'''
A python script for implement a MPC framework

Simulation model: EnergyPlus
Optimization algorithm: pygad

Author: ffeng@tamu.edu 
'''

import os, sys, shutil
import subprocess as sp

import time,datetime
from multiprocessing import Pool

import numpy as np
import numpy.random as random
import pandas as pd

## import mpi management package
from mpipool import MPIPool
from mpi4py import MPI

# Import optimization package
import pygad
def convert_NumOfSec_To_MonAndDay(NumOfSec): 
  '''
  Convert NumOfSec to Month/Day
  '''
  DayOfYear = int(NumOfSec/86400)
  HourOfDay = int((NumOfSec%86400)/3600)  # This is just a placehold..
  DateValue = datetime.datetime(2021, 1, 1) + datetime.timedelta(DayOfYear)
  Mon = DateValue.month
  DayValue = DateValue.day

  return Mon,DayValue,HourOfDay

def modifyIDF(fileName,targetFile,startMon,startDay,endMon, endDay,SchFileLOC):
  '''
  Modify idf file by specifying startMon,startDay,endMon, endDay
  '''
  with open(fileName,'r') as fp:
    lines = fp.readlines()
  
  # Replace
  with open(targetFile,'w') as fp:
    for line in lines:
      line = line.replace("%BeginMon%",str(startMon)) 
      line = line.replace("%BeginDay%",str(startDay))
      line = line.replace("%EndMon%",str(endMon))
      line = line.replace("%EndDay%",str(endDay))
      line = line.replace("%SchFile_Loc%",SchFileLOC)
      fp.writelines(line)
  
def fitness_func(x,solution_idx):
  # run simulation, this is deprecated. 
  res = run_prediction(tim,x,[1],X_sp_log,start_time,final_time,Eplus_timestep,Eplus_FileName,solution_idx)

  # utility rate
  uRate = [0.5,0.5,0.6,0.7,1,1]

  total_Cost = sum([x*uRate[i] for i,x in enumerate(res)])
  return total_Cost

def penalty_func(ZMAT,output_DF,tim):

  ## This function could be modified in the future if necessary
  SP_list = [30.7]*6+[25]+[24]*15+[30.7]*2 # [18,24]
  ThermalComfort_range = 0.5
  residuals = 0
  for j in range(5):
    for i in range(output_DF.shape[0]):
      dtime = output_DF.iloc[i,0]
      hourOfDay = int(dtime.hour)
      if SP_list[hourOfDay] >20:
        a=0
      residuals += max(ZMAT.iloc[i,j] - ThermalComfort_range- SP_list[hourOfDay],0)
  
  return residuals

def fitness_wrapper(x,solution_idx,hyperParam):
  # run simulation 
  Sim_Status, ZMAT,output_DF = run_prediction(x,solution_idx,hyperParam)
  # utility rate, read from an external file
  uRate = [3.59]*12+[4.69]*2+[8.86]*4+[4.69]*2+[3.59]*4  
  alpha = 10**20 ## 
  if Sim_Status:
    tim = hyperParam["tim"]
    PH = hyperParam["PH"]
    # Total electricity rate = E_{RadSys_Pump} + E_{Boiler} +E_{Plant pump}
    PowerConsumption = output_DF.iloc[:,7:].apply(sum, axis = 1)
    total_Cost = 0
    CurMon,CurDay,HourOfDay = convert_NumOfSec_To_MonAndDay(tim)
    for i in range(PH):
      curHour = (HourOfDay + i)%24
      #print(curHour,"PowerCom:",PowerConsumption)
      total_Cost = total_Cost + (uRate[curHour])*PowerConsumption.iloc[i]

    total_Cost = - total_Cost - alpha * penalty_func(ZMAT,output_DF,tim)
  else:
    total_Cost = -alpha
  return total_Cost

def run_prediction(CVar_list, solution_idx,hyperParam):
  # This function runs the EPlus model over prediction horizon at "tim", and the control variable
  # is overridden by CVar_list.

  ## Step 0. Pre-process inputs for the models. 
  tim = hyperParam["tim"]
  CVar_timestep = hyperParam["CVar_timestep"]  #unit: Second.
  X_sp_log = hyperParam["X_sp_log"]
  start_time = hyperParam["start_time"]
  final_time = hyperParam["final_time"]
  Eplus_timestep = hyperParam["Eplus_timestep"]
  Eplus_FileName = hyperParam["Eplus_FileName"]

  if len(X_sp_log)>0:
    X_sp = np.concatenate((X_sp_log ,CVar_list )) # concatenate these Sp log with new Sp
  else:
    X_sp = CVar_list

  # calculate the end time of current prediction horizon
  time_end = tim + len(CVar_list) * CVar_timestep

  ## Generate some placeholders to make sure the model simulation time is a multiple of 86400
  final_time = int((time_end - 1)/86400)*86400 + 86400  # 


  ##Step 1. make a copy of the base Eplus model with specific start time and end time
  Cur_WorkPath =  os.getcwd()
  Target_WorkPath = Cur_WorkPath + '//Model_T{}_{}'.format(tim/3600,solution_idx)

  if (os.path.exists(Target_WorkPath)):# delete the workpath
    shutil.rmtree(Target_WorkPath)

  os.makedirs(Target_WorkPath)

  startMon,startDay,HourofDay = convert_NumOfSec_To_MonAndDay(start_time-86400)  # Add one day before the start time as warmup day.
  endMon,endDay,HourofDay = convert_NumOfSec_To_MonAndDay(final_time)
  modifyIDF(Cur_WorkPath + "//" + Eplus_FileName,Target_WorkPath+"//"+Eplus_FileName,startMon,startDay,endMon,endDay,Target_WorkPath+"//RadInletWater_SP_schedule.csv")

  # write control signal to the .csv file, both historical and new
  shutil.copyfile(Cur_WorkPath + "//RadInletWater_SP_schedule.csv",Target_WorkPath+"//RadInletWater_SP_schedule.csv")

  Input_DF = pd.read_csv(Target_WorkPath+"//RadInletWater_SP_schedule.csv")
  start_idx,end_idx = int(start_time/3600),int(time_end/3600)
  Input_DF.iloc[start_idx:end_idx,0] = X_sp  #
  Input_DF.iloc[start_idx:end_idx,1] = X_sp
  Aval_Status = [int(xi<=17) for xi in X_sp]
  Input_DF.iloc[start_idx:end_idx,2] = Aval_Status

  Input_DF.to_csv(Target_WorkPath+"//RadInletWater_SP_schedule.csv",index = False)

  ## Step 2. Run EnergyPlus model
   #srun --time 30 --mem-per-cpu 2048 -n 1 energyplus -w USA_CO_Golden-NREL.724666_TMY3.epw 1ZoneUncontrolled.idf
  argument = ["energyplus", "-w",Cur_WorkPath + "//in.epw","-d",Target_WorkPath,Target_WorkPath+"//"+Eplus_FileName]
  print("============EPlus Sim Start==================/n")
  sp.call(argument)
  print("============EPlus Sim End====================/n")
  
  ## Step 3. After completion, retrieve results
  Sim_Status = check_SimulationStatus(Target_WorkPath+"//" + "eplusout.err")
  print(Sim_Status)
  if Sim_Status:
    output_DF = read_result(Target_WorkPath+"//" + "eplusout.eso")
    tim_idx,end_idx = int((tim-start_time)/3600),int((time_end-start_time)/3600)
    ZMAT = output_DF.iloc[tim_idx+24:end_idx+24,2:7]
    
    ## Step 4. Remove temporary files
    shutil.rmtree(Target_WorkPath)
    return Sim_Status, ZMAT,output_DF.iloc[tim_idx+24:end_idx+24,:]
  else:

    output_DF, ZMAT, Heating_Rate = -1,-1,-1
    ## Step 4. Remove temporary files
    shutil.rmtree(Target_WorkPath)
    return Sim_Status, ZMAT,output_DF

def check_SimulationStatus(fileName):
  with open(fileName,'r') as fp:
    lines = fp.readlines()
    LastLine = lines[-1]
    print(LastLine)
  if LastLine.find('Successfully')>-1:
    Sim_Status = True
  else:
    Sim_Status = False
  return Sim_Status

def read_result(filename):
  import datetime
  ## a function used to process ESO file

  output_idx =   output_idx = [703,704,705,706,707,1682,1724,1730,1737,1743,1933,2062] # Indices for  
  data = {'dtime':[],
          'dayType':[]}
  for id_i in output_idx:
    data[str(id_i)] = []

  with open(filename) as fp:
    while True:
      line = fp.readline()
      ## Skip the header part
      if line.startswith('End of Data Dictionary'):
        break
      else:
        continue
      
    while True:
      line = fp.readline()
      if line.startswith('End of Data'):
        break

      fields = [f.strip() for f in line.split(',')]
      id = int(fields[0])
      if id == 2: # this is the timestamp for all following outputs
        dtime = datetime.datetime(2021,int(fields[2]),int(fields[3]),int(float(fields[5]))-1,int(float(fields[6])))
        dayType = fields[-1]
        data['dtime'].append(dtime)
        data['dayType'].append(dayType)
        continue

      if id in output_idx:
        data[str(id)].append(float(fields[1]))
      else:
        # skip entries that are not output:variables
        continue

  data = pd.DataFrame(data)
  return data

class PooledGA(pygad.GA):

  def cal_pop_fitness(self):
    global pool,hyperParam
    pop_fitness = pool.starmap(fitness_wrapper, [(individual,i,hyperParam) for i,individual in enumerate(self.population)])
    #print(pop_fitness)
    pop_fitness = np.array(pop_fitness)
    return pop_fitness

# run optimization 
with MPIPool() as pool:
  pool.workers_exit() ## Only master process will proceed
  
  # simulation setup
  start_time= 60*60*24*205  # July 24
  final_time= 60*60*24*206
  Eplus_timestep = 60*3 # 3 min

  # setup for MPC
  pred_horizon = {"length":6,"timestep":3600}

  #### run optimization
  X_sp_log = []  # This trend variable is used to store all setpoints from start_time 

  Eplus_FileName = "MediumOffice_Houston_Cooling.idf"


  #prepare hyper parameter
  hyperParam  ={}
  hyperParam["CVar_timestep"] =  pred_horizon['timestep'] 
  hyperParam["PH"] = pred_horizon['length'] 
  hyperParam["start_time"] = start_time 
  hyperParam["final_time"] = final_time 
  hyperParam["Eplus_timestep"] = Eplus_timestep
  hyperParam["Eplus_FileName"] = Eplus_FileName

  
  ##
  tim = start_time
  while True:
    #
    hyperParam["tim"] = tim
    hyperParam["X_sp_log"] = X_sp_log

    # Do optimization
    
    #SP_cur = run_Optimization(hyperParam)

    ## At each time step, this function will implement an optimization.. \
    # Parameter for GA 
    num_parents_mating = 24
    num_genes = hyperParam["PH"]

    init_range_low = 25
    init_range_high = 50
    parent_selection_type = "tournament"
    keep_parents = 1

    # Optimization algorithm setting
    num_generations = 30
    sol_per_pop = 49   # Number of individuals

    crossover_type = "single_point"
    crossover_probability = 0.9

    mutation_type = "random"
    mutation_probability = 0.2

    gene_space = [{'low':10, 'high': 18}]*hyperParam['PH']

    ga_instance = PooledGA(num_generations=num_generations,
                    num_parents_mating=num_parents_mating,
                    fitness_func=fitness_func, # Actually this is not used.
                    sol_per_pop=sol_per_pop,
                    num_genes=num_genes,
                    init_range_low=init_range_low,
                    init_range_high=init_range_high,
                    parent_selection_type=parent_selection_type,
                    keep_parents=keep_parents,
                    crossover_type=crossover_type,
                    crossover_probability = crossover_probability,
                    mutation_type=mutation_type,
                    mutation_probability = mutation_probability,
                    gene_space = gene_space
                    )
    print("Start Optimization")
    ga_instance.run()
    print("Op completed")
    SP_cur = ga_instance.best_solution()[0][0]
    X_sp_log.append(SP_cur)
    print(X_sp_log,tim)
    # proceed to next timestep
    tim = tim + pred_horizon['timestep']
    if tim>= final_time:
      break
    
    
  print("all mpi process join again then")
      
