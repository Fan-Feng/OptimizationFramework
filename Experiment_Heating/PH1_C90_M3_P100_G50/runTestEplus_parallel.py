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

# Import optimization package
import pygad

## import mpi management package
from mpipool import MPIPool
from mpi4py import MPI

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

def penalty_func(ZMAT,output_DF):

  ## This function could be modified in the future if necessary
  SP_list = [15.6]*5+[17.6]+[19.6]+[21]*15+[15.6]*2 # [18,24]
  ThermalComfort_range = 1

  residuals = 0
  for i in range(output_DF.shape[0]):
    dtime = output_DF.iloc[i,0]
    hourOfDay = int(dtime.hour)
    residuals += max(SP_list[hourOfDay]-ThermalComfort_range-ZMAT[i],0)
  
  return residuals

def fitness_wrapper(x,solution_idx,hyperParam):
  # run simulation 
  Heating_rate, ZMAT,output_DF = run_prediction(x,solution_idx,hyperParam)

  # utility rate, read from an external file
  uRate = [3.462]*6+[5.842]*9+[10.378]*5+[5.842]*2+[3.462]*2  # Summer, workday. Replaced later. 
  
  alpha = 10**20 ## 

  tim = hyperParam["tim"]
  PH = hyperParam["PH"]
  # Total electricity rate = E_{RadSys_Pump} + E_{Boiler} +E_{Plant pump}
  PowerConsumption = output_DF.iloc[:,4:].apply(sum, axis = 1)
  total_Cost = 0
  CurMon,CurDay,HourOfDay = convert_NumOfSec_To_MonAndDay(start_time)
  for i in range(PH):
    curHour = (HourOfDay + i)%24
    total_Cost = total_Cost + (uRate(curHour))*PowerConsumption[i]

  total_Cost = - total_Cost - alpha * penalty_func(ZMAT,output_DF)

  return total_Cost

def run_prediction(CVar_list, solution_idx,hyperParam):
  # This function runs the EPlus model over prediction horizon at "tim", and the control variable
  # is overridden by CVar_list.
  # Because get_state and set_state are not supported, then we need to run the EPlus from beginning everytime. 

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

  startMon,startDay,HourofDay = convert_NumOfSec_To_MonAndDay(start_time)
  endMon,endDay,HourofDay = convert_NumOfSec_To_MonAndDay(final_time)
  modifyIDF(Cur_WorkPath + "//" + Eplus_FileName,Target_WorkPath+"//"+Eplus_FileName,startMon,startDay,endMon,endDay,Target_WorkPath+"//RadInletWater_SP_schedule.csv")

  # write control signal to the .csv file, both historical and new
  shutil.copyfile(Cur_WorkPath + "//RadInletWater_SP_schedule.csv",Target_WorkPath+"//RadInletWater_SP_schedule.csv")

  Input_DF = pd.read_csv(Target_WorkPath+"//RadInletWater_SP_schedule.csv")
  start_idx,end_idx = int(start_time/3600),int(time_end/3600)
  Input_DF.iloc[start_idx:end_idx,0] = X_sp  #
  Input_DF.iloc[start_idx:end_idx,1] = X_sp
  Aval_Status = [int(xi>=30) for xi in X_sp]
  Input_DF.iloc[start_idx:end_idx,2] = Aval_Status

  Input_DF.to_csv(Target_WorkPath+"//RadInletWater_SP_schedule.csv",index = False)

  ## Step 2. Run EnergyPlus model
   #srun --time 30 --mem-per-cpu 2048 -n 1 energyplus -w USA_CO_Golden-NREL.724666_TMY3.epw 1ZoneUncontrolled.idf
  argument = ["energyplus", "-w",Cur_WorkPath + "//in.epw","-d",Target_WorkPath,Target_WorkPath+"//"+Eplus_FileName]
  sp.call(argument)
  
  ## Step 3. After completion, retrieve results
  output_DF = read_result(Target_WorkPath+"//" + "eplusout.eso")
  tim_idx,end_idx = int((tim-start_time)/3600),int((time_end-start_time)/3600)
  ZMAT = list(output_DF.iloc[tim_idx:end_idx,2]) 
  Heating_Rate = list(output_DF.iloc[tim_idx:end_idx,3])  

  ## Step 4. Remove temporary files
  shutil.rmtree(Target_WorkPath)
  return Heating_Rate,ZMAT,output_DF.iloc[tim_idx:end_idx,:]

def read_result(filename):
  import datetime
  ## a function used to process ESO file

  output_idx = [675,1748,1753,1991,2033] # Indices for  ZMAT, heating rate, RadSyste Pump E_Rate(w), Boiler E_Rate, Plant Pump E_rate(w)
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
def run_Optimization(hyperParam):
  rng = random.default_rng(1234)
  CVar_list = rng.random(pred_horizon['length'])
  CVar_list = [CVar*5+12 for CVar in CVar_list]


  ## At each time step, this function will implement an optimization.. \
  num_parents_mating = 4
  num_genes = len(hyperParam["PH"])

  init_range_low = 25
  init_range_high = 50
  parent_selection_type = "sss"
  keep_parents = 1

  # Optimization algorithm setting
  num_generations = 50
  sol_per_pop = 49   # Number of individuals

  crossover_type = "single_point"
  crossover_probability = 0.9

  mutation_type = "random"
  mutation_probability = 0.03

  gene_space = [{'low': 25, 'high': 50}]*24

  hyperParam = hyperParam

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

  print(ga_instance.best_solution())  

  return ga_instance.best_solution()[0][0]

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
    start_time= 60*60*24*20 
    final_time= 60*60*24*21
    Eplus_timestep = 60*3 # 3 min

    # setup for MPC
    pred_horizon = {"length":1,"timestep":3600}

    #### run optimization
    X_sp_log = []  # This trend variable is used to store all setpoints from start_time 

    Eplus_FileName = "MediumOff_NewYork.idf"


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
      SP_cur = run_Optimization(hyperParam)

      # proceed to next timestep
      tim = tim + pred_horizon['timestep']
      if tim>= final_time:
        break
      X_sp_log.append(SP_cur)
      print(X_sp_log)


print("all mpi process join again then")
      
