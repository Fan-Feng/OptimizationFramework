import os, sys, shutil
import subprocess as sp

import time,datetime
from multiprocessing import Pool

import numpy as np
import numpy.random as random
import pandas as pd

# Import optimization package
from scipy.optimize import minimize

import pygad


def convert_NumOfSec_To_MonAndDay(NumOfSec): 
  DayOfYear = int(NumOfSec/86400)
  HourOfDay = int((NumOfSec%86400)/3600)
  DateValue = datetime.datetime(2021, 1, 1) + datetime.timedelta(DayOfYear)
  Mon = DateValue.month
  DayValue = DateValue.day

  return Mon,DayValue

def modifyIDF(fileName,targetFile,startMon,startDay,endMon, endDay):
  with open(fileName,'r') as fp:
    lines = fp.readlines()
  
  # Replace
  with open(targetFile,'w') as fp:
    for line in lines:
      line = line.replace("%BeginMon%",str(startMon)) 
      line = line.replace("%BeginDay%",str(startDay))
      line = line.replace("%EndMon%",str(endMon))
      line = line.replace("%EndDay%",str(endDay))
      fp.writelines(line)
    
def fitness_func(x,solution_idx):
    # run simulation
    res = run_prediction(tim,x,CVar_timestep,X_sp_log,start_time,final_time,Eplus_timestep,Eplus_FileName,solution_idx)

    # utility rate
    uRate = [0.5,0.5,0.6,0.7,1,1]

    total_Cost = sum([x*uRate[i] for i,x in enumerate(res)])
    return total_Cost
    
def fitness_wrapper(x,solution_idx,hyperParam):
        # run simulation
    tim = hyperParam["tim"]
    CVar_timestep = hyperParam["CVar_timestep"]
    X_sp_log = hyperParam["X_sp_log"]
    start_time = hyperParam["start_time"]
    final_time = hyperParam["final_time"]
    Eplus_timestep = hyperParam["Eplus_timestep"]
    Eplus_FileName = hyperParam["Eplus_FileName"]
    
    res = run_prediction(tim,x,CVar_timestep,X_sp_log,start_time,final_time,Eplus_timestep,Eplus_FileName,solution_idx)

    # utility rate
    uRate = [0.5,0.5,0.6,0.7,1,1]

    total_Cost = sum([x*uRate[i] for i,x in enumerate(res)])

    return total_Cost

def run_prediction(tim,CVar_list, CVar_timestep,X_sp_log, start_time,final_time,Eplus_timestep,Eplus_FileName, solution_idx):
    # This function runs the EPlus model over prediction horizon at "tim", and the control variable
    # is overridden by CVar_list.
    # Because get_state and set_state are not supported, then we need to run the EPlus from beginning everytime. 

    ## Step 0. Pre-process inputs for the models. 
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

    startMon,startDay = convert_NumOfSec_To_MonAndDay(start_time)
    endMon,endDay = convert_NumOfSec_To_MonAndDay(final_time)
    modifyIDF(Cur_WorkPath + "//" + Eplus_FileName,Target_WorkPath+"//"+Eplus_FileName,startMon,startDay,endMon,endDay)

    # write control signal to the .csv file, both historical and new

    shutil.copyfile(Cur_WorkPath + "//RadInletWater_SP_schedule.csv",Target_WorkPath+"//RadInletWater_SP_schedule.csv")

    Input_DF = pd.read_csv(Target_WorkPath+"//RadInletWater_SP_schedule.csv")
    start_idx,end_idx = int(start_time/3600),int(time_end/3600)
    print(start_idx,end_idx,X_sp_log,CVar_list,X_sp)
    Input_DF.iloc[start_idx:end_idx,0] = X_sp
    
    Input_DF.iloc[start_idx:end_idx,1] = X_sp
    Input_DF.to_csv(Target_WorkPath+"//RadInletWater_SP_schedule.csv",index = False)

    ## Step 2. Run EnergyPlus model
    sp.call(["energyplus", "-w",Cur_WorkPath + "//in.epw","-d",Target_WorkPath,"-r",Target_WorkPath+"//"+Eplus_FileName])
    
    ## Step 3. After completion, retrieve results
    # .
    output_DF = pd.read_csv(Target_WorkPath+"//" + "eplusout.csv")
    tim_idx,end_idx = int((tim-start_time)/3600),int((time_end-start_time)/3600)
    cooling_Rate = list(output_DF.iloc[tim_idx+48:end_idx+48,1])  # because of two design days start from 48.

    ## Step 4. Remove temporary files
    shutil.rmtree(Target_WorkPath)
    return cooling_Rate

class PooledGA(pygad.GA):

    def cal_pop_fitness(self):
        global pool,hyperParam
        pop_fitness = pool.starmap(fitness_wrapper, [(individual,i,hyperParam) for i,individual in enumerate(self.population)])
        #print(pop_fitness)
        pop_fitness = np.array(pop_fitness)
        return pop_fitness
        
if __name__ == "__main__":
    # write control signal to the .csv file, both historical and new
    X_sp = [12]*24

    # simulation setup
    start_time= 60*60*24*151 
    final_time= 60*60*24*158
    Eplus_timestep = 60

    # setup for MPC
    pred_horizon = {"length":6,"timestep":3600}

    #### run optimization
    X_sp_log = [12] * 6  # This trend variable is used to store all setpoints from start_time 
    CVar_timestep = pred_horizon['timestep']

    rng = random.default_rng(1234)
    CVar_list = rng.random(pred_horizon['length'])
    CVar_list = [CVar*5+12 for CVar in CVar_list]

    tim = start_time + 60*60*6
    Eplus_FileName = "testModel.idf"

    
    #prepare hyper parameter
    hyperParam  ={}
    hyperParam["tim"] = tim
    hyperParam["CVar_timestep"] =  CVar_timestep 
    hyperParam["X_sp_log"] = X_sp_log 
    hyperParam["start_time"] = start_time 
    hyperParam["final_time"] = final_time 
    hyperParam["Eplus_timestep"] = Eplus_timestep
    hyperParam["Eplus_FileName"] = Eplus_FileName
        
    fitness_function = fitness_func
    # Optimization algorithm setting
    num_generations = 50
    num_parents_mating = 4
    
    # Number of individuals
    sol_per_pop = 30
    num_genes = len(CVar_list)

    init_range_low = 7
    init_range_high = 18

    parent_selection_type = "sss"
    keep_parents = 1

    crossover_type = "single_point"

    mutation_type = "random"
    mutation_num_genes = 1
    
    ga_instance = PooledGA(num_generations=num_generations,
                   num_parents_mating=num_parents_mating,
                   fitness_func=fitness_function,
                   sol_per_pop=sol_per_pop,
                   num_genes=num_genes,
                   init_range_low=init_range_low,
                   init_range_high=init_range_high,
                   parent_selection_type=parent_selection_type,
                   keep_parents=keep_parents,
                   crossover_type=crossover_type,
                   mutation_type=mutation_type,
                   mutation_num_genes=mutation_num_genes)
    # run optimization 
    #ga_instance.run()

    with Pool(processes=2) as pool:
        ga_instance.run()
        
    print(ga_instance.best_solutions())