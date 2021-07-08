#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE        #Do not propagate environment
#SBATCH --get-user-env=L     #Replicate login environment
  
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=JobExample1     #Set the job name to "JobExample1"
#SBATCH --time=10:00:00            #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=500                #Request 400 task
#SBATCH --ntasks-per-node=20        #Request 10 task/core per node
#SBATCH --mem=20480M                #Request 204800MB (20GB) per node
#SBATCH --output=Example1Out.%j    #Send stdout/err to "Example1Out.[jobID]"

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --account=122774970664             #Set billing account to 122774970664

ml purge
ml intel/2019a

ml CMake/3.15.3-GCCcore-8.3.0
export PATH=$SCRATCH/programs/EnergyPlus-9-4-0:$PATH

mpirun python runTestEplus_parallel.py