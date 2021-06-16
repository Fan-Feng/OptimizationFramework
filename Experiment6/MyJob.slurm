#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE        #Do not propagate environment
#SBATCH --get-user-env=L     #Replicate login environment
  
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=JobExample1     #Set the job name to "JobExample1"
#SBATCH --time=03:30:00            #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=4                #Request 1 task
#SBATCH --ntasks-per-node=1        #Request 1 task/core per node
#SBATCH --mem=2560M                #Request 2560MB (2.5GB) per node
#SBATCH --output=Example1Out.%j    #Send stdout/err to "Example1Out.[jobID]"
ml purge
ml CMake/3.15.3-GCCcore-8.3.0
ml intel/2018b
ml mpi4py/3.0.2-iimpi-2019a-timed-pingpong

export PATH=$SCRATCH/programs/EnergyPlus-9-4-0:$SCRATCH/programs/anaconda3/bin:$PATH

mpirun python runTest.py