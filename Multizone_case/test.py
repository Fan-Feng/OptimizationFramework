import subprocess

for i in range(2):
    argument=["sbatch",'MyJob{}.slurm'.format(i+1)]
    subprocess.Popen(argument)

