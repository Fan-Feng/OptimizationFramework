from mpipool import MPIPool
from mpi4py import MPI

def menial_task(x):
  return x ** MPI.COMM_WORLD.Get_rank()

with MPIPool() as pool:
    pool.workers_exit()
    print("Only master")

    results = pool.map(menial_task, range(10))

print("All MPI processes join again here.")