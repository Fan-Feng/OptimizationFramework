from mpipool import MPIPool
from mpi4py import MPI

def menial_task(x):
  print(MPI.COMM_WORLD.Get_rank())
  return x ** MPI.COMM_WORLD.Get_rank()

with MPIPool() as pool:
  pool.workers_exit()
  print("Only the master executes this code.")

  # Block for results
  results = pool.map(menial_task, range(10))
  print(results)

  # Async
  result = pool.map_async(menial_task, range(10))
  print("Done already?", result.ready())

print("All MPI processes join again here.")