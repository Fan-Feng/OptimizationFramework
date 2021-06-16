from mpipool import Pool
from mpi4py import MPI

def menial_task(x):
  return x ** MPI.COMM_WORLD.Get_rank()

with Pool() as pool:
  # Block for results
  results = pool.map(menial_task, range(100))

  # Async
  result = pool.map_async(menial_task, range(100))
  print("Done already?", result.ready())

print("All MPI processes join again here.")