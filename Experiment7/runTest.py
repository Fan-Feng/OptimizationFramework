from mpipool import MPIPool
from mpi4py import MPI

def menial_task(x):
    print(x, MPI.COMM_WORLD.Get_rank())
    return x ** MPI.COMM_WORLD.Get_rank()

with MPIPool() as pool:
    pool.workers_exit()
    print("Only master")

    results = pool.map(menial_task, range(3))
    print(results)

print("All MPI processes join again here.")