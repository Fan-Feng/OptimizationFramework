from mpipool import MPIPool
from mpi4py import MPI

def menial_task(x):
    print(MPI.COMM_WORLD.Get_rank())
    return x ** MPI.COMM_WORLD.Get_rank()

with MPIPool() as pool:
    pool.workers_exit()
    print("Only master")

    results = pool.map(menial_task, range(4))

print("All MPI processes join again here.")