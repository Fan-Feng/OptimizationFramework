#import multiprocessing as mp
import mpi4py.futures
import random
import time,sys


def howmany_within_range(row,minimum=4,maximum=8):
    count = 0 
    for n in row:
        if minimum <= n <= maximum:
            count = count + 1
    #print(sys.version)
    return count

random.seed(100)
data = [[random.randint(0,10) for j in range(10000000)] for i in range(5)]

pool = mpi4py.futures.MPIPoolExecutor(max_workers=30)

tic = time.perf_counter()
results = pool.map(howmany_within_range,[row for row in data])
pool.shutdown()
toc = time.perf_counter()
print("Simulation time:",toc-tic)
