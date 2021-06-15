import multiprocessing as mp
import random
import time,sys


def howmany_within_range(row,minimum,maximum):
    count = 0 
    for n in row:
        if minimum <= n <= maximum:
            count = count + 1
    #print(sys.version)
    return count

random.seed(100)
data = [[random.randint(0,10) for j in range(100000000)] for i in range(5)]

pool = mp.Pool(30)

tic = time.perf_counter()
results = pool.starmap(howmany_within_range,[(row,4,8) for row in data])
pool.close()
toc = time.perf_counter()
print("Simulation time:",toc-tic)
