import multiprocessing as mp
import numpy as np


def howmany_within_range(row,minimum,maximum):
    count = 0 
    for n in row:
        if minimum <= n <= maximum:
            count = count + 1
    return count

np.random.RandomState(100)
arr = np.random.randint(0,10,size=[20000,5])
data = arr.tolist()


pool = mp.Pool(30)

results = [pool.apply(howmany_within_range,args = (row,4,8)) for row in data]

pool.close()
print(results[:10])