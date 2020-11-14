from numba import jit, cuda
import numpy as np
# to measure exec time
from timeit import default_timer as timer


# normal function to run on cpu
def func(a):
    for i in range(10000000):
        a[i] += 1

    # function optimized to run on gpu


@jit(nopython=True)
def func2(a):
    for i in range(10000000):
        a[i] += 1


if __name__ == "__main__":
    n = 10000000
    a = np.ones(n, dtype=np.float64)
    b = np.ones(n, dtype=np.float32)

    start = timer()
    func(a)
    cpu_time = timer() - start
    print("without GPU:", cpu_time)

    start = timer()
    func2(a)
    gpu_time = timer() - start
    print("with GPU:", gpu_time)
    print("x faster: ", cpu_time/gpu_time)
