import time

from numba import cuda


def synchronous_kernel_timeit( kernel, number=1, repeat=1 ):
    """Time a kernel function while synchronizing upon every function call.

    :param kernel: Lambda function containing the kernel function (including all arguments).
    :param number: Number of function calls in a single averaging interval.
    :param repeat: Number of repetitions.
    :return: List of timing results or a single value if repeat is equal to one.
    """

    times = []
    for r in range(repeat):
        start = time.time()
        for n in range(number):
            kernel()
            cuda.synchronize() # Do not queue up, instead wait for all previous kernel launches to finish executing
        stop = time.time()
        times.append((stop - start) / number)
    return times[0] if len(times)==1 else times
