import time
import numpy as np
from numba import cuda
import matplotlib.pyplot as plt

def synchronous_kernel_timeit(kernel_func, number=1, repeat=1):
    """Time a kernel function while synchronizing upon every function call.

    :param kernel: Lambda function containing the kernel function (including all arguments)
    :param number: Number of function calls in a single averaging interval
    :param repeat: Number of repetitions
    :return: List of timing results or a single value if repeat is equal to one
    """

    times = []
    for r in range(repeat):
        start = time.time()
        for n in range(number):
            kernel_func()
            cuda.synchronize() # Do not queue up, instead wait for all previous kernel launches to finish executing
        stop = time.time()
        times.append((stop - start) / number)

    return times[0] if len(times)==1 else times

@cuda.jit
def kernel_filter_outputFocus(samples, coeffs, result):
    # Calculate the thread's absolute position within the grid
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    # Set stride equal to the number of threads we have available in either direction
    stride_x = cuda.gridDim.x * cuda.blockDim.x

    for i in range(x, result.shape[0], stride_x):
        for j in range(coeffs.shape[0]):
            result[i]+=samples[i+j]*coeffs[j]

@cuda.jit
def kernel_filter_log(samples, coeffs, result, tempresult):
    # Calculate the thread's absolute position within the grid
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    # Set stride equal to the number of threads we have available in either direction
    stride_x = cuda.gridDim.x * cuda.blockDim.x

    for i in range(x, samples.shape[0], stride_x):
        for j in range(coeffs.shape[0]):
            cuda.atomic.add(result,i+j,samples[i]*coeffs[j])


def _kz_coeffs(m, k):
    """Calc KZ coefficients. Source https://github.com/Kismuz/kolmogorov-zurbenko-filter"""

    # Coefficients at degree one
    coef = np.ones(m)

    # Iterate k-1 times over coefficients
    for i in range(1, k):

        t = np.zeros((m, m + i * (m - 1)))
        for km in range(m):
            t[km, km:km + coef.size] = coef

        coef = np.sum(t, axis=0)

    assert coef.size == k * (m - 1) + 1

    return coef / m ** k

def time_filter(blocks, threads, samples, coeffs, result):
    kernel_filter_outputFocus[blocks, threads](samples, coeffs, result)
    t_par = synchronous_kernel_timeit(lambda: kernel_filter_outputFocus[blocks, threads](samples, coeffs, result), number=100)
    return t_par

########################

kz_degree = 3
kz_window = 101
kz_coeffs = _kz_coeffs(kz_window, kz_degree)

pad_size = int(kz_degree*(kz_window-1)/2)

times_no_mem = np.zeros(300, dtype=np.float)
times_mem = np.zeros(300, dtype=np.float)
x = np.zeros(300, dtype=np.int32)

for i in range(0, 300):
    dt = 0.1
    timePoints = np.arange(0, i+dt, dt)
    signal_original = np.sin(2*np.pi*0.05*timePoints)
    result = np.zeros(signal_original.size)
    signal = np.append(np.zeros(pad_size),np.append(signal_original,np.zeros(pad_size))) # Padded left and right

    # time transfer
    t_transfer_signal = synchronous_kernel_timeit(lambda: cuda.to_device(signal), number=100)
    t_transfer_coeffs = synchronous_kernel_timeit(lambda: cuda.to_device(kz_coeffs), number=100)
    t_transfer_result = synchronous_kernel_timeit(lambda: cuda.device_array(result.size), number=100)
    t_transfer = t_transfer_signal + t_transfer_result + t_transfer_coeffs

    # time no mem
    t_no_mem = time_filter(1, 100, signal, kz_coeffs, result)

    # time mem
    signal_dev = cuda.to_device(signal)
    result_dev = cuda.device_array(result.size)
    coeffs_dev = cuda.to_device(kz_coeffs)
    t_mem = time_filter(1, 100, signal_dev, coeffs_dev, result_dev)

    times_mem[i] = t_transfer+t_mem
    times_no_mem[i] = t_no_mem

plt.plot(times_mem, label='Mem management')
plt.plot(times_no_mem, label='No mem management')
plt.ylabel("Execution time")
plt.xlabel("Sample set size")
plt.legend(loc='lower right')
plt.show()