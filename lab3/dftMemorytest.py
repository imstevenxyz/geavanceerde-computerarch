import time
import numpy as np
from numba import cuda
import matplotlib.pyplot as plt
import math
from math import sin, cos, pi

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
           # result[i+j, j] = samples[i]*coeffs[j]
           result[i]+=samples[i+j]*coeffs[j]
           #  cuda.atomic.add(result,i,samples[i+j]*coeffs[j])

@cuda.jit
def kernel_parallel(samples, nSamples, frequencies_real, frequencies_img):
    '''Use the GPU for generateing a histogram. In parallel.'''

    # Calculate the thread's absolute position within the grid
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    sample = samples[x]

    for k in range(frequencies_img.shape[0]):
        cuda.atomic.add(frequencies_real, k, ((sample * (cos(2 * pi * k * x / nSamples)))))
        cuda.atomic.add(frequencies_img, k, ((sample * (-1* sin(2 * pi * k * x / nSamples) ))))



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

def function_dft_filter(signal, kz_coeffs, result, N, frequencies_real, frequencies_img):
    kernel_filter_outputFocus[1, 128](signal, kz_coeffs, result)
    kernel_parallel[2, 1000](result, N, frequencies_real, frequencies_img)

def create_signal_samples(sample_size):
    dt = 0.1
    timePoints = np.arange(0, sample_size+dt, dt)
    signal_samples = np.sin(2*np.pi*0.05*timePoints)
    return signal_samples

########################
Nsamples = [100, 2000]

kz_degree = 3
kz_window = 101
kz_coeffs = _kz_coeffs(kz_window, kz_degree)
pad_size = int(kz_degree*(kz_window-1)/2)

times = np.zeros(100, dtype=np.float)
x = np.zeros(100, dtype=np.int32)

for i in range(0, 100):
    signal = create_signal_samples(i)
    result = np.zeros(signal.size)
    signal_padded = np.append(np.zeros(pad_size),np.append(signal,np.zeros(pad_size)))
    frequencies_real = np.zeros(int(signal.size/2+1), dtype=np.float)
    frequencies_img = np.zeros(int(signal.size/2+1), dtype=np.float)

    #first compile
    function_dft_filter(signal_padded, kz_coeffs, result, signal.size, frequencies_real, frequencies_img)

    #time not on device
    t_no_mem = synchronous_kernel_timeit(lambda: function_dft_filter(signal_padded, kz_coeffs, result, signal.size, frequencies_real, frequencies_img), number=10)

    # time on device
    result_dev = cuda.to_device(result)
    signal_dev = cuda.to_device(signal_padded)
    coeff_dev = cuda.to_device(kz_coeffs)
    img_dev = cuda.to_device(frequencies_img)
    real_dev = cuda.to_device(frequencies_real)
    t_mem = synchronous_kernel_timeit(lambda: function_dft_filter(signal_dev, coeff_dev, result_dev, signal.size, real_dev, img_dev), number=10)

    # time transfer
    t_transfer = synchronous_kernel_timeit(lambda: cuda.to_device(result), number=10)

    print(i)
    print(t_no_mem-(t_mem+t_transfer))
    times[i] = t_no_mem-(t_mem)
    x[i] = i

plt.plot(x, times)
plt.show()