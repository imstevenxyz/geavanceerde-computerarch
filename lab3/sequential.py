import time
from numba import cuda
import numpy as np
from numpy.lib import stride_tricks
import matplotlib.pyplot as plt

#pylint: disable=too-many-function-args
#pylint: disable=unsubscriptable-object

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

def _kz_coeffs(m, k):
    """Calc KZ coefficients. """

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

@cuda.jit
def filter_kernel_sequential(samples, kz_coeffs, result):
    """Execute the filter sequentially on the GPU.

    :param samples: An array containing discrete time domain signal samples.
    :param kz_coeffs: An array containing the filter normalized coefficients
    """
    for i in range(samples.shape[0]):
        for j in range(kz_coeffs.shape[0]):
            cuda.atomic.add(result, i+j, samples[i]*kz_coeffs[j])

# Define signal samples
dt = 0.1
timePoints = np.arange(0, 200+dt, dt)
signal = np.sin(2*np.pi*0.05*timePoints)

# Define filter degree and window
kz_degree = 3
kz_window = 101
kz_coeffs = _kz_coeffs(kz_window, kz_degree)

pad_size = int(kz_degree*(kz_window-1)/2)

result = np.zeros(signal.size+2*pad_size) # Padded left and right
filter_kernel_sequential[1,1](signal, kz_coeffs, result)
result_unpadded = result[2*pad_size:-(2*pad_size)] # remove incorrectly calculated values at signal begin and end

# Plot signals
plt.xlabel('Time')
plt.ylabel('Signal')
plt.plot(timePoints, signal, label='Original')
plt.plot(timePoints[pad_size:-pad_size], result_unpadded, label='Filtered')
plt.show()