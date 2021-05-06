import time
import numpy as np
from numba import cuda
import matplotlib.pyplot as plt
import math
from math import sin, cos, pi
from scipy import signal

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

def function_dft_filter(signal,kz_coeffs,result,N,frequencies_real_1, frequencies_img_1,frequencies_real_2, frequencies_img_2):
    kernel_parallel[2, 1000](result, N, frequencies_real_1, frequencies_img_1)
    kernel_filter_outputFocus[1, 128](signal, kz_coeffs, result)
    kernel_parallel[2, 1000](result, N, frequencies_real_2, frequencies_img_2)

########################

# Define signal samples
dt = 0.1
timePoints = np.arange(0, 200+dt, dt)
signal_original = signal.sawtooth(2*np.pi*0.05*timePoints)
# signal_original = np.sin(2*np.pi*0.05*timePoints)
noise= np.random.normal(0.1, size=signal_original.size)

# Define filter degree and window
kz_degree = 3
kz_window = 101
kz_coeffs = _kz_coeffs(kz_window, kz_degree)

pad_size = int(kz_degree*(kz_window-1)/2)

result = np.zeros(signal_original.size)
signal = np.append(np.zeros(pad_size),np.append(noise,np.zeros(pad_size))) # Padded left and right
result_array_gs_device = cuda.to_device(result)

kernel_filter_outputFocus[1,128](signal, kz_coeffs, result_array_gs_device)

# dft

N = noise.size
# dft original
frequencies_real = np.zeros(int(N/2+1), dtype=np.float)
frequencies_img = np.zeros(int(N/2+1), dtype=np.float)
kernel_parallel[2, 1000](noise, N, frequencies_real, frequencies_img)


fig, (ax1, ax2) = plt.subplots(1, 2)
timePoints_2 = np.arange(0, 100+dt, dt)

ax1.plot(timePoints, noise)
ax2.plot(timePoints_2, abs(frequencies_real + frequencies_img * 1j), color='C3')
plt.show()
plt.savefig('dft_original.png')

# dft filter
frequencies_real = np.zeros(int(N/2+1), dtype=np.float)
frequencies_img = np.zeros(int(N/2+1), dtype=np.float)
kernel_parallel[2, 1000](result_array_gs_device, N, frequencies_real, frequencies_img)

fig, (ax1, ax2) = plt.subplots(1, 2)
timePoints_2 = np.arange(0, 100+dt, dt)

ax1.plot(timePoints, result_array_gs_device)
ax2.plot(timePoints_2, abs(frequencies_real + frequencies_img * 1j), color='C3')
plt.show()
plt.savefig('dft_filter.png')

#time no memoery

frequencies_real_1 = np.zeros(int(N/2+1), dtype=np.float)
frequencies_img_1 = np.zeros(int(N/2+1), dtype=np.float)
frequencies_real_2 = np.zeros(int(N/2+1), dtype=np.float)
frequencies_img_2 = np.zeros(int(N/2+1), dtype=np.float)
t_par = synchronous_kernel_timeit(lambda: function_dft_filter(signal,kz_coeffs,result,N,frequencies_real_1, frequencies_img_1,frequencies_real_2, frequencies_img_2), number=100)
print("time memory not on device ", t_par)

#time memory
frequencies_real = np.zeros(int(N/2+1), dtype=np.float)
frequencies_img = np.zeros(int(N/2+1), dtype=np.float)
result_array_gs_device = cuda.to_device(result)
t_par = synchronous_kernel_timeit(lambda: function_dft_filter(signal,kz_coeffs,result_array_gs_device,N,frequencies_real_1, frequencies_img_1,frequencies_real_2,frequencies_img_2), number=100)
print("time memory to device ", t_par)
