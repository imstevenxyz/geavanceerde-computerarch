import time
import numpy as np
from math import sin, cos, pi
from matplotlib import pyplot as plt
from numba import cuda

# pylint: disable=unsubscriptable-object

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

def DFT_sequential(samples, frequencies):
    """Execute the DFT sequentially on the CPU.

    :param samples: An array containing discrete time domain signal samples.
    :param frequencies: An empty array where the frequency components will be stored.
    """

    for k in range(frequencies.shape[0]):
        for n in range(N):
            frequencies[k] += samples[n] * (cos(2 * pi * k * n / N) - sin(2 * pi * k * n / N) * 1j)

@cuda.jit
def DFT_kernel_sequential(samples, frequencies_real,frequencies_img):
    """Execute the DFT sequentially on the CPU.

    :param samples: An array containing discrete time domain signal samples.
    :param frequencies: An empty array where the frequency components will be stored.
    """

    for k in range(frequencies_real.shape[0]):
        for n in range(N):
            cuda.atomic.add(frequencies_real, k, ((samples[n] * (cos(2 * pi * k * n / N)))))
            cuda.atomic.add(frequencies_img, k, ((samples[n] * (-1 * sin(2 * pi * k * n / N)))))\

@cuda.jit
def kernel_parallel(samples, frequencies_real, frequencies_img):
    '''Use the GPU for generateing a histogram. In parallel.'''

    # Calculate the thread's absolute position within the grid
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    sample = samples[x]

    for k in range(frequencies_img.shape[0]):
        cuda.atomic.add(frequencies_real, k, ((sample * (cos(2 * pi * k * x / N)))))
        cuda.atomic.add(frequencies_img, k, ((sample * (-1* sin(2 * pi * k * x / N) ))))

@cuda.jit
# def kernel_parallel_sequential(samples, frequencies_real, frequencies_img,threads):
def kernel_parallel_sequential(samples, frequencies_real, frequencies_img, threads, reps):
    '''Use the GPU for generateing a histogram. In parallel.'''

    # Calculate the thread's absolute position within the grid
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    # _N = samples.shape[0]

    for k in range(frequencies_real.shape[0]):
        # for i in range(N/threads):
        # for i in range(_N/threads):
        for i in range(reps):
            cuda.atomic.add(frequencies_real, k, ((samples[x+i*threads]* (cos(2 * pi * k * (x+i*threads) / N)))))
            cuda.atomic.add(frequencies_img, k, ((samples[x+i*threads] * (-1 * sin(2 * pi * k * (x+i*threads) / N)))))



##
## Timing functions
##

def time_cpu():
    frequencies = np.zeros(int(N/2+1), dtype=np.complex)
    DFT_sequential(sig_sum, frequencies)
    t_cpu = synchronous_kernel_timeit(lambda: DFT_sequential(sig_sum, frequencies), number=10)
    print('CPU:')
    print(t_cpu)

def time_gpu_seq():
    frequencies_real = np.zeros(int(N/2+1), dtype=np.float)
    frequencies_img = np.zeros(int(N/2+1), dtype=np.float)
    DFT_kernel_sequential[1, 1](sig_sum, frequencies_real, frequencies_img)
    t_seq = synchronous_kernel_timeit(lambda: DFT_kernel_sequential[1,1](sig_sum, frequencies_real, frequencies_img), number=10)
    print('GPU seq:')
    print(t_seq)

def time_gpu_par():
    frequencies_real = np.zeros(int(N/2+1), dtype=np.float)
    frequencies_img = np.zeros(int(N/2+1), dtype=np.float)
    kernel_parallel[1, 500](sig_sum, frequencies_real, frequencies_img)
    t_par = synchronous_kernel_timeit(lambda: kernel_parallel[1,500](sig_sum, frequencies_real, frequencies_img), number=10)
    print('GPU par:')
    print(t_par)

def time_gpu_par_seq(threads):
    frequencies_real = np.zeros(int(N/2+1), dtype=np.float)
    frequencies_img = np.zeros(int(N/2+1), dtype=np.float)
    kernel_parallel_sequential[1, threads](sig_sum, frequencies_real, frequencies_img, threads)
    t_par_seq = synchronous_kernel_timeit(lambda: kernel_parallel_sequential[1,threads](sig_sum, frequencies_real, frequencies_img,threads), number=10)
    print('GPU par seq with ' + str(threads) + ' threads : ' )
    print(t_par_seq)
###

import math
def time_gpu_par_seq_time(threads):
    frequencies_real = np.zeros(int(N/2+1), dtype=np.float)
    frequencies_img = np.zeros(int(N/2+1), dtype=np.float)
    reps = math.ceil(N / threads)

    print(threads, reps, N)
    kernel_parallel_sequential[1, threads](sig_sum, frequencies_real, frequencies_img, threads, reps)
    t_par_seq = synchronous_kernel_timeit(lambda: kernel_parallel_sequential[1, threads](sig_sum, frequencies_real, frequencies_img, threads, reps), number=10)
    return t_par_seq

def times(array_times,aray_threads,threads,stepsize):
    array_times = np.zeros(int(threads/stepsize), dtype=np.float)
    array_threads = np.zeros(int(threads/stepsize), dtype=np.integer)
    t_par_seq=0
    for i in range(aray_threads.shape[0]):
        t_par_seq = time_gpu_par_seq_time((i+1)*stepsize)
        print(t_par_seq)
        array_times[i]=t_par_seq
        array_threads[i] = (i+1)*stepsize

    plt.plot(array_threads, array_times)
    plt.show()
##
## Plotting functions
##



def create_plots():
    # plot CPU equential
    frequencies = np.zeros(int(N/2+1), dtype=np.complex)
    DFT_sequential(sig_sum, frequencies)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    xf = np.linspace(0, SAMPLING_RATE_HZ/2, int(N/2+1), endpoint=True)
    for sig in sigs:
        ax1.plot(x, sig, lw=0.5, color='#333333', alpha=0.5)

    ax1.plot(x, sig_sum)
    ax2.plot( xf, abs(frequencies), color='C3' )
    plt.show()

    # plot CPU equential
    frequencies_real = np.zeros(int(N/2+1), dtype=np.float)
    frequencies_img = np.zeros(int(N/2+1), dtype=np.float)
    DFT_kernel_sequential[1, 1](sig_sum, frequencies_real, frequencies_img)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    xf = np.linspace(0, SAMPLING_RATE_HZ/2, int(N/2+1), endpoint=True)
    for sig in sigs:
        ax1.plot(x, sig, lw=0.5, color='#333333', alpha=0.5)

    ax1.plot(x, sig_sum)
    ax2.plot(xf, abs(frequencies_real+frequencies_img*1j), color='C3')
    plt.show()

    # plot GPU paralell

    frequencies_real = np.zeros(int(N/2+1), dtype=np.float)
    frequencies_img = np.zeros(int(N/2+1), dtype=np.float)
    kernel_parallel[1, 500](sig_sum, frequencies_real, frequencies_img)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    xf = np.linspace(0, SAMPLING_RATE_HZ/2, int(N/2+1), endpoint=True)
    for sig in sigs:
        ax1.plot(x, sig, lw=0.5, color='#333333', alpha=0.5)

    ax1.plot(x, sig_sum)
    ax2.plot(xf, abs(frequencies_real+frequencies_img*1j), color='C3')
    plt.show()

    # plot GPU parallel sequential

    threads = 500
    frequencies_real = np.zeros(int(N/2+1), dtype=np.float)
    frequencies_img = np.zeros(int(N/2+1), dtype=np.float)
    kernel_parallel_sequential[1, threads](sig_sum, frequencies_real, frequencies_img,threads)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    xf = np.linspace(0, SAMPLING_RATE_HZ/2, int(N/2+1), endpoint=True)
    for sig in sigs:
        ax1.plot(x, sig, lw=0.5, color='#333333', alpha=0.5)

    ax1.plot(x, sig_sum)
    ax2.plot(xf, abs(frequencies_real+frequencies_img*1j), color='C3')
    plt.show()

##
## SETUP
##

SAMPLING_RATE_HZ = 100
TIME_S = 5 # Use only integers for correct DFT results
N = SAMPLING_RATE_HZ * TIME_S

x = np.linspace(0, TIME_S, int(N), endpoint=False)

# Define a group of signals and add them together
sigs = [ np.sin(x * (2*pi) * (i+1) * 2 + i*pi/16) / (i+1) for i in range(24) ]
sig_sum = np.array(sum(sigs) / len(sigs)) + 0.05

##
## Call timing functions
##

# time_cpu()
# time_gpu_seq()
# time_gpu_par()
# time_gpu_par_seq(1)
# time_gpu_par_seq(5)
# time_gpu_par_seq(50)
# time_gpu_par_seq(100)
# time_gpu_par_seq(250)
# time_gpu_par_seq(500)

##
## Call plotting functions
##

#create_plots()     # Uncomment to show the plots

threads = 500
stepsize = 1
array_times = np.zeros(int(threads / stepsize), dtype=np.float)
array_threads = np.zeros(int(threads / stepsize), dtype=np.integer)
times(array_times,array_threads,threads,stepsize)