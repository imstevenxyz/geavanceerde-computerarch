import time
import numpy as np
from math import sin, cos, pi
from matplotlib import pyplot as plt
from numba import cuda
import math

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

def DFT_sequential(samples, nSamples, frequencies):
    """Execute the DFT sequentially on the CPU.

    :param samples: An array containing discrete time domain signal samples.
    :param frequencies: An empty array where the frequency components will be stored.
    """

    for k in range(frequencies.shape[0]):
        for n in range(nSamples):
            frequencies[k] += samples[n] * (cos(2 * pi * k * n / nSamples) - sin(2 * pi * k * n / nSamples) * 1j)

@cuda.jit
def DFT_kernel_sequential(samples, nSamples, frequencies_real, frequencies_img):
    """Execute the DFT sequentially on the CPU.

    :param samples: An array containing discrete time domain signal samples.
    :param frequencies: An empty array where the frequency components will be stored.
    """

    for k in range(frequencies_real.shape[0]):
        for n in range(nSamples):
            cuda.atomic.add(frequencies_real, k, ((samples[n] * (cos(2 * pi * k * n / nSamples)))))
            cuda.atomic.add(frequencies_img, k, ((samples[n] * (-1 * sin(2 * pi * k * n / nSamples)))))\

@cuda.jit
def kernel_parallel(samples, nSamples, frequencies_real, frequencies_img):
    '''Use the GPU for generateing a histogram. In parallel.'''

    # Calculate the thread's absolute position within the grid
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    sample = samples[x]

    for k in range(frequencies_img.shape[0]):
        cuda.atomic.add(frequencies_real, k, ((sample * (cos(2 * pi * k * x / nSamples)))))
        cuda.atomic.add(frequencies_img, k, ((sample * (-1* sin(2 * pi * k * x / nSamples) ))))

@cuda.jit
# def kernel_parallel_sequential(samples, frequencies_real, frequencies_img,threads):
def kernel_parallel_sequential(samples, nSamples, frequencies_real, frequencies_img, threads, reps):
    '''Use the GPU for generateing a histogram. In parallel.'''

    # Calculate the thread's absolute position within the grid
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    # _N = samples.shape[0]

    for k in range(frequencies_real.shape[0]):
        # for i in range(N/threads):
        # for i in range(_N/threads):
        for i in range(reps):
            cuda.atomic.add(frequencies_real, k, ((samples[x+i*threads]* (cos(2 * pi * k * (x+i*threads) / nSamples)))))
            cuda.atomic.add(frequencies_img, k, ((samples[x+i*threads] * (-1 * sin(2 * pi * k * (x+i*threads) / nSamples)))))

##
## Timing functions
##

def time_cpu(samples, nSamples):
    frequencies = np.zeros(int(nSamples/2+1), dtype=np.complex)
    DFT_sequential(samples, nSamples, frequencies)
    t_cpu = synchronous_kernel_timeit(lambda: DFT_sequential(samples, nSamples, frequencies), number=10)
    return t_cpu

def time_gpu_seq(samples, nSamples):
    frequencies_real = np.zeros(int(nSamples/2+1), dtype=np.float)
    frequencies_img = np.zeros(int(nSamples/2+1), dtype=np.float)
    DFT_kernel_sequential[1, 1](samples, nSamples, frequencies_real, frequencies_img)
    t_seq = synchronous_kernel_timeit(lambda: DFT_kernel_sequential[1,1](samples, nSamples, frequencies_real, frequencies_img), number=10)
    return t_seq

def time_gpu_par(samples, nSamples):
    frequencies_real = np.zeros(int(nSamples/2+1), dtype=np.float)
    frequencies_img = np.zeros(int(nSamples/2+1), dtype=np.float)
    kernel_parallel[1, nSamples](samples, nSamples, frequencies_real, frequencies_img)
    t_par = synchronous_kernel_timeit(lambda: kernel_parallel[1,nSamples](samples, nSamples, frequencies_real, frequencies_img), number=10)
    return t_par

def time_gpu_semi(samples, nSamples, nThreads):
    frequencies_real = np.zeros(int(nSamples/2+1), dtype=np.float)
    frequencies_img = np.zeros(int(nSamples/2+1), dtype=np.float)
    reps = math.ceil(nSamples / nThreads)
    print(nThreads, reps, nSamples)

    kernel_parallel_sequential[1, nThreads](samples, nSamples, frequencies_real, frequencies_img, nThreads, reps)
    t_semi = synchronous_kernel_timeit(lambda: kernel_parallel_sequential[1, nThreads](samples, nSamples, frequencies_real, frequencies_img, nThreads, reps), number=10)
    return t_semi

##
## Plotting functions
##

def plot(time):
    N = SAMPLING_RATE_HZ * time
    x = np.linspace(0, time, int(N), endpoint=False)
    sigs = [ np.sin(x * (2*pi) * (i+1) * 2 + i*pi/16) / (i+1) for i in range(24) ]
    sig_sum = np.array(sum(sigs) / len(sigs)) + 0.05

    # CPU
    frequencies = np.zeros(int(N/2+1), dtype=np.complex)
    DFT_sequential(sig_sum, N, frequencies)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    xf = np.linspace(0, SAMPLING_RATE_HZ/2, int(N/2+1), endpoint=True)
    for sig in sigs:
        ax1.plot(x, sig, lw=0.5, color='#333333', alpha=0.5)

    ax1.plot(x, sig_sum)
    ax2.plot( xf, abs(frequencies), color='C3' )
    plt.show()
    plt.savefig('seq.png')

    # GPU seq
    frequencies_real = np.zeros(int(N/2+1), dtype=np.float)
    frequencies_img = np.zeros(int(N/2+1), dtype=np.float)
    DFT_kernel_sequential[1, 1](sig_sum, N, frequencies_real, frequencies_img)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    xf = np.linspace(0, SAMPLING_RATE_HZ/2, int(N/2+1), endpoint=True)
    for sig in sigs:
        ax1.plot(x, sig, lw=0.5, color='#333333', alpha=0.5)

    ax1.plot(x, sig_sum)
    ax2.plot(xf, abs(frequencies_real+frequencies_img*1j), color='C3')
    plt.show()
    plt.savefig('gpuseq.png')

    # GPU par
    frequencies_real = np.zeros(int(N/2+1), dtype=np.float)
    frequencies_img = np.zeros(int(N/2+1), dtype=np.float)
    kernel_parallel[1, 500](sig_sum, N, frequencies_real, frequencies_img)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    xf = np.linspace(0, SAMPLING_RATE_HZ/2, int(N/2+1), endpoint=True)
    for sig in sigs:
        ax1.plot(x, sig, lw=0.5, color='#333333', alpha=0.5)

    ax1.plot(x, sig_sum)
    ax2.plot(xf, abs(frequencies_real+frequencies_img*1j), color='C3')
    plt.show()
    plt.savefig('gpupar.png')

    # GPU semi 200
    reps = math.ceil(N / 200)
    frequencies_real = np.zeros(int(N/2+1), dtype=np.float)
    frequencies_img = np.zeros(int(N/2+1), dtype=np.float)
    kernel_parallel_sequential[1, 250](sig_sum, N, frequencies_real, frequencies_img, 200, reps)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    xf = np.linspace(0, SAMPLING_RATE_HZ/2, int(N/2+1), endpoint=True)
    for sig in sigs:
        ax1.plot(x, sig, lw=0.5, color='#333333', alpha=0.5)

    ax1.plot(x, sig_sum)
    ax2.plot(xf, abs(frequencies_real+frequencies_img*1j), color='C3')
    plt.show()
    plt.savefig('gpusemi.png')

##
## SETUP
##

SAMPLING_RATE_HZ = 100

def create_samples(time):
    N = SAMPLING_RATE_HZ * time

    x = np.linspace(0, time, int(N), endpoint=False)

    # Define a group of signals and add them together
    sigs = [ np.sin(x * (2*pi) * (i+1) * 2 + i*pi/16) / (i+1) for i in range(24) ]
    sig_sum = np.array(sum(sigs) / len(sigs)) + 0.05

    return sig_sum, N

##
## Call timing functions
##

def log_time(maxTime, stepSize):
    ''' Create timing performance graph for functions:
        CPU, GPU seq and GPU par using dynamic sample size
    '''
    times_cpu = np.zeros(int(maxTime/stepSize), dtype=np.float)
    times_gpu_seq = np.zeros(int(maxTime/stepSize), dtype=np.float)
    times_gpu_par = np.zeros(int(maxTime/stepSize), dtype=np.float)
    sampling_time = np.zeros(int(maxTime/stepSize), dtype=np.int32)

    for i in range(sampling_time.shape[0]):
        sampling_time[i] = (i+1)*stepSize
        sig_sum, N = create_samples((i+1)*stepSize)
        times_cpu[i] = time_cpu(sig_sum, N)
        times_gpu_seq[i] = time_gpu_seq(sig_sum, N)
        times_gpu_par[i] = time_gpu_par(sig_sum, N)

    plt.plot(sampling_time, times_gpu_seq, label='GPU seqential')
    plt.plot(sampling_time, times_gpu_par, label='GPU parallel')
    plt.plot(sampling_time, times_cpu, label='CPU')
    plt.xlabel('DFT sampling time [s]')
    plt.ylabel('Function execution time [s]')
    plt.legend()
    plt.show()

def log_time_semi(nThreads, stepSize):
    ''' Create timing performance graph for 500 samples using dynamic thread
        count
    '''
    array_times = np.zeros(int(nThreads / stepSize), dtype=np.float)
    array_threads = np.zeros(int(nThreads / stepSize), dtype=np.int32)

    sig_sum, N = create_samples(5)

    for i in range(array_threads.shape[0]):
        array_threads[i] = (i+1)*stepSize
        array_times[i] = time_gpu_semi(sig_sum, N, (i+1)*stepSize)
    
    plt.plot(array_threads, array_times)
    plt.show()
    plt.savefig('times.png')

# log_time(10, 1)
log_time_semi(500, 1)

# # 500 samples
# sig_sum, N = create_samples(5)
# print(time_cpu(sig_sum, N))
# print(time_gpu_seq(sig_sum, N))
# print(time_gpu_par(sig_sum, N))
# print(time_gpu_semi(sig_sum, N, 1))
# print(time_gpu_semi(sig_sum, N, 5))
# print(time_gpu_semi(sig_sum, N, 50))
# print(time_gpu_semi(sig_sum, N, 100))
# print(time_gpu_semi(sig_sum, N, 250))
# print(time_gpu_semi(sig_sum, N, 500))

##
## Call plotting functions
##

# plot(5)