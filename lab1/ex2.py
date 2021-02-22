from math import sin, cos, pi
import numpy as np
from matplotlib import pyplot as plt
from numba import cuda
import time

def synchronous_kernel_timeit( kernel, number=1, repeat=1 ):
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
            kernel()
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
def kernel_parallel(samples, frequencies_real, frequencies_img):
    '''Use the GPU for generateing a histogram. In parallel.'''

    # Calculate the thread's absolute position within the grid
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    sample = samples[x]

    for k in range(frequencies_img.shape[0]):
        cuda.atomic.add(frequencies_real, k, ((sample * (cos(2 * pi * k * x / N)))))
        cuda.atomic.add(frequencies_img, k, ((sample * (-1* sin(2 * pi * k * x / N) ))))

@cuda.jit
def kernel_parallel_one(samples, frequencies_real, frequencies_img,threads):
    '''Use the GPU for generateing a histogram. In parallel.'''

    # Calculate the thread's absolute position within the grid
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    for k in range(frequencies.shape[0]):
        for i in range(N/threads):
            cuda.atomic.add(frequencies_real, k, ((samples[x+i*threads]* (cos(2 * pi * k * (x+i*threads) / N)))))
            cuda.atomic.add(frequencies_img, k, ((samples[x+i*threads] * (-1 * sin(2 * pi * k * (x+i*threads) / N)))))



# # Define the sampling rate and observation time
SAMPLING_RATE_HZ = 100
TIME_S = 5 # Use only integers for correct DFT results
N = SAMPLING_RATE_HZ * TIME_S


# Define sample times
x = np.linspace(0, TIME_S, int(N), endpoint=False)
#
# # Define a group of signals and add them together
sigs = [ np.sin(x * (2*pi) * (i+1) * 2 + i*pi/16) / (i+1) for i in range(24) ]
sig_sum = np.array(sum(sigs) / len(sigs)) + 0.05
#
#
# # Initiate the empty frequency components
# frequencies = np.zeros(int(N/2+1), dtype=np.complex)
#
# # Time the sequential CPU function
# t = synchronous_kernel_timeit( lambda: DFT_sequential(sig_sum, frequencies), number=10 )
# print(t)
#
#
# # Reset the results and run the DFT
# frequencies = np.zeros(int(N/2+1), dtype=np.complex)
# DFT_sequential(sig_sum, frequencies)
#
# # Plot to evaluate whether the results are as expected
# fig, (ax1, ax2) = plt.subplots(1, 2)
#
# # Calculate the appropriate X-axis for the frequency components
# xf = np.linspace(0, SAMPLING_RATE_HZ/2, int(N/2+1), endpoint=True)
#
# # Plot all of the signal components and their sum
# for sig in sigs:
#     ax1.plot( x, sig, lw=0.5, color='#333333', alpha=0.5 )
# ax1.plot( x, sig_sum )
#
# # Plot the frequency components
# ax2.plot( xf, abs(frequencies), color='C3' )
#
# plt.show()

########################################################################"

frequencies_real = np.zeros(int(N/2+1), dtype=np.float)
frequencies_img = np.zeros(int(N/2+1), dtype=np.float)
#
# # start = time.time()
# kernel_parallel[1, 500]( sig_sum, frequencies_real, frequencies_img)
# print(time.time() - start)
#
# start = time.time()
# kernel_parallel[1, 500](sig_sum, frequencies_real, frequencies_img)
# print(time.time() - start)

# Reset the results and run the DFT
# frequencies_real = np.zeros(int(N/2+1), dtype=np.float)
# frequencies_img = np.zeros(int(N/2+1), dtype=np.float)
#
#kernel_parallel[1, 500](sig_sum, frequencies_real, frequencies_img)

kernel_parallel_one[1, 500](sig_sum, frequencies_real, frequencies_img,500)
# Plot to evaluate whether the results are as expected
fig, (ax1, ax2) = plt.subplots(1, 2)

# Calculate the appropriate X-axis for the frequency components
xf = np.linspace(0, SAMPLING_RATE_HZ/2, int(N/2+1), endpoint=True)

# Plot all of the signal components and their sum
for sig in sigs:
     ax1.plot( x, sig, lw=0.5, color='#333333', alpha=0.5 )
ax1.plot( x, sig_sum )

# Plot the frequency components
ax2.plot( xf, abs( frequencies_real+frequencies_img*1j ), color='C3' )

plt.show()

#################################################
SAMPLING_RATE_HZ = 100
TIME_S = 5 # Use only integers for correct DFT results
N = SAMPLING_RATE_HZ * TIME_S

x = np.linspace(0, TIME_S, int(N), endpoint=False)

# Define a group of signals and add them together
sigs = [ np.sin(x * (2*pi) * (i+1) * 2 + i*pi/16) / (i+1) for i in range(24) ]
sig_sum = np.array(sum(sigs) / len(sigs)) + 0.05

# Initiate the empty frequency components
frequencies = np.zeros(int(N/2+1), dtype=np.complex)
frequencies_real = np.zeros(int(N/2+1), dtype=np.float)
frequencies_img = np.zeros(int(N/2+1), dtype=np.float)

t_seq = synchronous_kernel_timeit( lambda: DFT_sequential(sig_sum, frequencies), number=10)
t_par = synchronous_kernel_timeit( lambda: kernel_parallel[1,500](sig_sum, frequencies_real, frequencies_img), number=10)

print('1,500')
print( t_seq )
print( t_par )

###############################################"

SAMPLING_RATE_HZ = 100
TIME_S = 10 # Use only integers for correct DFT results
N = SAMPLING_RATE_HZ * TIME_S

x = np.linspace(0, TIME_S, int(N), endpoint=False)

# Define a group of signals and add them together
sigs = [ np.sin(x * (2*pi) * (i+1) * 2 + i*pi/16) / (i+1) for i in range(24) ]
sig_sum = np.array(sum(sigs) / len(sigs)) + 0.05

frequencies_real = np.zeros(int(N/2+1), dtype=np.float)
frequencies_img = np.zeros(int(N/2+1), dtype=np.float)
kernel_parallel[1, 1000](sig_sum, frequencies_real, frequencies_img)

frequencies = np.zeros(int(N/2+1), dtype=np.complex)
frequencies_real = np.zeros(int(N/2+1), dtype=np.float)
frequencies_img = np.zeros(int(N/2+1), dtype=np.float)

t_seq = synchronous_kernel_timeit( lambda: DFT_sequential(sig_sum, frequencies), number=10)
t_par = synchronous_kernel_timeit( lambda: kernel_parallel[1,1000](sig_sum, frequencies_real, frequencies_img), number=10)

print('1,1000')
print( t_seq )
print( t_par )
