import time
import numpy as np
from matplotlib import pyplot as plt
from numba import cuda

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

def calc_histogram_on_cpu(samples, xmin, xmax, histogram_out):
    '''Use the CPU to generate a histogram of the sampled signal within the defined boundries.
    
    The resulting histogram will contian the following values: [xmin, xmax)
    
    :param x: Observed signal.
    :param xmin: Minimal observed value contained in the histogram.
    :param xmax: Maximal observed value contained in the histogram.
    :param histogram_out: The calculated histogram.
    '''
    
    # Calc the resolution of the histogram
    nbins = histogram_out.shape[0]
    bin_width = (xmax - xmin) / nbins

    # Associate each sample in the interval [xmin, xmax) with a bin and update the histogram. Skip outliers.
    for sample in samples:
        bin_number = int((sample - xmin ) / bin_width)
        if bin_number >= 0 and bin_number < histogram_out.shape[0]:
            histogram_out[bin_number] += 1

@cuda.jit
def kernel(samples, xmin, xmax, histogram_out):
    '''Use the GPU for generateing a histogram.'''

    # Calc the resolution of the histogram
    nbins = histogram_out.shape[0]
    bin_width = (xmax - xmin) / nbins

    # Associate each sample in the interval [xmin, xmax) with a bin and update the histogram. Skip outliers.
    for sample in samples:
        bin_number = int((sample - xmin) / bin_width)
        if bin_number >= 0 and bin_number < histogram_out.shape[0]:
            cuda.atomic.add(histogram_out, bin_number, 1)

@cuda.jit
def kernel_parallel(samples, xmin, xmax, histogram_out):
    '''Use the GPU for generateing a histogram. In parallel.'''

    # Calculate the thread's absolute position within the grid
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    
    # Calc the resolution of the histogram
    nbins = histogram_out.shape[0]
    bin_width = (xmax - xmin) / nbins
    
    sample = samples[x]

    # Associate each sample in the interval [xmin, xmax) with a bin and update the histogram. Skip outliers.
    bin_number = int((sample - xmin) / bin_width)
    if bin_number >= 0 and bin_number < histogram_out.shape[0]:
        cuda.atomic.add(histogram_out, bin_number, 1) # Prevent race conditions

def time_cpu():
    histogram_out = np.zeros(nbins)
    start = time.time()
    calc_histogram_on_cpu(signal, xmin, xmax, histogram_out)
    print('CPU:')
    print(time.time() - start)

def time_gpu_seq():
    histogram_out = np.zeros(nbins)
    t_seq = synchronous_kernel_timeit( lambda: kernel[1,1](signal, xmin, xmax, histogram_out), number=10)
    print('GPU seq:')
    print(t_seq)

def time_gpu_par():
    histogram_out = np.zeros(nbins)
    t_par = synchronous_kernel_timeit( lambda: kernel_parallel[16,512](signal, xmin, xmax, histogram_out), number=10)
    print('GPU par:')
    print(t_par)

def create_plots():
    functions = [
        lambda x: calc_histogram_on_cpu(signal, xmin, xmax, x),
        lambda x: kernel[1,1](signal, xmin, xmax, x),
        lambda x: kernel_parallel[16,512](signal, xmin, xmax, x)
    ]

    fig, ax = plt.subplots(1, 3, sharey=True)

    for i, function in enumerate(functions):
        histogram_out = np.zeros(nbins)
        function(histogram_out)
        ax[i].plot(x_vals, histogram_out)

    plt.show()

# Repeatable results
np.random.seed(0)

# Define the observed signal
signal = np.random.normal(size=8_192, loc=0, scale=1).astype(np.float32)

# Define the range
xmin = -4
xmax = 4

# Calculate x-axis values for plotting reasons (we also need to recalculate the bin width)
nbins = 500
bin_width = (xmax - xmin) / nbins
x_vals = np.linspace( xmin, xmax, nbins, endpoint=False ) + bin_width/2

time_cpu()
time_gpu_seq()
time_gpu_par()
create_plots()
