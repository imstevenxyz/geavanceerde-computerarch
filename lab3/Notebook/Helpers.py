import time
from math import sin, cos, pi

import numpy as np
from numba import cuda
from PIL import Image
import requests


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


def get_lenna():
    """Get the 'Lenna' sample image.

    :return: List containing RGB and grayscale image of Lenna in ndarray form.
    """

    # Fetch image from URL
    url = 'https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png'
    image = Image.open(requests.get(url, stream=True).raw)

    # Convert image to numpy array
    image_array = np.asarray(image, dtype=np.int64)

    # Convert to grayscale
    image_array_gs = np.sum(np.asarray(image), axis=2) / 3

    return [image_array, image_array_gs]

@cuda.jit
def kernel_2D(samples, xmin, xmax, histogram_out):
    '''Use the GPU for generateing a histogram of 2D input data.'''

    # Calculate the thread's absolute position within the grid
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    # Set stride equal to the number of threads we have available in either direction
    stride_x = cuda.gridDim.x * cuda.blockDim.x
    stride_y = cuda.gridDim.y * cuda.blockDim.y

    # Calc the resolution of the histogram
    nbins = histogram_out.shape[0]
    bin_width = (xmax - xmin) / nbins

    for i in range(x, samples.shape[0], stride_x):
        for j in range(y, samples.shape[1], stride_y):
            # Associate each sample in the interval [xmin, xmax) with a bin and update the histogram. Skip outliers.
            bin_number = int((samples[i, j] - xmin) / bin_width)
            if bin_number >= 0 and bin_number < histogram_out.shape[0]:
                cuda.atomic.add(histogram_out, bin_number, 1)  # Prevent race conditions

@cuda.jit
def invert_image(samples, out):
    '''Use the GPU to invert a 2D grayscale image. Designed by Robin Uyttendaele.'''

    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    stride_x = cuda.gridDim.x * cuda.blockDim.x
    stride_y = cuda.gridDim.y * cuda.blockDim.y

    for i in range(x, samples.shape[0], stride_x):
        for j in range(y, samples.shape[1], stride_y):
            out[i, j] = 255 - samples[i, j]

@cuda.jit
def flatten_image(samples, out):
    '''Use the GPU to flatten a 2D grayscale image. '''

    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    stride_x = cuda.gridDim.x * cuda.blockDim.x
    stride_y = cuda.gridDim.y * cuda.blockDim.y

    for i in range(x, samples.shape[0], stride_x):
        for j in range(y, samples.shape[1], stride_y):
            out[i, j] = 127  + (samples[i, j]-127)*0.5

@cuda.jit
def DFT_parallel_k(samples, frequencies):
    """Conduct an DFT on the image using the GPU."""

    x = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    y = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y

    stride_x = cuda.gridDim.x * cuda.blockDim.x
    stride_y = cuda.gridDim.y * cuda.blockDim.y

    N = samples.shape[0]
    M = samples.shape[1]

    for i in range(x, frequencies.shape[0], stride_x):
        for j in range(y, frequencies.shape[1], stride_y):
            for n in range(N):
                for m in range(M):
                    factor = 2*pi*( i*m/M + j*n/N )
                    frequencies[i,j] += samples[m,n] * ( cos(factor) - sin(factor)*1j )