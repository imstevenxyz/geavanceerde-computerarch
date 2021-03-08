from matplotlib import pyplot as plt
import numpy as np
from numba import cuda
from math import cos, sin, pi

@cuda.jit
def kernel_parallel(image_pixels, period):
    '''Use the GPU for generateing an image. In parallel two dimensional.'''

    # Calculate the thread's absolute position within the grid
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    image_pixels[x,y] = (sin(x*2*pi/period)+1)*(sin(y*2*pi/period)+1)*1/4

@cuda.jit
def kernel_parallel_striding(image_pixels, period):
    # Calculate the thread's absolute position within the grid
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y

    # Set stride equal to the number of threads we have available in either direction
    stride_x = cuda.gridDim.x * cuda.blockDim.x
    stride_y = cuda.gridDim.y * cuda.blockDim.y

    for i in range(x, image_pixels.shape[0], stride_x):
        for j in range(y, image_pixels.shape[1], stride_y):
            image_pixels[i,j] = (sin(i*2*pi/period)+1)*(sin(j*2*pi/period)+1)*1/4

def generate_image_parallel(period):
    image = np.zeros(shape=(1024,1024), dtype=np.float)

    # 32x32 = 1024 blocks with 32x32 = 1024 threads (1024 is thread per block limit on GT 730)
    kernel_parallel[(32,32), (32,32)](image, period)
    return image

def generate_image_parallel_quarter_threads(period):
    image = np.zeros(shape=(1024,1024), dtype=np.float)

    # Only use 1/4 the amount of threads, either (32,32), (16,16) or (16,16), (32,32) works
    kernel_parallel[(32,32), (16,16)](image, period)
    return image

def generate_image_parallel_striding(period):
    image = np.zeros(shape=(1024,1024), dtype=np.float)

    kernel_parallel_striding[(32,32), (32,32)](image, period)
    return image

def generate_image_parallel_quarter_threads_striding(period):
    image = np.zeros(shape=(1024,1024), dtype=np.float)

    kernel_parallel_striding[(32,32), (16,16)](image, period)
    return image

def generate_image_parallel_too_much_threads_striding(period):
    image = np.zeros(shape=(1024,1024), dtype=np.float)

    # Use 32 block too much = 32768 threads too much
    kernel_parallel_striding[(33,32), (32,32)](image, period)
    return image

# This gives us the wanted image
image1 = generate_image_parallel(period=100)

# This gives us only a quarter of the wanted image
image2 = generate_image_parallel_quarter_threads(period=100)

# Using too much threads gives an error
#

# Using striding but with correct amount of threads gives us as expected the correct result
image4 = generate_image_parallel_striding(period=100)

# Using striding but with 1/4 threads we now get a correct result but it should take longer
image5 = generate_image_parallel_quarter_threads_striding(period=100)

# Using striding but with too much threads we now get a correct result
image6 = generate_image_parallel_too_much_threads_striding(period=100)

fig, subplots = plt.subplots(2, 3)
subplots[0,0].imshow(image1)
subplots[0,1].imshow(image2)
subplots[0,2].text(0.5, 0.5, 'Error', horizontalalignment='center', verticalalignment='center', color='red')
subplots[1,0].imshow(image4)
subplots[1,1].imshow(image5)
subplots[1,2].imshow(image6)

fig.suptitle('n = 1024x1024')
subplots[0,0].set(ylabel='Without striding')
subplots[1,0].set(ylabel='With striding')
subplots[0,0].set_title('n threads')
subplots[0,1].set_title('(1/4)*n threads')
subplots[0,2].set_title('>n threads')

for subplot in subplots.flat:
    subplot.set_xticklabels([])
    subplot.set_yticklabels([])
    subplot.set_xticks([])
    subplot.set_yticks([])

plt.show()
