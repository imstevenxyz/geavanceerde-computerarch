from matplotlib import pyplot as plt
import numpy as np
from numba import cuda
from math import cos, sin, pi
import time
import pandas

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





def generate_image_parallel_striding(period, blocks, threads,image_size):
    image = np.zeros(shape=(image_size,image_size), dtype=np.float)

    kernel_parallel_striding[(blocks,blocks), (1,threads)](image, period)
    return image

def time_N_threads(period, blocks, threads,image_size):
    generate_image_parallel_striding(period, blocks, threads,image_size)
    t_par = synchronous_kernel_timeit(lambda: generate_image_parallel_striding(period, blocks, threads,image_size), number=10)
    return t_par

def result_time_N_threads(period,blocks,threads,image_size):
    N_threads = np.zeros(threads, dtype=np.float)
    times = np.zeros(threads ,dtype=np.float)
    for t in range(threads):
        N_threads[t]  =(t+1)*blocks
        times[t]=time_N_threads(period,32,t+1,image_size)
    result = pandas.DataFrame(times,N_threads,['times'])
    return result

def save_time_N_threads_csv(period,block,threads, image_size):
    result_time_N_threads(period, block, threads, image_size).to_csv("plot/plot_B"+str(block)+"_T"+str(threads)+"_I"+str(image_size))


####################################################
period = 100


#########


save_time_N_threads_csv(period, 32, 1024, 1024)
save_time_N_threads_csv(period, 32, 1024, 512)
save_time_N_threads_csv(period, 16, 1024, 1024)
save_time_N_threads_csv(period, 64, 1024, 1024)


import plotly.express as px
import pandas as pd

df = pandas.read_csv("plot1.csv", index_col=0)
# plot_1.scatter(y='times')

# pd.options.plotting.backend = "plotly"
#
# fig = df.plot.scatter(
#     x=df.index,
#     y='times'
# )
# fig = df.plot(
#     x=df.index,
#     y='times'
# )
# fig.write_html("plot.html")
# fig.show()

fig, ax = plt.subplots()

ax.plot( df.index, df['times'] )
ax.scatter( df.index, df['times'], label='1' )

ax.legend()

plt.show()

# plot_1.scatter(y='times')

plt.show()