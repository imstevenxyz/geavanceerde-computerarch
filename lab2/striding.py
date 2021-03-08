import time
import numpy as np
from math import sin, cos, pi
from matplotlib import pyplot as plt
from numba import cuda



# @cude.jit
# def functsion(picture, size):
#
#
#     for(i,size)
#         for(j, size)
#             cuda.atomic.add(sin(i*2*pi)
# (sin(i*2*pi/T)+1)*(sin(j*2*pi/T)+1)*1/4)

@cuda.jit
def kernel_2d_image(image_i):

    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    T=100
    image_i[x,y] = (sin(x*2*pi/T)+1)*(sin(y*2*pi/T)+1)*1/4

@cuda.jit
def kernel_2d_image_striding(image_i):
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    T=100

    stride_x = cuda.gridDim.x * cuda.blockDim.x
    stride_y = cuda.gridDim.y * cuda.blockDim.y

    for i in range(x,image_i.shape[0],stride_x):
        for j in range (y, image_i.shape[1],stride_y):
            if (x < image_i.shape[0]) or (y < image_i.shape[1]):
                image_i[i, j] = (sin(i * 2 * pi / T) + 1) * (sin(j * 2 * pi / T) + 1) * 1 / 4



###############################################



image_empty = np.zeros(shape=(1024,1024), dtype=np.float)
image_full = np.zeros(shape=(1024,1024), dtype=np.float)
image_quarter_1 = np.zeros(shape=(1024,1024), dtype=np.float)
image_quarter_2 = np.zeros(shape=(1024,1024), dtype=np.float)

image_striding_1 = np.zeros(shape=(1024,1024), dtype=np.float)
image_striding_2 = np.zeros(shape=(1024,1024), dtype=np.float)
image_striding_3 = np.zeros(shape=(1024,1024), dtype=np.float)
image_striding_4 = np.zeros(shape=(1024,1024), dtype=np.float)
# kernel_2d_image[(1,1),(512,512)](image)
kernel_2d_image[(32,32),(32,32)](image_full)
kernel_2d_image[(16,16),(32,32)](image_quarter_1)
kernel_2d_image[(22,22),(22,22)](image_quarter_2)


kernel_2d_image_striding[(32,32),(32,32)](image_striding_1)
kernel_2d_image_striding[(16,16),(32,32)](image_striding_2)
kernel_2d_image_striding[(64,64),(32,32)](image_striding_3)
kernel_2d_image_striding[(64,1),(32,1)](image_striding_4)

fig, ((ax1, ax2,ax3, ax4),(ay1, ay2,ay3, ay4)) = plt.subplots(2,4)
ax1.imshow(image_empty)
ax2.imshow(image_full)
ax3.imshow(image_quarter_1)
ax4.imshow(image_quarter_2)
ay1.imshow(image_striding_1)
ay2.imshow(image_striding_2)
ay3.imshow(image_striding_3)
ay4.imshow(image_striding_4)
plt.show()