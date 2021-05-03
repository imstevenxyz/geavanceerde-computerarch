import time
import numpy as np
from numba import cuda
import matplotlib.pyplot as plt

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
def kernel(samples, coeffs, result):
    # Calculate the thread's absolute position within the grid
    x = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    # Set stride equal to the number of threads we have available in either direction
    stride_x = cuda.gridDim.x * cuda.blockDim.x

    for i in range(x, samples.shape[0], stride_x):
        for j in range(coeffs.shape[0]):
           # result[i+j, j] = samples[i]*coeffs[j]
            cuda.atomic.add(result,i+j-(coeffs.shape[0]/2),samples[i]*coeffs[j])


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


########################

coeffs = _kz_coeffs(5, 5)


from plotly import graph_objects as go


timePoints = np.linspace(0,1,500)

x_data = np.linspace(0, 2*np.pi, 102)
y_data_1 = np.sin(x_data)
y_data_2 = np.cos(x_data)

ydata_saw = signal.sawtooth(2*np.pi*5*timePoints)

result = np.zeros(x_data.shape[0])
kernel[(1), (1)](ydata_saw, coeffs, result)



fig = go.Figure()
fig.add_traces(go.Scatter( x=timePoints, y=ydata_saw, name='original'))
fig.add_traces(go.Scatter( x=timePoints, y=result, name='result'))
fig.show()

# dt = 0.1
# t = np.arange(0, 200+dt, dt)
# x = np.sin(2*np.pi*0.5*t)+0.1*np.sin(2*np.pi*0.25*t)
# plt.plot(t, x)
# k = 1
# m = 5
# coeffs = _kz_coeffs(m, k)
# result = np.zeros(t.shape[0])
# kernel[(1), (1)](x, coeffs, result)
# plt.plot(t, result)
# plt.show()

coef = _kz_coeffs(6, 1)
print(coef)
print(coef.shape)
coef = _kz_coeffs(6, 2)
print(coef)
print(coef.shape)
coef = _kz_coeffs(6, 3)
print(coef)
print(coef.shape)
coef = _kz_coeffs(6, 4)
print(coef)
print(coef.shape)
coef = _kz_coeffs(6, 5)
print(coef)
print(coef.shape)