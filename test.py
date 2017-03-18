import sys
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import mas
import python.morfas.morfcomparison as mcl

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# fig, ax = plt.subplots()
#
# x = np.arange(0, 200, 1)
# print(x.shape)
# line, = ax.plot(x, x)
#
#
# def animate(i):
#     line.set_ydata(x+i)  # update the data
#     return line,
#
#
# ani = animation.FuncAnimation(fig, animate, np.arange(1, 20),
#                               interval=25, blit=True)
# plt.show()
# y = mcl.MorfComporator(3, 2)
#
# y.out()
# y.push([3,3])
# y.out()
# y.push([2,3])
# y.out()
# y.push([4,3])
# y.out()
#
# print(y.getsiftspectre())


x = np.arange(12)
x = x.reshape(3, 4)

print(x)
x = x.T
print(x)
x[0:-1] = x[1:]
x[-1] = [2, 2, 2]

# print(x)
# with open('data/Sig0909.txt', 'rb') as f:
#     x1, x2, x3, x4, x5 = np.loadtxt(f, delimiter=' ', usecols=(0, 1, 2, 3, 4), unpack=True)
#     print(x1.size)
#
# minB = 6000
# maxB = 7000
# shift = -6
# up = 0
#
# print(mas.proximity(x2[minB:maxB], x1[minB:maxB]))
# print(mas.proximity(x1[minB:maxB], x2[minB:maxB]))
# print(mas.proximity(x3[minB:maxB], x1[minB:maxB]))
# print(mas.proximity(x4[minB:maxB], x1[minB:maxB]))
# print(mas.proximity(x5[minB + shift:maxB + shift], x1[minB:maxB]))
#
# a = mas.getChunkProximity(x1, x1[minB:maxB])
# x = np.arange(x1.size)
# g = np.arange(maxB - minB)
#
# plt.figure(1)
# plt.subplot(511)
# plt.plot(x, np.concatenate((a, np.zeros(maxB - minB)), axis=0), 'c')
# plt.subplot(512)
# plt.plot(x, x1, 'c')
# plt.subplot(513)
# plt.plot(x, np.concatenate((np.zeros(minB - 1), x1[minB:maxB], np.zeros(x1.size - maxB + 1)), axis=0), 'c')

"""plt.figure(1)
plt.subplot(511)
plt.plot(x, x1[minB:maxB], 'c')
plt.subplot(512)
plt.plot(x, x2[minB:maxB], 'c')
plt.subplot(513)
plt.plot(x, x3[minB:maxB], 'c')
plt.subplot(514)
plt.plot(x, x4[minB:maxB], 'c')
plt.subplot(515)
plt.plot(x, x5[minB + shift:maxB + shift], 'c')"""

# plt.show()

# z = np.ones(maxB - minB) * 6
# a = np.asarray(mas.approximation(x2[minB:maxB] + up, z))
# a = np.asarray(mas.approximation(x2[minB:maxB] + up, x1[minB + shift:maxB + shift] + up))
# b = np.asarray(mas.approximation(x1[minB + shift:maxB + shift] + up, x2[minB:maxB] + up))
# b = np.asarray(mas.approximation(x1[minB + shift:maxB + shift], x2[minB:maxB]))
"""print a.size
x = np.arange(a.size)
plt.plot(x, x2[minB + shift:maxB + shift] + up, 'r')
plt.plot(x, x1[minB:maxB] + up, 'g')
plt.plot(x, a, 'c')
#plt.plot(x, b + 0.4, 'c')
plt.show()"""
