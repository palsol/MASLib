import sys
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import mas
import python.morfas.cumas as cumas
import python.morfas.morfcomparison as mcl
import time as t

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pydub import AudioSegment

# song = AudioSegment.from_mp3("C:/Users/palsol/ClionProjects/MASLib/data/Actress - Untitled 7.mp3")
#
# print(song.frame_rate)
# raw = song.get_array_of_samples()[::2]
# print(len(raw))
#
# print(((len(raw))/song.frame_rate)/60)

def morf_metric(vec1, vec2):
   return (mas.proximity(vec1, vec2) + mas.proximity(vec2, vec1))/2


vec1 = np.array([34, 32, 27, 25 ,23, 21, 22, 23, 24, 23, 23, 24, 22, 20, 18])
# vec2 = vec1[::-1]
vec3 = np.array([34 + 2, 32 + 2, 27 + 2, 25 + 2 ,23 + 2, 21 + 2, 22 + 2, 23 + 2, 24 + 2, 23 + 2, 23 + 2, 24 + 2, 22 + 2, 20+ 2, 18 + 2])
vec2 = np.array([34 - 1, 32 - 2, 27 - 2, 25 - 2 ,23 - 1, 21 - 1, 22 + 1, 23 + 1, 24 + 1.5, 23 - 0.5, 23 - 1, 24 - 0.5, 22 - 1, 20 - 2, 18 - 1])
vec4 = np.array([15, 14, 13, 12 ,12, 13, 14, 15, 13, 12, 11, 10, 11, 12, 13])
vec4 += 10
x_length = 15
x = range(0, x_length)
# fig = plt.figure(figsize=(5, 5))
#
# ax = fig.add_subplot(2,1,1)
#
# ax.plot(x, vec1, 'ro', color="g")
# ax.plot(x, vec2, 'ro', color="r")
# ax.plot(x, vec3, 'ro', color="b")
# ax.axis([0, x_length, 15, np.max(vec1) + 1])
#
# # major ticks every 20, minor ticks every 5
# major_ticks_x = np.arange(0, x_length, 3)
# major_ticks_y = np.arange(15, np.max(vec1)+1, 5)
# ax.set_xticks(major_ticks_x)
# ax.set_yticks(major_ticks_y)
# ax.grid(which='both')
# ax.grid(which='major', alpha=0.5)
#
# ax1 = fig.add_subplot(2,1,2)
# ax1.plot(x, vec4, 'ro', color="g")
# major_ticks_x = np.arange(0, x_length, 3)
# major_ticks_y = np.arange(15, np.max(vec1)+1, 5)
# ax1.set_xticks(major_ticks_x)
# ax1.set_yticks(major_ticks_y)
# ax1.grid(which='both')
# ax1.grid(which='major', alpha=0.5)
#
# plt.show()


vec3 = np.concatenate((vec1[0:4], vec2[4:8]), axis=0)
print(vec1)
print(vec2)
print(vec3)

cudaComparator = cumas.cuMas()

n = 128 * 128 * 2
dim = int(1024/8)
vectors = np.random.randn(n, dim).astype(np.double)
print(vectors.shape[0])
b = np.random.randn(dim).astype(np.double)


start = t.time()
cudaComparator.proximity_vectors_to_b(vectors, b)
print("GPU:  %f secs" % (t.time() - start))

start = t.time()
for i in range(0, n):
    morf_metric(vectors[i], b)
print("CPU:  %f secs" % (t.time() - start))


print(morf_metric(vec2, vec3))
print(morf_metric(vec1, vec3))
print(morf_metric(vec1, vec3) + morf_metric(vec2, vec3))


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


# x = np.arange(12)
# x = x.reshape(3, 4)
#
# print(x)
# x = x.T
# print(x)
# x[0:-1] = x[1:]
# x[-1] = [2, 2, 2]

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
