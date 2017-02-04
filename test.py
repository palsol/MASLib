import sys
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import mas

with open('data/Sig0909.txt', 'rb') as f:
    x1, x2, x3, x4, x5 = np.loadtxt(f, delimiter=' ', usecols=(0, 1, 2, 3, 4), unpack=True)
    print(x1.size)

minB = 6000
maxB = 7000
shift = -6
up = 0

print(mas.proximity(x2[minB:maxB], x1[minB:maxB]))
print(mas.proximity(x1[minB:maxB], x2[minB:maxB]))
print(mas.proximity(x3[minB:maxB], x1[minB:maxB]))
print(mas.proximity(x4[minB:maxB], x1[minB:maxB]))
print(mas.proximity(x5[minB + shift:maxB + shift], x1[minB:maxB]))

a = mas.getChunkProximity(x1, x1[minB:maxB])
x = np.arange(x1.size)
g = np.arange(maxB - minB)

plt.figure(1)
plt.subplot(511)
plt.plot(x, np.concatenate((a, np.zeros(maxB - minB)), axis=0), 'c')
plt.subplot(512)
plt.plot(x, x1, 'c')
plt.subplot(513)
plt.plot(x, np.concatenate((np.zeros(minB - 1), x1[minB:maxB], np.zeros(x1.size - maxB + 1)), axis=0), 'c')

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

plt.show()

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