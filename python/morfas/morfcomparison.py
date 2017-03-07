"""

MorfAS morfological comparisons.
Library morphological analysis of signals.
---------------------------------

"""

import mas
import math

import numpy as np


def compare_chunk(data, axis, cmf=None):
    axis_values = ["x", "y", "0", "1"]
    if axis not in axis_values:
        raise ValueError("axis [%s] must be one of %s" %
                         (axis, axis_values))

    if cmf is None:
        if axis in ("x", "0"):
            result = np.zeros((data.shape[0], data.shape[0]))
            print(data.shape)
            for i in range(data.shape[0]):
                for j in range(data.shape[0]):
                    if i < j:
                        result[i][j] = mas.proximity(data[i], data[j])
                    else:
                        result[i][j] = (mas.proximity(data[i], data[j]) + result[j][i]) / 2.0
                        result[j][i] = result[i][j]

        elif axis in ("y", "1"):
            result = np.zeros((data.shape[1], data.shape[1]))
            print(data.shape)
            for i in range(data.shape[1]):
                for j in range(data.shape[1]):
                    if i < j:
                        result[i][j] = mas.proximity(data[:, i], data[:, j])
                    else:
                        result[i][j] = (mas.proximity(data[:, i], data[:, j]) + result[j][i]) / 2.0
                        result[j][i] = result[i][j]
                        if math.isnan(result[i][j]):
                            print(str(result[i][j]) + ' ' + str(i) + ':' + str(j))

    else:
        if axis in ("x", "0"):
            result = np.zeros((data.shape[0], data.shape[0]))
            print(data.shape)
            for i in range(data.shape[0]):
                for j in range(data.shape[0]):
                    if i < j:
                        result[i][j] = mas.proximity(data[i], data[j])
                    else:
                        temp = mas.proximity(data[i], data[j])
                        mc_ratio = temp / result[j][i]
                        if 1 - cmf < mc_ratio and mc_ratio < 1 + cmf:
                            result[i][j] = (temp + result[j][i]) / 2.0
                        else:
                            result[i][j] = (temp + result[j][i]) * 20

                        result[j][i] = result[i][j]

        elif axis in ("y", "1"):
            result = np.zeros((data.shape[1], data.shape[1]))
            print(data.shape)
            for i in range(data.shape[1]):
                for j in range(data.shape[1]):
                    if i < j:
                        result[i][j] = mas.proximity(data[:, i], data[:, j])
                    else:
                        temp = mas.proximity(data[:, i], data[:, j])
                        mc_ratio = temp / result[j][i]
                        if 1 - cmf < mc_ratio and mc_ratio < 1 + cmf:
                            result[i][j] = (temp + result[j][i]) / 2.0
                        else:
                            result[i][j] = (temp + result[j][i]) * 20

                        result[j][i] = result[i][j]

    return result


def chunk_cross_comparison(data1, data2, axis, cmf=None):
    axis_values = ["x", "y", "0", "1"]

    if data1.shape != data2.shape:
        print(data1.shape)
        print(data2.shape)
        raise ValueError("chunks shape must be equal!")

    if axis not in axis_values:
        raise ValueError("axis [%s] must be one of %s" %
                         (axis, axis_values))

    if cmf is None:
        if axis in ("x", "0"):
            result = np.zeros((data1.shape[0], data1.shape[0]))
            print(data1.shape)
            for i in range(data1.shape[0]):
                for j in range(data1.shape[0]):
                    if i < j:
                        result[i][j] = (mas.proximity(data1[i], data2[j]) + mas.proximity(data2[j], data1[i])) / 2

        elif axis in ("y", "1"):
            result = np.zeros((data1.shape[1], data1.shape[1]))
            print(data1.shape)
            for i in range(data1.shape[1]):
                for j in range(data1.shape[1]):
                    result[i][j] = (mas.proximity(data1[:, i], data2[:, j]) +
                                    mas.proximity(data2[:, j], data1[:, i])) / 2
                    if math.isnan(result[i][j]):
                        print(str(result[i][j]) + ' ' + str(i) + ':' + str(j))

    return result


def compare_chunks(data, axis):
    axis_values = ["x", "y", "0", "1"]
    if axis not in axis_values:
        raise ValueError("axis [%s] must be one of %s" %
                         (axis, axis_values))

    if axis in ("x", "0"):
        result = np.zeros((data.shape[1], data.shape[0], data.shape[0]))

        print(result.shape)
        print(data.shape)
        for i in range(data.shape[1]):
            print(i)
            result[i] = compare_chunk(data[:, i], axis)

    elif axis in ("y", "1"):
        result = np.zeros((data.shape[1], data.shape[2], data.shape[2]))

        print(result.shape)
        print(data.shape)
        for i in range(data.shape[1]):
            print(i)
            result[i] = compare_chunk(data[:, i], axis)

    return result
