"""

MorfAS morfological comparisons.
Library morphological analysis of signals.
---------------------------------

"""

import mas
import python.morfas.cumas as cumas
import math

import numpy as np


def corr(vec1, vec2):
    return (mas.proximity(vec1, vec2) + mas.proximity(vec2, vec1)) / 2


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
                        # if math.isnan(result[i][j]):
                        #     print(str(result[i][j]) + ' ' + str(i) + ':' + str(j))

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


def compare_chunk_corr(data, axis, cmf=None):
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
                        result[i][j] = corr(data[i], data[j])
                    else:
                        result[i][j] = (corr(data[i], data[j]) + result[j][i]) / 2.0
                        result[j][i] = result[i][j]

        elif axis in ("y", "1"):
            result = np.zeros((data.shape[1], data.shape[1]))
            print(data.shape)
            for i in range(data.shape[1]):
                for j in range(data.shape[1]):
                    if i < j:
                        result[i][j] = corr(data[:, i], data[:, j])
                    else:
                        result[i][j] = (corr(data[:, i], data[:, j]) + result[j][i]) / 2.0
                        result[j][i] = result[i][j]
                        # if math.isnan(result[i][j]):
                        #     print(str(result[i][j]) + ' ' + str(i) + ':' + str(j))

    else:
        if axis in ("x", "0"):
            result = np.zeros((data.shape[0], data.shape[0]))
            print(data.shape)
            for i in range(data.shape[0]):
                for j in range(data.shape[0]):
                    if i < j:
                        result[i][j] = corr(data[i], data[j])
                    else:
                        temp = corr(data[i], data[j])
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
                        result[i][j] = corr(data[:, i], data[:, j])
                    else:
                        temp = np.correlate(data[:, i], data[:, j])
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


def chunk_cross_comparison_corr(data1, data2, axis, cmf=None):
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
                        result[i][j] = np.correlate(data1[i], data2[j])

        elif axis in ("y", "1"):
            result = np.zeros((data1.shape[1], data1.shape[1]))
            print(data1.shape)
            for i in range(data1.shape[1]):
                for j in range(data1.shape[1]):
                    result[i][j] = np.correlate(data1[:, i], data2[:, j])
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
            result[i] = compare_chunk(data[:, i], axis)

    elif axis in ("y", "1"):
        result = np.zeros((data.shape[1], data.shape[2], data.shape[2]))

        print(result.shape)
        print(data.shape)
        for i in range(data.shape[1]):
            result[i] = compare_chunk(data[:, i], axis)

    return result


def compare_chunks_corr(data, axis):
    axis_values = ["x", "y", "0", "1"]
    if axis not in axis_values:
        raise ValueError("axis [%s] must be one of %s" %
                         (axis, axis_values))

    if axis in ("x", "0"):
        result = np.zeros((data.shape[1], data.shape[0], data.shape[0]))

        print(result.shape)
        print(data.shape)
        for i in range(data.shape[1]):
            result[i] = compare_chunk_corr(data[:, i], axis)

    elif axis in ("y", "1"):
        result = np.zeros((data.shape[1], data.shape[2], data.shape[2]))

        print(result.shape)
        print(data.shape)
        for i in range(data.shape[1]):
            result[i] = compare_chunk_corr(data[:, i], axis)

    return result


class MorfComporator:
    def __init__(self, window_size, data_size):
        self.__window_size = window_size
        self.__data_size = data_size
        self.__storage_size = 0
        self.__data = np.zeros((window_size, data_size))
        self.__comparison_matrix = np.zeros(shape=(window_size, window_size))

    def push(self, item):
        self.__data[0:-1] = self.data[1:]
        self.__data[-1] = item
        self.__comparison_matrix[0:-1] = self.__comparison_matrix[1:]
        self.__storage_size += 1
        if self.__storage_size > self.__window_size:
            for i in range(self.__window_size):
                self.__comparison_matrix[-1, i] = (mas.proximity(self.__data[i], self.__data[-1]) +
                                                   mas.proximity(self.__data[-1], self.__data[i])) / 2

    def getshiftspectre(self):
        return self.__comparison_matrix.sum(axis=0)

    @property
    def data(self):
        return self.__data

    @property
    def comparison_matrix(self):
        return self.__comparison_matrix

    def out(self):
        print(self.__data)
        print(self.__comparison_matrix)

    def __repr__(self):
        return "MorfComporator() " + str(self.size)

    def __str__(self):
        return "MorfComporator " + str(self.size)

    # def isEmpty(self):
    #     return self.items == []
    #
    # def enqueue(self, item):
    #     self.items.insert(0,item)
    #
    # def dequeue(self):
    #     return self.items.pop()

    def size(self):
        return len(self.__data)


class cuMorfComporator:
    def __init__(self, window_size, data_size):
        self.__window_size = window_size
        self.__data_size = data_size
        self.__storage_size = 0
        self.__data = np.zeros((window_size, data_size))
        self.__data_forms = np.zeros((window_size, data_size))
        self.__comparison_matrix = np.zeros(shape=(window_size, window_size))
        self.__cudaComparator = cumas.cuMas()

    def push(self, item):
        self.__data[0:-1] = self.data[1:]
        self.__data_forms[0:-1] = self.__data_forms[1:]
        self.__data[-1] = item
        self.__data_forms[-1] = np.argsort(item)
        self.__storage_size += 1
        if self.__storage_size > self.__window_size:
            self.__comparison_matrix[0:-1] = self.__comparison_matrix[1:]
            self.__comparison_matrix[-1] = \
                np.append(self.__cudaComparator.proximity_vectors_to_b(self.__data[0:-1], self.__data[-1],
                                                                       self.__data_forms[0:-1], self.__data_forms[-1]
                                                                       ), 0)

    def getshiftspectre(self):
        return self.__comparison_matrix.sum(axis=0)

    @property
    def data(self):
        return self.__data

    @property
    def comparison_matrix(self):
        return self.__comparison_matrix

    def out(self):
        print(self.__data)
        print(self.__comparison_matrix)

    def __repr__(self):
        return "MorfComporator() " + str(self.size)

    def __str__(self):
        return "MorfComporator " + str(self.size)

    # def isEmpty(self):
    #     return self.items == []
    #
    # def enqueue(self, item):
    #     self.items.insert(0,item)
    #
    # def dequeue(self):
    #     return self.items.pop()

    def size(self):
        return len(self.__data)


class CorrComporator:
    def __init__(self, window_size, data_size):
        self.__window_size = window_size
        self.__data_size = data_size
        self.__data = np.zeros((window_size, data_size))
        self.__comparison_matrix = np.zeros(shape=(window_size, window_size))

    def push(self, item):
        self.__data[0:-1] = self.__data[1:]
        self.__data[-1] = item
        self.__comparison_matrix[0:-1] = self.__comparison_matrix[1:]
        for i in range(self.__window_size):
            self.__comparison_matrix[-1, i] = 1 - np.dot(self.__data[i], self.__data[-1]) / (
                np.linalg.norm(self.__data[i]) * np.linalg.norm(self.__data[-1]))

    def getshiftspectre(self):
        return self.__comparison_matrix[int(self.__comparison_matrix.shape[0] / 4):].sum(axis=0)

    @property
    def data(self):
        return self.__data

    @property
    def comparison_matrix(self):
        return self.__comparison_matrix

    def out(self):
        print(self.__data)
        print(self.__comparison_matrix)

    def __repr__(self):
        return "MorfComporator() " + str(self.size)

    def __str__(self):
        return "MorfComporator " + str(self.size)

    # def isEmpty(self):
    #     return self.items == []
    #
    # def enqueue(self, item):
    #     self.items.insert(0,item)
    #
    # def dequeue(self):
    #     return self.items.pop()

    def size(self):
        return len(self.__data)
