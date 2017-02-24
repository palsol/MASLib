from numpy.lib.stride_tricks import as_strided
import matplotlib.mlab as mlab
from scipy import signal
import time
import numpy as np
import matplotlib.pyplot as plt
import mas
import os.path

import scipy.io.wavfile as wav
from numpy.lib import stride_tricks


# diplom

def wav_to_spectrogram(wav_file, time_start, time_end, nfft=2048):
    rate, data = wav.read(wav_file)
    data = data[rate * time_start:rate * time_end, 0]
    spec, freqs, t = mlab.specgram(x=data, NFFT=nfft, Fs=rate,
                                   detrend=None, window=None,
                                   noverlap=128, pad_to=None,
                                   sides=None,
                                   scale_by_freq=None,
                                   mode=None)
    return spec / np.amax(spec)


def split_to_slide_chunks(data, chunk_size):
    data = np.array(data)
    data_item_size = data.itemsize
    data_length = data.shape[0]
    print(data_length, data_item_size)
    chunks = as_strided(data, strides=(data_item_size, data_item_size),
                        shape=(data_length - chunk_size + 1, chunk_size))
    return chunks


def split_to_chunks(data, chunk_size):
    print(data.shape)
    num_chunks = data.shape[1] // chunk_size
    chunk_height = data.shape[0]
    data_item_size = data.itemsize
    chunks = np.empty((chunk_height, num_chunks, chunk_size))

    for i in range(data.shape[0]):
        buff_data = np.array(data[i])

        chunks[i] = as_strided(buff_data, strides=(data_item_size * chunk_size, data_item_size),
                               shape=(num_chunks, chunk_size))
    return chunks


def compare_chunk(data, axis):
    axis_values = ["x", "y", "0", "1"]
    if axis not in axis_values:
        raise ValueError("axis [%s] must be one of %s" %
                         (axis, axis_values))

    if axis in ("x", "0"):
        result = np.zeros((data.shape[0], data.shape[0]))

        # print(result.shape)
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

        # print(result.shape)
        print(data.shape)
        for i in range(data.shape[1]):
            for j in range(data.shape[1]):
                if i < j:
                    result[i][j] = mas.proximity(data[:, i], data[:, j])
                else:
                    result[i][j] = (mas.proximity(data[:, i], data[:, j]) + result[j][i]) / 2.0
                    result[j][i] = result[i][j]
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


def plotting_comparison_between_x_chunks(chunk_size, audio_path, time_start, time_end):
    distance = mas.distance(np.ones(chunk_size), (-1 * np.ones(chunk_size)))
    print("distance =", distance)

    spectrogram = wav_to_spectrogram(audio_path, time_start, time_end)
    spectrogram = spectrogram[::-1]
    log_spectrogram = -1 * np.log(spectrogram)
    log_spectrogram /= np.nanmax(log_spectrogram)

    print(np.amax(spectrogram))
    spectrogram_chunks = split_to_chunks(spectrogram, chunk_size)
    print(spectrogram_chunks.shape)

    compare_results = compare_chunks(spectrogram_chunks, 'y')
    # compare_results = -1 * np.log(compare_results)
    max_res = np.nanmax(compare_results)
    print(compare_results)
    print(max_res)
    compare_results = compare_results
    print(compare_results)

    directory = '/home/palsol/CLionProjects/MASLib/data/res/cs(' + str(chunk_size) + ')'
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in range(compare_results.shape[0]):
        compare_result = compare_results[i]
        spec_chunk = log_spectrogram[:, i * chunk_size: i * chunk_size + chunk_size]

        fig, axes = plt.subplots(ncols=2, sharex='all', sharey='all', figsize=(20, 10))
        plt.subplot(121)
        plt.imshow(compare_result, vmin=0, vmax=1, cmap='jet', aspect='1')
        plt.subplot(122)
        plt.imshow(spec_chunk, vmin=0, vmax=1, cmap='jet', aspect='auto')
        # plt.axis('off')
        fig.subplots_adjust(.1, .1, .9, .9, .0, .0)
        fig.savefig('/home/palsol/CLionProjects/MASLib/data/res/cs(' + str(chunk_size) + ')/'
                    + str(i) + '.png')
        # plt.show()

    fig = plt.figure(1, figsize=(5, 10))
    plt.imshow(log_spectrogram, vmin=0, vmax=1, cmap='jet', aspect='auto')
    fig.savefig('/home/palsol/CLionProjects/MASLib/data/res/cs(' + str(chunk_size) + ')/spec.png')
    # x1 = spectrogram[10]
    # x2 = spectrogram[11]
    # x3 = spectrogram[13]
    # x4 = spectrogram[17]
    # print(np.amax(spectrogram[10]))
    # print(mas.proximity(x1, x2))
    # print(mas.proximity(x1, x3))
    # print(mas.proximity(x1, x4))
    #
    # x = np.arange(x1.size)

    # plt.figure(1)
    # plt.subplot(511)
    # plt.plot(x, x1, 'c')
    # plt.subplot(512)
    # plt.plot(x, x2, 'c')
    # plt.subplot(513)
    # plt.plot(x, x3, 'c')
    # plt.subplot(514)
    # plt.plot(x, x4, 'c')
    # plt.show()
    #
    # result = compare_chunks(chunks)
    # print(mas.proximity(spectrogram_chunks[10], spectrogram_chunks[11]))

    # test = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    #                  [13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
    #                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]], dtype="float64")
    # test_chunks = split_to_chunks(test, 3)
    # print(test_chunks[:,0])


def plotting_beat_spectrum(audio_path, audio_name, time_start, time_end, nfft):
    spectrogram = wav_to_spectrogram(audio_path, time_start, time_end, nfft)
    spectrogram = spectrogram[::-1]
    log_spectrogram = -1 * np.log(spectrogram)
    log_spectrogram /= np.nanmax(log_spectrogram)

    then = time.time()
    compare_result = compare_chunk(log_spectrogram, 'y')
    max_res = np.nanmax(compare_result)
    print(compare_result.shape)
    print(max_res)
    compare_result /= max_res
    beat_spectrum = np.zeros(compare_result.shape[0])

    for i in range(compare_result.shape[0]):
        for j in range(compare_result.shape[0] - i):
            beat_spectrum[i] += np.exp(-1 * np.log(compare_result[i, i + j] + 0.001))

    beat_spectrum /= np.nanmax(beat_spectrum)
    beat_spectrum *= nfft
    beat_spectrum = nfft - beat_spectrum

    now = time.time()
    diff = int(now - then)
    minutes, seconds = diff // 60, diff % 60
    print('comparison time: ' + str(minutes) + ':' + str(seconds).zfill(2))

    fig, axes = plt.subplots(nrows=2, sharex='all', sharey='all', figsize=(10, 20))
    plt.subplot(211)
    plt.imshow(compare_result, vmin=0, vmax=1, cmap='jet', aspect='1')
    plt.subplot(212)
    plt.imshow(log_spectrogram, vmin=0, vmax=1, cmap='jet', aspect='auto', zorder=0)
    plt.plot(beat_spectrum, linewidth=1, zorder=1, color='k')
    plt.axis(xmin=0, xmax=compare_result.shape[0])
    # plt.axis('off')
    fig.subplots_adjust(.1, .1, .9, .9, .0, .0)

    directory = '/home/palsol/CLionProjects/MASLib/data/res/PBS'
    if not os.path.exists(directory):
        os.makedirs(directory)

    fig.savefig('/home/palsol/CLionProjects/MASLib/data/res/PBS/'
                + 'pbs_(' + audio_name + ')_'
                + str(time_start) + ':' + str(time_end)
                + '_' + str(nfft)
                + '.png')

    plt.show()

    # fig = plt.figure(1, figsize=(5, 10))

    # plt.imshow(log_spectrogram, vmin=0, vmax=1, cmap='jet', aspect='auto')
    # fig.savefig('/home/palsol/CLionProjects/MASLib/data/res/PBS/'
    #             + 'pbs_(' + audio_name + ')_' + str(time_start) + ':' + str(time_end) + 'pec.png')


if __name__ == '__main__':
    plotting_beat_spectrum(audio_path="../data/d26.wav",
                           audio_name="d26",
                           time_start=1,
                           time_end=3,
                           nfft=1024)
