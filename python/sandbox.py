import mas
import os.path
import time as timer

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
from numpy.lib.stride_tricks import as_strided
from scipy.signal import argrelextrema
import soundfile as sf


# diplom

def smooth(x, window_len=16, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


def wav_to_spectrogram(wav_file, time_start, time_end, nfft=2048, noverlap=None, pad_to=None):
    if noverlap is None:
        noverlap = nfft / 2  # same default noverlap

    # rate, data = wav.read(wav_file)
    data, rate = sf.read(wav_file, always_2d=True)
    data = data[rate * time_start:rate * time_end, 0]

    # fs = 10e5
    # N = 1e5
    # amp = 2 * np.sqrt(2)
    # noise_power = 0.001 * fs / 2
    # time = np.arange(N)
    # freq = np.linspace(1e3, 2e3, N)
    # x = np.sin(2*np.pi*10000*time / fs)
    # data = x
    # rate = fs

    spec, freqs, t = mlab.specgram(x=data, NFFT=nfft, Fs=rate,
                                   detrend=None, window=None,
                                   noverlap=noverlap, pad_to=pad_to,
                                   sides=None,
                                   scale_by_freq=None,
                                   mode=None)
    return spec, freqs, t


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
                        result[j][i] = result[i][j]
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


def compress_spectrum(spectrogram):
    spectrogram_smooth = np.apply_along_axis(smooth, 0, spectrogram)
    spectrogram_compresed = np.zeros(spectrogram.shape[0] / 16)
    spectrogram_compresed = spectrogram_smooth[0:spectrogram.shape[0]:16]
    return spectrogram_compresed


def classify_beat_spectrum_extrema(beat_spectrum):
    extrema = argrelextrema(beat_spectrum, np.less, mode='wrap')
    beat_spectrum_max = beat_spectrum[extrema]
    ex = [list(extrema)]

    while len(ex[-1][0]) != 0:
        extrema = argrelextrema(beat_spectrum_max, np.less, mode='wrap')
        beat_spectrum_max = beat_spectrum_max[extrema]
        ex.append(list(extrema))

        # print(len(ex[-1][0]))

    ex.pop()
    # print(ex)
    # print(len(ex))

    beat_spectrum_max = np.zeros(beat_spectrum.shape[0])
    rank_pos = []
    for i in range(len(ex)):
        pos = ex[0][0]
        for j in range(i):
            pos = pos[ex[j + 1][0]]
        # print(pos)
        rank_pos.append(pos)
        beat_spectrum_max[pos] += 1

    return beat_spectrum_max, rank_pos


def beat_spectrum_peak_analysis(peaks):
    return 0


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


def plotting_beat_spectrum(audio_path, audio_name, time_start, time_end, nfft, noverlap=None, pad_to=None):
    if pad_to is None:
        pad_to = nfft

    then = timer.time()

    spectrogram, freq, time = wav_to_spectrogram(audio_path, time_start, time_end, nfft, noverlap, pad_to)
    spectrogram = spectrogram[::-1]
    spectrogram = compress_spectrum(spectrogram)
    log_spectrogram = -1 * np.log(spectrogram)
    log_spectrogram /= np.nanmax(log_spectrogram)

    compare_result = compare_chunk(log_spectrogram, 'y')
    max_res = np.nanmax(compare_result)
    print(compare_result.shape)
    print(max_res)
    nfft = log_spectrogram.shape[0]

    beat_spectrum = np.zeros(compare_result.shape[0])

    for k in range(compare_result.shape[0]):
        for i in range(compare_result.shape[0] - k):
            # beat_spectrum[i] += np.exp(-1 * np.log(compare_result[i, i + j] + 0.0001))
            beat_spectrum[k] += (compare_result[i, i + k])  # *(compare_result.shape[0]/(compare_result.shape[0] - k))

    beat_spectrum = smooth(beat_spectrum, window_len=4)[0:compare_result.shape[0]]
    beat_spectrum_max, rank_pos = classify_beat_spectrum_extrema(beat_spectrum)
    print(rank_pos)
    print(len(rank_pos))

    def analys_beat_spectrum_max(rank_pos, compare_result):
        max_sb_rank = len(rank_pos)
        if max_sb_rank > 1:
            mean_rank_len = len(rank_pos[max_sb_rank - 2])
            peak_proximity = np.zeros([mean_rank_len, mean_rank_len])
            for i in range(mean_rank_len):
                for j in range(mean_rank_len):
                    if i == j:
                        peak_proximity[i][j] = 1
                    else:
                        peak_proximity[i][j] = compare_result[
                            rank_pos[max_sb_rank - 2][i], rank_pos[max_sb_rank - 2][j]]

            print(peak_proximity)
            i = (peak_proximity).argsort(axis=None, kind='mergesort')
            j = np.unravel_index(i, peak_proximity.shape)
            res = np.vstack(j).T
            print(res)

    analys_beat_spectrum_max(rank_pos, compare_result)

    similarity = np.zeros(compare_result.shape[0])
    for k in range(compare_result.shape[0]):
        for i in range(compare_result.shape[0]):
            # beat_spectrum[i] += np.exp(-1 * np.log(compare_result[i, i + j] + 0.0001))
            similarity[k] += (compare_result[i, k])

    now = timer.time()
    diff = int(now - then)
    minutes, seconds = diff // 60, diff % 60
    print('comparison time: ' + str(minutes) + ':' + str(seconds).zfill(2))

    # Normalize for plot
    freq_scale = freq.max() - freq.min()
    compare_result /= max_res
    beat_spectrum /= np.nanmax(beat_spectrum)
    beat_spectrum *= freq_scale
    beat_spectrum = beat_spectrum + freq.min()
    beat_spectrum_max = beat_spectrum_max * freq_scale * 0.1 + freq.min()
    similarity /= np.nanmax(similarity)
    similarity *= freq_scale / 2
    similarity = similarity + freq.min()

    fig, axes = plt.subplots(nrows=2, sharex='all', sharey='all', figsize=(10, 20))
    plt.subplot(211)
    plt.imshow(compare_result, cmap='jet', aspect='1')
    plt.subplot(212)
    plt.imshow(log_spectrogram, vmin=0, vmax=1, cmap='jet', aspect='auto', zorder=0,
               extent=(0, compare_result.shape[0], freq.min(), freq.max()))
    plt.plot(beat_spectrum, linewidth=1, zorder=1, color='k')
    plt.plot(similarity, linewidth=1, zorder=1, color='w')
    plt.plot(beat_spectrum_max, linewidth=1, zorder=1, color='k')
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
    plt.close(fig)
    # fig = plt.figure(1, figsize=(5, 10))

    # plt.imshow(log_spectrogram, vmin=0, vmax=1, cmap='jet', aspect='auto')
    # fig.savefig('/home/palsol/CLionProjects/MASLib/data/res/PBS/'
    #             + 'pbs_(' + audio_name + ')_' + str(time_start) + ':' + str(time_end) + 'pec.png')


def analyze_clip(audio_path, audio_name, time_start, time_end, nfft, noverlap=None, pad_to=None):
    if pad_to is None:
        pad_to = nfft

    then = timer.time()

    spectrogram, freq, time = wav_to_spectrogram(audio_path, time_start, time_end, nfft, noverlap, pad_to)
    spectrogram = spectrogram[::-1]
    spectrogram = compress_spectrum(spectrogram)
    log_spectrogram = -1 * np.log(spectrogram)
    log_spectrogram /= np.nanmax(log_spectrogram)

    compare_result = compare_chunk(log_spectrogram, 'y')
    max_res = np.nanmax(compare_result)
    nfft = log_spectrogram.shape[0]

    beat_spectrum = np.zeros(compare_result.shape[0])

    for k in range(compare_result.shape[0]):
        for i in range(compare_result.shape[0] - k):
            # beat_spectrum[i] += np.exp(-1 * np.log(compare_result[i, i + j] + 0.0001))
            beat_spectrum[k] += (compare_result[i, i + k])  # *(compare_result.shape[0]/(compare_result.shape[0] - k))

    beat_spectrum = smooth(beat_spectrum, window_len=4)[0:compare_result.shape[0]]
    beat_spectrum_max, rank_pos = classify_beat_spectrum_extrema(beat_spectrum)

    def analys_beat_spectrum_max(rank_pos, compare_result):
        max_sb_rank = len(rank_pos)
        if max_sb_rank > 1:
            mean_rank_len = len(rank_pos[max_sb_rank - 2])
            peak_proximity = np.zeros([mean_rank_len, mean_rank_len])
            for i in range(mean_rank_len):
                for j in range(mean_rank_len):
                    if i == j:
                        peak_proximity[i][j] = 1
                    else:
                        peak_proximity[i][j] = compare_result[
                            rank_pos[max_sb_rank - 2][i], rank_pos[max_sb_rank - 2][j]]

            # print(peak_proximity)
            i = (peak_proximity).argsort(axis=None, kind='mergesort')
            j = np.unravel_index(i, peak_proximity.shape)
            res = np.vstack(j).T
            # print(res)

    analys_beat_spectrum_max(rank_pos, compare_result)

    similarity = np.zeros(compare_result.shape[0])
    for k in range(compare_result.shape[0]):
        for i in range(compare_result.shape[0]):
            # beat_spectrum[i] += np.exp(-1 * np.log(compare_result[i, i + j] + 0.0001))
            similarity[k] += (compare_result[i, k])

    now = timer.time()
    diff = int(now - then)
    minutes, seconds = diff // 60, diff % 60
    print('comparison time: ' + str(minutes) + ':' + str(seconds).zfill(2))

    # Normalize for plot
    compare_result /= max_res
    beat_spectrum /= np.nanmax(beat_spectrum)
    beat_spectrum *= nfft
    beat_spectrum = nfft - beat_spectrum
    beat_spectrum_max = nfft - beat_spectrum_max * nfft * 0.1
    similarity /= np.nanmax(similarity)
    similarity *= nfft / 2
    similarity = nfft / 2 - similarity

    fig, axes = plt.subplots(nrows=1, sharex='all', sharey='all', figsize=(20, 10))
    plt.imshow(log_spectrogram, vmin=0, vmax=1, cmap='jet', aspect='auto', zorder=0,
               extent=(time.min(), time.max(), freq.min(), freq.max()))
    plt.plot(beat_spectrum, linewidth=1, zorder=1, color='k')
    plt.plot(similarity, linewidth=1, zorder=1, color='w')
    plt.plot(beat_spectrum_max, linewidth=1, zorder=1, color='k')
    plt.axis(xmin=0, xmax=compare_result.shape[0])
    # plt.axis('off')
    fig.subplots_adjust(.1, .1, .9, .9, .0, .0)

    directory = '/home/palsol/CLionProjects/MASLib/data/res/PBS/' + 'pbs_(' + audio_name + ')/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    fig.savefig('/home/palsol/CLionProjects/MASLib/data/res/PBS/'
                + 'pbs_(' + audio_name + ')/'
                + str(int(time_start * 10)) + '_' + str(int(time_end * 10))
                + '.png')
    plt.close(fig)


def plotting_spectrum(audio_path, audio_name, time_start, time_end, nfft, noverlap, pad_to=None):
    spectrogram, freq, time = wav_to_spectrogram(audio_path, time_start, time_end, nfft, noverlap, pad_to)
    spectrogram = spectrogram[::-1]
    log_spectrogram = np.log(spectrogram)
    # log_spectrogram /= np.nanmax(log_spectrogram)

    fig, axes = plt.subplots(nrows=1, sharex='all', sharey='all', figsize=(20, 10))
    plt.imshow(log_spectrogram, cmap='jet', aspect='auto', zorder=0,
               extent=(time.min(), time.max(), freq.min(), freq.max()))

    fig.subplots_adjust(.1, .1, .9, .9, .0, .0)

    directory = '/home/palsol/CLionProjects/MASLib/data/res/spectrum'
    if not os.path.exists(directory):
        os.makedirs(directory)

    fig.savefig('/home/palsol/CLionProjects/MASLib/data/res/spectrum/'
                + 'spectrum_(' + audio_name + ')_'
                + str(time_start * 10) + ':' + str(time_end * 10)
                + '_' + str(nfft)
                + '.png')

    plt.show()
    plt.close(fig)

if __name__ == '__main__':
    # plotting_spectrum(audio_path="../data/sodar/2.waw",
    #                   audio_name="sodar2",
    #                   time_start=0,
    #                   time_end=9,
    #                   nfft=2048, noverlap=1024)
    start = 25
    plotting_beat_spectrum(audio_path="../data/Wesley Don't Surf.wav",
                           audio_name="Wesley Don't Surf",
                           time_start=start + 0.2,
                           time_end=start + 4,
                           nfft=1024, noverlap=0)

    # for i in range(40):
    #     start = 0.2
    #     analyze_clip(audio_path="../data/Wesley Don't Surf.wav",
    #                  audio_name="Wesley Don't Surf",
    #                  time_start=start + i * 2.5,
    #                  time_end=start + (i + 1) * 2.5 + 2.5,
    #                  nfft=1024, noverlap=0)
