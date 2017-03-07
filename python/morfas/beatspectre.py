"""

MorfAS beat spectre.
Library morphological analysis of signals.
---------------------------------
morfcompy

"""

import time as timer

import matplotlib.mlab as mlab
import numpy as np
import scipy.signal as signal

import python.morfas.morfcomparison as mcl
import python.morfas.tools as tools


def beat_spectre(log_spectrogram, smoothing=None):
    then = timer.time()
    compare_result = mcl.compare_chunk(log_spectrogram, 'y')

    max_res = np.nanmax(compare_result)
    print(max_res)
    beat_spectrum = np.zeros(compare_result.shape[0])

    for k in range(compare_result.shape[0]):
        for i in range(compare_result.shape[0] - k):
            beat_spectrum[k] += (compare_result[i, i + k])

    if smoothing is not None:
        beat_spectrum = tools.smooth(beat_spectrum, window_len=smoothing)[0:compare_result.shape[0]]

    now = timer.time()
    diff = int(now - then)
    minutes, seconds = diff // 60, diff % 60
    print('comparison time: ' + str(minutes) + ':' + str(seconds).zfill(2))

    return beat_spectrum, compare_result


def cross_beat_spectre(log_spectrogram1, log_spectrogram2, smoothing=None):
    then = timer.time()

    compare_result = mcl.chunk_cross_comparison(log_spectrogram1, log_spectrogram2, 'y')
    max_res = np.nanmax(compare_result)
    print(compare_result.shape)
    print(compare_result)
    print(max_res)

    beat_spectrum = np.zeros(compare_result.shape[0])

    for k in range(compare_result.shape[0]):
        for i in range(compare_result.shape[0] - k):
            if np.isinf(compare_result[i, i + k]):
                beat_spectrum[k] += 2 * beat_spectrum[k] / k
            else:
                beat_spectrum[k] += compare_result[i, i + k]

    if smoothing is not None:
        beat_spectrum = tools.smooth(beat_spectrum, window_len=smoothing)[0:compare_result.shape[0]]

    now = timer.time()
    diff = int(now - then)
    minutes, seconds = diff // 60, diff % 60
    print('comparison time: ' + str(minutes) + ':' + str(seconds).zfill(2))

    return beat_spectrum, compare_result


def plotting_bs_for_raw(data, rate, nfft, noverlap, pad_to=None):
    spectrogram, freq, time = mlab.specgram(x=data, NFFT=nfft, Fs=rate,
                                            detrend=None, window=None,
                                            noverlap=noverlap, pad_to=pad_to,
                                            sides=None,
                                            scale_by_freq=None,
                                            mode=None)

    spectrogram = spectrogram[::-1]
    log_spectrogram = np.log(spectrogram)
    return beat_spectre(log_spectrogram), freq, time


def plotting_cbs_for_raw(data1, data2, rate, nfft, noverlap, pad_to=None):
    spectrogram1, freq, time = mlab.specgram(x=data1, NFFT=nfft, Fs=rate,
                                             detrend=None, window=None,
                                             noverlap=noverlap, pad_to=pad_to,
                                             sides=None,
                                             scale_by_freq=None,
                                             mode=None)
    spectrogram2, freq, time = mlab.specgram(x=data2, NFFT=nfft, Fs=rate,
                                             detrend=None, window=None,
                                             noverlap=noverlap, pad_to=pad_to,
                                             sides=None,
                                             scale_by_freq=None,
                                             mode=None)

    spectrogram1 = spectrogram1[::-1]
    spectrogram2 = spectrogram2[::-1]
    log_spectrogram1 = np.log(spectrogram1)
    log_spectrogram2 = np.log(spectrogram2)
    return cross_beat_spectre(log_spectrogram1, log_spectrogram2, smoothing=4), freq, time


"""

WIP
---------------------------------

"""


def classify_beat_spectrum_extrema(beat_spectrum):
    extrema = signal.argrelextrema(beat_spectrum, np.less, mode='wrap')
    beat_spectrum_max = beat_spectrum[extrema]
    ex = [list(extrema)]

    while len(ex[-1][0]) != 0:
        extrema = signal.argrelextrema(beat_spectrum_max, np.less, mode='wrap')
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
