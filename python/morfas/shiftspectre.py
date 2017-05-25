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


def shift_spectre(log_spectrogram, smoothing=None):
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
    print('morf comparison time: ' + str(minutes) + ':' + str(seconds).zfill(2))

    return beat_spectrum, compare_result


def shift_spectre_corr(log_spectrogram, smoothing=None):
    then = timer.time()
    compare_result = mcl.compare_chunk_corr(log_spectrogram, 'y')

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
    print('corr comparison time: ' + str(minutes) + ':' + str(seconds).zfill(2))

    return beat_spectrum, compare_result


def cross_shift_spectre(log_spectrogram1, log_spectrogram2, smoothing=None):
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


def plotting_ss_for_raw(data, rate, nfft, noverlap, pad_to=None):
    spectrogram, freq, time = mlab.specgram(x=data, NFFT=nfft, Fs=rate,
                                            detrend=None, window=None,
                                            noverlap=noverlap, pad_to=pad_to,
                                            sides=None,
                                            scale_by_freq=None,
                                            mode=None)

    spectrogram = spectrogram[::-1]
    log_spectrogram = np.log(spectrogram)
    return shift_spectre(log_spectrogram), freq, time


def plotting_css_for_raw(data1, data2, rate, nfft, noverlap, pad_to=None):
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
    return cross_shift_spectre(log_spectrogram1, log_spectrogram2, smoothing=4), freq, time


def ss_with_window(data, win_size, mc=None, comparison_func='corr'):
    if mc is None:
        if comparison_func == 'morf':
            mc = mcl.MorfComporator(win_size, data.shape[0])
        elif comparison_func == 'cumorf':
            mc = mcl.cuMorfComporator(win_size, data.shape[0])
        else:
            mc = mcl.CorrComporator(win_size, data.shape[0])

    ss_result = np.zeros((data.shape[1], win_size))
    for i in range(0, data.shape[1], ):
        ss_result[i] = mc.getshiftspectre()[-1::-1]
        mc.push(data[:, i])

    ss_result = ss_result.T
    return ss_result

def analys_shift_spectrum(ssdata):
    extrema = signal.argrelextrema(ssdata, np.less, mode='wrap')
    if len(extrema[0]) > 1 :
        shift_spectrum_min = ssdata[extrema]
        min_index = np.where(shift_spectrum_min[0:] == shift_spectrum_min[1:].min())[0][0]
        return extrema[0][min_index]
    else:
        return 0


"""

WIP
---------------------------------

"""


def classify_beat_spectrum_extrema(beat_spectrum):
    a = np.array((1000000000, 1000000000, 1000000000, -1000000000, -2000000000, 1000000000))
    b = a[::-1]
    print(len(beat_spectrum))
    ex_beat_spectrum = np.concatenate((a, beat_spectrum), axis=0)
    ex_beat_spectrum = np.concatenate((ex_beat_spectrum, b), axis=0)

    extrema = signal.argrelextrema(ex_beat_spectrum, np.less, mode='wrap')
    beat_spectrum_max = ex_beat_spectrum[extrema]
    ex = [list(extrema)]

    while len(ex[-1][0]) != 0:
        extrema = signal.argrelextrema(beat_spectrum_max, np.less, mode='wrap')
        beat_spectrum_max = beat_spectrum_max[extrema]
        ex.append(list(extrema))

        # print(len(ex[-1][0]))

    ex.pop()
    # print(ex)

    beat_spectrum_max = np.zeros(ex_beat_spectrum.shape[0])
    rank_pos = []
    for i in range(len(ex)):
        pos = ex[0][0]
        for j in range(i):
            pos = pos[ex[j + 1][0]]
        # print(pos)
        rank_pos.append(pos)
        beat_spectrum_max[pos - 6] += 1

    return beat_spectrum_max[0:-12], rank_pos


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
