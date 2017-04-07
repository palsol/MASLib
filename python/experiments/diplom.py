import time as timer

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

import python.morfas.morfcomparison as mcl
import python.morfas.shiftspectre as ss
import python.morfas.tools as tools

import scipy.signal as signal


def morfcomporator():
    audio_path = "../../data/14 Dadamnphreaknoizphunk - Crocodile Leather Tile.wav"
    scale = 8

    start = 10
    time_start = start + 0
    time_end = start + 12
    nfft = 1024
    noverlap = 512
    win_size_t = 1

    spectrogram, freq, time = tools.wav_to_spectrogram(audio_path,
                                                       time_start,
                                                       time_end,
                                                       nfft=nfft,
                                                       noverlap=noverlap)
    # time_end = start + time.max()comparison_func='morf'
    spectrogram = spectrogram[::]
    spectrogram = tools.compress_spectrum(spectrogram, scale)
    spectrogram[spectrogram == 0] += 10 ** -22
    log_spectrogram = -1 * np.log(spectrogram)
    log_spectrogram /= np.nanmax(log_spectrogram)
    log_spectrogram = 1 - log_spectrogram

    clip_length = time_end - time_start
    time_step = 1.0 / (spectrogram.shape[1] / clip_length)
    frame_step = (spectrogram.shape[1] / clip_length)
    win_size = int(win_size_t * int(spectrogram.shape[1] / clip_length))
    data_size = spectrogram.shape[0]

    print('clip lenght(sec): ' + str(clip_length))
    print('frame step(count): ' + str(frame_step))
    print('time step(sec): ' + str(time_step))
    print('window size(count): ' + str(win_size))
    print('window size(sec): ' + str(win_size_t))

    ssdata = ss.ss_with_window(log_spectrogram, win_size, comparison_func='corr')

    fig = plt.figure(figsize=(10, 15))
    ssdata_without_nan = ssdata[np.logical_not(np.isnan(ssdata))]
    ssdata_without_nan = ssdata_without_nan[np.logical_not(np.isinf(ssdata_without_nan))]
    ssdata_max = np.percentile(ssdata_without_nan, 99)
    ssdata_min = ssdata_without_nan.min()
    print('min = ' + str(ssdata_min) + ' max = ' + str(ssdata_max))
    ax1 = fig.add_subplot(311)
    x, y = np.meshgrid(
        np.linspace(time_start, time_end, ssdata.shape[1]),
        np.linspace(0, 1, ssdata.shape[0]))
    ax1.set_title("Shiftgramm")
    ax1.set_ylabel("Window time[Seconds]")
    ax1.set_xlabel("Time [Seconds]")
    ax1.pcolormesh(x, y, ssdata, vmin=ssdata_min, vmax=ssdata_max, cmap='jet')

    win_size = int(4 * int(spectrogram.shape[1] / clip_length))
    ssdata = ss.ss_with_window(log_spectrogram, win_size, comparison_func='corr')
    ssdata_without_nan = ssdata[np.logical_not(np.isnan(ssdata))]
    ssdata_without_nan = ssdata_without_nan[np.logical_not(np.isinf(ssdata_without_nan))]
    ssdata_max = np.percentile(ssdata_without_nan, 99)
    ssdata_min = ssdata_without_nan.min()
    print('min = ' + str(ssdata_min) + ' max = ' + str(ssdata_max))
    ax1 = fig.add_subplot(312)
    x, y = np.meshgrid(
        np.linspace(time_start, time_end, ssdata.shape[1]),
        np.linspace(0, 4, ssdata.shape[0]))
    ax1.set_title("Shiftgramm")
    ax1.set_ylabel("Window time[Seconds]")
    ax1.set_xlabel("Time [Seconds]")
    ax1.pcolormesh(x, y, ssdata, vmin=ssdata_min, vmax=ssdata_max, cmap='jet')

    ax1 = fig.add_subplot(313)
    x, y = np.meshgrid(
        np.linspace(time_start, time_end, log_spectrogram.shape[1]),
        np.linspace(freq.min(), freq.max(), log_spectrogram.shape[0]))

    ax1.set_title("Spectrogramm")
    ax1.set_ylabel("Frequency [Hz]")
    ax1.set_xlabel("Time [Seconds]")
    ax1.pcolormesh(x, y, log_spectrogram, cmap='jet')

    plt.show()


def t_shiftspectre():
    audio_path = "../../data/Gershon Kingsley - popcorn (original 1969).wav"
    scale = 8

    start = 5
    time_start = start + 0
    time_end = start + 2.5
    nfft = 1024
    noverlap = 512
    win_size_t = 0.5

    spectrogram, freq, time = tools.wav_to_spectrogram(audio_path,
                                                       time_start,
                                                       time_end,
                                                       nfft=nfft,
                                                       noverlap=noverlap)
    # time_end = start + time.max()comparison_func='morf'
    cut_freq = int(nfft / 2 * 3 / 4)
    freq = freq[0:-1 * int(nfft / 2 - cut_freq)]
    spectrogram = spectrogram[0:cut_freq]
    spectrogram = tools.compress_spectrum(spectrogram, scale)
    spectrogram[spectrogram == 0] += 10 ** -22
    log_spectrogram = -1 * np.log(spectrogram)
    log_spectrogram /= np.nanmax(log_spectrogram)
    log_spectrogram = 1 - log_spectrogram

    clip_length = time_end - time_start
    time_step = 1.0 / (spectrogram.shape[1] / clip_length)
    frame_step = (spectrogram.shape[1] / clip_length)
    win_size = int(win_size_t * int(spectrogram.shape[1] / clip_length))
    data_size = spectrogram.shape[0]

    print('clip lenght(sec): ' + str(clip_length))
    print('frame step(count): ' + str(frame_step))
    print('time step(sec): ' + str(time_step))
    print('window size(count): ' + str(win_size))
    print('window size(sec): ' + str(win_size_t))

    mc = mcl.MorfComporator(win_size, data_size)

    freq_graff_coeff_low = 0.005
    freq_graff_coeff_high = 0.990
    fig = plt.figure(figsize=(12, 15))

    ssdata = np.zeros((spectrogram.shape[1], win_size))
    specdata = np.zeros((spectrogram.shape[1], data_size, win_size))
    frame_time = np.zeros(spectrogram.shape[1])
    print(specdata.shape)

    then = timer.time()
    for i in range(0, spectrogram.shape[1], ):
        frame_time[i] = time_start + i * time_step
        ssdata[i] = mc.getshiftspectre()[-1::-1]
        specdata[i] = mc.data.T
        mc.push(log_spectrogram[:, i])
        if i % 10 == 0:
            now = timer.time()
            diff = int(now - then)
            minutes, seconds = diff // 60, diff % 60
            print('step: ' + str(i) + ' time: ' + str(frame_time[i]) + ' comparison time: ' + str(minutes) + ':' + str(
                seconds).zfill(2))

    chunk_time = 6.5
    chunk = int((chunk_time - time_start) / time_step)
    chunk_time = frame_time[chunk]
    print('chunk_time = ' + str(chunk_time))
    print('chunk_step = ' + str(chunk))

    ssdata[chunk] /= np.max(ssdata[chunk])
    t = win_size_t / (len(ssdata[chunk]) - 1)
    extrema = signal.argrelextrema(ssdata[chunk], np.less, mode='wrap')
    shift_spectrum_min = ssdata[chunk][extrema]
    min_index = np.where(shift_spectrum_min[1:] == shift_spectrum_min[1:].min())
    min_time = chunk_time - win_size_t + extrema[0][min_index[0][0] + 1] * t
    min_scale = shift_spectrum_min[min_index[0][0] + 1]
    period_time = min_time - chunk_time + win_size_t
    print(shift_spectrum_min)
    print(min_index)
    print(min_time)
    print(min_scale)
    print(period_time)

    ax1 = fig.add_subplot(311)
    x1 = np.linspace(chunk_time - win_size_t, chunk_time, ssdata[chunk].shape[0])
    ax1.plot(x1, ssdata[chunk], linewidth=2, zorder=1, color='k')
    ax1.axis('tight')
    ax1.scatter(chunk_time - win_size_t + (extrema[0]) * t, ssdata[chunk][extrema], color='r', s=40)

    ax1.arrow(min_time, 0.02, 0, min_scale - 0.15, head_width=0.005, head_length=0.1, fc='k', ec='k')
    ax1.set_ylabel("Scale")
    ax1.set_title("Shiftspectre")

    ax2 = fig.add_subplot(312)
    x, y = np.meshgrid(
        np.linspace(chunk_time - win_size_t, chunk_time, win_size),
        np.linspace(freq.min(), freq.max(), data_size))
    ax2.pcolormesh(x, y, specdata[chunk], cmap='jet')
    t = chunk_time - win_size_t
    while t < chunk_time - period_time:
        ax2.add_patch(
            mpatches.Rectangle((t, freq.max() * freq_graff_coeff_low), period_time, freq.max() * freq_graff_coeff_high,
                               facecolor="red",
                               edgecolor="black",
                               linewidth=4,
                               alpha=0.6,
                               fill=False))
        ax2.add_patch(
            mpatches.Rectangle((t, freq.max() * freq_graff_coeff_low), period_time, freq.max() * freq_graff_coeff_high,
                               facecolor="white",
                               edgecolor="black",
                               linewidth=2,
                               alpha=0.2))

        t += period_time

    ax2.set_ylabel("Frequency [Hz]")
    ax2.xaxis.tick_top()
    ax2.axis('tight')

    ax3 = fig.add_subplot(313)
    x, y = np.meshgrid(
        np.linspace(time_start, time_end, log_spectrogram.shape[1]),
        np.linspace(freq.min(), freq.max(), log_spectrogram.shape[0]))

    ax3.set_ylabel("Frequency [Hz]")
    ax3.set_xlabel("Time [Seconds]")
    ax3.pcolormesh(x, y, log_spectrogram, cmap='jet')
    # chunk_time = (chunk_time - time_start)*(time_end - time_start)/log_spectrogram.shape[1]
    # win_size_t = win_size_t *(time_end - time_start)/log_spectrogram.shape[1]
    # ax3.add_patch(
    #     mpatches.Rectangle((chunk_time, freq.max() * freq_graff_coeff_low), win_size_t, freq.max() * freq_graff_coeff_high,
    #                        facecolor="red",
    #                        edgecolor="black",
    #                        linewidth=2,
    #                        fill=False))
    ax3.add_patch(mpatches.Rectangle((chunk_time - win_size_t, freq.max() * freq_graff_coeff_low), win_size_t,
                                     freq.max() * freq_graff_coeff_high,
                                     facecolor="white",
                                     edgecolor="black",
                                     linewidth=2,
                                     alpha=0.5))
    ax3.add_patch(mpatches.Rectangle((chunk_time - 2 *win_size_t, freq.max() * freq_graff_coeff_low), 2 * win_size_t,
                                     freq.max() * freq_graff_coeff_high,
                                     facecolor="red",
                                     edgecolor="black",
                                     linewidth=2,
                                     fill=False))
    t = chunk_time - win_size_t
    while t < chunk_time - win_size_t + 4 * period_time:
        ax3.add_patch(
            mpatches.Rectangle((t, freq.max() * freq_graff_coeff_low), period_time, freq.max() * freq_graff_coeff_high,
                               facecolor="red",
                               edgecolor="black",
                               linewidth=3,
                               linestyle='dotted',
                               fill=False))
        ax3.add_patch(
            mpatches.Rectangle((t, freq.max() * freq_graff_coeff_low), period_time, freq.max() * freq_graff_coeff_high,
                               facecolor="magenta",
                               edgecolor="black",
                               linewidth=2,
                               alpha=0.2))
        t += period_time

    ax3.axis('tight')

    plt.show()


def shift_spectre():
    audio_path = "/home/palsol/CLionProjects/MASLib/data/Gershon Kingsley - popcorn (original 1969).wav"
    scale = 2

    start = 10
    time_start = start + 0
    time_end = start + 4
    nfft = 1024
    noverlap = 512
    spectrogram, freq, time = tools.wav_to_spectrogram(audio_path,
                                                       time_start,
                                                       time_end,
                                                       nfft=nfft,
                                                       noverlap=noverlap)

    spectrogram = tools.compress_spectrum(spectrogram, scale)
    # /spectrogram[spectrogram == 0] += 10 ** -22
    log_spectrogram = -1 * np.log(spectrogram)
    log_spectrogram /= np.nanmax(log_spectrogram)
    log_spectrogram = 1 - log_spectrogram
    print(spectrogram.shape)
    beat_spectrum_corr, compare_result_corr = ss.shift_spectre_corr(log_spectrogram, smoothing=4)
    # beat_spectrum_morf, compare_result_morf = ss.shift_spectre(log_spectrogram, smoothing=4)

    # Normalize for plot
    freq_scale = freq.max() - freq.min()
    print(beat_spectrum_corr)
    print(np.nanmax(beat_spectrum_corr))
    beat_spectrum_corr /= np.nanmax(beat_spectrum_corr)
    beat_spectrum_corr *= freq_scale
    beat_spectrum_corr = beat_spectrum_corr + freq.min()

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(111)
    x, y = np.meshgrid(
        np.linspace(time_start, time_end, log_spectrogram.shape[1]),
        np.linspace(freq.min(), freq.max(), log_spectrogram.shape[0]))
    x1 = np.linspace(time_start, time_end, log_spectrogram.shape[1])
    ax1.set_title("Shift spectre")
    ax1.set_ylabel("Frequency [Hz]")
    ax1.set_xlabel("Time [Seconds]")
    ax1.pcolormesh(x, y, log_spectrogram, cmap='jet')
    ax1.plot(x1, beat_spectrum_corr, linewidth=2, zorder=1, color='k')

    # fig = plt.figure(figsize=(10, 5))
    # ax1 = fig.add_subplot(111)
    # x, y = np.meshgrid(
    #     np.linspace(time_start, time_end, log_spectrogram.shape[1]),
    #     np.linspace(freq.min(), freq.max(), log_spectrogram.shape[0]))
    # ax1.set_title("Spectrogram")
    # ax1.set_ylabel("Frequency [Hz]")
    # ax1.set_xlabel("Time [Seconds]")
    # ax1.pcolormesh(x, y, log_spectrogram,  cmap='jet')

    # fig = plt.figure(figsize=(10, 5))
    # ax1 = fig.add_subplot(121)
    # x, y = np.meshgrid(
    #     np.linspace(time_start, time_end, compare_result_corr.shape[1]),
    #     np.linspace(time_start, time_end, compare_result_corr.shape[0]))
    # ax1.set_title("Similarity matrix(correlation comparison)")
    # ax1.set_ylabel("Time [Seconds]")
    # ax1.set_xlabel("Time [Seconds]")
    # ax1.pcolormesh(x, y, compare_result_corr,  cmap='jet')
    #
    # ax2 = fig.add_subplot(122)
    # x, y = np.meshgrid(
    #     np.linspace(time_start, time_end, compare_result_morf.shape[1]),
    #     np.linspace(time_start, time_end, compare_result_morf.shape[0]))
    # ax2.set_title("Similarity matrix(morphological comparison)")
    # ax2.set_ylabel("Time [Seconds]")
    # ax2.set_xlabel("Time [Seconds]")
    # ax2.pcolormesh(x, y, compare_result_morf,  cmap='jet')

    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    t_shiftspectre()
