import time as timer

import matplotlib.pyplot as plt
import numpy as np

import python.morfas.morfcomparison as mcl
import python.morfas.shiftspectre as ss
import python.morfas.tools as tools


def morfcomporator():
    audio_path = "/home/palsol/CLionProjects/MASLib/data/Gershon Kingsley - popcorn (original 1969).wav"
    scale = 8

    start = 10
    time_start = start + 0
    time_end = start + 60 * 0.25
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

    ssdata = ss.ss_with_window(log_spectrogram, win_size, comparison_func='morf')

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
    ssdata = ss.ss_with_window(log_spectrogram, win_size, comparison_func='morf')
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
    ax1.pcolormesh(x, y, log_spectrogram,  cmap='jet')

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
    ax1.pcolormesh(x, y, log_spectrogram,  cmap='jet')
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
    morfcomporator()