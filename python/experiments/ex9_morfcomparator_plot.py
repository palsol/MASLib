import time as timer

import matplotlib.pyplot as plt
import numpy as np

import python.morfas.morfcomparison as mcl
import python.morfas.shiftspectre as ss
import python.morfas.tools as tools

if __name__ == '__main__':
    audio_path = "../../data/Gershon Kingsley - popcorn (original 1969).wav"
    scale = 8

    start = 1
    time_start = start + 0
    time_end = start + 8
    nfft = 1024
    noverlap = 512
    win_size_t = 0.5

    spectrogram, freq, time = tools.wav_to_spectrogram(audio_path,
                                                       time_start,
                                                       time_end,
                                                       nfft=nfft,
                                                       noverlap=noverlap)
    # time_end = start + time.max()comparison_func='morf'
    spectrogram = spectrogram[0:200]
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

    ssdata = ss.ss_with_window(log_spectrogram, win_size)

    fig = plt.figure(figsize=(10, 20))
    ssdata_without_nan = ssdata[np.logical_not(np.isnan(ssdata))]
    ssdata_without_nan = ssdata_without_nan[np.logical_not(np.isinf(ssdata_without_nan))]
    ssdata_max = np.percentile(ssdata_without_nan, 99)
    ssdata_min = ssdata_without_nan.min()
    print('min = ' + str(ssdata_min) + ' max = ' + str(ssdata_max))
    ax1 = fig.add_subplot(311)
    x, y = np.meshgrid(
        np.linspace(time_start, time_end, ssdata.shape[1]),
        np.linspace(0, win_size_t, ssdata.shape[0]))
    ax1.set_title("Shiftgramm")
    ax1.set_ylabel("Window time[Seconds]")
    ax1.set_xlabel("Time [Seconds]")
    ax1.pcolormesh(x, y, ssdata, vmin=ssdata_min, vmax=ssdata_max, cmap='jet')
    ax1.axis('tight')

    win_size_t = 4
    win_size1 = int(win_size_t * int(ssdata.shape[1] / clip_length))
    ssdata = ss.ss_with_window(ssdata, win_size1, comparison_func='corr')

    ssdata_without_nan = ssdata[np.logical_not(np.isnan(ssdata))]
    ssdata_without_nan = ssdata_without_nan[np.logical_not(np.isinf(ssdata_without_nan))]
    ssdata_max = np.percentile(ssdata_without_nan, 99)
    ssdata_min = ssdata_without_nan.min()
    print('min = ' + str(ssdata_min) + ' max = ' + str(ssdata_max))
    ax2 = fig.add_subplot(312)
    x, y = np.meshgrid(
        np.linspace(time_start, time_end, ssdata.shape[1]),
        np.linspace(0, win_size_t, ssdata.shape[0]))
    ax2.set_title("Shiftgramm")
    ax2.set_ylabel("Window time[Seconds]")
    ax2.set_xlabel("Time [Seconds]")
    ax2.pcolormesh(x, y, ssdata, vmin=ssdata_min, vmax=ssdata_max, cmap='jet')
    ax2.axis('tight')

    ax3 = fig.add_subplot(313)
    x, y = np.meshgrid(
        np.linspace(time_start, time_end, log_spectrogram.shape[1]),
        np.linspace(freq.min(), freq.max(), log_spectrogram.shape[0]))

    ax3.set_title("Spectrogramm")
    ax3.set_ylabel("Frequency [Hz]")
    ax3.set_xlabel("Time [Seconds]")
    ax3.pcolormesh(x, y, log_spectrogram,  cmap='jet')
    ax3.axis('tight')

    plt.show()
