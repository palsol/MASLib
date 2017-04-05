"""

Experiment #6
---------------------------------
Wavelet Ex

"""
import time as timer
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import soundfile as sf
import math

import obspy
from obspy.imaging.cm import obspy_sequential
from obspy.signal.tf_misfit import cwt

import python.morfas.shiftspectre as bs
import python.morfas.tools as tools

if __name__ == '__main__':
    with open('../../data/explosion/signal1.txt', 'rb') as f:
        x1, x2, x3 = np.loadtxt(f, delimiter=' ', usecols=(0, 1, 2), unpack=True)

    rate = 1 / 0.04
    nfft = 150
    noverlap = 149
    pad_to = None

    start = 0
    time_start = start + 7
    time_end = start + 14

    npts = math.ceil((time_end - time_start) * rate)
    print(npts)
    data = x3[time_start * rate: time_end * rate]

    nfft = 150
    noverlap = 149
    f_min = 1
    f_max = 256
    dt = 1 / rate

    then = timer.time()
    scalogram = cwt(data, dt, np.pi * 2, f_min, f_max, nf=1024)
    print(scalogram.shape)
    scalogram = tools.compress_scalogram(scalogram, 1)
    scalogram = tools.compress_spectrum(scalogram, 1)
    print(scalogram.shape)
    now = timer.time()
    diff = int(now - then)
    minutes, seconds = diff // 60, diff % 60
    print('comparison time: ' + str(minutes) + ':' + str(seconds).zfill(2))

    # print(scalogram.shape)
    # print(scalogram)
    spectrogram, freq, time = mlab.specgram(data, NFFT=nfft, Fs=rate,
                                            detrend=None, window=None,
                                            noverlap=noverlap, pad_to=pad_to,
                                            sides=None,
                                            scale_by_freq=None,
                                            mode=None)
    log_spectrogram = -1 * np.log(spectrogram)
    log_spectrogram /= np.nanmax(log_spectrogram)
    log_spectrogram = 1 - log_spectrogram

    scalogram = np.abs(scalogram)
    beat_spectrum, compare_result = bs.shift_spectre(scalogram)

    # Normalize for plot
    freq_scale = f_max - f_min
    print(beat_spectrum.shape)
    print(np.nanmax(beat_spectrum))
    beat_spectrum /= np.nanmax(beat_spectrum)
    beat_spectrum *= freq_scale
    beat_spectrum = beat_spectrum + f_min
    print(np.nanmax(compare_result))
    fig = plt.figure(figsize=(10, 20))

    ax1 = fig.add_subplot(321)
    x, y = np.meshgrid(
        np.linspace(time_start, time_end, scalogram.shape[1]),
        np.linspace(f_min, f_max, scalogram.shape[0]))
    ax1.set_title("scalogram")
    ax1.set_ylabel("Frequency [Hz]")
    ax1.set_xlabel("Time [Seconds]")
    ax1.pcolormesh(x, y, scalogram, cmap='jet')
    ax1.step(x[2,:], beat_spectrum, color="w")

    ax2 = fig.add_subplot(323)
    x, y = np.meshgrid(
        np.linspace(time_start, time_end, log_spectrogram.shape[1]),
        np.linspace(freq.min(), freq.max(), log_spectrogram.shape[0]))
    ax2.set_title("spectrogram")
    ax2.set_ylabel("Frequency [Hz]")
    ax2.set_xlabel("Time [Seconds]")
    ax2.pcolormesh(x, y, log_spectrogram, cmap='jet')

    ax4 = fig.add_subplot(325)
    ax4.set_title("data")
    ax4.set_ylabel("am")
    ax4.set_xlabel("Time")
    x = np.linspace(time_start, time_end, data.shape[0])
    ax4.step(x, data)

    ax3 = fig.add_subplot(122)
    x, y = np.meshgrid(
        np.linspace(time_start, time_end, scalogram.shape[1]),
        np.linspace(time_start, time_end, scalogram.shape[1]))
    ax3.set_title("compare_result")
    ax3.set_ylabel("Time [Seconds]")
    ax3.set_xlabel("Time [Seconds]")
    ax3.pcolormesh(x, y, compare_result, vmin=0, cmap='jet')

    plt.tight_layout()
    plt.tight_layout()
    plt.show()
