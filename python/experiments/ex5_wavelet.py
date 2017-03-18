"""

Experiment #5
---------------------------------
Wavelet

"""
import math
import time as timer

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from obspy.signal.tf_misfit import cwt
import pywt

import python.morfas.beatspectre as bs
import python.morfas.tools as tools
from wavelets import WaveletAnalysis

if __name__ == '__main__':
    audio_path = "/home/palsol/CLionProjects/MASLib/data/Wesley Don't Surf.wav"

    data, rate = sf.read(audio_path, always_2d=True)

    start = 0
    time_start = start + 0
    time_end = start + 60

    npts = math.ceil((time_end - time_start) * rate)
    print(npts)
    data = data[time_start * rate: time_end * rate, 0]

    nfft = 512
    noverlap = 256
    pad_to = None
    f_min = 1
    f_max = rate / 2
    dt = 1 / rate
    print(dt)

    then = timer.time()

    wa = WaveletAnalysis(data, dt=dt, dj=0.20)
    scalogram = wa.wavelet_power
    T, F = np.meshgrid(wa.time, wa.fourier_periods)
    freqs = 1 / F

    print(scalogram.shape)
    scalogram = tools.compress_scalogram(scalogram, 512)
    now = timer.time()
    diff = int(now - then)
    minutes, seconds = diff // 60, diff % 60
    print('wavelet time: ' + str(minutes) + ':' + str(seconds).zfill(2))

    spectrogram, freq, time = mlab.specgram(data, NFFT=nfft, Fs=rate,
                                            detrend=None, window=None,
                                            noverlap=noverlap, pad_to=pad_to,
                                            sides=None,
                                            scale_by_freq=None,
                                            mode=None)
    log_spectrogram = -1 * np.log(spectrogram)
    log_spectrogram /= np.nanmax(log_spectrogram)
    log_spectrogram = 1 - log_spectrogram

    log_spectrogram = tools.compress_spectrum(log_spectrogram, 8)

    # scalogram = tools.compress_scalogram(scalogram, 2)
    # scalogram = abs(scalogram)

    # print(log_spectrogram.shape)
    # beat_spectrum1, compare_result1 = bs.beat_spectre(log_spectrogram)
    # print(scalogram.shape)
    # beat_spectrum, compare_result = bs.beat_spectre(scalogram)

    # wa1 = WaveletAnalysis(beat_spectrum1,
    #                      dt=(time_end-time_start)/beat_spectrum1.shape[0],
    #                      dj=0.001)
    # scalogram_bs = wa1.wavelet_power
    # T, F = np.meshgrid(wa1.time, wa1.fourier_periods)
    # freqs1 = 1 / F


    # Normalize for plot
    freq_scale = f_max - f_min
    # print(beat_spectrum.shape)
    # print(np.nanmax(beat_spectrum))
    # beat_spectrum /= np.nanmax(beat_spectrum)
    # beat_spectrum *= freq_scale
    # beat_spectrum += f_min
    #
    # beat_spectrum1 /= np.nanmax(beat_spectrum1)
    # beat_spectrum1 *= freq_scale
    # beat_spectrum1 += f_min
    # print(np.nanmax(compare_result))
    fig = plt.figure(figsize=(10, 20))

    ax1 = fig.add_subplot(311)
    ax1.set_title("scalogram")
    ax1.set_ylabel("Frequency [Hz]")
    ax1.set_xlabel("Time [Seconds]")
    # x = np.linspace(time_start, time_end, beat_spectrum.shape[0])
    ax1.imshow(scalogram, extent=[time_start, time_end, freqs.min(), freqs.max()], cmap='jet', aspect='auto',
               vmax=scalogram.max() / 8, vmin=scalogram.min())
    # ax1.step(x, beat_spectrum, color="w")

    ax2 = fig.add_subplot(312)
    x, y = np.meshgrid(
        np.linspace(time_start, time_end, log_spectrogram.shape[1]),
        np.linspace(freq.min(), freq.max(), log_spectrogram.shape[0]))
    ax2.set_title("spectrogram")
    ax2.set_ylabel("Frequency [Hz]")
    ax2.set_xlabel("Time [Seconds]")
    ax2.pcolormesh(x, y, log_spectrogram, cmap='jet')
    # ax2.step(x[2, :], beat_spectrum1, color="k")

    ax4 = fig.add_subplot(313)
    ax4.set_title("data")
    ax4.set_ylabel("am")
    ax4.set_xlabel("Time")
    x = np.linspace(time_start, time_end, data.shape[0])
    ax4.step(x, data)

    # ax3 = fig.add_subplot(122)
    # x, y = np.meshgrid(
    #     np.linspace(time_start, time_end, scalogram.shape[1]),
    #     np.linspace(time_start, time_end, scalogram.shape[1]))
    # ax3.set_title("compare_result")
    # ax3.set_ylabel("Time [Seconds]")
    # ax3.set_xlabel("Time [Seconds]")
    # ax3.pcolormesh(x, y, compare_result, vmin=0, cmap='jet')

    # ax3 = fig.add_subplot(122)
    # ax3.set_title("scalogram_bs")
    # ax3.set_ylabel("Frequency [Hz]")
    # ax3.set_xlabel("Time [Seconds]")
    # x = np.linspace(time_start, time_end, beat_spectrum.shape[0])
    # ax3.imshow(scalogram_bs, extent=[time_start, time_end, freqs1.min(), freqs1.max()], cmap='jet', aspect='auto',
    #            vmax=scalogram_bs.max(), vmin=scalogram_bs.min())


    plt.tight_layout()
    plt.tight_layout()
    plt.show()
