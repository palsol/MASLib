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

    start = 20
    time_start = start + 0
    time_end = start + 1

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

    wa = WaveletAnalysis(data, dt=dt, unbias=True)
    scalogram = wa.wavelet_power

    print(scalogram.shape)
    now = timer.time()
    diff = int(now - then)
    minutes, seconds = diff // 60, diff % 60
    print('wavelet time: ' + str(minutes) + ':' + str(seconds).zfill(2))


    fig, ax = plt.subplots(figsize=(20, 10))

    T, F = np.meshgrid(wa.time, wa.fourier_periods)
    freqs = 1 / F
    ax.contourf(T, freqs, wa.wavelet_power, 100)
    ax.set_yscale('log')

    ax.set_ylabel('frequency (Hz)')
    ax.set_xlabel('time (s)')

    ax.set_ylim(100, 10000)

    fig.savefig('alarma_wavelet.png')
