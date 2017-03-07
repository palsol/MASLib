"""

Experiment #1
---------------------------------
Plot cross signal beat spectrum.

"""

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np

import python.morfas.beatspectre as bs

if __name__ == '__main__':
    with open('../../data/explosion/signal1.txt', 'rb') as f:
        x1, x2, x3 = np.loadtxt(f, delimiter=' ', usecols=(0, 1, 2), unpack=True)

    length = 2000
    fs = 1 / 0.04
    print(fs)
    print(len(x2) / fs)
    nfft = 50
    noverlap = 46
    pad_to = None

    spectrogram1, freq, time = mlab.specgram(x=x2[:length], NFFT=nfft, Fs=fs,
                                             detrend=None, window=None,
                                             noverlap=noverlap, pad_to=pad_to,
                                             sides=None,
                                             scale_by_freq=None,
                                             mode=None)
    spectrogram2, freq, time = mlab.specgram(x=x1[:length], NFFT=nfft, Fs=fs,
                                             detrend=None, window=None,
                                             noverlap=noverlap, pad_to=pad_to,
                                             sides=None,
                                             scale_by_freq=None,
                                             mode=None)

    spectrogram1 = spectrogram1[::-1]
    spectrogram2 = spectrogram2[::-1]
    log_spectrogram1 = np.log(spectrogram1)
    log_spectrogram2 = np.log(spectrogram2)
    beat_spectrum, compare_result = bs.cross_beat_spectre(log_spectrogram1, log_spectrogram2, smoothing=4)

    # Normalize for plot
    freq_scale = freq.max() - freq.min()
    print(beat_spectrum)
    print(np.nanmax(beat_spectrum))
    beat_spectrum /= np.nanmax(beat_spectrum)
    beat_spectrum *= freq_scale
    beat_spectrum = beat_spectrum + freq.min()

    fig, axes = plt.subplots(nrows=2, ncols=2, sharex='all', sharey='all', figsize=(10, 20))
    plt.subplot(122)
    plt.imshow(compare_result, vmin=0, vmax=1, cmap='jet', aspect='1')
    plt.subplot(221)
    plt.imshow(log_spectrogram1, cmap='jet', aspect='auto', zorder=0,
               extent=(0, compare_result.shape[0], freq.min(), freq.max()))
    plt.subplot(223)
    plt.imshow(log_spectrogram2, cmap='jet', aspect='auto', zorder=0,
               extent=(0, compare_result.shape[0], freq.min(), freq.max()))
    plt.plot(beat_spectrum, linewidth=1, zorder=1, color='k')

    plt.axis(xmin=0, xmax=compare_result.shape[0])
    fig.subplots_adjust(.1, .1, .9, .9, .0, .0)

    # directory = '/home/palsol/CLionProjects/MASLib/data/res/PBS'
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    #
    # fig.savefig('/home/palsol/CLionProjects/MASLib/data/res/PBS/'
    #             + 'pbs_(' + audio_name + ')_'
    #             + str(time_start) + ':' + str(time_end)
    #             + '_' + str(nfft)
    #             + '.png')

    plt.show()
    plt.close(fig)
