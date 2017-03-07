"""

Experiment #2
---------------------------------
Plot signal beat spectrum.

"""

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np

import python.morfas.beatspectre as bs

if __name__ == '__main__':
    with open('../../data/explosion/signal1.txt', 'rb') as f:
        x1, x2, x3 = np.loadtxt(f, delimiter=' ', usecols=(0, 1, 2), unpack=True)

    fs = 1 / 0.04
    print(fs)
    print(len(x2) / fs)
    nfft = 50
    noverlap = 46
    pad_to = None

    spectrogram, freq, time = mlab.specgram(x=x1, NFFT=nfft, Fs=fs,
                                            detrend=None, window=None,
                                            noverlap=noverlap, pad_to=pad_to,
                                            sides=None,
                                            scale_by_freq=None,
                                            mode=None)

    spectrogram = spectrogram[::-1]
    log_spectrogram = np.log(spectrogram)
    beat_spectrum, compare_result = bs.beat_spectre(log_spectrogram, smoothing=4)

    # Normalize for plot
    freq_scale = freq.max() - freq.min()
    print(beat_spectrum)
    print(np.nanmax(beat_spectrum))
    beat_spectrum /= np.nanmax(beat_spectrum)
    beat_spectrum *= freq_scale
    beat_spectrum = beat_spectrum + freq.min()

    fig, axes = plt.subplots(nrows=1, sharex='all', sharey='all', figsize=(20, 10))
    plt.subplot(211)
    plt.imshow(compare_result, vmin=0, vmax=1, cmap='jet', aspect='1')
    plt.subplot(212)
    plt.imshow(log_spectrogram, cmap='jet', aspect='auto', zorder=0)
    plt.plot(beat_spectrum, linewidth=1, zorder=1, color='k')
    plt.axis(xmin=0, xmax=compare_result.shape[0])
    fig.subplots_adjust(.1, .1, .9, .9, .0, .0)
    plt.show()

    # directory = '/home/palsol/CLionProjects/MASLib/data/res/PBS/' + 'pbs_(' + name + ')/'
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    #
    # fig.savefig('/home/palsol/CLionProjects/MASLib/data/res/PBS/'
    #             + 'pbs_(' + name + ').png')

    plt.close(fig)
