"""

Experiment #3
---------------------------------
Plot wav file beat spectrum.

"""


import matplotlib.pyplot as plt
import numpy as np

import python.morfas.tools as tools
import python.morfas.beatspectre as bs

if __name__ == '__main__':
    start = 25

    audio_path = "/home/palsol/CLionProjects/MASLib/data/02 - Tight Rope.wav"
    scale = 32
    time_start = start + 0.2
    time_end = start + 40
    nfft = 1024
    noverlap = 0

    spectrogram, freq, time = tools.wav_to_spectrogram(audio_path,
                                                       time_start,
                                                       time_end,
                                                       nfft=nfft,
                                                       noverlap=noverlap)
    spectrogram = spectrogram[::-1]
    spectrogram = tools.compress_spectrum(spectrogram, scale)
    log_spectrogram = -1 * np.log(spectrogram)
    log_spectrogram /= np.nanmax(log_spectrogram)

    beat_spectrum, compare_result = bs.beat_spectre(log_spectrogram, smoothing=4)

    # Normalize for plot
    freq_scale = freq.max() - freq.min()
    print(beat_spectrum)
    print(np.nanmax(beat_spectrum))
    beat_spectrum /= np.nanmax(beat_spectrum)
    beat_spectrum *= freq_scale
    beat_spectrum = beat_spectrum + freq.min()

    fig, axes = plt.subplots(nrows=2, sharex='all', sharey='all', figsize=(10, 20))
    plt.subplot(211)
    plt.imshow(compare_result, cmap='jet', aspect='1')
    plt.subplot(212)
    plt.imshow(log_spectrogram, vmin=0, vmax=1, cmap='jet', aspect='auto', zorder=0,
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
