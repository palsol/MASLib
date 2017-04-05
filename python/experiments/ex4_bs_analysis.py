"""

Experiment #4
---------------------------------
Analysis of wav file beat spectrum.

"""

import os.path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile
import soundfile as sf

import python.morfas.shiftspectre as bs
import python.morfas.tools as tools

if __name__ == '__main__':
    start = 0

    audio_path = "/home/palsol/CLionProjects/MASLib/data/Gershon Kingsley - popcorn (original 1969).wav"
    scale = 8
    time_start = start + 0.2
    time_end = start + 4
    nfft = 2048
    noverlap = 1024

    spectrogram, freq, time = tools.wav_to_spectrogram(audio_path,
                                                       time_start,
                                                       time_end,
                                                       nfft=nfft,
                                                       noverlap=noverlap)
    spectrogram = spectrogram[::-1]
    spectrogram = tools.compress_spectrum(spectrogram, scale)
    log_spectrogram = -1 * np.log(spectrogram)
    log_spectrogram /= np.nanmax(log_spectrogram)

    beat_spectrum, compare_result = bs.shift_spectre_corr(log_spectrogram, smoothing=4)

    beat_spectrum_max, rank_pos = bs.classify_beat_spectrum_extrema(beat_spectrum)

    # Normalize for plot
    freq_scale = freq.max() - freq.min()
    beat_spectrum /= np.nanmax(beat_spectrum)
    beat_spectrum *= freq_scale
    beat_spectrum = beat_spectrum + freq.min()
    beat_spectrum_max = beat_spectrum_max * freq_scale * 0.1 + freq.min()

    fig, axes = plt.subplots(nrows=2, sharex='all', sharey='all', figsize=(10, 20))
    plt.subplot(211)
    plt.imshow(compare_result, vmin=0, vmax=1, cmap='jet', aspect='1')
    plt.subplot(212)
    plt.imshow(log_spectrogram, vmin=0, vmax=1, cmap='jet', aspect='auto', zorder=0,
               extent=(0, compare_result.shape[0], freq.min(), freq.max()))
    plt.plot(beat_spectrum, linewidth=1, zorder=1, color='k')
    plt.plot(beat_spectrum_max, linewidth=1, zorder=1, color='m')
    plt.axis(xmin=0, xmax=compare_result.shape[0])
    fig.subplots_adjust(.1, .1, .9, .9, .0, .0)

    print(rank_pos)

    # data, rate = sf.read(audio_path, always_2d=True)
    #
    # def clip_save(data, rate, start, end):
    #     data = data[start:end, 0]
    #     directory = '/home/palsol/CLionProjects/MASLib/data/res/sound'
    #     if not os.path.exists(directory):
    #         os.makedirs(directory)
    #     scipy.io.wavfile.write('/home/palsol/CLionProjects/MASLib/data/res/sound/clip'
    #                            + str(start) + '_' + str(end) + '.wav', rate, data)
    #
    # clip_save(data, rate, start*rate + 0*1024, start*rate + 78*1024)
    # clip_save(data, rate, start*rate + 78*1024, start*rate + 155*1024)
    # clip_save(data, rate, start*rate + 155*1024, start*rate + 223*1024)

    # fig.savefig('/home/palsol/CLionProjects/MASLib/data/res/PBS/'
    #             + 'pbs_(' + audio_name + ')_'
    #             + str(time_start) + ':' + str(time_end)
    #             + '_' + str(nfft)
    #             + '.png')

    plt.show()
    plt.close(fig)
