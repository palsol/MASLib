"""

Experiment #11
---------------------------------
Cross ss for sinal reflection from the layered structure of the atmosphere.

"""

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np

import python.morfas.shiftspectre as bs

if __name__ == '__main__':
    with open('/home/palsol/CLionProjects/MASLib/data/atmospheric/IZHEVSK_2_h_line2.csv', 'rb') as f:
        x1, y1 = np.loadtxt(f, delimiter=',', usecols=(0, 1), unpack=True)
    with open('/home/palsol/CLionProjects/MASLib/data/atmospheric/IZHEVSK_2_h_line1.csv', 'rb') as f:
        x2, y2 = np.loadtxt(f, delimiter=',', usecols=(0, 1), unpack=True)

    length = 2560
    start_point = 1000
    nfft = 100
    noverlap = 96
    pad_to = None

    time_start = x1[start_point]
    time_end = x1[-1]

    fs = (time_end - time_start)/x2.shape[0]
    print(fs)
    print(len(x2) / fs)
    time_end = time_start + (length - start_point) * fs

    spectrogram1, freq1, time = mlab.specgram(x=y1[start_point:length], NFFT=nfft, Fs=fs,
                                             detrend=None, window=None,
                                             noverlap=noverlap, pad_to=pad_to,
                                             sides=None,
                                             scale_by_freq=None,
                                             mode=None)
    spectrogram2, freq2, time = mlab.specgram(x=y2[start_point:length], NFFT=nfft, Fs=fs,
                                             detrend=None, window=None,
                                             noverlap=noverlap, pad_to=pad_to,
                                             sides=None,
                                             scale_by_freq=None,
                                             mode=None)
    spectrogram1 = spectrogram1[::-1]
    spectrogram2 = spectrogram2[::-1]
    log_spectrogram1 = np.log(spectrogram1)
    log_spectrogram2 = np.log(spectrogram2)
    beat_spectrum, compare_result = bs.cross_shift_spectre(log_spectrogram1, log_spectrogram2, smoothing=4)

    # Normalize for plot
    # freq_scale = freq.max() - freq.min()
    # print(beat_spectrum)
    # print(np.nanmax(beat_spectrum))
    # beat_spectrum /= np.nanmax(beat_spectrum)
    # beat_spectrum *= freq_scale
    # beat_spectrum = beat_spectrum + freq.min()
    max = log_spectrogram1.max()
    fig = plt.figure(figsize=(10, 20))
    ax1 = fig.add_subplot(221)
    x, y = np.meshgrid(
        np.linspace(time_start, time_end, log_spectrogram1.shape[1]),
        np.linspace(freq1[-1], freq1[0], log_spectrogram1.shape[0]))
    ax1.set_title("signal1")
    ax1.set_ylabel("Frequency [Hz]")
    ax1.set_xlabel("Time [Seconds]")
    ax1.pcolormesh(x, y, log_spectrogram1, vmax=max, cmap='jet')
    # ax1.step(x[2,:], beat_spectrum, color="w")

    ax2 = fig.add_subplot(223)
    x, y = np.meshgrid(
        np.linspace(time_start, time_end, log_spectrogram2.shape[1]),
        np.linspace(freq2[-1], freq2[0], log_spectrogram2.shape[0]))
    ax2.set_title("signal2")
    ax2.set_ylabel("Frequency [Hz]")
    ax2.set_xlabel("Time [Seconds]")
    ax2.pcolormesh(x, y, log_spectrogram2, vmax=max, cmap='jet')

    ax3 = fig.add_subplot(122)
    x, y = np.meshgrid(
        np.linspace(time_start, time_end, compare_result.shape[1]),
        np.linspace(time_start, time_end, compare_result.shape[1]))
    ax3.set_title("morf_comparison")
    ax3.set_ylabel("Time [Seconds]")
    ax3.set_xlabel("Time [Seconds]")
    ax3.pcolormesh(x, y, compare_result, cmap='jet')

    # fig, axes = plt.subplots(nrows=2, ncols=2, sharex='all', sharey='all', figsize=(10, 20))
    # plt.subplot(122)
    # plt.imshow(compare_result, vmin=0, vmax=1, cmap='jet', aspect='1')
    # plt.subplot(221)
    # plt.imshow(log_spectrogram1, cmap='jet', aspect='auto', zorder=0,
    #            extent=(0, compare_result.shape[0], freq.min(), freq.max()))
    # plt.subplot(223)
    # plt.imshow(log_spectrogram2, cmap='jet', aspect='auto', zorder=0,
    #            extent=(0, compare_result.shape[0], freq.min(), freq.max()))
    # plt.plot(beat_spectrum, linewidth=1, zorder=1, color='k')
    #
    # plt.axis(xmin=0, xmax=compare_result.shape[0])
    # fig.subplots_adjust(.1, .1, .9, .9, .0, .0)

    # directory = '/home/palsol/CLionProjects/MASLib/data/res/PBS'
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    #
    # fig.savefig('/home/palsol/CLionProjects/MASLib/data/res/PBS/'
    #             + 'pbs_(' + audio_name + ')_'
    #             + str(time_start) + ':' + str(time_end)
    #             + '_' + str(nfft)
    #             + '.png')

    plt.tight_layout()
    plt.show()
    plt.close(fig)
