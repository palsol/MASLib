import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import numpy as np
from moviepy.editor import VideoClip
from moviepy.editor import AudioFileClip
from moviepy.video.io.bindings import mplfig_to_npimage

import python.morfas.tools as tools
import python.morfas.shiftspectre as bs
import python.morfas.morfcomparison as mcl
import scipy.signal as signal
import time as timer

if __name__ == '__main__':

    audio_path = "C:/Users/palsol/CLionProjects/MASLib/data/mozart vals.mp3"
    scale = 1

    start = 20
    time_start = start + 0.5
    time_end = start + 40
    nfft = 1024
    noverlap = 512
    win_size_t = 8

    # spectrogram, freq, time = tools.wav_to_spectrogram(audio_path,
    #                                                    time_start,
    #                                                    time_end,
    #                                                    nfft=nfft,
    #                                                    noverlap=noverlap)
    print(time_start)
    print(time_end)
    spectrogram, freq, time = tools.mp3_to_spectrogram(audio_path,
                                                       time_start,
                                                       time_end,
                                                       ch=1,
                                                       nfft=nfft,
                                                       noverlap=noverlap)
    cut_freq = int(nfft / 2 * 3 / 4)
    freq = freq[0:-1 * int(nfft / 2 - cut_freq)]
    spectrogram = spectrogram[::]
    spectrogram = tools.compress_spectrum(spectrogram, scale)
    spectrogram[spectrogram == 0] += 10 ** -22
    log_spectrogram = -1 * np.log(spectrogram)
    log_spectrogram /= np.nanmax(log_spectrogram)
    log_spectrogram = 1 - log_spectrogram

    clip_length = time_end - time_start
    time_step = 1.0 / (spectrogram.shape[1] / clip_length)
    frame_step = int(spectrogram.shape[1] / clip_length)
    win_size = int(win_size_t * frame_step)
    data_size = spectrogram.shape[0]

    print('clip lenght(sec): ' + str(clip_length))
    print('frame step(count): ' + str(frame_step))
    print('time step(sec): ' + str(time_step))
    print('window size(count): ' + str(win_size))
    print('window size(sec): ' + str(win_size_t))

    mc = mcl.CorrComporator(win_size, data_size)

    ssdata = np.zeros((spectrogram.shape[1], win_size))
    specdata = np.zeros((spectrogram.shape[1], data_size, win_size))
    frame_time = np.zeros(spectrogram.shape[1])
    print(specdata.shape)

    then = timer.time()
    for i in range(0, spectrogram.shape[1], ):
        frame_time[i] = i * time_step
        ssdata[i] = mc.getshiftspectre()[-1::-1]
        specdata[i] = mc.data.T
        mc.push(log_spectrogram[:, i])
        if i % 10 == 0:
            now = timer.time()
            diff = int(now - then)
            minutes, seconds = diff // 60, diff % 60
            print('step: ' + str(i) + ' time: ' + str(frame_time[i]) + ' comparison time: ' + str(minutes) + ':' + str(
                seconds).zfill(2))

    ssdata_without_nan = ssdata[np.logical_not(np.isnan(ssdata))]
    ssdata_without_nan = ssdata_without_nan[np.logical_not(np.isinf(ssdata_without_nan))]
    ssdata_max = np.percentile(ssdata_without_nan, 99)
    ssdata_min = ssdata_without_nan.min()

    fig = plt.figure(figsize=(20, 10))
    ax1 = fig.add_subplot(211)
    x = np.linspace(0, win_size_t, win_size)
    line, = ax1.plot(x, ssdata[0])
    scat = ax1.scatter(0, 0, color='r', s=40)
    time_text = ax1.text(2, 30, '')
    ax1.set_xlim([0, win_size_t])
    ax1.set_ylim([0, ssdata_max])

    ax2 = fig.add_subplot(212)
    x, y = np.meshgrid(
        np.linspace(win_size_t, 0, win_size),
        np.linspace(freq.min(), freq.max(), data_size))
    im = ax2.pcolormesh(x, y, specdata[20], cmap='jet')


    def animate(i):
        frame = int(i * frame_step)
        data = ssdata[frame]
        minss = bs.analys_shift_spectrum(data)
        scat.set_offsets([minss*(win_size_t/(data.shape[0]-1)), data[minss]])
        im.set_array(np.fliplr(specdata[frame][:-1, :-1]).ravel())
        line.set_ydata(data)  # update the data
        return mplfig_to_npimage(fig)


    animation = VideoClip(animate, duration=clip_length) \
        .set_audio(AudioFileClip(audio_path)
                   .subclip(time_start, time_end))

    animation.write_videofile("C:/Users/palsol/CLionProjects/MASLib/data/vals.mp4",
                              fps=25, codec='libx264')


    # ani = animation.FuncAnimation(fig, animate, np.arange(1, spectrogram.shape[1]), init_func=init,
    #                               interval=25, blit=True)
    #
    # ani.save('/home/palsol/CLionProjects/MASLib/data/res/animate/M O O N - Crystals.mp4', fps=init(spectrogram.shape[1] / (time_end - time_start)),
    #          extra_args=['-vcodec', 'libx264'])
    #
    # plt.show()
