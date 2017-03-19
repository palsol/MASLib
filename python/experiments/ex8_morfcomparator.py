import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

import python.morfas.tools as tools
import python.morfas.beatspectre as bs
import python.morfas.morfcomparison as mcl
import time as timer

if __name__ == '__main__':
    audio_path = "/home/palsol/CLionProjects/MASLib/data/mk.wav"
    scale = 8

    start = 0
    time_start = start + 0.4
    time_end = start + 132
    nfft = 1024
    noverlap = 512

    spectrogram, freq, time = tools.wav_to_spectrogram(audio_path,
                                                       time_start,
                                                       time_end,
                                                       nfft=nfft,
                                                       noverlap=noverlap)
    spectrogram = spectrogram[::-1]
    spectrogram = tools.compress_spectrum(spectrogram, scale)
    log_spectrogram = -1 * np.log(spectrogram)
    log_spectrogram /= np.nanmax(log_spectrogram)
    log_spectrogram = 1 - log_spectrogram

    clip_length = time_end - time_start
    time_step = 1.0 / (spectrogram.shape[1] / (clip_length))
    frame_step = (spectrogram.shape[1] / (clip_length))
    print(frame_step)
    print(time_step)
    win_size = 4 * int(spectrogram.shape[1] / (clip_length))
    data_size = spectrogram.shape[0]

    mc = mcl.MorfComporator(win_size, data_size)

    fig = plt.figure(figsize=(20, 10))

    ax = fig.add_subplot(211)
    x = np.arange(0, win_size, 1)
    print(x.shape)
    line, = ax.plot(x, mc.getsiftspectre())

    ax1 = fig.add_subplot(212)
    im = ax1.imshow(mc.data.T, cmap='jet', animated=True, vmin=0, vmax=1)
    time_text = ax.text(2, 30, '')

    ssdata = np.zeros((spectrogram.shape[1], win_size))
    specdata = np.zeros((spectrogram.shape[1], data_size, win_size))
    frame_time = np.zeros(spectrogram.shape[1])
    print(specdata.shape)

    then = timer.time()
    for i in range(0, spectrogram.shape[1], ):
        frame_time[i] = i * time_step
        ssdata[i] = mc.getsiftspectre()[-1::-1]
        specdata[i] = mc.data.T
        mc.push(log_spectrogram[:, i])
        if i % 10 == 0:
            now = timer.time()
            diff = int(now - then)
            minutes, seconds = diff // 60, diff % 60
            print('step: ' + str(i) + ' time: ' + str(frame_time[i]) + ' comparison time: ' + str(minutes) + ':' + str(
                seconds).zfill(2))

    ax.set_xlim([0, win_size])
    print(ssdata.max())
    ax.set_ylim([0, 20])


    def init():
        """initialize animation"""
        data = ssdata[0]
        line.set_data(x, data)
        im.set_array(specdata[0])
        time_text.set_text('nnnnnn')
        return line, im, time_text


    # def animate(i):
    #     time_text.set_text('frame = %.1f' % frame_time[i])
    #     data = ssdata[i]
    #     im.set_array(specdata[i])
    #     line.set_ydata(data)  # update the data
    #     return line, im, time_text

    def animate(i):
        frame = i*frame_step
        time_text.set_text('frame = %.1f' % frame_time[frame])
        data = ssdata[frame]
        im.set_array(specdata[frame])
        line.set_ydata(data)  # update the data
        return mplfig_to_npimage(fig)


    animation = VideoClip(animate, duration=clip_length)
    animation.write_videofile("/home/palsol/CLionProjects/MASLib/data/03.0011.mp4",
                              fps=25, codec='libx264')


    # ani = animation.FuncAnimation(fig, animate, np.arange(1, spectrogram.shape[1]), init_func=init,
    #                               interval=25, blit=True)
    #
    # ani.save('/home/palsol/CLionProjects/MASLib/data/res/animate/M O O N - Crystals.mp4', fps=init(spectrogram.shape[1] / (time_end - time_start)),
    #          extra_args=['-vcodec', 'libx264'])
    #
    # plt.show()
