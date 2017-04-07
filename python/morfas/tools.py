"""

MorfAS tools.
Library morphological analysis of signals.
---------------------------------

"""

import os.path
import re

import matplotlib.mlab as mlab
import numpy as np
import soundfile as sf
from numpy.lib.stride_tricks import as_strided

def normalize_whitespace_in_file(name):
    f = open(name)
    temp = open(name + 'temp', 'a')
    for month in f.readlines():
        str = month.strip()
        str = re.sub(r'\s+', ' ', str)
        print(str, file=temp)

    os.rename(name + 'temp', name)

def split_to_slide_chunks(data, chunk_size):
    data = np.array(data)
    data_item_size = data.itemsize
    data_length = data.shape[0]
    print(data_length, data_item_size)
    chunks = as_strided(data, strides=(data_item_size, data_item_size),
                        shape=(data_length - chunk_size + 1, chunk_size))
    return chunks


def split_to_chunks(data, chunk_size):
    print(data.shape)
    num_chunks = data.shape[1] // chunk_size
    chunk_height = data.shape[0]
    data_item_size = data.itemsize
    chunks = np.empty((chunk_height, num_chunks, chunk_size))

    for i in range(data.shape[0]):
        buff_data = np.array(data[i])

        chunks[i] = as_strided(buff_data, strides=(data_item_size * chunk_size, data_item_size),
                               shape=(num_chunks, chunk_size))
    return chunks


def smooth(x, window_len=16, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.150
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


def compress_spectrum(spectrogram, window_len):
    """Simple compress the spectrum using smooth.

     input:
        spectrogram: the input spectrogram
        window_len: the dimension of the smoothing window; should be an odd integer

    """
    spectrogram_smooth = np.apply_along_axis(smooth, 0, spectrogram, window_len)
    spectrogram_compresed = spectrogram_smooth[0:spectrogram.shape[0]: window_len]
    return spectrogram_compresed

def compress_scalogram(scalogram, window_len):
    """Simple compress the scalogram using smooth.

     input:
        scalogram: the input spectrogram
        window_len: the dimension of the smoothing window; should be an odd integer

    """
    scalogram_smooth = np.apply_along_axis(smooth, 1, scalogram, window_len)
    scalogram_compresed = scalogram_smooth[:, 0:scalogram.shape[1]: window_len]
    return scalogram_compresed

def wav_to_spectrogram(wav_file, time_start=0, time_end=None, nfft=2048, noverlap=None, pad_to=None):
    if noverlap is None:
        noverlap = nfft / 2  # same default noverlap

    data, rate = sf.read(wav_file, always_2d=True)

    if time_end is None:
        time_end = data.shape[0]/rate
        time_end -= 3
        print(time_end)

    data = data[int(rate * time_start):int(rate * time_end), 0]

    spec, freqs, t = mlab.specgram(x=data, NFFT=nfft, Fs=rate,
                                   detrend=None, window=None,
                                   noverlap=noverlap, pad_to=pad_to,
                                   sides=None,
                                   scale_by_freq=None,
                                   mode=None)
    return spec, freqs, t
