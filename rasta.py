from functools import partial, reduce
import numpy as np
import scipy.signal as signal
import spectrum
from typing import Tuple
import tensorflow as tf
import tensorflow._api.v2.math as math

def pipeline(*func):
    return lambda x: reduce(lambda f,g: g(f), func, x)

def next_pow_2(x: tf.Tensor) -> tf.Tensor:
    return tf.cast(
        tf.math.pow(
            2, math.ceil(
                math.log(tf.cast(x, dtype=np.float32))/math.log(2.0))
        ), dtype=tf.int64
    )

def rastaplp(
    x: tf.Tensor,
    fs: int = 16000,
    win_time: float = 0.04,
    hop_time: float = 0.02,
    dorasta: bool = True,
    model_order: int = 8
) -> Tuple[tf.Tensor, tf.Tensor]:

    power_spec = tf.pow(tf.math.abs(tf.signal.stft(
        x,
        frame_length=tf.constant(int(round(fs*win_time))),
        frame_step=tf.constant(int(round(fs*hop_time))),      
        fft_length=next_pow_2(tf.constant(int(round(fs*win_time)))),
        pad_end=True
    )),2)

    aspectrum = audspec(power_spec, fs)
    nbands = aspectrum.shape[0]

    if dorasta:
        nl_aspectrum = math.log(aspectrum)
        ras_nl_aspectrum = rastafilt(nl_aspectrum)
        aspectrum = math.exp(ras_nl_aspectrum)
    
    # dolpc
    # lpc2cep

def audspec(
    p_spectrum: tf.Tensor,
    fs: int = 16000,
    nfilts: int = 0,
    min_freq: float = 0,
    max_freq: float = 0,
    sum_power: bool = True,
    band_width: float = 1
) -> tf.Tensor:
    if nfilts == 0:
        nfilts = math.ceil(hz2bark(fs/2)) + 1
    if max_freq == 0:
        max_freq = fs/2
    nfreqs = p_spectrum[0]
    nfft = (int(nfreqs)-1)*2
    
    wts = fft2barkmx(nfft, fs, nfilts, band_width, min_freq, max_freq)[:, 0:nfreqs]

    if sum_power:
        return wts @ p_spectrum
    return math.pow(wts @ math.sqrt(p_spectrum), 2)
        

    

def hz2bark(f) -> tf.Tensor:
    return 6 * math.asinh(f/600)

def fft2barkmx(
    fft_length: int,
    fs: int,
    nfilts: int,
    bandwidth: float,
    min_freq: float,
    max_freq: float
) -> tf.Tensor:
    min_bark = hz2bark(min_freq)
    nyqbark = hz2bark(max_freq) - min_bark
    wts = tf.zeros((nfilts, fft_length))
    step_bark = nyqbark/(nfilts - 1)
    binbarks = hz2bark(
        tf.range((fft_length/2)+1)*(fs/fft_length)
    )
    for i in range(int(nfilts)):
        f_bark_mid = min_bark + i*step_bark
        lof = binbarks - f_bark_mid - 0.5
        hif = (binbarks -f_bark_mid) + 0.5
        wts[i, 0:(fft_length//2)+1] = math.pow(10, math.minimum(
            0, math.minimum(hif, -2.5*lof)/bandwidth
        ))
    return wts

def rastafilt(x: tf.Tensor) -> tf.Tensor:
    numer = lambda x: -x/math.reduce_sum(math.pow(x,2))(tf.range(-2,3))
    denom = tf.constant([1, -0.94])
    
    zi = tf.numpy_function(signal.lfilter_zi, (numer, 1), None)
    y = tf.zeros(x.shape)
    for i in range(x.shape[0]):
        y1, zi = tf.numpy_function(
            partial(signal.lfilter, axis=0, zi=zi),
            (numer, denom, x[i, 4:x.shape[1]]),
            None
        )
        y2, _ = tf.numpy_function(
            partial(signal.lfilter, axis=0, zi=zi),
            (numer, denom, x[i, 4:x.shape[1]]),
            None
        )
        y[i,:] = tf.concat((y1, y2), axis=0)
    return y

def dolpc(x: tf.Tensor, modelorder: int) -> tf.Tensor:
    nbands, nframes = x.shape
    ncorr = 2 * (nbands - 1)
    R = np.zeros(ncorr, nframes)
    R[0:nbands, :] = x
    for i in range(nbands-1):
        R[i+nbands-1, :] = x[nbands - (i+1), :]
    
    r = tf.signal.ifft(R.T).real.T[0:nbands,:]

    y = tf.ones((nframes, modelorder + 1))
    e = tf.zeros((nframes, 1))

