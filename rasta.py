from functools import partial, reduce
import numpy as np
import scipy.signal as signal
import spectrum
from typing import Tuple
import tensorflow as tf
import tensorflow._api.v2.math as math
import librosa.display
import matplotlib.pyplot as plt

def pipeline(*func):
    return lambda x: reduce(lambda f,g: g(f), func, x)


def pad(x: tf.Tensor, shape: Tuple[int], axis: int = 0) -> tf.Tensor:
    return tf.concat([tf.cast(x, dtype=tf.float32), tf.zeros(shape)], axis)


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
    tf.keras.backend.set_floatx('float64')
    power_spec = powspec(x, fs, win_time, hop_time)
    aspectrum = audspec(power_spec, fs)
    nbands = aspectrum.shape[0]

    if dorasta:
        nl_aspectrum = math.log(aspectrum)
        ras_nl_aspectrum = rastafilt(nl_aspectrum)
        aspectrum = math.exp(ras_nl_aspectrum)
    
    postspec, _ = postaud(aspectrum, fs/2)
        
    lpcas = dolpc(postspec, model_order)
    cepstra = lpc2cep(lpcas, model_order+1)
    spectra = lpc2spec(lpcas, nbands)
    return spectra, lifter(cepstra)


def powspec(
    x: tf.Tensor,
    fs: int,
    window_time: float,
    hop_time: float,
    dither: bool = True
) -> tf.Tensor:
    win_length = round(window_time * fs)
    hop_length = round(hop_time * fs)
    fft_length = 1024
    '''
    X = tf.signal.stft(
        x * (1 << 15),
        frame_length=win_length,
        frame_step=hop_length,
        fft_length=fft_length,
        window_fn=tf.signal.hann_window
    )
    '''
    X = librosa.stft(
        (1<<15) * x.numpy(),
        n_fft=fft_length,
        hop_length=hop_length,
        win_length=win_length,
        window='hann',
        center=True
    )
    pow_X = math.pow(tf.abs(X), 2)
    if dither:
        return tf.cast(pow_X + win_length, tf.float64)
    return tf.cast(pow_X, tf.float64)

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
    nfreqs = p_spectrum.shape[0]
    nfft = (int(nfreqs)-1)*2
    
    wts = fft2barkmx(nfft, fs, nfilts, band_width, min_freq, max_freq)
    wts = wts[:, 0:nfreqs]
    if sum_power:
        return wts @ p_spectrum
    return math.pow(wts @ math.sqrt(p_spectrum), 2)
        


def postaud(x: tf.Tensor, fmax: float, broaden: float = 0) -> Tuple[tf.Tensor, tf.Tensor]:
    n_bands, n_frames = x.shape
    nf_pts = int(n_bands + 2 * broaden)
    band_chz = bark2hz(tf.linspace(0.0, tf.cast(hz2bark(fmax), tf.float32), nf_pts))[int(broaden): int(nf_pts - broaden)]
    fsq = tf.pow(band_chz, 2)
    f_tmp = fsq + 1.6e5
    eql = tf.stack([tf.pow(fsq / f_tmp, 2) * ((fsq + 1.44e6) / (fsq + 9.61e6))], 0)
    z = tf.pow(tf.transpose(tf.tile(eql, (n_frames, 1))) * x, 0.33)

    if broaden:
        y = tf.concat([
                [z[0,:]],
                z, 
                [z[z.shape[0]-1, :]]
            ], axis=0
        )
    else:
        y = tf.concat([
                [z[1,:]],
                z[1:z.shape[0]-1, :],
                [z[z.shape[0]-2, :]]
            ], axis=0
        )
    return tf.cast(y, tf.float64), tf.cast(eql, tf.float64)
    
    
def hz2bark(f: tf.Tensor) -> tf.Tensor:
    return tf.cast(6 * math.asinh(f/600), dtype=tf.float64)

def bark2hz(z: tf.Tensor) -> tf.Tensor:
    return tf.cast(600 * math.sinh(z / 6), dtype=tf.float64)

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
    step_bark = nyqbark/(nfilts - 1)
    binbarks = hz2bark(
        tf.range((fft_length/2)+1)*(fs/fft_length)
    )
    wts_len = math.floor(fft_length/2) + 1
    pad_len = fft_length - wts_len
    
    def calc_weights(i: int):
        f_bark_mid = min_bark + i*step_bark
        lof = binbarks - f_bark_mid - 0.5
        hif = (binbarks -f_bark_mid) + 0.5
        wts = math.pow(10, math.minimum(0, math.minimum(hif, -2.5*lof)/bandwidth))
        return pad(wts, (pad_len,))
    
    return tf.cast(tf.stack(
        list(map(calc_weights, range(int(nfilts)))), axis=0
    ), tf.float64)


def rastafilt(x: tf.Tensor) -> tf.Tensor:
    x = x.numpy()
    numer = np.arange(-2, 3)
    numer = np.divide(-numer, np.sum(np.multiply(numer, numer)))
    denom = np.array([2, -0.94])
    zi = signal.lfilter_zi(numer,1)
    y = np.zeros((x.shape))
    for i in range(x.shape[0]):
        y1, zi = signal.lfilter(
            numer,
            1,
            x[i, 0:4],
            axis = 0,
            zi = zi * x[i, 0]
        )
        y1 = y1*0
        y2, _ = signal.lfilter(
            numer,
            denom,
            x[i, 4:x.shape[1]],
            axis = 0,
            zi = zi
        )
        y[i, :] = np.append(y1, y2)    
    return tf.convert_to_tensor(y, dtype=tf.float64)

def dolpc(x: tf.Tensor, modelorder: int) -> tf.Tensor:
    nbands, nframes = x.shape
    ncorr = 2 * (nbands - 1)
    R = np.zeros((ncorr, nframes))
    R[0:nbands, :] = x
    for i in range(nbands-1):
        R[i+nbands-1, :] = x[nbands - (i+1), :]
    
    r = tf.transpose(math.real(tf.signal.ifft(R.T)))[0:nbands,:]
    if modelorder == 0:
        def lpc(ndx: int) -> tf.Tensor:
            _, e_tmp, _ = tf.numpy_function(
                partial(spectrum.LEVINSON, allow_singularity=True),
                (r[:,ndx], modelorder),
                3*[tf.float32]
            )
            return e_tmp
        e = tf.stack(list(map(lpc, range(nframes))))
    else:
        def lpc(ndx: int) -> Tuple[tf.Tensor, tf.Tensor]:
            y_tmp, e_tmp, _ = tf.numpy_function(
                partial(spectrum.LEVINSON, allow_singularity=True),
                (r[:,ndx], modelorder),
                3*[tf.float32]
            )
            return tf.concat([tf.ones((1,), dtype=tf.float64), y_tmp], 0), e_tmp
        lpc_out = list(map(lpc, range(nframes)))
        y = np.stack(list(map(lambda x: x[0], lpc_out)), axis=0)
        e = np.stack([list(map(lambda x: x[1], lpc_out))], axis=1)

    return tf.cast(
        tf.transpose(y)/(tf.tile(tf.transpose(e), (modelorder+1, 1)) + 1e-8),
        tf.float32
    )


def lpc2cep(a: tf.Tensor, nout: int = 0) -> tf.Tensor:
    nin = a.shape[0]

    order = nin-1
    
    if nout == 0:
        nout = order + 1
    
    cep_arr = [-math.log(a[0,:])]

    norm_a = a / (tf.tile(tf.stack([a[0,:]]), (nin, 1)) + 1e-8)
    for n in range(1, nout):
        cumm = 0
        for m in range(1, n):
            cumm += (n-m) * norm_a[m,:] * cep_arr[n-m]
        cep_arr.append(-(norm_a[n,:] + (cumm/n)))
    return tf.stack(cep_arr)




def lpc2spec(
    a: tf.Tensor,
    nout: int = 17,
) -> tf.Tensor:
    rows = a.shape[0]
    model_order = rows - 1

    gain = tf.stack([a[0,:]], axis=0)
    aa = tf.cast(a / tf.tile(gain, (rows, 1)), tf.complex64)
    
    tmp_1 = tf.cast(tf.stack([tf.range(0, nout)], axis=1), tf.complex64)
    tmp_1 = tf.complex(0.0, -1.0) * tmp_1 * np.pi / (nout - 1)
    tmp_2 = tf.cast(tf.stack([tf.range(0, model_order + 1)], 0), tf.complex64)
    zz = math.exp(tmp_1 @ tmp_2)
    t = math.abs(zz @ aa)
    features = tf.pow(1/t, 2) / tf.tile(gain, (nout, 1))
    return features


    
def lifter(x: tf.Tensor, lift: float = 0.6, invs: bool = False) -> tf.Tensor:
    if lift == 0:
        return x
    
    n_cep = x.shape[0]
    lift = 0.6 if lift < 0.0 else lift
    liftwts = tf.concat([tf.ones(1), tf.pow(tf.range(1, n_cep, dtype=tf.float32), lift)], 0)
    if invs:
        liftwts = 1 / liftwts
    
    return tf.numpy_function(np.diag, (liftwts,), tf.float32) @ tf.cast(x, dtype=tf.float32)

def test():
    sig, fs = tf.audio.decode_wav(tf.io.read_file("ae_1a.wav")) 
    spectra, _ = rastaplp(tf.squeeze(sig), fs.numpy())
    librosa.display.specshow(
        10*np.log10(spectra.numpy()),
        sr=fs.numpy()
    )
    plt.show()
    

if __name__ == "__main__":
    test()
