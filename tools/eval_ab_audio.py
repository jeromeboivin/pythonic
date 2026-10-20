"""
Audio measures for tools/eval_ab.py: the loudness the page matches the sounds
at, and how far one render is from another.

``compare(ref, other)`` takes two mono signals that start at their hit and
returns the distance and the pieces it is made of:

- ``mel``: the loudness-weighted difference of the log-mel spectrograms, in dB.
  Each signal is scaled to the same energy first, so a plain level change
  costs nothing here (it is reported as ``level_db``). A cell counts from 0 at
  50 dB under the shared peak to 1 at the peak, as in the fitter's score;
- ``env``: the same weighting on the 5 ms energy envelope, in dB;
- ``pitch_st``: how far the energy-weighted centre of the 30 Hz to 2 kHz
  spectrum sits from the reference, in semitones (None when neither sound has
  energy there);
- ``bright_oct``: the full-band spectral centroid ratio, in octaves;
- ``decay_oct``: the ratio of the times the envelopes stay within 30 dB of
  their peak, in octaves;
- ``click_db``: the difference of the level of the first 5 ms under the peak.

``distance = mel + 0.5 * env + 2 * |pitch_st|`` (dB-like units). Two renders of
the same patch on a floating-point-identical engine are at 0.
"""

import math

import numpy as np
import soundfile as sf
from scipy import signal

SR = 44100
FLOOR_DB = 50.0        # a cell weighs 0 this far under the peak
N_FFT, HOP, N_MELS = 1024, 256, 48
ENV_HOP = 220          # 5 ms


def read_mono(path) -> tuple:
    """(samples averaged over the channels, sample rate) of an audio file."""
    x, sr = sf.read(str(path), always_2d=True)
    return x.mean(axis=1), sr


def _rbj(kind, f0, q, sr, gain_db=0.0):
    w = 2 * math.pi * f0 / sr
    al, c, a = math.sin(w) / (2 * q), math.cos(w), 10 ** (gain_db / 40)
    if kind == 'hp':
        b = [(1 + c) / 2, -(1 + c), (1 + c) / 2]
        d = [1 + al, -2 * c, 1 - al]
    else:   # high shelf
        sq = 2 * math.sqrt(a) * al
        b = [a * ((a + 1) + (a - 1) * c + sq), -2 * a * ((a - 1) + (a + 1) * c), a * ((a + 1) + (a - 1) * c - sq)]
        d = [(a + 1) - (a - 1) * c + sq, 2 * ((a - 1) - (a + 1) * c), (a + 1) - (a - 1) * c - sq]
    return np.array(b) / d[0], np.array(d) / d[0]


def loudness_db(path) -> float:
    """K-weighted energy of a file in dB, averaged over its channels: a mono
    file and the same sound on two channels read the same."""
    x, sr = sf.read(str(path), always_2d=True)
    for kind, f0, q, g in (('shelf', 1500.0, 1 / math.sqrt(2), 4.0), ('hp', 38.0, 0.5, 0.0)):
        b, a = _rbj(kind, f0, q, sr, g)
        x = signal.lfilter(b, a, x, axis=0)
    return 10 * math.log10(float(np.sum(x ** 2)) / x.shape[1] / sr + 1e-20)


def _mel_bank() -> np.ndarray:
    mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    edges = hz(np.linspace(mel(30.0), mel(SR / 2.0), N_MELS + 2))
    bins = np.fft.rfftfreq(N_FFT, 1.0 / SR)
    fb = np.zeros((N_MELS, len(bins)))
    for i in range(N_MELS):
        lo, mid, hi = edges[i:i + 3]
        fb[i] = np.clip(np.minimum((bins - lo) / (mid - lo), (hi - bins) / (hi - mid)), 0.0, None)
    for i in np.where(fb.sum(axis=1) == 0)[0]:
        fb[i, np.argmin(np.abs(bins - edges[i + 1]))] = 1.0
    return fb


_MEL = _mel_bank()
_WIN = np.hanning(N_FFT)


def _mel_power(x: np.ndarray) -> np.ndarray:
    """(frames, bands) mel power; the first frame is centred on sample 0."""
    y = np.concatenate([np.zeros(N_FFT // 2), x, np.zeros(N_FFT)])
    frames = np.lib.stride_tricks.sliding_window_view(y, N_FFT)[::HOP] * _WIN
    spec = np.fft.rfft(frames, axis=1)
    return (spec.real ** 2 + spec.imag ** 2) @ _MEL.T


def _envelope(x: np.ndarray) -> np.ndarray:
    n = max(len(x) // ENV_HOP, 1) * ENV_HOP
    y = np.pad(x, (0, max(n - len(x), 0)))[:n]
    return (y.reshape(-1, ENV_HOP) ** 2).mean(axis=1)


def _weighted(a: np.ndarray, b: np.ndarray) -> float:
    """Mean |dB difference| of two power arrays (same shape), cells weighted by
    how loud they are in the louder signal."""
    peak = max(a.max(), b.max(), 1e-30)
    la, lb = (10 * np.log10(np.maximum(v, peak * 1e-12) / peak) for v in (a, b))
    w = np.clip((np.maximum(la, lb) + FLOOR_DB) / FLOOR_DB, 0.0, 1.0)
    return float((w * np.abs(la - lb)).sum() / max(w.sum(), 1e-12))


def _pad_to(x: np.ndarray, n: int) -> np.ndarray:
    return np.pad(x, (0, n - len(x))) if len(x) < n else x[:n]


def _centre_hz(x: np.ndarray, lo: float, hi: float):
    """Energy-weighted log-frequency centre of the summed spectrum in [lo, hi],
    with the share of the total energy that is there."""
    n = 1 << int(math.ceil(math.log2(max(len(x), 2))))
    p = np.abs(np.fft.rfft(x, n)) ** 2
    f = np.fft.rfftfreq(n, 1.0 / SR)
    sel = (f >= lo) & (f <= hi)
    total = p.sum()
    if total <= 0 or p[sel].sum() <= 0:
        return None, 0.0
    return float(2 ** (np.sum(p[sel] * np.log2(f[sel])) / p[sel].sum())), float(p[sel].sum() / total)


def _t_within(env: np.ndarray, db: float) -> float:
    """Seconds up to the last 5 ms frame within ``db`` of the envelope's peak."""
    if env.max() <= 0:
        return ENV_HOP / SR
    above = np.nonzero(env > env.max() * 10 ** (-db / 10))[0]
    return (above[-1] + 1) * ENV_HOP / SR


def _click_db(x: np.ndarray) -> float:
    peak = max(float(np.abs(x).max()), 1e-12)
    return 20 * math.log10(max(float(np.sqrt(np.mean(x[:int(0.005 * SR)] ** 2))) / peak, 1e-6))


def compare(ref: np.ndarray, other: np.ndarray) -> dict:
    """How far ``other`` is from ``ref`` (see the module text); both mono at SR."""
    n = max(len(ref), len(other))
    a, b = _pad_to(np.asarray(ref, np.float64), n), _pad_to(np.asarray(other, np.float64), n)
    ea, eb = float(np.sum(a ** 2)), float(np.sum(b ** 2))
    if ea <= 0 or eb <= 0:
        return {'distance': 99.0, 'mel': 99.0, 'env': 99.0, 'pitch_st': None, 'bright_oct': None,
                'decay_oct': None, 'click_db': None, 'level_db': None}
    an, bn = a / math.sqrt(ea), b / math.sqrt(eb)
    mel = _weighted(_mel_power(an), _mel_power(bn))
    env = _weighted(_envelope(an), _envelope(bn))
    ca, wa = _centre_hz(a, 30.0, 2000.0)
    cb, wb = _centre_hz(b, 30.0, 2000.0)
    pitch = 12 * math.log2(cb / ca) if ca and cb and min(wa, wb) > 1e-3 else None
    fa, _ = _centre_hz(a, 30.0, SR / 2)
    fb, _ = _centre_hz(b, 30.0, SR / 2)
    return {
        'distance': mel + 0.5 * env + 2.0 * abs(pitch or 0.0),
        'mel': mel, 'env': env, 'pitch_st': pitch,
        'bright_oct': math.log2(fb / fa) if fa and fb else None,
        'decay_oct': math.log2(_t_within(_envelope(b), 30.0) / _t_within(_envelope(a), 30.0)),
        'click_db': _click_db(b) - _click_db(a),
        'level_db': 10 * math.log10(eb / ea),
    }
