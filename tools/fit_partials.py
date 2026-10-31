"""Fit the partials section of a drum patch to a hat or cymbal sample, by analysis.

The section (pythonic.voice, PARTIALS_PARAMS) is decaying partials, each with its
frequency, level, decay and phase, plus white noise in bands, each with its level and
decay and an optional click (a second, short level and decay), and an optional cut. As
judged by ear on the CR-78, 808, 909 and 606:

- the noise bands are 48 dB an octave each side: gentler ones smear a narrow noise peak
  (the CR-78's) into its neighbours, and brick-wall ones sound robotic;
- each band's decay is read through its own filter at about a millisecond, after the
  hit's first milliseconds, and its level keeps the band's energy; a click per band at the
  hit carries the rest. Read at the 12 ms of a short-time spectrum, a sub-millisecond
  click spreads over the band's whole decay, 10-20 dB of low-mid hiss that is not there;
- where the sound holds and then drops away fast (a gated closed hat), the section's cut
  and fade fall there, the band decays fitted before it; otherwise it stops where the
  sample ends;
- partials are the sample's real tones only: the peaks standing out more than any peak
  of noise with the sample's spectrum and envelope does, kept when together they carry
  TONE_SHARE_DB of its energy. The strongest peaks of a noise are noise (as tones they
  sound like a frozen, lo-fi hiss), and a weak line (hum, a whine) reads its level off
  the noise around it at the hit. Each tone's level and decay come from its own envelope
  (the sample heterodyned down to 0 Hz and low-passed), one exponential from its peak on,
  weighted by level; phases are random (drawn the same way for a given sample);
- the bands take what the tones leave of the sample's power. Each engine band carries its
  own band's power: a fit over all bands at once lets neighbours trade power, and the
  noise turns into a row of peaks.
"""
import math

import numpy as np
from scipy import ndimage, signal

from pythonic.voice import (PARTIAL_BANDS, format_bands, format_partials, partial_band_edges, partial_band_filters,
                            partial_band_power)

SR = 44100
PARTIALS = 96           # at most this many tones
FINE = 1 << 16          # FFT of the peak search
NOISE_FFT, NOISE_HOP = 512, 64
PHASE_SEED = 0
CUT_FADE_MS = 5.0       # without a gate the section stops at the sample's end, in this fade (-60 dB)
BAND_GRID = np.linspace(30.0, SR / 2 - 30.0, 4096)
NOISE_SLOPE = 48.0      # dB an octave each side of a band
TONE_DB = 15.0          # a real tone stands at least this far over the running median ...
TONE_TWIN_DB = 2.0      # ... and this much more than any peak of noise twins of the sample
TONE_SHARE_DB = -15.0   # the real tones are kept when together they carry this much of the energy
CLICK_SHARE = 0.1       # a band gets a click when its body leaves at least this share of its energy




def _envelope(x, freq, bw):
    """|x| heterodyned down from freq and low-passed at bw (zero phase), as an amplitude."""
    t = np.arange(len(x)) / SR
    z = x * np.exp(-2j * np.pi * freq * t)
    sos = signal.butter(2, bw, fs=SR, output='sos')
    pad = int(0.05 * SR)
    zz = np.concatenate([np.zeros(pad), z, np.zeros(pad)])
    lp = signal.sosfiltfilt(sos, zz.real) + 1j * signal.sosfiltfilt(sos, zz.imag)
    return 2.0 * np.abs(lp[pad:pad + len(x)])


def _decay_fit(tail, t, min_points=8, short=0.01):
    """(level at t = 0, decay s) of one exponential through tail from its start, weighted by
    level; (its first value, `short` s) when too little of it is above a thousandth of it."""
    good = tail > tail[0] * 1e-3
    if good.sum() <= min_points:
        return float(tail[0]), short
    w = tail[good]
    A = np.vstack([np.ones(good.sum()), t[good]]).T
    coef = np.linalg.lstsq(A * w[:, None], np.log(tail[good] + 1e-12) * w, rcond=None)[0]
    decay = float(np.clip(-1.0 / coef[1], 0.002, 8.0)) if coef[1] < 0 else 8.0
    return float(np.exp(coef[0])), decay




def render_rows(rows, n):
    t = np.arange(n) / SR
    y = np.zeros(n)
    for fq, amp, decay, ph in rows:
        y += amp * np.exp(-t / decay) * np.cos(2 * np.pi * fq * t + ph)
    return y


def _stft_power_to_psd() -> float:
    """One-sided power density (per Hz) of a signal whose short-time spectrum (scipy's
    'spectrum' scaling, Hann NOISE_FFT) reads 1 in every bin."""
    win = signal.get_window('hann', NOISE_FFT)
    return float(np.sum(win) ** 2 / np.sum(win ** 2)) / (SR / 2)




def engine_bands(rows, slope: float = 24.0) -> np.ndarray:
    """Engine noise band rows (gain, decay[, click gain, click decay]) for band rows (level
    per bin of the short-time spectrum, decay[, click level, click decay]): each engine
    band carries its own band's power, over what the engine's overlapping band filters
    (of `slope` dB an octave) give in that band with every band at the same gain."""
    total = partial_band_power(SR, BAND_GRID, slope=slope).sum(axis=0) / (SR / 2)
    edges = partial_band_edges()
    k = _stft_power_to_psd()
    out = np.array(rows, dtype=float).copy()
    levels = range(0, out.shape[1], 2)
    for b in range(PARTIAL_BANDS):
        m = (BAND_GRID >= edges[b]) & (BAND_GRID < edges[b + 1])
        g = math.sqrt(k * m.sum() / total[m].sum()) if m.any() and total[m].sum() > 0.0 else 0.0
        for c in levels:
            out[b, c] *= g
    for c in levels:
        out[:, c + 1] = np.maximum(out[:, c + 1], 1e-4 if c else 1e-3)
    return out


def _spectrum_over_median(x):
    """(Hz, dB, dB over the 300 Hz running median) of x's first second, fine."""
    seg = x[:SR]
    P = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), FINE)) ** 2
    f = np.fft.rfftfreq(FINE, 1.0 / SR)
    db = 10.0 * np.log10(P + P.max() * 1e-12 + 1e-30)
    return f, db, db - ndimage.median_filter(db, int(300 * FINE / SR) | 1, mode='nearest')






def _noise_twin(x, seed):
    """Noise with x's short-time spectrum (256 points) and random phases."""
    _, _, Z = signal.stft(x, SR, nperseg=256, noverlap=192)
    rng = np.random.default_rng(seed)
    _, y = signal.istft(np.abs(Z) * np.exp(2j * np.pi * rng.random(Z.shape)), SR, nperseg=256, noverlap=192)
    return y[:len(x)]


def real_tones(x, n: int = PARTIALS) -> np.ndarray:
    """(m, 4) partial rows (frequency Hz, level, decay s, phase rad) of x's real tones, from 30 Hz: the peaks standing
    TONE_DB over the median and TONE_TWIN_DB more than any peak of two noise twins of x;
    none unless together they carry TONE_SHARE_DB of x's energy."""
    def top(z):
        f, _, over = _spectrum_over_median(z)
        return over[(f > 30.0) & (f < 19000.0)].max()
    thr = max(TONE_DB, max(top(_noise_twin(x, s)) for s in (1, 2)) + TONE_TWIN_DB)
    f, db, over = _spectrum_over_median(x)
    pk, _ = signal.find_peaks(over, height=thr, distance=max(1, int(4 * FINE / SR)))
    pk = pk[(f[pk] > 30.0) & (f[pk] < 19000.0)]
    pk = pk[np.argsort(-db[pk])][:n]
    freqs = []
    for b in pk:
        a, c, d = db[b - 1:b + 2]
        den = a - 2.0 * c + d
        freqs.append(f[b] + (0.5 * (a - d) / den if den != 0 else 0.0) * SR / FINE)
    freqs = np.sort(np.array(freqs))
    rng = np.random.default_rng(PHASE_SEED)
    rows = []
    for i, fq in enumerate(freqs):
        near = np.min(np.abs(np.delete(freqs, i) - fq)) if len(freqs) > 1 else 200.0
        a = _envelope(x, fq, float(np.clip(0.45 * near, 12.0, 90.0)))
        k = int(np.argmax(a))
        amp, decay = _decay_fit(a[k:], np.arange(len(a) - k) / SR)
        rows.append([fq, amp, decay, rng.uniform(0.0, 2.0 * np.pi) - np.pi / 2.0])
    rows = np.array(rows).reshape(-1, 4)
    y = render_rows(rows, len(x))
    if 10.0 * math.log10(float(np.sum(y ** 2)) / max(float(np.sum(x ** 2)), 1e-30) + 1e-30) < TONE_SHARE_DB:
        return rows[:0]
    return rows


SHORT_S = 0.05          # a short hit's tones are the peaks of its first 50 ms ...
SHORT_PROM_DB = 6.0     # ... standing this far out of their neighbourhood ...
SHORT_RANGE_DB = 20.0   # ... within this much of the strongest
SHORT_TONES = 6


def short_tones(x, n: int = SHORT_TONES) -> np.ndarray:
    """(m, 4) partial rows of a short hit's tones, from 100 Hz to 12 kHz: the clear peaks of
    its first SHORT_S, where real_tones finds none (a 25-50 ms rim leaves too few cycles
    above the median); level and decay from each one's envelope, as in real_tones."""
    w = x[:int(SHORT_S * SR)]
    db = 20.0 * np.log10(np.abs(np.fft.rfft(w * np.hanning(len(w)), FINE)) + 1e-12)
    f = np.fft.rfftfreq(FINE, 1.0 / SR)
    pk, _ = signal.find_peaks(db, prominence=SHORT_PROM_DB)
    pk = pk[(f[pk] > 100.0) & (f[pk] < 12000.0) & (db[pk] > db.max() - SHORT_RANGE_DB)]
    freqs = np.sort(f[pk[np.argsort(-db[pk])][:n]])
    rows = []
    for i, fq in enumerate(freqs):
        near = np.min(np.abs(np.delete(freqs, i) - fq)) if len(freqs) > 1 else 200.0
        a = _envelope(x, fq, float(np.clip(0.45 * near, 12.0, 90.0)))
        k = int(np.argmax(a))
        amp, decay = _decay_fit(a[k:], np.arange(len(a) - k) / SR, min_points=3, short=0.003)
        rows.append([fq, amp, decay, -np.pi / 2.0])
    return np.array(rows).reshape(-1, 4)


def gate(x):
    """(cut s, fade s) where x holds and then drops away fast, or None: a two-line fit of
    its level above 2 kHz (1 ms RMS, dB) from the peak to 60 dB under it, the second line at
    least 3 times as steep as the first and over 15 dB long. The fade is -60 dB at the
    second line's slope."""
    hp = signal.sosfiltfilt(signal.butter(4, 2000.0, 'highpass', fs=SR, output='sos'), x)
    e = np.sqrt(np.convolve(hp ** 2, np.ones(44) / 44, 'same'))
    db = 20.0 * np.log10(e[::44] / max(e.max(), 1e-30) + 1e-9)       # 1 ms points
    k = int(np.argmax(db))
    below = np.nonzero(db[k:] < -60.0)[0]
    seg = db[k:k + (below[0] if len(below) else len(db) - k)]
    if len(seg) < 8:
        return None
    t = np.arange(len(seg))
    best = None
    for c in range(3, len(seg) - 3):
        a1, a2 = np.polyfit(t[:c], seg[:c], 1), np.polyfit(t[c:], seg[c:], 1)
        err = np.sum((np.polyval(a1, t[:c]) - seg[:c]) ** 2) + np.sum((np.polyval(a2, t[c:]) - seg[c:]) ** 2)
        if best is None or err < best[0]:
            best = (err, c, a1[0], a2[0])
    _, c, s1, s2 = best
    if s2 > -0.1 or s2 > 3.0 * min(s1, -0.01) or seg[c] - seg[-1] < 15.0:
        return None
    return (k + c) / 1000.0, float(np.clip(60.0 / -s2, 1.0, 4000.0)) / 1000.0


def _cut_gain(t, cut, fade):
    u = np.maximum(t - cut, 0.0)
    return np.where(u < fade, np.exp(-6.907755279 * u / fade), 0.0)


def click_bands(x, cut, fade, y=None) -> np.ndarray:
    """(PARTIAL_BANDS, 4) band rows (level per bin of the short-time spectrum, decay s,
    click level, click decay s) of what the partials y leave of x, the section cut at `cut`
    in `fade` (s). Each band's body decay is read through the band's own filter (0.5 ms
    RMS) after the hit's first milliseconds (3 ms and the band's ring), up to the cut; its
    level from that envelope back at the hit; the click (decay: the envelope's fall before
    that) carries the rest of the band's energy, when it is at least CLICK_SHARE of it.
    Energies are integrals over time, so short decays count right."""
    kw = dict(fs=SR, nperseg=NOISE_FFT, noverlap=NOISE_FFT - NOISE_HOP)
    f, _, Zx = signal.stft(x, **kw)
    res = np.abs(Zx) ** 2
    if y is not None:
        _, _, Zy = signal.stft(y[:len(x)], **kw)
        res = np.maximum(res - np.abs(Zy) ** 2, 0.0)
    edges = partial_band_edges()
    coef = partial_band_filters(SR, slope=NOISE_SLOPE).reshape(PARTIAL_BANDS, -1, 5)
    tt = np.arange(int((cut + fade + 0.5) * SR)) / SR
    gain2 = _cut_gain(tt, cut, fade) ** 2
    energy = lambda level, decay: level ** 2 * float(np.sum(np.exp(-2.0 * tt / decay) * gain2)) / NOISE_HOP
    rows = np.zeros((PARTIAL_BANDS, 4))
    rows[:, 1] = rows[:, 3] = 0.005
    step = 11                                   # 0.25 ms points
    last = int(cut * SR / step) + 1
    for b in range(PARTIAL_BANDS):
        m = (f >= edges[b]) & (f < edges[b + 1])
        total = float(res[m].mean(axis=0).sum()) if m.any() else 0.0
        if total <= 0.0 or not coef[b].any():
            continue
        sos = np.c_[coef[b][:, :3], np.ones(len(coef[b])), coef[b][:, 3:]]
        z = signal.sosfiltfilt(sos, x)
        env = np.sqrt(np.convolve(z ** 2, np.ones(22) / 22, 'same'))[::step][:last]
        t = np.arange(len(env)) * step / SR
        k = int(np.argmax(env))
        if env[k] <= 0.0:
            continue
        body = t >= max(0.003 + 1.0 / (edges[b + 1] - edges[b]), t[k])
        if body.sum() > 4:
            w = env[body]
            _, decay = _decay_fit(w, t[body] - t[body][0], min_points=3, short=0.002)
            level = math.exp(float(np.sum(w * (np.log(w + 1e-12) + t[body] / decay)) / np.sum(w)))
        else:
            _, decay = _decay_fit(env[k:], t[k:] - t[k], min_points=3, short=0.002)
            level = float(env[k])
        # envelope units to per-bin power, through the band's energy
        level *= math.sqrt(total * NOISE_HOP / max(float(np.sum(env ** 2)) * step, 1e-30))
        body_energy = energy(level, decay)
        if body_energy > total:
            level *= math.sqrt(total / body_energy)
            body_energy = total
        rest = total - body_energy
        click, cdecay = 0.0, 0.001
        early = (t >= t[k]) & (t < max(0.003 + 1.0 / (edges[b + 1] - edges[b]), t[k] + 0.001))
        if early.sum() > 2:
            slope = np.polyfit(t[early] - t[k], np.log(env[early] / env[k] + 1e-9), 1)[0]
            if slope < 0.0:
                cdecay = float(np.clip(-1.0 / slope, 0.0003, 0.005))
        if rest > CLICK_SHARE * total:
            click = math.sqrt(rest / max(energy(1.0, cdecay), 1e-30))
        rows[b] = (level, decay, click, cdecay)
    return rows




def fit(x: np.ndarray, short: bool = False) -> tuple:
    """(VoiceParams fields of the section, description) for sample x, level 0 dB (the caller
    sets it); `short`: a short hit, whose tones are its short_tones when it has no real ones."""
    x = x / max(float(np.abs(x).max()), 1e-12)     # a quiet recording would need more level than the knob has
    g = gate(x)
    cut, fade = g if g else (len(x) / SR, CUT_FADE_MS / 1000.0)
    tones = real_tones(x)
    if short and not len(tones):
        tones = short_tones(x)
    y = render_rows(tones, len(x)) if len(tones) else None
    bands = engine_bands(click_bands(x, cut, fade, y), NOISE_SLOPE)
    fields = {'partials': format_partials(tones), 'partial_bands': format_bands(bands),
              'partials_level_db': 0.0, 'partials_tune': 0.0, 'partials_decay': 1.0,
              'partials_cut_ms': cut * 1000.0, 'partials_fade_ms': fade * 1000.0, 'partials_slope': NOISE_SLOPE}
    return fields, f'{len(tones)} tones' + (', gated' if g else '')


def peak_level_db(fields, render, peak_db: float) -> float:
    """partials_level_db for which the section peaks at peak_db (dBFS) at velocity 127."""
    peak = float(np.abs(render(fields)).max())
    return float(np.clip(peak_db - 20.0 * math.log10(max(peak, 1e-9)), -60.0, 24.0))
