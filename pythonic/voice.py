"""
Drum voice engine.

Patch parameters, derived engine constants and the DrumVoice front end; the
per-sample render loop is compiled with numba (voice_kernel.py).

Signal flow (per channel):

    oscillator (2x oversampled, band-limited shapes, pitch modulation; the decay
      mode can hold its start pitch before it decays)
      -> polyphase decimator -> amplitude envelope
    noise (uniform white -> state-variable filter, power-normalised)
      -> noise envelope (exponential / linear / clap "mod")
    metal (optional: six free-running square waves -> two band-passes, each
      saturated (drive), with its own decay envelope and level -> high-pass)
    mix (osc / noise gains, plus metal) -> drive -> asymmetric soft shaper
      -> SVF peaking EQ -> level / pan / makeup gains
    partials (optional, for hats and cymbals: decaying partials, each with its
      frequency, level, decay and phase, plus white noise in 32 bands of 24 or 48 dB
      an octave, each with its level and decay and an optional click (a second,
      short level and decay); an optional cut fades it out) -> level / pan

With both metal band levels at 0 the metal section is off and the voice
renders exactly as without it. While the partials section is on, a hit starts
it instead of the oscillator, noise and metal: those keep a classic version of
the same sound, for engines without the section.

Everything is processed on a 4-sample block grid in absolute time: envelopes advance per sample but change stage only at block ends,
triggers take effect at the next block boundary, and slow pitch modulation is
updated on an 8 (oversampled) / 128 (oversampled) sample grid.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np

from . import voice_kernel as K


# Output scaling of the whole engine (a 0 dB patch with a full sine peaks at
# 0.8165 * 0.75 * 0.5 = 0.306 per channel).
MASTER_SCALE = 0.5

# Velocity curve: x * exp(K (x - 1))
VEL_K = 4.259782314
OSC_GAIN = 0.4634255171  # compensates the decimator DC gain (2 / (1 - DEC_A))

WAVE_SINE, WAVE_TRI, WAVE_SAW = 0, 1, 2
MOD_DECAY, MOD_SINE, MOD_NOISE = K.MOD_DECAY, K.MOD_SINE, K.MOD_NOISE
FILT_LP, FILT_BP, FILT_HP = 0, 1, 2
NENV_EXP, NENV_LINEAR, NENV_MOD = K.NENV_EXP, K.NENV_LINEAR, K.NENV_MOD

# Metal section: VoiceParams field -> (default, .mtdrum key, unit, minimum, maximum).
# The default oscillator tuning is the classic six-square cymbal bank
# (205.3 ... 800 Hz). Levels and velocity are ratios (percent in files); the
# section is off while both band levels are 0, and a patch that leaves it off
# has none of these keys.
MOD_HOLD_MAX_MS = 100.0  # longest hold of the decay pitch modulation

METAL_PARAMS = {
    'metal_freq1': (205.3, 'MetalFreq1', ' Hz', 20.0, 2000.0),
    'metal_freq2': (304.4, 'MetalFreq2', ' Hz', 20.0, 2000.0),
    'metal_freq3': (369.6, 'MetalFreq3', ' Hz', 20.0, 2000.0),
    'metal_freq4': (522.7, 'MetalFreq4', ' Hz', 20.0, 2000.0),
    'metal_freq5': (540.0, 'MetalFreq5', ' Hz', 20.0, 2000.0),
    'metal_freq6': (800.0, 'MetalFreq6', ' Hz', 20.0, 2000.0),
    'metal_tune': (0.0, 'MetalTune', ' sm', -24.0, 24.0),
    'metal_band1_freq': (3440.0, 'MetalB1Freq', ' Hz', 200.0, 16000.0),
    'metal_band1_decay_ms': (300.0, 'MetalB1Dcy', ' ms', 10.0, 10000.0),
    'metal_band1_level': (0.0, 'MetalB1Lvl', '%', 0.0, 1.0),
    'metal_band1_q': (1.0, 'MetalB1Q', '', 0.5, 20.0),
    'metal_band2_freq': (7100.0, 'MetalB2Freq', ' Hz', 200.0, 16000.0),
    'metal_band2_decay_ms': (800.0, 'MetalB2Dcy', ' ms', 10.0, 10000.0),
    'metal_band2_level': (0.0, 'MetalB2Lvl', '%', 0.0, 1.0),
    'metal_band2_q': (1.0, 'MetalB2Q', '', 0.5, 20.0),
    'metal_curve': (0.0, 'MetalCurve', '%', 0.0, 1.0),
    'metal_drive': (0.0, 'MetalDrive', '%', 0.0, 1.0),
    'metal_hp_freq': (2000.0, 'MetalHPFreq', ' Hz', 20.0, 16000.0),
    'metal_vel': (0.0, 'MetalVel', '%', 0.0, 2.0),
}
METAL_DEFAULTS = {name: spec[0] for name, spec in METAL_PARAMS.items()}
METAL_PERCENT = {'metal_band1_level', 'metal_band2_level', 'metal_curve', 'metal_drive', 'metal_vel'}
METAL_FREQS = tuple(f'metal_freq{i}' for i in range(1, 7))
METAL_GAIN = 0.6        # a band at level 1 is about as loud (RMS) as the noise at full mix


# Partials section: VoiceParams field -> (default, .mtdrum key, unit, minimum, maximum).
PARTIALS_PARAMS = {
    'partials_level_db': (0.0, 'PartLevel', ' dB', -60.0, 24.0),
    'partials_tune': (0.0, 'PartTune', ' sm', -24.0, 24.0),
    'partials_decay': (1.0, 'PartDecay', '%', 0.1, 4.0),         # scales every decay
    'partials_vel': (0.0, 'PartVel', '%', 0.0, 2.0),
    'partials_rise_ms': (0.0, 'PartRise', ' ms', 0.0, 100.0),     # fades in over this (0 = at once)
    'partials_cut_ms': (0.0, 'PartCut', ' ms', 0.0, 10000.0),    # 0 = no cut
    'partials_fade_ms': (5.0, 'PartFade', ' ms', 1.0, 4000.0),        # -60 dB at its end
    'partials_slope': (24.0, 'PartSlope', ' dB', 24.0, 48.0),    # each side of a noise band: 24 or 48 dB an octave
}
PARTIALS_PERCENT = {'partials_decay', 'partials_vel'}
# The two tables, as text (format_partials, format_bands): VoiceParams field -> .mtdrum key
PARTIALS_TABLES = {'partials': 'Partials', 'partial_bands': 'PartBands'}
PARTIALS_DEFAULTS = {**{name: spec[0] for name, spec in PARTIALS_PARAMS.items()},
                     **{name: '' for name in PARTIALS_TABLES}}
MAX_PARTIALS = 384
PARTIAL_BANDS = 32
PARTIAL_BAND_LO = 100.0     # Hz, lower edge of the first band (at pitch 0)
PARTIAL_BAND_HI = 22050.0   # Hz, upper edge of the last
PART_END = 1e-5             # a partial or band this far under the section's loudest has ended
PART_TOP = 0.45             # cycles/sample: partials and band edges above are left out


def metal_on(params) -> bool:
    """True when a VoiceParams or channel-format dict turns the metal section on."""
    get = params.get if isinstance(params, dict) else lambda k, d: getattr(params, k, d)
    return get('metal_band1_level', 0.0) > 0.0 or get('metal_band2_level', 0.0) > 0.0


def partials_on(params) -> bool:
    """True when a VoiceParams or channel-format dict turns the partials section on."""
    get = params.get if isinstance(params, dict) else lambda k, d: getattr(params, k, d)
    return bool(get('partials', '')) or bool(get('partial_bands', ''))


def format_partials(table) -> str:
    """Text of a partials table, rows (frequency Hz, amplitude, decay s, phase rad): one
    'Hz dB ms rad' entry per partial, separated by commas."""
    rows = np.asarray(table, dtype=float).reshape(-1, 4)
    return ', '.join(f'{f:.2f} {20.0 * math.log10(max(a, 1e-12)):.1f} {d * 1000.0:.4g} {ph:.2f}'
                     for f, a, d, ph in rows)


def format_bands(table) -> str:
    """Text of a noise band table, rows (amplitude, decay s[, click amplitude, click decay
    s]) from the lowest band up: one 'dB ms' entry per band, 'dB ms dB ms' for a band with
    a click, separated by commas."""
    rows = np.asarray(table, dtype=float)
    rows = rows.reshape(-1, 4 if rows.ndim == 2 and rows.shape[1] == 4 else 2)
    db = lambda a: f'{20.0 * math.log10(max(a, 1e-12)):.1f}'
    return ', '.join(f'{db(r[0])} {r[1] * 1000.0:.4g}' + (f' {db(r[2])} {r[3] * 1000.0:.4g}' if len(r) == 4 and r[2] > 0.0 else '')
                     for r in rows)


def _table(text: str, widths: tuple, limit: int) -> np.ndarray:
    """Rows of a table text: comma-separated entries of one of `widths` numbers, the
    narrower ones padded with zeros to the widest; (dB, ms) pairs from the first number
    on (partials: from the second) become (amplitude, s)."""
    width = max(widths)
    rows = []
    for entry in str(text).split(','):
        try:
            v = [float(x) for x in entry.split()]
        except ValueError:
            continue
        if len(v) in widths and all(math.isfinite(x) for x in v):
            rows.append(v + [None] * (width - len(v)))
    out = np.array(rows[:limit], dtype=float).reshape(-1, width)
    pad = np.isnan(out)
    first = 1 if width == 4 and widths == (4,) else 0
    for c in range(first, width - 1 if first else width, 2):
        out[:, c] = 10.0 ** (out[:, c] / 20.0)
        out[:, c + 1] /= 1000.0
    out[pad] = 0.0
    out.flags.writeable = False
    return out


@functools.lru_cache(maxsize=256)
def parse_partials(text: str) -> np.ndarray:
    """(n, 4) rows (frequency Hz, amplitude, decay s, phase rad) of a partials text
    (format_partials); at most MAX_PARTIALS, malformed entries left out."""
    return _table(text, (4,), MAX_PARTIALS)


@functools.lru_cache(maxsize=256)
def parse_bands(text: str) -> np.ndarray:
    """(n, 4) rows (amplitude, decay s, click amplitude, click decay s) of a noise band
    text (format_bands), the click 0 where a band has none; at most PARTIAL_BANDS, from the
    lowest band up."""
    return _table(text, (2, 4), PARTIAL_BANDS)


def partial_band_edges(ratio: float = 1.0) -> np.ndarray:
    """The PARTIAL_BANDS + 1 band edges in Hz, geometric from PARTIAL_BAND_LO to
    PARTIAL_BAND_HI, times the pitch ratio."""
    k = np.arange(PARTIAL_BANDS + 1) / PARTIAL_BANDS
    return PARTIAL_BAND_LO * (PARTIAL_BAND_HI / PARTIAL_BAND_LO) ** k * ratio


# The second-order sections of a Butterworth filter of order 4 (24 dB an octave) and 8 (48)
BAND_QS = {4: (0.5411961, 1.3065630),
           8: (0.5097956, 0.6013449, 0.8999762, 2.5629154)}


def band_order(slope: float) -> int:
    """Butterworth order of each side of a noise band for a PartSlope (dB an octave)."""
    return 8 if slope >= 36.0 else 4


def partial_band_filters(sr: float, ratio: float = 1.0, slope: float = 24.0) -> np.ndarray:
    """(PARTIAL_BANDS, 2 * order * 5) coefficients: band k is a Butterworth high-pass at its
    lower edge into one low-pass at its upper edge (of the order band_order(slope): 24 or
    48 dB an octave each side), as second-order sections, b0 b1 b2 a1 a2 each, the
    high-pass ones first. A band reaching past PART_TOP has no low-pass; one starting past
    it is silent."""
    qs = BAND_QS[band_order(slope)]
    half = len(qs)
    out = np.zeros((PARTIAL_BANDS, 2 * half, 5))
    edges = partial_band_edges(ratio) / sr
    for k in range(PARTIAL_BANDS):
        lo, hi = edges[k], edges[k + 1]
        if lo >= PART_TOP:
            continue
        for i, q in enumerate(qs):
            b, a = _rbj(lo, q, 'hp')
            out[k, i] = (b[0], b[1], b[2], a[1], a[2])
            if hi < PART_TOP:
                b, a = _rbj(hi, q, 'lp')
                out[k, half + i] = (b[0], b[1], b[2], a[1], a[2])
            else:
                out[k, half + i, 0] = 1.0
    return out.reshape(PARTIAL_BANDS, 2 * half * 5)


def partial_band_power(sr: float, freqs, ratio: float = 1.0, slope: float = 24.0) -> np.ndarray:
    """(PARTIAL_BANDS, len(freqs)) power gains of the noise band filters at freqs (Hz)."""
    z = np.exp(-2j * np.pi * np.asarray(freqs, dtype=float) / sr)
    out = []
    for c in partial_band_filters(sr, ratio, slope):
        h = np.ones_like(z)
        for b0, b1, b2, a1, a2 in c.reshape(-1, 5):
            h = h * (b0 + b1 * z + b2 * z * z) / (1.0 + a1 * z + a2 * z * z)
        out.append(np.abs(h) ** 2)
    return np.array(out)


@functools.lru_cache(maxsize=32)
def _section(partials: str, bands: str, tune: float, decay: float, ratio: float, sr: float, slope: float = 24.0):
    """(partials, bands, kernel constants sp, end level, filter sections a band, whether
    the bands have clicks) of a partials section; see voice_kernel._section_run for the
    layout of sp."""
    rt = ratio * 2.0 ** (min(max(tune, -48.0), 48.0) / 12.0)
    scale = min(max(decay, 0.05), 8.0)
    P = parse_partials(partials)
    f = P[:, 0] * rt
    P = P[(f > 0.0) & (f < PART_TOP * sr)]
    amp = P[:, 1]
    tau = np.maximum(P[:, 2] * scale, 1e-4) * sr        # samples
    B = parse_bands(bands)
    bamp = B[:, 0]
    btau = np.maximum(B[:, 1] * scale, 1e-4) * sr
    camp = B[:, 2]
    ctau = np.maximum(B[:, 3] * scale, 1e-4) * sr
    peak = max(float(amp.max(initial=0.0)), float(bamp.max(initial=0.0)), float(camp.max(initial=0.0)))
    end = PART_END * peak
    life = np.where(amp > end, np.log(np.maximum(amp, 1e-300) / max(end, 1e-300)) * tau, 0.0)
    order = np.argsort(-life, kind='stable')
    P, amp, tau, life = P[order], amp[order], tau[order], life[order]
    w = 2.0 * np.pi * P[:, 0] * rt / sr
    sp = np.concatenate([w, np.exp(-1.0 / tau), life, amp * np.cos(P[:, 3]), amp * np.sin(P[:, 3]),
                         partial_band_filters(sr, rt, slope)[:len(B)].ravel(), np.exp(-1.0 / btau), bamp,
                         np.exp(-1.0 / ctau), camp])
    sp.flags.writeable = False
    return len(P), len(B), sp, end, band_order(slope), bool((camp > 0.0).any())


# --------------------------------------------------------------------------- helpers
def attack_time_s(attack_ms: float) -> float:
    """Attack times are quantised through the knob curve t = p * exp(5.526 p - 3.224)."""
    t = max(0.0, attack_ms) / 1000.0
    if t <= 0.0:
        return 0.0
    p = attack_norm(attack_ms)
    return p * math.exp(p * 5.526204586 - 3.223619223)


def attack_norm(attack_ms: float) -> float:
    """Inverse of the attack knob curve (normalised knob position 0..1)."""
    t = max(0.0, attack_ms) / 1000.0
    if t <= 0.0:
        return 0.0
    if t >= 10.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if mid * math.exp(mid * 5.526204586 - 3.223619223) < t:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def velocity_factor(velocity: float, sensitivity: float) -> float:
    """Linear velocity factor; sensitivity 1.0 == 100 % (range 0..2)."""
    inv = 2.0 - 2.0 * (float(velocity) - 1.0) / 126.0
    return max(0.0, 1.0 - inv * sensitivity * 0.5)


def velocity_gain(velocity: float, sensitivity: float) -> float:
    x = velocity_factor(velocity, sensitivity)
    g = x * math.exp(x * VEL_K - VEL_K)
    return g if abs(g) >= 1e-10 else 0.0


def svf_coefs(f: float, k: float) -> np.ndarray:
    """Coefficients of the normalised two-integrator SVF (f in cycles/sample)."""
    s = math.sin(f * 2.0 * math.pi) * 0.5
    t = math.sin(f * math.pi)
    a = 1.0 / (s * k + 1.0)
    b = a * t * t
    c = 1.0 - t * t
    sb = math.sqrt(b)
    p = np.empty(8)
    p[0] = c
    p[1] = 1.0 / c
    p[2] = b / c
    p[3] = a * s / c
    p[4] = (a - b) / c
    p[5] = p[3] / sb
    p[6] = 2.0 * sb
    p[7] = (1.0 - a) / sb
    return p


def svf_tf(p: np.ndarray, mode: int):
    """Transfer function (b, a) of an SVF output tap (LP/BP/HP)."""
    p0, p1, p2, p3, p4, p5, p6, p7 = p
    # state x = [s1, s2]; s1' = s1 + p6 s2 ; v = p0 u - s1' - p7 s2 ; s2' = s2 + p6 v
    A = np.array([[1.0, p6], [-p6, 1.0 - p6 * (p6 + p7)]])
    B = np.array([0.0, p6 * p0])
    if mode == FILT_LP:     # y = p1 s1' + p2 v
        C = np.array([p1 - p2, p1 * p6 - p2 * (p6 + p7)])
        D = p2 * p0
    elif mode == FILT_BP:   # y = p5 s2' - p3 v
        q = p5 * p6 - p3
        C = np.array([-q, p5 - q * (p6 + p7)])
        D = q * p0
    else:                   # y = p4 v
        C = np.array([-p4, -p4 * (p6 + p7)])
        D = p4 * p0
    # H(z) = C (zI - A)^-1 B + D  ->  2nd order rational function
    tr = A[0, 0] + A[1, 1]
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    a = np.array([1.0, -tr, det])
    # numerator: C adj(zI - A) B + D * den
    adj_z = np.array([[1.0, 0.0], [0.0, 1.0]])  # coefficient of z in adj(zI-A)
    adj_1 = np.array([[-A[1, 1], A[0, 1]], [A[1, 0], -A[0, 0]]])
    n1 = C @ adj_z @ B
    n0 = C @ adj_1 @ B
    b = np.array([D, n1 + D * a[1], n0 + D * a[2]])
    return b, a


def noise_norm(mode: int, f: float, k: float, sr: float) -> float:
    """Gain that normalises the filtered uniform noise power."""
    if mode == FILT_BP:
        return math.sqrt((1.0 / 6.0) / ((0.5 - f) * f) * k)
    if mode == FILT_HP:
        fv5 = min(22050.0 / sr, 0.5)
        pw = (0.5 / fv5) ** min(k, 1.0)
        return math.sqrt(pw * k / ((((2 * k - 3) * f - k) * (2 * f - 1))))
    return math.sqrt(k / (((4 * k - 6) * f + 3) * f))


def shaper_params(amount: float):
    """(shape d, clamp k, dc offset, pre-drive, makeup) for distortion 0..1."""
    if amount < 1e-4:
        return 0.0, 1.0, 0.0, 1.0, 1.0
    d = amount ** 3 * 200.0
    if d < 1.0:
        k = (math.sqrt((d + 3.0) / d) + 1.0) / 3.0
        c = -1.0 / 3.0
        dc = ((abs(c) - c * c) * d + 1.0) * c
        return d, k, dc, 1.0, 1.0 - d / 30.0
    return d, 1.0, -0.407407403, d, 0.2 / ((d - 0.9589490891) + 0.4761904776) + 0.58


# --------------------------------------------------------------------------- params
@dataclass(frozen=True)
class VoiceParams:
    """Patch parameters in display units (as in .mtpreset files)."""
    wave: int = WAVE_SINE
    osc_freq: float = 632.46            # Hz
    osc_attack_ms: float = 0.0
    osc_decay_ms: float = 316.23
    mod_mode: int = MOD_DECAY
    mod_rate: float = 353.33            # ms (decay) or Hz (sine / noise)
    mod_amount: float = 0.0             # semitones
    mod_hold_ms: float = 0.0            # decay mode: the start pitch holds this long, then decays
    noise_filter: int = FILT_LP
    noise_freq: float = 20000.0         # Hz
    noise_q: float = 0.70710683
    noise_stereo: bool = False
    noise_env: int = NENV_EXP
    noise_attack_ms: float = 0.0
    noise_decay_ms: float = 316.23
    osc_mix: float = 0.5                # 1 = oscillator only, 0 = noise only
    distortion: float = 0.0             # 0..1
    eq_freq: float = 632.46
    eq_gain_db: float = 0.0
    level_db: float = 0.0
    pan: float = 0.0                    # -100..100
    osc_vel: float = 0.0                # 1.0 == 100 %
    noise_vel: float = 0.0
    mod_vel: float = 0.0
    pitch_ratio: float = 1.0            # note pitch offset (scales osc, noise filter, EQ, metal)
    mono: bool = False
    # metal section (off while both band levels are 0), see METAL_PARAMS
    metal_freq1: float = 205.3          # Hz
    metal_freq2: float = 304.4
    metal_freq3: float = 369.6
    metal_freq4: float = 522.7
    metal_freq5: float = 540.0
    metal_freq6: float = 800.0
    metal_tune: float = 0.0             # semitones, on all six oscillators
    metal_band1_freq: float = 3440.0    # Hz
    metal_band1_decay_ms: float = 300.0
    metal_band1_level: float = 0.0      # 0..1
    metal_band1_q: float = 1.0
    metal_band2_freq: float = 7100.0
    metal_band2_decay_ms: float = 800.0
    metal_band2_level: float = 0.0
    metal_band2_q: float = 1.0
    metal_curve: float = 0.0            # band envelopes: 0 exponential .. 1 linear
    metal_drive: float = 0.0            # band saturation, 0..1 (0 = clean)
    metal_hp_freq: float = 2000.0       # Hz
    metal_vel: float = 0.0              # 1.0 == 100 %
    # partials section (off while both tables are empty), see PARTIALS_PARAMS / PARTIALS_TABLES
    partials: str = ''                  # format_partials text
    partial_bands: str = ''             # format_bands text
    partials_level_db: float = 0.0
    partials_tune: float = 0.0          # semitones, on every partial and band edge
    partials_decay: float = 1.0         # scales every decay (1.0 == 100 %)
    partials_vel: float = 0.0           # 1.0 == 100 %
    partials_rise_ms: float = 0.0       # the section fades in (linearly) over this long from the hit
    partials_cut_ms: float = 0.0        # the section fades out from this long after the hit (0 = never)
    partials_fade_ms: float = 5.0       # exponentially, -60 dB at this time, then silent
    partials_slope: float = 24.0        # dB an octave each side of a noise band: 24 or 48


class _Derived:
    """Engine constants derived from VoiceParams."""

    def __init__(self, p: VoiceParams, sr: float):
        self.sr = sr
        self.max_inc = min(10000.0 / sr, 0.235)
        self.wave = int(p.wave)
        self.inc0 = min(max(p.osc_freq, 1.0) * p.pitch_ratio / (2.0 * sr), self.max_inc)

        # oscillator envelope
        t_atk = attack_time_s(p.osc_attack_ms)
        self.osc_atk_n = t_atk * sr
        self.osc_fade_n = max(min(t_atk, 0.01) * sr, 1.0)
        self.osc_atk_c = 1000.0 ** (1.0 / max(self.osc_atk_n, 1.0))
        t_dcy = min(max(p.osc_decay_ms, 10.0), 10000.0) / 1000.0
        self.osc_dcy_c = 1000.0 ** (-1.0 / max(t_dcy * sr, 1.0))

        # pitch modulation
        self.mod_mode = int(p.mod_mode)
        rng = 96.0 if self.mod_mode == MOD_DECAY else 48.0
        amt = max(-rng, min(rng, float(p.mod_amount)))
        self.mod_a = math.copysign(math.sqrt(abs(amt) / rng), amt)
        self.mod_scale = 8.0 if self.mod_mode == MOD_DECAY else 4.0
        self.mod_dc = 1.0
        self.mod_b0 = 0.0
        self.mod_b4 = 0.0
        self.mod_lpc = 0.0
        self.mod_hold8 = False
        self.mod_hold = 0.0     # 4-sample blocks
        if self.mod_mode == MOD_DECAY:
            hold = float(p.mod_hold_ms)
            if math.isfinite(hold) and hold > 0.0:
                self.mod_hold = min(hold, MOD_HOLD_MAX_MS) / 1000.0 * sr / 4.0
            T = float(p.mod_rate) / 1000.0
            if math.isfinite(T) and T > 0.0:
                self.mod_dc = 1000.0 ** (-1.0 / max(T * sr * 2.0, 1.0))
        elif self.mod_mode == MOD_SINE:
            self.mod_b0 = min(max(float(p.mod_rate), 0.0) / (2.0 * sr), self.max_inc)
        else:
            b0 = min(max(float(p.mod_rate), 0.0) / (2.0 * sr), self.max_inc)
            r = 2.0 * b0
            if b0 <= 0.0015:
                r *= 4.0
                self.mod_hold8 = True
            self.mod_b4 = math.sqrt(0.5 / r) if r > 1e-13 else 0.0
            self.mod_lpc = ((r * 11.01200485 - 13.33381844) * r + 5.846880913) * r

        # noise
        self.noise_filter = int(p.noise_filter)
        self.stereo = bool(p.noise_stereo) and not p.mono
        f = min(max(p.noise_freq * p.pitch_ratio / sr, 1e-4), 2.0 * self.max_inc)
        k = 1.0 / min(max(float(p.noise_q), 0.1), 10000.0)
        self.nf_b, self.nf_a = svf_tf(svf_coefs(f, k), self.noise_filter)
        self.noise_gain = noise_norm(self.noise_filter, f, k, sr)

        self.noise_env = int(p.noise_env)
        if self.noise_env == NENV_LINEAR:
            n_atk = max(p.noise_attack_ms, 0.0) / 1000.0
            n_dcy = min(max(p.noise_decay_ms, 6.6667), 6666.67) / 1000.0
            atk_p = 0.0
        else:
            atk_p = attack_norm(p.noise_attack_ms)
            n_atk = atk_p * math.exp(atk_p * 5.526204586 - 3.223619223) if atk_p > 1e-13 else 0.0
            n_dcy = min(max(p.noise_decay_ms, 10.0), 10000.0) / 1000.0
        self.n_atk_n = n_atk * sr
        self.n_fade_n = max(min(n_atk, 0.01) * sr, 1.0)
        self.n_atk_c = 1000.0 ** (1.0 / max(self.n_atk_n, 1.0))
        self.n_dcy_c = 1000.0 ** (-1.0 / max(n_dcy * sr, 1.0))
        self.n_lin_up = 1.0 / max(self.n_atk_n, 1.0)
        self.n_lin_dn = 1.0 / max(n_dcy * sr, 1.0)
        self.n_bursts = 1
        self.n_burst_c = 1.0
        if self.noise_env == NENV_MOD and self.n_atk_n > 0.0:
            self.n_bursts = max(1, int(n_atk / (atk_p * 0.095 + 0.005) + 2.12))
            self.n_burst_c = 15.84893227 ** (-1.0 / max(n_atk * sr / self.n_bursts, 1.0))

        # mix, shaper, EQ, output
        u = (1.0 - min(max(p.osc_mix, 0.0), 1.0)) * 2.0 - 1.0
        self.mix_osc = math.exp(u * -2.878231525) * (1.0 - u) if u > 0.0 else 1.0
        self.mix_noise = math.exp(u * 2.878231525) * (1.0 + u) if u < 0.0 else 1.0
        self.sh_d, self.sh_k, self.sh_dc, self.drive, post = shaper_params(min(max(p.distortion, 0.0), 1.0))

        gnorm = (min(max(p.eq_gain_db, -40.0), 40.0) / 40.0 + 1.0) / 2.0
        G = 10.0 ** (2.0 * gnorm - 1.0)
        K = 2.0 if gnorm >= 0.5 else 0.5
        self.eq_amt = G * K - K / G
        ef = min(max(p.eq_freq * p.pitch_ratio / sr, 1e-4), 2.0 * self.max_inc)
        self.eq_b, self.eq_a = svf_tf(svf_coefs(ef, K / G), FILT_BP)
        eq_att = 1.0 / G if gnorm > 0.5 else 1.0

        level = 10.0 ** (p.level_db / 20.0) if p.level_db > -60.0 else 0.0
        pn = (min(max(p.pan, -100.0), 100.0) + 100.0) / 200.0
        g = MASTER_SCALE * level * eq_att * post
        if p.mono:
            self.out_l = self.out_r = 0.75 * g
        else:
            self.out_l = (1.0 - pn * pn) * g
            self.out_r = (2.0 * pn - pn * pn) * g

        self.smooth = 1.0 - math.exp(-2.0 * math.pi * min(40.0 / sr, 0.5))
        self.release_slope = -100.0 / sr

        # partials section: straight to level and pan
        self.section = partials_on(p)
        if self.section:
            self.ps_n, self.ps_nb, self.ps_sp, self.ps_end, self.ps_nsec, self.ps_click = _section(
                str(p.partials), str(p.partial_bands), float(p.partials_tune), float(p.partials_decay),
                float(p.pitch_ratio), float(sr), float(band_order(p.partials_slope) * 6))
            lv = p.partials_level_db
            g = MASTER_SCALE * level * (10.0 ** (min(lv, 24.0) / 20.0) if lv > -60.0 else 0.0)
            if p.mono:
                self.ps_out_l = self.ps_out_r = 0.75 * g
            else:
                self.ps_out_l = (1.0 - pn * pn) * g
                self.ps_out_r = (2.0 * pn - pn * pn) * g
            cut = float(p.partials_cut_ms)
            self.ps_cut = cut / 1000.0 * sr if math.isfinite(cut) and cut > 0.0 else 0.0
            self.ps_fade = max(float(p.partials_fade_ms), 1.0) / 1000.0 * sr
            rise = float(p.partials_rise_ms)
            self.ps_rise = rise / 1000.0 * sr if math.isfinite(rise) and rise > 0.0 else 0.0

        # metal section
        self.metal = metal_on(p)
        if self.metal:
            ratio = p.pitch_ratio * 2.0 ** (min(max(p.metal_tune, -48.0), 48.0) / 12.0)
            self.metal_inc = [min(max(getattr(p, f), 0.0) * ratio / sr, 0.25) for f in METAL_FREQS]
            q = lambda v: min(max(v, 0.1), 50.0)
            self.metal_bp1 = _rbj(p.metal_band1_freq * p.pitch_ratio / sr, q(p.metal_band1_q), 'bp')
            self.metal_bp2 = _rbj(p.metal_band2_freq * p.pitch_ratio / sr, q(p.metal_band2_q), 'bp')
            self.metal_hp = _rbj(p.metal_hp_freq * p.pitch_ratio / sr, 0.70710678, 'hp')
            times = [max(min(max(t, 10.0), 10000.0) / 1000.0 * sr, 1.0)
                     for t in (p.metal_band1_decay_ms, p.metal_band2_decay_ms)]
            self.metal_dcy = [1000.0 ** (-1.0 / n) for n in times]
            self.metal_lstep = [1.0 / n for n in times]
            self.metal_curve = min(max(p.metal_curve, 0.0), 1.0)
            self.metal_drive = 20.0 * min(max(p.metal_drive, 0.0), 1.0)
            self.metal_lvl = [min(max(v, 0.0), 1.0) * METAL_GAIN
                              for v in (p.metal_band1_level, p.metal_band2_level)]


def _rbj(f: float, q: float, kind: str):
    """(b, a) of a bilinear band-pass (0 dB at its centre), high-pass or low-pass, f in
    cycles/sample."""
    w = 2.0 * math.pi * min(max(f, 1e-5), 0.45)
    alpha = math.sin(w) / (2.0 * q)
    c = math.cos(w)
    a0 = 1.0 + alpha
    a = np.array([1.0, -2.0 * c / a0, (1.0 - alpha) / a0])
    if kind == 'bp':
        b = np.array([alpha, 0.0, -alpha]) / a0
    elif kind == 'lp':
        b = np.array([(1.0 - c) / 2.0, 1.0 - c, (1.0 - c) / 2.0]) / a0
    else:
        b = np.array([(1.0 + c) / 2.0, -(1.0 + c), (1.0 + c) / 2.0]) / a0
    return b, a


# --------------------------------------------------------------------------- voice
def _pack(d: _Derived):
    """Flat parameter arrays for the render kernel."""
    p = np.zeros(K.NUM_P)
    p[K.P_INC0] = d.inc0
    p[K.P_MAX_INC] = d.max_inc
    p[K.P_OSC_ATK_C] = d.osc_atk_c
    p[K.P_OSC_DCY_C] = d.osc_dcy_c
    p[K.P_MOD_DC] = d.mod_dc
    p[K.P_MOD_B0] = d.mod_b0
    p[K.P_MOD_B4] = d.mod_b4
    p[K.P_MOD_LPC] = d.mod_lpc
    p[K.P_MOD_HOLD] = d.mod_hold
    p[K.P_NF_B:K.P_NF_B + 3] = d.nf_b
    p[K.P_NF_A:K.P_NF_A + 3] = d.nf_a
    p[K.P_NOISE_GAIN] = d.noise_gain
    p[K.P_N_ATK_C] = d.n_atk_c
    p[K.P_N_DCY_C] = d.n_dcy_c
    p[K.P_N_LIN_UP] = d.n_lin_up
    p[K.P_N_LIN_DN] = d.n_lin_dn
    p[K.P_N_BURST_C] = d.n_burst_c
    p[K.P_GAINS:K.P_GAINS + 6] = (d.mix_osc, d.mix_noise, d.drive, d.out_l, d.out_r, d.eq_amt)
    p[K.P_SH_D] = d.sh_d
    p[K.P_SH_K] = d.sh_k
    p[K.P_SH_DC] = d.sh_dc
    p[K.P_EQ_B:K.P_EQ_B + 3] = d.eq_b
    p[K.P_EQ_A:K.P_EQ_A + 3] = d.eq_a
    p[K.P_SMOOTH] = d.smooth
    p[K.P_RELEASE] = d.release_slope
    ip = np.zeros(K.NUM_IP, dtype=np.int64)
    ip[K.IP_WAVE] = d.wave
    ip[K.IP_MOD_MODE] = d.mod_mode
    ip[K.IP_NOISE_ENV] = d.noise_env
    ip[K.IP_STEREO] = int(d.stereo)
    ip[K.IP_MOD_HOLD8] = int(d.mod_hold8)
    ip[K.IP_N_BURSTS] = d.n_bursts
    if d.metal:
        ip[K.IP_METAL] = 1
        p[K.P_M_INC:K.P_M_INC + 6] = d.metal_inc
        for (b, a), ib, ia in ((d.metal_bp1, K.P_M_B1, K.P_M_A1), (d.metal_bp2, K.P_M_B2, K.P_M_A2),
                               (d.metal_hp, K.P_M_HB, K.P_M_HA)):
            p[ib:ib + 3] = b
            p[ia:ia + 3] = a
        p[K.P_M_DCY:K.P_M_DCY + 2] = d.metal_dcy
        p[K.P_M_LVL:K.P_M_LVL + 2] = d.metal_lvl
        p[K.P_M_LSTEP:K.P_M_LSTEP + 2] = d.metal_lstep
        p[K.P_M_CURVE] = d.metal_curve
        p[K.P_M_DRIVE] = d.metal_drive
    if d.section:
        ip[K.IP_PS] = 1
        ip[K.IP_PS_N] = d.ps_n
        ip[K.IP_PS_NB] = d.ps_nb
        ip[K.IP_PS_NSEC] = d.ps_nsec
        ip[K.IP_PS_CLICK] = int(d.ps_click)
        p[K.P_PS_OUT_L] = d.ps_out_l
        p[K.P_PS_OUT_R] = d.ps_out_r
        p[K.P_PS_CUT] = d.ps_cut
        p[K.P_PS_FADE] = d.ps_fade
        p[K.P_PS_RISE] = d.ps_rise
        p[K.P_PS_REL] = -d.release_slope
        p[K.P_PS_END] = d.ps_end
    return p, ip


class DrumVoice:
    """One drum voice (one per channel, monophonic, retriggerable).

    The voice state lives in two flat arrays that the compiled kernel
    (voice_kernel.py) advances; events are applied here, at the start of the
    next render.
    """

    def __init__(self, sample_rate: int = 44100, seed: int = 0):
        self.sr = float(sample_rate)
        # Separate streams for the noise source and the random pitch modulation, so
        # the output does not depend on how the audio is split into blocks
        self._rng = np.random.default_rng(seed)
        self._mod_rng = np.random.default_rng((seed, 1))
        # the metal section draws its oscillator phases from its own stream, so a
        # patch without it renders as before
        self._metal_rng = np.random.default_rng((seed, 2))
        # and the partials section seeds its band noise from another
        self._ps_rng = np.random.default_rng((seed, 3))
        self._sp = np.zeros(0)
        self._ss = np.zeros(0)
        self._su = np.zeros(0, dtype=np.uint64)
        self.params = VoiceParams()
        self.d = _Derived(self.params, self.sr)
        self._p, self._ip = _pack(self.d)
        self._fs = np.zeros(K.NUM_S)
        self._is = np.zeros(K.NUM_I, dtype=np.int64)
        self._is[K.I_OSTAGE] = 4
        self._is[K.I_NSTAGE] = 4
        self._fs[K.S_HOLD] = self.d.inc0
        self._fs[K.S_PENV] = 1.0
        self._fs[K.S_DRIFT] = 1.0
        self._fs[K.S_G:K.S_G + 6] = self._p[K.P_GAINS:K.P_GAINS + 6]
        self.clock = 0                  # samples rendered (block grid is absolute)
        self._carry = np.zeros((0, 2), dtype=np.float32)
        self._pending = []              # ('trig', velocity) / ('choke',)
        if not DrumVoice._kernel_ready:
            self._warm_up()

    _kernel_ready = False

    def _warm_up(self):
        """Compile (or load from the cache) the kernel now rather than in the audio callback."""
        fs, st = self._fs.copy(), self._is.copy()
        rng = np.random.default_rng(0)
        K.reset_osc(self._p, self._ip, fs, st, rng)
        K.render(self._p, self._ip, fs, st, rng, rng, 1, 0, np.empty((4, 2)), self._sp, self._ss, self._su)
        DrumVoice._kernel_ready = True

    @property
    def pitch_drift(self) -> float:
        return float(self._fs[K.S_DRIFT])

    @pitch_drift.setter
    def pitch_drift(self, value: float):
        self._fs[K.S_DRIFT] = value

    # ---------------------------------------------------------------- params
    def set_params(self, params: VoiceParams):
        if params == self.params:
            return
        old = self.params
        self.params = params
        self.d = _Derived(params, self.sr)
        self._p, self._ip = _pack(self.d)
        self._update_section(old)
        if old.noise_filter != params.noise_filter:
            self._fs[K.S_NF_ZI:K.S_NF_ZI + 4] = 0.0
        if not self.is_active:
            self._fs[K.S_G:K.S_G + 6] = self._p[K.P_GAINS:K.P_GAINS + 6]

    def _update_section(self, old: VoiceParams):
        """Point the kernel at the partials section's constants. Its state is kept when only
        a knob changed; other tables silence it (the next hit starts them)."""
        d = self.d
        if not d.section:
            self._sp = np.zeros(0)
            self._is[K.I_PSTAGE] = 0
            return
        self._sp = d.ps_sp
        n, nb = d.ps_n, d.ps_nb
        same = ((old.partials, old.partial_bands, band_order(old.partials_slope))
                == (self.params.partials, self.params.partial_bands, band_order(self.params.partials_slope)))
        size = 4 * n + (2 * d.ps_nsec + 2) * nb
        if not same or len(self._ss) != size or len(self._su) != nb:
            self._ss = np.zeros(size)
            self._su = self._ps_rng.integers(1, 2 ** 63, size=nb, dtype=np.uint64)
            self._is[K.I_PSTAGE] = 0
        self._fs[K.S_PS_DRIFT] = 0.0    # the rotations follow the new tuning at the next render

    @property
    def is_active(self) -> bool:
        # a silent carry (block remainder) is not activity: idle() consumes it as well
        return (self._is[K.I_OSTAGE] != 4 or self._is[K.I_NSTAGE] != 4 or self._is[K.I_MSTAGE] != 0
                or self._is[K.I_PSTAGE] != 0
                or len(self._pending) > 0 or bool(self._is[K.I_EQ_TAIL]) or bool(self._carry.any()))

    # ---------------------------------------------------------------- events
    def trigger(self, velocity: float = 127.0):
        self._pending.append(('trig', float(velocity)))

    def choke(self):
        self._pending.append(('choke',))

    def _apply_pending(self):
        fs, st = self._fs, self._is
        for ev in self._pending:
            if ev[0] == 'choke':
                for stage, lvl, slope in ((K.I_OSTAGE, K.S_OLVL, K.S_OSLOPE),
                                          (K.I_NSTAGE, K.S_NLVL, K.S_NSLOPE)):
                    if st[stage] != 4:
                        fs[slope] = max(fs[lvl], 1e-6) * self.d.release_slope
                        st[stage] = 3
                if st[K.I_MSTAGE] != 0:
                    c = self._p[K.P_M_CURVE]
                    for i in range(2):
                        level = fs[K.S_M_ENV + i] * (1.0 - c) + fs[K.S_M_LIN + i] * c
                        fs[K.S_M_ENV + i] = fs[K.S_M_LIN + i] = level
                        fs[K.S_M_REL + i] = -level * self.d.release_slope
                    st[K.I_MSTAGE] = 2
                if st[K.I_PSTAGE] == 1:
                    st[K.I_PSTAGE] = 2
            else:
                self._do_trigger(ev[1])
        self._pending = []

    def _do_trigger(self, velocity):
        d = self.d
        p = self.params
        fs, st = self._fs, self._is
        fs[K.S_G:K.S_G + 6] = self._p[K.P_GAINS:K.P_GAINS + 6]
        if d.section:
            self._start_section(velocity)
            return
        osc_vel = velocity_gain(velocity, p.osc_vel) * OSC_GAIN
        noise_vel = velocity_gain(velocity, p.noise_vel)
        fs[K.S_OSC_VEL] = osc_vel
        fs[K.S_NOISE_VEL] = noise_vel
        am = d.mod_a * velocity_factor(velocity, p.mod_vel)
        fs[K.S_A] = abs(am) * d.mod_scale * am
        if d.osc_atk_n > 0.0:
            st[K.I_OSTAGE] = 0
            fs[K.S_OSLOPE] = max(fs[K.S_OLVL], 1e-6) * (-1.0 / d.osc_fade_n)
        else:
            fs[K.S_OLVL] = osc_vel
            st[K.I_OSTAGE] = 2
            K.reset_osc(self._p, self._ip, fs, st, self._mod_rng)
        if d.noise_env == NENV_MOD or d.n_atk_n <= 0.0:
            fs[K.S_NLVL] = noise_vel
            st[K.I_NSTAGE] = 2 if d.n_atk_n <= 0.0 else 1
            st[K.I_NCOUNT] = 0
        else:
            st[K.I_NSTAGE] = 0
            fs[K.S_NSLOPE] = max(fs[K.S_NLVL], 1e-6) * (-1.0 / d.n_fade_n)
        if d.metal:
            # free-running oscillators: their phases at a hit are unrelated
            fs[K.S_M_PH:K.S_M_PH + 6] = self._metal_rng.random(6)
            fs[K.S_M_ENV:K.S_M_ENV + 2] = 1.0
            fs[K.S_M_LIN:K.S_M_LIN + 2] = 1.0
            fs[K.S_M_VEL] = velocity_gain(velocity, p.metal_vel)
            st[K.I_MSTAGE] = 1

    def _start_section(self, velocity):
        """A hit of the partials section: every partial at its start, the band filters
        cleared, the band and click gains at their levels."""
        d = self.d
        n, nb = d.ps_n, d.ps_nb
        sp, ss = self._sp, self._ss
        sec = d.ps_nsec
        ss[0:n] = sp[3 * n:4 * n]
        ss[n:2 * n] = sp[4 * n:5 * n]
        ss[4 * n:4 * n + 2 * sec * nb] = 0.0
        o = 5 * n + 5 * sec * nb             # band decays, band gains, click decays, click gains
        ss[4 * n + 2 * sec * nb:4 * n + (2 * sec + 1) * nb] = sp[o + nb:o + 2 * nb]
        ss[4 * n + (2 * sec + 1) * nb:] = sp[o + 3 * nb:o + 4 * nb]
        self._fs[K.S_PS_VEL] = velocity_gain(velocity, self.params.partials_vel)
        self._fs[K.S_PS_REL] = 1.0
        self._fs[K.S_PS_DRIFT] = 0.0
        self._is[K.I_PSTAGE] = 1
        self._is[K.I_PS_T] = 0
        self._is[K.I_PS_M] = n

    # ---------------------------------------------------------------- render
    def _render_blocks(self, nb):
        out = np.empty((nb * 4, 2))
        K.render(self._p, self._ip, self._fs, self._is, self._rng, self._mod_rng,
                 nb, self.clock // 4, out, self._sp, self._ss, self._su)
        self.clock += nb * 4
        return out

    def idle(self, num_samples: int):
        """Advance time without rendering (silent / muted channel)."""
        self._pending = []
        k = min(len(self._carry), num_samples)
        self._carry = self._carry[k:]
        n = num_samples - k
        if n > 0:
            nb = (n + 3) // 4
            self.clock += nb * 4
            self._carry = np.zeros((nb * 4 - n, 2), dtype=np.float32)

    def process(self, num_samples: int) -> np.ndarray:
        """Render `num_samples` stereo samples (float32, shape (n, 2))."""
        out = np.empty((num_samples, 2), dtype=np.float32)
        pos = 0
        if len(self._carry):
            k = min(len(self._carry), num_samples)
            out[:k] = self._carry[:k]
            self._carry = self._carry[k:]
            pos = k
        if pos < num_samples:
            if self._pending:
                self._apply_pending()
            need = num_samples - pos
            nb = (need + 3) // 4
            blk = self._render_blocks(nb)
            out[pos:] = blk[:need]
            self._carry = blk[need:].astype(np.float32)
        return out


# --------------------------------------------------------------------------- preset helpers
def _num(v, default=0.0) -> float:
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().strip('"')
    if s.lower().startswith('inf'):
        return float('inf')
    num = ''
    for ch in s:
        if ch in '+-.0123456789eE' and not (ch in 'eE' and not num):
            num += ch
        else:
            break
    try:
        return float(num)
    except ValueError:
        return default


def params_from_patch(patch: dict, pitch_ratio: float = 1.0, mono: bool = False) -> VoiceParams:
    """Build VoiceParams from a parsed .mtpreset / .mtdrum patch (display units)."""
    g = patch.get
    s = lambda k, d: str(g(k, d)).strip().strip('"')
    mix = g('Mix', 50.0)
    if isinstance(mix, (tuple, list)):
        osc_pct = _num(mix[0])
    elif isinstance(mix, str) and '/' in mix:
        osc_pct = _num(mix.split('/')[0])
    else:
        osc_pct = _num(mix, 50.0)
    stereo = g('NStereo', False)
    stereo = stereo if isinstance(stereo, bool) else s('NStereo', 'Off').lower() in ('on', 'true', '1')
    return VoiceParams(
        wave={'Sine': 0, 'Triangle': 1, 'Saw': 2}.get(s('OscWave', 'Sine'), 0),
        osc_freq=_num(g('OscFreq', 632.46)),
        osc_attack_ms=_num(g('OscAtk', 0.0)),
        osc_decay_ms=_num(g('OscDcy', 316.23)),
        mod_mode={'Decay': 0, 'Sine': 1, 'Noise': 2}.get(s('ModMode', 'Decay'), 0),
        mod_rate=_num(g('ModRate', 353.33)),
        mod_amount=_num(g('ModAmt', 0.0)),
        mod_hold_ms=_num(g('ModHold', 0.0)),
        noise_filter={'LP': 0, 'BP': 1, 'HP': 2}.get(s('NFilMod', 'LP'), 0),
        noise_freq=_num(g('NFilFrq', 20000.0)),
        noise_q=_num(g('NFilQ', 0.70710683)),
        noise_stereo=bool(stereo),
        noise_env={'Exp': 0, 'Linear': 1, 'Mod': 2}.get(s('NEnvMod', 'Exp'), 0),
        noise_attack_ms=_num(g('NEnvAtk', 0.0)),
        noise_decay_ms=_num(g('NEnvDcy', 316.23)),
        osc_mix=osc_pct / 100.0,
        distortion=_num(g('DistAmt', 0.0)) / 100.0,
        eq_freq=_num(g('EQFreq', 632.46)),
        eq_gain_db=_num(g('EQGain', 0.0)),
        level_db=_num(g('Level', 0.0)),
        pan=_num(g('Pan', 0.0)),
        osc_vel=_num(g('OscVel', 0.0)) / 100.0,
        noise_vel=_num(g('NVel', 0.0)) / 100.0,
        mod_vel=_num(g('ModVel', 0.0)) / 100.0,
        pitch_ratio=pitch_ratio,
        mono=mono,
        **metal_from_patch(patch),
        **partials_from_patch(patch),
    )


def partials_from_patch(patch: dict) -> dict:
    """Partials section fields of a parsed .mtdrum / .mtpreset patch (all defaults, so off,
    when it has none of its keys)."""
    out = {}
    for name, (default, key, *_) in PARTIALS_PARAMS.items():
        v = _num(patch[key], default) if key in patch else default
        out[name] = v / 100.0 if name in PARTIALS_PERCENT and key in patch else v
    for name, key in PARTIALS_TABLES.items():
        v = patch.get(key, '')
        out[name] = v.strip().strip('"') if isinstance(v, str) else ''
    return out


def metal_from_patch(patch: dict) -> dict:
    """Metal section fields of a parsed .mtdrum / .mtpreset patch (all defaults, so off,
    when it has no metal keys)."""
    out = {}
    for name, (default, key, *_) in METAL_PARAMS.items():
        v = _num(patch[key], default) if key in patch else default
        out[name] = v / 100.0 if name in METAL_PERCENT and key in patch else v
    return out
