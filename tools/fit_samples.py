"""
Fit a .mtdrum drum patch to each one-shot WAV in a folder.

    venv/bin/python tools/fit_samples.py <samples dir> <output dir> [--wav]

Each sample is matched in four steps:

A sample whose energy above 2 kHz is mostly steady partials (cymbals, hats,
cowbells) is fitted with the metal section on: the six oscillator
frequencies are the square-wave fundamentals whose odd harmonics explain those
partials, and the search adds the band-passes, their decays and levels, the Q
and the high-pass.

1. Starts: patches built on the partials found in the sample, the
   predictions of the model trained by tools/fit_prior.py (fit_prior.pt at the
   repository root), and, with --library-dir, random drum patches of the
   sample's category (guessed from its folder and file name). Each is rendered
   and scored once.
2. Screening: a short CMA-ES run from the best starts of each kind, each with
   its own switch settings (wave, pitch mod mode, noise filter, noise envelope).
3. Long runs: CMA-ES from the best screened starts, with the switches fixed,
   restarted from its best point with a doubled population (IPOP).
4. Switches: a short run with each single switch changed; a better one is
   polished with another long run.

The score compares the render with the sample after scaling the render to the
sample's energy, so it ignores loudness; Level is then set so the patch peaks
at --peak-db. It sums:

- log-mel spectrogram differences in dB at three resolutions, plus a fine
  linear spectrum below 2 kHz and the frame energy envelope. Each cell is
  weighted by how loud it is in the louder of the two signals, from 0 at
  50 dB under the peak to 1 at the peak, so the audible parts dominate;
- the relative magnitude error of the mel spectrograms (x10);
- how far the energy below 2 kHz sits in pitch from the sample's, as the
  earth mover's distance between the two spectra on an octave scale (x12, so
  a semitone costs 1), weighted by the share of the sample's energy below
  2 kHz. Unlike the spectral differences, it shrinks as a partial moves
  toward the right pitch;
- above 2 kHz, where the mel bands are too wide to tell partials from noise,
  the dB difference of the fine spectra summed over the sound (x0.25) and the
  difference in the share of that energy held by partials (x5), both weighted
  by the share of the sample's energy above 2 kHz;
- the dB difference of the octave bands from 125 Hz to 16 kHz frame by frame
  (x0.5), every octave counting the same however quiet, so the body under a
  loud band (the lows of a cymbal under its sizzle) is matched too;
- for a metal sample, the dB difference of each of its partials, summed over
  the sound (x0.4): the render has them at the same frequencies, and their
  levels carry the shape of the sample's filters.

Samples are mixed to mono and trimmed to their onset. The output folder gets
one <sample name>.mtdrum per sample, in the same subfolder as the sample in
the sample folder, and fit_report.csv; --wav adds <sample name>.fit.wav next
to it, the patch rendered next to the trimmed sample (left: sample, right:
patch) for listening.
"""

from __future__ import annotations

import os

# one BLAS thread per worker process: the samples are fitted in parallel instead
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import argparse
import concurrent.futures
import csv
import importlib.util
import itertools
import math
import pathlib
import random
import sys
import time

import cma
import numpy as np
import scipy.fft
import scipy.ndimage
import scipy.signal
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pythonic.preset_manager import DrumPatchParser, DrumPatchWriter  # noqa: E402
from pythonic.voice import (DrumVoice, FILT_BP, FILT_HP, FILT_LP, METAL_FREQS, MOD_DECAY,  # noqa: E402
                            NENV_EXP, VoiceParams, WAVE_SAW, WAVE_SINE, WAVE_TRI,
                            attack_norm, metal_on, params_from_patch)

SR = 44100
DEFAULT_PRIOR = ROOT / 'fit_prior.pt'

# File or folder name keyword -> --library-dir subfolders to draw starts from.
# The first match wins; a sample that matches nothing draws from every subfolder.
CATEGORIES = [
    ('closed', ['ch']), ('open', ['oh']), ('hat', ['ch', 'oh']),
    ('kick', ['bd']), ('bass drum', ['bd']), ('bd', ['bd']),
    ('snare', ['sd']), ('sd', ['sd']),
    ('clap', ['clap']), ('snap', ['clap', 'perc']),
    ('rim', ['perc', 'sd']),
    ('tom', ['tom']), ('conga', ['tom', 'perc']), ('timbale', ['tom', 'perc']),
    ('bongo', ['tom', 'perc']),
    ('cowbell', ['cowbell']),
    ('crash', ['cy']), ('ride', ['cy']), ('cymbal', ['cy']),
    ('shaker', ['shaker']), ('rachet', ['shaker', 'perc']), ('ratchet', ['shaker', 'perc']),
    ('maraca', ['shaker']), ('tamb', ['shaker']), ('cabasa', ['shaker']),
    ('perc', ['perc']),
]
CHOKE_CATEGORIES = {'ch', 'oh'}

# Continuous controls: (VoiceParams field, low, high, scale). 'knob' attacks
# are fitted as the knob position of the attack curve; 'amount' is the
# signed square-root pitch mod amount the engine uses, scaled by the mode range.
CONTINUOUS = [
    ('osc_freq', 20.0, 20000.0, 'log'),
    ('osc_attack_ms', 0.0, 1.0, 'knob'),
    ('osc_decay_ms', 10.0, 10000.0, 'log'),
    ('mod_rate', 1.0, 2000.0, 'log'),
    ('mod_amount', -1.0, 1.0, 'amount'),
    ('noise_freq', 20.0, 20000.0, 'log'),
    ('noise_q', 0.5, 20.0, 'log'),
    ('noise_attack_ms', 0.0, 1.0, 'knob'),
    ('noise_decay_ms', 10.0, 10000.0, 'log'),
    ('osc_mix', 0.0, 1.0, 'linear'),
    ('distortion', 0.0, 1.0, 'linear'),
    ('eq_freq', 20.0, 20000.0, 'log'),
    ('eq_gain_db', -40.0, 40.0, 'linear'),
]
# The metal section's controls, appended to the vector of a metal sample (its
# oscillator frequencies come from the analysis and stay fixed)
METAL_CONTINUOUS = [
    ('metal_band1_freq', 500.0, 16000.0, 'log'),
    ('metal_band1_decay_ms', 10.0, 10000.0, 'log'),
    ('metal_band1_level', 0.0, 1.0, 'linear'),
    ('metal_band1_q', 0.5, 20.0, 'log'),
    ('metal_band2_freq', 500.0, 16000.0, 'log'),
    ('metal_band2_decay_ms', 10.0, 10000.0, 'log'),
    ('metal_band2_level', 0.0, 1.0, 'linear'),
    ('metal_band2_q', 0.5, 20.0, 'log'),
    ('metal_curve', 0.0, 1.0, 'linear'),
    ('metal_drive', 0.0, 1.0, 'linear'),
    ('metal_hp_freq', 20.0, 16000.0, 'log'),
]
SWITCHES = ('wave', 'mod_mode', 'noise_filter', 'noise_env')
SWITCH_CHOICES = 3


def mod_range(mod_mode: int) -> float:
    return 96.0 if mod_mode == MOD_DECAY else 48.0


def encode(p: VoiceParams, metal: bool = False) -> np.ndarray:
    """Unit-cube vector of the continuous controls of p (and of its metal section)."""
    u = []
    for name, lo, hi, scale in CONTINUOUS + (METAL_CONTINUOUS if metal else []):
        v = float(getattr(p, name))
        if not math.isfinite(v):
            v = hi if v > 0 else lo
        if scale == 'log':
            x = math.log(min(max(v, lo), hi) / lo) / math.log(hi / lo)
        elif scale == 'knob':
            x = attack_norm(v)
        elif scale == 'amount':
            a = min(max(v / mod_range(p.mod_mode), -1.0), 1.0)
            x = (math.copysign(math.sqrt(abs(a)), a) + 1.0) / 2.0
        else:
            x = (min(max(v, lo), hi) - lo) / (hi - lo)
        u.append(x)
    return np.clip(np.array(u), 0.0, 1.0)


# The controls a metal sample leaves alone: its colour comes from the metal band-passes
# (the drum machines have no EQ on their metal path), and an EQ boost up high makes
# the metal sound smaller
METAL_FIXED = {'eq_gain_db': 0.0}


def decode(u: np.ndarray, switches: tuple, freqs: tuple = ()) -> VoiceParams:
    """VoiceParams from a unit-cube vector and the (wave, mod mode, filter, env) switches;
    a vector longer than CONTINUOUS carries the metal section, on the oscillator freqs
    (and leaves METAL_FIXED as it says)."""
    fields = dict(zip(SWITCHES, switches))
    if len(u) > len(CONTINUOUS):
        fields.update(zip(METAL_FREQS, freqs))
    for x, (name, lo, hi, scale) in zip(np.clip(u, 0.0, 1.0), CONTINUOUS + METAL_CONTINUOUS):
        x = float(x)
        if scale == 'log':
            v = lo * (hi / lo) ** x
        elif scale == 'knob':
            v = attack_time_s_from_knob(x) * 1000.0
        elif scale == 'amount':
            s = 2.0 * x - 1.0
            v = math.copysign(s * s, s) * mod_range(fields['mod_mode'])
        else:
            v = lo + x * (hi - lo)
        fields[name] = v
    if len(u) > len(CONTINUOUS):
        fields.update(METAL_FIXED)
    return VoiceParams(**fields)


def attack_time_s_from_knob(p: float) -> float:
    return p * math.exp(p * 5.526204586 - 3.223619223) if p > 1e-6 else 0.0


def switches_of(p: VoiceParams) -> tuple:
    return tuple(int(getattr(p, s)) for s in SWITCHES)


def neutral(p: VoiceParams) -> VoiceParams:
    """p with the controls the fit leaves alone at their written values."""
    return VoiceParams(**{**p.__dict__, 'noise_stereo': False, 'pan': 0.0, 'level_db': 0.0,
                          'osc_vel': 0.0, 'noise_vel': 0.0, 'mod_vel': 0.0, 'pitch_ratio': 1.0,
                          'mono': False})


# --------------------------------------------------------------------------- audio
def render(p: VoiceParams, n: int) -> np.ndarray:
    v = DrumVoice(SR, seed=0)
    v.set_params(p)
    v.trigger(127)
    return v.process(n).astype(np.float64).mean(axis=1)


def load_target(path: str, max_seconds: float) -> np.ndarray:
    """Mono sample, resampled to SR if needed, trimmed to its onset and to max_seconds."""
    x, sr = sf.read(path, always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        g = math.gcd(sr, SR)
        x = scipy.signal.resample_poly(x, SR // g, sr // g)
    peak = np.abs(x).max()
    if peak <= 0:
        raise ValueError('silent sample')
    start = int(np.argmax(np.abs(x) > peak * 1e-3))
    return x[max(start - 2, 0):][:int(max_seconds * SR)]


def mel_bank(n_fft: int, n_mels: int) -> np.ndarray:
    """(bins, bands) triangular mel filters from 30 Hz to Nyquist."""
    mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    edges = hz(np.linspace(mel(30.0), mel(SR / 2.0), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1.0 / SR)
    fb = np.zeros((n_mels, len(bins)))
    for i in range(n_mels):
        lo, mid, hi = edges[i:i + 3]
        rise = (bins - lo) / max(mid - lo, 1e-9)
        fall = (hi - bins) / max(hi - mid, 1e-9)
        fb[i] = np.clip(np.minimum(rise, fall), 0.0, None)
    # bands narrower than a bin still get the nearest bin
    for i in np.where(fb.sum(axis=1) == 0)[0]:
        fb[i, np.argmin(np.abs(bins - edges[i + 1]))] = 1.0
    return fb.T.astype(np.float32)


def stft_power(x: np.ndarray, n_fft: int, hop: int, win: np.ndarray) -> np.ndarray:
    """(frames, bins) power spectrogram, the first frame centred on sample 0."""
    y = np.concatenate([np.zeros(n_fft // 2, np.float32), x.astype(np.float32),
                        np.zeros(n_fft, np.float32)])
    spec = scipy.fft.rfft(np.lib.stride_tricks.sliding_window_view(y, n_fft)[::hop] * win, axis=1)
    return spec.real ** 2 + spec.imag ** 2


RESOLUTIONS = [(n_fft, n_fft // 4, np.hanning(n_fft).astype(np.float32), mel_bank(n_fft, n_mels))
               for n_fft, n_mels in ((2048, 96), (512, 64), (128, 32))]
ENV_WIN, ENV_HOP = 256, 64

# Fine spectrum for the pitch of low partials: 5.4 Hz bins from 30 Hz to 2 kHz
FINE_FFT, FINE_HOP = 8192, 1024
FINE_WIN = np.hanning(FINE_FFT).astype(np.float32)
_fine_hz = np.fft.rfftfreq(FINE_FFT, 1.0 / SR)
FINE_BINS = np.where((_fine_hz >= 30.0) & (_fine_hz <= 2000.0))[0]
FINE_STEP_OCT = np.diff(np.log2(_fine_hz[FINE_BINS]))   # octave width between neighbouring bins
# The same fine spectrum from 2 to 16 kHz, summed over the sound: the partials of
# metal sounds, which the mel bands (some 270 Hz wide at 7 kHz) blur into noise
SUM_BINS = np.where((_fine_hz > 300.0) & (_fine_hz <= 16000.0))[0]
HIGH_IN_SUM = _fine_hz[SUM_BINS] > 2000.0
TONAL_SPAN = 151       # bins (about 800 Hz) of the running median a partial stands above
TONAL_DB = 8.0         # a bin this far above the running median belongs to a partial

FLOOR_DB = 50.0        # weights and log floors reach zero this far under the peak
MAG_WEIGHT = 10.0      # score per unit of relative magnitude error
PITCH_WEIGHT = 12.0    # score per octave of pitch distance below 2 kHz
HIGH_WEIGHT = 0.25     # score per dB of summed high spectrum difference above 2 kHz
BALANCE_WEIGHT = 0.5   # score per dB of octave band balance difference
BALANCE_RANGE = 60.0   # dB under a frame's loudest octave that its balance reaches
PARTIAL_WEIGHT = 0.4   # score per dB of partial level difference (metal samples)


def _octave_groups() -> np.ndarray:
    """(mel bands, octaves) matrix summing the 2048-point mel bands into the octaves from
    125 Hz to 16 kHz, by band centre."""
    mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
    n_mels = RESOLUTIONS[0][3].shape[1]
    centres = hz(np.linspace(mel(30.0), mel(SR / 2.0), n_mels + 2))[1:-1]
    octave = np.floor(np.log2(centres / 125.0)).astype(int)
    groups = np.zeros((n_mels, 7))
    for i, o in enumerate(octave):
        if 0 <= o < 7:
            groups[i, o] = 1.0
    return groups


OCTAVES = _octave_groups()

TONAL_WEIGHT = 5.0     # score per unit of difference in the share of high energy in partials


def spectra(x: np.ndarray) -> tuple:
    """(mel spectrograms, the fine low spectrum and the frame energy envelope; the fine
    spectrum from 300 Hz to 16 kHz summed over the sound)."""
    out = [stft_power(x, n_fft, hop, win) @ fb for n_fft, hop, win, fb in RESOLUTIONS]
    fine = stft_power(x, FINE_FFT, FINE_HOP, FINE_WIN)
    out.append(fine[:, FINE_BINS])
    y = np.concatenate([np.zeros(ENV_WIN // 2, np.float32), x.astype(np.float32),
                        np.zeros(ENV_WIN, np.float32)])
    out.append(np.mean(np.lib.stride_tricks.sliding_window_view(y, ENV_WIN)[::ENV_HOP] ** 2,
                       axis=1)[:, None])
    return out, fine[:, SUM_BINS].sum(axis=0)


def tonality(summed: np.ndarray) -> float:
    """Share of the energy above 2 kHz of a summed spectrum (spectra()) in partials: bins
    TONAL_DB above the running median. Near 0 for noise, whose summed spectrum is smooth."""
    high = summed[HIGH_IN_SUM]
    db = 10.0 * np.log10(high + high.max() * 1e-12 + 1e-30)
    peaks = db - scipy.ndimage.median_filter(db, size=TONAL_SPAN, mode='nearest') > TONAL_DB
    return float(high[peaks].sum() / max(high.sum(), 1e-30))


def _cdf(p: np.ndarray) -> np.ndarray:
    """Per-frame cumulative distribution of a (frames, bins) power spectrum."""
    c = np.cumsum(p, axis=1)
    return c / np.maximum(c[:, -1:], 1e-30)


class Target:
    """A trimmed sample and its spectra; scores renders against it."""

    def __init__(self, x: np.ndarray):
        self.x = x
        peak = np.abs(x).max()
        tail = np.sqrt(np.mean(x[-max(len(x) // 20, 32):] ** 2))
        # A sample that fades out is compared past its end against silence, so
        # a patch ringing on is penalised; a cut-off one only up to its end.
        self.n = len(x) + (max(len(x) // 4, int(0.05 * SR)) if tail < peak * 0.005 else 0)
        padded = np.concatenate([x, np.zeros(self.n - len(x))])
        self.energy = float(np.sum(padded ** 2))
        self.specs, summed = spectra(padded)
        self.summed = summed
        high = summed[HIGH_IN_SUM]
        self.floors = [s.max() * 10.0 ** (-FLOOR_DB / 10.0) for s in self.specs]
        self.logs = [10.0 * np.log10(s + fl) for s, fl in zip(self.specs, self.floors)]
        self.refs = [lg.max() - FLOOR_DB for lg in self.logs]
        self.wsums = [max(float(np.sum(np.clip((lg - r) / FLOOR_DB, 0.0, 1.0))), 1e-9)
                      for lg, r in zip(self.logs, self.refs)]
        self.mags = [np.sqrt(s) for s in self.specs[:len(RESOLUTIONS)]]
        self.mag_norms = [float(np.linalg.norm(m)) for m in self.mags]
        fine = self.specs[len(RESOLUTIONS)]
        self.fine_cdf = _cdf(fine)
        self.fine_energy = fine.sum(axis=1)
        full = stft_power(padded, FINE_FFT, FINE_HOP, FINE_WIN)
        self.low_share = float(fine.sum() / max(full.sum(), 1e-30))
        self.high_share = float(high.sum() / max(full.sum(), 1e-30))
        self.high_floor = high.max() * 10.0 ** (-FLOOR_DB / 10.0)
        self.high_log = 10.0 * np.log10(high + self.high_floor)
        self.high_ref = self.high_log.max() - FLOOR_DB
        self.tonality = tonality(summed)
        self.octaves, self.octave_floor = self._octave_db(self.specs[0])
        self.metal_freqs = ()
        self.partial_bins = None

    def use_metal(self, freqs: tuple):
        """Fit with the metal section on these oscillator frequencies: the render has the
        sample's partials at the same frequencies, so their levels are compared one by one
        (they carry the shape of the sample's filters)."""
        self.metal_freqs = tuple(freqs)
        if not freqs:
            self.partial_bins = None
            return
        db = 10.0 * np.log10(self.summed + self.summed.max() * 1e-12)
        peaks, _ = scipy.signal.find_peaks(
            db - scipy.ndimage.median_filter(db, TONAL_SPAN, mode='nearest'), height=TONAL_DB, distance=3)
        peaks = peaks[(peaks > 0) & (peaks < len(db) - 1)]
        self.partial_floor = self.summed.max() * 10.0 ** (-FLOOR_DB / 10.0)
        self.partial_db = 10.0 * np.log10(self.summed[peaks] + self.partial_floor)
        self.partial_w = np.clip((self.partial_db - self.partial_db.max() + FLOOR_DB) / FLOOR_DB, 0.0, 1.0)
        self.partial_bins = peaks

    def _octave_db(self, mel: np.ndarray, floor=None):
        """(frames, 7) octave band levels in dB, floored 120 dB under the loudest (the
        target's), and that floor."""
        bands = mel @ OCTAVES
        if floor is None:
            floor = bands.max() * 1e-12 + 1e-30
        return 10.0 * np.log10(bands + floor), floor

    def loss(self, y: np.ndarray, parts: bool = False):
        """Score of render y; with parts, (spectra dB, magnitude, pitch, high dB, tonality,
        octave balance) before weighting."""
        e = float(np.sum(y ** 2))
        if not math.isfinite(e) or e < 1e-20:
            return (1e3,) * 7 if parts else 1e3
        specs, summed = spectra(y * math.sqrt(self.energy / e))
        high = summed[HIGH_IN_SUM]
        log_err = 0.0
        for s, fl, lt, ref, ws in zip(specs, self.floors, self.logs, self.refs, self.wsums):
            ls = 10.0 * np.log10(s + fl)
            w = np.clip((np.maximum(lt, ls) - ref) / FLOOR_DB, 0.0, 1.0)
            log_err += float(np.sum(w * np.abs(ls - lt))) / ws
        log_err /= len(specs)
        mag_err = np.mean([float(np.linalg.norm(np.sqrt(s) - m)) / max(nm, 1e-30)
                           for s, m, nm in zip(specs, self.mags, self.mag_norms)])
        fine = specs[len(RESOLUTIONS)]
        frame_w = self.fine_energy + fine.sum(axis=1)
        dist = np.sum(np.abs(_cdf(fine) - self.fine_cdf)[:, :-1] * FINE_STEP_OCT, axis=1)
        pitch_err = float(np.sum(dist * frame_w) / max(frame_w.sum(), 1e-30)) * self.low_share
        hl = 10.0 * np.log10(high + self.high_floor)
        w = np.clip((np.maximum(self.high_log, hl) - self.high_ref) / FLOOR_DB, 0.0, 1.0)
        high_err = float(np.sum(w * np.abs(hl - self.high_log)) / max(w.sum(), 1e-9)) * self.high_share
        tonal_err = abs(tonality(summed) - self.tonality) * self.high_share
        # Octave balance: every octave counts the same however quiet it is next to the
        # loudest one, so a body 25 dB under the sizzle of a cymbal is not cut away;
        # frames weighted by loudness from 50 dB under the peak, as above
        octaves, _ = self._octave_db(specs[0], self.octave_floor)
        level = np.maximum(self.octaves, octaves).max(axis=1)
        w = np.clip((level - (self.octaves.max() - FLOOR_DB)) / FLOOR_DB, 0.0, 1.0)
        # each frame's balance: its octaves against its loudest one, down to 60 dB under it,
        # so a quiet band in the tail counts as much as at the strike
        rel = lambda o: np.maximum(o - o.max(axis=1, keepdims=True), -BALANCE_RANGE)
        diff = rel(octaves) - rel(self.octaves)
        balance_err = float(np.sum(w * np.abs(diff).mean(axis=1)) / max(w.sum(), 1e-9))
        partial_err = 0.0
        if self.partial_bins is not None:
            b = self.partial_bins
            got = np.maximum(np.maximum(summed[b - 1], summed[b]), summed[b + 1])
            partial_err = float(np.sum(self.partial_w * np.abs(10.0 * np.log10(got + self.partial_floor)
                                                                - self.partial_db)) / self.partial_w.sum())
        if parts:
            return log_err, mag_err, pitch_err, high_err, tonal_err, balance_err, partial_err
        return (log_err + MAG_WEIGHT * mag_err + PITCH_WEIGHT * pitch_err
                + HIGH_WEIGHT * high_err + TONAL_WEIGHT * tonal_err + BALANCE_WEIGHT * balance_err
                + PARTIAL_WEIGHT * partial_err)

    def score(self, p: VoiceParams) -> float:
        return self.loss(render(p, self.n))


# --------------------------------------------------------------------------- starts
def categories_for(path: pathlib.Path, root: pathlib.Path) -> list:
    text = ' '.join(path.relative_to(root).with_suffix('').parts).lower().replace('_', ' ')
    for key, cats in CATEGORIES:
        if key in text.split() or (len(key) > 3 and key in text):
            return cats
    return []


def load_library(library: pathlib.Path, cats: list, size: int, seed: int) -> list:
    """Up to `size` random patches of the categories, as VoiceParams."""
    if not library.is_dir():
        return []
    dirs = [library / c for c in cats] if cats else [d for d in library.iterdir() if d.is_dir()]
    files = [e.path for d in dirs if d.is_dir() for e in os.scandir(d) if e.name.endswith('.mtdrum')]
    random.Random(seed).shuffle(files)
    parser, out = DrumPatchParser(), []
    for f in files[:size]:
        try:
            out.append(neutral(params_from_patch(parser.parse_file(f))))
        except Exception:
            continue
    return out


def analyse(x: np.ndarray) -> dict:
    """Partials (Hz, loudest first), decay time to -60 dB (ms) and spectral centroid (Hz)."""
    seg = x[:max(int(0.3 * SR), 2048)]
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 1 << 16))
    hz = np.fft.rfftfreq(1 << 16, 1.0 / SR)
    db = 20.0 * np.log10(spec / spec.max() + 1e-9)
    band = (hz >= 30.0) & (hz <= 5000.0)
    peaks, props = scipy.signal.find_peaks(np.where(band, db, -200.0), height=-30.0,
                                           prominence=12.0, distance=int(20.0 / hz[1]))
    partials = [float(hz[i]) for i in peaks[np.argsort(-props['peak_heights'])][:3]]

    hop = int(0.005 * SR)
    frames = np.lib.stride_tricks.sliding_window_view(np.concatenate([x, np.zeros(hop)]), hop)[::hop]
    env = 10.0 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)
    top = int(np.argmax(env))
    below = np.where(env[top:] < env[top] - 30.0)[0]
    t30 = (below[0] if len(below) else len(env) - top) * hop / SR
    power = spec ** 2
    centroid = float(np.sum(hz * power) / np.sum(power))
    return {'partials': partials, 'decay_ms': min(max(t30 * 2000.0, 10.0), 10000.0),
            'centroid': min(max(centroid, 100.0), 16000.0)}


def seed_starts(x: np.ndarray) -> list:
    """Patches built on the partials, decay and brightness found in the sample."""
    a = analyse(x)
    dcy, centroid, partials = a['decay_ms'], a['centroid'], a['partials']
    base = dict(osc_attack_ms=0.0, osc_decay_ms=dcy, mod_mode=MOD_DECAY, mod_rate=30.0,
                mod_amount=2.0, noise_env=NENV_EXP, noise_attack_ms=0.0,
                noise_decay_ms=dcy * 0.5, distortion=0.0, eq_freq=1000.0, eq_gain_db=0.0)
    out = []
    for i, f0 in enumerate(partials):
        other = partials[1] if i == 0 and len(partials) > 1 else (partials[0] if i else centroid)
        for wave in (WAVE_SINE, WAVE_TRI, WAVE_SAW):
            # second partial on narrow band-pass noise, or a bright noise click
            out.append(VoiceParams(**base, wave=wave, osc_freq=f0, noise_filter=FILT_BP,
                                   noise_freq=other, noise_q=15.0, osc_mix=0.6))
            out.append(VoiceParams(**base, wave=wave, osc_freq=f0, noise_filter=FILT_HP,
                                   noise_freq=centroid, noise_q=0.7, osc_mix=0.85))
    for filt, f, q in ((FILT_BP, centroid, 1.0), (FILT_HP, centroid * 0.5, 0.7),
                       (FILT_LP, centroid * 2.0, 0.7)):
        out.append(VoiceParams(**{**base, 'noise_decay_ms': dcy}, wave=WAVE_SINE,
                               osc_freq=partials[0] if partials else 200.0,
                               noise_filter=filt, noise_freq=min(f, 20000.0), noise_q=q, osc_mix=0.1))
    return out


def _partials(x: np.ndarray) -> tuple:
    """(Hz, relative power, frequency resolution in Hz) of the steady partials from 400 Hz
    to 16 kHz in the first 1.5 s (at least 2048 samples)."""
    seg = x[int(0.005 * SR):int(1.5 * SR)]
    n = min(65536, len(seg))
    if n < 2048:
        return np.zeros(0), np.zeros(0), 0.0
    hz, pw = scipy.signal.welch(seg, SR, nperseg=n, noverlap=n // 2, nfft=2 * 65536)
    db = 10.0 * np.log10(pw + pw.max() * 1e-14 + 1e-30)
    peaks, _ = scipy.signal.find_peaks(db - scipy.ndimage.median_filter(db, 801, mode='nearest'),
                                       height=10.0, distance=8)
    peaks = peaks[(hz[peaks] > 400.0) & (hz[peaks] < 16000.0) & (peaks > 0) & (peaks < len(db) - 1)]
    a, b, c = db[peaks - 1], db[peaks], db[peaks + 1]
    shift = 0.5 * (a - c) / np.where(a - 2 * b + c == 0, -1e-9, a - 2 * b + c)
    return hz[peaks] + shift * hz[1], pw[peaks] / pw[peaks].max(), SR / n


EVEN_WEIGHT = 0.5       # a square of 48 % duty has weak even harmonics too


def _odd_matches(freqs: np.ndarray, tol: np.ndarray, f0: float) -> tuple:
    """(mask of the partials a harmonic of f0 explains, their harmonic numbers)."""
    k = np.round(freqs / f0)
    return (k >= 1) & (np.abs(freqs - k * f0) < tol), k


def _bank_share(freqs: np.ndarray, power: np.ndarray, tol: np.ndarray, bank) -> float:
    """Share of the partial power the odd harmonics of a bank explain, beyond chance."""
    hit = np.zeros(len(freqs), bool)
    for f0 in bank:
        hit |= _odd_matches(freqs, tol, f0)[0]
    chance = power @ np.minimum(sum(2.0 * tol / f0 for f0 in bank), 1.0)
    return float((power[hit].sum() - chance) / power.sum())


def metal_fundamentals(x: np.ndarray, banks=()) -> tuple:
    """The six square-wave fundamentals (Hz, ascending) whose odd harmonics explain the
    steady partials above 2 kHz of a metallic sample, or () for any other sample.

    Greedy: the fundamental from 100 Hz to 2 kHz explaining the most partial power (its
    even harmonics at half weight) beyond what its comb would catch by chance (a share
    tolerance / f0 of every partial per harmonic parity), its partials removed, six times. A fundamental whose matches mostly sit on multiples of 3
    (or 5) of its harmonics is the third (fifth) harmonic of the real one, kept instead.
    A sample too short to resolve its partials finely takes the best of ``banks``, the
    fundamentals found in other samples of the same machine (they share one bank)."""
    if tonality(spectra(x)[1]) < 0.3:
        return ()
    freqs, power, resolution = _partials(x)
    if len(freqs) < 12:
        return ()
    tol = 2.0 + 3e-4 * freqs + 0.5 * resolution
    found = []
    if resolution <= BANK_RESOLUTION:
        grid = np.arange(100.0, 2000.0, 0.05)
        left = np.ones(len(freqs), bool)
        for _ in range(6):
            k = np.round(freqs[None, :] / grid[:, None])
            hit = (k >= 1) & (np.abs(freqs - k * grid[:, None]) < tol) & left
            weight = np.where(k % 2 == 1, 1.0, EVEN_WEIGHT)
            chance = (power * left) @ np.minimum((1.0 + EVEN_WEIGHT) * tol[:, None] / grid[None, :], 1.0)
            f0 = float(grid[np.argmax((hit * weight) @ power - chance)])
            ok, k = _odd_matches(freqs, tol, f0)
            ok &= left
            for m in (3, 5):
                if power[ok & (k % m == 0)].sum() > 0.7 * power[ok].sum():
                    f0 *= m
                    ok, k = _odd_matches(freqs, tol, f0)
                    ok &= left
            f0 = float(np.sum(k[ok] * freqs[ok]) / max(np.sum(k[ok] ** 2), 1e-9)) if ok.any() else f0
            found.append(min(max(f0, 20.0), 2000.0))
            left &= ~ok
    if found:
        # a greedy pick can be a submultiple of the real fundamental (2.5 x 114.7 Hz = 286.8 Hz):
        # keep any multiple or submultiple that explains the partials better as a whole bank
        share = _bank_share(freqs, power, tol, found)
        for i in range(len(found)):
            for m in (2.0, 2.5, 3.0, 4.0, 5.0, 0.5, 0.4, 1.0 / 3.0):
                trial = found[:i] + [found[i] * m] + found[i + 1:]
                if 100.0 <= trial[i] <= 2000.0 and _bank_share(freqs, power, tol, trial) > share + 0.01:
                    found, share = trial, _bank_share(freqs, power, tol, trial)
    # A bank found in this sample must explain half its partials beyond chance: a real
    # drum machine bank explains 0.75-0.85, while the modes of an acoustic cymbal leave
    # about 0.3 to a bank of low fundamentals. A bank of the folder, already found in a
    # long sample of the machine, only needs to explain a quarter of a short hit's.
    candidates = [(tuple(sorted(found)), OWN_BANK_SHARE)] if found else []
    candidates += [(tuple(b), BORROWED_BANK_SHARE) for b in banks]
    best = max(((_bank_share(freqs, power, tol, b) - need, b) for b, need in candidates),
               default=(-1.0, ()))
    return best[1] if best[0] >= 0.0 else ()


def _shared(a, b) -> int:
    """How many frequencies of bank a are within 0.5 % of one of bank b."""
    return sum(any(abs(f - g) < 0.005 * f for g in b) for f in a)


def _consensus(banks: list) -> tuple:
    """The six frequencies found in the most banks of one machine (median of each)."""
    clusters = []
    for f in sorted(f for b in banks for f in b):
        if clusters and f - clusters[-1][-1] < 0.005 * f:
            clusters[-1].append(f)
        else:
            clusters.append([f])
    top = sorted(clusters, key=lambda c: -len(c))[:6]
    return tuple(sorted(float(np.median(c)) for c in top))


def metal_banks(paths, max_seconds: float) -> list:
    """The banks found in the samples long enough to resolve them, one per machine unit:
    the banks sharing four frequencies or more are merged into their consensus, which
    drops an oscillator a weak sample got wrong."""
    groups = []
    for path in paths:
        try:
            x = load_target(str(path), max_seconds)
        except Exception:
            continue
        if len(x) < BANK_SECONDS * SR:
            continue
        bank = metal_fundamentals(x)
        if not bank:
            continue
        group = next((g for g in groups if _shared(bank, _consensus(g)) >= 4), None)
        if group is None:
            groups.append([bank])
        else:
            group.append(bank)
    return [_consensus(g) for g in groups]


SQUARE_DUTY = 0.48      # of the metal oscillators (voice_kernel.METAL_DUTY)


def metal_response(x: np.ndarray, freqs: tuple) -> np.ndarray:
    """(points, 2) Hz and dB of the filters a metal sample went through: each partial of the
    first 0.4 s over the level its square wave gives that harmonic, |sin(pi k duty)| / k."""
    seg = x[int(0.005 * SR):int(0.4 * SR)]
    n = min(16384, len(seg))
    if n < 2048:
        return np.zeros((0, 2))
    hz, pw = scipy.signal.welch(seg, SR, nperseg=n, noverlap=n // 2, nfft=65536)
    db = 10.0 * np.log10(pw + pw.max() * 1e-14)
    peaks, _ = scipy.signal.find_peaks(db - scipy.ndimage.median_filter(db, 201, mode='nearest'),
                                       height=8.0, distance=3)
    points = []
    for i in peaks[(hz[peaks] > 150.0) & (hz[peaks] < 18000.0)]:
        f, best = hz[i], 0.0
        for f0 in freqs:
            k = round(f / f0)
            if k >= 1 and abs(f - k * f0) < 3.0 + 4e-4 * f + SR / n:
                best = max(best, abs(math.sin(math.pi * k * SQUARE_DUTY)) / k)
        if best > 0.02:
            points.append((f, db[i] - 20.0 * math.log10(best)))
    return np.array(points).reshape(-1, 2)


def _bands_db(theta, hz):
    """dB response of two band-passes (level ratio) in parallel, then a high-pass, at hz."""
    from pythonic.voice import _rbj
    f1, q1, f2, q2, ratio, fhp = np.exp(theta[:6])
    w = 2.0 * np.pi * hz / SR
    resp = lambda ba: scipy.signal.freqz(ba[0], ba[1], worN=w)[1]
    h = (resp(_rbj(f1 / SR, q1, 'bp')) + ratio * resp(_rbj(f2 / SR, q2, 'bp'))) \
        * resp(_rbj(fhp / SR, 0.70710678, 'hp'))
    return 20.0 * np.log10(np.abs(h) + 1e-9) + theta[6]


def metal_filter(x: np.ndarray, freqs: tuple):
    """(band 1 freq, Q, band 2 freq, Q, band 2 / band 1 level, high-pass freq) whose response
    matches metal_response(), or None: a least-squares fit in dB from a few starts."""
    import scipy.optimize
    points = metal_response(x, freqs)
    if len(points) < 20:
        return None
    hz, db = points[:, 0], points[:, 1]
    # the loudest partials carry the shape: weigh each by its level, down to 40 dB under the top
    weight = np.clip((db - db.max() + 40.0) / 40.0, 0.05, 1.0)
    peak = float(hz[np.argmax(db)])
    lo = np.log([500.0, 0.5, 500.0, 0.5, 1e-3, 20.0])
    hi = np.log([16000.0, 20.0, 16000.0, 20.0, 10.0, 16000.0])
    best = None
    for f2, q1, q2, ratio, fhp in ((peak * 1.6, 3.0, 0.7, 0.3, 2000.0), (peak / 2.0, 3.0, 1.0, 0.3, 1000.0),
                                   (peak * 1.3, 1.5, 1.5, 1.0, 3000.0), (peak / 1.5, 2.0, 2.0, 0.5, 500.0)):
        theta0 = np.r_[np.clip(np.log([peak, q1, min(f2, 15000.0), q2, ratio, fhp]), lo, hi), 0.0]
        theta0[6] = float(np.median(db - _bands_db(theta0, hz)))
        fit = scipy.optimize.least_squares(
            lambda t: weight * (_bands_db(t, hz) - db), theta0,
            bounds=(np.r_[lo, -np.inf], np.r_[hi, np.inf]), loss='soft_l1', f_scale=3.0)
        if best is None or fit.cost < best.cost:
            best = fit
    return tuple(float(v) for v in np.exp(best.x[:6]))


def metal_starts(p: VoiceParams, x: np.ndarray, freqs: tuple) -> list:
    """p with the metal section on, on the fundamentals. First the filters measured on the
    sample's partials (metal_filter), with both bands decaying together, the second one
    longer or shorter, and a linear decay that holds then falls away (open hats), also
    without noise (the oscillator a 10 ms tick at 10 kHz); then, should the measure fail,
    both band-passes around the partials' centre of power in two spreads."""
    dcy = analyse(x)['decay_ms']
    base = {**p.__dict__, **dict(zip(METAL_FREQS, freqs)), 'osc_mix': min(p.osc_mix, 0.3),
            **METAL_FIXED}
    quiet = {'osc_mix': 1.0, 'wave': WAVE_SINE, 'osc_freq': 10000.0, 'osc_decay_ms': 10.0}
    measured = metal_filter(x, freqs)
    out = []
    if measured is not None:
        f1, q1, f2, q2, ratio, fhp = measured
        lv1, lv2 = (0.7, 0.7 * ratio) if ratio <= 1.0 else (0.7 / ratio, 0.7)
        bands = {'metal_band1_freq': f1, 'metal_band1_q': q1, 'metal_band2_freq': f2,
                 'metal_band2_q': q2, 'metal_hp_freq': fhp, 'metal_band1_level': min(lv1, 1.0),
                 'metal_band2_level': min(lv2, 1.0)}
        for d1, d2, curve, extra in ((dcy, dcy, 0.0, {}), (dcy, dcy * 0.3, 0.0, {}),
                                     (dcy * 0.3, dcy, 0.0, {}), (dcy * 0.8, dcy * 0.8, 1.0, {}),
                                     (dcy, dcy, 0.0, quiet), (dcy * 0.8, dcy * 0.8, 1.0, quiet)):
            out.append(VoiceParams(**{**base, **bands, 'metal_band1_decay_ms': d1,
                                      'metal_band2_decay_ms': d2, 'metal_curve': curve, **extra}))
        return out
    hz, power, _ = _partials(x)
    high = hz > 2000.0
    centre = float(np.exp(np.sum(np.log(hz[high]) * power[high]) / power[high].sum())) \
        if high.any() else 7000.0
    for spread, q in ((1.25, 3.0), (1.8, 1.5)):
        out.append(VoiceParams(**{**base, 'metal_band1_freq': centre / spread,
                                  'metal_band2_freq': min(centre * spread, 16000.0),
                                  'metal_band1_decay_ms': dcy * 0.5, 'metal_band2_decay_ms': dcy,
                                  'metal_band1_level': 0.5, 'metal_band2_level': 0.5,
                                  'metal_band1_q': q, 'metal_band2_q': q, 'metal_hp_freq': centre / 3.0}))
    return out


BANK_RESOLUTION = 3.0   # Hz: a sample analysed more coarsely takes a bank from another one
BANK_SECONDS = 0.5      # samples at least this long give the banks of a folder
OWN_BANK_SHARE = 0.5    # partial power a sample's own bank must explain, beyond chance
BORROWED_BANK_SHARE = 0.25  # the same for a bank found in another sample of the folder

_prior = {}


def prior_starts(x: np.ndarray, path: str) -> list:
    """Patches predicted by the fit_prior.py model (none without the model)."""
    if not path or not os.path.exists(path):
        return []
    if path not in _prior:
        import fit_prior
        _prior[path] = fit_prior.load(path)
    import fit_prior
    return fit_prior.predict(_prior[path], x)


# --------------------------------------------------------------------------- search
def cma_run(target: Target, x0, sw: tuple, sigma: float, evals: int, seed: int, popsize=None):
    """(best score, best vector, evaluations) of one CMA-ES run with the switches fixed."""
    opts = {'bounds': [0.0, 1.0], 'maxfevals': max(evals, 1), 'seed': seed, 'verbose': -9,
            'tolfun': 1e-4, 'tolx': 1e-4}
    if popsize:
        opts['popsize'] = popsize
    es = cma.CMAEvolutionStrategy(np.asarray(x0, float), sigma, opts)
    while not es.stop():
        xs = es.ask()
        es.tell(xs, [target.score(decode(x, sw, target.metal_freqs)) for x in xs])
    if es.result.xbest is None:
        return float('inf'), np.asarray(x0, float), es.result.evaluations
    return float(es.result.fbest), np.array(es.result.xbest), es.result.evaluations


def long_run(target: Target, x0, sw: tuple, job: dict, seed: int):
    """CMA-ES restarted from its best point with a doubled population while budget remains."""
    best = (float('inf'), np.asarray(x0, float))
    budget, popsize, sigma = job['evals'], None, job['sigma']
    for r in range(job['restarts'] + 1):
        share = budget if r == job['restarts'] else int(budget * 0.6)
        f, x, used = cma_run(target, best[1], sw, sigma, share, seed + 100 * r, popsize)
        if f < best[0]:
            best = (f, x)
        budget -= used
        if budget < 200:
            break
        popsize = (popsize or 4 + int(3 * math.log(len(best[1])))) * 2
        sigma = 0.2
    return best


def pick(cands: list, n: int) -> list:
    """The n best candidates with distinct switch settings."""
    out, seen = [], set()
    for c in sorted(cands, key=lambda c: c[0]):
        if switches_of(c[2]) not in seen:
            seen.add(switches_of(c[2]))
            out.append(c)
            if len(out) == n:
                break
    return out


def fit_one(job: dict) -> dict:
    t0 = time.time()
    x = load_target(job['path'], job['max_seconds'])
    target = Target(x)
    # Only a sample whose partials are those of six squares gets the metal section: on an
    # acoustic cymbal, six squares sound like a drum machine, however well they score
    target.use_metal(metal_fundamentals(x, job.get('banks', ())))
    metal = bool(target.metal_freqs)
    seeds = itertools.count(job['seed'] + 1)
    code = lambda p: encode(p, metal)
    back = lambda u, sw: decode(u, sw, target.metal_freqs)

    # 1. starts, scored once
    sources = {'library': job['library'], 'partials': seed_starts(x),
               'prior': [neutral(p) for p in prior_starts(x, job['prior'])]}
    if metal:
        sources = {src: [m for p in ps for m in metal_starts(p, x, target.metal_freqs)]
                   for src, ps in sources.items()}
    cands = [(target.score(p), src, p) for src, ps in sources.items() for p in ps]
    if not cands:
        default = metal_starts(VoiceParams(), x, target.metal_freqs)[0] if metal else VoiceParams()
        cands = [(target.score(default), 'default', default)]
    start_loss = min(c[0] for c in cands)

    # 2. screening: short runs from the best starts of each kind
    others = max(job['screen'] // 2, 1) if job['library'] else job['screen']
    quota = {'library': job['screen'], 'partials': others, 'prior': others, 'default': 1}
    screened = []
    for src in quota:
        for loss, _, p in pick([c for c in cands if c[1] == src], quota[src]):
            f, u, _ = cma_run(target, code(p), switches_of(p), job['sigma'] * 0.7,
                              job['screen_evals'], next(seeds))
            screened.append((f, src, back(u, switches_of(p))))

    # 3. long runs from the best screened starts
    finals = []
    for _, src, p in pick(screened, job['starts']):
        f, u = long_run(target, code(p), switches_of(p), job, next(seeds))
        finals.append((f, src, back(u, switches_of(p))))
    best_loss, best_src, best = min(finals, key=lambda c: c[0])

    # 4. single switch changes
    tried, best_sw = [], switches_of(best)
    for k in range(len(SWITCHES)):
        for v in range(SWITCH_CHOICES):
            sw = list(best_sw)
            if sw[k] == v:
                continue
            sw[k] = v
            f, u, _ = cma_run(target, code(best), tuple(sw), job['sigma'] * 0.7,
                              job['screen_evals'], next(seeds))
            tried.append((f, tuple(sw), u))
    f, sw, u = min(tried, key=lambda c: c[0])
    switched = ''
    if f < best_loss:
        f, u = long_run(target, u, sw, {**job, 'evals': job['evals'] // 2}, next(seeds))
        if f < best_loss:
            best_loss, best = f, back(u, sw)
            switched = '+'.join(name for name, a, b in zip(SWITCHES, sw, best_sw) if a != b)

    # Level: peak at --peak-db with velocity 127
    peak = np.abs(render(best, target.n)).max()
    level = job['peak_db'] - 20.0 * math.log10(max(peak, 1e-9))
    best = VoiceParams(**{**best.__dict__, 'level_db': min(max(level, -60.0), 10.0)})
    return {'path': job['path'], 'name': job['name'], 'cats': job['cats'], 'params': best,
            'start_loss': start_loss, 'loss': best_loss, 'source': best_src + (' ' + switched if switched else ''),
            'seconds': time.time() - t0, 'target': target.x, 'n': target.n}


def patch_dict(p: VoiceParams, name: str, choke: bool) -> dict:
    """Display values of p, keyed and ordered as in a .mtdrum file."""
    return {
        'Name': name,
        'Modified': False,
        'OscWave': ['Sine', 'Triangle', 'Saw'][p.wave],
        'OscFreq': p.osc_freq,
        'OscAtk': p.osc_attack_ms,
        'OscDcy': p.osc_decay_ms,
        'ModMode': ['Decay', 'Sine', 'Noise'][p.mod_mode],
        'ModRate': p.mod_rate,
        'ModAmt': p.mod_amount,
        'NFilMod': ['LP', 'BP', 'HP'][p.noise_filter],
        'NFilFrq': p.noise_freq,
        'NFilQ': p.noise_q,
        'NStereo': p.noise_stereo,
        'NEnvMod': ['Exp', 'Linear', 'Mod'][p.noise_env],
        'NEnvAtk': p.noise_attack_ms,
        'NEnvDcy': p.noise_decay_ms,
        'Mix': (p.osc_mix * 100.0, (1.0 - p.osc_mix) * 100.0),
        'DistAmt': p.distortion * 100.0,
        'EQFreq': p.eq_freq,
        'EQGain': p.eq_gain_db,
        'Level': p.level_db,
        'Pan': 0.0,
        'Output': 'A',
        'Choke': choke,
        'OscVel': 25.0,
        'NVel': 25.0,
        'ModVel': 0.0,
        **(DrumPatchWriter.metal_keys({**p.__dict__, 'metal_vel': 0.25}) if metal_on(p) else {}),
    }


REPORT_FIELDS = ['sample', 'patch', 'categories', 'start_score', 'score', 'won_from', 'seconds']


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('samples', help='folder of one-shot WAVs (searched recursively)')
    ap.add_argument('out', help='output folder for the .mtdrum files')
    ap.add_argument('--only', default='', help='fit only the samples whose path contains this text')
    ap.add_argument('--library-dir', default='',
                    help='optional folder of .mtdrum patches in category subfolders '
                         '(bd, sd, ch, oh, clap, tom, perc, cowbell, cy, shaker) to start from too')
    ap.add_argument('--library', type=int, default=1000, help='library patches scored per sample')
    ap.add_argument('--prior', default=str(DEFAULT_PRIOR),
                    help='model trained by tools/fit_prior.py (skipped if missing; "" to disable)')
    ap.add_argument('--screen', type=int, default=6, help='library starts screened (half as many of the others)')
    ap.add_argument('--screen-evals', type=int, default=400, help='evaluations per screening or switch run')
    ap.add_argument('--starts', type=int, default=3, help='long CMA-ES runs from the best screened starts')
    ap.add_argument('--evals', type=int, default=3000, help='evaluations per long run, restarts included')
    ap.add_argument('--restarts', type=int, default=1, help='IPOP restarts within a long run')
    ap.add_argument('--sigma', type=float, default=0.12, help='initial CMA-ES step in unit space')
    ap.add_argument('--max-seconds', type=float, default=2.5, help='length of each sample that is matched')
    ap.add_argument('--peak-db', type=float, default=-10.0, help='peak level of each patch at velocity 127')
    ap.add_argument('--jobs', type=int, default=os.cpu_count() or 1, help='samples fitted in parallel')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--wav', action='store_true', help='also write <name>.fit.wav (sample left, patch right)')
    args = ap.parse_args()

    root = pathlib.Path(args.samples)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in root.rglob('*') if p.suffix.lower() in ('.wav', '.aif', '.aiff', '.flac')
                   and args.only.lower() in str(p).lower())
    if not paths:
        sys.exit(f'no samples under {root}')
    if args.prior and os.path.exists(args.prior) and importlib.util.find_spec('torch') is None:
        print('PyTorch is not installed (pip install -e ".[ml]"): the prior model is not used')
        args.prior = ''
    if args.prior and not os.path.exists(args.prior):
        print(f'no prior model at {args.prior}: starting from the partials'
              + (' and the library' if args.library_dir else '') + ' only')

    libraries, jobs = {}, []
    for path in paths:
        cats = categories_for(path, root)
        key = tuple(cats)
        if key not in libraries:
            libraries[key] = (load_library(pathlib.Path(args.library_dir), cats, args.library, args.seed)
                              if args.library_dir else [])
        jobs.append({'path': str(path), 'name': path.stem, 'cats': cats, 'library': libraries[key],
                     'prior': args.prior, 'screen': args.screen, 'screen_evals': args.screen_evals,
                     'starts': args.starts, 'evals': args.evals, 'restarts': args.restarts,
                     'sigma': args.sigma, 'max_seconds': args.max_seconds, 'peak_db': args.peak_db,
                     'seed': args.seed})
    banks = metal_banks(paths, args.max_seconds)
    for job in jobs:
        job['banks'] = banks
    print(f'{len(jobs)} samples, {args.jobs} jobs'
          + ''.join(f'\nmetal bank: {", ".join(f"{f:.1f}" for f in b)} Hz' for b in banks), flush=True)

    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(fit_one, job): job for job in jobs}
        for fut in concurrent.futures.as_completed(futures):
            job = futures[fut]
            try:
                r = fut.result()
            except Exception as exc:
                print(f'FAILED {job["name"]}: {exc!r}', flush=True)
                continue
            # the output folder mirrors the sample folder's tree
            folder = out / pathlib.Path(r['path']).relative_to(root).parent
            folder.mkdir(parents=True, exist_ok=True)
            dst = folder / f'{r["name"]}.mtdrum'
            patch = patch_dict(r['params'], r['name'], bool(set(r['cats']) & CHOKE_CATEGORIES))
            dst.write_text(DrumPatchWriter.format_patch(patch), encoding='utf-8')
            if args.wav:
                y = render(params_from_patch(DrumPatchParser().parse_file(str(dst))), r['n'])
                xt = np.concatenate([r['target'], np.zeros(r['n'] - len(r['target']))])
                y = y * (np.abs(xt).max() / max(np.abs(y).max(), 1e-9))
                sf.write(folder / f'{r["name"]}.fit.wav', np.stack([xt, y], 1), SR, subtype='PCM_24')
            rows.append({'sample': os.path.relpath(r['path'], root), 'patch': os.path.relpath(dst, out),
                         'categories': '+'.join(r['cats']) or 'all',
                         'start_score': round(r['start_loss'], 3), 'score': round(r['loss'], 3),
                         'won_from': r['source'], 'seconds': round(r['seconds'], 1)})
            print(f'{r["name"]:32s} {rows[-1]["categories"]:12s} start {r["start_loss"]:6.2f}'
                  f' -> fit {r["loss"]:6.2f}  from {r["source"]:20s} ({r["seconds"]:.0f} s)', flush=True)

    # keep the rows of samples this run did not fit (--only)
    report = out / 'fit_report.csv'
    done = {row['sample'] for row in rows}
    if report.exists():
        with open(report, newline='', encoding='utf-8') as f:
            old = list(csv.DictReader(f))
        if old and list(old[0]) == REPORT_FIELDS:
            rows += [row for row in old if row['sample'] not in done]
    rows.sort(key=lambda row: row['sample'])
    with open(report, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f'wrote {len(rows)} patches to {out}')


if __name__ == '__main__':
    main()
