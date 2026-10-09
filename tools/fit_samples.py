"""
Fit a .mtdrum drum patch to each one-shot WAV in a folder.

    venv/bin/python tools/fit_samples.py <samples dir> <output dir> [--wav]

Each sample is matched in four steps:

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
  toward the right pitch.

Samples are mixed to mono and trimmed to their onset. The output folder gets
one <sample name>.mtdrum per sample and fit_report.csv; --wav adds
<sample name>.fit.wav, the patch rendered next to the trimmed sample (left:
sample, right: patch) for listening.
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
import scipy.signal
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pythonic.preset_manager import DrumPatchParser, DrumPatchWriter  # noqa: E402
from pythonic.voice import (DrumVoice, FILT_BP, FILT_HP, FILT_LP, MOD_DECAY,  # noqa: E402
                            NENV_EXP, VoiceParams, WAVE_SAW, WAVE_SINE, WAVE_TRI,
                            attack_norm, params_from_patch)

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
SWITCHES = ('wave', 'mod_mode', 'noise_filter', 'noise_env')
SWITCH_CHOICES = 3


def mod_range(mod_mode: int) -> float:
    return 96.0 if mod_mode == MOD_DECAY else 48.0


def encode(p: VoiceParams) -> np.ndarray:
    """Unit-cube vector of the continuous controls of p."""
    u = []
    for name, lo, hi, scale in CONTINUOUS:
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


def decode(u: np.ndarray, switches: tuple) -> VoiceParams:
    """VoiceParams from a unit-cube vector and the (wave, mod mode, filter, env) switches."""
    fields = dict(zip(SWITCHES, switches))
    for x, (name, lo, hi, scale) in zip(np.clip(u, 0.0, 1.0), CONTINUOUS):
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

FLOOR_DB = 50.0        # weights and log floors reach zero this far under the peak
MAG_WEIGHT = 10.0      # score per unit of relative magnitude error
PITCH_WEIGHT = 12.0    # score per octave of pitch distance below 2 kHz


def spectra(x: np.ndarray) -> list:
    """Mel spectrograms, the fine low spectrum and the frame energy envelope."""
    out = [stft_power(x, n_fft, hop, win) @ fb for n_fft, hop, win, fb in RESOLUTIONS]
    out.append(stft_power(x, FINE_FFT, FINE_HOP, FINE_WIN)[:, FINE_BINS])
    y = np.concatenate([np.zeros(ENV_WIN // 2, np.float32), x.astype(np.float32),
                        np.zeros(ENV_WIN, np.float32)])
    out.append(np.mean(np.lib.stride_tricks.sliding_window_view(y, ENV_WIN)[::ENV_HOP] ** 2,
                       axis=1)[:, None])
    return out


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
        self.specs = spectra(padded)
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

    def loss(self, y: np.ndarray, parts: bool = False):
        """Score of render y; with parts, (spectra dB, magnitude, pitch) before weighting."""
        e = float(np.sum(y ** 2))
        if not math.isfinite(e) or e < 1e-20:
            return (1e3, 1e3, 1e3) if parts else 1e3
        specs = spectra(y * math.sqrt(self.energy / e))
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
        if parts:
            return log_err, mag_err, pitch_err
        return log_err + MAG_WEIGHT * mag_err + PITCH_WEIGHT * pitch_err

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
        es.tell(xs, [target.score(decode(x, sw)) for x in xs])
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
        popsize = (popsize or 4 + int(3 * math.log(len(CONTINUOUS)))) * 2
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
    seeds = itertools.count(job['seed'] + 1)

    # 1. starts, scored once
    sources = {'library': job['library'], 'partials': seed_starts(x),
               'prior': [neutral(p) for p in prior_starts(x, job['prior'])]}
    cands = [(target.score(p), src, p) for src, ps in sources.items() for p in ps]
    if not cands:
        cands = [(target.score(VoiceParams()), 'default', VoiceParams())]
    start_loss = min(c[0] for c in cands)

    # 2. screening: short runs from the best starts of each kind
    others = max(job['screen'] // 2, 1) if job['library'] else job['screen']
    quota = {'library': job['screen'], 'partials': others, 'prior': others, 'default': 1}
    screened = []
    for src in quota:
        for loss, _, p in pick([c for c in cands if c[1] == src], quota[src]):
            f, u, _ = cma_run(target, encode(p), switches_of(p), job['sigma'] * 0.7,
                              job['screen_evals'], next(seeds))
            screened.append((f, src, decode(u, switches_of(p))))

    # 3. long runs from the best screened starts
    finals = []
    for _, src, p in pick(screened, job['starts']):
        f, u = long_run(target, encode(p), switches_of(p), job, next(seeds))
        finals.append((f, src, decode(u, switches_of(p))))
    best_loss, best_src, best = min(finals, key=lambda c: c[0])

    # 4. single switch changes
    tried, best_sw = [], switches_of(best)
    for k in range(len(SWITCHES)):
        for v in range(SWITCH_CHOICES):
            sw = list(best_sw)
            if sw[k] == v:
                continue
            sw[k] = v
            f, u, _ = cma_run(target, encode(best), tuple(sw), job['sigma'] * 0.7,
                              job['screen_evals'], next(seeds))
            tried.append((f, tuple(sw), u))
    f, sw, u = min(tried, key=lambda c: c[0])
    switched = ''
    if f < best_loss:
        f, u = long_run(target, u, sw, {**job, 'evals': job['evals'] // 2}, next(seeds))
        if f < best_loss:
            best_loss, best = f, decode(u, sw)
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
    print(f'{len(jobs)} samples, {args.jobs} jobs', flush=True)

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
            dst = out / f'{r["name"]}.mtdrum'
            patch = patch_dict(r['params'], r['name'], bool(set(r['cats']) & CHOKE_CATEGORIES))
            dst.write_text(DrumPatchWriter.format_patch(patch), encoding='utf-8')
            if args.wav:
                y = render(params_from_patch(DrumPatchParser().parse_file(str(dst))), r['n'])
                xt = np.concatenate([r['target'], np.zeros(r['n'] - len(r['target']))])
                y = y * (np.abs(xt).max() / max(np.abs(y).max(), 1e-9))
                sf.write(out / f'{r["name"]}.fit.wav', np.stack([xt, y], 1), SR, subtype='PCM_24')
            rows.append({'sample': os.path.relpath(r['path'], root), 'patch': dst.name,
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
