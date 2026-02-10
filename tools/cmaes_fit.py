"""
CMA-ES based drum patch fitting against a target WAV.

Simple usage (all optimizations enabled by default):
  python tools/cmaes_fit.py --target path/to/808_kick.wav

Full options:
  python tools/cmaes_fit.py --target path/to/kick.wav --out fitted.mtdrum --name "My Kick"

Defaults enabled:
  - Initial waveform analysis for smart parameter estimation
  - Multi-stage fitting (coarse → mid → fine)
  - Grid search for discrete params (waveform, filter mode)
  - Multiple restarts per stage
  - 1.5x population size multiplier
  - CMA-ES internal stopping disabled (runs full evaluations)

Initial Analysis Phase:
  Before CMA-ES starts, the target waveform is analyzed to estimate:
  - Fundamental frequency (autocorrelation)
  - Decay time (envelope analysis)
  - Noise vs tonal ratio (spectral flatness)
  - Pitch sweep detection (kick/tom characteristic)
  - Waveform type (sine/triangle/saw)
  - Filter mode and frequency
  
  This gives CMA-ES a much better starting point than random initialization.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.io import wavfile
from scipy import signal
import cma

# Optional GPU acceleration via CuPy
try:
    import cupy as cp
    from cupyx.scipy import signal as cp_signal
    _CUPY_AVAILABLE = True
except ImportError:
    cp = None
    cp_signal = None
    _CUPY_AVAILABLE = False

# Global GPU state
_USE_GPU = False
_GPU_DEVICE = None


def _init_gpu(force_cpu: bool = False) -> bool:
    """Initialize GPU if available. Returns True if GPU is active."""
    global _USE_GPU, _GPU_DEVICE
    
    if force_cpu:
        _USE_GPU = False
        _GPU_DEVICE = None
        return False
    
    if not _CUPY_AVAILABLE:
        _USE_GPU = False
        _GPU_DEVICE = None
        return False
    
    try:
        # Test GPU availability
        _GPU_DEVICE = cp.cuda.Device(0)
        _ = cp.array([1.0])  # Test allocation
        _USE_GPU = True
        return True
    except Exception as e:
        print(f"[GPU] CuPy available but GPU init failed: {e}")
        _USE_GPU = False
        _GPU_DEVICE = None
        return False


def _to_gpu(arr: np.ndarray) -> "cp.ndarray":
    """Transfer numpy array to GPU."""
    return cp.asarray(arr)


def _to_cpu(arr) -> np.ndarray:
    """Transfer CuPy array back to CPU."""
    if _CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return arr

# Ensure project root is on sys.path for direct script execution
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pythonic.synthesizer import PythonicSynthesizer
from pythonic.preset_manager import DrumPatchWriter


@dataclass
class ParamSpec:
    name: str
    min_val: float
    max_val: float
    scale: str = "linear"  # "linear" or "log"
    discrete: Optional[int] = None  # number of discrete values (0..discrete-1)

    def clamp(self, value: float) -> float:
        return float(np.clip(value, self.min_val, self.max_val))


CORE_PARAMS: List[ParamSpec] = [
    ParamSpec("osc_frequency", 20.0, 2000.0, "log"),
    ParamSpec("osc_waveform", 0, 2, "linear", discrete=3),
    ParamSpec("pitch_mod_mode", 0, 2, "linear", discrete=3),
    ParamSpec("pitch_mod_amount", -48.0, 48.0, "linear"),
    ParamSpec("pitch_mod_rate", 1.0, 2000.0, "log"),
    ParamSpec("osc_attack", 0.0, 100.0, "linear"),
    ParamSpec("osc_decay", 1.0, 2000.0, "log"),
    ParamSpec("noise_filter_mode", 0, 2, "linear", discrete=3),
    ParamSpec("noise_filter_freq", 20.0, 20000.0, "log"),
    ParamSpec("noise_filter_q", 0.5, 100.0, "log"),
    ParamSpec("noise_envelope_mode", 0, 2, "linear", discrete=3),
    ParamSpec("noise_attack", 0.0, 100.0, "linear"),
    ParamSpec("noise_decay", 1.0, 2000.0, "log"),
    ParamSpec("osc_noise_mix", 0.0, 1.0, "linear"),
    ParamSpec("distortion", 0.0, 0.6, "linear"),
    ParamSpec("eq_frequency", 20.0, 20000.0, "log"),
    ParamSpec("eq_gain_db", -40.0, 40.0, "linear"),
    ParamSpec("level_db", -12.0, 6.0, "linear"),
]

REALLY_MAJOR_PARAMS: List[ParamSpec] = [
    ParamSpec("osc_waveform", 0, 2, "linear", discrete=3),
    ParamSpec("pitch_mod_mode", 0, 2, "linear", discrete=3),
    ParamSpec("osc_frequency", 20.0, 2000.0, "log"),
    ParamSpec("osc_decay", 1.0, 2000.0, "log"),
    ParamSpec("noise_filter_mode", 0, 2, "linear", discrete=3),
    ParamSpec("noise_filter_freq", 100.0, 12000.0, "log"),
    ParamSpec("osc_noise_mix", 0.0, 1.0, "linear"),
]

MINOR_PARAMS: List[ParamSpec] = [
    # Core tonal parameters (must be present for refinement)
    ParamSpec("osc_frequency", 20.0, 2000.0, "log"),
    ParamSpec("osc_decay", 1.0, 2000.0, "log"),
    # Pitch modulation
    ParamSpec("pitch_mod_amount", -48.0, 48.0, "linear"),
    ParamSpec("pitch_mod_rate", 1.0, 2000.0, "log"),
    ParamSpec("osc_attack", 0.0, 100.0, "linear"),
    # Noise parameters
    ParamSpec("noise_filter_q", 0.5, 100.0, "log"),
    ParamSpec("noise_envelope_mode", 0, 2, "linear", discrete=3),
    ParamSpec("noise_attack", 0.0, 100.0, "linear"),
    ParamSpec("noise_decay", 1.0, 2000.0, "log"),
    # Output parameters
    ParamSpec("distortion", 0.0, 0.6, "linear"),
    ParamSpec("eq_frequency", 20.0, 20000.0, "log"),
    ParamSpec("eq_gain_db", -40.0, 40.0, "linear"),
    ParamSpec("level_db", -12.0, 6.0, "linear"),
]


def _to_unit(value: float, spec: ParamSpec) -> float:
    value = spec.clamp(value)
    if spec.discrete:
        return float(np.clip(value / max(spec.discrete - 1, 1), 0.0, 1.0))
    if spec.scale == "log":
        min_log = math.log(spec.min_val)
        max_log = math.log(spec.max_val)
        return (math.log(value) - min_log) / (max_log - min_log)
    return (value - spec.min_val) / (spec.max_val - spec.min_val)


def _from_unit(u: float, spec: ParamSpec) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    if spec.discrete:
        idx = int(np.round(u * (spec.discrete - 1)))
        return float(idx)
    if spec.scale == "log":
        min_log = math.log(spec.min_val)
        max_log = math.log(spec.max_val)
        return float(math.exp(min_log + u * (max_log - min_log)))
    return spec.min_val + u * (spec.max_val - spec.min_val)


def _load_wav(path: str) -> Tuple[int, np.ndarray]:
    """Load WAV file and convert to mono float32."""
    sr, audio = wavfile.read(path)
    if audio.dtype.kind in ("i", "u"):
        max_val = np.iinfo(audio.dtype).max
        audio = audio.astype(np.float32) / max_val
    else:
        audio = audio.astype(np.float32)
    # Always convert to mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return sr, audio


def _resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return audio
    gcd = math.gcd(sr, target_sr)
    up = target_sr // gcd
    down = sr // gcd
    return signal.resample_poly(audio, up, down).astype(np.float32)


def _trim_onset(audio: np.ndarray, threshold: float = 1e-3) -> np.ndarray:
    idx = np.argmax(np.abs(audio) > threshold)
    if np.abs(audio[idx]) <= threshold:
        return audio
    return audio[idx:]


def _compute_features_cpu(audio: np.ndarray, sr: int, nperseg: int, hop: int) -> Dict[str, np.ndarray]:
    """CPU-based feature extraction using scipy."""
    if len(audio) < nperseg:
        pad = nperseg - len(audio)
        audio = np.pad(audio, (0, pad))

    _, _, zxx = signal.stft(audio, fs=sr, nperseg=nperseg, noverlap=nperseg - hop, boundary=None)
    mag = np.abs(zxx)
    log_mag = np.log1p(mag)

    freqs = np.linspace(0.0, sr / 2.0, mag.shape[0], dtype=np.float32)
    mag_sum = np.maximum(np.sum(mag, axis=0), 1e-8)
    centroid = np.sum(mag * freqs[:, None], axis=0) / mag_sum

    env_win = max(8, int(0.005 * sr))
    env_kernel = np.ones(env_win, dtype=np.float32) / env_win
    env = np.sqrt(np.convolve(audio * audio, env_kernel, mode="same"))

    return {
        "log_mag": log_mag.astype(np.float32),
        "centroid": centroid.astype(np.float32),
        "env": env.astype(np.float32),
    }


def _compute_features_gpu(audio: np.ndarray, sr: int, nperseg: int, hop: int) -> Dict[str, np.ndarray]:
    """GPU-accelerated feature extraction using CuPy."""
    if len(audio) < nperseg:
        pad = nperseg - len(audio)
        audio = np.pad(audio, (0, pad))

    # Transfer to GPU
    audio_gpu = _to_gpu(audio.astype(np.float32))
    
    # STFT on GPU
    _, _, zxx = cp_signal.stft(audio_gpu, fs=sr, nperseg=nperseg, noverlap=nperseg - hop, boundary=None)
    mag = cp.abs(zxx)
    log_mag = cp.log1p(mag)

    # Spectral centroid on GPU
    freqs = cp.linspace(0.0, sr / 2.0, mag.shape[0], dtype=cp.float32)
    mag_sum = cp.maximum(cp.sum(mag, axis=0), 1e-8)
    centroid = cp.sum(mag * freqs[:, None], axis=0) / mag_sum

    # Envelope on GPU
    env_win = max(8, int(0.005 * sr))
    env_kernel = cp.ones(env_win, dtype=cp.float32) / env_win
    env = cp.sqrt(cp.convolve(audio_gpu * audio_gpu, env_kernel, mode="same"))

    # Transfer back to CPU
    return {
        "log_mag": _to_cpu(log_mag).astype(np.float32),
        "centroid": _to_cpu(centroid).astype(np.float32),
        "env": _to_cpu(env).astype(np.float32),
    }


def _compute_features(audio: np.ndarray, sr: int, nperseg: int, hop: int) -> Dict[str, np.ndarray]:
    """Compute audio features, using GPU if available."""
    if _USE_GPU:
        try:
            return _compute_features_gpu(audio, sr, nperseg, hop)
        except Exception:
            # Fallback to CPU on any GPU error
            pass
    return _compute_features_cpu(audio, sr, nperseg, hop)


def _feature_loss(features: Dict[str, np.ndarray], target: Dict[str, np.ndarray], weights: Dict[str, float]) -> float:
    loss = 0.0
    for key, weight in weights.items():
        if weight <= 0:
            continue
        a = features.get(key)
        b = target.get(key)
        if a is None or b is None:
            continue
        min_len = min(a.shape[-1], b.shape[-1])
        if min_len == 0:
            continue
        if a.ndim == 1:
            diff = a[:min_len] - b[:min_len]
            mse = np.mean(diff * diff)
        else:
            diff = a[:, :min_len] - b[:, :min_len]
            mse = np.mean(diff * diff)
        
        # Normalize centroid loss to be comparable with other features
        # Centroid is in Hz, so we normalize by converting to relative difference
        # This makes a 10% frequency difference have similar weight as 10% amplitude difference
        if key == "centroid":
            # Use relative MSE: (diff / target)^2
            # Add small epsilon to avoid division by zero
            target_vals = b[:min_len]
            rel_diff = diff / (np.abs(target_vals) + 1e-6)
            mse = np.mean(rel_diff * rel_diff)
        
        if np.isfinite(mse):
            loss += weight * float(mse)
    return loss if np.isfinite(loss) else 1e6


# ---------------------------------------------------------------------------
# Initial parameter estimation from target WAV analysis
# ---------------------------------------------------------------------------

def _estimate_fundamental_freq(audio: np.ndarray, sr: int) -> Tuple[float, float]:
    """
    Estimate fundamental frequency using autocorrelation.
    Returns (frequency_hz, confidence 0-1).
    """
    # High-pass filter to remove DC
    b, a = signal.butter(2, 30.0 / (sr / 2), btype='high')
    audio_hp = signal.filtfilt(b, a, audio)
    
    # Use first ~50ms for attack pitch detection
    attack_samples = min(len(audio_hp), int(sr * 0.05))
    audio_attack = audio_hp[:attack_samples]
    
    if len(audio_attack) < 256:
        return 200.0, 0.0  # Default if too short
    
    # Autocorrelation
    corr = np.correlate(audio_attack, audio_attack, mode='full')
    corr = corr[len(corr) // 2:]  # Take positive lags only
    
    # Normalize
    corr = corr / (corr[0] + 1e-10)
    
    # Find peaks (min freq ~30Hz, max ~2000Hz)
    min_lag = max(1, int(sr / 2000))
    max_lag = min(len(corr) - 1, int(sr / 30))
    
    if max_lag <= min_lag:
        return 200.0, 0.0
    
    corr_search = corr[min_lag:max_lag]
    
    # Find first significant peak
    peaks, properties = signal.find_peaks(corr_search, height=0.1, distance=min_lag // 2 + 1)
    
    if len(peaks) == 0:
        return 200.0, 0.0
    
    # Take highest peak
    best_idx = peaks[np.argmax(properties['peak_heights'])]
    lag = best_idx + min_lag
    
    freq = sr / lag
    confidence = float(corr[lag])
    
    return float(np.clip(freq, 20.0, 2000.0)), float(np.clip(confidence, 0.0, 1.0))


def _estimate_envelope_decay(audio: np.ndarray, sr: int) -> float:
    """
    Estimate decay time in ms by finding time to reach -60dB from peak.
    """
    # Compute amplitude envelope
    window = max(8, int(sr * 0.002))
    env = np.sqrt(np.convolve(audio * audio, np.ones(window) / window, mode='same'))
    
    peak_idx = np.argmax(env)
    peak_val = env[peak_idx]
    
    if peak_val < 1e-8:
        return 100.0  # Default
    
    # Find -60dB point (0.001 of peak)
    threshold = peak_val * 0.001
    
    decay_region = env[peak_idx:]
    below_threshold = np.where(decay_region < threshold)[0]
    
    if len(below_threshold) == 0:
        # Never reaches threshold, use length
        decay_samples = len(decay_region)
    else:
        decay_samples = below_threshold[0]
    
    decay_ms = (decay_samples / sr) * 1000.0
    return float(np.clip(decay_ms, 1.0, 2000.0))


def _estimate_noise_ratio(audio: np.ndarray, sr: int) -> float:
    """
    Estimate tonal vs noise ratio (0 = pure tone, 1 = pure noise).
    Uses spectral flatness (Wiener entropy) computed over STFT frames.
    """
    # Analyze first 0.5 seconds or full audio if shorter
    win_samples = min(len(audio), int(sr * 0.5))
    audio_win = audio[:win_samples]
    
    if len(audio_win) < 256:
        return 0.5
    
    # Use STFT for time-varying analysis
    f, t, Zxx = signal.stft(audio_win, fs=sr, nperseg=512, noverlap=384)
    mag = np.abs(Zxx)
    
    # Compute spectral flatness for each frame
    flatness_values = []
    num_frames = min(20, mag.shape[1])  # Analyze first 20 frames
    for i in range(num_frames):
        frame = mag[:, i] + 1e-10
        geom_mean = np.exp(np.mean(np.log(frame)))
        arith_mean = np.mean(frame)
        flatness = geom_mean / arith_mean
        flatness_values.append(flatness)
    
    avg_flatness = float(np.mean(flatness_values))
    
    # Map to 0-1 range
    # flatness < 0.15 = highly tonal (< 15% noise)
    # flatness > 0.5 = mostly noise (> 50% noise)
    noise_ratio = float(np.clip(avg_flatness / 0.5, 0.0, 1.0))
    return noise_ratio


def _estimate_spectral_centroid(audio: np.ndarray, sr: int) -> float:
    """
    Estimate spectral centroid frequency in Hz.
    """
    win_samples = min(len(audio), int(sr * 0.05))
    audio_win = audio[:win_samples]
    
    if len(audio_win) < 256:
        return 2000.0
    
    nperseg = min(512, len(audio_win))
    f, psd = signal.welch(audio_win, sr, nperseg=nperseg)
    
    psd_sum = np.sum(psd) + 1e-10
    centroid = np.sum(f * psd) / psd_sum
    
    return float(np.clip(centroid, 100.0, 12000.0))


def _estimate_pitch_sweep(audio: np.ndarray, sr: int) -> Tuple[float, float, int]:
    """
    Detect if there's a pitch sweep (common in kicks/toms).
    Returns (start_freq, end_freq, sweep_direction).
    sweep_direction: 0=none, 1=down, 2=up
    """
    # Analyze pitch in short windows
    win_ms = 10
    hop_ms = 5
    win_samples = int(sr * win_ms / 1000)
    hop_samples = int(sr * hop_ms / 1000)
    
    freqs = []
    n_windows = min(10, (len(audio) - win_samples) // hop_samples)
    
    if n_windows < 3:
        return 200.0, 200.0, 0
    
    for i in range(n_windows):
        start = i * hop_samples
        win = audio[start:start + win_samples]
        
        if len(win) < win_samples:
            break
            
        # Simple zero-crossing rate as frequency proxy
        zero_crossings = np.sum(np.abs(np.diff(np.sign(win))) > 0)
        freq_estimate = (zero_crossings / 2) * (sr / win_samples)
        
        if 20 < freq_estimate < 2000:
            freqs.append(freq_estimate)
    
    if len(freqs) < 3:
        return 200.0, 200.0, 0
    
    # Detect trend
    start_freq = np.mean(freqs[:2])
    end_freq = np.mean(freqs[-2:])
    
    ratio = start_freq / (end_freq + 1e-6)
    
    if ratio > 1.5:
        # Pitch drops (common in kicks)
        return start_freq, end_freq, 1  # Down sweep
    elif ratio < 0.67:
        # Pitch rises
        return start_freq, end_freq, 2  # Up sweep
    else:
        return start_freq, end_freq, 0  # No significant sweep


def _detect_waveform_type(audio: np.ndarray, sr: int) -> int:
    """
    Guess waveform type: 0=sine, 1=triangle, 2=saw.
    Based on harmonic content analysis.
    
    For drum sounds, we need to be careful: pitch sweeps create apparent
    harmonics in short windows. We use multiple strategies:
    1. Analyze a longer, more stable portion (after initial transient)
    2. Use stricter thresholds for classifying as non-sine
    3. Check harmonic decay pattern (1/n for saw, 1/n² for triangle)
    """
    # Use a window from after the initial attack transient
    # Start at ~5ms to skip click/transient, analyze ~30ms
    start_samples = min(int(sr * 0.005), len(audio) // 4)
    win_samples = min(len(audio) - start_samples, int(sr * 0.03))
    
    if win_samples < 256:
        # Too short, try from start
        start_samples = 0
        win_samples = min(len(audio), int(sr * 0.02))
    
    if win_samples < 256:
        return 0  # Default to sine
    
    audio_win = audio[start_samples:start_samples + win_samples]
    
    # Apply window function to reduce spectral leakage
    window = np.hanning(len(audio_win))
    audio_win = audio_win * window
    
    # Get spectrum
    spectrum = np.abs(np.fft.rfft(audio_win))
    
    if len(spectrum) < 10:
        return 0
    
    # Find fundamental - look for strongest peak above 30 Hz
    min_bin = max(1, int(30 * len(audio_win) / sr))
    fundamental_idx = np.argmax(spectrum[min_bin:]) + min_bin
    
    if fundamental_idx < 1:
        return 0
    
    # Check harmonics presence and relative levels
    # Use a small search window around each harmonic to handle slight inharmonicity
    n_harmonics = min(6, len(spectrum) // fundamental_idx - 1)
    
    if n_harmonics < 2:
        return 0  # Not enough data, assume sine
    
    harmonic_levels = []
    search_width = max(1, fundamental_idx // 8)  # Allow some frequency drift
    
    for h in range(1, n_harmonics + 1):
        idx = fundamental_idx * h
        start = max(0, idx - search_width)
        end = min(len(spectrum), idx + search_width + 1)
        if end > start:
            # Take max in neighborhood to handle slight detuning
            harmonic_levels.append(np.max(spectrum[start:end]))
    
    if len(harmonic_levels) < 2:
        return 0
    
    harmonic_levels = np.array(harmonic_levels)
    fundamental_level = harmonic_levels[0]
    
    if fundamental_level < 1e-10:
        return 0
    
    harmonic_levels_norm = harmonic_levels / fundamental_level
    
    # Sine: very weak harmonics
    # For real-world signals, allow slightly higher threshold
    # A pure sine should have harmonics < 0.05 of fundamental
    total_harmonic_energy = np.sum(harmonic_levels_norm[1:])
    max_harmonic_ratio = np.max(harmonic_levels_norm[1:]) if len(harmonic_levels_norm) > 1 else 0
    
    # If strongest harmonic is < 20% of fundamental, likely sine
    # (real drums with pitch sweep can have harmonics around 25-30%)
    if max_harmonic_ratio < 0.20 and total_harmonic_energy < 0.5:
        return 0  # Sine
    
    # If harmonics are very weak overall, it's sine
    if total_harmonic_energy < 0.15:
        return 0  # Sine
    
    # Check harmonic decay pattern to distinguish triangle from saw
    # Saw: harmonics fall off as 1/n (levels: 1, 0.5, 0.33, 0.25...)
    # Triangle: odd harmonics only, fall off as 1/n² (levels: 1, 0, 0.11, 0, 0.04...)
    
    # Check odd vs even harmonic balance
    odd_harmonics = harmonic_levels_norm[::2]  # 1st, 3rd, 5th... (indices 0, 2, 4)
    even_harmonics = harmonic_levels_norm[1::2]  # 2nd, 4th, 6th... (indices 1, 3, 5)
    
    odd_energy = np.sum(odd_harmonics)
    even_energy = np.sum(even_harmonics)
    
    odd_ratio = odd_energy / (odd_energy + even_energy + 1e-10)
    
    # Triangle has mostly odd harmonics (>85%)
    if odd_ratio > 0.85:
        return 1  # Triangle
    
    # Check if harmonics follow 1/n pattern (saw)
    # Compare actual levels to theoretical 1/n
    if len(harmonic_levels_norm) >= 3:
        theoretical_saw = np.array([1.0 / h for h in range(1, len(harmonic_levels_norm) + 1)])
        correlation = np.corrcoef(harmonic_levels_norm, theoretical_saw)[0, 1]
        
        # Strong correlation with 1/n pattern suggests saw
        if correlation > 0.9 and total_harmonic_energy > 0.5:
            return 2  # Saw
    
    # Default: if harmonics present but pattern unclear, check intensity
    if total_harmonic_energy > 0.8:
        return 2  # Saw (rich harmonics)
    
    # Moderate harmonics without clear pattern - likely sine with some distortion
    return 0  # Sine


def _detect_filter_mode(audio: np.ndarray, sr: int) -> Tuple[int, float]:
    """
    Detect filter type and cutoff frequency.
    Returns (mode: 0=LP, 1=BP, 2=HP, cutoff_hz).
    """
    # Analyze spectral shape
    win_samples = min(len(audio), int(sr * 0.05))
    audio_win = audio[:win_samples]
    
    if len(audio_win) < 256:
        return 0, 4000.0
    
    nperseg = min(512, len(audio_win))
    f, psd = signal.welch(audio_win, sr, nperseg=nperseg)
    
    psd_db = 10 * np.log10(psd + 1e-10)
    
    # Find spectral peak
    peak_idx = np.argmax(psd_db)
    peak_freq = f[peak_idx]
    
    # Analyze roll-off
    half_len = len(psd_db) // 2
    low_energy = np.mean(psd_db[:half_len])
    high_energy = np.mean(psd_db[half_len:])
    
    # Check spectral centroid relative to Nyquist
    centroid = _estimate_spectral_centroid(audio, sr)
    nyquist = sr / 2
    centroid_ratio = centroid / nyquist
    
    if centroid_ratio < 0.15:
        # Low frequency content dominant -> LP filter likely
        return 0, float(np.clip(centroid * 2, 100, 12000))
    elif centroid_ratio > 0.5:
        # High frequency content -> HP filter likely
        return 2, float(np.clip(centroid * 0.5, 100, 12000))
    else:
        # Bandpass
        return 1, float(np.clip(peak_freq, 100, 12000))


def _analyze_target_wav(audio: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Analyze target WAV and return estimated initial parameters.
    """
    print("[Analysis] Analyzing target waveform...")
    
    # Normalize for analysis
    audio_norm = audio / (np.max(np.abs(audio)) + 1e-8)
    
    # Fundamental frequency
    freq, freq_confidence = _estimate_fundamental_freq(audio_norm, sr)
    print(f"  Fundamental frequency: {freq:.1f} Hz (confidence: {freq_confidence:.2f})")
    
    # Decay time
    decay_ms = _estimate_envelope_decay(audio_norm, sr)
    print(f"  Decay time: {decay_ms:.1f} ms")
    
    # Noise ratio
    noise_ratio = _estimate_noise_ratio(audio_norm, sr)
    print(f"  Noise ratio: {noise_ratio:.2f} (0=tonal, 1=noisy)")
    
    # Recommend --no-noise if sound is highly tonal
    if noise_ratio < 0.3:  # Less than 30% noise content
        print(f"  [Recommendation] Consider using --no-noise flag (highly tonal sound)")
    
    # Spectral centroid
    centroid = _estimate_spectral_centroid(audio_norm, sr)
    print(f"  Spectral centroid: {centroid:.1f} Hz")
    
    # Pitch sweep detection
    start_freq, end_freq, sweep_dir = _estimate_pitch_sweep(audio_norm, sr)
    sweep_names = ["none", "down", "up"]
    print(f"  Pitch sweep: {sweep_names[sweep_dir]} ({start_freq:.1f} -> {end_freq:.1f} Hz)")
    
    # Waveform type
    waveform = _detect_waveform_type(audio_norm, sr)
    waveform_names = ["sine", "triangle", "saw"]
    print(f"  Waveform guess: {waveform_names[waveform]}")
    
    # Filter mode
    filter_mode, filter_freq = _detect_filter_mode(audio_norm, sr)
    filter_names = ["lowpass", "bandpass", "highpass"]
    print(f"  Filter mode: {filter_names[filter_mode]} @ {filter_freq:.1f} Hz")
    
    # Build estimated parameters
    estimated = {
        # Oscillator
        "osc_waveform": float(waveform),
        "osc_frequency": freq if freq_confidence > 0.3 else start_freq,
        "osc_decay": decay_ms,
        "osc_attack": 0.0,  # Most drums have instant attack
        
        # Pitch modulation
        "pitch_mod_mode": float(sweep_dir),  # 0=off, 1=down, 2=up
        "pitch_mod_amount": 24.0 if sweep_dir != 0 else 0.0,  # Semitones
        "pitch_mod_rate": decay_ms * 0.5 if sweep_dir != 0 else 100.0,
        
        # Noise
        "osc_noise_mix": noise_ratio,
        "noise_filter_mode": float(filter_mode),
        "noise_filter_freq": filter_freq,
        "noise_filter_q": 2.0,
        "noise_decay": decay_ms,
        "noise_attack": 0.0,
        "noise_envelope_mode": 1,  # Decay mode
        
        # EQ / output
        "eq_frequency": centroid,
        "eq_gain_db": 0.0,
        "distortion": 0.0,
        "level_db": 0.0,
    }
    
    print("[Analysis] Initial parameter estimation complete")
    return estimated


def _prepare_base_params() -> Dict[str, float]:
    """Return fixed parameters that are not optimized (pan, effects, etc.)."""
    return {
        "pan": 0.0,  # Always centered (mono target)
        "vintage_amount": 0.0,
        "reverb_decay": 0.0,
        "reverb_mix": 0.0,
        "reverb_width": 1.0,
        "delay_feedback": 0.0,
        "delay_mix": 0.0,
        "delay_ping_pong": False,
    }


def _get_discrete_params(param_list: List[ParamSpec]) -> List[ParamSpec]:
    """Return only discrete parameters from the list."""
    return [p for p in param_list if p.discrete is not None]


def _get_continuous_params(param_list: List[ParamSpec]) -> List[ParamSpec]:
    """Return only continuous parameters from the list."""
    return [p for p in param_list if p.discrete is None]


def _filter_noise_params(param_list: List[ParamSpec]) -> List[ParamSpec]:
    """Remove noise-related parameters from the list."""
    noise_param_names = {
        "osc_noise_mix",
        "noise_filter_mode",
        "noise_filter_freq",
        "noise_filter_q",
        "noise_envelope_mode",
        "noise_attack",
        "noise_decay",
    }
    return [p for p in param_list if p.name not in noise_param_names]


def _filter_waveform_param(param_list: List[ParamSpec]) -> List[ParamSpec]:
    """Remove oscillator waveform parameter from the list."""
    return [p for p in param_list if p.name != "osc_waveform"]


def _filter_pitch_mod_params(param_list: List[ParamSpec]) -> List[ParamSpec]:
    """Remove pitch modulation parameters from the list."""
    pitch_mod_param_names = {
        "pitch_mod_mode",
        "pitch_mod_amount",
        "pitch_mod_rate",
    }
    return [p for p in param_list if p.name not in pitch_mod_param_names]


def _grid_search_discrete(
    discrete_params: List[ParamSpec],
    base_params: Dict[str, float],
    evaluate_fn,
    continuous_params: List[ParamSpec],
) -> Tuple[Dict[str, float], float]:
    """Grid search over all discrete parameter combinations."""
    if not discrete_params:
        return base_params, float("inf")
    
    # Generate all combinations of discrete values
    discrete_ranges = []
    for p in discrete_params:
        discrete_ranges.append(list(range(p.discrete)))
    
    best_params = dict(base_params)
    best_loss = float("inf")
    
    combinations = list(itertools.product(*discrete_ranges))
    print(f"  [Grid] Searching {len(combinations)} discrete combinations...")
    
    for combo in combinations:
        test_params = dict(base_params)
        for val, spec in zip(combo, discrete_params):
            test_params[spec.name] = float(val)
        
        # Build unit vector for continuous params only (use midpoint)
        unit_vec = np.array([
            _to_unit(test_params.get(p.name, (p.min_val + p.max_val) / 2), p)
            for p in continuous_params
        ], dtype=np.float32)
        
        # Temporarily set discrete params and evaluate
        loss = evaluate_fn(unit_vec, test_params, use_main_synth=True)
        
        if loss < best_loss:
            best_loss = loss
            best_params = dict(test_params)
    
    print(f"  [Grid] Best discrete combo loss: {best_loss:.6f}")
    return best_params, best_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a drum patch to a target WAV using CMA-ES")
    parser.add_argument("--target", required=True, help="Path to target WAV")
    parser.add_argument("--channel", type=int, default=0, help="Channel index (0-7)")
    parser.add_argument("--out", default="", help="Output .mtdrum path (default: <target_name>.mtdrum)")
    parser.add_argument("--name", default="", help="Patch name (default: <target_name>)")
    parser.add_argument("--max-evals", type=int, default=0, help="Maximum CMA-ES evaluations (0=auto: ~33k total)")
    parser.add_argument("--max-iter", type=int, default=0, help="Maximum CMA-ES iterations (0=auto)")
    parser.add_argument("--popsize", type=int, default=0, help="Population size (0=auto)")
    parser.add_argument("--sigma", type=float, default=0.25, help="Initial CMA-ES sigma in unit space")
    parser.add_argument("--target-loss", type=float, default=1e-5, help="Stop early only if best loss <= target-loss")
    parser.add_argument("--min-evals", type=int, default=200, help="Minimum evaluations before allowing early stop")
    parser.add_argument("--tolx", type=float, default=1e-10, help="CMA-ES tolx (smaller = stricter)")
    parser.add_argument("--tolfun", type=float, default=1e-12, help="CMA-ES tolfun (smaller = stricter)")
    parser.add_argument("--tolstagnation", type=int, default=200, help="CMA-ES tolstagnation (higher = less early stop)")
    parser.add_argument("--tolfunhist", type=float, default=1e-12, help="CMA-ES tolfunhist (smaller = stricter)")
    parser.add_argument("--tolflatfitness", type=float, default=1e-12, help="CMA-ES tolflatfitness (smaller = stricter)")
    parser.add_argument("--allow-cma-stop", action="store_true", help="Allow CMA-ES internal stop criteria (disabled by default)")
    parser.add_argument("--checkpoint-dir", default="", help="Directory to write checkpoints (empty=auto)")
    parser.add_argument("--checkpoint-interval", type=int, default=0, help="Deprecated (no effect).")
    parser.add_argument("--resume", default="", help="Path to checkpoint JSON to resume from")
    parser.add_argument("--resume-latest", action="store_true", help="Resume from latest checkpoint in checkpoint-dir")
    parser.add_argument("--no-auto-resume", action="store_true", help="Disable automatic resume from default checkpoint")
    parser.add_argument("--single-stage", action="store_true", help="Disable multi-stage fitting (use single stage)")
    parser.add_argument("--stages-config", default="", help="Path to JSON stages config")
    parser.add_argument("--feature-sr", type=int, default=11025, help="Sample rate for feature extraction")
    parser.add_argument("--nperseg", type=int, default=512, help="STFT window size")
    parser.add_argument("--hop", type=int, default=128, help="STFT hop size")
    parser.add_argument("--velocity", type=int, default=127, help="Trigger velocity")
    parser.add_argument("--param-set", choices=["core", "extended"], default="core")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--align", action="store_true", help="Trim leading silence before feature extraction")
    parser.add_argument("--export-wav", default="", help="Optional output WAV of best fit")
    parser.add_argument("--weights", default="", help="JSON weights: {\"spec\":1,\"env\":0.5,\"centroid\":0.2,\"rms\":0.1}")
    parser.add_argument("--workers", type=int, default=0, help="Parallel workers (0=auto, 1=serial)")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration (auto-detect)")
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU-only mode")
    parser.add_argument("--no-grid-search", action="store_true", help="Disable grid search for discrete params")
    parser.add_argument("--restarts", type=int, default=0, help="Number of random restarts (0=use stage default)")
    parser.add_argument("--popsize-multiplier", type=float, default=1.5, help="Multiply default popsize (default 1.5x)")
    parser.add_argument("--no-analyze", action="store_true", help="Skip initial waveform analysis for parameter estimation")
    parser.add_argument("--no-noise", action="store_true", help="Disable noise oscillator optimization (for kicks/toms without noise)")
    parser.add_argument("--no-pitch-mod", action="store_true", help="Disable pitch modulation optimization (for sounds without pitch sweep)")
    parser.add_argument("--force-waveform", choices=["sine", "triangle", "saw"], default="", help="Force specific oscillator waveform (locks it, excludes from optimization)")
    args = parser.parse_args()

    if not os.path.isfile(args.target):
        raise FileNotFoundError(f"Target WAV not found: {args.target}")

    target_base = os.path.splitext(os.path.basename(args.target))[0]
    
    # Use target filename for output and patch name if not specified
    if not args.out:
        args.out = f"{target_base}.mtdrum"
    if not args.name:
        args.name = target_base

    # Initialize GPU if requested or auto-detect
    if args.no_gpu:
        gpu_active = _init_gpu(force_cpu=True)
        print("[GPU] Disabled by --no-gpu flag")
    elif args.gpu or _CUPY_AVAILABLE:
        gpu_active = _init_gpu(force_cpu=False)
        if gpu_active:
            dev_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
            print(f"[GPU] Enabled: {dev_name}")
        else:
            print("[GPU] Not available, using CPU")
    else:
        print("[GPU] CuPy not installed, using CPU")

    if not args.checkpoint_dir:
        args.checkpoint_dir = os.path.join(PROJECT_ROOT, "checkpoints", target_base)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    np.random.seed(args.seed)

    target_sr, target_audio = _load_wav(args.target)
    if args.align:
        target_audio = _trim_onset(target_audio)
    
    # Always normalize target audio to full scale
    max_abs = np.max(np.abs(target_audio))
    if max_abs > 0:
        target_audio = target_audio / max_abs
        print(f"[Normalization] Target audio normalized (peak: {max_abs:.4f})")

    synth_sr = target_sr
    synth = PythonicSynthesizer(sample_rate=synth_sr)
    channel = synth.channels[args.channel]

    params = CORE_PARAMS
    original_param_count = len(CORE_PARAMS)
    if args.no_noise:
        params = _filter_noise_params(params)
    if args.no_pitch_mod:
        params = _filter_pitch_mod_params(params)
    if args.force_waveform:
        params = _filter_waveform_param(params)
    if args.no_noise or args.no_pitch_mod or args.force_waveform:
        exclusions = []
        if args.no_noise:
            exclusions.append("noise")
        if args.no_pitch_mod:
            exclusions.append("pitch_mod")
        if args.force_waveform:
            exclusions.append("waveform")
        print(f"[Config] Reduced parameter space from {original_param_count} to {len(params)} parameters ({', '.join(exclusions)} excluded)")
    base_params = channel.get_parameters()
    base_params.update(_prepare_base_params())

    # Initial parameter estimation from waveform analysis (enabled by default)
    if not args.no_analyze:
        estimated_params = _analyze_target_wav(target_audio, target_sr)
        base_params.update(estimated_params)
        print("[Analysis] Using estimated parameters as starting point")
    
    # Disable noise if requested
    if args.no_noise:
        base_params["osc_noise_mix"] = 1.0  # 1.0 = pure oscillator, 0.0 = pure noise
        base_params["noise_filter_mode"] = 0.0
        base_params["noise_filter_freq"] = 1000.0
        base_params["noise_filter_q"] = 1.0
        base_params["noise_envelope_mode"] = 0.0
        base_params["noise_attack"] = 0.0
        base_params["noise_decay"] = 0.0
        print("[Config] Noise oscillator disabled (--no-noise)")
    
    # Disable pitch modulation if requested
    if args.no_pitch_mod:
        base_params["pitch_mod_mode"] = 0.0  # DECAYING mode with 0 amount = no modulation
        base_params["pitch_mod_amount"] = 0.0
        base_params["pitch_mod_rate"] = 100.0
        print("[Config] Pitch modulation disabled (--no-pitch-mod)")
    
    # Force waveform if requested
    if args.force_waveform:
        waveform_map = {"sine": 0.0, "triangle": 1.0, "saw": 2.0}
        base_params["osc_waveform"] = waveform_map[args.force_waveform]
        print(f"[Config] Oscillator waveform locked to {args.force_waveform} (--force-waveform)")

    def load_checkpoint_params() -> Optional[Dict[str, float]]:
        if args.resume:
            with open(args.resume, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("params")
        if (args.resume_latest or not args.no_auto_resume) and args.checkpoint_dir:
            if not os.path.isdir(args.checkpoint_dir):
                return None
            candidates = [
                os.path.join(args.checkpoint_dir, f)
                for f in os.listdir(args.checkpoint_dir)
                if f.lower().endswith(".json")
            ]
            best_path = None
            best_evals = -1
            for path in candidates:
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    evals = int(data.get("evals", -1))
                    if evals > best_evals:
                        best_evals = evals
                        best_path = path
                except (OSError, json.JSONDecodeError, ValueError):
                    continue
            if best_path:
                with open(best_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("params")
        return None

    resume_params = load_checkpoint_params()
    if resume_params:
        base_params.update(resume_params)
        print("[Resume] Using checkpoint parameters (overriding analysis)")
    elif not args.no_analyze:
        print("[Init] Starting from analyzed parameters")

    unit_init = np.array([_to_unit(base_params.get(p.name, p.min_val), p) for p in params], dtype=np.float32)

    weights = {
        "log_mag": 1.0,
        "env": 0.5,
        "centroid": 0.2,
        "rms": 0.1,
    }
    if args.weights:
        user_weights = json.loads(args.weights)
        weights.update({
            "log_mag": float(user_weights.get("spec", weights["log_mag"])),
            "env": float(user_weights.get("env", weights["env"])),
            "centroid": float(user_weights.get("centroid", weights["centroid"])),
            "rms": float(user_weights.get("rms", weights["rms"])),
        })

    def save_checkpoint(
        tag: str,
        loss: float,
        evals: int,
        stage_name: str,
        vector: np.ndarray,
        param_list: List[ParamSpec],
    ) -> None:
        if not args.checkpoint_dir:
            return
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        base_path = os.path.join(args.checkpoint_dir, f"{target_base}_best")
        checkpoint_params = dict(base_params)
        for u, spec in zip(vector, param_list):
            checkpoint_params[spec.name] = _from_unit(u, spec)
        checkpoint_params["name"] = args.name
        channel.set_parameters(checkpoint_params)
        DrumPatchWriter.write_drum_patch(base_path + ".mtdrum", channel, args.name)
        with open(base_path + ".json", "w", encoding="utf-8") as f:
            json.dump({"loss": loss, "evals": evals, "stage": stage_name, "params": checkpoint_params}, f, indent=2)

    def load_stages() -> List[Dict[str, float]]:
        if args.stages_config:
            with open(args.stages_config, "r", encoding="utf-8") as f:
                return json.load(f)
        if args.single_stage:
            return [{
                "name": "single",
                "feature_sr": args.feature_sr,
                "nperseg": args.nperseg,
                "hop": args.hop,
                "max_evals": args.max_evals if args.max_evals > 0 else 20000,
                "sigma": args.sigma,
                "target_loss": args.target_loss,
                "min_evals": args.min_evals,
                "grid_search_discrete": True,
                "restarts": 3,
            }]
        return [
            {
                "name": "coarse",
                "feature_sr": 8000,
                "nperseg": 256,
                "hop": 64,
                "max_evals": 8000,
                "sigma": 0.5,
                "target_loss": 1e-2,
                "min_evals": 1000,
                "param_group": "major",
                "grid_search_discrete": True,
                "restarts": 3,
            },
            {
                "name": "mid",
                "feature_sr": 11025,
                "nperseg": 512,
                "hop": 128,
                "max_evals": 10000,
                "sigma": 0.35,
                "target_loss": 5e-4,
                "min_evals": 1500,
                "param_group": "minor",
                "restarts": 2,
            },
            {
                "name": "fine",
                "feature_sr": 22050,
                "nperseg": 1024,
                "hop": 256,
                "max_evals": 15000,
                "sigma": 0.25,
                "target_loss": 1e-5,
                "min_evals": 2000,
                "param_group": "core",
                "restarts": 1,
            },
        ]

    stages = load_stages()

    best_u = unit_init.copy()
    best_loss = float("inf")

    # Determine worker count
    if args.workers == 0:
        num_workers = min(os.cpu_count() or 4, 8)
    else:
        num_workers = max(1, args.workers)

    # Worker-local synthesizer storage for thread pool
    _worker_synths: Dict[int, PythonicSynthesizer] = {}
    _worker_lock = threading.Lock()

    def get_worker_synth() -> Tuple[PythonicSynthesizer, any]:
        """Get or create a thread-local synthesizer."""
        tid = threading.get_ident()
        with _worker_lock:
            if tid not in _worker_synths:
                _worker_synths[tid] = PythonicSynthesizer(sample_rate=synth_sr)
        return _worker_synths[tid], _worker_synths[tid].channels[args.channel]

    for stage in stages:
        stage_name = stage.get("name", "stage")
        feature_sr = int(stage.get("feature_sr", args.feature_sr))
        if feature_sr <= 0:
            feature_sr = target_sr
        nperseg = int(stage.get("nperseg", args.nperseg))
        hop = int(stage.get("hop", args.hop))
        if args.allow_cma_stop:
            max_evals = int(stage.get("max_evals", args.max_evals))
        else:
            max_evals = int(args.max_evals) if args.max_evals > 0 else int(stage.get("max_evals", 10000))
        sigma = float(stage.get("sigma", args.sigma))
        target_loss = float(stage.get("target_loss", args.target_loss))
        min_evals = int(stage.get("min_evals", args.min_evals))

        param_group = stage.get("param_group", "default")
        if param_group == "major":
            stage_params = REALLY_MAJOR_PARAMS
        elif param_group == "minor":
            stage_params = MINOR_PARAMS
        elif param_group == "core":
            stage_params = CORE_PARAMS
        else:
            stage_params = params
        
        # Apply noise filter if enabled
        if args.no_noise:
            stage_params = _filter_noise_params(stage_params)
        # Apply pitch mod filter if disabled
        if args.no_pitch_mod:
            stage_params = _filter_pitch_mod_params(stage_params)
        # Apply waveform filter if forced
        if args.force_waveform:
            stage_params = _filter_waveform_param(stage_params)

        target_audio_rs = _resample(target_audio, target_sr, feature_sr)
        target_audio_rs = target_audio_rs / (np.max(np.abs(target_audio_rs)) + 1e-8)
        target_features = _compute_features(target_audio_rs, feature_sr, nperseg, hop)
        target_rms = float(np.sqrt(np.mean(target_audio_rs * target_audio_rs)))

        # Pre-allocate buffers for evaluation to reduce allocations
        target_len = len(target_audio)
        target_rs_len = len(target_audio_rs)
        
        # Thread-local pre-allocated buffers
        class WorkerBuffers:
            __slots__ = ('audio_buf', 'audio_rs_buf', 'audio_norm_buf')
            def __init__(self):
                self.audio_buf = np.zeros(target_len, dtype=np.float32)
                self.audio_rs_buf = np.zeros(target_rs_len, dtype=np.float32)
                self.audio_norm_buf = np.zeros(target_rs_len, dtype=np.float32)
        
        _worker_buffers: Dict[int, WorkerBuffers] = {}
        _buffers_lock = threading.Lock()
        
        # Main thread buffer
        _main_buffers = WorkerBuffers()
        
        def get_buffers(use_main: bool) -> WorkerBuffers:
            if use_main or num_workers == 1:
                return _main_buffers
            tid = threading.get_ident()
            with _buffers_lock:
                if tid not in _worker_buffers:
                    _worker_buffers[tid] = WorkerBuffers()
                return _worker_buffers[tid]

        cache: Dict[Tuple[float, ...], float] = {}
        cache_lock = threading.Lock()
        
        # Separate discrete and continuous params for this stage
        discrete_stage_params = _get_discrete_params(stage_params)
        continuous_stage_params = _get_continuous_params(stage_params)

        def evaluate(unit_vector: np.ndarray, override_params: Optional[Dict[str, float]] = None, use_main_synth: bool = False) -> float:
            """Evaluate a solution. unit_vector is for continuous params only when override_params is set."""
            key = tuple(np.round(unit_vector, 6))
            if override_params:
                # Include discrete param values in cache key
                discrete_key = tuple(override_params.get(p.name, 0) for p in discrete_stage_params)
                key = key + discrete_key
            
            with cache_lock:
                if key in cache:
                    return cache[key]

            # Use worker-local synth for parallel evaluation, main synth for serial
            if use_main_synth or num_workers == 1:
                ch = channel
            else:
                _, ch = get_worker_synth()

            bufs = get_buffers(use_main_synth)

            # Build patch params
            if override_params:
                patch_params = dict(override_params)
                for u, spec in zip(unit_vector, continuous_stage_params):
                    patch_params[spec.name] = _from_unit(u, spec)
            else:
                patch_params = dict(base_params)
                for u, spec in zip(unit_vector, stage_params):
                    patch_params[spec.name] = _from_unit(u, spec)

            ch.set_parameters(patch_params)
            ch.trigger(args.velocity)
            audio_stereo = ch.process(target_len)
            # In-place mean to mono
            np.mean(audio_stereo, axis=1, out=bufs.audio_buf)
            audio = bufs.audio_buf
            if args.align:
                audio = _trim_onset(audio)

            audio_rs = _resample(audio, synth_sr, feature_sr)
            # Copy into pre-allocated buffer with proper length handling
            if len(audio_rs) < target_rs_len:
                bufs.audio_rs_buf[:len(audio_rs)] = audio_rs
                bufs.audio_rs_buf[len(audio_rs):] = 0.0
                audio_rs = bufs.audio_rs_buf
            elif len(audio_rs) > target_rs_len:
                bufs.audio_rs_buf[:] = audio_rs[:target_rs_len]
                audio_rs = bufs.audio_rs_buf
            else:
                bufs.audio_rs_buf[:] = audio_rs
                audio_rs = bufs.audio_rs_buf

            # Normalize in-place (with NaN protection)
            max_abs = np.max(np.abs(audio_rs))
            if max_abs < 1e-10 or not np.isfinite(max_abs):
                # Silent or invalid audio - return high penalty
                loss = 1e6
                with cache_lock:
                    cache[key] = loss
                return loss
            np.divide(audio_rs, max_abs + 1e-8, out=bufs.audio_norm_buf)
            
            features = _compute_features(bufs.audio_norm_buf, feature_sr, nperseg, hop)
            loss = _feature_loss(features, target_features, {
                "log_mag": weights["log_mag"],
                "env": weights["env"],
                "centroid": weights["centroid"],
            })

            rms = float(np.sqrt(np.mean(audio_rs * audio_rs)))
            if np.isfinite(rms) and np.isfinite(target_rms):
                loss += weights["rms"] * float((rms - target_rms) ** 2)

            # Final NaN check - return high penalty if loss is invalid
            if not np.isfinite(loss):
                loss = 1e6

            with cache_lock:
                cache[key] = loss
            return loss

        def evaluate_batch(solutions: List[np.ndarray], override_params: Optional[Dict[str, float]] = None) -> List[float]:
            """Evaluate a batch of solutions, possibly in parallel."""
            if num_workers == 1:
                return [evaluate(np.array(x, dtype=np.float32), override_params, use_main_synth=True) for x in solutions]
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [
                    executor.submit(evaluate, np.array(x, dtype=np.float32), override_params, False)
                    for x in solutions
                ]
                return [f.result() for f in futures]

        # Grid search for discrete params (enabled by default)
        do_grid_search = stage.get("grid_search_discrete", True) and not args.no_grid_search
        stage_base_params = dict(base_params)
        
        if do_grid_search and discrete_stage_params:
            stage_base_params, grid_loss = _grid_search_discrete(
                discrete_stage_params,
                base_params,
                evaluate,
                continuous_stage_params,
            )
            # Update base_params with best discrete values
            for p in discrete_stage_params:
                base_params[p.name] = stage_base_params[p.name]
        
        # Determine number of restarts
        num_restarts = args.restarts if args.restarts > 0 else int(stage.get("restarts", 1))
        
        # Determine population size
        base_popsize = args.popsize if args.popsize > 0 else 0
        popsize_mult = args.popsize_multiplier
        
        cma_opts_base = {
            "seed": args.seed,
            "verbose": -9,
            "tolx": args.tolx,
            "tolfun": args.tolfun,
            "tolstagnation": args.tolstagnation,
            "tolfunhist": args.tolfunhist,
            "tolflatfitness": args.tolflatfitness,
        }
        
        # Use continuous params for CMA-ES when grid search is used
        if do_grid_search and discrete_stage_params:
            cma_params = continuous_stage_params
        else:
            cma_params = stage_params
        
        stage_best = None
        stage_best_loss = float("inf")
        total_evals = 0
        
        print(f"[{stage_name}] Starting with {num_workers} worker(s), {num_restarts} restart(s)")
        
        for restart_idx in range(num_restarts):
            # Initialize: first restart uses current best, others use random
            if restart_idx == 0:
                if stage_params is not params:
                    init_vec = np.array([
                        _to_unit(base_params.get(p.name, (p.min_val + p.max_val) / 2), p) 
                        for p in cma_params
                    ], dtype=np.float32)
                else:
                    init_vec = np.array([
                        _to_unit(base_params.get(p.name, (p.min_val + p.max_val) / 2), p) 
                        for p in cma_params
                    ], dtype=np.float32)
            else:
                # Random initialization for restarts
                init_vec = np.random.uniform(0.1, 0.9, len(cma_params)).astype(np.float32)
            
            cma_opts = dict(cma_opts_base)
            cma_opts["seed"] = args.seed + restart_idx * 1000
            if base_popsize > 0:
                cma_opts["popsize"] = int(base_popsize * popsize_mult)
            elif popsize_mult != 1.0:
                # Will be set after es is created
                pass
            
            es = cma.CMAEvolutionStrategy(init_vec, sigma, cma_opts)
            
            # Apply popsize multiplier if needed
            if popsize_mult != 1.0 and base_popsize == 0:
                new_popsize = max(4, int(es.popsize * popsize_mult))
                es = cma.CMAEvolutionStrategy(init_vec, sigma, {**cma_opts, "popsize": new_popsize})
            
            if restart_idx == 0:
                print(f"  [Restart {restart_idx+1}/{num_restarts}] popsize={es.popsize}, sigma={sigma:.3f}")
            else:
                print(f"  [Restart {restart_idx+1}/{num_restarts}] popsize={es.popsize}, sigma={sigma:.3f} (random init)")
            
            restart_best = init_vec.copy()
            restart_best_loss = evaluate(init_vec, stage_base_params if do_grid_search else None, use_main_synth=True)
            
            evals_this_restart = 0
            max_evals_per_restart = max_evals // num_restarts if max_evals > 0 else 0
            
            while True:
                if args.allow_cma_stop and es.stop():
                    break
                solutions = es.ask()
                losses = evaluate_batch(solutions, stage_base_params if do_grid_search else None)
                es.tell(solutions, losses)
                evals_this_restart = es.countevals
                
                idx = int(np.argmin(losses))
                if losses[idx] < restart_best_loss:
                    restart_best_loss = losses[idx]
                    restart_best = np.array(solutions[idx], dtype=np.float32)
                
                if args.allow_cma_stop:
                    if restart_best_loss <= target_loss and evals_this_restart >= min_evals // num_restarts:
                        break
                    if max_evals_per_restart > 0 and evals_this_restart >= max_evals_per_restart:
                        break
                else:
                    if max_evals_per_restart > 0 and evals_this_restart >= max_evals_per_restart:
                        break
                
                if args.max_iter > 0 and getattr(es, "countiter", 0) >= args.max_iter // num_restarts:
                    break
                
                if evals_this_restart % max(1, int(es.popsize * 5)) == 0:
                    print(f"    evals={total_evals + evals_this_restart} loss={restart_best_loss:.6f}")
            
            total_evals += evals_this_restart
            
            # Update stage best if this restart found better solution
            if restart_best_loss < stage_best_loss:
                stage_best_loss = restart_best_loss
                stage_best = restart_best.copy()
                
                # Update base_params with best continuous values
                if do_grid_search and discrete_stage_params:
                    for u, spec in zip(stage_best, continuous_stage_params):
                        base_params[spec.name] = _from_unit(u, spec)
                else:
                    if stage_params is params:
                        best_u = stage_best.copy()
                    for u, spec in zip(stage_best, cma_params):
                        base_params[spec.name] = _from_unit(u, spec)
                
                if stage_best_loss < best_loss:
                    best_loss = stage_best_loss
                
                # Build full param vector for checkpoint
                full_vec = np.array([
                    _to_unit(base_params.get(p.name, p.min_val), p) for p in stage_params
                ], dtype=np.float32)
                save_checkpoint(
                    f"{stage_name}_best",
                    stage_best_loss,
                    total_evals,
                    stage_name,
                    full_vec,
                    stage_params,
                )
            
            print(f"  [Restart {restart_idx+1}] Finished: evals={evals_this_restart}, best_loss={restart_best_loss:.6f}")
            
            # Early termination if we hit target
            if stage_best_loss <= target_loss:
                print(f"  [Early stop] Target loss reached")
                break
        
        print(f"[{stage_name}] Completed: total_evals={total_evals}, best_loss={stage_best_loss:.6f}")

    final_params = dict(base_params)
    for u, spec in zip(best_u, params):
        final_params[spec.name] = _from_unit(u, spec)
    final_params["name"] = args.name

    channel.set_parameters(final_params)
    DrumPatchWriter.write_drum_patch(args.out, channel, args.name)
    print(f"Saved patch: {args.out}")

    if args.export_wav:
        channel.trigger(args.velocity)
        audio = channel.process(len(target_audio))
        audio = np.clip(audio, -1.0, 1.0)
        audio_int = (audio * 32767).astype(np.int16)
        wavfile.write(args.export_wav, synth_sr, audio_int)
        print(f"Saved WAV: {args.export_wav}")


if __name__ == "__main__":
    main()
