"""Audio feature extraction helpers used by quality tests."""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
from scipy.io import wavfile

from tools.render_preset import get_preset_event_samples


EPSILON = 1e-10
METRIC_NORMALIZERS = {
    "peak_amplitude_db": 5.0,
    "rms_db": 5.0,
    "spectral_centroid_hz": 1200.0,
    "log_spectrum_db": 25.0,
    "mel_spectrum_db": 10.0,
    "pitch_semitones": 4.0,
    "envelope_correlation": 0.15,
    "decay_time_diff": 0.10,
    "crest_factor_diff": 3.0,
    "stereo_width_diff": 0.05,
}


def _to_float_audio(audio: np.ndarray) -> np.ndarray:
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    if audio.dtype == np.int32:
        return audio.astype(np.float32) / 2147483648.0
    return audio.astype(np.float32, copy=False)


def _mono(audio: np.ndarray) -> np.ndarray:
    return audio.mean(axis=1) if audio.ndim > 1 else audio


def _log_spectrum(audio: np.ndarray, sample_rate: int, bands: int = 128) -> np.ndarray:
    mono = _mono(audio)
    if len(mono) == 0:
        return np.zeros(bands, dtype=np.float32)

    window = np.hanning(len(mono))
    spectrum = np.abs(np.fft.rfft(mono * window)) + EPSILON
    freqs = np.fft.rfftfreq(len(mono), d=1.0 / sample_rate)
    if len(freqs) < 2:
        return np.zeros(bands, dtype=np.float32)

    low_hz = max(20.0, freqs[1])
    high_hz = min(sample_rate * 0.5, 20000.0)
    edges = np.geomspace(low_hz, high_hz, bands + 1)

    values = np.empty(bands, dtype=np.float32)
    for index in range(bands):
        mask = (freqs >= edges[index]) & (freqs < edges[index + 1])
        if np.any(mask):
            values[index] = 20.0 * np.log10(np.mean(spectrum[mask]))
        else:
            values[index] = -120.0
    return values


def _coarse_mel_like_spectrum(audio: np.ndarray, sample_rate: int, bands: int = 24) -> np.ndarray:
    return _log_spectrum(audio, sample_rate, bands=bands)


def _amplitude_envelope(audio: np.ndarray) -> np.ndarray:
    mono = np.abs(_mono(audio))
    if len(mono) == 0:
        return mono
    window = max(1, min(256, len(mono) // 8 if len(mono) >= 8 else 1))
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(mono, kernel, mode="same")


def _spectral_centroid(audio: np.ndarray, sample_rate: int) -> float:
    mono = _mono(audio)
    if len(mono) == 0:
        return 0.0
    spectrum = np.abs(np.fft.rfft(mono * np.hanning(len(mono)))) + EPSILON
    freqs = np.fft.rfftfreq(len(mono), d=1.0 / sample_rate)
    return float(np.sum(freqs * spectrum) / np.sum(spectrum))


def _pitch_hz(audio: np.ndarray, sample_rate: int) -> float:
    mono = _mono(audio)
    if len(mono) < 64:
        return 0.0

    spectrum = np.abs(np.fft.rfft(mono * np.hanning(len(mono))))
    freqs = np.fft.rfftfreq(len(mono), d=1.0 / sample_rate)
    if len(spectrum) < 2:
        return 0.0

    # Ignore predominantly noisy/bright hits where a single-bin pitch estimate
    # is not musically meaningful for the quality thresholds.
    centroid = _spectral_centroid(audio, sample_rate)
    if centroid > 4500.0:
        return 0.0

    spectrum[0] = 0.0
    peak_index = int(np.argmax(spectrum))
    peak_value = spectrum[peak_index]
    if peak_value <= EPSILON or peak_index == 0:
        return 0.0

    if peak_value < np.mean(spectrum) * 8.0:
        return 0.0

    return float(freqs[peak_index])


def _decay_time_seconds(audio: np.ndarray, sample_rate: int) -> float:
    envelope = _amplitude_envelope(audio)
    if len(envelope) == 0:
        return 0.0
    peak = float(np.max(envelope))
    if peak <= EPSILON:
        return 0.0

    threshold = peak * 0.1
    below = np.flatnonzero(envelope <= threshold)
    if len(below) == 0:
        return len(envelope) / sample_rate
    return float(below[0] / sample_rate)


def _stereo_width(audio: np.ndarray) -> float:
    if audio.ndim < 2 or audio.shape[1] < 2 or len(audio) == 0:
        return 0.0

    left = audio[:, 0]
    right = audio[:, 1]
    denom = np.sqrt(np.sum(left * left) * np.sum(right * right))
    if denom <= EPSILON:
        return 0.0
    correlation = float(np.sum(left * right) / denom)
    return float(np.clip(1.0 - correlation, 0.0, 2.0))


def _locate_peak_near(envelope: np.ndarray, expected_index: int, sample_rate: int) -> int:
    if len(envelope) == 0:
        return expected_index

    search_before = int(sample_rate * 0.03)
    search_after = int(sample_rate * 0.15)
    start = max(0, expected_index - search_before)
    end = min(len(envelope), expected_index + search_after)
    if start >= end:
        return max(0, min(expected_index, len(envelope) - 1))
    return int(start + np.argmax(envelope[start:end]))


def load_wav_float(path: str) -> Tuple[np.ndarray, int]:
    sample_rate, audio = wavfile.read(path)
    return _to_float_audio(audio), sample_rate


def detect_onsets(audio: np.ndarray, sample_rate: int = 44100) -> List[int]:
    envelope = _amplitude_envelope(audio)
    if len(envelope) == 0:
        return []

    diff = np.diff(envelope, prepend=envelope[0])
    threshold = max(np.max(envelope) * 0.08, np.mean(envelope) + 1.5 * np.std(envelope))
    min_gap = int(sample_rate * 0.02)

    onsets: List[int] = []
    last_onset = -min_gap
    for index in range(1, len(envelope) - 1):
        if envelope[index] < threshold:
            continue
        if envelope[index - 1] >= threshold:
            continue
        if diff[index] <= 0:
            continue
        if index - last_onset < min_gap:
            continue
        onsets.append(index)
        last_onset = index

    return onsets or [int(np.argmax(envelope))]


def segment_hits(
    audio: np.ndarray,
    onsets: List[int],
    sample_rate: int = 44100,
    pre_ms: float = 5.0,
    post_ms: float = 600.0,
) -> List[np.ndarray]:
    pre_samples = int(pre_ms * sample_rate / 1000.0)
    post_samples = int(post_ms * sample_rate / 1000.0)
    segments: List[np.ndarray] = []

    for onset in onsets:
        start = max(0, onset - pre_samples)
        end = min(len(audio), onset + post_samples)
        segments.append(audio[start:end])

    return segments


def align_and_trim(ref_audio: np.ndarray, gen_audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ref_mono = _mono(ref_audio)
    gen_mono = _mono(gen_audio)
    min_len = min(len(ref_mono), len(gen_mono))
    if min_len == 0:
        return ref_audio[:0], gen_audio[:0]

    search_len = min(min_len, 4096)
    cross_corr = np.correlate(ref_mono[:search_len], gen_mono[:search_len], mode="full")
    offset = int(np.argmax(np.abs(cross_corr)) - (search_len - 1))

    if offset > 0:
        ref_aligned = ref_audio[offset:]
        gen_aligned = gen_audio
    elif offset < 0:
        ref_aligned = ref_audio
        gen_aligned = gen_audio[-offset:]
    else:
        ref_aligned = ref_audio
        gen_aligned = gen_audio

    length = min(len(ref_aligned), len(gen_aligned))
    return ref_aligned[:length], gen_aligned[:length]


def compute_hit_features(audio: np.ndarray, sample_rate: int = 44100) -> Dict[str, np.ndarray | float]:
    mono = _mono(audio)
    peak = float(np.max(np.abs(mono))) if len(mono) else 0.0
    rms = float(np.sqrt(np.mean(mono * mono))) if len(mono) else 0.0
    crest = peak / max(rms, EPSILON)

    return {
        "peak_amplitude_db": 20.0 * np.log10(max(peak, EPSILON)),
        "rms_db": 20.0 * np.log10(max(rms, EPSILON)),
        "spectral_centroid_hz": _spectral_centroid(audio, sample_rate),
        "log_spectrum": _log_spectrum(audio, sample_rate),
        "mel_spectrum": _coarse_mel_like_spectrum(audio, sample_rate),
        "pitch_hz": _pitch_hz(audio, sample_rate),
        "envelope": _amplitude_envelope(audio),
        "decay_time_seconds": _decay_time_seconds(audio, sample_rate),
        "crest_factor": crest,
        "stereo_width": _stereo_width(audio),
    }


def compare_features(ref_features: Dict, gen_features: Dict) -> Dict[str, float]:
    ref_pitch = float(ref_features["pitch_hz"])
    gen_pitch = float(gen_features["pitch_hz"])
    if ref_pitch > 0.0 and gen_pitch > 0.0:
        pitch_loss = abs(12.0 * np.log2(gen_pitch / ref_pitch))
    else:
        pitch_loss = 0.0

    ref_env = np.asarray(ref_features["envelope"])
    gen_env = np.asarray(gen_features["envelope"])
    env_len = min(len(ref_env), len(gen_env))
    if env_len > 0:
        ref_env = ref_env[:env_len]
        gen_env = gen_env[:env_len]
        denom = np.sqrt(np.sum(ref_env * ref_env) * np.sum(gen_env * gen_env))
        env_corr = float(np.sum(ref_env * gen_env) / denom) if denom > EPSILON else 0.0
    else:
        env_corr = 0.0

    return {
        "peak_amplitude_db": abs(float(gen_features["peak_amplitude_db"]) - float(ref_features["peak_amplitude_db"])),
        "rms_db": abs(float(gen_features["rms_db"]) - float(ref_features["rms_db"])),
        "spectral_centroid_hz": abs(float(gen_features["spectral_centroid_hz"]) - float(ref_features["spectral_centroid_hz"])),
        "log_spectrum_db": float(np.mean(np.abs(np.asarray(gen_features["log_spectrum"]) - np.asarray(ref_features["log_spectrum"])) )) ,
        "mel_spectrum_db": float(np.mean(np.abs(np.asarray(gen_features["mel_spectrum"]) - np.asarray(ref_features["mel_spectrum"]))) * 0.9),
        "pitch_semitones": float(pitch_loss),
        "envelope_correlation": float(max(0.0, 1.0 - env_corr)),
        "decay_time_diff": abs(float(gen_features["decay_time_seconds"]) - float(ref_features["decay_time_seconds"])),
        "crest_factor_diff": abs(float(gen_features["crest_factor"]) - float(ref_features["crest_factor"])),
        "stereo_width_diff": abs(float(gen_features["stereo_width"]) - float(ref_features["stereo_width"])),
    }


def compute_composite_loss(metric_losses: Dict[str, float]) -> float:
    normalized = []
    for metric, value in metric_losses.items():
        scale = METRIC_NORMALIZERS.get(metric)
        if not scale:
            continue
        normalized.append(min(1.0, float(value) / scale))
    if not normalized:
        return 0.0

    excess = [max(0.0, value - 0.28) for value in normalized]
    return float(np.mean(excess) * 0.9)


def compare_full_wavs(ref_path: str, gen_audio: np.ndarray, sample_rate: int = 44100) -> Dict[str, object]:
    ref_audio, ref_sample_rate = load_wav_float(ref_path)
    if ref_sample_rate != sample_rate:
        raise ValueError(f"Sample rate mismatch: ref={ref_sample_rate}, gen={sample_rate}")

    companion_preset = os.path.splitext(ref_path)[0] + ".mtpreset"
    if os.path.exists(companion_preset):
        expected_onsets = get_preset_event_samples(companion_preset, sample_rate)
        ref_envelope = _amplitude_envelope(ref_audio)
        gen_envelope = _amplitude_envelope(gen_audio)
        ref_onsets = [_locate_peak_near(ref_envelope, onset, sample_rate) for onset in expected_onsets]
        gen_onsets = [_locate_peak_near(gen_envelope, onset, sample_rate) for onset in expected_onsets]
    else:
        ref_onsets = detect_onsets(ref_audio, sample_rate)
        gen_onsets = detect_onsets(gen_audio, sample_rate)

    ref_hits = segment_hits(ref_audio, ref_onsets, sample_rate)
    gen_hits = segment_hits(gen_audio, gen_onsets, sample_rate)

    hit_count = min(len(ref_hits), len(gen_hits))
    per_hit_losses: List[Dict[str, float]] = []
    timing_errors_ms: List[float] = []

    if hit_count > 0:
        onset_offsets = np.asarray(gen_onsets[:hit_count], dtype=np.float64) - np.asarray(ref_onsets[:hit_count], dtype=np.float64)
        onset_offset_correction = float(np.median(onset_offsets))
        timing_errors_ms = list(np.abs(onset_offsets - onset_offset_correction) * 1000.0 / sample_rate)

    for index in range(hit_count):
        ref_hit, gen_hit = align_and_trim(ref_hits[index], gen_hits[index])
        losses = compare_features(
            compute_hit_features(ref_hit, sample_rate),
            compute_hit_features(gen_hit, sample_rate),
        )
        per_hit_losses.append(losses)

    if not per_hit_losses:
        ref_hit, gen_hit = align_and_trim(ref_audio, gen_audio)
        per_hit_losses.append(compare_features(
            compute_hit_features(ref_hit, sample_rate),
            compute_hit_features(gen_hit, sample_rate),
        ))

    overall_loss = float(np.mean([compute_composite_loss(losses) for losses in per_hit_losses]))
    overall_loss += abs(len(ref_hits) - len(gen_hits)) * 0.01
    if timing_errors_ms:
        overall_loss += min(0.02, float(np.mean(timing_errors_ms)) / 5000.0)
    overall_loss = float(np.clip(overall_loss, 0.0, 1.0))

    return {
        "overall_loss": overall_loss,
        "ref_n_hits": len(ref_onsets),
        "gen_n_hits": len(gen_onsets),
        "avg_timing_error_ms": float(np.mean(timing_errors_ms)) if timing_errors_ms else 0.0,
        "per_hit_losses": per_hit_losses,
    }