#!/usr/bin/env python3
"""
PO-32 Modem Decoder — WAV/audio → patch data pipeline.

Decodes FSK audio from PO-32 Tonic into patch parameters, patterns, and state.

Decoder pipeline (bottom to top):
1. FSK demodulation (zero-crossing → half-periods → groups → n-values → bits → bytes)
2. Bit reversal
3. CRC additive cipher descrambling
4. TLV parsing (tag + length + data + CRC)
5. Parameter denormalization (uint16 → 0-1 → display values → synth params)
"""

import struct
import math
import wave
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field


# ==============================================================================
# Protocol Constants
# ==============================================================================

# FSK parameters
SAMPLE_RATE = 44100
FREQ_HIGH = 3675.0   # ~6 samples per half-period
FREQ_LOW = 1837.5    # ~12 samples per half-period

# CRC
CRC_INIT = 0x1D0F

# Modem header (not scrambled)
MODEM_HEADER = bytes([0x14, 0x19, 0x9D, 0xCF])

# TLV tags (little-endian uint16)
TAG_PATCH   = 0x37B2
TAG_PATTERN = 0xD022
TAG_STATE   = 0x505A
TAG_TRAILER = 0x71C3

# SDK frequency normalization constants
FREQ_VALUE_C4 = 0.4725600599127338
OCTAVE_STEP = 0.1003433318879937

# Parameter serialization order (21 params × 2 bytes = 42 bytes per patch)
PARAM_NAMES = [
    'OscWave', 'OscFreq', 'OscAtk', 'OscDcy', 'ModMode', 'ModRate', 'ModAmt',
    'NFilMod', 'NFilFrq', 'NFilQ', 'NEnvMod', 'NEnvAtk', 'NEnvDcy', 'Mix',
    'DistAmt', 'EQFreq', 'EQGain', 'Level', 'OscVel', 'NVel', 'ModVel'
]

# Discrete parameter mappings
DISCRETE_PARAMS = {
    'OscWave': ['Sine', 'Triangle', 'Saw'],
    'ModMode': ['Decay', 'Sine', 'Noise'],
    'NFilMod': ['LP', 'BP', 'HP'],
    'NEnvMod': ['Exp', 'Linear', 'Mod'],
}


# ==============================================================================
# Low-Level Utilities
# ==============================================================================

def crc16_ccitt_update(crc: int, byte: int) -> int:
    """Update CRC-16-CCITT with one byte."""
    x = ((crc >> 8) ^ byte) & 0xFF
    x ^= (x >> 4)
    return ((crc << 8) ^ (x << 12) ^ (x << 5) ^ x) & 0xFFFF


def reverse_bits(b: int) -> int:
    """Reverse bits in a byte."""
    result = 0
    for i in range(8):
        if b & (1 << i):
            result |= 1 << (7 - i)
    return result


def descramble(data: bytes, crc_init: int = CRC_INIT) -> Tuple[bytes, int]:
    """Descramble modem data using CRC additive cipher."""
    crc = crc_init
    result = bytearray()
    for sb in data:
        original = (sb - (crc & 0xFF)) & 0xFF
        result.append(original)
        crc = crc16_ccitt_update(crc, original)
    return bytes(result), crc


def load_wav(filename: str) -> Tuple[np.ndarray, int]:
    """Load WAV file as float64 mono samples.

    Handles stereo by taking the first (left) channel.
    Supports PCM 8/16/24/32-bit and IEEE float 32-bit WAV files.
    """
    try:
        with wave.open(filename, 'r') as wf:
            rate = wf.getframerate()
            nchannels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())
            if sampwidth == 2:
                samples = np.frombuffer(frames, dtype=np.int16).astype(np.float64) / 32768.0
            elif sampwidth == 3:
                # 24-bit PCM: unpack 3-byte little-endian samples
                n = len(frames) // 3
                raw = np.frombuffer(frames, dtype=np.uint8).reshape(n, 3)
                i32 = (raw[:, 0].astype(np.int32)
                       | (raw[:, 1].astype(np.int32) << 8)
                       | (raw[:, 2].astype(np.int32) << 16))
                # Sign-extend from 24 bits
                i32[i32 >= 0x800000] -= 0x1000000
                samples = i32.astype(np.float64) / 8388608.0
            elif sampwidth == 4:
                samples = np.frombuffer(frames, dtype=np.int32).astype(np.float64) / 2147483648.0
            else:
                samples = np.frombuffer(frames, dtype=np.uint8).astype(np.float64) / 128.0 - 1.0
            # Deinterleave stereo → take first channel
            if nchannels > 1:
                samples = samples[0::nchannels]
        return samples, rate
    except wave.Error:
        # Python's wave module doesn't support IEEE float (format 3) or
        # extensible WAV. Fall back to manual header parsing.
        pass

    import struct as _struct

    with open(filename, 'rb') as f:
        riff = f.read(12)
        if riff[:4] != b'RIFF' or riff[8:12] != b'WAVE':
            raise ValueError("Not a WAV file")

        fmt_code = None
        nchannels = 1
        rate = 44100
        bits = 16
        data_bytes = b''

        while True:
            chunk_hdr = f.read(8)
            if len(chunk_hdr) < 8:
                break
            chunk_id = chunk_hdr[:4]
            chunk_size = _struct.unpack('<I', chunk_hdr[4:8])[0]

            if chunk_id == b'fmt ':
                fmt_data = f.read(chunk_size)
                fmt_code = _struct.unpack('<H', fmt_data[:2])[0]
                nchannels = _struct.unpack('<H', fmt_data[2:4])[0]
                rate = _struct.unpack('<I', fmt_data[4:8])[0]
                bits = _struct.unpack('<H', fmt_data[14:16])[0]
            elif chunk_id == b'data':
                data_bytes = f.read(chunk_size)
            else:
                # Skip unknown chunks (pad to even boundary)
                f.seek(chunk_size + (chunk_size % 2), 1)

        if fmt_code is None or len(data_bytes) == 0:
            raise ValueError("WAV file missing fmt or data chunk")

        if fmt_code == 3:  # IEEE float
            if bits == 32:
                samples = np.frombuffer(data_bytes, dtype=np.float32).astype(np.float64)
            elif bits == 64:
                samples = np.frombuffer(data_bytes, dtype=np.float64).copy()
            else:
                raise ValueError(f"Unsupported IEEE float bit depth: {bits}")
        elif fmt_code == 1:  # PCM (shouldn't get here, but just in case)
            if bits == 16:
                samples = np.frombuffer(data_bytes, dtype=np.int16).astype(np.float64) / 32768.0
            elif bits == 24:
                n = len(data_bytes) // 3
                raw = np.frombuffer(data_bytes, dtype=np.uint8).reshape(n, 3)
                i32 = (raw[:, 0].astype(np.int32)
                       | (raw[:, 1].astype(np.int32) << 8)
                       | (raw[:, 2].astype(np.int32) << 16))
                i32[i32 >= 0x800000] -= 0x1000000
                samples = i32.astype(np.float64) / 8388608.0
            elif bits == 32:
                samples = np.frombuffer(data_bytes, dtype=np.int32).astype(np.float64) / 2147483648.0
            else:
                samples = np.frombuffer(data_bytes, dtype=np.uint8).astype(np.float64) / 128.0 - 1.0
        elif fmt_code == 0xFFFE:  # EXTENSIBLE — check sub-format GUID
            # The actual format is in bytes 24-40 of fmt data
            if len(fmt_data) >= 40:
                sub_fmt = _struct.unpack('<H', fmt_data[24:26])[0]
                if sub_fmt == 3 and bits == 32:
                    samples = np.frombuffer(data_bytes, dtype=np.float32).astype(np.float64)
                elif sub_fmt == 1 and bits == 16:
                    samples = np.frombuffer(data_bytes, dtype=np.int16).astype(np.float64) / 32768.0
                elif sub_fmt == 1 and bits == 24:
                    n = len(data_bytes) // 3
                    raw = np.frombuffer(data_bytes, dtype=np.uint8).reshape(n, 3)
                    i32 = (raw[:, 0].astype(np.int32)
                           | (raw[:, 1].astype(np.int32) << 8)
                           | (raw[:, 2].astype(np.int32) << 16))
                    i32[i32 >= 0x800000] -= 0x1000000
                    samples = i32.astype(np.float64) / 8388608.0
                elif sub_fmt == 1 and bits == 32:
                    samples = np.frombuffer(data_bytes, dtype=np.int32).astype(np.float64) / 2147483648.0
                else:
                    raise ValueError(f"Unsupported WAVEFORMATEXTENSIBLE sub-format {sub_fmt}, bits={bits}")
            else:
                raise ValueError("WAVEFORMATEXTENSIBLE header too short")
        else:
            raise ValueError(f"Unsupported WAV format code: {fmt_code}")

        # Deinterleave stereo → take first channel
        if nchannels > 1:
            samples = samples[0::nchannels]

    return samples, rate


# ==============================================================================
# FSK Demodulation
# ==============================================================================

def demodulate_fsk(samples: np.ndarray, sample_rate: int = SAMPLE_RATE,
                   threshold_ratio: float = 0.1) -> Optional[bytes]:
    """Demodulate FSK audio signal back to raw modem bytes.

    Algorithm:
    1. Detect zero crossings in the audio
    2. Measure half-period lengths between crossings
    3. Classify as HIGH (~6 samples) or LOW (~12 samples)
    4. Count consecutive HIGHs between LOWs → group sizes
    5. Skip preamble, decode data groups to n-values → bits → bytes

    Args:
        samples: Audio samples (float, mono)
        sample_rate: Audio sample rate (default 44100)
        threshold_ratio: Amplitude threshold as fraction of peak (ignores silence)

    Returns:
        Raw modem bytes (bit-reversed, before descrambling), or None if decode fails
    """
    if len(samples) == 0:
        return None

    # Handle stereo by taking first channel
    if samples.ndim > 1:
        samples = samples[:, 0]

    # Normalize
    peak = np.max(np.abs(samples))
    if peak < 0.001:
        return None  # Too quiet
    samples = samples / peak

    # Find the active region (above noise threshold)
    amplitude_threshold = threshold_ratio
    envelope = _compute_envelope(samples, window=256)
    active = envelope > amplitude_threshold

    # Find first and last active sample
    active_indices = np.where(active)[0]
    if len(active_indices) < 1000:
        return None  # Too short
    start_idx = max(0, active_indices[0] - 100)
    end_idx = min(len(samples), active_indices[-1] + 100)
    samples = samples[start_idx:end_idx]

    # Find zero crossings
    half_periods = _measure_half_periods(samples)
    if len(half_periods) < 100:
        return None  # Not enough data

    # Expected half-period lengths at sample_rate
    hp_high = sample_rate / (2 * FREQ_HIGH)  # ~6.0
    hp_low = sample_rate / (2 * FREQ_LOW)    # ~12.0

    # Classify threshold: geometric mean
    hp_threshold = math.sqrt(hp_high * hp_low)  # ~8.49

    # Classify half-periods as HIGH or LOW
    classifications = []
    for hp_len in half_periods:
        if hp_len < hp_threshold:
            classifications.append('H')
        else:
            classifications.append('L')

    # Extract groups: count consecutive HIGHs between LOWs
    groups = _extract_groups(classifications)
    if len(groups) < 50:
        return None  # Too few groups

    # Find preamble end and data start
    data_start = _find_data_start(groups)
    if data_start is None:
        return None

    data_groups = groups[data_start:]

    # Convert groups to n-values: n = (group_size - 4) / 6
    n_values = []
    for g in data_groups:
        n = (g - 4) / 6
        n_rounded = round(n)
        if n_rounded < 0:
            n_rounded = 0
        n_values.append(n_rounded)

    # Convert n-values to bitstream: each n produces n ones followed by one zero
    bits = []
    for n in n_values:
        bits.extend([1] * n)
        bits.append(0)

    # Assemble bits MSB-first into bytes
    raw_bytes = _bits_to_bytes(bits)

    if len(raw_bytes) < 5:
        return None

    return bytes(raw_bytes)


def _compute_envelope(samples: np.ndarray, window: int = 256) -> np.ndarray:
    """Compute amplitude envelope using a sliding window."""
    abs_samples = np.abs(samples)
    kernel = np.ones(window) / window
    envelope = np.convolve(abs_samples, kernel, mode='same')
    return envelope


def _measure_half_periods(samples: np.ndarray) -> List[float]:
    """Find zero crossings and measure distances between them."""
    signs = np.sign(samples)
    for i in range(1, len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i - 1]

    crossings = np.where(np.diff(signs) != 0)[0]

    if len(crossings) < 2:
        return []

    # Refine crossing positions using linear interpolation
    refined = []
    for idx in crossings:
        if idx + 1 < len(samples):
            s0 = samples[idx]
            s1 = samples[idx + 1]
            if s0 != s1:
                frac = -s0 / (s1 - s0)
                refined.append(idx + frac)
            else:
                refined.append(float(idx))
        else:
            refined.append(float(idx))

    half_periods = []
    for i in range(1, len(refined)):
        hp = refined[i] - refined[i - 1]
        half_periods.append(hp)

    return half_periods


def _extract_groups(classifications: List[str]) -> List[int]:
    """Count consecutive HIGHs between LOWs to get group sizes."""
    groups = []
    current_count = 0

    for c in classifications:
        if c == 'H':
            current_count += 1
        else:
            if current_count > 0:
                groups.append(current_count)
            current_count = 0

    if current_count > 0:
        groups.append(current_count)

    return groups


def _find_data_start(groups: List[int], min_preamble: int = 20) -> Optional[int]:
    """Find where preamble ends and data begins.

    The preamble consists of groups of size 10 (n=1, representing bit pattern '10').
    Data starts when group sizes deviate from 10.
    """
    preamble_count = 0

    for i, g in enumerate(groups):
        if 8 <= g <= 12:
            preamble_count += 1
        else:
            if preamble_count >= min_preamble:
                return i
            preamble_count = 0

    return None


def _bits_to_bytes(bits: List[int]) -> List[int]:
    """Assemble bitstream to bytes (MSB-first)."""
    result = []
    for i in range(0, len(bits) - 7, 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        result.append(byte)
    return result


# ==============================================================================
# Packet Decoder
# ==============================================================================

@dataclass
class DecodedPatch:
    """A decoded drum patch with normalized and synth-ready parameters."""
    prefix: int                          # TLV prefix byte: 0x1X=left, 0x2X=right
    drum_index: int                      # 0-based drum index (from prefix low nibble)
    is_left: bool                        # True if left (A) patch
    raw_params: bytes                    # Raw 42-byte param data
    normalized: Dict[str, float] = field(default_factory=dict)  # 0-1 values
    synth_params: Dict[str, Any] = field(default_factory=dict)  # Ready for set_parameters()


@dataclass
class DecodedPattern:
    """A decoded PO-32 pattern with per-step trigger data."""
    pattern_index: int                   # 0-based pattern index (0-15)
    raw_data: bytes                      # Raw 211-byte pattern data
    triggers: List[int] = field(default_factory=list)  # 16 step trigger bitmasks (bit N = drum N)
    bank: int = 0                        # Detected bank: 0 = drums 0-7, 1 = drums 8-15
    active_drums: List[int] = field(default_factory=list)  # List of drum indices with triggers


@dataclass
class DecodedPreset:
    """Complete decoded PO-32 preset."""
    left_patches: Dict[int, DecodedPatch] = field(default_factory=dict)   # drum_idx → patch
    right_patches: Dict[int, DecodedPatch] = field(default_factory=dict)  # drum_idx → patch
    patterns: List[bytes] = field(default_factory=list)
    decoded_patterns: List[DecodedPattern] = field(default_factory=list)  # Decoded pattern data
    state: Optional[bytes] = None
    error: Optional[str] = None


def decode_modem_bytes(raw_bytes: bytes) -> DecodedPreset:
    """Decode raw modem bytes (from FSK demod) into a preset structure.

    Pipeline: bit-reverse → verify header → descramble → parse TLV → extract patches
    """
    preset = DecodedPreset()

    if len(raw_bytes) < 8:
        preset.error = "Data too short"
        return preset

    # Step 1: Bit-reverse all bytes
    modem_data = bytes(reverse_bits(b) for b in raw_bytes)

    # Step 2: Verify modem header
    if modem_data[:4] != MODEM_HEADER:
        header_pos = _find_header(modem_data)
        if header_pos is None:
            preset.error = f"Modem header not found (got {modem_data[:4].hex()})"
            return preset
        modem_data = modem_data[header_pos:]

    # Step 3: Descramble (skip 4-byte header)
    descrambled, final_crc = descramble(modem_data[4:])

    # Step 4: Parse TLV fields
    pos = 0
    while pos + 5 <= len(descrambled):
        tag = descrambled[pos] | (descrambled[pos + 1] << 8)
        flen = descrambled[pos + 2]

        if pos + 3 + flen + 2 > len(descrambled):
            break

        fdata = descrambled[pos + 3:pos + 3 + flen]

        # Field CRC (consumed but not verified — used for packet integrity)
        pos += 3 + flen + 2

        if tag == TAG_PATCH and flen >= 1:
            prefix = fdata[0]
            param_data = fdata[1:]
            drum_idx = prefix & 0x0F
            is_left = (prefix & 0xF0) == 0x10

            patch = DecodedPatch(
                prefix=prefix,
                drum_index=drum_idx,
                is_left=is_left,
                raw_params=param_data,
            )

            # Deserialize 21 uint16 LE values to normalized params
            if len(param_data) >= 42:
                patch.normalized = _deserialize_params(param_data)
                patch.synth_params = normalized_to_synth_params(patch.normalized)

            if is_left:
                preset.left_patches[drum_idx] = patch
            else:
                preset.right_patches[drum_idx] = patch

        elif tag == TAG_PATTERN:
            preset.patterns.append(fdata)
            decoded_pat = decode_pattern_data(fdata)
            if decoded_pat:
                preset.decoded_patterns.append(decoded_pat)

        elif tag == TAG_STATE:
            preset.state = fdata

        elif tag == TAG_TRAILER:
            break

    return preset


def _find_header(data: bytes, max_offset: int = 50) -> Optional[int]:
    """Search for modem header in the first max_offset bytes."""
    for i in range(min(max_offset, len(data) - 4)):
        if data[i:i + 4] == MODEM_HEADER:
            return i
    return None


def _deserialize_params(param_data: bytes) -> Dict[str, float]:
    """Deserialize 42 bytes (21 × uint16 LE) to normalized 0-1 parameter dict."""
    result = {}
    for i, name in enumerate(PARAM_NAMES):
        if i * 2 + 2 <= len(param_data):
            u16 = struct.unpack_from('<H', param_data, i * 2)[0]
            result[name] = u16 / 65536.0
        else:
            result[name] = 0.0
    return result


# ==============================================================================
# Inverse Normalization (0-1 → display/synth values)
# ==============================================================================

def norm_to_freq(n: float) -> float:
    """Convert normalized 0-1 to frequency (Hz).
    Inverse of freq_to_norm: hz = 523.25 × 2^((n - FREQ_VALUE_C4) / OCTAVE_STEP)
    """
    return 523.25 * 2.0 ** ((n - FREQ_VALUE_C4) / OCTAVE_STEP)


def norm_to_decay(n: float) -> float:
    """Convert normalized 0-1 to decay time (ms).
    Inverse of decay_to_norm: t = 10^(3n - 2) seconds → ms × 1000
    """
    if n <= 0:
        return 0.01
    t_sec = 10.0 ** (3.0 * n - 2.0)
    return t_sec * 1000.0


def norm_to_attack(n: float) -> float:
    """Convert normalized 0-1 to attack time (ms).
    Formula: t = n × 10^((12n - 7) / 5) seconds → ms × 1000
    """
    if n <= 0:
        return 0.0
    t_sec = n * 10.0 ** ((12.0 * n - 7.0) / 5.0)
    return t_sec * 1000.0


def norm_to_modrate_sine(n: float) -> float:
    """Convert normalized 0-1 to ModRate Sine frequency (Hz)."""
    if n <= 0:
        return 0.0
    return 2000.0 * n * 10.0 ** (3.5 * (n - 1.0))


def norm_to_modrate_noise(n: float) -> float:
    """Convert normalized 0-1 to ModRate Noise frequency (Hz)."""
    if n <= 0:
        return 0.0
    return 20000.0 * n * 10.0 ** (4.0 * (n - 1.0))


def norm_to_modrate_decay(n: float) -> float:
    """Convert normalized 0-1 to ModRate Decay time (ms)."""
    if n <= 0:
        return 10000.0
    return (1030.0 - 1000.0 * n) / (3.0 * n)


def norm_to_modamt(n: float, mod_mode: str) -> float:
    """Convert normalized 0-1 to ModAmt (semitones)."""
    max_sm = 96.0 if mod_mode == 'Decay' else 48.0
    offset = n - 0.5
    sign = 1.0 if offset >= 0 else -1.0
    return sign * max_sm * (2.0 * abs(offset)) ** 2


def norm_to_level(n: float) -> float:
    """Convert normalized 0-1 to Level (dB).
    Formula: dB = 60n - 49 - 1/n
    """
    if n <= 0.001:
        return -100.0
    return 60.0 * n - 49.0 - 1.0 / n


def norm_to_nfilq(n: float) -> float:
    """Convert normalized 0-1 to noise filter Q."""
    if n <= 0:
        return 0.1
    return 0.1 * (10001.0 / 0.1) ** n


def norm_to_discrete(n: float, options: List[str]) -> Tuple[int, str]:
    """Convert normalized 0-1 to discrete parameter value."""
    num = len(options)
    idx = round(n * (num - 1))
    idx = max(0, min(num - 1, idx))
    return idx, options[idx]


# ==============================================================================
# Normalized → Synth Parameters
# ==============================================================================

# Maps PO-32 discrete strings to synth enum values
STR_TO_WAVEFORM = {'Sine': 0, 'Triangle': 1, 'Saw': 2}
STR_TO_PITCHMOD = {'Decay': 0, 'Sine': 1, 'Noise': 2}
STR_TO_FILTERMODE = {'LP': 0, 'BP': 1, 'HP': 2}
STR_TO_ENVMODE = {'Exp': 0, 'Linear': 1, 'Mod': 2}


def normalized_to_synth_params(norm: Dict[str, float]) -> Dict[str, Any]:
    """Convert normalized 0-1 PO-32 parameters to Pythonic synth parameter dict.

    Returns a dict compatible with DrumChannel.set_parameters().
    """
    params = {}

    # --- Discrete parameters ---
    wf_idx, wf_str = norm_to_discrete(norm.get('OscWave', 0.0), ['Sine', 'Triangle', 'Saw'])
    params['osc_waveform'] = wf_idx

    mm_idx, mm_str = norm_to_discrete(norm.get('ModMode', 0.0), ['Decay', 'Sine', 'Noise'])
    params['pitch_mod_mode'] = mm_idx
    mod_mode = mm_str

    fm_idx, fm_str = norm_to_discrete(norm.get('NFilMod', 0.0), ['LP', 'BP', 'HP'])
    params['noise_filter_mode'] = fm_idx

    em_idx, em_str = norm_to_discrete(norm.get('NEnvMod', 0.0), ['Exp', 'Linear', 'Mod'])
    params['noise_envelope_mode'] = em_idx
    nenv_mode = em_str

    # --- Frequencies ---
    params['osc_frequency'] = norm_to_freq(norm.get('OscFreq', 0.5))
    params['noise_filter_freq'] = norm_to_freq(norm.get('NFilFrq', 0.5))
    params['eq_frequency'] = norm_to_freq(norm.get('EQFreq', 0.5))

    # --- Attack times ---
    params['osc_attack'] = norm_to_attack(norm.get('OscAtk', 0.0))

    noise_atk = norm_to_attack(norm.get('NEnvAtk', 0.0))
    if nenv_mode == 'Linear':
        noise_atk /= 1.5
    params['noise_attack'] = noise_atk

    # --- Decay times ---
    params['osc_decay'] = norm_to_decay(norm.get('OscDcy', 0.5))

    noise_dcy = norm_to_decay(norm.get('NEnvDcy', 0.5))
    if nenv_mode == 'Linear':
        noise_dcy /= 1.5
    params['noise_decay'] = noise_dcy

    # --- Modulation ---
    mod_rate_norm = norm.get('ModRate', 0.5)
    if mod_mode == 'Sine':
        params['pitch_mod_rate'] = norm_to_modrate_sine(mod_rate_norm)
    elif mod_mode == 'Noise':
        params['pitch_mod_rate'] = norm_to_modrate_noise(mod_rate_norm)
    else:
        params['pitch_mod_rate'] = norm_to_modrate_decay(mod_rate_norm)

    params['pitch_mod_amount'] = norm_to_modamt(norm.get('ModAmt', 0.5), mod_mode)

    # --- Noise filter Q ---
    params['noise_filter_q'] = norm_to_nfilq(norm.get('NFilQ', 0.0))

    # --- Mix ---
    noise_pct = norm.get('Mix', 0.0)
    params['osc_noise_mix'] = 1.0 - noise_pct

    # --- Distortion ---
    params['distortion'] = norm.get('DistAmt', 0.0)

    # --- EQ Gain ---
    params['eq_gain_db'] = norm.get('EQGain', 0.5) * 80.0 - 40.0

    # --- Level ---
    params['level_db'] = norm_to_level(norm.get('Level', 0.5))

    # --- Velocity sensitivity ---
    params['osc_vel_sensitivity'] = norm.get('OscVel', 0.0)
    params['noise_vel_sensitivity'] = norm.get('NVel', 0.0)
    params['mod_vel_sensitivity'] = norm.get('ModVel', 0.0)

    # PO-32 protocol has no pitch offset — reset to zero on import
    params['pitch_semitones'] = 0.0

    return params


# ==============================================================================
# Pattern Decoder
# ==============================================================================

def decode_pattern_data(raw_data: bytes) -> Optional[DecodedPattern]:
    """Decode a raw pattern TLV data block into step triggers.

    PO-32 pattern format (211 bytes):
    - Byte 0: Pattern index (0-15)
    - Bytes 1-16: Per-step trigger bitmask (16 steps)
      Each byte is a bitmask: bit N set = drum N triggers at this step
    - Bytes 17-210: Per-step velocity/variation data (not decoded yet)

    Args:
        raw_data: Raw 211-byte pattern data from TLV

    Returns:
        DecodedPattern with trigger data, or None if invalid
    """
    if len(raw_data) < 17:
        return None

    pattern_index = raw_data[0]
    triggers = list(raw_data[1:17])

    # Determine which drums are active
    active_mask = 0
    for t in triggers:
        active_mask |= t

    active_drums = [bit for bit in range(16) if active_mask & (1 << bit)]

    # Detect bank
    has_bank0 = any(d < 8 for d in active_drums)
    has_bank1 = any(d >= 8 for d in active_drums)

    if has_bank1 and not has_bank0:
        bank = 1
    elif has_bank0 and not has_bank1:
        bank = 0
    else:
        bank = 0

    return DecodedPattern(
        pattern_index=pattern_index,
        raw_data=raw_data,
        triggers=triggers,
        bank=bank,
        active_drums=active_drums,
    )


def get_pattern_triggers_for_bank(pattern: DecodedPattern, bank: int = 0) -> Dict[int, List[bool]]:
    """Extract per-drum trigger arrays for a specific bank from a decoded pattern.

    Args:
        pattern: Decoded pattern data
        bank: 0 for drums 0-7, 1 for drums 8-15

    Returns:
        Dict mapping drum channel (0-7) to list of 16 booleans (step triggers)
    """
    result = {}
    bank_offset = bank * 8

    for drum_ch in range(8):
        drum_idx = drum_ch + bank_offset
        bit_mask = 1 << drum_idx
        steps = [(t & bit_mask) != 0 for t in pattern.triggers]
        result[drum_ch] = steps

    return result


def get_pattern_summary(pattern: DecodedPattern) -> str:
    """Get a human-readable summary of a decoded pattern."""
    if not pattern.triggers:
        return "Empty pattern"

    total_triggers = sum(bin(t).count('1') for t in pattern.triggers)
    active_steps = sum(1 for t in pattern.triggers if t != 0)

    if total_triggers == 0:
        return "Empty pattern"

    return (f"Pattern {pattern.pattern_index + 1}: "
            f"{active_steps}/16 steps active, "
            f"{len(pattern.active_drums)} drums, bank {pattern.bank}")


# ==============================================================================
# High-Level Decode Functions
# ==============================================================================

def decode_wav_file(filename: str) -> DecodedPreset:
    """Decode a PO-32 transfer WAV file into patches and patterns.

    Args:
        filename: Path to WAV file containing PO-32 modem audio

    Returns:
        DecodedPreset with extracted patches and patterns
    """
    try:
        samples, rate = load_wav(filename)
    except Exception as e:
        preset = DecodedPreset()
        preset.error = f"Failed to load WAV: {e}"
        return preset

    return decode_audio_samples(samples, rate)


def decode_audio_samples(samples: np.ndarray,
                         sample_rate: int = SAMPLE_RATE) -> DecodedPreset:
    """Decode FSK audio samples into patches and patterns.

    Args:
        samples: Audio samples (float, mono or stereo)
        sample_rate: Audio sample rate

    Returns:
        DecodedPreset with extracted patches and patterns
    """
    # Convert stereo to mono if needed
    if samples.ndim > 1:
        samples = samples[:, 0]

    # Resample if needed
    if sample_rate != SAMPLE_RATE:
        ratio = SAMPLE_RATE / sample_rate
        new_len = int(len(samples) * ratio)
        indices = np.linspace(0, len(samples) - 1, new_len)
        samples = np.interp(indices, np.arange(len(samples)), samples)

    # Demodulate FSK
    raw_bytes = demodulate_fsk(samples, SAMPLE_RATE)

    if raw_bytes is None:
        preset = DecodedPreset()
        preset.error = "FSK demodulation failed - no valid signal detected"
        return preset

    # Decode modem packet
    return decode_modem_bytes(raw_bytes)


def get_patch_summary(patch: DecodedPatch) -> str:
    """Get a human-readable summary of a decoded patch."""
    if not patch.synth_params:
        return "Empty patch"

    p = patch.synth_params
    wf_names = ['Sine', 'Triangle', 'Saw']
    wf = wf_names[p.get('osc_waveform', 0)]
    freq = p.get('osc_frequency', 0)
    decay = p.get('osc_decay', 0)
    mix_val = p.get('osc_noise_mix', 1.0)

    if mix_val < 0.3:
        character = "Noise"
    elif mix_val > 0.7:
        character = wf
    else:
        character = f"{wf}+Noise"

    return f"{character} {freq:.0f}Hz, decay {decay:.0f}ms"


# ==============================================================================
# CLI
# ==============================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print(f"  {sys.argv[0]} <input.wav>              # Decode WAV file")
        print(f"  {sys.argv[0]} <input.wav> --verbose     # Verbose output")
        sys.exit(1)

    wav_file = sys.argv[1]
    verbose = '--verbose' in sys.argv or '-v' in sys.argv

    print(f"Decoding: {wav_file}")
    preset = decode_wav_file(wav_file)

    if preset.error:
        print(f"Error: {preset.error}")
        sys.exit(1)

    print(f"Decoded successfully!")
    print(f"  Left patches: {len(preset.left_patches)}")
    print(f"  Right patches: {len(preset.right_patches)}")
    print(f"  Patterns: {len(preset.patterns)}")
    print(f"  State: {'yes' if preset.state else 'no'}")

    for idx in sorted(preset.left_patches.keys()):
        patch = preset.left_patches[idx]
        summary = get_patch_summary(patch)
        print(f"  Channel {idx + 1}: {summary}")

        if verbose and patch.normalized:
            for name in PARAM_NAMES:
                n = patch.normalized.get(name, 0)
                print(f"    {name:10s}: norm={n:.4f}", end="")
                if name in patch.synth_params:
                    print(f"  → synth={patch.synth_params.get(name, '?')}", end="")
                print()
