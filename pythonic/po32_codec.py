"""
PO-32 Tonic Modem Codec

Self-contained FSK modem encoder and decoder for PO-32 Tonic audio transfer.
Handles encoding synth parameters to FSK audio and decoding FSK audio back
to synth parameters.

Protocol layers:
  Encode: params → uint16 LE → TLV framing → CRC scramble → bit-reverse → FSK audio
  Decode: FSK audio → zero-crossings → groups → bits → bytes → bit-reverse → descramble → TLV → params
"""

import struct
import math
import wave
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field


# ==============================================================================
# Constants
# ==============================================================================

SAMPLE_RATE = 44100
FREQ_HIGH = 3675.0    # ~6 samples per half-period
FREQ_LOW = 1837.5     # ~12 samples per half-period

CRC_INIT = 0x1D0F
MODEM_HEADER = bytes([0x14, 0x19, 0x9D, 0xCF])

# TLV tags (little-endian uint16)
TAG_PATCH   = 0x37B2
TAG_PATTERN = 0xD022
TAG_STATE   = 0x505A
TAG_TRAILER = 0x71C3

# SDK frequency normalization constants
FREQ_VALUE_C4 = 0.4725600599127338
OCTAVE_STEP = 0.1003433318879937

# Parameter serialization order (21 parameters per patch)
PARAM_NAMES = [
    'OscWave', 'OscFreq', 'OscAtk', 'OscDcy', 'ModMode', 'ModRate', 'ModAmt',
    'NFilMod', 'NFilFrq', 'NFilQ', 'NEnvMod', 'NEnvAtk', 'NEnvDcy', 'Mix',
    'DistAmt', 'EQFreq', 'EQGain', 'Level', 'OscVel', 'NVel', 'ModVel'
]

DISCRETE_PARAMS = {
    'OscWave': ['Sine', 'Triangle', 'Saw'],
    'ModMode': ['Decay', 'Sine', 'Noise'],
    'NFilMod': ['LP', 'BP', 'HP'],
    'NEnvMod': ['Exp', 'Linear', 'Mod'],
}


# ==============================================================================
# CRC-16-CCITT
# ==============================================================================

def crc16_ccitt_update(crc: int, byte: int) -> int:
    x = ((crc >> 8) ^ byte) & 0xFF
    x ^= (x >> 4)
    return ((crc << 8) ^ (x << 12) ^ (x << 5) ^ x) & 0xFFFF


# ==============================================================================
# Bit Reversal
# ==============================================================================

def reverse_bits(b: int) -> int:
    result = 0
    for i in range(8):
        if b & (1 << i):
            result |= 1 << (7 - i)
    return result


def bit_reverse_bytes(data: bytes) -> bytes:
    return bytes(reverse_bits(b) for b in data)


# ==============================================================================
# WAV I/O
# ==============================================================================

def save_wav(samples: np.ndarray, filename: str, sample_rate: int = SAMPLE_RATE):
    """Save float samples as 16-bit WAV."""
    peak = np.max(np.abs(samples))
    if peak > 0:
        samples = samples / peak * 0.9
    int_samples = (samples * 32767).astype(np.int16)
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int_samples.tobytes())


def load_wav(filename: str) -> Tuple[np.ndarray, int]:
    """Load WAV file as float64 samples."""
    with wave.open(filename, 'r') as wf:
        rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        if wf.getsampwidth() == 2:
            samples = np.frombuffer(frames, dtype=np.int16).astype(np.float64) / 32768.0
        else:
            samples = np.frombuffer(frames, dtype=np.uint8).astype(np.float64) / 128.0 - 1.0
    return samples, rate


# ==============================================================================
# Modem Encoder (scrambling + TLV framing)
# ==============================================================================

class ModemEncoder:
    """Builds a scrambled modem packet with TLV framing."""

    def __init__(self):
        self.crc = CRC_INIT
        self.buffer = bytearray()

    def _write_byte(self, byte: int):
        scrambled = (byte + (self.crc & 0xFF)) & 0xFF
        self.buffer.append(scrambled)
        self.crc = crc16_ccitt_update(self.crc, byte)

    def append(self, tag: int, data: bytes):
        self._write_byte(tag & 0xFF)
        self._write_byte((tag >> 8) & 0xFF)
        self._write_byte(len(data))
        for b in data:
            self._write_byte(b)
        crc_lo = self.crc & 0xFF
        crc_hi = (self.crc >> 8) & 0xFF
        self._write_byte(crc_lo)
        self._write_byte(crc_hi)

    def get_packet(self) -> bytes:
        return MODEM_HEADER + bytes(self.buffer)


# ==============================================================================
# Descrambler
# ==============================================================================

def descramble(data: bytes, crc_init: int = CRC_INIT) -> Tuple[bytes, int]:
    crc = crc_init
    result = bytearray()
    for sb in data:
        original = (sb - (crc & 0xFF)) & 0xFF
        result.append(original)
        crc = crc16_ccitt_update(crc, original)
    return bytes(result), crc


# ==============================================================================
# Forward Normalization (synth values → 0-1)
# ==============================================================================

def freq_to_norm(hz: float) -> float:
    if hz <= 0:
        return 0.0
    return max(0.0, min(1.0, OCTAVE_STEP * math.log2(hz / 523.25) + FREQ_VALUE_C4))


def modrate_sine_to_norm(hz: float) -> float:
    if hz <= 0:
        return 0.0
    if hz >= 2000.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 2000.0 * mid * 10 ** (3.5 * (mid - 1)) < hz:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-15:
            break
    return (lo + hi) / 2


def modrate_noise_to_norm(hz: float) -> float:
    if hz <= 0:
        return 0.0
    if hz >= 20000.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 20000.0 * mid * 10 ** (4.0 * (mid - 1)) < hz:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-15:
            break
    return (lo + hi) / 2


def modrate_decay_to_norm(ms: float) -> float:
    return max(0.0, min(1.0, 1030.0 / (3.0 * ms + 1000.0)))


def decay_to_norm(ms: float) -> float:
    if ms <= 0:
        return 0.0
    t_sec = ms / 1000.0
    x = (math.log10(t_sec) + 2) / 3
    return max(0.0, min(1.0, x))


def attack_to_norm(ms: float) -> float:
    if ms <= 0:
        return 0.0
    t_sec = ms / 1000.0

    def f(x):
        if x <= 0:
            return 0
        return x * 10 ** ((12 * x - 7) / 5)

    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if f(mid) < t_sec:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def modamt_to_norm(semitones: float, mod_mode: str) -> float:
    max_sm = 96.0 if mod_mode == 'Decay' else 48.0
    if abs(semitones) < 1e-10:
        return 0.5
    sign = 1.0 if semitones > 0 else -1.0
    return 0.5 + sign * 0.5 * math.sqrt(min(abs(semitones), max_sm) / max_sm)


def level_to_norm(db: float) -> float:
    s = 49.0 + db
    disc = s * s + 240.0
    if disc < 0:
        return 0.0
    return max(0.0, min(1.0, (s + math.sqrt(disc)) / 120.0))


# ==============================================================================
# Inverse Normalization (0-1 → synth values)
# ==============================================================================

def norm_to_freq(n: float) -> float:
    return 523.25 * 2.0 ** ((n - FREQ_VALUE_C4) / OCTAVE_STEP)


def norm_to_decay(n: float) -> float:
    if n <= 0:
        return 0.01
    return 10.0 ** (3.0 * n - 2.0) * 1000.0


def norm_to_attack(n: float) -> float:
    if n <= 0:
        return 0.0
    return n * 10.0 ** ((12.0 * n - 7.0) / 5.0) * 1000.0


def norm_to_modrate_sine(n: float) -> float:
    if n <= 0:
        return 0.0
    return 2000.0 * n * 10.0 ** (3.5 * (n - 1.0))


def norm_to_modrate_noise(n: float) -> float:
    if n <= 0:
        return 0.0
    return 20000.0 * n * 10.0 ** (4.0 * (n - 1.0))


def norm_to_modrate_decay(n: float) -> float:
    if n <= 0:
        return 10000.0
    return (1030.0 - 1000.0 * n) / (3.0 * n)


def norm_to_modamt(n: float, mod_mode: str) -> float:
    max_sm = 96.0 if mod_mode == 'Decay' else 48.0
    offset = n - 0.5
    sign = 1.0 if offset >= 0 else -1.0
    return sign * max_sm * (2.0 * abs(offset)) ** 2


def norm_to_level(n: float) -> float:
    if n <= 0.001:
        return -100.0
    return 60.0 * n - 49.0 - 1.0 / n


def norm_to_nfilq(n: float) -> float:
    if n <= 0:
        return 0.1
    return 0.1 * (10001.0 / 0.1) ** n


def norm_to_discrete(n: float, options: List[str]) -> Tuple[int, str]:
    num = len(options)
    idx = round(n * (num - 1))
    idx = max(0, min(num - 1, idx))
    return idx, options[idx]


# ==============================================================================
# FSK Encoding
# ==============================================================================

def _bytes_to_bitstream(data: bytes) -> List[int]:
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits


def _bitstream_to_n_values(bits: List[int]) -> List[int]:
    n_values = []
    count = 0
    for bit in bits:
        if bit == 1:
            count += 1
        else:
            n_values.append(count)
            count = 0
    if count > 0:
        n_values.append(count)
    return n_values


def generate_fsk_signal(data: bytes, preamble_groups: int = 496,
                        sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Generate FSK audio using phase-continuous sine wave generation."""
    bits = _bytes_to_bitstream(data)
    n_values = _bitstream_to_n_values(bits)
    groups = [4 + 6 * n for n in n_values]

    preamble = [10] * preamble_groups
    all_groups = preamble + groups

    samples = []
    phase = 0.0

    for group_size in all_groups:
        samples_per_half_h = sample_rate / (2 * FREQ_HIGH)
        total_h_samples = int(round(group_size * samples_per_half_h))
        for _ in range(total_h_samples):
            samples.append(math.sin(phase))
            phase += 2 * math.pi * FREQ_HIGH / sample_rate

        samples_per_half_l = sample_rate / (2 * FREQ_LOW)
        total_l_samples = int(round(samples_per_half_l))
        for _ in range(total_l_samples):
            samples.append(math.sin(phase))
            phase += 2 * math.pi * FREQ_LOW / sample_rate

    return np.array(samples, dtype=np.float64)


# ==============================================================================
# Default Data (morph targets, empty patterns, default state)
# ==============================================================================

_DEFAULT_RIGHT_PATCH = bytes([
    0x00, 0x80, 0xDE, 0x1D, 0x00, 0x00, 0xE5, 0xB0,
    0x00, 0x00, 0x8F, 0x82, 0x5C, 0xCF, 0x00, 0x00,
    0x14, 0x2E, 0x24, 0x47, 0x00, 0x00, 0xC2, 0x35,
    0x75, 0x6B, 0x14, 0x2E, 0xB0, 0x66, 0xB6, 0x75,
    0x00, 0x00, 0xE8, 0xDE, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00
])


def default_right_patch(drum_idx: int = 0) -> bytes:
    """Return a default right (morph target) patch."""
    return _DEFAULT_RIGHT_PATCH


def default_pattern(pattern_num: int) -> bytes:
    """Generate a default empty pattern (all steps off)."""
    return bytes([pattern_num]) + bytes(210)


def default_state() -> bytes:
    """Generate default state data (37 bytes)."""
    return bytes(37)


# ==============================================================================
# FSK Demodulation
# ==============================================================================

def demodulate_fsk(samples: np.ndarray, sample_rate: int = SAMPLE_RATE,
                   threshold_ratio: float = 0.1) -> Optional[bytes]:
    """Demodulate FSK audio signal back to raw modem bytes.

    Returns raw bytes (bit-reversed, before descrambling), or None on failure.
    """
    if len(samples) == 0:
        return None

    if samples.ndim > 1:
        samples = samples[:, 0]

    peak = np.max(np.abs(samples))
    if peak < 0.001:
        return None
    samples = samples / peak

    # Find active region
    envelope = _compute_envelope(samples, window=256)
    active_indices = np.where(envelope > threshold_ratio)[0]
    if len(active_indices) < 1000:
        return None
    start_idx = max(0, active_indices[0] - 100)
    end_idx = min(len(samples), active_indices[-1] + 100)
    samples = samples[start_idx:end_idx]

    half_periods = _measure_half_periods(samples)
    if len(half_periods) < 100:
        return None

    hp_high = sample_rate / (2 * FREQ_HIGH)
    hp_low = sample_rate / (2 * FREQ_LOW)
    hp_threshold = math.sqrt(hp_high * hp_low)

    classifications = ['H' if hp < hp_threshold else 'L' for hp in half_periods]

    groups = _extract_groups(classifications)
    if len(groups) < 50:
        return None

    data_start = _find_data_start(groups)
    if data_start is None:
        return None

    n_values = []
    for g in groups[data_start:]:
        n = round((g - 4) / 6)
        n_values.append(max(0, n))

    bits = []
    for n in n_values:
        bits.extend([1] * n)
        bits.append(0)

    raw_bytes = _bits_to_bytes(bits)
    if len(raw_bytes) < 5:
        return None

    return bytes(raw_bytes)


def _compute_envelope(samples: np.ndarray, window: int = 256) -> np.ndarray:
    kernel = np.ones(window) / window
    return np.convolve(np.abs(samples), kernel, mode='same')


def _measure_half_periods(samples: np.ndarray) -> List[float]:
    signs = np.sign(samples)
    for i in range(1, len(signs)):
        if signs[i] == 0:
            signs[i] = signs[i - 1]

    crossings = np.where(np.diff(signs) != 0)[0]
    if len(crossings) < 2:
        return []

    refined = []
    for idx in crossings:
        if idx + 1 < len(samples):
            s0, s1 = samples[idx], samples[idx + 1]
            if s0 != s1:
                refined.append(idx + (-s0 / (s1 - s0)))
            else:
                refined.append(float(idx))
        else:
            refined.append(float(idx))

    return [refined[i] - refined[i - 1] for i in range(1, len(refined))]


def _extract_groups(classifications: List[str]) -> List[int]:
    groups = []
    count = 0
    for c in classifications:
        if c == 'H':
            count += 1
        else:
            if count > 0:
                groups.append(count)
            count = 0
    if count > 0:
        groups.append(count)
    return groups


def _find_data_start(groups: List[int], min_preamble: int = 20) -> Optional[int]:
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
    result = []
    for i in range(0, len(bits) - 7, 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        result.append(byte)
    return result


# ==============================================================================
# Decoded Data Structures
# ==============================================================================

@dataclass
class DecodedPatch:
    """A decoded drum patch with normalized and synth-ready parameters."""
    prefix: int
    drum_index: int
    is_left: bool
    raw_params: bytes
    normalized: Dict[str, float] = field(default_factory=dict)
    synth_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecodedPreset:
    """Complete decoded PO-32 preset."""
    left_patches: Dict[int, DecodedPatch] = field(default_factory=dict)
    right_patches: Dict[int, DecodedPatch] = field(default_factory=dict)
    patterns: List[bytes] = field(default_factory=list)
    state: Optional[bytes] = None
    error: Optional[str] = None


# ==============================================================================
# Packet Decoder
# ==============================================================================

def decode_modem_bytes(raw_bytes: bytes) -> DecodedPreset:
    """Decode raw modem bytes (from FSK demod) into a preset structure."""
    preset = DecodedPreset()

    if len(raw_bytes) < 8:
        preset.error = "Data too short"
        return preset

    modem_data = bytes(reverse_bits(b) for b in raw_bytes)

    if modem_data[:4] != MODEM_HEADER:
        header_pos = _find_header(modem_data)
        if header_pos is None:
            preset.error = f"Modem header not found (got {modem_data[:4].hex()})"
            return preset
        modem_data = modem_data[header_pos:]

    descrambled, _ = descramble(modem_data[4:])

    pos = 0
    while pos + 5 <= len(descrambled):
        tag = descrambled[pos] | (descrambled[pos + 1] << 8)
        flen = descrambled[pos + 2]

        if pos + 3 + flen + 2 > len(descrambled):
            break

        fdata = descrambled[pos + 3:pos + 3 + flen]
        pos += 3 + flen + 2

        if tag == TAG_PATCH and flen >= 1:
            prefix = fdata[0]
            param_data = fdata[1:]
            drum_idx = prefix & 0x0F
            is_left = (prefix & 0xF0) == 0x10

            patch = DecodedPatch(
                prefix=prefix, drum_index=drum_idx,
                is_left=is_left, raw_params=param_data,
            )
            if len(param_data) >= 42:
                patch.normalized = _deserialize_params(param_data)
                patch.synth_params = normalized_to_synth_params(patch.normalized)

            if is_left:
                preset.left_patches[drum_idx] = patch
            else:
                preset.right_patches[drum_idx] = patch

        elif tag == TAG_PATTERN:
            preset.patterns.append(fdata)
        elif tag == TAG_STATE:
            preset.state = fdata
        elif tag == TAG_TRAILER:
            break

    return preset


def _find_header(data: bytes, max_offset: int = 50) -> Optional[int]:
    for i in range(min(max_offset, len(data) - 4)):
        if data[i:i + 4] == MODEM_HEADER:
            return i
    return None


def _deserialize_params(param_data: bytes) -> Dict[str, float]:
    result = {}
    for i, name in enumerate(PARAM_NAMES):
        if i * 2 + 2 <= len(param_data):
            u16 = struct.unpack_from('<H', param_data, i * 2)[0]
            result[name] = u16 / 65536.0
        else:
            result[name] = 0.0
    return result


# ==============================================================================
# Normalized → Synth Parameters
# ==============================================================================

def normalized_to_synth_params(norm: Dict[str, float]) -> Dict[str, Any]:
    """Convert normalized 0-1 PO-32 parameters to Pythonic synth parameter dict.

    Returns a dict compatible with DrumChannel.set_parameters().
    """
    params = {}

    wf_idx, _ = norm_to_discrete(norm.get('OscWave', 0.0), ['Sine', 'Triangle', 'Saw'])
    params['osc_waveform'] = wf_idx

    mm_idx, mm_str = norm_to_discrete(norm.get('ModMode', 0.0), ['Decay', 'Sine', 'Noise'])
    params['pitch_mod_mode'] = mm_idx
    mod_mode = mm_str

    fm_idx, _ = norm_to_discrete(norm.get('NFilMod', 0.0), ['LP', 'BP', 'HP'])
    params['noise_filter_mode'] = fm_idx

    em_idx, em_str = norm_to_discrete(norm.get('NEnvMod', 0.0), ['Exp', 'Linear', 'Mod'])
    params['noise_envelope_mode'] = em_idx
    nenv_mode = em_str

    params['osc_frequency'] = norm_to_freq(norm.get('OscFreq', 0.5))
    params['noise_filter_freq'] = norm_to_freq(norm.get('NFilFrq', 0.5))
    params['eq_frequency'] = norm_to_freq(norm.get('EQFreq', 0.5))

    params['osc_attack'] = norm_to_attack(norm.get('OscAtk', 0.0))
    noise_atk = norm_to_attack(norm.get('NEnvAtk', 0.0))
    if nenv_mode == 'Linear':
        noise_atk /= 1.5
    params['noise_attack'] = noise_atk

    params['osc_decay'] = norm_to_decay(norm.get('OscDcy', 0.5))
    noise_dcy = norm_to_decay(norm.get('NEnvDcy', 0.5))
    if nenv_mode == 'Linear':
        noise_dcy /= 1.5
    params['noise_decay'] = noise_dcy

    mod_rate_norm = norm.get('ModRate', 0.5)
    if mod_mode == 'Sine':
        params['pitch_mod_rate'] = norm_to_modrate_sine(mod_rate_norm)
    elif mod_mode == 'Noise':
        params['pitch_mod_rate'] = norm_to_modrate_noise(mod_rate_norm)
    else:
        params['pitch_mod_rate'] = norm_to_modrate_decay(mod_rate_norm)

    params['pitch_mod_amount'] = norm_to_modamt(norm.get('ModAmt', 0.5), mod_mode)
    params['noise_filter_q'] = norm_to_nfilq(norm.get('NFilQ', 0.0))
    params['osc_noise_mix'] = 1.0 - norm.get('Mix', 0.0)
    params['distortion'] = norm.get('DistAmt', 0.0)
    params['eq_gain_db'] = norm.get('EQGain', 0.5) * 80.0 - 40.0
    params['level_db'] = norm_to_level(norm.get('Level', 0.5))
    params['osc_vel_sensitivity'] = norm.get('OscVel', 0.0)
    params['noise_vel_sensitivity'] = norm.get('NVel', 0.0)
    params['mod_vel_sensitivity'] = norm.get('ModVel', 0.0)

    # PO-32 protocol has no pitch offset — reset to zero on import
    params['pitch_semitones'] = 0.0

    return params


# ==============================================================================
# High-Level Decode
# ==============================================================================

def decode_wav_file(filename: str) -> DecodedPreset:
    """Decode a PO-32 transfer WAV file into patches and patterns."""
    try:
        samples, rate = load_wav(filename)
    except Exception as e:
        preset = DecodedPreset()
        preset.error = f"Failed to load WAV: {e}"
        return preset
    return decode_audio_samples(samples, rate)


def decode_audio_samples(samples: np.ndarray,
                         sample_rate: int = SAMPLE_RATE) -> DecodedPreset:
    """Decode FSK audio samples into patches and patterns."""
    if sample_rate != SAMPLE_RATE:
        ratio = SAMPLE_RATE / sample_rate
        new_len = int(len(samples) * ratio)
        indices = np.linspace(0, len(samples) - 1, new_len)
        if samples.ndim == 1:
            samples = np.interp(indices, np.arange(len(samples)), samples)
        else:
            samples = np.interp(indices, np.arange(len(samples)), samples[:, 0])

    raw_bytes = demodulate_fsk(samples, SAMPLE_RATE)
    if raw_bytes is None:
        preset = DecodedPreset()
        preset.error = "FSK demodulation failed - no valid signal detected"
        return preset

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
