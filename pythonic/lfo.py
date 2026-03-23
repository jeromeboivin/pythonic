"""
LFO and Modulation system for Pythonic drum synthesizer.

Provides:
- LFOWaveform enum (sine, triangle, saw up/down, square, sample-and-hold)
- ModTarget enum (full catalog of assignable per-channel + global targets)
- LFO class (per-voice LFO with retrigger, phase offset, sync, polarity)
- PumpSource class (sidechain-style ducking envelope)
- ModulationRouter (applies mod source outputs to target parameters)
"""

import numpy as np
from enum import Enum


# ---------------------------------------------------------------------------
# Waveforms
# ---------------------------------------------------------------------------

class LFOWaveform(Enum):
    SINE = 0
    TRIANGLE = 1
    SAW_UP = 2
    SAW_DOWN = 3
    SQUARE = 4
    SAMPLE_AND_HOLD = 5


# ---------------------------------------------------------------------------
# Retrigger behaviour
# ---------------------------------------------------------------------------

class LFORetrigger(Enum):
    FREE = 0       # Free-running – phase never resets
    RETRIGGER = 1  # Reset phase on every note trigger


# ---------------------------------------------------------------------------
# Polarity
# ---------------------------------------------------------------------------

class LFOPolarity(Enum):
    BIPOLAR = 0   # -1 … +1
    UNIPOLAR = 1  # 0 … +1


# ---------------------------------------------------------------------------
# Sync divisions (relative to quarter note)
# ---------------------------------------------------------------------------

class SyncDivision(Enum):
    OFF = 0        # Free Hz rate
    WHOLE = 1      # 1/1
    HALF = 2       # 1/2
    QUARTER = 3    # 1/4
    EIGHTH = 4     # 1/8
    SIXTEENTH = 5  # 1/16
    DOTTED_QUARTER = 6   # 1/4.
    DOTTED_EIGHTH = 7    # 1/8.
    TRIPLET_QUARTER = 8  # 1/4T
    TRIPLET_EIGHTH = 9   # 1/8T
    TWO_BARS = 10  # 2/1
    FOUR_BARS = 11 # 4/1

# Quarter-note multipliers for each sync division
_SYNC_QUARTER_MULTIPLIERS = {
    SyncDivision.OFF: 1.0,
    SyncDivision.WHOLE: 4.0,
    SyncDivision.HALF: 2.0,
    SyncDivision.QUARTER: 1.0,
    SyncDivision.EIGHTH: 0.5,
    SyncDivision.SIXTEENTH: 0.25,
    SyncDivision.DOTTED_QUARTER: 1.5,
    SyncDivision.DOTTED_EIGHTH: 0.75,
    SyncDivision.TRIPLET_QUARTER: 2.0 / 3.0,
    SyncDivision.TRIPLET_EIGHTH: 1.0 / 3.0,
    SyncDivision.TWO_BARS: 8.0,
    SyncDivision.FOUR_BARS: 16.0,
}


# ---------------------------------------------------------------------------
# Modulation target catalog
# ---------------------------------------------------------------------------

class ModTarget(Enum):
    """Every parameter an LFO or pump source can modulate.

    The enum value is a stable string ID used for preset serialisation.
    """

    # -- No target (LFO disabled routing) --
    NONE = 'none'

    # -- Oscillator --
    OSC_FREQUENCY = 'osc_frequency'
    OSC_ATTACK = 'osc_attack'
    OSC_DECAY = 'osc_decay'
    PITCH_MOD_AMOUNT = 'pitch_mod_amount'
    PITCH_MOD_RATE = 'pitch_mod_rate'
    PITCH_SEMITONES = 'pitch_semitones'

    # -- Noise --
    NOISE_FILTER_FREQ = 'noise_filter_freq'
    NOISE_FILTER_Q = 'noise_filter_q'
    NOISE_ATTACK = 'noise_attack'
    NOISE_DECAY = 'noise_decay'

    # -- Mix / level --
    OSC_NOISE_MIX = 'osc_noise_mix'
    LEVEL_DB = 'level_db'
    PAN = 'pan'
    DISTORTION = 'distortion'

    # -- EQ --
    EQ_FREQUENCY = 'eq_frequency'
    EQ_GAIN_DB = 'eq_gain_db'

    # -- FX --
    VINTAGE_AMOUNT = 'vintage_amount'
    REVERB_DECAY = 'reverb_decay'
    REVERB_MIX = 'reverb_mix'
    REVERB_WIDTH = 'reverb_width'
    DELAY_FEEDBACK = 'delay_feedback'
    DELAY_MIX = 'delay_mix'

    # -- Velocity --
    OSC_VEL_SENSITIVITY = 'osc_vel_sensitivity'
    NOISE_VEL_SENSITIVITY = 'noise_vel_sensitivity'
    MOD_VEL_SENSITIVITY = 'mod_vel_sensitivity'

    # -- Global --
    MASTER_VOLUME = 'master_volume'
    MORPH = 'morph'


# Human-readable labels and group names for the destination picker UI
MOD_TARGET_GROUPS = {
    'Oscillator': [
        ModTarget.OSC_FREQUENCY, ModTarget.PITCH_SEMITONES,
        ModTarget.PITCH_MOD_AMOUNT, ModTarget.PITCH_MOD_RATE,
        ModTarget.OSC_ATTACK, ModTarget.OSC_DECAY,
    ],
    'Noise': [
        ModTarget.NOISE_FILTER_FREQ, ModTarget.NOISE_FILTER_Q,
        ModTarget.NOISE_ATTACK, ModTarget.NOISE_DECAY,
    ],
    'Mix': [
        ModTarget.OSC_NOISE_MIX, ModTarget.LEVEL_DB,
        ModTarget.PAN, ModTarget.DISTORTION,
    ],
    'EQ': [
        ModTarget.EQ_FREQUENCY, ModTarget.EQ_GAIN_DB,
    ],
    'FX': [
        ModTarget.VINTAGE_AMOUNT,
        ModTarget.REVERB_DECAY, ModTarget.REVERB_MIX, ModTarget.REVERB_WIDTH,
        ModTarget.DELAY_FEEDBACK, ModTarget.DELAY_MIX,
    ],
    'Velocity': [
        ModTarget.OSC_VEL_SENSITIVITY, ModTarget.NOISE_VEL_SENSITIVITY,
        ModTarget.MOD_VEL_SENSITIVITY,
    ],
    'Global': [
        ModTarget.MASTER_VOLUME, ModTarget.MORPH,
    ],
}

MOD_TARGET_LABELS = {
    ModTarget.NONE: 'Off',
    ModTarget.OSC_FREQUENCY: 'Osc Freq',
    ModTarget.PITCH_SEMITONES: 'Pitch',
    ModTarget.PITCH_MOD_AMOUNT: 'Pitch Mod Amt',
    ModTarget.PITCH_MOD_RATE: 'Pitch Mod Rate',
    ModTarget.OSC_ATTACK: 'Osc Attack',
    ModTarget.OSC_DECAY: 'Osc Decay',
    ModTarget.NOISE_FILTER_FREQ: 'Noise Freq',
    ModTarget.NOISE_FILTER_Q: 'Noise Q',
    ModTarget.NOISE_ATTACK: 'Noise Attack',
    ModTarget.NOISE_DECAY: 'Noise Decay',
    ModTarget.OSC_NOISE_MIX: 'Osc/Noise Mix',
    ModTarget.LEVEL_DB: 'Level',
    ModTarget.PAN: 'Pan',
    ModTarget.DISTORTION: 'Distortion',
    ModTarget.EQ_FREQUENCY: 'EQ Freq',
    ModTarget.EQ_GAIN_DB: 'EQ Gain',
    ModTarget.VINTAGE_AMOUNT: 'Vintage',
    ModTarget.REVERB_DECAY: 'Reverb Time',
    ModTarget.REVERB_MIX: 'Reverb Mix',
    ModTarget.REVERB_WIDTH: 'Reverb Width',
    ModTarget.DELAY_FEEDBACK: 'Delay Feedback',
    ModTarget.DELAY_MIX: 'Delay Mix',
    ModTarget.OSC_VEL_SENSITIVITY: 'Osc Vel',
    ModTarget.NOISE_VEL_SENSITIVITY: 'Noise Vel',
    ModTarget.MOD_VEL_SENSITIVITY: 'Mod Vel',
    ModTarget.MASTER_VOLUME: 'Master Vol',
    ModTarget.MORPH: 'Morph',
}

# Default modulation depth ranges per target (min_depth, max_depth, unit)
# Bipolar depth: the LFO output (-1..+1) is scaled by depth,
# then added to the base parameter value.
MOD_TARGET_DEPTH_RANGES = {
    ModTarget.NONE: (0.0, 0.0, ''),
    ModTarget.OSC_FREQUENCY: (0.0, 20000.0, 'Hz'),
    ModTarget.PITCH_SEMITONES: (0.0, 48.0, 'st'),
    ModTarget.PITCH_MOD_AMOUNT: (0.0, 120.0, 'st'),
    ModTarget.PITCH_MOD_RATE: (0.0, 2000.0, 'Hz'),
    ModTarget.OSC_ATTACK: (0.0, 10000.0, 'ms'),
    ModTarget.OSC_DECAY: (0.0, 10000.0, 'ms'),
    ModTarget.NOISE_FILTER_FREQ: (0.0, 20000.0, 'Hz'),
    ModTarget.NOISE_FILTER_Q: (0.0, 10.0, ''),
    ModTarget.NOISE_ATTACK: (0.0, 10000.0, 'ms'),
    ModTarget.NOISE_DECAY: (0.0, 10000.0, 'ms'),
    ModTarget.OSC_NOISE_MIX: (0.0, 1.0, ''),
    ModTarget.LEVEL_DB: (0.0, 40.0, 'dB'),
    ModTarget.PAN: (0.0, 100.0, ''),
    ModTarget.DISTORTION: (0.0, 1.0, ''),
    ModTarget.EQ_FREQUENCY: (0.0, 20000.0, 'Hz'),
    ModTarget.EQ_GAIN_DB: (0.0, 40.0, 'dB'),
    ModTarget.VINTAGE_AMOUNT: (0.0, 1.0, ''),
    ModTarget.REVERB_DECAY: (0.0, 1.0, ''),
    ModTarget.REVERB_MIX: (0.0, 1.0, ''),
    ModTarget.REVERB_WIDTH: (0.0, 2.0, ''),
    ModTarget.DELAY_FEEDBACK: (0.0, 0.95, ''),
    ModTarget.DELAY_MIX: (0.0, 1.0, ''),
    ModTarget.OSC_VEL_SENSITIVITY: (0.0, 2.0, ''),
    ModTarget.NOISE_VEL_SENSITIVITY: (0.0, 2.0, ''),
    ModTarget.MOD_VEL_SENSITIVITY: (0.0, 2.0, ''),
    ModTarget.MASTER_VOLUME: (0.0, 20.0, 'dB'),
    ModTarget.MORPH: (0.0, 1.0, ''),
}


# ---------------------------------------------------------------------------
# LFO generator
# ---------------------------------------------------------------------------

class LFO:
    """Per-voice low-frequency oscillator.

    Generates one scalar value per audio block (block-rate, not sample-rate).
    Output range depends on polarity setting:
      BIPOLAR:  -1 … +1
      UNIPOLAR:  0 … +1
    """

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate

        # User-facing parameters
        self.enabled = False
        self.waveform = LFOWaveform.SINE
        self.rate_hz = 1.0         # Free-running rate (0.01 – 100 Hz)
        self.sync = SyncDivision.OFF
        self.depth = 0.0           # 0 … max (unit depends on target)
        self.target = ModTarget.NONE
        self.retrigger = LFORetrigger.RETRIGGER
        self.phase_offset = 0.0    # 0 … 1 (fraction of cycle)
        self.polarity = LFOPolarity.BIPOLAR

        # Internal state
        self._phase = 0.0          # 0 … 1 normalised phase
        self._sh_value = 0.0       # Latched sample-and-hold value
        self._last_output = 0.0    # Last raw waveform output (-1..+1 or 0..+1)

    # ---- state management ----

    def reset_phase(self):
        """Reset phase to the configured phase_offset. Called on voice trigger
        when retrigger mode is RETRIGGER."""
        self._phase = self.phase_offset % 1.0
        self._sh_value = 0.0

    def get_parameters(self) -> dict:
        """Serialise LFO state to a plain dict."""
        return {
            'enabled': self.enabled,
            'waveform': self.waveform.value,
            'rate_hz': self.rate_hz,
            'sync': self.sync.value,
            'depth': self.depth,
            'target': self.target.value,
            'retrigger': self.retrigger.value,
            'phase_offset': self.phase_offset,
            'polarity': self.polarity.value,
        }

    def set_parameters(self, params: dict):
        """Restore LFO state from a plain dict (safe for missing keys)."""
        if 'enabled' in params:
            self.enabled = bool(params['enabled'])
        if 'waveform' in params:
            self.waveform = LFOWaveform(params['waveform'])
        if 'rate_hz' in params:
            self.rate_hz = float(params['rate_hz'])
        if 'sync' in params:
            self.sync = SyncDivision(params['sync'])
        if 'depth' in params:
            self.depth = float(params['depth'])
        if 'target' in params:
            self.target = ModTarget(params['target'])
        if 'retrigger' in params:
            self.retrigger = LFORetrigger(params['retrigger'])
        if 'phase_offset' in params:
            self.phase_offset = float(params['phase_offset'])
        if 'polarity' in params:
            self.polarity = LFOPolarity(params['polarity'])

    # ---- waveform computation (block-rate) ----

    def process(self, num_samples: int, bpm: float = 120.0) -> float:
        """Advance the LFO by *num_samples* and return a single scalar output.

        Args:
            num_samples: Audio block size (used to advance phase).
            bpm: Current tempo (used when sync != OFF).

        Returns:
            Modulation value.  Raw waveform * depth, ready to be added to the
            base parameter value by the router.
        """
        if not self.enabled or self.target == ModTarget.NONE:
            self._last_output = 0.0
            return 0.0

        # Determine effective rate
        if self.sync != SyncDivision.OFF and bpm > 0:
            quarter_seconds = 60.0 / bpm
            period_seconds = quarter_seconds * _SYNC_QUARTER_MULTIPLIERS[self.sync]
            effective_rate = 1.0 / period_seconds if period_seconds > 0 else self.rate_hz
        else:
            effective_rate = self.rate_hz

        # Advance phase
        phase_inc = effective_rate * num_samples / self.sr
        old_phase = self._phase
        self._phase = (self._phase + phase_inc) % 1.0

        # Waveform generation (phase 0..1 → raw -1..+1)
        p = self._phase
        if self.waveform == LFOWaveform.SINE:
            raw = np.sin(2.0 * np.pi * p)
        elif self.waveform == LFOWaveform.TRIANGLE:
            raw = 4.0 * abs(p - 0.5) - 1.0
        elif self.waveform == LFOWaveform.SAW_UP:
            raw = 2.0 * p - 1.0
        elif self.waveform == LFOWaveform.SAW_DOWN:
            raw = 1.0 - 2.0 * p
        elif self.waveform == LFOWaveform.SQUARE:
            raw = 1.0 if p < 0.5 else -1.0
        elif self.waveform == LFOWaveform.SAMPLE_AND_HOLD:
            # Latch new random value on phase wrap
            if self._phase < old_phase:  # wrapped
                self._sh_value = np.random.uniform(-1.0, 1.0)
            raw = self._sh_value
        else:
            raw = 0.0

        # Polarity
        if self.polarity == LFOPolarity.UNIPOLAR:
            raw = (raw + 1.0) * 0.5  # map -1..+1 → 0..+1

        self._last_output = raw
        # depth is 0-100 (percentage); scale by target's max range
        max_range = MOD_TARGET_DEPTH_RANGES.get(self.target, (0, 0, ''))[1]
        return raw * (self.depth / 100.0) * max_range

    @property
    def last_output(self) -> float:
        """Last computed raw waveform value (before depth scaling)."""
        return self._last_output


# ---------------------------------------------------------------------------
# Pump / sidechain envelope source
# ---------------------------------------------------------------------------

class PumpSource:
    """Simple internal sidechain-style ducking envelope.

    On every trigger the envelope ducks to 0 then recovers with configurable
    attack (duck speed) and release (recovery speed), following a shaped curve.
    The output is a unipolar value 0…1 representing the *attenuation amount*
    (1 = fully ducked, 0 = no ducking).
    """

    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate

        # User parameters
        self.enabled = False
        self.amount = 0.0          # 0 … 1 (depth of ducking)
        self.attack_ms = 1.0       # Duck-in time (ms)
        self.release_ms = 100.0    # Recovery time (ms)
        self.curve = 0.5           # Shape: 0=linear, 0.5=smooth, 1=steep
        self.target = ModTarget.LEVEL_DB
        self.sync = SyncDivision.OFF  # Sync recovery to tempo division

        # Internal state
        self._envelope = 0.0       # 0 = no duck, 1 = fully ducked
        self._triggered = False
        self._phase = 0.0          # 0..1 for attack-release cycle

    def trigger(self):
        """Called on drum voice trigger."""
        if self.enabled:
            self._triggered = True
            self._phase = 0.0
            self._envelope = 1.0  # Immediate duck

    def reset(self):
        self._envelope = 0.0
        self._triggered = False
        self._phase = 0.0

    def get_parameters(self) -> dict:
        return {
            'enabled': self.enabled,
            'amount': self.amount,
            'attack_ms': self.attack_ms,
            'release_ms': self.release_ms,
            'curve': self.curve,
            'target': self.target.value,
            'sync': self.sync.value,
        }

    def set_parameters(self, params: dict):
        if 'enabled' in params:
            self.enabled = bool(params['enabled'])
        if 'amount' in params:
            self.amount = float(params['amount'])
        if 'attack_ms' in params:
            self.attack_ms = float(params['attack_ms'])
        if 'release_ms' in params:
            self.release_ms = float(params['release_ms'])
        if 'curve' in params:
            self.curve = float(params['curve'])
        if 'target' in params:
            self.target = ModTarget(params['target'])
        if 'sync' in params:
            self.sync = SyncDivision(params['sync'])

    def process(self, num_samples: int, bpm: float = 120.0) -> float:
        """Advance envelope and return duck amount (0 = no duck, amount = full duck).

        Returns:
            Negative modulation value: -amount * envelope.
            For LEVEL_DB target this means attenuation.
        """
        if not self.enabled:
            self._envelope = 0.0
            return 0.0

        block_seconds = num_samples / self.sr

        if self._envelope > 0.001:
            # Release phase: envelope decays from 1→0
            release_s = max(self.release_ms / 1000.0, 0.001)
            decay_rate = block_seconds / release_s
            # Apply curve shape
            if self.curve > 0.01:
                decay_rate *= (1.0 + self.curve * 2.0)
            self._envelope = max(0.0, self._envelope - decay_rate)
        else:
            self._envelope = 0.0

        return -self.amount * self._envelope * MOD_TARGET_DEPTH_RANGES.get(self.target, (0, 0, ''))[1]


# ---------------------------------------------------------------------------
# Modulation router
# ---------------------------------------------------------------------------

class ModulationRouter:
    """Applies modulation source outputs to channel/global parameters.

    This is a stateless helper: given a target ID + modulation value,
    it writes to the correct parameter on the channel or synthesizer.
    """

    @staticmethod
    def apply(target: ModTarget, mod_value: float,
              channel, synthesizer=None):
        """Add *mod_value* to the base parameter identified by *target*.

        For channel-local targets, reads the current value, adds mod_value,
        clamps appropriately, and writes back through the channel setter.
        For global targets (MASTER_VOLUME, MORPH), writes through the
        synthesizer reference.

        Args:
            target: Which parameter to modulate.
            mod_value: Signed offset to add.
            channel: DrumChannel instance (always available).
            synthesizer: PythonicSynthesizer instance (needed for global targets).
        """
        if target == ModTarget.NONE or mod_value == 0.0:
            return

        # ---- Channel-local targets ----
        if target == ModTarget.OSC_FREQUENCY:
            base = channel.oscillator.frequency
            channel.oscillator.set_frequency(max(20.0, min(20000.0, base + mod_value)))
        elif target == ModTarget.PITCH_SEMITONES:
            base = channel.pitch_semitones
            channel.pitch_semitones = np.clip(base + mod_value, -48.0, 48.0)
        elif target == ModTarget.PITCH_MOD_AMOUNT:
            base = channel.oscillator.pitch_mod_amount
            channel.oscillator.set_pitch_mod_amount(base + mod_value)
        elif target == ModTarget.PITCH_MOD_RATE:
            base = channel.oscillator.pitch_mod_rate
            channel.oscillator.set_pitch_mod_rate(max(0.1, base + mod_value))
        elif target == ModTarget.OSC_ATTACK:
            base = channel.osc_envelope.attack_ms
            channel.osc_envelope.set_attack(max(0.0, base + mod_value))
        elif target == ModTarget.OSC_DECAY:
            base = channel.osc_envelope.decay_ms
            channel.osc_envelope.set_decay(max(1.0, base + mod_value))
        elif target == ModTarget.NOISE_FILTER_FREQ:
            base = channel.noise_gen.filter_frequency
            channel.noise_gen.set_filter_frequency(max(20.0, min(20000.0, base + mod_value)))
        elif target == ModTarget.NOISE_FILTER_Q:
            base = channel.noise_gen.filter_q
            channel.noise_gen.set_filter_q(max(0.1, min(100.0, base + mod_value)))
        elif target == ModTarget.NOISE_ATTACK:
            base = channel.noise_gen.attack_ms
            channel.noise_gen.set_attack(max(0.0, base + mod_value))
        elif target == ModTarget.NOISE_DECAY:
            base = channel.noise_gen.decay_ms
            channel.noise_gen.set_decay(max(1.0, base + mod_value))
        elif target == ModTarget.OSC_NOISE_MIX:
            base = channel.osc_noise_mix
            channel.set_osc_noise_mix_immediate(np.clip(base + mod_value, 0.0, 1.0))
        elif target == ModTarget.LEVEL_DB:
            base = channel.level_db
            channel.level_db = np.clip(base + mod_value, -60.0, 40.0)
        elif target == ModTarget.PAN:
            base = channel.pan
            channel.pan = np.clip(base + mod_value, -100.0, 100.0)
        elif target == ModTarget.DISTORTION:
            base = channel.distortion
            channel.set_distortion_immediate(np.clip(base + mod_value, 0.0, 1.0))
        elif target == ModTarget.EQ_FREQUENCY:
            base = channel.eq_frequency
            channel.set_eq_frequency_immediate(max(20.0, min(20000.0, base + mod_value)))
        elif target == ModTarget.EQ_GAIN_DB:
            base = channel.eq_gain_db
            channel.set_eq_gain(np.clip(base + mod_value, -40.0, 40.0))
        elif target == ModTarget.VINTAGE_AMOUNT:
            base = channel.vintage_amount
            channel.vintage_amount = np.clip(base + mod_value, 0.0, 1.0)
        elif target == ModTarget.REVERB_DECAY:
            base = channel.reverb_decay
            channel.reverb_decay = np.clip(base + mod_value, 0.0, 1.0)
        elif target == ModTarget.REVERB_MIX:
            base = channel.reverb_mix
            channel.reverb_mix = np.clip(base + mod_value, 0.0, 1.0)
        elif target == ModTarget.REVERB_WIDTH:
            base = channel.reverb_width
            channel.reverb_width = np.clip(base + mod_value, 0.0, 2.0)
        elif target == ModTarget.DELAY_FEEDBACK:
            base = channel.delay_feedback
            channel.delay_feedback = np.clip(base + mod_value, 0.0, 0.95)
        elif target == ModTarget.DELAY_MIX:
            base = channel.delay_mix
            channel.delay_mix = np.clip(base + mod_value, 0.0, 1.0)
        elif target == ModTarget.OSC_VEL_SENSITIVITY:
            base = channel.osc_vel_sensitivity
            channel.osc_vel_sensitivity = np.clip(base + mod_value, 0.0, 2.0)
        elif target == ModTarget.NOISE_VEL_SENSITIVITY:
            base = channel.noise_vel_sensitivity
            channel.noise_vel_sensitivity = np.clip(base + mod_value, 0.0, 2.0)
        elif target == ModTarget.MOD_VEL_SENSITIVITY:
            base = channel.mod_vel_sensitivity
            channel.mod_vel_sensitivity = np.clip(base + mod_value, 0.0, 2.0)

        # ---- Global targets ----
        elif target == ModTarget.MASTER_VOLUME and synthesizer is not None:
            base = synthesizer.master_volume_db
            synthesizer.set_master_volume(np.clip(base + mod_value, -60.0, 10.0))
        elif target == ModTarget.MORPH and synthesizer is not None:
            if hasattr(synthesizer, '_morph_manager'):
                base = synthesizer._morph_manager._position
                synthesizer._morph_manager.set_position(
                    np.clip(base + mod_value, 0.0, 1.0))
