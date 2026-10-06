"""
Sound addresses of the app core: every per-channel sound parameter of a drum
patch, named ``ch<N>.<group>.<param>`` with N = 1..8 (``ch3.osc.decay``).

Values are in engine units: Hz, ms, dB, semitones (``st``), and enum names
(the engine enum member names in lower case, such as ``sine`` or
``low_pass``). Two unit names need a word:

- ``ratio``: a 0..1 (or 0..2) fraction, shown as a percentage (0.5 is 50 %).
- ``%``: a value already in percent (LFO depth, 0..100).

Ranges are the ranges of the controls (the tkinter widgets); defaults are the
values of a freshly created channel. ``SOUND_PARAMS`` is the single table:
the core registers it for each channel, and Edit all fans out over it.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from pythonic.delay import DelayTime
from pythonic.lfo import LFOPolarity, LFORetrigger, LFOWaveform, ModTarget, SyncDivision
from pythonic.noise import NoiseEnvelopeMode, NoiseFilterMode
from pythonic.oscillator import PitchModMode, WaveformType


@dataclass(frozen=True)
class SoundParam:
    """One per-channel sound parameter.

    ``suffix`` is the address after ``ch<N>.``; ``key`` is the engine's preset
    key (``DrumChannel.get_parameters()``, ``lfo1.depth`` for nested ones).
    ``get(channel)`` / ``set(channel, value)`` work on a ``DrumChannel``.
    """

    suffix: str
    key: str
    get: Callable[[Any], Any]
    set: Callable[[Any, Any], None]
    kind: str = 'float'
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    unit: str = ''
    curve: str = 'linear'
    labels: Sequence[str] = field(default_factory=tuple)


def enum_labels(enum_cls):
    return tuple(member.name.lower() for member in enum_cls)


def _number(suffix, key, get, set, minimum, maximum, unit='', curve='linear'):
    return SoundParam(suffix, key, lambda ch: float(get(ch)), set, 'float',
                      float(minimum), float(maximum), unit, curve)


def _ratio(suffix, key, get, set, maximum=1.0):
    return _number(suffix, key, get, set, 0.0, maximum, 'ratio')


def _flag(suffix, key, get, set):
    return SoundParam(suffix, key, lambda ch: bool(get(ch)), lambda ch, v: set(ch, bool(v)),
                      'bool')


def _enum(suffix, key, enum_cls, get, set):
    """An engine enum, addressed by its lower-case member name."""
    return SoundParam(suffix, key,
                      lambda ch: get(ch).name.lower(),
                      lambda ch, v: set(ch, enum_cls[v.upper()]),
                      'enum', labels=enum_labels(enum_cls))


def _set_attr(path, attr):
    def setter(ch, value):
        setattr(path(ch), attr, value)
    return setter


def _modulation_params(name):
    """Addresses of one modulation source (lfo1, lfo2) of a channel."""
    def src(ch):
        return getattr(ch, name)

    return [
        _flag(f'{name}.on', f'{name}.enabled', lambda ch: src(ch).enabled,
              _set_attr(src, 'enabled')),
        _enum(f'{name}.wave', f'{name}.waveform', LFOWaveform,
              lambda ch: src(ch).waveform, _set_attr(src, 'waveform')),
        _number(f'{name}.rate', f'{name}.rate_hz', lambda ch: src(ch).rate_hz,
                _set_attr(src, 'rate_hz'), 0.01, 50.0, 'Hz', 'log'),
        _number(f'{name}.depth', f'{name}.depth', lambda ch: src(ch).depth,
                _set_attr(src, 'depth'), 0.0, 100.0, '%'),
        _enum(f'{name}.sync', f'{name}.sync', SyncDivision,
              lambda ch: src(ch).sync, _set_attr(src, 'sync')),
        _flag(f'{name}.retrig', f'{name}.retrigger',
              lambda ch: src(ch).retrigger == LFORetrigger.RETRIGGER,
              lambda ch, v: setattr(src(ch), 'retrigger',
                                    LFORetrigger.RETRIGGER if v else LFORetrigger.FREE)),
        _flag(f'{name}.unipolar', f'{name}.polarity',
              lambda ch: src(ch).polarity == LFOPolarity.UNIPOLAR,
              lambda ch, v: setattr(src(ch), 'polarity',
                                    LFOPolarity.UNIPOLAR if v else LFOPolarity.BIPOLAR)),
        _enum(f'{name}.target', f'{name}.target', ModTarget,
              lambda ch: src(ch).target, _set_attr(src, 'target')),
        _ratio(f'{name}.phase', f'{name}.phase_offset', lambda ch: src(ch).phase_offset,
               _set_attr(src, 'phase_offset')),
    ]


def _pump(ch):
    return ch.pump


SOUND_PARAMS = tuple([
    # Oscillator
    _enum('osc.wave', 'osc_waveform', WaveformType,
          lambda ch: ch.oscillator.waveform, lambda ch, v: ch.set_osc_waveform(v)),
    _number('osc.freq', 'osc_frequency', lambda ch: ch.oscillator.frequency,
            lambda ch, v: ch.set_osc_frequency(v), 20.0, 20000.0, 'Hz', 'log'),
    _number('osc.pitch', 'pitch_semitones', lambda ch: ch.pitch_semitones,
            lambda ch, v: ch.set_pitch_semitones(v), -24.0, 24.0, 'st'),
    _enum('osc.mod_mode', 'pitch_mod_mode', PitchModMode,
          lambda ch: ch.oscillator.pitch_mod_mode, lambda ch, v: ch.set_pitch_mod_mode(v)),
    # The engine clamps the amount further by mode (±96 st decaying, ±48 sine)
    _number('osc.mod_amount', 'pitch_mod_amount', lambda ch: ch.oscillator.pitch_mod_amount,
            lambda ch, v: ch.set_pitch_mod_amount(v), -120.0, 120.0, 'st'),
    # ms in decaying mode, Hz in sine and random mode
    _number('osc.mod_rate', 'pitch_mod_rate', lambda ch: ch.oscillator.pitch_mod_rate,
            lambda ch, v: ch.set_pitch_mod_rate(v), 1.0, 2000.0, '', 'log'),
    _number('osc.attack', 'osc_attack', lambda ch: ch._osc_attack_base_ms,
            lambda ch, v: ch.set_osc_attack(v), 0.0, 10000.0, 'ms', 'log'),
    _number('osc.decay', 'osc_decay', lambda ch: ch.osc_envelope.decay_ms,
            lambda ch, v: ch.set_osc_decay(v), 10.0, 10000.0, 'ms', 'log'),
    # Noise
    _enum('noise.filter', 'noise_filter_mode', NoiseFilterMode,
          lambda ch: ch.noise_gen.filter_mode, lambda ch, v: ch.set_noise_filter_mode(v)),
    _number('noise.freq', 'noise_filter_freq', lambda ch: ch._noise_filter_freq_base,
            lambda ch, v: ch.set_noise_filter_freq(v), 20.0, 20000.0, 'Hz', 'log'),
    _number('noise.q', 'noise_filter_q', lambda ch: ch.noise_gen.filter_q,
            lambda ch, v: ch.set_noise_filter_q(v), 0.5, 20.0, '', 'log'),
    _flag('noise.stereo', 'noise_stereo', lambda ch: ch.noise_gen.stereo,
          lambda ch, v: ch.set_noise_stereo(v)),
    _enum('noise.env', 'noise_envelope_mode', NoiseEnvelopeMode,
          lambda ch: ch.noise_gen.envelope_mode, lambda ch, v: ch.set_noise_envelope_mode(v)),
    _number('noise.attack', 'noise_attack', lambda ch: ch._noise_attack_base_ms,
            lambda ch, v: ch.set_noise_attack(v), 0.0, 10000.0, 'ms', 'log'),
    _number('noise.decay', 'noise_decay', lambda ch: ch.noise_gen.decay_ms,
            lambda ch, v: ch.set_noise_decay(v), 10.0, 10000.0, 'ms', 'log'),
    # Mixing (1 = all oscillator, 0 = all noise)
    _ratio('mix.osc_noise', 'osc_noise_mix', lambda ch: ch.osc_noise_mix,
           lambda ch, v: ch.set_osc_noise_mix(v)),
    _number('mix.level', 'level_db', lambda ch: ch.level_db,
            lambda ch, v: setattr(ch, 'level_db', v), -60.0, 10.0, 'dB'),
    _number('mix.pan', 'pan', lambda ch: ch.pan,
            lambda ch, v: setattr(ch, 'pan', v), -100.0, 100.0, 'pan'),
    _ratio('mix.distortion', 'distortion', lambda ch: ch.distortion,
           lambda ch, v: ch.set_distortion(v)),
    _flag('mix.choke', 'choke_enabled', lambda ch: ch.choke_enabled,
          lambda ch, v: setattr(ch, 'choke_enabled', v)),
    SoundParam('mix.output', 'output_pair', lambda ch: ch.output_pair,
               lambda ch, v: setattr(ch, 'output_pair', v), 'enum', labels=('A', 'B')),
    # EQ
    _number('eq.freq', 'eq_frequency', lambda ch: ch.eq_frequency,
            lambda ch, v: ch.set_eq_frequency(v), 20.0, 20000.0, 'Hz', 'log'),
    _number('eq.gain', 'eq_gain_db', lambda ch: ch.eq_gain_db,
            lambda ch, v: ch.set_eq_gain(v), -40.0, 40.0, 'dB'),
    # FX
    _ratio('fx.vintage', 'vintage_amount', lambda ch: ch.vintage_amount,
           lambda ch, v: setattr(ch, 'vintage_amount', v)),
    _ratio('fx.reverb_decay', 'reverb_decay', lambda ch: ch.reverb_decay,
           lambda ch, v: setattr(ch, 'reverb_decay', v)),
    _ratio('fx.reverb_mix', 'reverb_mix', lambda ch: ch.reverb_mix,
           lambda ch, v: setattr(ch, 'reverb_mix', v)),
    _ratio('fx.reverb_width', 'reverb_width', lambda ch: ch.reverb_width,
           lambda ch, v: setattr(ch, 'reverb_width', v), maximum=2.0),
    _enum('fx.delay_time', 'delay_time', DelayTime,
          lambda ch: ch.delay_time, lambda ch, v: setattr(ch, 'delay_time', v)),
    _ratio('fx.delay_feedback', 'delay_feedback', lambda ch: ch.delay_feedback,
           lambda ch, v: setattr(ch, 'delay_feedback', v), maximum=0.95),
    _ratio('fx.delay_mix', 'delay_mix', lambda ch: ch.delay_mix,
           lambda ch, v: setattr(ch, 'delay_mix', v)),
    _flag('fx.delay_pingpong', 'delay_ping_pong', lambda ch: ch.delay_ping_pong,
          lambda ch, v: setattr(ch, 'delay_ping_pong', v)),
    # Velocity sensitivity (1.0 = 100 %)
    _ratio('vel.osc', 'osc_vel_sensitivity', lambda ch: ch.osc_vel_sensitivity,
           lambda ch, v: setattr(ch, 'osc_vel_sensitivity', v), maximum=2.0),
    _ratio('vel.noise', 'noise_vel_sensitivity', lambda ch: ch.noise_vel_sensitivity,
           lambda ch, v: setattr(ch, 'noise_vel_sensitivity', v), maximum=2.0),
    _ratio('vel.mod', 'mod_vel_sensitivity', lambda ch: ch.mod_vel_sensitivity,
           lambda ch, v: setattr(ch, 'mod_vel_sensitivity', v), maximum=2.0),
    # Modulation
    *_modulation_params('lfo1'),
    *_modulation_params('lfo2'),
    _flag('pump.on', 'pump.enabled', lambda ch: ch.pump.enabled, _set_attr(_pump, 'enabled')),
    _ratio('pump.amount', 'pump.amount', lambda ch: ch.pump.amount, _set_attr(_pump, 'amount')),
    _number('pump.attack', 'pump.attack_ms', lambda ch: ch.pump.attack_ms,
            _set_attr(_pump, 'attack_ms'), 0.1, 100.0, 'ms', 'log'),
    _number('pump.release', 'pump.release_ms', lambda ch: ch.pump.release_ms,
            _set_attr(_pump, 'release_ms'), 1.0, 1000.0, 'ms', 'log'),
    _ratio('pump.curve', 'pump.curve', lambda ch: ch.pump.curve, _set_attr(_pump, 'curve')),
    _enum('pump.target', 'pump.target', ModTarget,
          lambda ch: ch.pump.target, _set_attr(_pump, 'target')),
    _enum('pump.sync', 'pump.sync', SyncDivision,
          lambda ch: ch.pump.sync, _set_attr(_pump, 'sync')),
])

SOUND_SUFFIXES = frozenset(p.suffix for p in SOUND_PARAMS)


# ---------------------------------------------------------------------------
# MIDI CC parameter names saved in preferences (``midi_cc_mappings``)
# ---------------------------------------------------------------------------

# Per-channel names: the address suffix, on the selected channel
CHANNEL_CC_PARAMETERS = {
    'level': 'mix.level', 'pan': 'mix.pan', 'distortion': 'mix.distortion',
    'osc_noise_mix': 'mix.osc_noise',
    'eq_freq': 'eq.freq', 'eq_gain': 'eq.gain',
    'vintage': 'fx.vintage', 'reverb_decay': 'fx.reverb_decay', 'reverb_mix': 'fx.reverb_mix',
    'reverb_width': 'fx.reverb_width', 'delay_feedback': 'fx.delay_feedback',
    'delay_mix': 'fx.delay_mix',
    'osc_freq': 'osc.freq', 'pitch': 'osc.pitch', 'pitch_amount': 'osc.mod_amount',
    'pitch_rate': 'osc.mod_rate', 'osc_attack': 'osc.attack', 'osc_decay': 'osc.decay',
    'noise_freq': 'noise.freq', 'noise_q': 'noise.q', 'noise_attack': 'noise.attack',
    'noise_decay': 'noise.decay',
    'osc_vel': 'vel.osc', 'noise_vel': 'vel.noise', 'mod_vel': 'vel.mod',
    'lfo1_rate': 'lfo1.rate', 'lfo1_depth': 'lfo1.depth',
    'lfo2_rate': 'lfo2.rate', 'lfo2_depth': 'lfo2.depth',
    'pump_amount': 'pump.amount', 'pump_attack': 'pump.attack',
    'pump_release': 'pump.release', 'pump_curve': 'pump.curve',
}

# Global names: the full address
GLOBAL_CC_PARAMETERS = {
    'master_volume': 'global.master',
    'sound_morph': 'morph.position',
}


# A controller target on the selected channel: ``selected.osc.decay`` is the
# osc.decay of whichever channel is selected when the message arrives
SELECTED_PREFIX = 'selected.'


def cc_parameter_target(name):
    """The controller target of a saved CC parameter name (global address or
    ``selected.<suffix>``)."""
    if name in GLOBAL_CC_PARAMETERS:
        return GLOBAL_CC_PARAMETERS[name]
    return SELECTED_PREFIX + CHANNEL_CC_PARAMETERS[name]


def resolve_target(target, selected_channel):
    """The address a controller target names now (selected_channel is 1..8)."""
    if target.startswith(SELECTED_PREFIX):
        return f'ch{selected_channel}.{target[len(SELECTED_PREFIX):]}'
    return target


def cc_parameter_address(name, selected_channel):
    """The address of a saved CC parameter name (selected_channel is 1..8)."""
    return resolve_target(cc_parameter_target(name), selected_channel)
