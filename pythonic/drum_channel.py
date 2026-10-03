"""
Drum Channel for Pythonic
Complete drum voice with oscillator, noise, mixing, and effects.

The sound itself is produced by :class:`pythonic.voice.DrumVoice`. The oscillator / noise / envelope
objects hold the parameters so the GUI, presets and the
drum generator keep working against the same attributes.

Pythonic extras (LFO/pump modulation, vintage, delay, reverb) are applied
around the voice.
"""

import numpy as np

from .oscillator import Oscillator, WaveformType, PitchModMode
from .noise import NoiseGenerator, NoiseFilterMode, NoiseEnvelopeMode
from .envelope import Envelope
from .lfo import LFO, PumpSource, ModTarget, LFORetrigger
from .vintage import VintageProcessor
from .reverb import FastStereoReverb
from .delay import FastStereoDelay, DelayTime
from .voice import DrumVoice, VoiceParams


class DrumChannel:
    """One of the 8 drum channels."""

    DEFAULT_SMOOTHING_MS = 30.0

    def __init__(self, channel_id: int, sample_rate: int = 44100):
        self.channel_id = channel_id
        self.sr = sample_rate

        # Parameter holders
        self.oscillator = Oscillator()
        self.noise_gen = NoiseGenerator()
        self.osc_envelope = Envelope()

        # Voice engine
        self.voice = DrumVoice(sample_rate, seed=channel_id + 1)

        # Pythonic extras
        self.vintage = VintageProcessor(sample_rate)
        self.reverb = FastStereoReverb(sample_rate)
        self.delay = FastStereoDelay(sample_rate)
        self.vintage_amount = 0.0
        self.reverb_decay = 0.0
        self.reverb_mix = 0.0
        self.reverb_width = 1.0
        self.delay_time = DelayTime.EIGHTH
        self.delay_feedback = 0.3
        self.delay_mix = 0.0
        self.delay_ping_pong = False
        self._prev_delay_time = None
        self._prev_delay_feedback = None
        self._prev_delay_mix = None
        self._prev_delay_pp = None
        self._prev_reverb_decay = None
        self._prev_reverb_mix = None
        self._prev_reverb_width = None

        # Mixing parameters
        self.osc_noise_mix = 0.5   # 0 = all noise, 1 = all oscillator
        self.level_db = 0.0
        self.pan = 0.0             # -100..100
        self.distortion = 0.0      # 0..1
        self.eq_frequency = 632.46
        self.eq_gain_db = 0.0
        self._osc_attack_base_ms = 0.0
        self._noise_filter_freq_base = 20000.0
        self._noise_attack_base_ms = 0.0
        self._smoothing_ms = self.DEFAULT_SMOOTHING_MS

        self.mono = False
        self.choke_enabled = False
        self.output_pair = 'A'
        self.muted = False

        # Velocity sensitivity (0..2, 1.0 = 100 %)
        self.osc_vel_sensitivity = 0.0
        self.noise_vel_sensitivity = 0.0
        self.mod_vel_sensitivity = 0.0

        # Pitch offset in semitones (scales oscillator, noise filter and EQ)
        self.pitch_semitones = 0.0

        self.is_active = False
        self.current_velocity = 1.0
        self.name = f"Channel {channel_id + 1}"

        self._zeros = np.zeros((8192, 2), dtype=np.float32)

        # Modulation: 2 LFOs + 1 pump source per channel
        self.lfo1 = LFO(sample_rate)
        self.lfo2 = LFO(sample_rate)
        self.pump = PumpSource(sample_rate)
        self._synthesizer = None
        self._global_mod_offsets = {}
        self._last_mod_offsets = {}

        self._init_defaults()

    def _init_defaults(self):
        self.oscillator.set_frequency(200.0)
        self.oscillator.set_waveform(WaveformType.SINE)
        self.oscillator.set_pitch_mod_mode(PitchModMode.DECAYING)
        self.oscillator.set_pitch_mod_amount(0.0)
        self.oscillator.set_pitch_mod_rate(100.0)
        self.set_osc_attack(0.0)
        self.osc_envelope.set_decay(316.23)
        self.noise_gen.set_filter_mode(NoiseFilterMode.LOW_PASS)
        self.set_noise_filter_freq_immediate(20000.0)
        self.noise_gen.set_filter_q(0.707)
        self.noise_gen.set_stereo(False)
        self.set_noise_attack(0.0)
        self.noise_gen.set_decay(316.23)

    # ------------------------------------------------------------------ voice params
    def _voice_params(self, mod=None) -> VoiceParams:
        """Snapshot of the channel parameters (with modulation offsets) for the voice."""
        osc = self.oscillator
        noise = self.noise_gen
        osc_freq = osc.frequency
        noise_freq = self._noise_filter_freq_base
        noise_q = noise.filter_q
        eq_freq = self.eq_frequency
        eq_gain = self.eq_gain_db
        dist = self.distortion
        mix = self.osc_noise_mix
        level = self.level_db
        pan = self.pan
        pitch = self.pitch_semitones
        mod_amt = osc.pitch_mod_amount
        mod_rate = osc.pitch_mod_rate
        osc_atk = self._osc_attack_base_ms
        osc_dcy = self.osc_envelope.decay_ms
        n_atk = self._noise_attack_base_ms
        n_dcy = noise.decay_ms
        ovel, nvel, mvel = self.osc_vel_sensitivity, self.noise_vel_sensitivity, self.mod_vel_sensitivity
        if mod:
            g = mod.get
            if ModTarget.OSC_FREQUENCY in mod:
                osc_freq = min(20000.0, max(20.0, osc_freq + g(ModTarget.OSC_FREQUENCY)))
            if ModTarget.NOISE_FILTER_FREQ in mod:
                noise_freq = min(20000.0, max(20.0, noise_freq + g(ModTarget.NOISE_FILTER_FREQ)))
            if ModTarget.NOISE_FILTER_Q in mod:
                noise_q = min(10000.0, max(0.1, noise_q + g(ModTarget.NOISE_FILTER_Q)))
            if ModTarget.EQ_FREQUENCY in mod:
                eq_freq = min(20000.0, max(20.0, eq_freq + g(ModTarget.EQ_FREQUENCY)))
            if ModTarget.EQ_GAIN_DB in mod:
                eq_gain = min(40.0, max(-40.0, eq_gain + g(ModTarget.EQ_GAIN_DB)))
            if ModTarget.DISTORTION in mod:
                dist = min(1.0, max(0.0, dist + g(ModTarget.DISTORTION)))
            if ModTarget.OSC_NOISE_MIX in mod:
                mix = min(1.0, max(0.0, mix + g(ModTarget.OSC_NOISE_MIX)))
            if ModTarget.LEVEL_DB in mod:
                level = min(40.0, max(-60.0, level + g(ModTarget.LEVEL_DB)))
            if ModTarget.PAN in mod:
                pan = min(100.0, max(-100.0, pan + g(ModTarget.PAN)))
            if ModTarget.PITCH_SEMITONES in mod:
                pitch = min(48.0, max(-48.0, pitch + g(ModTarget.PITCH_SEMITONES)))
            if ModTarget.PITCH_MOD_AMOUNT in mod:
                mod_amt = mod_amt + g(ModTarget.PITCH_MOD_AMOUNT)
            if ModTarget.PITCH_MOD_RATE in mod:
                mod_rate = max(0.1, mod_rate + g(ModTarget.PITCH_MOD_RATE))
            if ModTarget.OSC_ATTACK in mod:
                osc_atk = max(0.0, osc_atk + g(ModTarget.OSC_ATTACK))
            if ModTarget.OSC_DECAY in mod:
                osc_dcy = max(1.0, osc_dcy + g(ModTarget.OSC_DECAY))
            if ModTarget.NOISE_ATTACK in mod:
                n_atk = max(0.0, n_atk + g(ModTarget.NOISE_ATTACK))
            if ModTarget.NOISE_DECAY in mod:
                n_dcy = max(1.0, n_dcy + g(ModTarget.NOISE_DECAY))
            if ModTarget.OSC_VEL_SENSITIVITY in mod:
                ovel = min(2.0, max(0.0, ovel + g(ModTarget.OSC_VEL_SENSITIVITY)))
            if ModTarget.NOISE_VEL_SENSITIVITY in mod:
                nvel = min(2.0, max(0.0, nvel + g(ModTarget.NOISE_VEL_SENSITIVITY)))
            if ModTarget.MOD_VEL_SENSITIVITY in mod:
                mvel = min(2.0, max(0.0, mvel + g(ModTarget.MOD_VEL_SENSITIVITY)))
        return VoiceParams(
            wave=osc.waveform.value,
            osc_freq=float(osc_freq),
            osc_attack_ms=float(osc_atk),
            osc_decay_ms=float(osc_dcy),
            mod_mode=osc.pitch_mod_mode.value,
            mod_rate=float(mod_rate),
            mod_amount=float(mod_amt),
            noise_filter=noise.filter_mode.value,
            noise_freq=float(noise_freq),
            noise_q=float(noise_q),
            noise_stereo=bool(noise.stereo),
            noise_env=noise.envelope_mode.value,
            noise_attack_ms=float(n_atk),
            noise_decay_ms=float(n_dcy),
            osc_mix=float(mix),
            distortion=float(dist),
            eq_freq=float(eq_freq),
            eq_gain_db=float(eq_gain),
            level_db=float(level),
            pan=float(pan),
            osc_vel=float(ovel),
            noise_vel=float(nvel),
            mod_vel=float(mvel),
            pitch_ratio=float(2.0 ** (pitch / 12.0)),
            mono=bool(self.mono),
        )

    # ------------------------------------------------------------------ playback
    def trigger(self, velocity: int = 127, note: int = 60):
        """Trigger the drum (velocity 0..127)."""
        self.is_active = True
        self.current_velocity = velocity / 127.0
        self.voice.set_params(self._voice_params(self._last_mod_offsets or None))
        self.voice.trigger(velocity)
        self.vintage.reset()
        if self.lfo1.retrigger == LFORetrigger.RETRIGGER:
            self.lfo1.reset_phase()
        if self.lfo2.retrigger == LFORetrigger.RETRIGGER:
            self.lfo2.reset_phase()
        self.pump.trigger()
        self.reverb.reset()
        self.delay.reset()

    def choke(self):
        """Fade the voice out over 10 ms (choke group)."""
        if self.is_active:
            self.voice.choke()

    def process(self, num_samples: int) -> np.ndarray:
        """Render `num_samples` stereo samples."""
        if not self.is_active or self.muted:
            self.voice.idle(num_samples)
            if num_samples <= len(self._zeros):
                return self._zeros[:num_samples]
            return np.zeros((num_samples, 2), dtype=np.float32)

        # LFO / pump modulation (block rate)
        mod = {}
        if ((self.lfo1.enabled and self.lfo1.target != ModTarget.NONE)
                or (self.lfo2.enabled and self.lfo2.target != ModTarget.NONE)
                or (self.pump.enabled and self.pump.target != ModTarget.NONE)):
            bpm = getattr(self._synthesizer, '_bpm', 120.0) if self._synthesizer else 120.0
            for src in (self.lfo1, self.lfo2, self.pump):
                if src.enabled and src.target != ModTarget.NONE:
                    v = src.process(num_samples, bpm)
                    if v != 0.0:
                        mod[src.target] = mod.get(src.target, 0.0) + v
        self._global_mod_offsets = {t: mod[t] for t in (ModTarget.MASTER_VOLUME, ModTarget.MORPH) if t in mod}
        self._last_mod_offsets = dict(mod)

        vintage = self.vintage_amount
        if ModTarget.VINTAGE_AMOUNT in mod:
            vintage = min(1.0, max(0.0, vintage + mod[ModTarget.VINTAGE_AMOUNT]))
        if vintage > 0.001:
            self.vintage.set_amount(vintage)
            self.voice.pitch_drift = self.vintage.get_pitch_multiplier()
        else:
            self.voice.pitch_drift = 1.0

        self.voice.set_params(self._voice_params(mod or None))
        out = self.voice.process(num_samples)

        if vintage > 0.001:
            out = self.vintage.process(out)

        delay_mix = self.delay_mix
        delay_fb = self.delay_feedback
        if ModTarget.DELAY_MIX in mod:
            delay_mix = min(1.0, max(0.0, delay_mix + mod[ModTarget.DELAY_MIX]))
        if ModTarget.DELAY_FEEDBACK in mod:
            delay_fb = min(0.95, max(0.0, delay_fb + mod[ModTarget.DELAY_FEEDBACK]))
        if delay_mix > 0.001:
            if (self.delay_time != self._prev_delay_time or delay_fb != self._prev_delay_feedback
                    or delay_mix != self._prev_delay_mix or self.delay_ping_pong != self._prev_delay_pp):
                self.delay.set_delay_time(self.delay_time)
                self.delay.set_feedback(delay_fb)
                self.delay.set_mix(delay_mix)
                self.delay.set_ping_pong(self.delay_ping_pong)
                self._prev_delay_time = self.delay_time
                self._prev_delay_feedback = delay_fb
                self._prev_delay_mix = delay_mix
                self._prev_delay_pp = self.delay_ping_pong
            out = self.delay.process(out)

        rv_mix = self.reverb_mix
        rv_decay = self.reverb_decay
        rv_width = self.reverb_width
        if ModTarget.REVERB_MIX in mod:
            rv_mix = min(1.0, max(0.0, rv_mix + mod[ModTarget.REVERB_MIX]))
        if ModTarget.REVERB_DECAY in mod:
            rv_decay = min(1.0, max(0.0, rv_decay + mod[ModTarget.REVERB_DECAY]))
        if ModTarget.REVERB_WIDTH in mod:
            rv_width = min(2.0, max(0.0, rv_width + mod[ModTarget.REVERB_WIDTH]))
        if rv_mix > 0.001:
            width = 0.0 if self.mono else rv_width
            if (rv_decay != self._prev_reverb_decay or rv_mix != self._prev_reverb_mix
                    or width != self._prev_reverb_width):
                self.reverb.set_decay(rv_decay)
                self.reverb.set_mix(rv_mix)
                self.reverb.set_width(width)
                self._prev_reverb_decay = rv_decay
                self._prev_reverb_mix = rv_mix
                self._prev_reverb_width = width
            out = self.reverb.process(out)

        if self.mono:
            m = (out[:, 0] + out[:, 1]) * 0.5
            out[:, 0] = m
            out[:, 1] = m

        if not self.voice.is_active:
            self.is_active = False
        return out

    # ------------------------------------------------------------------ smoothing (API compat)
    def set_smoothing_time(self, time_ms: float):
        """Kept for API compatibility; the voice smooths gains internally."""
        self._smoothing_ms = max(0.0, time_ms)

    def get_smoothing_time(self) -> float:
        return self._smoothing_ms

    # ------------------------------------------------------------------ setters
    def set_pitch_semitones(self, semitones: float):
        """Pitch offset in semitones (-24..24); shifts osc, noise filter and EQ."""
        self.pitch_semitones = float(np.clip(semitones, -24.0, 24.0))

    def set_osc_frequency(self, freq: float):
        self.oscillator.set_frequency(freq)

    def set_osc_frequency_immediate(self, freq: float):
        self.oscillator.set_frequency(freq)

    def set_osc_waveform(self, waveform: WaveformType):
        self.oscillator.set_waveform(waveform)

    def set_pitch_mod_mode(self, mode: PitchModMode):
        self.oscillator.set_pitch_mod_mode(mode)

    def set_pitch_mod_amount(self, amount: float):
        self.oscillator.set_pitch_mod_amount(amount)

    def set_pitch_mod_rate(self, rate: float):
        """Pitch modulation rate: ms in Decay mode, Hz in Sine / Noise mode."""
        self.oscillator.set_pitch_mod_rate(rate)

    def set_osc_attack(self, attack_ms: float):
        self._osc_attack_base_ms = float(np.clip(attack_ms, 0.0, 10000.0))
        self.osc_envelope.set_attack(self._osc_attack_base_ms)

    def set_osc_decay(self, decay_ms: float):
        self.osc_envelope.set_decay(decay_ms)

    def set_noise_filter_mode(self, mode: NoiseFilterMode):
        self.noise_gen.set_filter_mode(mode)

    def set_noise_filter_freq(self, freq: float):
        self._noise_filter_freq_base = float(np.clip(freq, 20.0, 20000.0))
        self.noise_gen.set_filter_frequency(self._noise_filter_freq_base)

    def set_noise_filter_freq_immediate(self, freq: float):
        self.set_noise_filter_freq(freq)

    def set_noise_filter_q(self, q: float):
        self.noise_gen.set_filter_q(q)

    def set_noise_filter_q_immediate(self, q: float):
        self.noise_gen.set_filter_q(q)

    def set_noise_stereo(self, enabled: bool):
        self.noise_gen.set_stereo(enabled)

    def set_noise_envelope_mode(self, mode: NoiseEnvelopeMode):
        self.noise_gen.set_envelope_mode(mode)

    def set_noise_attack(self, attack_ms: float):
        self._noise_attack_base_ms = float(np.clip(attack_ms, 0.0, 10000.0))
        self.noise_gen.set_attack(self._noise_attack_base_ms)

    def set_noise_decay(self, decay_ms: float):
        self.noise_gen.set_decay(decay_ms)

    def set_osc_noise_mix(self, mix: float):
        """0.0 = all noise, 1.0 = all oscillator."""
        self.osc_noise_mix = float(np.clip(mix, 0.0, 1.0))

    def set_osc_noise_mix_immediate(self, mix: float):
        self.set_osc_noise_mix(mix)

    def set_eq_frequency(self, freq: float):
        self.eq_frequency = float(np.clip(freq, 20.0, 20000.0))

    def set_eq_frequency_immediate(self, freq: float):
        self.set_eq_frequency(freq)

    def set_eq_gain(self, gain_db: float):
        self.eq_gain_db = float(np.clip(gain_db, -40.0, 40.0))

    def set_distortion(self, amount: float):
        self.distortion = float(np.clip(amount, 0.0, 1.0))

    def set_distortion_immediate(self, amount: float):
        self.set_distortion(amount)

    def set_bpm(self, bpm: float):
        """Set BPM for the tempo-synced delay effect."""
        self.delay.set_bpm(bpm)

    # ------------------------------------------------------------------ (de)serialisation
    def get_parameters(self) -> dict:
        return {
            'name': self.name,
            'pitch_semitones': self.pitch_semitones,
            'osc_frequency': self.oscillator.frequency,
            'osc_waveform': self.oscillator.waveform.value,
            'pitch_mod_mode': self.oscillator.pitch_mod_mode.value,
            'pitch_mod_amount': self.oscillator.pitch_mod_amount,
            'pitch_mod_rate': self.oscillator.pitch_mod_rate,
            'osc_attack': self._osc_attack_base_ms,
            'osc_decay': self.osc_envelope.decay_ms,
            'noise_filter_mode': self.noise_gen.filter_mode.value,
            'noise_filter_freq': self._noise_filter_freq_base,
            'noise_filter_q': self.noise_gen.filter_q,
            'noise_stereo': self.noise_gen.stereo,
            'noise_envelope_mode': self.noise_gen.envelope_mode.value,
            'noise_attack': self._noise_attack_base_ms,
            'noise_decay': self.noise_gen.decay_ms,
            'osc_noise_mix': self.osc_noise_mix,
            'distortion': self.distortion,
            'eq_frequency': self.eq_frequency,
            'eq_gain_db': self.eq_gain_db,
            'level_db': self.level_db,
            'pan': self.pan,
            'choke_enabled': self.choke_enabled,
            'output_pair': self.output_pair,
            'osc_vel_sensitivity': self.osc_vel_sensitivity,
            'noise_vel_sensitivity': self.noise_vel_sensitivity,
            'mod_vel_sensitivity': self.mod_vel_sensitivity,
            'vintage_amount': self.vintage_amount,
            'reverb_decay': self.reverb_decay,
            'reverb_mix': self.reverb_mix,
            'reverb_width': self.reverb_width,
            'delay_time': self.delay_time.value,
            'delay_feedback': self.delay_feedback,
            'delay_mix': self.delay_mix,
            'delay_ping_pong': self.delay_ping_pong,
            'lfo1': self.lfo1.get_parameters(),
            'lfo2': self.lfo2.get_parameters(),
            'pump': self.pump.get_parameters(),
        }

    def set_parameters(self, params: dict, immediate: bool = True):
        """Set all parameters from a dictionary (preset loading)."""
        if 'name' in params:
            self.name = params['name']
        if 'pitch_semitones' in params:
            self.set_pitch_semitones(params['pitch_semitones'])
        if 'osc_frequency' in params:
            self.set_osc_frequency(params['osc_frequency'])
        if 'osc_waveform' in params:
            self.set_osc_waveform(WaveformType(params['osc_waveform']))
        if 'pitch_mod_mode' in params:
            self.set_pitch_mod_mode(PitchModMode(params['pitch_mod_mode']))
        if 'pitch_mod_amount' in params:
            self.set_pitch_mod_amount(params['pitch_mod_amount'])
        if 'pitch_mod_rate' in params:
            self.set_pitch_mod_rate(params['pitch_mod_rate'])
        if 'osc_attack' in params:
            self.set_osc_attack(params['osc_attack'])
        if 'osc_decay' in params:
            self.set_osc_decay(params['osc_decay'])
        if 'noise_filter_mode' in params:
            self.set_noise_filter_mode(NoiseFilterMode(params['noise_filter_mode']))
        if 'noise_filter_freq' in params:
            self.set_noise_filter_freq(params['noise_filter_freq'])
        if 'noise_filter_q' in params:
            self.set_noise_filter_q(params['noise_filter_q'])
        if 'noise_stereo' in params:
            self.set_noise_stereo(params['noise_stereo'])
        if 'noise_envelope_mode' in params:
            self.set_noise_envelope_mode(NoiseEnvelopeMode(params['noise_envelope_mode']))
        if 'noise_attack' in params:
            self.set_noise_attack(params['noise_attack'])
        if 'noise_decay' in params:
            self.set_noise_decay(params['noise_decay'])
        if 'osc_noise_mix' in params:
            self.set_osc_noise_mix(params['osc_noise_mix'])
        if 'distortion' in params:
            self.set_distortion(params['distortion'])
        if 'eq_frequency' in params:
            self.set_eq_frequency(params['eq_frequency'])
        if 'eq_gain_db' in params:
            self.set_eq_gain(params['eq_gain_db'])
        if 'level_db' in params:
            self.level_db = float(np.clip(params['level_db'], -60.0, 40.0))
        if 'pan' in params:
            self.pan = float(np.clip(params['pan'], -100.0, 100.0))
        if 'choke_enabled' in params:
            self.choke_enabled = bool(params['choke_enabled'])
        if 'output_pair' in params:
            self.output_pair = params['output_pair']
        if 'osc_vel_sensitivity' in params:
            self.osc_vel_sensitivity = float(np.clip(params['osc_vel_sensitivity'], 0.0, 2.0))
        if 'noise_vel_sensitivity' in params:
            self.noise_vel_sensitivity = float(np.clip(params['noise_vel_sensitivity'], 0.0, 2.0))
        if 'mod_vel_sensitivity' in params:
            self.mod_vel_sensitivity = float(np.clip(params['mod_vel_sensitivity'], 0.0, 2.0))
        if 'vintage_amount' in params:
            self.vintage_amount = float(np.clip(params['vintage_amount'], 0.0, 1.0))
        if 'reverb_decay' in params:
            self.reverb_decay = float(np.clip(params['reverb_decay'], 0.0, 1.0))
        if 'reverb_mix' in params:
            self.reverb_mix = float(np.clip(params['reverb_mix'], 0.0, 1.0))
        if 'reverb_width' in params:
            self.reverb_width = float(np.clip(params['reverb_width'], 0.0, 2.0))
        if 'delay_time' in params:
            self.delay_time = DelayTime(int(params['delay_time']))
        if 'delay_feedback' in params:
            self.delay_feedback = float(np.clip(params['delay_feedback'], 0.0, 0.95))
        if 'delay_mix' in params:
            self.delay_mix = float(np.clip(params['delay_mix'], 0.0, 1.0))
        if 'delay_ping_pong' in params:
            self.delay_ping_pong = bool(params['delay_ping_pong'])
        if 'lfo1' in params:
            self.lfo1.set_parameters(params['lfo1'])
        if 'lfo2' in params:
            self.lfo2.set_parameters(params['lfo2'])
        if 'pump' in params:
            self.pump.set_parameters(params['pump'])
