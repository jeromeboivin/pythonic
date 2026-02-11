"""
Drum Channel for Pythonic
Complete drum voice with oscillator, noise, mixing, and effects
"""

import numpy as np
from .oscillator import Oscillator, WaveformType, PitchModMode
from .noise import NoiseGenerator, NoiseFilterMode, NoiseEnvelopeMode
from .envelope import Envelope
from .filter import EQFilter
from .vintage import VintageProcessor
from .reverb import FastStereoReverb
from .delay import FastStereoDelay, DelayTime
from .smoothed_parameter import SmoothedParameter, LogSmoothedParameter


class DrumChannel:
    """
    Complete drum synthesis channel
    Combines oscillator, noise generator, mixing, distortion, and EQ
    
    Parameter Smoothing:
    Frequency and filter parameters use smoothed values to prevent
    clicks and zippering artifacts when adjusting in real-time.
    """
    
    # Internal headroom
    INTERNAL_HEADROOM_DB = 0.0
    INTERNAL_HEADROOM_LINEAR = 10.0 ** (INTERNAL_HEADROOM_DB / 20.0)  # 1.0
    
    # Oscillator level scaling relative to noise
    # Oscillator level is lower than a full-scale waveform
    # This scaling matches the observed output ratio
    OSC_LEVEL_SCALING = 0.27
    
    # Default smoothing time constant (ms)
    DEFAULT_SMOOTHING_MS = 30.0
    
    def __init__(self, channel_id: int, sample_rate: int = 44100):
        self.channel_id = channel_id
        self.sr = sample_rate
        
        # Smoothing time constant (can be changed globally)
        self._smoothing_ms = self.DEFAULT_SMOOTHING_MS
        
        # Components
        self.oscillator = Oscillator(sample_rate)
        self.noise_gen = NoiseGenerator(sample_rate)
        self.osc_envelope = Envelope(sample_rate)
        self.eq_filter_l = EQFilter(sample_rate)
        self.eq_filter_r = EQFilter(sample_rate)
        self.vintage = VintageProcessor(sample_rate)
        
        # Per-channel stereo reverb effect (using FastStereoReverb for real-time performance)
        self.reverb = FastStereoReverb(sample_rate)
        
        # Per-channel tempo-synced delay/echo effect
        self.delay = FastStereoDelay(sample_rate)
        
        # Vintage amount (0.0 to 1.0) - simulates analog circuit behavior
        self.vintage_amount = 0.0
        
        # Reverb parameters (per-channel decay/reverb stereo effect)
        self.reverb_decay = 0.0   # 0.0 to 1.0 (reverb time)
        self.reverb_mix = 0.0     # 0.0 to 1.0 (dry/wet)
        self.reverb_width = 1.0   # 0.0 to 2.0 (stereo width)
        
        # Delay/echo parameters (tempo-synced)
        self.delay_time = DelayTime.EIGHTH  # Default to 1/8 note
        self.delay_feedback = 0.3           # 0.0 to 0.95
        self.delay_mix = 0.0                # 0.0 to 1.0 (dry/wet)
        self.delay_ping_pong = False        # Stereo ping-pong mode
        
        # Mixing parameters
        self.osc_noise_mix = 0.5  # 0 = all noise, 1 = all oscillator
        self.level_db = 0.0  # Output level in dB (-inf to +10)
        self.pan = 0.0  # Stereo pan (-100 to +100)
        
        # Distortion
        self.distortion = 0.0  # 0 to 1
        
        # EQ (raw target values - smoothing applied in process)
        self.eq_frequency = 632.46  # Hz
        self.eq_gain_db = 0.0  # dB (-40 to +40)
        
        # Smoothed parameters for click-free real-time control
        self._smoothed_osc_freq = LogSmoothedParameter(
            initial_value=440.0, time_constant_ms=self._smoothing_ms,
            sample_rate=sample_rate, min_val=20.0, max_val=20000.0
        )
        self._smoothed_noise_freq = LogSmoothedParameter(
            initial_value=20000.0, time_constant_ms=self._smoothing_ms,
            sample_rate=sample_rate, min_val=20.0, max_val=20000.0
        )
        self._smoothed_noise_q = LogSmoothedParameter(
            initial_value=0.707, time_constant_ms=self._smoothing_ms,
            sample_rate=sample_rate, min_val=0.5, max_val=20.0
        )
        self._smoothed_eq_freq = LogSmoothedParameter(
            initial_value=632.46, time_constant_ms=self._smoothing_ms,
            sample_rate=sample_rate, min_val=20.0, max_val=20000.0
        )
        self._smoothed_distortion = SmoothedParameter(
            initial_value=0.0, time_constant_ms=self._smoothing_ms,
            sample_rate=sample_rate, min_val=0.0, max_val=1.0
        )
        self._smoothed_mix = SmoothedParameter(
            initial_value=0.5, time_constant_ms=self._smoothing_ms,
            sample_rate=sample_rate, min_val=0.0, max_val=1.0
        )
        
        # Choke and output
        self.choke_enabled = False
        self.output_pair = 'A'  # 'A' or 'B'
        self.muted = False
        
        # Velocity sensitivity (0 to 2.0, where 1.0 = 100%)
        self.osc_vel_sensitivity = 0.0
        self.noise_vel_sensitivity = 0.0
        self.mod_vel_sensitivity = 0.0
        
        # Pitch offset in semitones (-24 to +24, applied to oscillator frequency)
        self.pitch_semitones = 0.0
        
        # Probability (0-100, chance of triggering during pattern playback)
        self.probability = 100
        
        # State
        self.is_active = False
        self.current_velocity = 1.0
        self.name = f"Channel {channel_id + 1}"
        
        # Pre-allocated buffers for performance
        self._output_buffer = np.zeros((8192, 2), dtype=np.float32)
        self._osc_stereo_buffer = np.zeros((8192, 2), dtype=np.float32)
        self._mixed_buffer = np.zeros((8192, 2), dtype=np.float32)
        
        # Initialize default drum sound
        self._init_defaults()
    
    def _init_defaults(self):
        """Set default parameters for a basic drum sound"""
        # Oscillator defaults
        self.oscillator.set_frequency(200.0)
        self.oscillator.set_waveform(WaveformType.SINE)
        self.oscillator.set_pitch_mod_mode(PitchModMode.DECAYING)
        self.oscillator.set_pitch_mod_amount(0.0)
        self.oscillator.set_pitch_mod_rate(100.0)
        
        # Envelope defaults
        self.osc_envelope.set_attack(0.0)
        self.osc_envelope.set_decay(316.23)
        
        # Noise defaults
        self.noise_gen.set_filter_mode(NoiseFilterMode.LOW_PASS)
        self.noise_gen.set_filter_frequency(20000.0)
        self.noise_gen.set_filter_q(0.707)
        self.noise_gen.set_stereo(False)
        self.noise_gen.set_attack(0.0)
        self.noise_gen.set_decay(316.23)
        
        # EQ defaults
        self.eq_filter_l.set_frequency(632.46)
        self.eq_filter_l.set_gain(0.0)
        self.eq_filter_l.set_q(2.8)
        self.eq_filter_r.set_frequency(632.46)
        self.eq_filter_r.set_gain(0.0)
        self.eq_filter_r.set_q(2.8)
    
    def trigger(self, velocity: int = 127, note: int = 60):
        """
        Trigger the drum channel
        
        Args:
            velocity: MIDI velocity (0-127)
            note: MIDI note number (for pitched mode)
        """
        self.is_active = True
        self.current_velocity = velocity / 127.0
        
        # Calculate velocity-modified gains
        vel_factor = self.current_velocity
        
        # Velocity sensitivity power multiplier
        # Uses an aggressive power curve: gain = vel^(sens * k)
        # where k ≈ 5 based on reference analysis
        _VEL_POWER_MULT = 5.0
        
        # Oscillator velocity
        osc_vel_gain = 1.0
        if self.osc_vel_sensitivity > 0:
            osc_vel_gain = vel_factor ** (self.osc_vel_sensitivity * _VEL_POWER_MULT)
        self.oscillator.set_velocity_gain(osc_vel_gain)
        
        # Noise velocity
        noise_vel_gain = 1.0
        if self.noise_vel_sensitivity > 0:
            noise_vel_gain = vel_factor ** (self.noise_vel_sensitivity * _VEL_POWER_MULT)
        self.noise_gen.set_velocity_gain(noise_vel_gain)
        
        # Modulation velocity (affects pitch mod amount)
        mod_vel_scale = 1.0
        if self.mod_vel_sensitivity > 0:
            mod_vel_scale = vel_factor ** (self.mod_vel_sensitivity * _VEL_POWER_MULT)
        self.oscillator.set_velocity_mod_scale(mod_vel_scale)
        
        # Reset and trigger components
        self.oscillator.reset_phase()
        self.osc_envelope.trigger()
        self.noise_gen.trigger()
        self.vintage.reset()
    
    def process(self, num_samples: int) -> np.ndarray:
        """
        Process and generate audio
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Stereo output array [num_samples, 2]
        """
        if not self.is_active or self.muted:
            # Return slice of pre-allocated zero buffer
            result = self._output_buffer[:num_samples]
            result.fill(0)
            return result
        
        # Update smoothed parameters (get current smoothed values)
        # These advance the smoothing filters per-block for efficiency
        smoothed_osc_freq = self._smoothed_osc_freq.get_next_value()
        smoothed_noise_freq = self._smoothed_noise_freq.get_next_value()
        smoothed_noise_q = self._smoothed_noise_q.get_next_value()
        smoothed_eq_freq = self._smoothed_eq_freq.get_next_value()
        smoothed_distortion = self._smoothed_distortion.get_next_value()
        smoothed_mix = self._smoothed_mix.get_next_value()
        
        # Apply smoothed values to components only when changed
        # (avoids expensive filter coefficient recomputation every block)
        # Apply pitch offset (semitones) as a frequency multiplier
        # Shifts both oscillator and noise filter frequency so the entire
        # drum character moves (important for noise-heavy sounds like snares)
        if self.pitch_semitones != 0.0:
            pitch_ratio = 2.0 ** (self.pitch_semitones / 12.0)
            effective_osc_freq = smoothed_osc_freq * pitch_ratio
            effective_noise_freq = np.clip(smoothed_noise_freq * pitch_ratio, 20.0, 20000.0)
        else:
            pitch_ratio = 1.0
            effective_osc_freq = smoothed_osc_freq
            effective_noise_freq = smoothed_noise_freq
        if not self._smoothed_osc_freq.is_settled() or pitch_ratio != 1.0:
            self.oscillator.set_frequency(effective_osc_freq)
        if not self._smoothed_noise_freq.is_settled() or not self._smoothed_noise_q.is_settled() or pitch_ratio != 1.0:
            self.noise_gen.set_filter_frequency(effective_noise_freq)
            self.noise_gen.set_filter_q(smoothed_noise_q)
        
        # Scale pitch modulation time with pitch ratio so higher-pitched
        # drums have faster pitch sweeps (natural drum behavior)
        self.oscillator.mod_time_scale = pitch_ratio
        
        # Apply vintage pitch drift if enabled
        if self.vintage_amount > 0.001:
            self.vintage.set_amount(self.vintage_amount)
            pitch_drift_mult = self.vintage.get_pitch_multiplier()
            self.oscillator.set_pitch_drift(pitch_drift_mult)
        else:
            self.oscillator.set_pitch_drift(1.0)
        
        # Generate oscillator signal
        osc_raw = self.oscillator.process(num_samples)
        osc_env = self.osc_envelope.process(num_samples)
        osc_signal = osc_raw * osc_env
        osc_signal *= self.OSC_LEVEL_SCALING
        
        # Generate noise signal (already stereo)
        noise_signal = self.noise_gen.process(num_samples)
        
        # Mix oscillator (mono) with noise (stereo)
        # Use pre-allocated buffer
        if num_samples <= len(self._osc_stereo_buffer):
            osc_stereo = self._osc_stereo_buffer[:num_samples]
        else:
            osc_stereo = np.empty((num_samples, 2), dtype=np.float32)
        
        osc_stereo[:, 0] = osc_signal
        osc_stereo[:, 1] = osc_signal
        
        # Mix based on smoothed osc_noise_mix parameter using power-scaled crossfade
        mix = smoothed_mix
        if mix > 0.999:
            # Pure oscillator - reuse osc_stereo
            mixed = osc_stereo
        elif mix < 0.001:
            # Pure noise
            mixed = noise_signal
        else:
            # Power-preserving squared crossfade
            osc_gain = mix * mix
            noise_gain = (1.0 - mix) * (1.0 - mix)
            # Use pre-allocated buffer for mixed output
            if num_samples <= len(self._mixed_buffer):
                mixed = self._mixed_buffer[:num_samples]
            else:
                mixed = np.empty((num_samples, 2), dtype=np.float32)
            # In-place operations
            np.multiply(osc_stereo, osc_gain, out=mixed)
            mixed += noise_signal * noise_gain
        
        # Apply smoothed distortion
        if smoothed_distortion > 0.001:
            self.distortion = smoothed_distortion  # Update for _apply_distortion
            mixed = self._apply_distortion(mixed)
        
        # Apply EQ with smoothed frequency (process left and right separately)
        if abs(self.eq_gain_db) > 0.1:
            # Only update EQ params when smoothed frequency is still changing
            if not self._smoothed_eq_freq.is_settled():
                self.eq_filter_l.set_frequency(smoothed_eq_freq)
                self.eq_filter_r.set_frequency(smoothed_eq_freq)
            
            # Process left channel
            mixed[:, 0] = self.eq_filter_l.process(mixed[:, 0])
            
            # Process right channel
            mixed[:, 1] = self.eq_filter_r.process(mixed[:, 1])
        
        # Apply level (dB to linear) with internal headroom
        if self.level_db <= -60:
            level_linear = 0.0
        else:
            level_linear = 10.0 ** (self.level_db / 20.0)
        # Apply internal headroom (in-place)
        gain = level_linear * self.INTERNAL_HEADROOM_LINEAR
        np.multiply(mixed, gain, out=mixed)
        
        # Apply vintage analog simulation (after level, before pan)
        if self.vintage_amount > 0.001:
            self.vintage.set_amount(self.vintage_amount)
            mixed = self.vintage.process(mixed)
        
        # Apply per-channel tempo-synced delay/echo (after vintage)
        if self.delay_mix > 0.001:
            self.delay.set_delay_time(self.delay_time)
            self.delay.set_feedback(self.delay_feedback)
            self.delay.set_mix(self.delay_mix)
            self.delay.set_ping_pong(self.delay_ping_pong)
            mixed = self.delay.process(mixed)
        
        # Apply per-channel stereo reverb (after delay, before pan)
        if self.reverb_mix > 0.001:
            self.reverb.set_decay(self.reverb_decay)
            self.reverb.set_mix(self.reverb_mix)
            self.reverb.set_width(self.reverb_width)
            mixed = self.reverb.process(mixed)
        
        # Apply pan (in-place where possible)
        output = self._apply_pan_inplace(mixed)
        
        # Check if voice is done
        if not self.osc_envelope.is_active and not self.noise_gen.is_active:
            self.is_active = False
        
        return output
    
    def _apply_distortion(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply distortion using soft clipping
        
        Distortion generates strong odd harmonics while maintaining
        overall signal level. Uses tanh for smooth saturation.
        
        Distortion levels:
        - 0-25%: Low drive (1-3), subtle saturation
        - 50%+: High drive (50+), approaches hard clipping
        - Uses exponential drive curve for natural response
        
        Args:
            signal: Input stereo signal
            
        Returns:
            Distorted signal
        """
        if self.distortion < 0.001:
            return signal
        
        # Exponential drive curve:
        # Formula: drive = exp(2.5 * distortion) gives:
        #   0% -> 1.0, 50% -> 3.5, 100% -> 12.2
        drive = np.exp(2.5 * self.distortion)
        
        # Pre-gain the signal
        driven = signal * drive
        
        # Soft clipping using tanh - generates odd harmonics
        clipped = np.tanh(driven)
        
        # Normalize to maintain consistent output level
        # tanh(drive) is the max output for a +1 input, use this to scale back
        normalization = np.tanh(drive)
        if normalization > 0.1:
            clipped = clipped / normalization
        
        # Full wet at any distortion amount
        return clipped
    
    def _apply_pan(self, stereo_signal: np.ndarray) -> np.ndarray:
        """
        Apply stereo panning using linear pan law
        
        Args:
            stereo_signal: Input stereo signal [samples, 2]
            
        Returns:
            Panned stereo signal
        """
        # Convert pan (-100 to +100) to normalized (-1 to 1)
        pan_normalized = self.pan / 100.0
        
        # Linear pan law: at center (0), both channels at 1.0
        # At full left (-1), left=1.0, right=0.0
        # At full right (+1), left=0.0, right=1.0
        if pan_normalized <= 0:
            # Panning left
            left_gain = 1.0
            right_gain = 1.0 + pan_normalized  # 0 at full left, 1 at center
        else:
            # Panning right
            left_gain = 1.0 - pan_normalized  # 1 at center, 0 at full right
            right_gain = 1.0
        
        output = np.zeros_like(stereo_signal)
        output[:, 0] = stereo_signal[:, 0] * left_gain
        output[:, 1] = stereo_signal[:, 1] * right_gain
        
        return output
    
    def _apply_pan_inplace(self, stereo_signal: np.ndarray) -> np.ndarray:
        """
        Apply stereo panning in-place using linear pan law
        
        Args:
            stereo_signal: Input stereo signal [samples, 2] - MODIFIED IN PLACE
            
        Returns:
            Panned stereo signal (same array)
        """
        # Convert pan (-100 to +100) to normalized (-1 to 1)
        pan_normalized = self.pan / 100.0
        
        # Linear pan law
        if pan_normalized <= 0:
            right_gain = 1.0 + pan_normalized
            if right_gain != 1.0:
                stereo_signal[:, 1] *= right_gain
        else:
            left_gain = 1.0 - pan_normalized
            if left_gain != 1.0:
                stereo_signal[:, 0] *= left_gain
        
        return stereo_signal
        
        output = np.zeros_like(stereo_signal)
        output[:, 0] = stereo_signal[:, 0] * left_gain
        output[:, 1] = stereo_signal[:, 1] * right_gain
        
        return output
    
    # Smoothing configuration
    def set_smoothing_time(self, time_ms: float):
        """Set the smoothing time constant for all smoothed parameters.
        
        Args:
            time_ms: Smoothing time in milliseconds (20-50 recommended)
        """
        self._smoothing_ms = max(0.0, time_ms)
        self._smoothed_osc_freq.set_time_constant(self._smoothing_ms)
        self._smoothed_noise_freq.set_time_constant(self._smoothing_ms)
        self._smoothed_noise_q.set_time_constant(self._smoothing_ms)
        self._smoothed_eq_freq.set_time_constant(self._smoothing_ms)
        self._smoothed_distortion.set_time_constant(self._smoothing_ms)
        self._smoothed_mix.set_time_constant(self._smoothing_ms)
    
    def get_smoothing_time(self) -> float:
        """Get the current smoothing time constant in milliseconds."""
        return self._smoothing_ms
    
    # Parameter setters for GUI binding
    def set_pitch_semitones(self, semitones: float):
        """Set pitch offset in semitones (-24 to +24)
        
        Applies a musical pitch shift to the oscillator frequency.
        Does not affect the noise generator.
        """
        self.pitch_semitones = np.clip(semitones, -24.0, 24.0)
    
    def set_osc_frequency(self, freq: float):
        """Set oscillator frequency (smoothed)"""
        self._smoothed_osc_freq.set_target(freq)
        # Also set immediately for non-realtime use
        self.oscillator.set_frequency(freq)
    
    def set_osc_frequency_immediate(self, freq: float):
        """Set oscillator frequency immediately (no smoothing)"""
        self._smoothed_osc_freq.set_immediate(freq)
        self.oscillator.set_frequency(freq)
    
    def set_osc_waveform(self, waveform: WaveformType):
        """Set oscillator waveform"""
        self.oscillator.set_waveform(waveform)
    
    def set_pitch_mod_mode(self, mode: PitchModMode):
        """Set pitch modulation mode"""
        self.oscillator.set_pitch_mod_mode(mode)
    
    def set_pitch_mod_amount(self, amount: float):
        """Set pitch modulation amount"""
        self.oscillator.set_pitch_mod_amount(amount)
    
    def set_pitch_mod_rate(self, rate: float):
        """Set pitch modulation rate"""
        self.oscillator.set_pitch_mod_rate(rate)
    
    def set_osc_attack(self, attack_ms: float):
        """Set oscillator envelope attack"""
        self.osc_envelope.set_attack(attack_ms)
    
    def set_osc_decay(self, decay_ms: float):
        """Set oscillator envelope decay"""
        self.osc_envelope.set_decay(decay_ms)
    
    def set_noise_filter_mode(self, mode: NoiseFilterMode):
        """Set noise filter mode"""
        self.noise_gen.set_filter_mode(mode)
    
    def set_noise_filter_freq(self, freq: float):
        """Set noise filter frequency (smoothed)"""
        self._smoothed_noise_freq.set_target(freq)
        # Also set immediately for non-realtime use
        self.noise_gen.set_filter_frequency(freq)
    
    def set_noise_filter_freq_immediate(self, freq: float):
        """Set noise filter frequency immediately (no smoothing)"""
        self._smoothed_noise_freq.set_immediate(freq)
        self.noise_gen.set_filter_frequency(freq)
    
    def set_noise_filter_q(self, q: float):
        """Set noise filter Q (smoothed)"""
        self._smoothed_noise_q.set_target(q)
        # Also set immediately for non-realtime use
        self.noise_gen.set_filter_q(q)
    
    def set_noise_filter_q_immediate(self, q: float):
        """Set noise filter Q immediately (no smoothing)"""
        self._smoothed_noise_q.set_immediate(q)
        self.noise_gen.set_filter_q(q)
    
    def set_noise_stereo(self, enabled: bool):
        """Set noise stereo mode"""
        self.noise_gen.set_stereo(enabled)
    
    def set_noise_envelope_mode(self, mode: NoiseEnvelopeMode):
        """Set noise envelope mode"""
        self.noise_gen.set_envelope_mode(mode)
    
    def set_noise_attack(self, attack_ms: float):
        """Set noise envelope attack"""
        self.noise_gen.set_attack(attack_ms)
    
    def set_noise_decay(self, decay_ms: float):
        """Set noise envelope decay"""
        self.noise_gen.set_decay(decay_ms)
    
    def set_osc_noise_mix(self, mix: float):
        """Set oscillator/noise mix (smoothed)
        
        Args:
            mix: 0.0 = all noise, 1.0 = all oscillator
        """
        mix = np.clip(mix, 0.0, 1.0)
        self.osc_noise_mix = mix
        self._smoothed_mix.set_target(mix)
    
    def set_osc_noise_mix_immediate(self, mix: float):
        """Set oscillator/noise mix immediately (no smoothing)"""
        mix = np.clip(mix, 0.0, 1.0)
        self.osc_noise_mix = mix
        self._smoothed_mix.set_immediate(mix)
    
    def set_eq_frequency(self, freq: float):
        """Set EQ frequency (smoothed)"""
        self.eq_frequency = np.clip(freq, 20.0, 20000.0)
        self._smoothed_eq_freq.set_target(self.eq_frequency)
    
    def set_eq_frequency_immediate(self, freq: float):
        """Set EQ frequency immediately (no smoothing)"""
        self.eq_frequency = np.clip(freq, 20.0, 20000.0)
        self._smoothed_eq_freq.set_immediate(self.eq_frequency)
    
    def set_distortion(self, amount: float):
        """Set distortion amount (smoothed)"""
        amount = np.clip(amount, 0.0, 1.0)
        self.distortion = amount
        self._smoothed_distortion.set_target(amount)
    
    def set_distortion_immediate(self, amount: float):
        """Set distortion amount immediately (no smoothing)"""
        amount = np.clip(amount, 0.0, 1.0)
        self.distortion = amount
        self._smoothed_distortion.set_immediate(amount)
    
    def set_bpm(self, bpm: float):
        """Set BPM for tempo-synced delay effect"""
        self.delay.set_bpm(bpm)
    
    def get_parameters(self) -> dict:
        """Get all parameters as a dictionary"""
        return {
            'name': self.name,
            'pitch_semitones': self.pitch_semitones,
            'osc_frequency': self.oscillator.frequency,
            'osc_waveform': self.oscillator.waveform.value,
            'pitch_mod_mode': self.oscillator.pitch_mod_mode.value,
            'pitch_mod_amount': self.oscillator.pitch_mod_amount,
            'pitch_mod_rate': self.oscillator.pitch_mod_rate,
            'osc_attack': self.osc_envelope.attack_ms,
            'osc_decay': self.osc_envelope.decay_ms,
            'noise_filter_mode': self.noise_gen.filter_mode.value,
            'noise_filter_freq': self.noise_gen.filter_frequency,
            'noise_filter_q': self.noise_gen.filter_q,
            'noise_stereo': self.noise_gen.stereo,
            'noise_envelope_mode': self.noise_gen.envelope_mode.value,
            'noise_attack': self.noise_gen.attack_ms,
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
        }
    
    def set_parameters(self, params: dict, immediate: bool = True):
        """Set all parameters from a dictionary
        
        Args:
            params: Dictionary of parameter values
            immediate: If True, set values immediately without smoothing (for preset loading)
        """
        if 'name' in params:
            self.name = params['name']
        if 'pitch_semitones' in params:
            self.set_pitch_semitones(params['pitch_semitones'])
        if 'osc_frequency' in params:
            if immediate:
                self.set_osc_frequency_immediate(params['osc_frequency'])
            else:
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
            if immediate:
                self.set_noise_filter_freq_immediate(params['noise_filter_freq'])
            else:
                self.set_noise_filter_freq(params['noise_filter_freq'])
        if 'noise_filter_q' in params:
            if immediate:
                self.set_noise_filter_q_immediate(params['noise_filter_q'])
            else:
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
            if immediate:
                self.set_osc_noise_mix_immediate(params['osc_noise_mix'])
            else:
                self.set_osc_noise_mix(params['osc_noise_mix'])
        if 'distortion' in params:
            if immediate:
                self.set_distortion_immediate(params['distortion'])
            else:
                self.set_distortion(params['distortion'])
        if 'eq_frequency' in params:
            if immediate:
                self.set_eq_frequency_immediate(params['eq_frequency'])
            else:
                self.set_eq_frequency(params['eq_frequency'])
        if 'eq_gain_db' in params:
            self.eq_gain_db = np.clip(params['eq_gain_db'], -40.0, 40.0)
        if 'level_db' in params:
            self.level_db = np.clip(params['level_db'], -60.0, 40.0)
        if 'pan' in params:
            self.pan = np.clip(params['pan'], -100.0, 100.0)
        if 'choke_enabled' in params:
            self.choke_enabled = params['choke_enabled']
        if 'output_pair' in params:
            self.output_pair = params['output_pair']
        if 'osc_vel_sensitivity' in params:
            self.osc_vel_sensitivity = np.clip(params['osc_vel_sensitivity'], 0.0, 2.0)
        if 'noise_vel_sensitivity' in params:
            self.noise_vel_sensitivity = np.clip(params['noise_vel_sensitivity'], 0.0, 2.0)
        if 'mod_vel_sensitivity' in params:
            self.mod_vel_sensitivity = np.clip(params['mod_vel_sensitivity'], 0.0, 2.0)
        if 'vintage_amount' in params:
            self.vintage_amount = np.clip(params['vintage_amount'], 0.0, 1.0)
        if 'reverb_decay' in params:
            self.reverb_decay = np.clip(params['reverb_decay'], 0.0, 1.0)
        if 'reverb_mix' in params:
            self.reverb_mix = np.clip(params['reverb_mix'], 0.0, 1.0)
        if 'reverb_width' in params:
            self.reverb_width = np.clip(params['reverb_width'], 0.0, 2.0)
        if 'delay_time' in params:
            self.delay_time = DelayTime(int(params['delay_time']))
        if 'delay_feedback' in params:
            self.delay_feedback = np.clip(params['delay_feedback'], 0.0, 0.95)
        if 'delay_mix' in params:
            self.delay_mix = np.clip(params['delay_mix'], 0.0, 1.0)
        if 'delay_ping_pong' in params:
            self.delay_ping_pong = bool(params['delay_ping_pong'])
