"""
Noise Generator for Pythonic
Implements filtered noise with multiple envelope modes
"""

import numpy as np
from enum import Enum
from .filter import StateVariableFilter, FilterMode
from .envelope import Envelope, ModulatedEnvelope


class NoiseFilterMode(Enum):
    LOW_PASS = 0
    BAND_PASS = 1
    HIGH_PASS = 2


class NoiseEnvelopeMode(Enum):
    EXPONENTIAL = 0  # Natural decay
    LINEAR = 1       # Gated effect
    MODULATED = 2    # Handclap effect


class NoiseGenerator:
    """
    Stereo noise generator with multi-mode filter and envelope
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        
        # Filter parameters
        self.filter_mode = NoiseFilterMode.LOW_PASS
        self.filter_frequency = 20000.0  # Hz (20-20000)
        self.filter_q = 0.707  # Butterworth default
        
        # Stereo mode
        self.stereo = False
        
        # Envelope mode and parameters
        self.envelope_mode = NoiseEnvelopeMode.EXPONENTIAL
        self.attack_ms = 0.0
        self.decay_ms = 316.23
        self.mod_frequency = 50.0  # For modulated mode
        
        # Internal components
        self.filter_left = StateVariableFilter(sample_rate)
        self.filter_right = StateVariableFilter(sample_rate)
        # Create envelope with exponential attack shape for EXPONENTIAL mode
        self.envelope = Envelope(sample_rate, attack_shape='exponential')
        self.mod_envelope = ModulatedEnvelope(sample_rate)
        
        # State
        self.is_active = False
        self.velocity_gain = 1.0
        
        # Random state for reproducible stereo decorrelation
        # Use the new Generator interface which supports out= parameter
        self._rng_left = np.random.default_rng(12345)
        self._rng_right = np.random.default_rng(67890)
        
        # Pre-allocated buffers for performance
        self._output_buffer = np.zeros((8192, 2), dtype=np.float32)
        self._noise_left = np.zeros(8192, dtype=np.float32)
        self._noise_right = np.zeros(8192, dtype=np.float32)
        
        self._update_filter()
        self._update_envelope()
    
    def _update_filter(self):
        """Update filter settings"""
        # Convert our mode to filter mode
        mode_map = {
            NoiseFilterMode.LOW_PASS: FilterMode.LOW_PASS,
            NoiseFilterMode.BAND_PASS: FilterMode.BAND_PASS,
            NoiseFilterMode.HIGH_PASS: FilterMode.HIGH_PASS
        }
        
        filter_mode = mode_map[self.filter_mode]
        
        # Q parameter is used directly with a minimum of 0.5
        # Q=0.5: gentle rolloff, Q=15: moderate resonance, Q=60: high resonance
        effective_q = max(0.5, self.filter_q)
        
        self.filter_left.set_frequency(self.filter_frequency)
        self.filter_left.set_q(effective_q)
        self.filter_left.set_mode(filter_mode)
        
        self.filter_right.set_frequency(self.filter_frequency)
        self.filter_right.set_q(effective_q)
        self.filter_right.set_mode(filter_mode)
    
    def _update_envelope(self):
        """Update envelope settings"""
        self.envelope.set_attack(self.attack_ms)
        self.envelope.set_decay(self.decay_ms)
        
        self.mod_envelope.set_attack(self.attack_ms)
        self.mod_envelope.set_decay(self.decay_ms)
        self.mod_envelope.set_mod_frequency(self.mod_frequency)
    
    def set_filter_mode(self, mode: NoiseFilterMode):
        """Set filter mode (LP, BP, HP)"""
        self.filter_mode = mode
        self._update_filter()
    
    def set_filter_frequency(self, freq: float):
        """Set filter cutoff/center frequency"""
        self.filter_frequency = np.clip(freq, 20.0, 20000.0)
        self._update_filter()
    
    def set_filter_q(self, q: float):
        """Set filter Q/resonance (direct Q value, typically 0.5 to 100)"""
        self.filter_q = np.clip(q, 0.0, 100.0)
        self._update_filter()
    
    def set_stereo(self, enabled: bool):
        """Enable/disable stereo mode"""
        self.stereo = enabled
    
    def set_envelope_mode(self, mode: NoiseEnvelopeMode):
        """Set envelope mode and update envelope attack shape accordingly"""
        self.envelope_mode = mode
        # Update attack shape based on envelope mode
        # EXPONENTIAL mode uses exponential attack (slow start, fast finish)
        # LINEAR mode uses linear attack
        # MODULATED mode uses the ModulatedEnvelope which has its own behavior
        if mode == NoiseEnvelopeMode.EXPONENTIAL:
            self.envelope.attack_shape = 'exponential'
        elif mode == NoiseEnvelopeMode.LINEAR:
            self.envelope.attack_shape = 'linear'
    
    def set_attack(self, attack_ms: float):
        """Set attack time in ms"""
        self.attack_ms = np.clip(attack_ms, 0.0, 10000.0)
        self._update_envelope()
    
    def set_decay(self, decay_ms: float):
        """Set decay time in ms (or mod frequency for modulated mode)"""
        if self.envelope_mode == NoiseEnvelopeMode.MODULATED:
            self.mod_frequency = np.clip(decay_ms, 0.0, 100.0)
        else:
            self.decay_ms = np.clip(decay_ms, 1.0, 10000.0)
        self._update_envelope()
    
    def set_velocity_gain(self, gain: float):
        """Set velocity-based gain"""
        self.velocity_gain = gain
    
    def trigger(self):
        """Trigger the noise generator"""
        self.is_active = True
        
        # Reset filters
        self.filter_left.reset()
        self.filter_right.reset()
        
        # Trigger appropriate envelope
        if self.envelope_mode == NoiseEnvelopeMode.MODULATED:
            self.mod_envelope.trigger()
        else:
            self.envelope.trigger()
    
    def process(self, num_samples: int) -> np.ndarray:
        """
        Generate filtered noise samples
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Stereo output array [num_samples, 2]
        """
        if not self.is_active:
            if num_samples <= len(self._output_buffer):
                result = self._output_buffer[:num_samples]
                result.fill(0)
                return result
            return np.zeros((num_samples, 2), dtype=np.float32)
        
        # Generate white noise using pre-allocated buffers
        if num_samples <= len(self._noise_left):
            noise_left = self._noise_left[:num_samples]
            noise_left[:] = self._rng_left.standard_normal(num_samples, dtype=np.float32)
        else:
            noise_left = self._rng_left.standard_normal(num_samples).astype(np.float32)
        
        if self.stereo:
            if num_samples <= len(self._noise_right):
                noise_right = self._noise_right[:num_samples]
                noise_right[:] = self._rng_right.standard_normal(num_samples, dtype=np.float32)
            else:
                noise_right = self._rng_right.standard_normal(num_samples).astype(np.float32)
        else:
            noise_right = noise_left
        
        # Apply filter
        filtered_left = self.filter_left.process(noise_left)
        filtered_right = self.filter_right.process(noise_right)
        
        # Apply filter normalization
        effective_q = max(0.5, self.filter_q)
        
        # Normalization to maintain consistent output level
        if self.filter_mode == NoiseFilterMode.LOW_PASS:
            norm_factor = np.sqrt(effective_q)
        elif self.filter_mode == NoiseFilterMode.HIGH_PASS:
            norm_factor = 0.5 * np.sqrt(effective_q)
        else:  # BAND_PASS
            norm_factor = 0.7 * effective_q
        
        # Apply normalization in-place
        filtered_left *= norm_factor
        filtered_right *= norm_factor
        
        # Get envelope
        if self.envelope_mode == NoiseEnvelopeMode.MODULATED:
            env = self.mod_envelope.process(num_samples)
            self.is_active = self.mod_envelope.is_active
        elif self.envelope_mode == NoiseEnvelopeMode.LINEAR:
            env = self._linear_envelope(num_samples)
        else:  # EXPONENTIAL
            env = self.envelope.process(num_samples)
            self.is_active = self.envelope.is_active
        
        # Apply envelope and velocity - use pre-allocated buffer
        if num_samples <= len(self._output_buffer):
            output = self._output_buffer[:num_samples]
        else:
            output = np.empty((num_samples, 2), dtype=np.float32)
        
        # Apply envelope and velocity gain together
        env_vel = env * self.velocity_gain
        output[:, 0] = filtered_left * env_vel
        output[:, 1] = filtered_right * env_vel
        
        return output
    
    def _linear_envelope(self, num_samples: int) -> np.ndarray:
        """Generate linear envelope"""
        env = self.envelope.process(num_samples)
        
        # Convert exponential to linear by taking square root
        # This approximates a linear decay
        return np.sqrt(np.maximum(env, 0.0))
