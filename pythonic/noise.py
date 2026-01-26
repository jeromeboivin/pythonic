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
        self.envelope = Envelope(sample_rate)
        self.mod_envelope = ModulatedEnvelope(sample_rate)
        
        # State
        self.is_active = False
        self.velocity_gain = 1.0
        
        # Random state for reproducible stereo decorrelation
        self._rng_left = np.random.RandomState(12345)
        self._rng_right = np.random.RandomState(67890)
        
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
        
        self.filter_left.set_frequency(self.filter_frequency)
        self.filter_left.set_q(self.filter_q)
        self.filter_left.set_mode(filter_mode)
        
        self.filter_right.set_frequency(self.filter_frequency)
        self.filter_right.set_q(self.filter_q)
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
        """Set filter Q/resonance"""
        self.filter_q = np.clip(q, 0.1, 10000.0)
        self._update_filter()
    
    def set_stereo(self, enabled: bool):
        """Enable/disable stereo mode"""
        self.stereo = enabled
    
    def set_envelope_mode(self, mode: NoiseEnvelopeMode):
        """Set envelope mode"""
        self.envelope_mode = mode
    
    def set_attack(self, attack_ms: float):
        """Set attack time in ms"""
        self.attack_ms = np.clip(attack_ms, 0.0, 10000.0)
        self._update_envelope()
    
    def set_decay(self, decay_ms: float):
        """Set decay time in ms (or mod frequency for modulated mode)"""
        if self.envelope_mode == NoiseEnvelopeMode.MODULATED:
            self.mod_frequency = np.clip(decay_ms, 0.0, 100.0)
        else:
            self.decay_ms = np.clip(decay_ms, 10.0, 10000.0)
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
            return np.zeros((num_samples, 2), dtype=np.float32)
        
        # Generate white noise
        noise_left = self._rng_left.randn(num_samples).astype(np.float32)
        
        if self.stereo:
            noise_right = self._rng_right.randn(num_samples).astype(np.float32)
        else:
            noise_right = noise_left.copy()
        
        # Apply filter
        filtered_left = self.filter_left.process(noise_left)
        filtered_right = self.filter_right.process(noise_right)
        
        # Get envelope
        if self.envelope_mode == NoiseEnvelopeMode.MODULATED:
            env = self.mod_envelope.process(num_samples)
            self.is_active = self.mod_envelope.is_active
        elif self.envelope_mode == NoiseEnvelopeMode.LINEAR:
            env = self._linear_envelope(num_samples)
        else:  # EXPONENTIAL
            env = self.envelope.process(num_samples)
            self.is_active = self.envelope.is_active
        
        # Apply envelope and velocity
        output = np.zeros((num_samples, 2), dtype=np.float32)
        output[:, 0] = filtered_left * env * self.velocity_gain
        output[:, 1] = filtered_right * env * self.velocity_gain
        
        return output
    
    def _linear_envelope(self, num_samples: int) -> np.ndarray:
        """Generate linear envelope"""
        env = self.envelope.process(num_samples)
        
        # Convert exponential to linear by taking square root
        # This approximates a linear decay
        return np.sqrt(np.maximum(env, 0.0))
