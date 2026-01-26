"""
Envelope Generator for Pythonic
Implements exponential and linear envelope shapes with attack and decay
"""

import numpy as np
from enum import Enum


class EnvelopeStage(Enum):
    IDLE = 0
    ATTACK = 1
    DECAY = 2
    DONE = 3


class Envelope:
    """
    Attack-Decay envelope generator
    Supports exponential and linear curves
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        
        # Parameters (in milliseconds)
        self.attack_ms = 0.0
        self.decay_ms = 316.23  # Default from Pythonic
        
        # State
        self.stage = EnvelopeStage.IDLE
        self.current_level = 0.0
        self.sample_index = 0
        self.is_active = False
        
        # Cached values
        self._attack_samples = 0
        self._decay_samples = 0
        self._attack_increment = 0.0
        self._decay_coefficient = 0.0
        
        self._update_coefficients()
    
    def _update_coefficients(self):
        """Recalculate envelope timing coefficients"""
        # Attack samples (minimum 1 to avoid division by zero)
        self._attack_samples = max(1, int(self.attack_ms * self.sr / 1000.0))
        self._attack_increment = 1.0 / self._attack_samples
        
        # Decay samples
        self._decay_samples = max(1, int(self.decay_ms * self.sr / 1000.0))
        
        # Exponential decay coefficient (reaches ~0.001 at end)
        # Using time constant: level = exp(-t/tau) where tau = decay_samples/6.9
        self._decay_coefficient = np.exp(-6.9 / self._decay_samples)
    
    def set_attack(self, attack_ms: float):
        """Set attack time in milliseconds (0-10000)"""
        self.attack_ms = np.clip(attack_ms, 0.0, 10000.0)
        self._update_coefficients()
    
    def set_decay(self, decay_ms: float):
        """Set decay time in milliseconds (10-10000)"""
        self.decay_ms = np.clip(decay_ms, 10.0, 10000.0)
        self._update_coefficients()
    
    def trigger(self):
        """Start the envelope from the beginning"""
        self.is_active = True
        self.sample_index = 0
        
        if self.attack_ms > 0:
            self.stage = EnvelopeStage.ATTACK
            self.current_level = 0.0
        else:
            self.stage = EnvelopeStage.DECAY
            self.current_level = 1.0
    
    def release(self):
        """Force release (not used in Pythonic's AD envelope)"""
        self.stage = EnvelopeStage.DONE
        self.is_active = False
    
    def process(self, num_samples: int) -> np.ndarray:
        """
        Generate envelope samples
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            numpy array of envelope values (0.0 to 1.0)
        """
        output = np.zeros(num_samples, dtype=np.float32)
        
        if not self.is_active:
            return output
        
        for i in range(num_samples):
            if self.stage == EnvelopeStage.ATTACK:
                # Linear attack
                self.current_level += self._attack_increment
                if self.current_level >= 1.0:
                    self.current_level = 1.0
                    self.stage = EnvelopeStage.DECAY
                    self.sample_index = 0
                output[i] = self.current_level
                
            elif self.stage == EnvelopeStage.DECAY:
                # Exponential decay
                self.current_level *= self._decay_coefficient
                output[i] = self.current_level
                self.sample_index += 1
                
                # Check if envelope has essentially finished
                if self.current_level < 0.0001:
                    self.stage = EnvelopeStage.DONE
                    self.is_active = False
                    
            elif self.stage == EnvelopeStage.DONE or self.stage == EnvelopeStage.IDLE:
                output[i] = 0.0
                
        return output
    
    def process_sample(self) -> float:
        """Process single sample (for sample-by-sample processing)"""
        if not self.is_active:
            return 0.0
            
        if self.stage == EnvelopeStage.ATTACK:
            self.current_level += self._attack_increment
            if self.current_level >= 1.0:
                self.current_level = 1.0
                self.stage = EnvelopeStage.DECAY
                self.sample_index = 0
            return self.current_level
            
        elif self.stage == EnvelopeStage.DECAY:
            self.current_level *= self._decay_coefficient
            self.sample_index += 1
            
            if self.current_level < 0.0001:
                self.stage = EnvelopeStage.DONE
                self.is_active = False
                
            return self.current_level
            
        return 0.0


class ModulatedEnvelope(Envelope):
    """
    Envelope with amplitude modulation (for handclap sounds)
    """
    
    def __init__(self, sample_rate: int = 44100):
        super().__init__(sample_rate)
        self.mod_frequency = 50.0  # Hz
        self.mod_phase = 0.0
    
    def set_mod_frequency(self, freq: float):
        """Set modulation frequency in Hz"""
        self.mod_frequency = np.clip(freq, 0.0, 100.0)
    
    def process(self, num_samples: int) -> np.ndarray:
        """Generate modulated envelope samples"""
        # Get base envelope
        base_env = super().process(num_samples)
        
        if not self.is_active and np.max(base_env) < 0.0001:
            return base_env
        
        # Apply amplitude modulation
        t = np.arange(num_samples) / self.sr + self.mod_phase
        modulation = 0.5 + 0.5 * np.sin(2.0 * np.pi * self.mod_frequency * t)
        
        # Update phase
        self.mod_phase = t[-1] if num_samples > 0 else self.mod_phase
        
        return base_env * modulation
