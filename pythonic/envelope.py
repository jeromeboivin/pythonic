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
    
    When attack time is 0, the envelope starts at full level immediately.
    Click prevention is handled by the oscillator starting at phase 0 (zero crossing).
    When attack time > 0, uses an S-curve (smoothstep) for smooth attack.
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
        # Attack samples (0 means instant full level)
        self._attack_samples = max(0, int(self.attack_ms * self.sr / 1000.0))
        if self._attack_samples > 0:
            self._attack_increment = 1.0 / self._attack_samples
        else:
            self._attack_increment = 1.0
        
        # Decay samples
        self._decay_samples = max(1, int(self.decay_ms * self.sr / 1000.0))
        
        # Exponential decay coefficient
        # Using time constant: level = exp(-t/tau) where tau = decay_samples/K
        # 
        # K varies based on decay time:
        # - Short decays (<100ms): K=9 for punchy attack/decay
        # - Medium decays (100-500ms): K=9 for natural percussion
        # - Long decays (500-1500ms): K=9 to 12 (gradual transition)
        # - Very long decays (>1500ms): K=12 to 16 for faster tail cutoff
        #
        # This interpolation helps decay behavior across
        # different drum types (kicks with long decay vs hats with short decay)
        if self.decay_ms <= 500:
            K = 9.0
        elif self.decay_ms <= 1500:
            # Linear interpolation from 9 to 12 over 500-1500ms range
            K = 9.0 + (self.decay_ms - 500) / 1000 * 3.0
        else:
            # Linear interpolation from 12 to 16 over 1500-5000ms range
            K = 12.0 + min((self.decay_ms - 1500) / 3500 * 4.0, 4.0)
        
        self._decay_coefficient = np.exp(-K / self._decay_samples)
    
    def set_attack(self, attack_ms: float):
        """Set attack time in milliseconds (0-10000)
        
        When set to 0, envelope starts at full level immediately.
        Click prevention relies on oscillator starting at zero-crossing phase.
        """
        self.attack_ms = np.clip(attack_ms, 0.0, 10000.0)
        self._update_coefficients()
    
    def set_decay(self, decay_ms: float):
        """Set decay time in milliseconds (1-10000)"""
        self.decay_ms = np.clip(decay_ms, 1.0, 10000.0)
        self._update_coefficients()
    
    def trigger(self):
        """Start the envelope from the beginning
        
        If attack_ms > 0, starts in ATTACK stage.
        If attack_ms = 0, jumps straight to DECAY at full level.
        """
        self.is_active = True
        self.sample_index = 0
        
        if self._attack_samples > 0:
            self.stage = EnvelopeStage.ATTACK
            self.current_level = 0.0
        else:
            self.stage = EnvelopeStage.DECAY
            self.current_level = 1.0
    
    def release(self):
        """Force release (not used in Pythonic's AD envelope)"""
        self.stage = EnvelopeStage.DONE
        self.is_active = False
    
    @staticmethod
    def _smoothstep(t: np.ndarray) -> np.ndarray:
        """S-curve attack shape: 3t² - 2t³
        
        This produces a smooth attack that:
        - Starts slow (no click)
        - Accelerates through the middle
        - Approaches peak smoothly
        """
        return 3 * t * t - 2 * t * t * t
    
    def process(self, num_samples: int) -> np.ndarray:
        """
        Generate envelope samples (vectorized)
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            numpy array of envelope values (0.0 to 1.0)
        """
        if not self.is_active:
            return np.zeros(num_samples, dtype=np.float32)
        
        output = np.zeros(num_samples, dtype=np.float32)
        idx = 0
        
        # Process attack phase
        if self.stage == EnvelopeStage.ATTACK and idx < num_samples:
            if self._attack_samples > 0:
                # Calculate samples remaining in attack
                # sample_index tracks position within attack phase
                remaining_attack = self._attack_samples - self.sample_index
                attack_samples = min(remaining_attack, num_samples - idx)
                
                if attack_samples > 0:
                    # S-curve attack: smooth start, accelerates, smooth finish
                    # Uses smoothstep formula: 3t² - 2t³
                    sample_positions = np.arange(self.sample_index, self.sample_index + attack_samples)
                    normalized_time = sample_positions / self._attack_samples
                    levels = self._smoothstep(normalized_time)
                    levels = np.minimum(levels, 1.0)
                    output[idx:idx + attack_samples] = levels
                    self.current_level = levels[-1]
                    self.sample_index += attack_samples
                    idx += attack_samples
                    
                    if self.sample_index >= self._attack_samples:
                        self.current_level = 1.0
                        self.stage = EnvelopeStage.DECAY
                        self.sample_index = 0
            else:
                # Should not reach here due to minimum attack, but handle gracefully
                self.current_level = 1.0
                self.stage = EnvelopeStage.DECAY
                self.sample_index = 0
        
        # Process decay phase
        if self.stage == EnvelopeStage.DECAY and idx < num_samples:
            decay_samples = num_samples - idx
            
            # Vectorized exponential decay: level * coeff^n
            # Start from n=0 so first sample is at full level (coeff^0 = 1)
            # This ensures punchy attacks when attack_ms = 0
            exponents = np.arange(0, decay_samples, dtype=np.float32)
            decay_curve = self.current_level * np.power(self._decay_coefficient, exponents)
            
            output[idx:] = decay_curve
            # Update current_level to be the level AFTER this block (for next call)
            # This is coeff^decay_samples relative to start
            self.current_level = self.current_level * np.power(self._decay_coefficient, decay_samples)
            self.sample_index += decay_samples
            
            # Check if envelope finished
            if self.current_level < 0.0001:
                self.stage = EnvelopeStage.DONE
                self.is_active = False
        
        return output
    
    def process_sample(self) -> float:
        """Process single sample (for sample-by-sample processing)"""
        if not self.is_active:
            return 0.0
            
        if self.stage == EnvelopeStage.ATTACK:
            # S-curve attack: smooth start, accelerates, smooth finish
            self.sample_index += 1
            normalized_time = self.sample_index / self._attack_samples
            # Smoothstep: 3t² - 2t³
            self.current_level = 3 * normalized_time * normalized_time - 2 * normalized_time * normalized_time * normalized_time
            
            if self.sample_index >= self._attack_samples:
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
