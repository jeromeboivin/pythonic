"""
State Variable Filter for Pythonic
Implements low-pass, band-pass, and high-pass filtering
"""

import numpy as np
from enum import Enum


class FilterMode(Enum):
    LOW_PASS = 0
    BAND_PASS = 1
    HIGH_PASS = 2


class StateVariableFilter:
    """
    State Variable Filter (SVF) implementation
    Provides simultaneous LP, BP, HP outputs with resonance control
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        
        # Parameters
        self.frequency = 1000.0  # Hz
        self.q = 0.707  # Butterworth Q by default
        self.mode = FilterMode.LOW_PASS
        
        # State variables (integrator memories)
        self.ic1eq = 0.0  # First integrator state
        self.ic2eq = 0.0  # Second integrator state
        
        # Stereo state
        self.ic1eq_r = 0.0
        self.ic2eq_r = 0.0
        
        # Coefficients
        self.g = 0.0
        self.k = 0.0
        self.a1 = 0.0
        self.a2 = 0.0
        self.a3 = 0.0
        
        self._update_coefficients()
    
    def _update_coefficients(self):
        """Calculate filter coefficients based on frequency and Q"""
        # Clamp frequency to safe range
        freq_clamped = np.clip(self.frequency, 20.0, self.sr * 0.49)
        
        # SVF coefficients
        self.g = np.tan(np.pi * freq_clamped / self.sr)
        self.k = 1.0 / self.q
        
        self.a1 = 1.0 / (1.0 + self.g * (self.g + self.k))
        self.a2 = self.g * self.a1
        self.a3 = self.g * self.a2
    
    def set_frequency(self, freq: float):
        """Set cutoff/center frequency in Hz (20-20000)"""
        self.frequency = np.clip(freq, 20.0, 20000.0)
        self._update_coefficients()
    
    def set_q(self, q: float):
        """Set Q/resonance (0.1 to 10000)"""
        self.q = np.clip(q, 0.1, 10000.0)
        self._update_coefficients()
    
    def set_mode(self, mode: FilterMode):
        """Set filter mode (LP, BP, HP)"""
        self.mode = mode
    
    def reset(self):
        """Reset filter state"""
        self.ic1eq = 0.0
        self.ic2eq = 0.0
        self.ic1eq_r = 0.0
        self.ic2eq_r = 0.0
    
    def process_sample(self, x: float) -> float:
        """
        Process single sample through SVF
        
        Args:
            x: Input sample
            
        Returns:
            Filtered output based on current mode
        """
        v3 = x - self.ic2eq
        v1 = self.a1 * self.ic1eq + self.a2 * v3
        v2 = self.ic2eq + self.a2 * self.ic1eq + self.a3 * v3
        
        self.ic1eq = 2.0 * v1 - self.ic1eq
        self.ic2eq = 2.0 * v2 - self.ic2eq
        
        # Return output based on mode
        if self.mode == FilterMode.LOW_PASS:
            return v2
        elif self.mode == FilterMode.BAND_PASS:
            return v1
        else:  # HIGH_PASS
            return x - self.k * v1 - v2
    
    def process(self, x_array: np.ndarray) -> np.ndarray:
        """
        Process array of samples
        
        Args:
            x_array: Input samples (1D array for mono, 2D for stereo)
            
        Returns:
            Filtered output array
        """
        if x_array.ndim == 1:
            # Mono processing
            output = np.zeros_like(x_array)
            for i in range(len(x_array)):
                output[i] = self.process_sample(x_array[i])
            return output
        else:
            # Stereo processing
            output = np.zeros_like(x_array)
            for i in range(len(x_array)):
                # Left channel
                v3 = x_array[i, 0] - self.ic2eq
                v1 = self.a1 * self.ic1eq + self.a2 * v3
                v2 = self.ic2eq + self.a2 * self.ic1eq + self.a3 * v3
                self.ic1eq = 2.0 * v1 - self.ic1eq
                self.ic2eq = 2.0 * v2 - self.ic2eq
                
                if self.mode == FilterMode.LOW_PASS:
                    output[i, 0] = v2
                elif self.mode == FilterMode.BAND_PASS:
                    output[i, 0] = v1
                else:
                    output[i, 0] = x_array[i, 0] - self.k * v1 - v2
                
                # Right channel
                v3_r = x_array[i, 1] - self.ic2eq_r
                v1_r = self.a1 * self.ic1eq_r + self.a2 * v3_r
                v2_r = self.ic2eq_r + self.a2 * self.ic1eq_r + self.a3 * v3_r
                self.ic1eq_r = 2.0 * v1_r - self.ic1eq_r
                self.ic2eq_r = 2.0 * v2_r - self.ic2eq_r
                
                if self.mode == FilterMode.LOW_PASS:
                    output[i, 1] = v2_r
                elif self.mode == FilterMode.BAND_PASS:
                    output[i, 1] = v1_r
                else:
                    output[i, 1] = x_array[i, 1] - self.k * v1_r - v2_r
            
            return output


class EQFilter:
    """
    Parametric EQ filter (peaking/shelving)
    Used in the mixing section
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        
        # Parameters
        self.frequency = 1000.0  # Center frequency
        self.gain_db = 0.0  # Gain in dB (-40 to +40)
        self.q = 1.0  # Bandwidth control
        
        # State
        self.x1 = 0.0
        self.x2 = 0.0
        self.y1 = 0.0
        self.y2 = 0.0
        
        # Coefficients
        self.b0 = 1.0
        self.b1 = 0.0
        self.b2 = 0.0
        self.a1 = 0.0
        self.a2 = 0.0
        
        self._update_coefficients()
    
    def _update_coefficients(self):
        """Calculate biquad coefficients for peaking EQ"""
        if abs(self.gain_db) < 0.01:
            # Bypass when gain is near zero
            self.b0 = 1.0
            self.b1 = 0.0
            self.b2 = 0.0
            self.a1 = 0.0
            self.a2 = 0.0
            return
        
        A = 10.0 ** (self.gain_db / 40.0)  # Square root of linear gain
        omega = 2.0 * np.pi * self.frequency / self.sr
        omega = np.clip(omega, 0.001, np.pi * 0.99)
        
        sin_omega = np.sin(omega)
        cos_omega = np.cos(omega)
        alpha = sin_omega / (2.0 * self.q)
        
        # Peaking EQ coefficients
        self.b0 = 1.0 + alpha * A
        self.b1 = -2.0 * cos_omega
        self.b2 = 1.0 - alpha * A
        a0 = 1.0 + alpha / A
        self.a1 = -2.0 * cos_omega
        self.a2 = 1.0 - alpha / A
        
        # Normalize by a0
        self.b0 /= a0
        self.b1 /= a0
        self.b2 /= a0
        self.a1 /= a0
        self.a2 /= a0
    
    def set_frequency(self, freq: float):
        """Set center frequency"""
        self.frequency = np.clip(freq, 20.0, 20000.0)
        self._update_coefficients()
    
    def set_gain(self, gain_db: float):
        """Set gain in dB"""
        self.gain_db = np.clip(gain_db, -40.0, 40.0)
        self._update_coefficients()
    
    def set_q(self, q: float):
        """Set Q/bandwidth"""
        self.q = np.clip(q, 0.1, 10.0)
        self._update_coefficients()
    
    def reset(self):
        """Reset filter state"""
        self.x1 = 0.0
        self.x2 = 0.0
        self.y1 = 0.0
        self.y2 = 0.0
    
    def process(self, x_array: np.ndarray) -> np.ndarray:
        """Process array of samples through biquad"""
        if abs(self.gain_db) < 0.01:
            return x_array.copy()
        
        output = np.zeros_like(x_array)
        
        for i in range(len(x_array)):
            x = x_array[i]
            y = (self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2 
                 - self.a1 * self.y1 - self.a2 * self.y2)
            
            self.x2 = self.x1
            self.x1 = x
            self.y2 = self.y1
            self.y1 = y
            
            output[i] = y
        
        return output
