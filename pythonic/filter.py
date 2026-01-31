"""
State Variable Filter for Pythonic
Implements low-pass, band-pass, and high-pass filtering
"""

import numpy as np
from enum import Enum
from scipy.signal import lfilter, lfilter_zi


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
        
        # lfilter state (will be initialized on first use)
        self._zi_left = None
        self._zi_right = None
        
        # Cached biquad coefficients (avoid recomputing every process() call)
        self._cached_b = None
        self._cached_a = None
        self._coeffs_dirty = True
        
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
        new_freq = np.clip(freq, 20.0, 20000.0)
        if new_freq != self.frequency:
            self.frequency = new_freq
            self._coeffs_dirty = True
            self._update_coefficients()
    
    def set_q(self, q: float):
        """Set Q/resonance (0.1 to 10000)"""
        new_q = np.clip(q, 0.1, 10000.0)
        if new_q != self.q:
            self.q = new_q
            self._coeffs_dirty = True
            self._update_coefficients()
    
    def set_mode(self, mode: FilterMode):
        """Set filter mode (LP, BP, HP)"""
        if mode != self.mode:
            self.mode = mode
            self._coeffs_dirty = True
            self._zi_left = None
            self._zi_right = None
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
    
    def _get_biquad_coeffs(self):
        """Get biquad filter coefficients for scipy.signal.lfilter"""
        # Convert SVF parameters to biquad coefficients
        w0 = 2.0 * np.pi * self.frequency / self.sr
        w0 = min(w0, np.pi * 0.99)  # Clamp to avoid instability
        
        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)
        alpha = sin_w0 / (2.0 * self.q)
        
        if self.mode == FilterMode.LOW_PASS:
            b0 = (1.0 - cos_w0) / 2.0
            b1 = 1.0 - cos_w0
            b2 = (1.0 - cos_w0) / 2.0
        elif self.mode == FilterMode.HIGH_PASS:
            b0 = (1.0 + cos_w0) / 2.0
            b1 = -(1.0 + cos_w0)
            b2 = (1.0 + cos_w0) / 2.0
        else:  # BAND_PASS
            b0 = alpha
            b1 = 0.0
            b2 = -alpha
        
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha
        
        # Normalize
        b = np.array([b0/a0, b1/a0, b2/a0], dtype=np.float64)
        a = np.array([1.0, a1/a0, a2/a0], dtype=np.float64)
        
        return b, a
    
    def process(self, x_array: np.ndarray) -> np.ndarray:
        """
        Process array of samples (vectorized using scipy.signal.lfilter)
        
        Args:
            x_array: Input samples (1D array for mono, 2D for stereo)
            
        Returns:
            Filtered output array
        """
        # Use cached coefficients or recompute if dirty
        if self._coeffs_dirty or self._cached_b is None:
            b, a = self._get_biquad_coeffs()
            self._cached_b = b
            self._cached_a = a
            self._coeffs_dirty = False
            # Reset filter state when coefficients change
            self._zi_left = None
            self._zi_right = None
        else:
            b = self._cached_b
            a = self._cached_a
        
        if x_array.ndim == 1:
            # Mono processing
            if self._zi_left is None:
                self._zi_left = np.zeros(2, dtype=np.float64)
            
            output, self._zi_left = lfilter(b, a, x_array.astype(np.float64), zi=self._zi_left)
            return output.astype(np.float32)
        else:
            # Stereo processing
            if self._zi_left is None:
                self._zi_left = np.zeros(2, dtype=np.float64)
            if self._zi_right is None:
                self._zi_right = np.zeros(2, dtype=np.float64)
            
            output = np.zeros_like(x_array, dtype=np.float32)
            
            # Left channel
            out_l, self._zi_left = lfilter(b, a, x_array[:, 0].astype(np.float64), zi=self._zi_left)
            output[:, 0] = out_l.astype(np.float32)
            
            # Right channel
            out_r, self._zi_right = lfilter(b, a, x_array[:, 1].astype(np.float64), zi=self._zi_right)
            output[:, 1] = out_r.astype(np.float32)
            
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
        
        # State for lfilter
        self._zi = None
        self._last_b = None
        self._last_a = None
        
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
        self._zi = None
        self._last_b = None
        self._last_a = None
    
    def process(self, x_array: np.ndarray) -> np.ndarray:
        """Process array of samples through biquad (vectorized)"""
        if abs(self.gain_db) < 0.01:
            return x_array.copy()
        
        # Use scipy.signal.lfilter for C-optimized filtering
        b = np.array([self.b0, self.b1, self.b2], dtype=np.float64)
        a = np.array([1.0, self.a1, self.a2], dtype=np.float64)
        
        # Reset state if coefficients changed
        if self._last_b is None or self._last_a is None or \
           not np.array_equal(b, self._last_b) or not np.array_equal(a, self._last_a):
            self._zi = None
            self._last_b = b.copy()
            self._last_a = a.copy()
        
        # Initialize state if needed
        if self._zi is None:
            self._zi = np.zeros(2, dtype=np.float64)
        
        output, self._zi = lfilter(b, a, x_array.astype(np.float64), zi=self._zi)
        
        return output.astype(np.float32)
