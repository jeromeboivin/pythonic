"""
Smoothed Parameter for Pythonic
Provides per-parameter smoothing to eliminate clicks and zippering artifacts
when adjusting synthesizer parameters in real-time.
"""

import numpy as np
from typing import Optional


class SmoothedParameter:
    """
    A parameter with exponential smoothing for click-free transitions.
    
    Uses a one-pole lowpass filter to smooth parameter changes over time.
    The smoothing time constant determines how quickly the parameter
    reaches its target value.
    
    Usage:
        param = SmoothedParameter(initial_value=440.0, time_constant_ms=20.0)
        param.set_target(880.0)  # Set new target
        
        # In audio processing loop:
        current_value = param.get_next_value()  # Per-sample smoothing
        # or
        values = param.process(num_samples)  # Block processing
    """
    
    def __init__(self, initial_value: float = 0.0, time_constant_ms: float = 20.0,
                 sample_rate: int = 44100, min_val: float = None, max_val: float = None):
        """
        Initialize a smoothed parameter.
        
        Args:
            initial_value: Starting value
            time_constant_ms: Smoothing time constant in milliseconds (5-100 typical)
            sample_rate: Audio sample rate
            min_val: Optional minimum value clamp
            max_val: Optional maximum value clamp
        """
        self.sample_rate = sample_rate
        self.min_val = min_val
        self.max_val = max_val
        
        # Current and target values
        self._target = initial_value
        self._current = initial_value
        
        # Smoothing coefficient
        self._time_constant_ms = time_constant_ms
        self._update_coefficient()
        
        # Pre-allocated buffer for block processing
        self._buffer = np.zeros(8192, dtype=np.float32)
    
    def _update_coefficient(self):
        """Update the smoothing coefficient based on time constant"""
        if self._time_constant_ms <= 0:
            self._alpha = 1.0  # Instant change
        else:
            # One-pole lowpass coefficient
            # Time constant is time to reach ~63% of target
            time_constant_samples = (self._time_constant_ms / 1000.0) * self.sample_rate
            self._alpha = 1.0 - np.exp(-1.0 / time_constant_samples)
    
    def set_time_constant(self, time_constant_ms: float):
        """Set the smoothing time constant in milliseconds"""
        self._time_constant_ms = max(0.0, time_constant_ms)
        self._update_coefficient()
    
    def set_sample_rate(self, sample_rate: int):
        """Update sample rate and recalculate coefficient"""
        self.sample_rate = sample_rate
        self._update_coefficient()
    
    def set_target(self, value: float):
        """Set the target value (will smooth towards this)"""
        if self.min_val is not None:
            value = max(self.min_val, value)
        if self.max_val is not None:
            value = min(self.max_val, value)
        self._target = value
    
    def set_immediate(self, value: float):
        """Set value immediately without smoothing"""
        if self.min_val is not None:
            value = max(self.min_val, value)
        if self.max_val is not None:
            value = min(self.max_val, value)
        self._target = value
        self._current = value
    
    def get_target(self) -> float:
        """Get the current target value"""
        return self._target
    
    def get_current(self) -> float:
        """Get the current smoothed value without advancing"""
        return self._current
    
    def get_next_value(self) -> float:
        """Get the next smoothed value (advances by one sample)"""
        self._current += (self._target - self._current) * self._alpha
        return self._current
    
    def process(self, num_samples: int) -> np.ndarray:
        """
        Process a block of samples, returning smoothed values.
        
        Args:
            num_samples: Number of samples to process
            
        Returns:
            Array of smoothed values
        """
        if num_samples <= len(self._buffer):
            output = self._buffer[:num_samples]
        else:
            output = np.zeros(num_samples, dtype=np.float32)
        
        # Check if already at target (optimization)
        if abs(self._current - self._target) < 1e-10:
            output.fill(self._target)
            return output
        
        # Apply smoothing sample by sample
        # For efficiency, use vectorized approach when possible
        alpha = self._alpha
        current = self._current
        target = self._target
        
        for i in range(num_samples):
            current += (target - current) * alpha
            output[i] = current
        
        self._current = current
        return output
    
    def is_settled(self, threshold: float = 1e-6) -> bool:
        """Check if parameter has reached its target"""
        return abs(self._current - self._target) < threshold
    
    @property
    def value(self) -> float:
        """Property access to current smoothed value"""
        return self._current
    
    @value.setter
    def value(self, val: float):
        """Set target value via property"""
        self.set_target(val)


class LogSmoothedParameter(SmoothedParameter):
    """
    Smoothed parameter that operates in logarithmic domain.
    
    Useful for frequency and time parameters where changes should
    be perceptually linear (i.e., smoothing in log space).
    """
    
    def __init__(self, initial_value: float = 1.0, time_constant_ms: float = 20.0,
                 sample_rate: int = 44100, min_val: float = 0.001, max_val: float = None):
        # Ensure positive values for log operation
        if min_val is None or min_val <= 0:
            min_val = 0.001
        super().__init__(initial_value, time_constant_ms, sample_rate, min_val, max_val)
        
        # Store log values for smoother interpolation
        self._log_target = np.log(max(self._target, self.min_val))
        self._log_current = np.log(max(self._current, self.min_val))
    
    def set_target(self, value: float):
        """Set target value (operates in log domain internally)"""
        if self.min_val is not None:
            value = max(self.min_val, value)
        if self.max_val is not None:
            value = min(self.max_val, value)
        self._target = value
        self._log_target = np.log(value)
    
    def set_immediate(self, value: float):
        """Set value immediately without smoothing"""
        if self.min_val is not None:
            value = max(self.min_val, value)
        if self.max_val is not None:
            value = min(self.max_val, value)
        self._target = value
        self._current = value
        self._log_target = np.log(value)
        self._log_current = np.log(value)
    
    def get_next_value(self) -> float:
        """Get the next smoothed value (advances by one sample)"""
        self._log_current += (self._log_target - self._log_current) * self._alpha
        self._current = np.exp(self._log_current)
        return self._current
    
    def process(self, num_samples: int) -> np.ndarray:
        """Process a block of samples with log-domain smoothing"""
        if num_samples <= len(self._buffer):
            output = self._buffer[:num_samples]
        else:
            output = np.zeros(num_samples, dtype=np.float32)
        
        # Check if already at target
        if abs(self._log_current - self._log_target) < 1e-10:
            output.fill(self._target)
            return output
        
        alpha = self._alpha
        log_current = self._log_current
        log_target = self._log_target
        
        for i in range(num_samples):
            log_current += (log_target - log_current) * alpha
            output[i] = np.exp(log_current)
        
        self._log_current = log_current
        self._current = np.exp(log_current)
        return output


# Logarithmic scaling utility functions for GUI widgets

def linear_to_log(value: float, min_val: float, max_val: float) -> float:
    """
    Convert a linear 0-1 normalized value to logarithmic scale.
    
    Args:
        value: Linear value (0.0 to 1.0)
        min_val: Minimum output value
        max_val: Maximum output value
        
    Returns:
        Logarithmically scaled value
    """
    if min_val <= 0:
        min_val = 0.001  # Avoid log(0)
    
    # Exponential mapping: output = min * (max/min)^value
    return min_val * np.power(max_val / min_val, value)


def log_to_linear(value: float, min_val: float, max_val: float) -> float:
    """
    Convert a logarithmic value back to linear 0-1 normalized.
    
    Args:
        value: Logarithmic value
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Linear normalized value (0.0 to 1.0)
    """
    if min_val <= 0:
        min_val = 0.001
    if value <= 0:
        value = min_val
    
    # Inverse of exponential mapping
    return np.log(value / min_val) / np.log(max_val / min_val)


def linear_to_log_time(value: float, min_val: float, max_val: float) -> float:
    """
    Convert linear 0-1 to logarithmic scale optimized for time parameters.
    Uses a curve that gives more resolution at short times.
    
    Args:
        value: Linear value (0.0 to 1.0)
        min_val: Minimum time in ms
        max_val: Maximum time in ms
        
    Returns:
        Logarithmically scaled time value
    """
    if min_val <= 0:
        min_val = 0.1  # Minimum 0.1ms
    
    return min_val * np.power(max_val / min_val, value)


def log_time_to_linear(value: float, min_val: float, max_val: float) -> float:
    """
    Convert logarithmic time value back to linear 0-1.
    
    Args:
        value: Time value in ms
        min_val: Minimum time in ms
        max_val: Maximum time in ms
        
    Returns:
        Linear normalized value (0.0 to 1.0)
    """
    if min_val <= 0:
        min_val = 0.1
    if value <= 0:
        value = min_val
    
    result = np.log(value / min_val) / np.log(max_val / min_val)
    return np.clip(result, 0.0, 1.0)
