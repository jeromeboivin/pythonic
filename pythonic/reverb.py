"""
Stereo Reverb Effect for Pythonic
Optimized Schroeder reverb with parallel comb filters and cascaded allpass filters

Features:
- Per-channel reverb with adjustable decay
- Stereo width control for spatial effects
- Highly optimized using NumPy vectorization
- Optional Numba JIT compilation for maximum performance
- Low CPU usage suitable for real-time synthesis
"""

import numpy as np
from typing import Tuple

# Try to import Numba for JIT compilation (optional, falls back to pure NumPy)
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Define a no-op decorator if Numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


# JIT-compiled helper functions for maximum performance
@jit(nopython=True, cache=True, fastmath=True)
def _process_comb_filter_jit(input_signal, buffer, delay, gain, start_idx):
    """Process a single comb filter with JIT compilation"""
    num_samples = len(input_signal)
    buffer_size = len(buffer)
    output = np.zeros(num_samples, dtype=np.float32)
    idx = start_idx
    
    for n in range(num_samples):
        read_idx = (idx - delay) % buffer_size
        delayed = buffer[read_idx]
        output[n] = delayed
        buffer[idx % buffer_size] = input_signal[n] + delayed * gain
        idx += 1
    
    return output, idx % buffer_size


@jit(nopython=True, cache=True, fastmath=True)
def _process_allpass_filter_jit(input_signal, buffer, delay, g, start_idx):
    """Process a single allpass filter with JIT compilation"""
    num_samples = len(input_signal)
    buffer_size = len(buffer)
    output = np.zeros(num_samples, dtype=np.float32)
    idx = start_idx
    
    for n in range(num_samples):
        read_idx = (idx - delay) % buffer_size
        delayed = buffer[read_idx]
        output[n] = -g * input_signal[n] + delayed
        buffer[idx % buffer_size] = input_signal[n] + g * delayed
        idx += 1
    
    return output, idx % buffer_size


class StereoReverb:
    """
    Optimized stereo reverb using Schroeder algorithm
    
    Architecture:
    - 8 parallel comb filters (4 per channel, different delay times for stereo)
    - 4 cascaded allpass filters for diffusion (2 per channel)
    - Stereo width control via mid-side processing
    
    All operations vectorized with NumPy for maximum performance.
    """
    
    # Comb filter delay times in samples at 44100Hz
    # Using prime-related numbers to avoid resonance
    COMB_DELAYS_L = np.array([1557, 1617, 1491, 1422], dtype=np.int32)
    COMB_DELAYS_R = np.array([1617, 1557, 1422, 1491], dtype=np.int32)  # Slightly different for stereo
    
    # Allpass filter delay times
    ALLPASS_DELAYS_L = np.array([225, 556], dtype=np.int32)
    ALLPASS_DELAYS_R = np.array([241, 579], dtype=np.int32)  # Different for stereo spread
    
    # Allpass coefficient (controls diffusion)
    ALLPASS_COEFF = 0.5
    
    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        
        # Scale delays for different sample rates
        self._scale = sample_rate / 44100.0
        
        # Parameters
        self._decay = 0.3  # 0.0 to 1.0 (maps to RT60 roughly 0.1s to 3s)
        self._mix = 0.0    # 0.0 = dry, 1.0 = fully wet
        self._width = 1.0  # 0.0 = mono, 1.0 = full stereo, 2.0 = extra wide
        
        # Pre-compute scaled delays
        self._comb_delays_l = (self.COMB_DELAYS_L * self._scale).astype(np.int32)
        self._comb_delays_r = (self.COMB_DELAYS_R * self._scale).astype(np.int32)
        self._allpass_delays_l = (self.ALLPASS_DELAYS_L * self._scale).astype(np.int32)
        self._allpass_delays_r = (self.ALLPASS_DELAYS_R * self._scale).astype(np.int32)
        
        # Maximum delay needed for buffers
        self._max_comb_delay = max(np.max(self._comb_delays_l), np.max(self._comb_delays_r))
        self._max_allpass_delay = max(np.max(self._allpass_delays_l), np.max(self._allpass_delays_r))
        
        # Comb filter delay lines (circular buffers)
        # Shape: [num_filters, max_delay]
        self._comb_buffers_l = np.zeros((4, self._max_comb_delay + 1), dtype=np.float32)
        self._comb_buffers_r = np.zeros((4, self._max_comb_delay + 1), dtype=np.float32)
        self._comb_indices = np.zeros(4, dtype=np.int32)  # Write positions
        
        # Allpass filter delay lines
        self._allpass_buffers_l = np.zeros((2, self._max_allpass_delay + 1), dtype=np.float32)
        self._allpass_buffers_r = np.zeros((2, self._max_allpass_delay + 1), dtype=np.float32)
        self._allpass_indices = np.zeros(2, dtype=np.int32)
        
        # Pre-compute feedback gains for comb filters based on decay
        self._update_feedback_gains()
        
        # Pre-allocated work buffers for efficiency
        self._work_buffer_size = 8192
        self._comb_out_l = np.zeros(self._work_buffer_size, dtype=np.float32)
        self._comb_out_r = np.zeros(self._work_buffer_size, dtype=np.float32)
    
    def _update_feedback_gains(self):
        """Calculate comb filter feedback gains based on decay parameter"""
        # Map decay (0-1) to RT60 time (0.1s to 4s)
        rt60 = 0.1 + self._decay * 3.9
        
        # Feedback gain formula: g = 10^(-3 * delay / (RT60 * sr))
        # This ensures each comb filter decays to -60dB in RT60 seconds
        self._comb_gains_l = np.zeros(4, dtype=np.float32)
        self._comb_gains_r = np.zeros(4, dtype=np.float32)
        
        for i in range(4):
            delay_l = self._comb_delays_l[i]
            delay_r = self._comb_delays_r[i]
            self._comb_gains_l[i] = 10.0 ** (-3.0 * delay_l / (rt60 * self.sr))
            self._comb_gains_r[i] = 10.0 ** (-3.0 * delay_r / (rt60 * self.sr))
        
        # Clamp gains to prevent instability
        self._comb_gains_l = np.clip(self._comb_gains_l, 0.0, 0.98)
        self._comb_gains_r = np.clip(self._comb_gains_r, 0.0, 0.98)
    
    def set_decay(self, decay: float):
        """Set reverb decay time (0.0 to 1.0)"""
        new_decay = np.clip(decay, 0.0, 1.0)
        if new_decay != self._decay:
            self._decay = new_decay
            self._update_feedback_gains()
    
    def set_mix(self, mix: float):
        """Set dry/wet mix (0.0 = dry, 1.0 = wet)"""
        self._mix = np.clip(mix, 0.0, 1.0)
    
    def set_width(self, width: float):
        """Set stereo width (0.0 = mono, 1.0 = normal, 2.0 = extra wide)"""
        self._width = np.clip(width, 0.0, 2.0)
    
    def reset(self):
        """Reset all delay buffers"""
        self._comb_buffers_l.fill(0)
        self._comb_buffers_r.fill(0)
        self._allpass_buffers_l.fill(0)
        self._allpass_buffers_r.fill(0)
        self._comb_indices.fill(0)
        self._allpass_indices.fill(0)
    
    def process(self, input_signal: np.ndarray) -> np.ndarray:
        """
        Process stereo signal through reverb
        
        Args:
            input_signal: Stereo input array [num_samples, 2]
            
        Returns:
            Processed stereo array [num_samples, 2]
        """
        if self._mix < 0.001:
            # Bypass when fully dry
            return input_signal
        
        num_samples = len(input_signal)
        
        # Extract channels
        in_l = input_signal[:, 0]
        in_r = input_signal[:, 1]
        
        # Process through parallel comb filters (vectorized per-filter)
        wet_l = self._process_comb_bank(in_l, self._comb_buffers_l, 
                                        self._comb_delays_l, self._comb_gains_l, 0)
        wet_r = self._process_comb_bank(in_r, self._comb_buffers_r,
                                        self._comb_delays_r, self._comb_gains_r, 1)
        
        # Process through cascaded allpass filters for diffusion
        wet_l = self._process_allpass_cascade(wet_l, self._allpass_buffers_l,
                                              self._allpass_delays_l, 0)
        wet_r = self._process_allpass_cascade(wet_r, self._allpass_buffers_r,
                                              self._allpass_delays_r, 1)
        
        # Apply stereo width using mid-side processing
        if abs(self._width - 1.0) > 0.001:
            mid = (wet_l + wet_r) * 0.5
            side = (wet_l - wet_r) * 0.5
            # Adjust width: width=0 means mono (no side), width=1 is normal
            side *= self._width
            wet_l = mid + side
            wet_r = mid - side
        
        # Mix dry and wet signals
        dry_gain = 1.0 - self._mix
        wet_gain = self._mix * 0.25  # Scale wet down to prevent clipping
        
        # Create output array
        output = np.empty_like(input_signal)
        output[:, 0] = in_l * dry_gain + wet_l * wet_gain
        output[:, 1] = in_r * dry_gain + wet_r * wet_gain
        
        return output
    
    def _process_comb_bank(self, input_signal: np.ndarray, buffers: np.ndarray,
                          delays: np.ndarray, gains: np.ndarray, 
                          channel: int) -> np.ndarray:
        """
        Process through 4 parallel comb filters and sum
        
        Uses JIT-compiled functions when Numba is available for maximum performance.
        Falls back to optimized Python loops otherwise.
        """
        num_samples = len(input_signal)
        output = np.zeros(num_samples, dtype=np.float32)
        
        if NUMBA_AVAILABLE:
            # Use JIT-compiled version for each comb filter
            for i in range(4):
                comb_out, new_idx = _process_comb_filter_jit(
                    input_signal.astype(np.float32),
                    buffers[i],
                    delays[i],
                    gains[i],
                    self._comb_indices[i]
                )
                self._comb_indices[i] = new_idx
                output += comb_out
        else:
            # Fallback: optimized Python implementation
            for i in range(4):
                delay = delays[i]
                gain = gains[i]
                buffer = buffers[i]
                buffer_size = len(buffer)
                idx = self._comb_indices[i]
                
                comb_out = np.zeros(num_samples, dtype=np.float32)
                
                # Chunked processing for cache efficiency
                chunk_size = 64
                n = 0
                while n < num_samples:
                    end = min(n + chunk_size, num_samples)
                    for j in range(n, end):
                        read_idx = (idx - delay) % buffer_size
                        delayed = buffer[read_idx]
                        comb_out[j] = delayed
                        buffer[idx % buffer_size] = input_signal[j] + delayed * gain
                        idx += 1
                    n = end
                
                self._comb_indices[i] = idx % buffer_size
                output += comb_out
        
        # Average the 4 comb filter outputs
        return output * 0.25
    
    def _process_allpass_cascade(self, input_signal: np.ndarray, buffers: np.ndarray,
                                 delays: np.ndarray, channel: int) -> np.ndarray:
        """
        Process through 2 cascaded allpass filters for diffusion
        
        Uses JIT-compiled functions when Numba is available for maximum performance.
        """
        signal = input_signal.copy().astype(np.float32)
        g = self.ALLPASS_COEFF
        
        if NUMBA_AVAILABLE:
            for i in range(2):
                signal, new_idx = _process_allpass_filter_jit(
                    signal,
                    buffers[i],
                    delays[i],
                    g,
                    self._allpass_indices[i]
                )
                self._allpass_indices[i] = new_idx
        else:
            # Fallback: optimized Python implementation
            for i in range(2):
                delay = delays[i]
                buffer = buffers[i]
                buffer_size = len(buffer)
                idx = self._allpass_indices[i]
                
                output = np.zeros_like(signal)
                num_samples = len(signal)
                
                # Chunked processing for cache efficiency
                chunk_size = 64
                n = 0
                while n < num_samples:
                    end = min(n + chunk_size, num_samples)
                    for j in range(n, end):
                        # Read from delay line
                        read_idx = (idx - delay) % buffer_size
                        delayed = buffer[read_idx]
                        
                        # Allpass filter equation
                        output[j] = -g * signal[j] + delayed
                        buffer[idx % buffer_size] = signal[j] + g * delayed
                        
                        idx += 1
                    n = end
                
                self._allpass_indices[i] = idx % buffer_size
                signal = output
        
        return signal


class FastStereoReverb:
    """
    Ultra-fast stereo reverb using vectorized FIR-based approach
    
    This is a simplified reverb for when CPU is at premium.
    Uses pre-computed impulse response convolution with overlap-add.
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        
        # Parameters
        self._decay = 0.3
        self._mix = 0.0
        self._width = 1.0
        
        # IR length in samples (max reverb tail ~1 second)
        self._ir_length = min(int(sample_rate * 1.0), 44100)  # 1.0s for audible reverb tails
        
        # Pre-computed impulse responses (stereo)
        self._ir_l = np.zeros(self._ir_length, dtype=np.float32)
        self._ir_r = np.zeros(self._ir_length, dtype=np.float32)
        
        # Overlap-add buffers
        self._overlap_l = np.zeros(self._ir_length, dtype=np.float32)
        self._overlap_r = np.zeros(self._ir_length, dtype=np.float32)
        
        # Generate initial IR
        self._generate_impulse_response()
    
    def _generate_impulse_response(self):
        """Generate synthetic impulse response based on decay"""
        t = np.arange(self._ir_length, dtype=np.float32) / self.sr
        
        # Exponential decay envelope
        rt60 = 0.1 + self._decay * 3.9  # 0.1s to 4s
        decay_rate = -6.91 / rt60  # -60dB at rt60
        envelope = np.exp(decay_rate * t)
        
        # Add some randomness for natural sound (different seeds for L/R)
        np.random.seed(42)
        noise_l = np.random.randn(self._ir_length).astype(np.float32)
        np.random.seed(43)
        noise_r = np.random.randn(self._ir_length).astype(np.float32)
        
        # Early reflections (discrete echoes in first 50ms)
        early_length = int(0.05 * self.sr)
        early_times = [0.007, 0.011, 0.017, 0.023, 0.031, 0.041]  # seconds
        early_gains = [0.7, 0.6, 0.5, 0.4, 0.35, 0.3]
        
        early_l = np.zeros(early_length, dtype=np.float32)
        early_r = np.zeros(early_length, dtype=np.float32)
        
        for time, gain in zip(early_times, early_gains):
            idx = int(time * self.sr)
            if idx < early_length:
                early_l[idx] = gain * (1.0 + 0.2 * np.random.randn())
                early_r[idx] = gain * (1.0 + 0.2 * np.random.randn())
        
        # Combine early reflections with diffuse tail
        self._ir_l = noise_l * envelope * 0.6
        self._ir_r = noise_r * envelope * 0.6
        
        if early_length <= self._ir_length:
            self._ir_l[:early_length] += early_l
            self._ir_r[:early_length] += early_r
        
        # Normalize
        max_val = max(np.max(np.abs(self._ir_l)), np.max(np.abs(self._ir_r)))
        if max_val > 0:
            self._ir_l /= max_val
            self._ir_r /= max_val
    
    def set_decay(self, decay: float):
        """Set reverb decay time (0.0 to 1.0)"""
        new_decay = np.clip(decay, 0.0, 1.0)
        if abs(new_decay - self._decay) > 0.01:
            self._decay = new_decay
            self._generate_impulse_response()
    
    def set_mix(self, mix: float):
        """Set dry/wet mix (0.0 = dry, 1.0 = wet)"""
        self._mix = np.clip(mix, 0.0, 1.0)
    
    def set_width(self, width: float):
        """Set stereo width (0.0 = mono, 1.0 = normal, 2.0 = extra wide)"""
        self._width = np.clip(width, 0.0, 2.0)
    
    def reset(self):
        """Reset overlap buffers"""
        self._overlap_l.fill(0)
        self._overlap_r.fill(0)
    
    def process(self, input_signal: np.ndarray) -> np.ndarray:
        """
        Process stereo signal through fast reverb
        
        Uses overlap-add convolution for efficiency
        """
        if self._mix < 0.001:
            return input_signal
        
        num_samples = len(input_signal)
        in_l = input_signal[:, 0]
        in_r = input_signal[:, 1]
        
        # Convolution with truncated IR for speed
        # Use enough of the IR for an audible reverb tail (~186ms at 44100Hz)
        ir_len_used = min(self._ir_length, 8192)
        
        # Convolve (numpy is fast for small kernels)
        wet_l_full = np.convolve(in_l, self._ir_l[:ir_len_used], mode='full')[:num_samples + ir_len_used - 1]
        wet_r_full = np.convolve(in_r, self._ir_r[:ir_len_used], mode='full')[:num_samples + ir_len_used - 1]
        
        # Add overlap from previous block
        overlap_len = min(ir_len_used - 1, len(self._overlap_l), num_samples)
        wet_l = wet_l_full[:num_samples].copy()
        wet_r = wet_r_full[:num_samples].copy()
        
        wet_l[:overlap_len] += self._overlap_l[:overlap_len]
        wet_r[:overlap_len] += self._overlap_r[:overlap_len]
        
        # Save new overlap (accumulate remaining tail beyond current block)
        tail_len = len(wet_l_full) - num_samples
        if tail_len > 0:
            new_overlap = np.zeros(len(self._overlap_l), dtype=np.float32)
            copy_len = min(tail_len, len(new_overlap))
            new_overlap[:copy_len] = wet_l_full[num_samples:num_samples + copy_len]
            # Add any leftover overlap from previous block that extends beyond current block
            leftover_start = num_samples
            leftover_end = min(len(self._overlap_l), len(new_overlap) + num_samples)
            if overlap_len < len(self._overlap_l):
                remaining = min(len(self._overlap_l) - overlap_len, len(new_overlap))
                new_overlap[:remaining] += self._overlap_l[overlap_len:overlap_len + remaining]
            self._overlap_l = new_overlap
            
            new_overlap_r = np.zeros(len(self._overlap_r), dtype=np.float32)
            new_overlap_r[:copy_len] = wet_r_full[num_samples:num_samples + copy_len]
            if overlap_len < len(self._overlap_r):
                remaining = min(len(self._overlap_r) - overlap_len, len(new_overlap_r))
                new_overlap_r[:remaining] += self._overlap_r[overlap_len:overlap_len + remaining]
            self._overlap_r = new_overlap_r
        else:
            self._overlap_l.fill(0)
            self._overlap_r.fill(0)
        
        # Apply stereo width
        if abs(self._width - 1.0) > 0.001:
            mid = (wet_l + wet_r) * 0.5
            side = (wet_l - wet_r) * 0.5 * self._width
            wet_l = mid + side
            wet_r = mid - side
        
        # Mix
        dry_gain = 1.0 - self._mix
        wet_gain = self._mix
        
        output = np.empty_like(input_signal)
        output[:, 0] = in_l * dry_gain + wet_l * wet_gain
        output[:, 1] = in_r * dry_gain + wet_r * wet_gain
        
        return output
