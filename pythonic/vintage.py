"""
Vintage Analog Simulation for Pythonic
Simulates analog circuit behavior characteristics

Based on research from musicdsp.org and DSP literature:
- Oscillator drift: Random walk pitch instability
- Thermal noise: Low-level circuit noise floor
- Soft saturation: Component-induced harmonic coloring
- High-frequency roll-off: Capacitor loading simulation

Optimized for real-time performance using vectorized numpy operations.
"""

import numpy as np
from scipy.signal import lfilter


class VintageProcessor:
    """
    Processor that adds analog-style imperfections and warmth
    
    The vintage amount controls the intensity of all effects:
    - 0%: Clean digital sound
    - 25%: Subtle analog warmth
    - 50%: Classic analog character
    - 75%: Vintage hardware feel
    - 100%: Heavy analog coloring
    
    Optimized for real-time audio processing with vectorized operations.
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        
        # Vintage amount (0.0 to 1.0)
        self.amount = 0.0
        
        # Drift state (random walk oscillator instability)
        self._drift_position = 0.0
        self._drift_velocity = 0.0
        self._drift_target = 0.0
        self._drift_counter = 0
        
        # Noise state - use faster numpy random
        self._rng = np.random.default_rng()
        
        # Filter states for stereo (left, right)
        # Noise filter state
        self._noise_z1 = np.zeros(2, dtype=np.float32)
        
        # High-frequency roll-off filter state
        self._hf_z1 = np.zeros(2, dtype=np.float32)
        
        # DC blocker state (per channel)
        self._dc_x1 = np.zeros(2, dtype=np.float32)
        self._dc_y1 = np.zeros(2, dtype=np.float32)
        
        # Pre-computed filter coefficients (updated when amount changes)
        self._cached_amount = -1.0
        self._hf_alpha = 0.0
        self._noise_alpha = 0.1
        
        # Pre-allocate work buffer
        self._work_buffer = np.zeros((8192, 2), dtype=np.float32)
    
    def reset(self):
        """Reset processor state for new note"""
        # Don't reset drift - it should be continuous
        pass
    
    def get_pitch_drift(self) -> float:
        """
        Get current pitch drift in cents (1/100 semitone)
        
        Uses a smoothed random walk algorithm inspired by musicdsp.org
        drift generator. The drift simulates thermal instability in
        analog oscillators.
        
        Returns:
            Pitch deviation in cents (-max_drift to +max_drift)
        """
        if self.amount < 0.001:
            return 0.0
        
        # Update drift using random walk with mean reversion
        # Max drift in cents scales with amount (±8 cents at 100%)
        max_drift_cents = self.amount * 8.0
        
        # Update target occasionally (every ~200 calls)
        self._drift_counter += 1
        if self._drift_counter > 200:
            self._drift_counter = 0
            # New random target with mean reversion toward 0
            self._drift_target = (self._rng.uniform(-1, 1) * 0.7 + 
                                  self._drift_target * 0.3)
            # Fast clip
            if self._drift_target > 1.0:
                self._drift_target = 1.0
            elif self._drift_target < -1.0:
                self._drift_target = -1.0
        
        # Smooth approach to target
        rate = 0.0001 * (44100.0 / self.sr)
        noise = self._rng.uniform(-0.05, 0.05)
        
        # Update velocity with damping
        self._drift_velocity = self._drift_velocity * 0.99 + \
                              (self._drift_target - self._drift_position + noise) * rate
        
        # Apply velocity
        self._drift_position += self._drift_velocity
        
        # Soft limit using fast approximation
        if self._drift_position > 1.0:
            self._drift_position = 1.0 - 0.3 * (self._drift_position - 1.0)
        elif self._drift_position < -1.0:
            self._drift_position = -1.0 - 0.3 * (self._drift_position + 1.0)
        
        return self._drift_position * max_drift_cents
    
    def get_pitch_multiplier(self) -> float:
        """
        Get pitch multiplier based on current drift
        
        Returns:
            Frequency multiplier (e.g., 1.0 = no change, 1.001 = +1.7 cents)
        """
        drift_cents = self.get_pitch_drift()
        # Convert cents to frequency multiplier: 2^(cents/1200)
        # Use fast approximation for small values
        if abs(drift_cents) < 20:
            # Linear approximation: 2^(x/1200) ≈ 1 + x * ln(2)/1200 for small x
            return 1.0 + drift_cents * 0.0005776226505  # ln(2)/1200
        return 2.0 ** (drift_cents / 1200.0)
    
    def process(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply vintage processing to stereo signal (vectorized)
        
        Processing chain:
        1. Add thermal noise floor
        2. Apply soft saturation for harmonic warmth  
        3. Apply gentle high-frequency roll-off
        4. DC blocking
        
        Args:
            signal: Stereo audio [num_samples, 2]
            
        Returns:
            Processed stereo signal
        """
        if self.amount < 0.001:
            return signal
        
        num_samples = len(signal)
        
        # Update cached coefficients if amount changed
        if self._cached_amount != self.amount:
            self._cached_amount = self.amount
            # HF roll-off coefficient
            if self.amount > 0.1:
                cutoff = 20000.0 - self.amount * 8000.0
                rc = 1.0 / (2.0 * np.pi * cutoff)
                dt = 1.0 / self.sr
                self._hf_alpha = dt / (rc + dt)
        
        # Work in-place on copy
        output = signal.copy()
        
        # 1. Add thermal noise (vectorized)
        noise_level = self.amount * 0.001
        if noise_level > 0.00001:
            # Generate white noise
            white = self._rng.standard_normal((num_samples, 2), dtype=np.float32)
            
            # Apply 1-pole lowpass for pink-ish character using scipy.lfilter
            # This is much faster than Python loops
            alpha = self._noise_alpha
            b = np.array([alpha], dtype=np.float32)
            a = np.array([1.0, -(1.0 - alpha)], dtype=np.float32)
            
            # Filter each channel
            for ch in range(2):
                zi = np.array([self._noise_z1[ch] * (1.0 - alpha)])
                filtered, zf = lfilter(b, a, white[:, ch], zi=zi)
                self._noise_z1[ch] = filtered[-1] if num_samples > 0 else self._noise_z1[ch]
                output[:, ch] += filtered * noise_level
        
        # 2. Apply soft saturation (fully vectorized)
        saturation_amount = self.amount * 0.3
        if saturation_amount > 0.01:
            drive = 1.0 + saturation_amount * 2.0
            
            # Vectorized saturation
            driven = output * drive
            np.tanh(driven, out=driven)  # In-place tanh
            
            # Normalize
            normalization = np.tanh(drive)
            if normalization > 0.1:
                driven *= (1.0 / normalization)
            
            # Blend (vectorized)
            blend = saturation_amount
            output *= (1.0 - blend)
            output += driven * blend
        
        # 3. Apply high-frequency roll-off (vectorized with lfilter)
        if self.amount > 0.1:
            alpha = self._hf_alpha
            b = np.array([alpha], dtype=np.float32)
            a = np.array([1.0, -(1.0 - alpha)], dtype=np.float32)
            
            for ch in range(2):
                zi = np.array([self._hf_z1[ch] * (1.0 - alpha)])
                output[:, ch], zf = lfilter(b, a, output[:, ch], zi=zi)
                self._hf_z1[ch] = output[-1, ch] if num_samples > 0 else self._hf_z1[ch]
        
        # 4. DC blocking (vectorized with lfilter) - only if saturation applied
        if saturation_amount > 0.01:
            # DC blocker: y[n] = x[n] - x[n-1] + R * y[n-1], R = 0.995
            R = 0.995
            b_dc = np.array([1.0, -1.0], dtype=np.float32)
            a_dc = np.array([1.0, -R], dtype=np.float32)
            
            for ch in range(2):
                zi = np.array([self._dc_x1[ch] - R * self._dc_y1[ch]])
                output[:, ch], zf = lfilter(b_dc, a_dc, output[:, ch], zi=zi)
                if num_samples > 0:
                    self._dc_x1[ch] = signal[-1, ch]  # Use original input
                    self._dc_y1[ch] = output[-1, ch]
        
        return output
    
    def set_amount(self, amount: float):
        """Set vintage amount (0.0 to 1.0)"""
        self.amount = float(np.clip(amount, 0.0, 1.0))
    
    def get_parameters(self) -> dict:
        """Get parameters for saving"""
        return {
            'vintage_amount': self.amount
        }
    
    def set_parameters(self, params: dict):
        """Set parameters from saved data"""
        if 'vintage_amount' in params:
            self.set_amount(params['vintage_amount'])
