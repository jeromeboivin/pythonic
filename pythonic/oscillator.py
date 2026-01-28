"""
Oscillator for Pythonic
Implements sine, triangle, and sawtooth waveforms with pitch modulation
"""

import numpy as np
from enum import Enum


class WaveformType(Enum):
    SINE = 0
    TRIANGLE = 1
    SAWTOOTH = 2


class PitchModMode(Enum):
    DECAYING = 0  # Exponential pitch decay (classic drum pitch envelope)
    SINE = 1      # Sine LFO / FM modulation
    RANDOM = 2    # Random noise modulation


class Oscillator:
    """
    Oscillator with multiple waveforms and pitch modulation modes
    """
    
    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        
        # Core parameters
        self.frequency = 440.0  # Base frequency in Hz (20-20000)
        self.waveform = WaveformType.SINE
        
        # Pitch modulation
        self.pitch_mod_mode = PitchModMode.DECAYING
        self.pitch_mod_amount = 0.0  # Semitones (-120 to +120 for decay, ±48 for sine)
        self.pitch_mod_rate = 100.0  # Decay time constant (ms) or LFO frequency (Hz)
        
        # State
        self.phase = 0.0
        self.mod_time = 0.0  # Time since trigger (seconds)
        self.velocity_gain = 1.0
        
        # Random modulation state
        self._random_state = np.random.RandomState(42)
        self._noise_buffer = np.zeros(512)
        self._noise_index = 0
    
    def reset_phase(self):
        """Reset oscillator phase (call on trigger)"""
        self.phase = 0.0
        self.mod_time = 0.0
    
    def set_frequency(self, freq: float):
        """Set base frequency in Hz"""
        self.frequency = np.clip(freq, 20.0, 20000.0)
    
    def set_waveform(self, waveform: WaveformType):
        """Set waveform type"""
        self.waveform = waveform
    
    def set_pitch_mod_mode(self, mode: PitchModMode):
        """Set pitch modulation mode"""
        self.pitch_mod_mode = mode
    
    def set_pitch_mod_amount(self, amount: float):
        """Set pitch modulation amount in semitones"""
        if self.pitch_mod_mode == PitchModMode.DECAYING:
            self.pitch_mod_amount = np.clip(amount, -120.0, 120.0)
        elif self.pitch_mod_mode == PitchModMode.SINE:
            self.pitch_mod_amount = np.clip(amount, -48.0, 48.0)
        else:  # RANDOM
            self.pitch_mod_amount = np.clip(amount, 0.0, 20000.0)
    
    def set_pitch_mod_rate(self, rate: float):
        """Set pitch modulation rate (ms for decay, Hz for sine/random)"""
        self.pitch_mod_rate = max(0.001, rate)
    
    def set_velocity_gain(self, gain: float):
        """Set velocity-based gain multiplier"""
        self.velocity_gain = gain
    
    def process(self, num_samples: int) -> np.ndarray:
        """
        Generate audio samples
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            numpy array of audio samples
        """
        # Time array for this block
        sample_times = np.arange(num_samples) / self.sr
        time_array = self.mod_time + sample_times
        
        # Calculate instantaneous frequency based on modulation mode
        freq_array = self._calculate_frequency(time_array, num_samples)
        
        # Calculate phase increments
        phase_increments = 2.0 * np.pi * freq_array / self.sr
        
        # Accumulate phase
        phases = self.phase + np.cumsum(phase_increments)
        
        # Wrap phase to [0, 2π]
        phases = np.mod(phases, 2.0 * np.pi)
        
        # Update state for next block
        self.phase = phases[-1] if num_samples > 0 else self.phase
        self.mod_time += num_samples / self.sr
        
        # Generate waveform
        output = self._generate_waveform(phases)
        
        # Apply velocity gain
        return output * self.velocity_gain
    
    def _calculate_frequency(self, time_array: np.ndarray, num_samples: int) -> np.ndarray:
        """Calculate frequency array based on modulation mode"""
        
        if self.pitch_mod_mode == PitchModMode.DECAYING:
            return self._apply_decaying_mod(time_array)
        elif self.pitch_mod_mode == PitchModMode.SINE:
            return self._apply_sine_mod(time_array)
        else:  # RANDOM
            return self._apply_random_mod(num_samples)
    
    def _apply_decaying_mod(self, time_array: np.ndarray) -> np.ndarray:
        """Apply exponential pitch decay modulation"""
        if abs(self.pitch_mod_amount) < 0.01:
            return np.full_like(time_array, self.frequency)
        
        # Convert semitones to frequency ratio
        mod_ratio = 2.0 ** (self.pitch_mod_amount / 12.0)
        
        # Exponential decay with rate parameter (in ms)
        decay_constant = self.pitch_mod_rate / 1000.0  # Convert ms to seconds
        
        if decay_constant > 0:
            # Decay envelope from 1 to 0
            decay_envelope = np.exp(-time_array / decay_constant)
        else:
            decay_envelope = np.zeros_like(time_array)
        
        # Start frequency is base * ratio, decay towards base
        # f(t) = f0 * (1 + (ratio - 1) * envelope)
        freq_multiplier = 1.0 + (mod_ratio - 1.0) * decay_envelope
        
        return self.frequency * freq_multiplier
    
    def _apply_sine_mod(self, time_array: np.ndarray) -> np.ndarray:
        """Apply sine/FM modulation"""
        if abs(self.pitch_mod_amount) < 0.01:
            return np.full_like(time_array, self.frequency)
        
        # Convert semitones to frequency ratio
        mod_ratio = 2.0 ** (self.pitch_mod_amount / 12.0) - 1.0
        
        # Sine LFO
        lfo = np.sin(2.0 * np.pi * self.pitch_mod_rate * time_array)
        
        # Apply modulation
        freq_multiplier = 1.0 + mod_ratio * lfo
        
        return self.frequency * freq_multiplier
    
    def _apply_random_mod(self, num_samples: int) -> np.ndarray:
        """Apply random/noise modulation (vectorized)"""
        if self.pitch_mod_amount < 0.01:
            return np.full(num_samples, self.frequency)
        
        # Generate filtered random noise
        noise = self._random_state.randn(num_samples)
        
        # Simple one-pole lowpass filter for smoothing (vectorized with cumsum trick)
        cutoff_normalized = min(0.99, self.pitch_mod_rate / (self.sr / 2))
        alpha = cutoff_normalized
        
        # Vectorized IIR filter using scipy.signal.lfilter
        from scipy.signal import lfilter
        b = np.array([alpha])
        a = np.array([1.0, -(1.0 - alpha)])
        
        prev = self._noise_buffer[-1] if len(self._noise_buffer) > 0 else 0.0
        zi = np.array([prev * (1.0 - alpha)])
        
        filtered_noise, zf = lfilter(b, a, noise, zi=zi)
        self._noise_buffer = filtered_noise  # Store for next call
        
        # Scale by modulation amount (treating it as bandwidth in Hz)
        # Convert to semitone range
        mod_semitones = self.pitch_mod_amount / 1000.0 * 12.0  # Scale down
        freq_multiplier = 2.0 ** (filtered_noise * mod_semitones / 12.0)
        
        return self.frequency * freq_multiplier
    
    def _generate_waveform(self, phases: np.ndarray) -> np.ndarray:
        """Generate waveform samples from phase array"""
        
        if self.waveform == WaveformType.SINE:
            return np.sin(phases)
        
        elif self.waveform == WaveformType.TRIANGLE:
            # Triangle wave: 2 * |2 * (phase/2π - floor(phase/2π + 0.5))| - 1
            normalized = phases / np.pi  # [0, 2] per cycle
            return 2.0 * np.abs(2.0 * (normalized / 2.0 - np.floor(normalized / 2.0 + 0.5))) - 1.0
        
        elif self.waveform == WaveformType.SAWTOOTH:
            # Sawtooth: 2 * (phase/2π) - 1, then wrap
            return 2.0 * (phases / (2.0 * np.pi)) - 1.0
        
        return np.sin(phases)  # Default to sine


class PolyBLEPOscillator(Oscillator):
    """
    Oscillator with PolyBLEP anti-aliasing for triangle and sawtooth
    Reduces aliasing artifacts at high frequencies
    """
    
    def __init__(self, sample_rate: int = 44100):
        super().__init__(sample_rate)
        self._prev_phase = 0.0
    
    def _polyblep(self, t: float, dt: float) -> float:
        """
        PolyBLEP correction factor
        t: phase position (0-1)
        dt: phase increment per sample
        """
        if t < dt:
            # Near start of cycle
            t = t / dt
            return t + t - t * t - 1.0
        elif t > 1.0 - dt:
            # Near end of cycle
            t = (t - 1.0) / dt
            return t * t + t + t + 1.0
        return 0.0
    
    def _generate_waveform(self, phases: np.ndarray) -> np.ndarray:
        """Generate anti-aliased waveform"""
        
        if self.waveform == WaveformType.SINE:
            return np.sin(phases)
        
        # Normalized phase (0 to 1)
        t = phases / (2.0 * np.pi)
        dt = self.frequency / self.sr
        
        if self.waveform == WaveformType.SAWTOOTH:
            # Naive sawtooth
            output = 2.0 * t - 1.0
            
            # Apply PolyBLEP correction
            for i in range(len(output)):
                output[i] -= self._polyblep(t[i], dt)
            
            return output
        
        elif self.waveform == WaveformType.TRIANGLE:
            # Generate from integrated square wave
            # First generate square with PolyBLEP
            square = np.where(t < 0.5, 1.0, -1.0)
            
            # Apply PolyBLEP at transitions
            for i in range(len(square)):
                square[i] += self._polyblep(t[i], dt)
                square[i] -= self._polyblep(np.mod(t[i] + 0.5, 1.0), dt)
            
            # Leaky integrator to create triangle
            output = np.zeros_like(square)
            leak = 0.999
            prev = 0.0
            
            for i in range(len(square)):
                output[i] = leak * prev + square[i] * dt * 4.0
                prev = output[i]
            
            # Normalize
            max_val = np.max(np.abs(output))
            if max_val > 0:
                output /= max_val
            
            return output
        
        return np.sin(phases)
