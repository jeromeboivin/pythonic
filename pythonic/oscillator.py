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
    
    # Waveform gain compensation to match loudness balancing
    # Boosts triangle and sawtooth to have similar perceived loudness to sine
    # Reference measurements from TEST samples (relative to sine at 1.0):
    # - Sine: 1.0
    # - Triangle: 1.218 (close to theoretical sqrt(2)/sqrt(3) = 1.225 for equal RMS)
    # - Sawtooth: 1.164 (slightly less than theoretical, possibly due to perceptual adjustment)
    WAVEFORM_GAIN = {
        WaveformType.SINE: 1.0,
        WaveformType.TRIANGLE: 1.241,
        WaveformType.SAWTOOTH: 1.214,
    }
    
    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        
        # Core parameters
        self.frequency = 440.0  # Base frequency in Hz (20-20000)
        self.waveform = WaveformType.SINE
        
        # Pitch modulation
        self.pitch_mod_mode = PitchModMode.DECAYING
        self.pitch_mod_amount = 0.0  # Semitones (-120 to +120 for decay, ±48 for sine)
        self.pitch_mod_rate = 100.0  # Decay time constant (ms) or LFO frequency (Hz)
        
        # Pitch drift (from vintage analog simulation)
        self.pitch_drift_multiplier = 1.0  # Frequency multiplier for analog drift
        
        # Modulation time scaling (for pitch-dependent envelope speed)
        # >1.0 = faster modulation (higher pitch), <1.0 = slower (lower pitch)
        self.mod_time_scale = 1.0
        
        # State
        self.phase = 0.0
        self.mod_time = 0.0  # Time since trigger (seconds)
        self.velocity_gain = 1.0
        self.velocity_mod_scale = 1.0  # Velocity-based modulation amount multiplier
        
        # Random modulation state
        self._random_state = np.random.RandomState(42)
        self._noise_buffer = np.zeros(512)
        self._noise_index = 0
        
        # Pre-allocated buffers for performance
        self._output_buffer = np.zeros(8192, dtype=np.float32)
        self._freq_buffer = np.zeros(8192, dtype=np.float32)
        self._phase_buffer = np.zeros(8192, dtype=np.float32)
        
        # Oversampling for band-limited sawtooth
        self.OVERSAMPLE_FACTOR = 4
        self._os_sos = None   # Decimation filter coefficients
        self._os_zi = None    # Decimation filter state
    
    def reset_phase(self):
        """Reset oscillator phase to zero crossing going positive.
        
        All waveforms start going positive:
        - Sine: sin(0) = 0, next sample positive
        - Triangle: at phase π/2, triangle crosses zero going positive
        - Sawtooth: rising waveform, at phase π gives 0
        
        For clean trigger, all start at or near zero amplitude.
        """
        if self.waveform == WaveformType.SINE:
            # sin(0) = 0, goes positive first
            self.phase = 0.0
        elif self.waveform == WaveformType.TRIANGLE:
            # Triangle: at phase π/2 crosses zero going positive
            self.phase = np.pi / 2.0
        elif self.waveform == WaveformType.SAWTOOTH:
            # Rising saw: (2*phase/2π - 1), at phase π gives 0
            self.phase = np.pi
        else:
            self.phase = 0.0
        self.mod_time = 0.0
        # Reset decimation filter state on new note
        self._os_zi = None
    
    def _ensure_decimation_filter(self):
        """Initialize the anti-aliasing decimation filter for oversampling."""
        if self._os_sos is None:
            from scipy.signal import cheby1
            factor = self.OVERSAMPLE_FACTOR
            # Chebyshev Type I: steep rolloff, 0.5dB passband ripple
            # Cutoff at 90% of original Nyquist, normalized to oversampled Nyquist
            self._os_sos = cheby1(8, 0.5, 0.9 / factor, btype='low', output='sos')
        if self._os_zi is None:
            self._os_zi = np.zeros((self._os_sos.shape[0], 2))
    
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
            self.pitch_mod_amount = np.clip(amount, -96.0, 96.0)
        elif self.pitch_mod_mode == PitchModMode.SINE:
            self.pitch_mod_amount = np.clip(amount, -48.0, 48.0)
        else:  # RANDOM
            # Store Noise mode ModAmt as negative values (e.g. -48 sm)
            # where the magnitude controls modulation depth
            self.pitch_mod_amount = np.clip(abs(amount), 0.0, 20000.0)
    
    def set_pitch_mod_rate(self, rate: float):
        """Set pitch modulation rate (ms for decay, Hz for sine/random)"""
        self.pitch_mod_rate = max(0.001, rate)
    
    def set_velocity_gain(self, gain: float):
        """Set velocity-based gain multiplier"""
        self.velocity_gain = gain

    def set_velocity_mod_scale(self, scale: float):
        """Set velocity-based modulation amount multiplier"""
        self.velocity_mod_scale = scale

    def set_pitch_drift(self, multiplier: float):
        """Set pitch drift multiplier (for analog simulation)
        
        Args:
            multiplier: Frequency multiplier (1.0 = no drift)
        """
        self.pitch_drift_multiplier = multiplier

    def process(self, num_samples: int) -> np.ndarray:
        """
        Generate audio samples
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            numpy array of audio samples
        """
        # Route sawtooth through oversampled path when frequency is high enough
        # that aliasing becomes problematic (harmonics near/above Nyquist)
        if self.waveform == WaveformType.SAWTOOTH:
            # Estimate max instantaneous frequency including pitch modulation
            max_freq = self.frequency
            if self.pitch_mod_mode == PitchModMode.SINE and abs(self.pitch_mod_amount) > 0.01:
                max_freq *= 2.0 ** (abs(self.pitch_mod_amount) / 12.0)
            elif self.pitch_mod_mode == PitchModMode.DECAYING and self.pitch_mod_amount > 0.01:
                max_freq *= 2.0 ** (self.pitch_mod_amount / 12.0)
            
            # Use oversampling when harmonics could alias significantly
            # At sr/4, only ~4 harmonics fit below Nyquist for a sawtooth
            if max_freq > self.sr / 4:
                return self._process_oversampled(num_samples)
        
        # --- Regular path for SINE, TRIANGLE, and low-frequency SAWTOOTH ---
        # Time array for this block - start from 0 for first sample
        # mod_time_scale accelerates/decelerates modulation for pitch shifting
        time_array = self.mod_time + np.arange(num_samples, dtype=np.float32) * self.mod_time_scale / self.sr
        
        # Calculate instantaneous frequency based on modulation mode
        freq_array = self._calculate_frequency(time_array, num_samples)
        
        # Calculate phase increments
        phase_increments = (2.0 * np.pi / self.sr) * freq_array
        
        # Build phase array using pre-allocated buffer if possible
        if num_samples <= len(self._phase_buffer):
            phases = self._phase_buffer[:num_samples]
        else:
            phases = np.empty(num_samples, dtype=np.float32)
        
        # Cumulative phase
        cumulative = np.cumsum(phase_increments)
        phases[0] = self.phase
        if num_samples > 1:
            phases[1:] = self.phase + cumulative[:-1]
        
        # Wrap phase to [0, 2π]
        np.mod(phases, 2.0 * np.pi, out=phases)
        
        # Update state for next block
        self.phase = np.mod(self.phase + cumulative[-1], 2.0 * np.pi) if num_samples > 0 else self.phase
        self.mod_time += num_samples * self.mod_time_scale / self.sr
        
        # Generate waveform (no PolyBLEP on regular path to preserve existing behavior)
        output = self._generate_waveform(phases)
        
        # Apply velocity gain
        if self.velocity_gain != 1.0:
            output *= self.velocity_gain
        
        return output
    
    def _process_oversampled(self, num_samples: int) -> np.ndarray:
        """
        Generate sawtooth with 4x oversampling + PolyBLEP for alias-free output.
        
        At high frequencies (e.g., 20kHz hi-hats with FM), the instantaneous
        frequency can exceed Nyquist. Oversampling extends the usable range
        to 4x Nyquist (88.2kHz at 44.1kHz), then decimates with an anti-aliasing
        filter to produce clean output.
        """
        factor = self.OVERSAMPLE_FACTOR
        os_num = num_samples * factor
        os_sr = self.sr * factor
        
        # Time array at oversampled rate
        time_array = self.mod_time + np.arange(os_num, dtype=np.float64) * self.mod_time_scale / os_sr
        
        # Calculate frequency at oversampled rate
        freq_array = self._calculate_frequency(time_array, os_num)
        
        # Phase increments at oversampled rate
        phase_increments = (2.0 * np.pi / os_sr) * freq_array
        
        # Build phase
        cumulative = np.cumsum(phase_increments)
        phases = np.empty(os_num, dtype=np.float64)
        phases[0] = self.phase
        if os_num > 1:
            phases[1:] = self.phase + cumulative[:-1]
        np.mod(phases, 2.0 * np.pi, out=phases)
        
        # Update state for next block (advance by original-rate time)
        self.phase = float(np.mod(self.phase + cumulative[-1], 2.0 * np.pi)) if os_num > 0 else self.phase
        self.mod_time += num_samples * self.mod_time_scale / self.sr
        
        # Generate PolyBLEP sawtooth at oversampled rate
        oversampled = self._generate_waveform(phases, phase_increments)
        
        # Anti-aliasing decimation filter (persistent state for block continuity)
        self._ensure_decimation_filter()
        from scipy.signal import sosfilt
        filtered, self._os_zi = sosfilt(self._os_sos, oversampled, zi=self._os_zi)
        
        # Downsample
        output = filtered[::factor].copy()
        
        # Apply velocity gain
        if self.velocity_gain != 1.0:
            output *= self.velocity_gain
        
        return output.astype(np.float32)
    
    def _calculate_frequency(self, time_array: np.ndarray, num_samples: int) -> np.ndarray:
        """Calculate frequency array based on modulation mode"""
        
        if self.pitch_mod_mode == PitchModMode.DECAYING:
            return self._apply_decaying_mod(time_array)
        elif self.pitch_mod_mode == PitchModMode.SINE:
            return self._apply_sine_mod(time_array)
        else:  # RANDOM
            return self._apply_random_mod(num_samples)
    
    def _apply_decaying_mod(self, time_array: np.ndarray) -> np.ndarray:
        """Apply exponential pitch decay modulation
        
        mod_rate parameter controls how fast the pitch decays.
        
        The pitch reaches near-zero modulation well before mod_rate time.
        """
        # Apply velocity-based modulation scaling
        effective_mod_amount = self.pitch_mod_amount * self.velocity_mod_scale
        
        # Base frequency with pitch drift applied
        base_freq = self.frequency * self.pitch_drift_multiplier

        if abs(effective_mod_amount) < 0.01:
            return np.full_like(time_array, base_freq)
        
        # Convert mod_rate (in ms) to decay time constant
        # pitchEnv = 0.001^(t/T) = exp(-6.908*t/T) where T is decay time
        mod_rate_sec = self.pitch_mod_rate / 1000.0  # Convert ms to seconds
        tau = mod_rate_sec / 6.908  # 6.908 = 3*ln(10), matches 0.001^(t/T)
        
        if tau > 0:
            # Exponential decay envelope: reaches -60dB at mod_rate time
            decay_envelope = np.exp(-time_array / tau)
            
            # Current pitch offset in semitones
            current_semitones = effective_mod_amount * decay_envelope
            
            # Convert semitones to frequency multiplier
            freq_multiplier = np.power(2.0, current_semitones / 12.0)
        else:
            freq_multiplier = np.ones_like(time_array)
        
        return base_freq * freq_multiplier
    
    def _apply_sine_mod(self, time_array: np.ndarray) -> np.ndarray:
        """Apply sine/FM modulation"""
        # Apply velocity-based modulation scaling
        effective_mod_amount = self.pitch_mod_amount * self.velocity_mod_scale
        
        # Base frequency with pitch drift applied
        base_freq = self.frequency * self.pitch_drift_multiplier
        
        if abs(effective_mod_amount) < 0.01:
            return np.full_like(time_array, base_freq)
        
        # Convert semitones to frequency ratio
        mod_ratio = 2.0 ** (effective_mod_amount / 12.0) - 1.0
        
        # Sine LFO
        lfo = np.sin(2.0 * np.pi * self.pitch_mod_rate * time_array)
        
        # Apply modulation
        freq_multiplier = 1.0 + mod_ratio * lfo
        
        return base_freq * freq_multiplier
    
    def _apply_random_mod(self, num_samples: int) -> np.ndarray:
        """Apply random/noise modulation (vectorized)"""
        # Apply velocity-based modulation scaling
        effective_mod_amount = abs(self.pitch_mod_amount * self.velocity_mod_scale)
        
        # Base frequency with pitch drift applied
        base_freq = self.frequency * self.pitch_drift_multiplier
        
        if effective_mod_amount < 0.01:
            return np.full(num_samples, base_freq)
        
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
        
        # Scale by modulation amount
        # Convert internal parameter to semitone range for pitch deviation
        mod_semitones = effective_mod_amount / 1000.0 * 12.0
        freq_multiplier = 2.0 ** (filtered_noise * mod_semitones / 12.0)
        
        return base_freq * freq_multiplier
    
    def _generate_waveform(self, phases: np.ndarray, phase_increments: np.ndarray = None) -> np.ndarray:
        """Generate waveform samples from phase array with gain compensation
        
        Note: All waveforms start going positive (starts positive after trigger).
        
        Args:
            phases: Phase array in [0, 2π]
            phase_increments: Per-sample phase increment (for PolyBLEP anti-aliasing)
        """
        
        gain = self.WAVEFORM_GAIN.get(self.waveform, 1.0)
        
        if self.waveform == WaveformType.SINE:
            # Positive sine
            return np.sin(phases) * gain
        
        elif self.waveform == WaveformType.TRIANGLE:
            # Triangle wave: 2 * |2 * (phase/2π - floor(phase/2π + 0.5))| - 1
            normalized = phases / np.pi  # [0, 2] per cycle
            raw = 2.0 * np.abs(2.0 * (normalized / 2.0 - np.floor(normalized / 2.0 + 0.5))) - 1.0
            # Positive polarity
            return raw * gain
        
        elif self.waveform == WaveformType.SAWTOOTH:
            # Band-limited sawtooth with vectorized PolyBLEP anti-aliasing
            # Eliminates aliasing artifacts that cause metallic sound at high frequencies
            t = phases / (2.0 * np.pi)  # Normalize phase to [0, 1)
            
            # Naive rising sawtooth: -1 to +1
            output = 2.0 * t - 1.0
            
            # Apply PolyBLEP correction at the discontinuity (phase wrap)
            if phase_increments is not None:
                dt = np.abs(phase_increments) / (2.0 * np.pi)  # Normalized frequency
                dt = np.maximum(dt, 1e-10)  # Avoid division by zero
                
                # Correction just after discontinuity (t < dt)
                mask_start = t < dt
                if np.any(mask_start):
                    t_n = t[mask_start] / dt[mask_start]
                    output[mask_start] -= (2.0 * t_n - t_n * t_n - 1.0)
                
                # Correction just before discontinuity (t > 1 - dt)
                mask_end = t > (1.0 - dt)
                if np.any(mask_end):
                    t_n = (t[mask_end] - 1.0) / dt[mask_end]
                    output[mask_end] -= (t_n * t_n + 2.0 * t_n + 1.0)
            
            return output * gain
        
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
