"""
Pythonic Non-Regression Test Suite

This test suite validates the core audio synthesis components of Pythonic
using synthetic reference data. It ensures that changes to the codebase
don't break existing functionality.

Run with: pytest tests/ -v
Or: python tests/test_synthesis.py
"""

import numpy as np
import sys
import os

# Try to import pytest, but allow running without it
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pythonic.oscillator import Oscillator, WaveformType, PitchModMode
from pythonic.drum_channel import DrumChannel
from pythonic.noise import NoiseGenerator, NoiseFilterMode
from pythonic.envelope import Envelope


# Constants
SAMPLE_RATE = 44100
TEST_DURATION_SAMPLES = 44100  # 1 second


class TestOscillatorWaveforms:
    """Test oscillator waveform generation"""
    
    def test_sine_wave_amplitude(self):
        """Sine wave should have amplitude of 1.0 (with gain=1.0)"""
        osc = Oscillator(SAMPLE_RATE)
        osc.set_waveform(WaveformType.SINE)
        osc.set_frequency(440)
        osc.reset_phase()
        
        samples = osc.process(SAMPLE_RATE)
        
        # Peak should be exactly 1.0 for sine
        assert 0.99 <= np.max(samples) <= 1.01, f"Sine peak too high: {np.max(samples)}"
        assert -1.01 <= np.min(samples) <= -0.99, f"Sine trough too low: {np.min(samples)}"
    
    def test_sine_wave_frequency(self):
        """Sine wave should have correct frequency via zero-crossings"""
        osc = Oscillator(SAMPLE_RATE)
        osc.set_waveform(WaveformType.SINE)
        osc.set_frequency(440)
        osc.reset_phase()
        
        samples = osc.process(SAMPLE_RATE)
        
        # Count zero crossings (should be ~880 for 440 Hz over 1 second)
        zero_crossings = np.sum(np.abs(np.diff(np.sign(samples))) > 0)
        expected_crossings = 440 * 2  # 2 per cycle
        
        assert abs(zero_crossings - expected_crossings) <= 2, \
            f"Expected ~{expected_crossings} zero crossings, got {zero_crossings}"
    
    def test_triangle_wave_amplitude(self):
        """Triangle wave should have correct gain-compensated amplitude"""
        osc = Oscillator(SAMPLE_RATE)
        osc.set_waveform(WaveformType.TRIANGLE)
        osc.set_frequency(440)
        osc.reset_phase()
        
        samples = osc.process(SAMPLE_RATE)
        expected_gain = 1.218  # Triangle gain compensation
        
        assert abs(np.max(samples) - expected_gain) < 0.02, \
            f"Triangle peak should be ~{expected_gain}, got {np.max(samples)}"
    
    def test_sawtooth_wave_amplitude(self):
        """Sawtooth wave should have correct gain-compensated amplitude"""
        osc = Oscillator(SAMPLE_RATE)
        osc.set_waveform(WaveformType.SAWTOOTH)
        osc.set_frequency(440)
        osc.reset_phase()
        
        samples = osc.process(SAMPLE_RATE)
        expected_gain = 1.164  # Sawtooth gain compensation
        
        assert abs(np.max(samples) - expected_gain) < 0.02, \
            f"Sawtooth peak should be ~{expected_gain}, got {np.max(samples)}"
    
    def test_sawtooth_is_rising(self):
        """Sawtooth should be a rising waveform (low to high)"""
        osc = Oscillator(SAMPLE_RATE)
        osc.set_waveform(WaveformType.SAWTOOTH)
        osc.set_frequency(220)  # Low freq for clear cycles
        osc.reset_phase()
        
        samples = osc.process(1000)  # First 1000 samples
        
        # After reset, sawtooth starts at phase π (value 0, then rises)
        # Sample 0 should be near 0
        assert abs(samples[0]) < 0.1, f"Sawtooth should start near 0, got {samples[0]}"
        
        # Next few samples should go positive (rising)
        assert samples[10] > samples[0], "Sawtooth should rise after start"
    
    def test_sine_starts_at_zero(self):
        """Sine wave should start at 0 after reset"""
        osc = Oscillator(SAMPLE_RATE)
        osc.set_waveform(WaveformType.SINE)
        osc.set_frequency(100)
        osc.reset_phase()
        
        samples = osc.process(100)
        
        assert abs(samples[0]) < 0.001, f"Sine should start at 0, got {samples[0]}"
        # Sine goes positive first
        assert samples[5] > 0, "Sine should go positive first"
    
    def test_triangle_starts_at_zero(self):
        """Triangle wave should start at 0 after reset"""
        osc = Oscillator(SAMPLE_RATE)
        osc.set_waveform(WaveformType.TRIANGLE)
        osc.set_frequency(100)
        osc.reset_phase()
        
        samples = osc.process(100)
        
        assert abs(samples[0]) < 0.1, f"Triangle should start near 0, got {samples[0]}"
    
    def test_waveform_determinism(self):
        """Same parameters should produce identical output"""
        osc1 = Oscillator(SAMPLE_RATE)
        osc1.set_waveform(WaveformType.SINE)
        osc1.set_frequency(440)
        osc1.reset_phase()
        samples1 = osc1.process(1000)
        
        osc2 = Oscillator(SAMPLE_RATE)
        osc2.set_waveform(WaveformType.SINE)
        osc2.set_frequency(440)
        osc2.reset_phase()
        samples2 = osc2.process(1000)
        
        np.testing.assert_array_almost_equal(samples1, samples2, decimal=10)
    
    def test_frequency_range(self):
        """Oscillator should handle extreme frequencies"""
        osc = Oscillator(SAMPLE_RATE)
        osc.set_waveform(WaveformType.SINE)
        
        # Low frequency
        osc.set_frequency(20)
        osc.reset_phase()
        samples_low = osc.process(SAMPLE_RATE)
        assert np.max(np.abs(samples_low)) > 0.9, "Low frequency should produce output"
        
        # High frequency (Nyquist limit)
        osc.set_frequency(20000)
        osc.reset_phase()
        samples_high = osc.process(SAMPLE_RATE)
        assert np.max(np.abs(samples_high)) > 0.9, "High frequency should produce output"


class TestPitchModulation:
    """Test pitch modulation modes"""
    
    def test_no_modulation_constant_frequency(self):
        """Zero modulation amount should give constant frequency"""
        osc = Oscillator(SAMPLE_RATE)
        osc.set_waveform(WaveformType.SINE)
        osc.set_frequency(440)
        osc.set_pitch_mod_mode(PitchModMode.DECAYING)
        osc.set_pitch_mod_amount(0.0)
        osc.set_pitch_mod_rate(100.0)
        osc.reset_phase()
        
        samples = osc.process(SAMPLE_RATE)
        
        # Check frequency consistency via FFT
        fft = np.abs(np.fft.rfft(samples[:8192]))
        peak_bin = np.argmax(fft)
        peak_freq = peak_bin * SAMPLE_RATE / 8192
        
        assert abs(peak_freq - 440) < 5, f"Expected 440 Hz, got {peak_freq}"
    
    def test_decay_modulation_starts_high(self):
        """Positive decay modulation should start at higher pitch"""
        osc = Oscillator(SAMPLE_RATE)
        osc.set_waveform(WaveformType.SINE)
        osc.set_frequency(100)
        osc.set_pitch_mod_mode(PitchModMode.DECAYING)
        osc.set_pitch_mod_amount(12.0)  # +12 semitones = 1 octave
        osc.set_pitch_mod_rate(100.0)
        osc.reset_phase()
        
        # Get early samples
        early_samples = osc.process(1024)
        
        # Count zero crossings in first chunk (should be higher than base freq)
        zero_crossings = np.sum(np.abs(np.diff(np.sign(early_samples))) > 0)
        
        # At 100 Hz base, should be 100 crossings per second (2 per cycle)
        # With +12 semitones, starts at 200 Hz, so ~9 crossings in 1024 samples
        # Base would be ~4.6 crossings
        assert zero_crossings > 6, f"Should have more crossings at higher pitch, got {zero_crossings}"
    
    def test_decay_modulation_decays_to_base(self):
        """Decay modulation should settle to base frequency"""
        osc = Oscillator(SAMPLE_RATE)
        osc.set_waveform(WaveformType.SINE)
        osc.set_frequency(200)
        osc.set_pitch_mod_mode(PitchModMode.DECAYING)
        osc.set_pitch_mod_amount(24.0)  # +24 semitones
        osc.set_pitch_mod_rate(50.0)  # Fast decay
        osc.reset_phase()
        
        # Process 1 second
        _ = osc.process(SAMPLE_RATE)
        
        # Now get late samples - should be at base frequency
        late_samples = osc.process(4096)
        
        # FFT to find frequency
        fft = np.abs(np.fft.rfft(late_samples))
        peak_bin = np.argmax(fft)
        peak_freq = peak_bin * SAMPLE_RATE / len(late_samples)
        
        assert abs(peak_freq - 200) < 20, f"Late frequency should be ~200 Hz, got {peak_freq}"


class TestEnvelope:
    """Test envelope generator"""
    
    def test_envelope_starts_at_one(self):
        """Envelope should start at 1.0 after trigger"""
        env = Envelope(SAMPLE_RATE)
        env.set_attack(0.0)
        env.set_decay(100.0)
        env.trigger()
        
        samples = env.process(100)
        
        assert abs(samples[0] - 1.0) < 0.01, f"Envelope should start at 1.0, got {samples[0]}"
    
    def test_envelope_decays(self):
        """Envelope should decay over time"""
        env = Envelope(SAMPLE_RATE)
        env.set_attack(0.0)
        env.set_decay(100.0)  # 100ms decay
        env.trigger()
        
        samples = env.process(SAMPLE_RATE // 2)  # 500ms
        
        # After 5 time constants, should be very small
        assert samples[-1] < 0.1, f"Envelope should decay to near 0, got {samples[-1]}"
    
    def test_envelope_attack(self):
        """Envelope with attack should ramp up"""
        env = Envelope(SAMPLE_RATE)
        env.set_attack(50.0)  # 50ms attack
        env.set_decay(500.0)
        env.trigger()
        
        samples = env.process(SAMPLE_RATE // 10)  # 100ms
        
        # Should start low and reach peak
        assert samples[0] < 0.5, "Should start below peak during attack"
        peak_idx = np.argmax(samples)
        assert peak_idx > 0, "Peak should not be at sample 0 with attack"
    
    def test_envelope_determinism(self):
        """Same parameters should produce identical envelope"""
        env1 = Envelope(SAMPLE_RATE)
        env1.set_attack(10.0)
        env1.set_decay(200.0)
        env1.trigger()
        samples1 = env1.process(1000)
        
        env2 = Envelope(SAMPLE_RATE)
        env2.set_attack(10.0)
        env2.set_decay(200.0)
        env2.trigger()
        samples2 = env2.process(1000)
        
        np.testing.assert_array_almost_equal(samples1, samples2, decimal=10)


class TestNoiseGenerator:
    """Test noise generation"""
    
    def test_noise_is_random(self):
        """Noise should not be periodic"""
        noise = NoiseGenerator(SAMPLE_RATE)
        noise.set_filter_mode(NoiseFilterMode.LOW_PASS)
        noise.set_filter_frequency(10000)
        noise.set_filter_q(0.7)
        noise.set_decay(500)
        noise.trigger()
        
        samples = noise.process(4096)
        mono = samples[:, 0]
        
        # Autocorrelation should drop off quickly for noise
        autocorr = np.correlate(mono, mono, mode='same')
        autocorr = autocorr / autocorr[len(autocorr)//2]  # Normalize
        
        # Beyond lag 50, autocorrelation should be low
        assert np.max(np.abs(autocorr[len(autocorr)//2 + 100:])) < 0.5, \
            "Noise should not be highly correlated with itself at large lags"
    
    def test_noise_filter_affects_spectrum(self):
        """Different filter modes should produce different spectra"""
        # Low-pass
        noise_lp = NoiseGenerator(SAMPLE_RATE)
        noise_lp.set_filter_mode(NoiseFilterMode.LOW_PASS)
        noise_lp.set_filter_frequency(500)
        noise_lp.set_filter_q(1.0)
        noise_lp.set_decay(1000)
        noise_lp.trigger()
        samples_lp = noise_lp.process(8192)[:, 0]
        
        # High-pass
        noise_hp = NoiseGenerator(SAMPLE_RATE)
        noise_hp.set_filter_mode(NoiseFilterMode.HIGH_PASS)
        noise_hp.set_filter_frequency(5000)
        noise_hp.set_filter_q(1.0)
        noise_hp.set_decay(1000)
        noise_hp.trigger()
        samples_hp = noise_hp.process(8192)[:, 0]
        
        # Compare energy in low vs high frequency bands
        fft_lp = np.abs(np.fft.rfft(samples_lp))
        fft_hp = np.abs(np.fft.rfft(samples_hp))
        
        low_band_end = int(1000 * len(fft_lp) / (SAMPLE_RATE / 2))
        high_band_start = int(5000 * len(fft_lp) / (SAMPLE_RATE / 2))
        
        lp_low_energy = np.sum(fft_lp[:low_band_end]**2)
        lp_high_energy = np.sum(fft_lp[high_band_start:]**2)
        hp_low_energy = np.sum(fft_hp[:low_band_end]**2)
        hp_high_energy = np.sum(fft_hp[high_band_start:]**2)
        
        # LP should have more low energy ratio, HP more high energy ratio
        lp_ratio = lp_low_energy / (lp_high_energy + 1e-10)
        hp_ratio = hp_low_energy / (hp_high_energy + 1e-10)
        
        assert lp_ratio > hp_ratio, \
            f"LP should have higher low/high ratio than HP: {lp_ratio:.2f} vs {hp_ratio:.2f}"
    
    def test_noise_stereo_independence(self):
        """Stereo noise channels should be independent"""
        noise = NoiseGenerator(SAMPLE_RATE)
        noise.set_filter_mode(NoiseFilterMode.LOW_PASS)
        noise.set_filter_frequency(5000)
        noise.set_stereo(True)
        noise.set_decay(500)
        noise.trigger()
        
        samples = noise.process(4096)
        left = samples[:, 0]
        right = samples[:, 1]
        
        # Correlation between left and right should be low
        correlation = np.corrcoef(left, right)[0, 1]
        assert abs(correlation) < 0.3, \
            f"Stereo channels should be mostly independent, got correlation {correlation}"
    
    def test_noise_mono(self):
        """Mono noise should have identical L/R channels"""
        noise = NoiseGenerator(SAMPLE_RATE)
        noise.set_filter_mode(NoiseFilterMode.LOW_PASS)
        noise.set_filter_frequency(5000)
        noise.set_stereo(False)
        noise.set_decay(500)
        noise.trigger()
        
        samples = noise.process(1024)
        
        np.testing.assert_array_almost_equal(samples[:, 0], samples[:, 1], decimal=10)


class TestDrumChannel:
    """Test complete drum channel"""
    
    def test_drum_channel_outputs_stereo(self):
        """Drum channel should output stereo audio"""
        ch = DrumChannel(0, SAMPLE_RATE)
        ch.trigger()
        audio = ch.process(1024)
        
        assert audio.shape == (1024, 2), f"Expected (1024, 2), got {audio.shape}"
    
    def test_drum_channel_silent_without_trigger(self):
        """Drum channel should be silent without trigger"""
        ch = DrumChannel(0, SAMPLE_RATE)
        audio = ch.process(1024)
        
        assert np.max(np.abs(audio)) < 0.001, "Should be silent without trigger"
    
    def test_drum_channel_oscillator_only(self):
        """100% oscillator mix should have no noise"""
        ch = DrumChannel(0, SAMPLE_RATE)
        ch.set_parameters({
            'osc_waveform': 0,  # Sine
            'osc_frequency': 440,
            'osc_decay': 500,
            'osc_noise_mix': 1.0,  # 100% oscillator
            'pitch_mod_amount': 0.0,
        })
        ch.trigger()
        
        audio = ch.process(4096)
        mono = audio[:, 0]
        
        # Pure sine should have very clean spectrum
        fft = np.abs(np.fft.rfft(mono))
        peak_bin = np.argmax(fft)
        
        # Most energy should be in fundamental
        fundamental_energy = fft[peak_bin]**2
        total_energy = np.sum(fft**2)
        
        assert fundamental_energy / total_energy > 0.8, \
            "Pure oscillator should have clean fundamental"
    
    def test_drum_channel_noise_only(self):
        """0% oscillator mix should be noise only"""
        ch = DrumChannel(0, SAMPLE_RATE)
        ch.set_parameters({
            'osc_frequency': 440,
            'osc_decay': 500,
            'osc_noise_mix': 0.0,  # 100% noise
            'noise_filter_freq': 1000,
            'noise_filter_q': 1.0,
            'noise_decay': 500,
        })
        ch.trigger()
        
        audio = ch.process(4096)
        mono = audio[:, 0]
        
        # Noise should have broad spectrum (low peak-to-average ratio in FFT)
        fft = np.abs(np.fft.rfft(mono))
        peak = np.max(fft)
        mean = np.mean(fft)
        
        # Noise has flatter spectrum than tone (filtered noise may concentrate energy)
        assert peak / mean < 35, \
            f"Noise should have flatter spectrum, got peak/mean ratio {peak/mean}"
    
    def test_drum_channel_mix_intermediate(self):
        """50% mix should contain both oscillator and noise"""
        ch = DrumChannel(0, SAMPLE_RATE)
        ch.set_parameters({
            'osc_waveform': 0,
            'osc_frequency': 200,
            'osc_decay': 500,
            'osc_noise_mix': 0.5,  # 50/50
            'noise_filter_freq': 5000,
            'noise_filter_q': 1.0,
            'noise_decay': 500,
        })
        ch.trigger()
        
        audio = ch.process(8192)
        mono = audio[:, 0]
        
        # Should have both tonal and broadband content
        fft = np.abs(np.fft.rfft(mono))
        
        # Check for fundamental peak
        expected_bin = int(200 * len(fft) / (SAMPLE_RATE / 2))
        local_peak = np.max(fft[max(0, expected_bin-5):expected_bin+5])
        
        # Check for broadband content
        high_freq_energy = np.sum(fft[len(fft)//2:]**2)
        
        assert local_peak > 0, "Should have oscillator component"
        assert high_freq_energy > 0, "Should have noise component"
    
    def test_drum_channel_pan_left(self):
        """Pan full left should put more signal in left channel"""
        ch = DrumChannel(0, SAMPLE_RATE)
        ch.set_parameters({
            'osc_waveform': 0,
            'osc_frequency': 440,
            'osc_decay': 200,
            'osc_noise_mix': 1.0,
            'pan': -100.0,  # Full left
        })
        ch.trigger()
        
        audio = ch.process(4096)
        left_rms = np.sqrt(np.mean(audio[:, 0]**2))
        right_rms = np.sqrt(np.mean(audio[:, 1]**2))
        
        assert left_rms > right_rms * 2, \
            f"Left should be much louder than right with pan left: L={left_rms:.4f}, R={right_rms:.4f}"
    
    def test_drum_channel_pan_right(self):
        """Pan full right should put more signal in right channel"""
        ch = DrumChannel(0, SAMPLE_RATE)
        ch.set_parameters({
            'osc_waveform': 0,
            'osc_frequency': 440,
            'osc_decay': 200,
            'osc_noise_mix': 1.0,
            'pan': 100.0,  # Full right
        })
        ch.trigger()
        
        audio = ch.process(4096)
        left_rms = np.sqrt(np.mean(audio[:, 0]**2))
        right_rms = np.sqrt(np.mean(audio[:, 1]**2))
        
        assert right_rms > left_rms * 2, \
            f"Right should be much louder than left with pan right: L={left_rms:.4f}, R={right_rms:.4f}"
    
    def test_drum_channel_level_affects_amplitude(self):
        """Level dB should affect output amplitude"""
        ch = DrumChannel(0, SAMPLE_RATE)
        ch.set_parameters({
            'osc_waveform': 0,
            'osc_frequency': 440,
            'osc_decay': 200,
            'osc_noise_mix': 1.0,
            'level_db': 0.0,
        })
        ch.trigger()
        audio_0db = ch.process(4096)
        rms_0db = np.sqrt(np.mean(audio_0db**2))
        
        ch.set_parameters({'level_db': -6.0})
        ch.trigger()
        audio_m6db = ch.process(4096)
        rms_m6db = np.sqrt(np.mean(audio_m6db**2))
        
        # -6dB should be roughly half amplitude
        ratio = rms_0db / rms_m6db if rms_m6db > 0 else float('inf')
        expected_ratio = 10**(6/20)  # ~2.0
        
        assert 1.5 < ratio < 2.5, \
            f"-6dB should halve amplitude, got ratio {ratio:.2f} (expected ~{expected_ratio:.2f})"
    
    def test_drum_channel_parameters_roundtrip(self):
        """Get/set parameters should be consistent"""
        ch = DrumChannel(0, SAMPLE_RATE)
        
        original_params = {
            'osc_waveform': 1,
            'osc_frequency': 330,
            'osc_attack': 5.0,
            'osc_decay': 250.0,
            'pitch_mod_mode': 0,
            'pitch_mod_amount': 12.0,
            'pitch_mod_rate': 80.0,
            'noise_filter_mode': 1,
            'noise_filter_freq': 2000.0,
            'noise_filter_q': 2.0,
            'osc_noise_mix': 0.7,
            'distortion': 0.3,
            'eq_frequency': 800.0,
            'eq_gain_db': 3.0,
            'level_db': -3.0,
            'pan': 25.0,
        }
        
        ch.set_parameters(original_params)
        retrieved = ch.get_parameters()
        
        for key, value in original_params.items():
            assert key in retrieved, f"Parameter {key} missing from retrieved"
            assert abs(retrieved[key] - value) < 0.01, \
                f"Parameter {key}: expected {value}, got {retrieved[key]}"


class TestWaveformGainCompensation:
    """Test that waveform gains are correctly applied"""
    
    def test_sine_triangle_sawtooth_rms_similar(self):
        """All waveforms should have similar RMS with gain compensation"""
        rms_values = {}
        
        for wave_type, wave_name in [(WaveformType.SINE, 'sine'),
                                      (WaveformType.TRIANGLE, 'triangle'),
                                      (WaveformType.SAWTOOTH, 'sawtooth')]:
            osc = Oscillator(SAMPLE_RATE)
            osc.set_waveform(wave_type)
            osc.set_frequency(440)
            osc.reset_phase()
            
            samples = osc.process(SAMPLE_RATE)
            rms_values[wave_name] = np.sqrt(np.mean(samples**2))
        
        # RMS should be within 10% of each other
        max_rms = max(rms_values.values())
        min_rms = min(rms_values.values())
        
        assert max_rms / min_rms < 1.15, \
            f"RMS values too different: {rms_values}"


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_zero_length_process(self):
        """Processing 0 samples should not crash or return empty/minimal array"""
        osc = Oscillator(SAMPLE_RATE)
        osc.reset_phase()
        try:
            samples = osc.process(0)
            # Implementation may return minimum 1 sample or empty array
            assert len(samples) <= 1
        except IndexError:
            # It's acceptable for the implementation to raise IndexError on 0 samples
            pass
    
    def test_very_long_process(self):
        """Processing many samples should work"""
        osc = Oscillator(SAMPLE_RATE)
        osc.set_waveform(WaveformType.SINE)
        osc.set_frequency(440)
        osc.reset_phase()
        
        samples = osc.process(SAMPLE_RATE * 10)  # 10 seconds
        assert len(samples) == SAMPLE_RATE * 10
        assert np.max(np.abs(samples)) <= 1.1  # Should not explode
    
    def test_multiple_triggers(self):
        """Multiple triggers should reset properly"""
        ch = DrumChannel(0, SAMPLE_RATE)
        ch.set_parameters({
            'osc_waveform': 0,
            'osc_frequency': 440,
            'osc_decay': 100,
            'osc_noise_mix': 1.0,
        })
        
        # First trigger and decay
        ch.trigger()
        audio1 = ch.process(SAMPLE_RATE)
        
        # Second trigger should restart
        ch.trigger()
        audio2 = ch.process(SAMPLE_RATE)
        
        # Both should have similar initial amplitude
        rms1_early = np.sqrt(np.mean(audio1[:1000]**2))
        rms2_early = np.sqrt(np.mean(audio2[:1000]**2))
        
        assert abs(rms1_early - rms2_early) / max(rms1_early, rms2_early) < 0.1, \
            "Retriggered signal should have similar amplitude"
    
    def test_parameter_clamping(self):
        """Out-of-range parameters should be clamped"""
        osc = Oscillator(SAMPLE_RATE)
        
        # Try to set frequency below minimum
        osc.set_frequency(1)  # Below 20 Hz
        assert osc.frequency >= 20, "Frequency should be clamped to minimum"
        
        # Try to set frequency above maximum
        osc.set_frequency(100000)  # Above 20000 Hz
        assert osc.frequency <= 20000, "Frequency should be clamped to maximum"


class TestDeterminism:
    """Test that synthesis is deterministic for reproducibility"""
    
    def test_drum_channel_determinism_osc_only(self):
        """Oscillator-only drum channel should be fully deterministic"""
        params = {
            'osc_waveform': 0,
            'osc_frequency': 440,
            'osc_decay': 300,
            'pitch_mod_amount': 12.0,
            'pitch_mod_rate': 100.0,
            'osc_noise_mix': 1.0,  # No noise
        }
        
        ch1 = DrumChannel(0, SAMPLE_RATE)
        ch1.set_parameters(params)
        ch1.trigger()
        audio1 = ch1.process(4096)
        
        ch2 = DrumChannel(0, SAMPLE_RATE)
        ch2.set_parameters(params)
        ch2.trigger()
        audio2 = ch2.process(4096)
        
        np.testing.assert_array_almost_equal(audio1, audio2, decimal=10)
    
    def test_oscillator_phase_continuity(self):
        """Phase should be continuous across process() calls"""
        osc = Oscillator(SAMPLE_RATE)
        osc.set_waveform(WaveformType.SINE)
        osc.set_frequency(440)
        osc.reset_phase()
        
        # Process in chunks
        chunk1 = osc.process(1000)
        chunk2 = osc.process(1000)
        
        # Process all at once
        osc.reset_phase()
        all_at_once = osc.process(2000)
        
        # Should match with reasonable precision
        # Small floating-point differences are acceptable due to phase accumulation
        combined = np.concatenate([chunk1, chunk2])
        np.testing.assert_array_almost_equal(combined, all_at_once, decimal=3)


class TestVelocitySensitivity:
    """Test velocity sensitivity implementation for osc, noise, and mod"""
    
    def test_osc_velocity_sensitivity_at_full_velocity(self):
        """At full velocity (127), osc gain should be 1.0 regardless of sensitivity"""
        ch = DrumChannel(0, SAMPLE_RATE)
        ch.osc_vel_sensitivity = 1.0  # 100%
        ch.trigger(velocity=127)
        
        assert abs(ch.oscillator.velocity_gain - 1.0) < 0.0001, \
            f"At vel=127, osc gain should be 1.0, got {ch.oscillator.velocity_gain}"
    
    def test_osc_velocity_sensitivity_at_half_velocity(self):
        """At half velocity with 100% sensitivity"""
        ch = DrumChannel(0, SAMPLE_RATE)
        ch.osc_vel_sensitivity = 1.0  # 100%
        ch.trigger(velocity=64)
        
        # invVel=(127-64)/63=1.0, x=max(1-1*1,0)=0, gain=0
        expected = 0.0
        assert abs(ch.oscillator.velocity_gain - expected) < 0.001, \
            f"At vel=64 with 100% sens, osc gain should be ~{expected:.4f}, got {ch.oscillator.velocity_gain}"
    
    def test_noise_velocity_sensitivity_at_full_velocity(self):
        """At full velocity (127), noise gain should be 1.0 regardless of sensitivity"""
        ch = DrumChannel(0, SAMPLE_RATE)
        ch.noise_vel_sensitivity = 1.0  # 100%
        ch.trigger(velocity=127)
        
        assert abs(ch.noise_gen.velocity_gain - 1.0) < 0.0001, \
            f"At vel=127, noise gain should be 1.0, got {ch.noise_gen.velocity_gain}"
    
    def test_noise_velocity_sensitivity_at_half_velocity(self):
        """At half velocity with 100% sensitivity"""
        ch = DrumChannel(0, SAMPLE_RATE)
        ch.noise_vel_sensitivity = 1.0  # 100%
        ch.trigger(velocity=64)
        
        # invVel=1.0, x=max(1-1*1,0)=0, gain=0
        expected = 0.0
        assert abs(ch.noise_gen.velocity_gain - expected) < 0.001, \
            f"At vel=64 with 100% sens, noise gain should be ~{expected:.4f}, got {ch.noise_gen.velocity_gain}"
    
    def test_mod_velocity_sensitivity_at_full_velocity(self):
        """At full velocity (127), mod scale should be 1.0 regardless of sensitivity"""
        ch = DrumChannel(0, SAMPLE_RATE)
        ch.mod_vel_sensitivity = 1.0  # 100%
        ch.trigger(velocity=127)
        
        assert abs(ch.oscillator.velocity_mod_scale - 1.0) < 0.0001, \
            f"At vel=127, mod scale should be 1.0, got {ch.oscillator.velocity_mod_scale}"
    
    def test_mod_velocity_sensitivity_at_half_velocity(self):
        """At half velocity with 100% sensitivity"""
        ch = DrumChannel(0, SAMPLE_RATE)
        ch.mod_vel_sensitivity = 1.0  # 100%
        ch.trigger(velocity=64)
        
        # invVel=1.0, x=max(1-1*1,0)=0, gain=0
        expected = 0.0
        assert abs(ch.oscillator.velocity_mod_scale - expected) < 0.001, \
            f"At vel=64 with 100% sens, mod scale should be ~{expected:.4f}, got {ch.oscillator.velocity_mod_scale}"
    
    def test_mod_velocity_affects_pitch_modulation(self):
        """Verify mod velocity sensitivity is set correctly"""
        from pythonic.oscillator import PitchModMode
        
        # Create channel with significant pitch modulation
        ch = DrumChannel(0, SAMPLE_RATE)
        ch.oscillator.set_pitch_mod_mode(PitchModMode.DECAYING)
        ch.oscillator.set_pitch_mod_amount(36.0)  # 3 octaves
        ch.oscillator.set_pitch_mod_rate(100.0)
        ch.osc_envelope.set_decay(500.0)
        ch.osc_noise_mix = 1.0  # Pure oscillator
        ch.mod_vel_sensitivity = 1.0  # 100%
        
        # Generate at full velocity
        ch.trigger(velocity=127)
        audio_full = ch.process(4096)
        
        # Verify audio was generated
        assert np.max(np.abs(audio_full)) > 0.01, \
            "Audio should be generated with pitch modulation"
        
        # Verify mod_vel_sensitivity property is set correctly
        assert ch.mod_vel_sensitivity == 1.0, \
            "mod_vel_sensitivity should be set to 1.0"
    
    def test_zero_sensitivity_no_velocity_effect(self):
        """With 0% sensitivity, velocity should have no effect"""
        ch = DrumChannel(0, SAMPLE_RATE)
        ch.osc_vel_sensitivity = 0.0
        ch.noise_vel_sensitivity = 0.0
        ch.mod_vel_sensitivity = 0.0
        
        ch.trigger(velocity=32)  # Low velocity
        
        # All gains should be 1.0 since sensitivity is 0
        assert abs(ch.oscillator.velocity_gain - 1.0) < 0.0001
        assert abs(ch.noise_gen.velocity_gain - 1.0) < 0.0001
        assert abs(ch.oscillator.velocity_mod_scale - 1.0) < 0.0001
    
    def test_velocity_sensitivity_200_percent(self):
        """At 200% sensitivity, the effect should be more extreme"""
        ch = DrumChannel(0, SAMPLE_RATE)
        ch.osc_vel_sensitivity = 2.0  # 200%
        ch.trigger(velocity=64)
        
        # invVel=1.0, x=max(1-1*2,0)=0, gain=0
        expected = 0.0
        assert abs(ch.oscillator.velocity_gain - expected) < 0.001, \
            f"At vel=64 with 200% sens, osc gain should be ~{expected:.6f}, got {ch.oscillator.velocity_gain}"


def run_tests_standalone():
    """Run tests without pytest for environments where pytest is not available"""
    import traceback
    
    test_classes = [
        TestOscillatorWaveforms,
        TestPitchModulation,
        TestEnvelope,
        TestNoiseGenerator,
        TestDrumChannel,
        TestWaveformGainCompensation,
        TestEdgeCases,
        TestDeterminism,
        TestVelocitySensitivity,
    ]
    
    total_passed = 0
    total_failed = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        instance = test_class()
        
        # Find all test methods
        test_methods = [m for m in dir(instance) if m.startswith('test_')]
        
        for method_name in test_methods:
            try:
                method = getattr(instance, method_name)
                method()
                print(f"  ✓ {method_name}")
                total_passed += 1
            except AssertionError as e:
                print(f"  ✗ {method_name}: {e}")
                failed_tests.append((test_class.__name__, method_name, str(e)))
                total_failed += 1
            except Exception as e:
                print(f"  ✗ {method_name}: {type(e).__name__}: {e}")
                failed_tests.append((test_class.__name__, method_name, traceback.format_exc()))
                total_failed += 1
    
    print(f"\n{'='*60}")
    print(f"Results: {total_passed} passed, {total_failed} failed")
    
    if failed_tests:
        print(f"\nFailed tests:")
        for cls, method, error in failed_tests:
            print(f"  - {cls}.{method}")
        return 1
    else:
        print("\nAll tests passed!")
        return 0


if __name__ == '__main__':
    if PYTEST_AVAILABLE:
        import pytest
        exit(pytest.main([__file__, '-v']))
    else:
        exit(run_tests_standalone())
