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

from pythonic.drum_channel import DrumChannel


# Constants
SAMPLE_RATE = 44100
TEST_DURATION_SAMPLES = 44100  # 1 second


def _render(params, n=SAMPLE_RATE, chunk=None):
    """Trigger a DrumChannel with `params` and render `n` samples."""
    ch = DrumChannel(0, SAMPLE_RATE)
    ch.set_parameters(params)
    ch.trigger()
    if chunk is None:
        return ch.process(n)
    return np.concatenate([ch.process(min(chunk, n - i)) for i in range(0, n, chunk)])


def _peak_freq(x):
    fft = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.argmax(fft) * SAMPLE_RATE / len(x)


class TestPitchModulation:
    """Pitch modulation (oscillator only)"""

    OSC = {'osc_waveform': 0, 'osc_decay': 2000, 'osc_noise_mix': 1.0, 'pitch_mod_mode': 0}

    def test_no_modulation_constant_frequency(self):
        audio = _render({**self.OSC, 'osc_frequency': 440, 'pitch_mod_amount': 0.0})
        assert abs(_peak_freq(audio[:8192, 0]) - 440) < 5
        assert abs(_peak_freq(audio[-8192:, 0]) - 440) < 5

    def test_decay_modulation_starts_high(self):
        def crossings(amount):
            audio = _render({**self.OSC, 'osc_frequency': 100, 'pitch_mod_amount': amount,
                             'pitch_mod_rate': 100.0}, n=1024)
            return np.sum(np.abs(np.diff(np.sign(audio[:, 0]))) > 0)
        assert crossings(24.0) > crossings(0.0) + 2

    def test_decay_modulation_decays_to_base(self):
        audio = _render({**self.OSC, 'osc_frequency': 200, 'pitch_mod_amount': 24.0,
                         'pitch_mod_rate': 50.0})
        assert abs(_peak_freq(audio[-4096:, 0]) - 200) < 20

    def test_high_hat_sine_modulation_stays_finite(self):
        """Large sine-mod amounts on a 20 kHz saw must stay bounded."""
        audio = _render({'osc_waveform': 2, 'osc_frequency': 20000, 'osc_decay': 200,
                         'osc_noise_mix': 1.0, 'pitch_mod_mode': 1,
                         'pitch_mod_amount': 19.26, 'pitch_mod_rate': 803.45}, n=4096)
        assert np.all(np.isfinite(audio)) and np.max(np.abs(audio)) < 2.0


class TestNoise:
    """Noise source (noise only)"""

    NOISE = {'osc_noise_mix': 0.0, 'noise_decay': 1000, 'noise_filter_q': 0.707}

    def test_noise_filter_affects_spectrum(self):
        def centroid(params):
            x = _render({**self.NOISE, **params}, n=8192)[:, 0]
            spec = np.abs(np.fft.rfft(x)) ** 2
            f = np.fft.rfftfreq(len(x), 1 / SAMPLE_RATE)
            return np.sum(f * spec) / np.sum(spec)
        lp = centroid({'noise_filter_mode': 0, 'noise_filter_freq': 1000})
        hp = centroid({'noise_filter_mode': 2, 'noise_filter_freq': 5000})
        assert hp > 2 * lp, f"HP centroid {hp:.0f} Hz should be well above LP {lp:.0f} Hz"

    def test_noise_stereo_independence(self):
        audio = _render({**self.NOISE, 'noise_stereo': True}, n=8192)
        corr = np.corrcoef(audio[:, 0], audio[:, 1])[0, 1]
        assert abs(corr) < 0.3, f"Stereo noise channels should be decorrelated, corr={corr:.3f}"

    def test_noise_mono(self):
        audio = _render({**self.NOISE, 'noise_stereo': False}, n=8192)
        np.testing.assert_allclose(audio[:, 0], audio[:, 1], atol=1e-6)


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

    def test_drum_channel_mix_law(self):
        """Mix law: 50/50 keeps both sources at unity, the minority source
        is attenuated by (1-u) * exp(-2.878 u) (u = distance from the centre)."""
        from pythonic.voice import VoiceParams, _Derived
        mid = _Derived(VoiceParams(osc_mix=0.5), SAMPLE_RATE)
        assert mid.mix_osc == 1.0 and mid.mix_noise == 1.0
        osc_heavy = _Derived(VoiceParams(osc_mix=0.75), SAMPLE_RATE)
        assert osc_heavy.mix_osc == 1.0
        assert abs(osc_heavy.mix_noise - 0.5 * np.exp(-2.878231525 * 0.5)) < 1e-9
        assert _Derived(VoiceParams(osc_mix=1.0), SAMPLE_RATE).mix_noise == 0.0
        assert _Derived(VoiceParams(osc_mix=0.0), SAMPLE_RATE).mix_osc == 0.0
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


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_zero_length_process(self):
        ch = DrumChannel(0, SAMPLE_RATE)
        ch.trigger()
        assert ch.process(0).shape == (0, 2)

    def test_very_long_process(self):
        audio = _render({'osc_frequency': 440, 'osc_decay': 5000, 'osc_noise_mix': 1.0},
                        n=SAMPLE_RATE * 10)
        assert audio.shape == (SAMPLE_RATE * 10, 2)
        assert np.max(np.abs(audio)) <= 1.1

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
        ch = DrumChannel(0, SAMPLE_RATE)
        ch.set_osc_frequency(1)
        assert ch.oscillator.frequency == 20.0
        ch.set_osc_frequency(100000)
        assert ch.oscillator.frequency == 20000.0
        ch.set_noise_filter_q(1e9)
        assert ch.noise_gen.filter_q == 10000.0


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
    
    def test_block_size_independence(self):
        """Rendering in small chunks must match a single render"""
        params = {'osc_waveform': 2, 'osc_frequency': 440, 'osc_decay': 300,
                  'pitch_mod_amount': 12.0, 'pitch_mod_rate': 100.0, 'osc_noise_mix': 1.0}
        whole = _render(params, n=4096)
        chunked = _render(params, n=4096, chunk=37)
        np.testing.assert_allclose(chunked, whole, atol=1e-6)


class TestVelocitySensitivity:
    """Velocity law: x = 1 - 2 (127 - v) / 126 * sens/2, gain = x * exp(4.26 (x - 1))."""

    def test_full_velocity_is_unity(self):
        from pythonic.voice import velocity_gain, velocity_factor
        for sens in (0.0, 0.5, 1.0, 2.0):
            assert abs(velocity_gain(127, sens) - 1.0) < 1e-9
            assert abs(velocity_factor(127, sens) - 1.0) < 1e-9

    def test_reference_values(self):
        """Ratios measured on the VEL 50/100 reference renders (v=64)."""
        from pythonic.voice import velocity_gain
        assert abs(velocity_gain(64, 1.0) - 0.0594) < 5e-4
        assert abs(velocity_gain(64, 0.5) - 0.2586) < 5e-4

    def test_200_percent_silences_half_velocity(self):
        from pythonic.voice import velocity_gain
        assert velocity_gain(64, 2.0) == 0.0

    def test_zero_sensitivity_no_velocity_effect(self):
        from pythonic.voice import velocity_gain
        assert velocity_gain(1, 0.0) == 1.0

    def test_channel_velocity_scales_output(self):
        ch_hi = DrumChannel(0, SAMPLE_RATE)
        ch_lo = DrumChannel(0, SAMPLE_RATE)
        for ch in (ch_hi, ch_lo):
            ch.set_parameters({'osc_noise_mix': 1.0, 'osc_vel_sensitivity': 1.0})
        ch_hi.trigger(127)
        ch_lo.trigger(64)
        hi = np.max(np.abs(ch_hi.process(2048)))
        lo = np.max(np.abs(ch_lo.process(2048)))
        assert abs(lo / hi - 0.0594) < 2e-3


def run_tests_standalone():
    """Run tests without pytest for environments where pytest is not available"""
    import traceback
    
    test_classes = [
        TestPitchModulation,
        TestNoise,
        TestDrumChannel,
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
