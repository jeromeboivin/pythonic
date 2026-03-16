"""
Tests for the AI Drum Generator inference adapter and patch conversion bridge.
"""

import numpy as np
import sys
import os

try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pythonic.drum_generator import (
    PatchGenerator, PatchPreprocessor, SLOT_MAP, DRUM_TYPES,
    CONTINUOUS_PARAMS, CATEGORICAL_PARAMS, is_torch_available,
)
from pythonic.preset_manager import convert_drum_patch_data, apply_drum_patch_to_channel
from pythonic.synthesizer import PythonicSynthesizer
from gui.drum_generator_dialog import BufferedPatternPreviewSource


# ─────────────────────────────────────────────
# Preprocessor / decode tests (no torch needed)
# ─────────────────────────────────────────────

class TestPatchPreprocessor:
    def _make_preprocessor(self):
        pp = PatchPreprocessor()
        # Load a fake but valid scaler state
        n = pp.cont_dim
        pp.scaler.load_state({
            "scale_": np.ones(n).tolist(),
            "min_": np.zeros(n).tolist(),
            "data_min_": np.zeros(n).tolist(),
            "data_max_": np.ones(n).tolist(),
        })
        return pp

    def test_encode_type_shape(self):
        pp = self._make_preprocessor()
        oh = pp.encode_type("bd")
        assert oh.shape == (pp.type_dim,)
        assert oh.sum() == 1.0

    def test_encode_type_unknown_defaults_to_other(self):
        pp = self._make_preprocessor()
        oh_unknown = pp.encode_type("nonexistent_type")
        oh_other = pp.encode_type("other")
        assert np.array_equal(oh_unknown, oh_other)

    def test_decode_patch_keys(self):
        pp = self._make_preprocessor()
        vec = np.zeros(pp.param_dim, dtype=np.float32)
        # Set one-hot for each categorical to the first value
        offset = pp.cont_dim
        for vals in CATEGORICAL_PARAMS.values():
            vec[offset] = 1.0
            offset += len(vals)

        patch = pp.decode_patch(vec, "bd", name="TestKick")
        # Must have all continuous params
        for p in CONTINUOUS_PARAMS:
            assert p in patch, f"Missing continuous param: {p}"
        # Must have all categorical params
        for p in CATEGORICAL_PARAMS:
            assert p in patch, f"Missing categorical param: {p}"
        # Must have metadata
        assert patch["Name"] == "TestKick"
        assert "Pan" in patch
        assert "Output" in patch

    def test_decode_patch_categorical_argmax(self):
        pp = self._make_preprocessor()
        vec = np.zeros(pp.param_dim, dtype=np.float32)
        # Set OscWave → Saw (index 2)
        offset = pp.cont_dim
        vec[offset + 2] = 10.0  # large logit for Saw
        patch = pp.decode_patch(vec, "sd")
        assert patch["OscWave"] == "Saw"


# ─────────────────────────────────────────────
# Slot map validation
# ─────────────────────────────────────────────

class TestSlotMap:
    def test_eight_slots(self):
        assert len(SLOT_MAP) == 8

    def test_all_types_valid(self):
        for idx, (label, allowed) in SLOT_MAP.items():
            for t in allowed:
                assert t in DRUM_TYPES, f"Slot {idx} has invalid type: {t}"

    def test_slot_indices(self):
        assert set(SLOT_MAP.keys()) == set(range(8))


# ─────────────────────────────────────────────
# Patch conversion bridge
# ─────────────────────────────────────────────

class TestConversionBridge:
    def _make_raw_patch(self):
        return {
            "Name": "TestPatch",
            "OscFreq": 100.0,
            "OscWave": "Sine",
            "OscDcy": 500.0,
            "ModMode": "Decay",
            "ModAmt": 12.0,
            "ModRate": 80.0,
            "NFilMod": "LP",
            "NFilFrq": 3000.0,
            "NFilQ": 2.0,
            "NStereo": "Off",
            "NEnvMod": "Exp",
            "NEnvAtk": 0.0,
            "NEnvDcy": 300.0,
            "Mix": 50.0,
            "DistAmt": 10.0,
            "EQFreq": 1000.0,
            "EQGain": 0.0,
            "Level": 0.0,
            "Pan": 0.0,
            "Output": "A",
            "OscVel": 100.0,
            "NVel": 100.0,
            "ModVel": 0.0,
        }

    def test_convert_returns_expected_keys(self):
        patch = self._make_raw_patch()
        result = convert_drum_patch_data(patch)
        expected_keys = [
            'name', 'osc_frequency', 'osc_waveform', 'osc_attack', 'osc_decay',
            'pitch_mod_mode', 'pitch_mod_amount', 'pitch_mod_rate',
            'noise_filter_mode', 'noise_filter_freq', 'noise_filter_q',
            'noise_envelope_mode', 'noise_attack', 'noise_decay',
            'osc_noise_mix', 'distortion', 'eq_frequency', 'eq_gain_db',
            'level_db', 'pan', 'osc_vel_sensitivity', 'noise_vel_sensitivity',
            'mod_vel_sensitivity',
        ]
        for k in expected_keys:
            assert k in result, f"Missing key: {k}"

    def test_convert_ms_to_seconds(self):
        patch = self._make_raw_patch()
        patch["OscDcy"] = 1000.0  # 1000ms
        result = convert_drum_patch_data(patch)
        assert abs(result['osc_decay'] - 1.0) < 1e-6

    def test_convert_mix_scale(self):
        patch = self._make_raw_patch()
        patch["Mix"] = 75.0
        result = convert_drum_patch_data(patch)
        assert abs(result['osc_noise_mix'] - 0.75) < 1e-6

    def test_convert_waveform_mapping(self):
        patch = self._make_raw_patch()
        patch["OscWave"] = "Triangle"
        result = convert_drum_patch_data(patch)
        assert result['osc_waveform'] == 1  # Triangle = 1

    def test_apply_to_channel(self):
        from pythonic.drum_channel import DrumChannel
        channel = DrumChannel(0, 44100)
        patch = self._make_raw_patch()
        data = convert_drum_patch_data(patch)
        # Should not raise
        apply_drum_patch_to_channel(channel, data)
        assert channel.name == "TestPatch"
        assert abs(channel.oscillator.frequency - 100.0) < 1e-6


class TestLoopPreviewAudioPath:
    class _DummyPreviewSource:
        def __init__(self, value: float):
            self.value = value
            self.stop_called = False

        def read(self, num_samples: int) -> np.ndarray:
            return np.full((num_samples, 2), self.value, dtype=np.float32)

        def stop(self):
            self.stop_called = True

    def test_synth_mixes_preview_source(self):
        synth = PythonicSynthesizer(44100)
        source = self._DummyPreviewSource(0.125)

        synth.set_preview_source(source)
        audio = synth.process_audio(256)
        synth.clear_preview_source()

        assert np.allclose(audio, 0.125, atol=1e-6)
        assert source.stop_called

    def test_buffered_preview_renderer_prefills_audio(self):
        preview_synth = PythonicSynthesizer(
            44100,
            parallel_channel_processing=True,
        )
        trigger_table = [[0]] + [[] for _ in range(15)]
        source = BufferedPatternPreviewSource(
            preview_synth,
            [trigger_table],
            bpm=120,
        )

        source.start(prefill_blocks=1)
        audio = source.read(1050)
        source.stop()

        assert audio.shape == (1050, 2)
        assert np.max(np.abs(audio)) > 0.0


# ─────────────────────────────────────────────
# Generator (requires torch)
# ─────────────────────────────────────────────

class TestPatchGenerator:
    """Tests that require a real checkpoint. Skipped when torch or checkpoint is unavailable."""

    @staticmethod
    def _get_checkpoint():
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for name in ("drum_cvae_best.pt", "drum_cvae_final.pt"):
            path = os.path.join(base, name)
            if os.path.isfile(path):
                return path
        return None

    def _skip_if_unavailable(self):
        if not is_torch_available():
            if PYTEST_AVAILABLE:
                pytest.skip("torch not installed")
            return True
        if self._get_checkpoint() is None:
            if PYTEST_AVAILABLE:
                pytest.skip("no checkpoint found")
            return True
        return False

    def test_load_and_generate(self):
        if self._skip_if_unavailable():
            return
        gen = PatchGenerator()
        gen.load_model(self._get_checkpoint())
        assert gen.is_loaded

        patches = gen.generate("bd", n=3, temperature=1.0, seed=42)
        assert len(patches) == 3
        for p in patches:
            for param in CONTINUOUS_PARAMS:
                assert param in p
            for param in CATEGORICAL_PARAMS:
                assert param in p

    def test_seed_reproducibility(self):
        if self._skip_if_unavailable():
            return
        gen = PatchGenerator()
        gen.load_model(self._get_checkpoint())

        a = gen.generate("sd", n=2, seed=123)
        b = gen.generate("sd", n=2, seed=123)
        for pa, pb in zip(a, b):
            for param in CONTINUOUS_PARAMS:
                assert abs(pa[param] - pb[param]) < 1e-6, f"Mismatch on {param}"

    def test_sampling_mode_validation(self):
        if self._skip_if_unavailable():
            return
        gen = PatchGenerator()
        gen.load_model(self._get_checkpoint())

        with pytest.raises(ValueError):
            gen.generate("bd", n=1, sampling_mode="not-a-mode")

    def test_explicit_prior_sampling(self):
        if self._skip_if_unavailable():
            return
        gen = PatchGenerator()
        gen.load_model(self._get_checkpoint())

        patches = gen.generate("bd", n=2, temperature=0.1, seed=7, sampling_mode="prior")
        assert len(patches) == 2

    def test_empirical_bank_detected_when_cache_present(self):
        if self._skip_if_unavailable():
            return
        gen = PatchGenerator()
        gen.load_model(self._get_checkpoint())

        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_path = os.path.join(base, "drum_dataset_cache.pt")
        if os.path.isfile(cache_path):
            assert gen.has_empirical_bank

    def test_generate_for_slot_validates_type(self):
        if self._skip_if_unavailable():
            return
        gen = PatchGenerator()
        gen.load_model(self._get_checkpoint())

        # Valid override
        patches = gen.generate_for_slot(2, type_override="shaker")
        assert len(patches) == 1

        # Invalid override
        try:
            gen.generate_for_slot(0, type_override="oh")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_patch_field_ranges(self):
        """Verify generated patches have plausible field values."""
        if self._skip_if_unavailable():
            return
        gen = PatchGenerator()
        gen.load_model(self._get_checkpoint())
        patches = gen.generate("bd", n=10, temperature=1.0, seed=99)
        for p in patches:
            # Frequency should be positive
            assert p["OscFreq"] > 0
            # Categorical values should be valid
            assert p["OscWave"] in CATEGORICAL_PARAMS["OscWave"]
            assert p["ModMode"] in CATEGORICAL_PARAMS["ModMode"]
            assert p["NFilMod"] in CATEGORICAL_PARAMS["NFilMod"]
            assert p["NEnvMod"] in CATEGORICAL_PARAMS["NEnvMod"]


if __name__ == "__main__":
    if PYTEST_AVAILABLE:
        pytest.main([__file__, "-v"])
    else:
        # Minimal manual test runner
        import traceback
        test_classes = [
            TestPatchPreprocessor,
            TestSlotMap,
            TestConversionBridge,
            TestLoopPreviewAudioPath,
            TestPatchGenerator,
        ]
        passed = failed = skipped = 0
        for cls in test_classes:
            inst = cls()
            for name in dir(inst):
                if name.startswith("test_"):
                    try:
                        getattr(inst, name)()
                        passed += 1
                        print(f"  PASS: {cls.__name__}.{name}")
                    except Exception as e:
                        if "skip" in str(e).lower():
                            skipped += 1
                            print(f"  SKIP: {cls.__name__}.{name}")
                        else:
                            failed += 1
                            print(f"  FAIL: {cls.__name__}.{name}")
                            traceback.print_exc()
        print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
