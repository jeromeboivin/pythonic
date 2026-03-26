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
from pythonic.pattern_manager import PatternManager
from pythonic.pattern_generator import PatternGenerator, _BUNDLED_CHECKPOINT


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


# ─────────────────────────────────────────────
# Dialog preview logic (no GUI required)
# ─────────────────────────────────────────────

class _FakeWidget:
    """Minimal widget stub for dialog tests."""
    def config(self, **kw):
        self._last_config = kw
    def get(self):
        return True


def _make_dialog_headless():
    """Build a DrumGeneratorDialog without Tk, by monkey-patching _build_dialog."""
    from gui.drum_generator_dialog import DrumGeneratorDialog

    synth = PythonicSynthesizer(44100)
    pm = PatternManager()
    # Setup a simple pattern: channel 0 triggers on steps 0, 4, 8, 12
    pattern = pm.get_pattern(0)
    for step in [0, 4, 8, 12]:
        pattern.get_channel(0).set_trigger(step, True)
    pattern.get_channel(0).set_accent(0, True)

    class FakePrefs:
        _data = {}
        def get(self, key, default=None):
            return self._data.get(key, default)
        def set(self, key, value):
            self._data[key] = value

    transport_log = []

    def start_transport():
        pm.start_playback(pm.selected_pattern_index)
        transport_log.append('start')

    def stop_transport():
        pm.stop_playback()
        transport_log.append('stop')

    # Patch out Tk UI construction
    original_build = DrumGeneratorDialog._build_dialog
    original_status = DrumGeneratorDialog._update_model_status
    DrumGeneratorDialog._build_dialog = lambda self: None
    DrumGeneratorDialog._update_model_status = lambda self: None
    try:
        dialog = DrumGeneratorDialog(
            parent=None,
            synth=synth,
            pattern_manager=pm,
            preferences_manager=FakePrefs(),
            start_transport=start_transport,
            stop_transport=stop_transport,
        )
    finally:
        DrumGeneratorDialog._build_dialog = original_build
        DrumGeneratorDialog._update_model_status = original_status

    # Stub the UI widgets / Tk objects that methods reference
    dialog.dialog = type('FakeToplevel', (), {'destroy': lambda self: None})()
    dialog.preview_loop_btn = _FakeWidget()
    dialog.preview_bank_btn = _FakeWidget()
    dialog.model_status_label = _FakeWidget()
    dialog.temp_var = type('V', (), {'get': lambda s: 1.0})()
    dialog.pattern_temp_var = type('V', (), {'get': lambda s: 1.0})()

    return dialog, synth, pm, transport_log


class TestDialogOneShotPreview:
    """One-shot preview must use the live synth at velocity 127 (like keys 1-8)."""

    def test_preview_slot_triggers_live_synth_at_velocity_127(self):
        dialog, synth, pm, _ = _make_dialog_headless()

        # Inject a candidate patch into slot 0
        raw_patch = {
            "Name": "TestKick", "OscFreq": 55.0, "OscWave": "Sine",
            "OscAtk": 0.0, "OscDcy": 500.0, "ModMode": "Decay",
            "ModAmt": 12.0, "ModRate": 80.0, "NFilMod": "LP",
            "NFilFrq": 3000.0, "NFilQ": 2.0, "NStereo": "Off",
            "NEnvMod": "Exp", "NEnvAtk": 0.0, "NEnvDcy": 300.0,
            "Mix": 50.0, "DistAmt": 10.0, "EQFreq": 1000.0,
            "EQGain": 0.0, "Level": 0.0, "Pan": 0.0, "Output": "A",
            "OscVel": 100.0, "NVel": 100.0, "ModVel": 0.0,
        }
        dialog.slot_state[0]['candidates'] = [raw_patch]

        # Record trigger calls
        trigger_log = []
        orig_trigger = synth.trigger_drum
        def spy_trigger(ch, velocity=127):
            trigger_log.append((ch, velocity))
            orig_trigger(ch, velocity)
        synth.trigger_drum = spy_trigger

        dialog._on_preview_slot(0)

        # Must have triggered channel 0 at velocity 127
        assert len(trigger_log) == 1
        assert trigger_log[0] == (0, 127)

        # The patch should now be on the live synth channel
        assert synth.channels[0].name == "TestKick"

    def test_preview_slot_noop_when_no_candidates(self):
        dialog, synth, pm, _ = _make_dialog_headless()

        trigger_log = []
        orig_trigger = synth.trigger_drum
        synth.trigger_drum = lambda ch, velocity=127: trigger_log.append((ch, velocity))

        dialog._on_preview_slot(3)
        assert len(trigger_log) == 0


class TestDialogLoopPreview:
    """Loop preview must use the real transport callbacks."""

    def test_start_loop_invokes_transport(self):
        dialog, synth, pm, log = _make_dialog_headless()
        dialog._start_loop_preview()
        assert 'start' in log
        assert dialog.preview_playing is True

    def test_stop_loop_invokes_transport_and_restores(self):
        dialog, synth, pm, log = _make_dialog_headless()
        # Save original frequency for channel 0
        orig_freq = synth.channels[0].oscillator.frequency

        dialog._start_loop_preview()
        dialog._stop_loop_preview()

        assert 'stop' in log
        assert dialog.preview_playing is False
        # Channel should be restored to original state
        assert abs(synth.channels[0].oscillator.frequency - orig_freq) < 1e-6


class TestDialogBankPreview:
    """Bank preview must chain patterns and use the real transport."""

    def test_bank_preview_chains_patterns(self):
        dialog, synth, pm, log = _make_dialog_headless()
        dialog._start_bank_preview()

        # All patterns except last should be chained
        for i in range(11):
            assert pm.patterns[i].chained_to_next is True
        assert pm.patterns[11].chained_to_next is False

        assert 'start' in log
        assert dialog.preview_playing is True

    def test_bank_preview_restores_chain_state(self):
        dialog, synth, pm, log = _make_dialog_headless()

        # Ensure no patterns are chained initially
        for p in pm.patterns:
            assert p.chained_to_next is False

        dialog._start_bank_preview()
        dialog._stop_loop_preview()

        # Patterns should be restored to unchained
        for p in pm.patterns:
            assert p.chained_to_next is False


class TestDialogChannelRestore:
    """Dialog must restore non-applied channels on close."""

    def test_close_restores_unapplied_channels(self):
        dialog, synth, pm, _ = _make_dialog_headless()
        freq_before = synth.channels[0].oscillator.frequency

        # Manually change channel 0
        synth.channels[0].set_osc_frequency(9999.0)
        assert abs(synth.channels[0].oscillator.frequency - 9999.0) < 1

        # Close dialog (without having applied slot 0)
        dialog._on_close()

        # Channel 0 should be restored
        assert abs(synth.channels[0].oscillator.frequency - freq_before) < 1e-6

    def test_close_preserves_applied_channels(self):
        dialog, synth, pm, _ = _make_dialog_headless()

        # Mark slot 2 as applied
        dialog._applied_slots.add(2)
        dialog._saved_channel_states[2] = synth.channels[2].get_parameters()

        # Change channel 2
        synth.channels[2].set_osc_frequency(9999.0)

        dialog._on_close()

        # Channel 2 should NOT be restored (it was applied)
        assert abs(synth.channels[2].oscillator.frequency - 9999.0) < 1

    def test_close_restores_unapplied_patterns(self):
        dialog, synth, pm, _ = _make_dialog_headless()

        # Pattern A has triggers on steps 0,4,8,12
        assert pm.patterns[0].channels[0].steps[0].trigger is True

        # Simulate what loop preview does: save patterns, then modify
        dialog._save_patterns()
        pm.patterns[0].clear()
        assert pm.patterns[0].channels[0].steps[0].trigger is False

        dialog._on_close()

        # Patterns should be restored
        assert pm.patterns[0].channels[0].steps[0].trigger is True

    def test_close_preserves_applied_patterns(self):
        dialog, synth, pm, _ = _make_dialog_headless()

        dialog._patterns_applied = True
        pm.patterns[0].clear()

        dialog._on_close()

        # Patterns should stay cleared (they were applied)
        assert pm.patterns[0].channels[0].steps[0].trigger is False


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


class TestPatternModelResolution:
    """Tests for PatternGenerator.resolve_model_path and ensure_loaded."""

    def test_resolve_returns_bundled_when_no_pref(self):
        """resolve_model_path should return bundled checkpoint when it exists."""
        path = PatternGenerator.resolve_model_path(preferences_manager=None)
        if os.path.isfile(_BUNDLED_CHECKPOINT):
            assert path == _BUNDLED_CHECKPOINT
        else:
            assert path is None

    def test_resolve_prefers_saved_path(self):
        """resolve_model_path should prefer a saved preference over bundled."""
        class FakePrefs:
            def get(self, key, default=None):
                if key == 'drum_generator_pattern_model_path':
                    return _BUNDLED_CHECKPOINT  # reuse bundled as stand-in
                return default
        if os.path.isfile(_BUNDLED_CHECKPOINT):
            path = PatternGenerator.resolve_model_path(FakePrefs())
            assert path == _BUNDLED_CHECKPOINT

    def test_resolve_ignores_missing_saved_path(self):
        """resolve_model_path should skip a saved path that doesn't exist."""
        class FakePrefs:
            def get(self, key, default=None):
                if key == 'drum_generator_pattern_model_path':
                    return '/nonexistent/path/model.pt'
                return default
        path = PatternGenerator.resolve_model_path(FakePrefs())
        if os.path.isfile(_BUNDLED_CHECKPOINT):
            assert path == _BUNDLED_CHECKPOINT
        else:
            assert path is None

    def test_ensure_loaded_returns_false_when_no_model(self):
        """ensure_loaded should return False when no model is available."""
        class FakePrefs:
            def get(self, key, default=None):
                return default
        # Temporarily hide the bundled checkpoint
        gen = PatternGenerator()
        import pythonic.pattern_generator as pg
        orig = pg._BUNDLED_CHECKPOINT
        pg._BUNDLED_CHECKPOINT = '/nonexistent/path.pt'
        try:
            result = gen.ensure_loaded(FakePrefs())
            assert result is False
        finally:
            pg._BUNDLED_CHECKPOINT = orig

    def test_ensure_loaded_succeeds_with_bundled(self):
        """ensure_loaded should succeed when the bundled checkpoint exists."""
        if not is_torch_available():
            if PYTEST_AVAILABLE:
                pytest.skip("torch not installed")
            return
        if not os.path.isfile(_BUNDLED_CHECKPOINT):
            if PYTEST_AVAILABLE:
                pytest.skip("no bundled checkpoint")
            return
        gen = PatternGenerator()
        assert gen.ensure_loaded() is True
        assert gen.is_loaded


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
            TestDialogOneShotPreview,
            TestDialogLoopPreview,
            TestDialogBankPreview,
            TestDialogChannelRestore,
            TestPatchGenerator,
            TestPatternModelResolution,
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
