"""
LFO and Modulation System Tests

Tests for LFO waveform generation, PumpSource envelope, modulation routing,
DrumChannel integration, preset serialization, and synthesizer global targets.

Run with: pytest tests/test_lfo.py -v
Or: python tests/test_lfo.py
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

from pythonic.lfo import (
    LFO, PumpSource, ModulationRouter,
    LFOWaveform, LFORetrigger, LFOPolarity, SyncDivision,
    ModTarget, MOD_TARGET_GROUPS, MOD_TARGET_LABELS, MOD_TARGET_DEPTH_RANGES,
)
from pythonic.drum_channel import DrumChannel
from pythonic.synthesizer import PythonicSynthesizer

SAMPLE_RATE = 44100
BLOCK_SIZE = 512


# ---------------------------------------------------------------------------
# LFO waveform generation
# ---------------------------------------------------------------------------

class TestLFOWaveforms:
    """Test that each waveform produces expected output characteristics."""

    # PITCH_SEMITONES max range = 48 st
    _TARGET_MAX = MOD_TARGET_DEPTH_RANGES[ModTarget.PITCH_SEMITONES][1]

    def _make_lfo(self, waveform, rate_hz=10.0, depth=1.0,
                  polarity=LFOPolarity.BIPOLAR):
        lfo = LFO(SAMPLE_RATE)
        lfo.enabled = True
        lfo.waveform = waveform
        lfo.rate_hz = rate_hz
        lfo.depth = depth
        lfo.polarity = polarity
        lfo.target = ModTarget.PITCH_SEMITONES
        lfo.retrigger = LFORetrigger.FREE
        return lfo

    def test_disabled_lfo_returns_zero(self):
        """A disabled LFO should always return 0."""
        lfo = LFO(SAMPLE_RATE)
        lfo.enabled = False
        lfo.target = ModTarget.PITCH_SEMITONES
        lfo.depth = 10.0
        assert lfo.process(BLOCK_SIZE) == 0.0

    def test_no_target_returns_zero(self):
        """An LFO with target NONE should return 0 even when enabled."""
        lfo = LFO(SAMPLE_RATE)
        lfo.enabled = True
        lfo.target = ModTarget.NONE
        lfo.depth = 10.0
        assert lfo.process(BLOCK_SIZE) == 0.0

    def test_sine_bounded_bipolar(self):
        """Sine LFO in bipolar mode should stay within effective depth bounds."""
        lfo = self._make_lfo(LFOWaveform.SINE, depth=50.0)
        eff = (50.0 / 100.0) * self._TARGET_MAX
        values = [lfo.process(BLOCK_SIZE) for _ in range(200)]
        assert min(values) >= -(eff + 0.01)
        assert max(values) <= (eff + 0.01)

    def test_sine_bounded_unipolar(self):
        """Sine LFO in unipolar mode should stay within [0, effective depth]."""
        lfo = self._make_lfo(LFOWaveform.SINE, depth=50.0,
                             polarity=LFOPolarity.UNIPOLAR)
        eff = (50.0 / 100.0) * self._TARGET_MAX
        values = [lfo.process(BLOCK_SIZE) for _ in range(200)]
        assert min(values) >= -0.01
        assert max(values) <= (eff + 0.01)

    def test_triangle_bounded(self):
        """Triangle LFO bipolar within effective depth bounds."""
        lfo = self._make_lfo(LFOWaveform.TRIANGLE, depth=30.0)
        eff = (30.0 / 100.0) * self._TARGET_MAX
        values = [lfo.process(BLOCK_SIZE) for _ in range(200)]
        assert min(values) >= -(eff + 0.01)
        assert max(values) <= (eff + 0.01)

    def test_saw_up_bounded(self):
        """Saw-up LFO bipolar within effective depth bounds."""
        lfo = self._make_lfo(LFOWaveform.SAW_UP, depth=20.0)
        eff = (20.0 / 100.0) * self._TARGET_MAX
        values = [lfo.process(BLOCK_SIZE) for _ in range(200)]
        assert min(values) >= -(eff + 0.01)
        assert max(values) <= (eff + 0.01)

    def test_saw_down_bounded(self):
        """Saw-down LFO bipolar within effective depth bounds."""
        lfo = self._make_lfo(LFOWaveform.SAW_DOWN, depth=20.0)
        eff = (20.0 / 100.0) * self._TARGET_MAX
        values = [lfo.process(BLOCK_SIZE) for _ in range(200)]
        assert min(values) >= -(eff + 0.01)
        assert max(values) <= (eff + 0.01)

    def test_square_values(self):
        """Square LFO should only produce +eff or -eff."""
        lfo = self._make_lfo(LFOWaveform.SQUARE, depth=70.0)
        eff = (70.0 / 100.0) * self._TARGET_MAX
        values = [lfo.process(BLOCK_SIZE) for _ in range(200)]
        for v in values:
            assert abs(abs(v) - eff) < 0.01, f"Square value {v} not ±{eff}"

    def test_sample_and_hold_changes(self):
        """S&H should produce different values across many blocks."""
        lfo = self._make_lfo(LFOWaveform.SAMPLE_AND_HOLD, rate_hz=20.0, depth=1.0)
        values = [lfo.process(BLOCK_SIZE) for _ in range(500)]
        unique_values = len(set(round(v, 6) for v in values))
        # Should have more than one unique value (statistical certainty)
        assert unique_values > 5, f"S&H produced only {unique_values} unique values"

    def test_zero_depth_returns_zero(self):
        """Depth 0 should always produce 0 regardless of waveform."""
        for wf in LFOWaveform:
            lfo = self._make_lfo(wf, depth=0.0)
            for _ in range(20):
                assert lfo.process(BLOCK_SIZE) == 0.0, f"{wf.name} with depth=0 not zero"


# ---------------------------------------------------------------------------
# LFO phase and retrigger
# ---------------------------------------------------------------------------

class TestLFORetrigger:
    """Test phase reset and retrigger behavior."""

    def test_retrigger_resets_phase(self):
        """reset_phase() should restart the LFO cycle."""
        lfo = LFO(SAMPLE_RATE)
        lfo.enabled = True
        lfo.waveform = LFOWaveform.SAW_UP
        lfo.rate_hz = 1.0
        lfo.depth = 1.0
        lfo.target = ModTarget.PITCH_SEMITONES
        lfo.retrigger = LFORetrigger.RETRIGGER

        # Advance some blocks
        for _ in range(50):
            lfo.process(BLOCK_SIZE)

        # Reset and read first value — should be near start of cycle
        lfo.reset_phase()
        val_after_reset = lfo.process(BLOCK_SIZE)
        # Saw-up at phase ≈0 → raw ≈ -1 → output ≈ -depth
        assert val_after_reset < 0, "Saw-up should be negative right after phase reset"

    def test_phase_offset(self):
        """phase_offset=0.5 should start the saw-up cycle at midpoint."""
        lfo = LFO(SAMPLE_RATE)
        lfo.enabled = True
        lfo.waveform = LFOWaveform.SAW_UP
        lfo.rate_hz = 0.5
        lfo.depth = 1.0
        lfo.target = ModTarget.PITCH_SEMITONES
        lfo.phase_offset = 0.5

        lfo.reset_phase()
        val = lfo.process(BLOCK_SIZE)
        # At phase=0.5, raw = 2*0.5 - 1 = 0 → output ≈ 0
        assert abs(val) < 0.1, f"Saw-up at phase_offset=0.5 should be near 0, got {val}"


# ---------------------------------------------------------------------------
# LFO tempo sync
# ---------------------------------------------------------------------------

class TestLFOSync:
    """Test tempo-synced rate calculation."""

    def test_quarter_note_sync(self):
        """At 120 BPM, 1/4 sync = 2 Hz (one cycle per quarter note)."""
        lfo = LFO(SAMPLE_RATE)
        lfo.enabled = True
        lfo.waveform = LFOWaveform.SINE
        lfo.depth = 1.0
        lfo.target = ModTarget.PITCH_SEMITONES
        lfo.sync = SyncDivision.QUARTER

        # At 120 BPM, one quarter = 0.5s, rate = 2 Hz
        # Collect values over 1 second (= 2 full cycles expected)
        blocks_per_second = SAMPLE_RATE // BLOCK_SIZE
        values = [lfo.process(BLOCK_SIZE, bpm=120.0) for _ in range(blocks_per_second)]

        # Count sign changes → indicates oscillation
        sign_changes = sum(1 for i in range(1, len(values))
                          if values[i] * values[i-1] < 0)
        # 2 Hz sine → ~4 zero crossings per second
        assert sign_changes >= 2, f"Expected oscillation, got {sign_changes} sign changes"

    def test_sync_off_uses_free_rate(self):
        """With sync OFF, the free rate_hz should be used."""
        lfo = LFO(SAMPLE_RATE)
        lfo.enabled = True
        lfo.waveform = LFOWaveform.SINE
        lfo.rate_hz = 10.0
        lfo.depth = 1.0
        lfo.target = ModTarget.PITCH_SEMITONES
        lfo.sync = SyncDivision.OFF

        values = [lfo.process(BLOCK_SIZE, bpm=120.0) for _ in range(200)]
        sign_changes = sum(1 for i in range(1, len(values))
                          if values[i] * values[i-1] < 0)
        # 10 Hz should produce many sign changes
        assert sign_changes > 20, f"10 Hz LFO should oscillate rapidly, got {sign_changes} sign changes"


# ---------------------------------------------------------------------------
# PumpSource
# ---------------------------------------------------------------------------

class TestPumpSource:
    """Test the sidechain / pump envelope source."""

    def test_disabled_returns_zero(self):
        """Disabled pump returns 0."""
        pump = PumpSource(SAMPLE_RATE)
        pump.enabled = False
        pump.amount = 1.0
        pump.trigger()
        assert pump.process(BLOCK_SIZE) == 0.0

    def test_trigger_produces_negative_output(self):
        """After trigger, pump should produce negative modulation."""
        pump = PumpSource(SAMPLE_RATE)
        pump.enabled = True
        pump.amount = 1.0
        pump.release_ms = 500.0
        pump.target = ModTarget.LEVEL_DB

        pump.trigger()
        val = pump.process(BLOCK_SIZE)
        assert val < 0, f"Pump should produce negative value after trigger, got {val}"

    def test_pump_recovers_to_zero(self):
        """Pump should recover to 0 after enough time."""
        pump = PumpSource(SAMPLE_RATE)
        pump.enabled = True
        pump.amount = 1.0
        pump.release_ms = 50.0
        pump.target = ModTarget.LEVEL_DB

        pump.trigger()
        # Process enough blocks for full recovery
        for _ in range(200):
            val = pump.process(BLOCK_SIZE)
        assert abs(val) < 0.001, f"Pump should recover to ~0, got {val}"

    def test_pump_amount_scales_output(self):
        """Higher amount should produce deeper ducking."""
        pump_lo = PumpSource(SAMPLE_RATE)
        pump_lo.enabled = True
        pump_lo.amount = 0.3
        pump_lo.release_ms = 200.0
        pump_lo.target = ModTarget.LEVEL_DB

        pump_hi = PumpSource(SAMPLE_RATE)
        pump_hi.enabled = True
        pump_hi.amount = 1.0
        pump_hi.release_ms = 200.0
        pump_hi.target = ModTarget.LEVEL_DB

        pump_lo.trigger()
        pump_hi.trigger()
        val_lo = pump_lo.process(BLOCK_SIZE)
        val_hi = pump_hi.process(BLOCK_SIZE)

        assert abs(val_hi) > abs(val_lo), \
            f"Higher amount should duck more: lo={val_lo}, hi={val_hi}"


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------

class TestLFOSerialization:
    """Test get_parameters / set_parameters round-trip."""

    def test_lfo_roundtrip(self):
        """LFO parameters survive get/set cycle."""
        lfo = LFO(SAMPLE_RATE)
        lfo.enabled = True
        lfo.waveform = LFOWaveform.SAW_DOWN
        lfo.rate_hz = 3.7
        lfo.sync = SyncDivision.DOTTED_EIGHTH
        lfo.depth = 42.0
        lfo.target = ModTarget.NOISE_FILTER_FREQ
        lfo.retrigger = LFORetrigger.FREE
        lfo.phase_offset = 0.25
        lfo.polarity = LFOPolarity.UNIPOLAR

        params = lfo.get_parameters()
        lfo2 = LFO(SAMPLE_RATE)
        lfo2.set_parameters(params)

        assert lfo2.enabled == True
        assert lfo2.waveform == LFOWaveform.SAW_DOWN
        assert abs(lfo2.rate_hz - 3.7) < 0.001
        assert lfo2.sync == SyncDivision.DOTTED_EIGHTH
        assert abs(lfo2.depth - 42.0) < 0.001
        assert lfo2.target == ModTarget.NOISE_FILTER_FREQ
        assert lfo2.retrigger == LFORetrigger.FREE
        assert abs(lfo2.phase_offset - 0.25) < 0.001
        assert lfo2.polarity == LFOPolarity.UNIPOLAR

    def test_pump_roundtrip(self):
        """PumpSource parameters survive get/set cycle."""
        pump = PumpSource(SAMPLE_RATE)
        pump.enabled = True
        pump.amount = 0.65
        pump.attack_ms = 5.0
        pump.release_ms = 250.0
        pump.curve = 0.8
        pump.target = ModTarget.REVERB_MIX
        pump.sync = SyncDivision.EIGHTH

        params = pump.get_parameters()
        pump2 = PumpSource(SAMPLE_RATE)
        pump2.set_parameters(params)

        assert pump2.enabled == True
        assert abs(pump2.amount - 0.65) < 0.001
        assert abs(pump2.attack_ms - 5.0) < 0.001
        assert abs(pump2.release_ms - 250.0) < 0.001
        assert abs(pump2.curve - 0.8) < 0.001
        assert pump2.target == ModTarget.REVERB_MIX
        assert pump2.sync == SyncDivision.EIGHTH

    def test_lfo_set_parameters_handles_missing_keys(self):
        """set_parameters should not crash on partial data."""
        lfo = LFO(SAMPLE_RATE)
        lfo.set_parameters({'depth': 10.0})
        assert abs(lfo.depth - 10.0) < 0.001
        # Other fields should remain at defaults
        assert lfo.enabled == False
        assert lfo.target == ModTarget.NONE

    def test_drum_channel_lfo_serialization(self):
        """DrumChannel get/set_parameters should include LFO state."""
        ch = DrumChannel(0, SAMPLE_RATE)
        ch.lfo1.enabled = True
        ch.lfo1.waveform = LFOWaveform.TRIANGLE
        ch.lfo1.rate_hz = 7.5
        ch.lfo1.depth = 24.0
        ch.lfo1.target = ModTarget.OSC_FREQUENCY
        ch.pump.enabled = True
        ch.pump.amount = 0.9
        ch.pump.target = ModTarget.LEVEL_DB

        params = ch.get_parameters()
        assert 'lfo1' in params
        assert 'lfo2' in params
        assert 'pump' in params
        assert params['lfo1']['enabled'] == True
        assert params['lfo1']['target'] == 'osc_frequency'
        assert params['pump']['amount'] == 0.9

        # Restore to a different channel
        ch2 = DrumChannel(1, SAMPLE_RATE)
        ch2.set_parameters(params)

        assert ch2.lfo1.enabled == True
        assert ch2.lfo1.waveform == LFOWaveform.TRIANGLE
        assert abs(ch2.lfo1.depth - 24.0) < 0.001
        assert ch2.lfo1.target == ModTarget.OSC_FREQUENCY
        assert ch2.pump.enabled == True
        assert abs(ch2.pump.amount - 0.9) < 0.001


# ---------------------------------------------------------------------------
# DrumChannel modulation integration
# ---------------------------------------------------------------------------

class TestDrumChannelModulation:
    """Test that LFOs actually modulate the audio path in DrumChannel."""

    def test_parameter_restored_after_process(self):
        """Base parameter values must be restored after process()."""
        ch = DrumChannel(0, SAMPLE_RATE)
        ch.lfo1.enabled = True
        ch.lfo1.waveform = LFOWaveform.SQUARE
        ch.lfo1.rate_hz = 5.0
        ch.lfo1.depth = 12.0
        ch.lfo1.target = ModTarget.PITCH_SEMITONES

        original = ch.pitch_semitones
        ch.trigger(127)
        ch.process(BLOCK_SIZE)
        assert abs(ch.pitch_semitones - original) < 0.001, \
            "pitch_semitones must be restored after process()"

    def test_level_restored_after_pump(self):
        """Level dB must be restored after pump modulation in process()."""
        ch = DrumChannel(0, SAMPLE_RATE)
        ch.pump.enabled = True
        ch.pump.amount = 1.0
        ch.pump.target = ModTarget.LEVEL_DB
        ch.level_db = -3.0

        ch.trigger(127)
        ch.process(BLOCK_SIZE)
        assert abs(ch.level_db - (-3.0)) < 0.001, \
            "level_db must be restored after process()"

    def test_lfo_produces_audible_difference(self):
        """Audio with active LFO should differ from without."""
        # Without LFO
        ch = DrumChannel(0, SAMPLE_RATE)
        ch.trigger(127)
        audio_no_lfo = ch.process(BLOCK_SIZE).copy()

        # With LFO modulating pitch
        ch2 = DrumChannel(0, SAMPLE_RATE)
        ch2.lfo1.enabled = True
        ch2.lfo1.waveform = LFOWaveform.SQUARE
        ch2.lfo1.rate_hz = 5.0
        ch2.lfo1.depth = 24.0
        ch2.lfo1.target = ModTarget.PITCH_SEMITONES
        ch2.trigger(127)
        audio_with_lfo = ch2.process(BLOCK_SIZE).copy()

        # The outputs should be different
        diff = np.max(np.abs(audio_no_lfo - audio_with_lfo))
        assert diff > 0.01, f"LFO should change audio output, max diff = {diff}"

    def test_inactive_lfo_no_change(self):
        """Disabled LFO should not alter audio."""
        ch1 = DrumChannel(0, SAMPLE_RATE)
        ch1.trigger(127)
        audio1 = ch1.process(BLOCK_SIZE).copy()

        ch2 = DrumChannel(0, SAMPLE_RATE)
        ch2.lfo1.enabled = False
        ch2.lfo1.depth = 50.0
        ch2.lfo1.target = ModTarget.PITCH_SEMITONES
        ch2.trigger(127)
        audio2 = ch2.process(BLOCK_SIZE).copy()

        np.testing.assert_array_almost_equal(audio1, audio2, decimal=10)

    def test_smoothed_param_modulation(self):
        """Modulating a smoothed param (osc_noise_mix) should restore correctly."""
        ch = DrumChannel(0, SAMPLE_RATE)
        ch.set_osc_noise_mix_immediate(0.7)
        ch.lfo1.enabled = True
        ch.lfo1.waveform = LFOWaveform.SINE
        ch.lfo1.rate_hz = 2.0
        ch.lfo1.depth = 0.3
        ch.lfo1.target = ModTarget.OSC_NOISE_MIX

        ch.trigger(127)
        for _ in range(10):
            ch.process(BLOCK_SIZE)
        # The osc_noise_mix attribute itself should not drift
        # (smoothed params are modulated via local variables, not saved/restored)
        assert abs(ch.osc_noise_mix - 0.7) < 0.001

    def test_dual_lfo_accumulate(self):
        """Two LFOs targeting the same param should accumulate their offsets."""
        ch = DrumChannel(0, SAMPLE_RATE)
        ch.lfo1.enabled = True
        ch.lfo1.waveform = LFOWaveform.SQUARE
        ch.lfo1.rate_hz = 1.0
        ch.lfo1.depth = 6.0
        ch.lfo1.target = ModTarget.PITCH_SEMITONES

        ch.lfo2.enabled = True
        ch.lfo2.waveform = LFOWaveform.SQUARE
        ch.lfo2.rate_hz = 1.0
        ch.lfo2.depth = 6.0
        ch.lfo2.target = ModTarget.PITCH_SEMITONES

        # Both square waves at same rate should be in phase → double offset
        ch_single = DrumChannel(0, SAMPLE_RATE)
        ch_single.lfo1.enabled = True
        ch_single.lfo1.waveform = LFOWaveform.SQUARE
        ch_single.lfo1.rate_hz = 1.0
        ch_single.lfo1.depth = 12.0
        ch_single.lfo1.target = ModTarget.PITCH_SEMITONES

        ch.trigger(127)
        ch_single.trigger(127)
        audio_dual = ch.process(BLOCK_SIZE).copy()
        audio_single = ch_single.process(BLOCK_SIZE).copy()

        # Should be very similar since 6+6 = 12
        diff = np.max(np.abs(audio_dual - audio_single))
        assert diff < 0.01, f"Dual LFOs should accumulate like single with 2x depth, diff={diff}"

    def test_multiple_blocks_no_parameter_drift(self):
        """Processing many blocks with active LFO should not cause parameter drift."""
        ch = DrumChannel(0, SAMPLE_RATE)
        ch.lfo1.enabled = True
        ch.lfo1.waveform = LFOWaveform.SINE
        ch.lfo1.rate_hz = 8.0
        ch.lfo1.depth = 20.0
        ch.lfo1.target = ModTarget.LEVEL_DB
        ch.level_db = -6.0
        ch.reverb_decay = 0.5
        ch.reverb_mix = 0.3

        ch.trigger(127)
        for _ in range(100):
            ch.process(BLOCK_SIZE)

        assert abs(ch.level_db - (-6.0)) < 0.001, "level_db drifted"
        assert abs(ch.reverb_decay - 0.5) < 0.001, "reverb_decay drifted"
        assert abs(ch.reverb_mix - 0.3) < 0.001, "reverb_mix drifted"


# ---------------------------------------------------------------------------
# Synthesizer-level integration
# ---------------------------------------------------------------------------

class TestSynthesizerLFOIntegration:
    """Test LFO integration at the synthesizer level."""

    def test_synthesizer_back_reference(self):
        """Each channel should have a back-reference to its synthesizer."""
        s = PythonicSynthesizer()
        for ch in s.channels:
            assert ch._synthesizer is s

    def test_bpm_stored(self):
        """set_bpm should store the BPM on the synthesizer."""
        s = PythonicSynthesizer()
        s.set_bpm(140.0)
        assert abs(s._bpm - 140.0) < 0.001

    def test_synthesizer_processes_with_lfo(self):
        """Full mix path should work with LFO active on a channel."""
        s = PythonicSynthesizer()
        ch = s.channels[0]
        ch.lfo1.enabled = True
        ch.lfo1.waveform = LFOWaveform.SINE
        ch.lfo1.rate_hz = 4.0
        ch.lfo1.depth = 10.0
        ch.lfo1.target = ModTarget.PITCH_SEMITONES

        ch.trigger(127)
        audio = s.process_audio(1024)
        assert audio.shape == (1024, 2)
        assert np.max(np.abs(audio)) > 0.01, "Synthesizer should produce audio with LFO"

    def test_global_mod_offsets_cleared(self):
        """Global mod offsets should not persist between process_audio calls."""
        s = PythonicSynthesizer()
        ch = s.channels[0]
        ch.lfo1.enabled = True
        ch.lfo1.waveform = LFOWaveform.SQUARE
        ch.lfo1.depth = 3.0
        ch.lfo1.target = ModTarget.MASTER_VOLUME

        ch.trigger(127)
        s.process_audio(BLOCK_SIZE)
        # master_volume_db should not have been permanently altered
        assert abs(s.master_volume_db - 0.0) < 0.001

    def test_morph_lfo_saves_and_restores(self):
        """LFO targeting MORPH should not permanently change morph position."""
        s = PythonicSynthesizer()
        # The morph manager is created by GUI normally; add a minimal one
        from pythonic.morph_manager import MorphManager
        s._morph_manager = MorphManager(s)
        s._morph_manager.set_position(0.3)

        ch = s.channels[0]
        ch.lfo1.enabled = True
        ch.lfo1.waveform = LFOWaveform.SQUARE
        ch.lfo1.rate_hz = 5.0
        ch.lfo1.depth = 0.5
        ch.lfo1.target = ModTarget.MORPH

        ch.trigger(127)
        s.process_audio(BLOCK_SIZE)
        s.process_audio(BLOCK_SIZE)

        # Morph position should be restored to 0.3 after process_audio
        assert abs(s._morph_manager._position - 0.3) < 0.01, \
            f"Morph position drifted to {s._morph_manager._position}"

    def test_last_mod_offsets_populated(self):
        """_last_mod_offsets should record current block's LFO offsets."""
        ch = DrumChannel(0, SAMPLE_RATE)
        ch.lfo1.enabled = True
        ch.lfo1.waveform = LFOWaveform.SQUARE
        ch.lfo1.rate_hz = 5.0
        ch.lfo1.depth = 10.0
        ch.lfo1.target = ModTarget.PITCH_SEMITONES

        ch.trigger(127)
        ch.process(BLOCK_SIZE)

        assert ModTarget.PITCH_SEMITONES in ch._last_mod_offsets
        assert ch._last_mod_offsets[ModTarget.PITCH_SEMITONES] != 0.0


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Test that old presets without LFO data still load correctly."""

    def test_old_preset_loads_without_lfo_keys(self):
        """set_parameters with no LFO keys should leave LFOs disabled."""
        ch = DrumChannel(0, SAMPLE_RATE)
        old_params = {
            'name': 'OldPatch',
            'osc_frequency': 100.0,
            'osc_waveform': 0,
            'level_db': -3.0,
            'pan': 25.0,
        }
        ch.set_parameters(old_params)
        assert ch.lfo1.enabled == False
        assert ch.lfo2.enabled == False
        assert ch.pump.enabled == False
        assert ch.lfo1.target == ModTarget.NONE

    def test_preset_roundtrip_via_synthesizer(self):
        """Full synthesizer preset save/load should preserve LFO state."""
        s = PythonicSynthesizer()
        s.channels[3].lfo1.enabled = True
        s.channels[3].lfo1.target = ModTarget.DISTORTION
        s.channels[3].lfo1.depth = 0.5
        s.channels[3].pump.enabled = True
        s.channels[3].pump.amount = 0.7

        data = s.get_preset_data()

        s2 = PythonicSynthesizer()
        s2.load_preset_data(data)

        assert s2.channels[3].lfo1.enabled == True
        assert s2.channels[3].lfo1.target == ModTarget.DISTORTION
        assert abs(s2.channels[3].lfo1.depth - 0.5) < 0.001
        assert s2.channels[3].pump.enabled == True
        assert abs(s2.channels[3].pump.amount - 0.7) < 0.001
        # Other channels should remain at defaults
        assert s2.channels[0].lfo1.enabled == False


# ---------------------------------------------------------------------------
# ModTarget enum completeness
# ---------------------------------------------------------------------------

class TestModTargetCatalog:
    """Verify catalog data is complete and consistent."""

    def test_all_targets_have_labels(self):
        """Every ModTarget should have an entry in MOD_TARGET_LABELS."""
        for target in ModTarget:
            assert target in MOD_TARGET_LABELS, f"{target} missing from MOD_TARGET_LABELS"

    def test_all_targets_have_depth_ranges(self):
        """Every ModTarget should have an entry in MOD_TARGET_DEPTH_RANGES."""
        for target in ModTarget:
            assert target in MOD_TARGET_DEPTH_RANGES, f"{target} missing from MOD_TARGET_DEPTH_RANGES"

    def test_groups_cover_all_non_none_targets(self):
        """All non-NONE targets should appear in exactly one group."""
        grouped = set()
        for targets in MOD_TARGET_GROUPS.values():
            for t in targets:
                assert t not in grouped, f"{t} appears in multiple groups"
                grouped.add(t)
        for target in ModTarget:
            if target != ModTarget.NONE:
                assert target in grouped, f"{target} not in any group"


# ---------------------------------------------------------------------------
# Per-target audibility tests
# ---------------------------------------------------------------------------

class TestAllModTargetsAudible:
    """For each non-global ModTarget, verify that modulation produces
    different audio than no modulation.  This catches routing dead-ends
    like the is_settled() bypass bug.
    """

    # Targets handled by synthesizer (global), not per-channel audio diff
    GLOBAL_TARGETS = {ModTarget.MASTER_VOLUME, ModTarget.MORPH}
    # Velocity sensitivity only affects next trigger, not mid-note audio
    VEL_TARGETS = {
        ModTarget.OSC_VEL_SENSITIVITY,
        ModTarget.NOISE_VEL_SENSITIVITY,
        ModTarget.MOD_VEL_SENSITIVITY,
    }

    def _make_channel(self, target, depth=None):
        """Create a channel with LFO1 modulating the given target."""
        ch = DrumChannel(0, SAMPLE_RATE)
        # Set up sensible base values so modulation has room to move
        ch.set_osc_frequency(200.0)
        ch.set_osc_noise_mix_immediate(0.5)
        ch.set_noise_filter_freq(2000.0)
        ch.set_noise_filter_q(2.0)
        ch.eq_gain_db = 6.0
        ch.eq_frequency = 1000.0
        ch.vintage_amount = 0.3
        ch.reverb_decay = 0.4
        ch.reverb_mix = 0.3
        ch.reverb_width = 0.5
        ch.delay_feedback = 0.3
        ch.delay_mix = 0.3
        ch.distortion = 0.3

        # Determine depth from range table if not given
        # depth is now a percentage (0-100) scaled by target's max range
        if depth is None:
            depth = 50  # 50% of max range — enough to be audible

        ch.lfo1.enabled = True
        ch.lfo1.waveform = LFOWaveform.SQUARE
        ch.lfo1.rate_hz = 5.0
        ch.lfo1.depth = depth
        ch.lfo1.target = target
        ch.lfo1.polarity = LFOPolarity.BIPOLAR
        ch.lfo1.retrigger = LFORetrigger.RETRIGGER
        return ch

    def _make_ref_channel(self):
        """Create a channel with NO modulation (reference)."""
        ch = DrumChannel(0, SAMPLE_RATE)
        ch.set_osc_frequency(200.0)
        ch.set_osc_noise_mix_immediate(0.5)
        ch.set_noise_filter_freq(2000.0)
        ch.set_noise_filter_q(2.0)
        ch.eq_gain_db = 6.0
        ch.eq_frequency = 1000.0
        ch.vintage_amount = 0.3
        ch.reverb_decay = 0.4
        ch.reverb_mix = 0.3
        ch.reverb_width = 0.5
        ch.delay_feedback = 0.3
        ch.delay_mix = 0.3
        ch.distortion = 0.3
        return ch

    def _audio_differs(self, target, depth=None, blocks=4):
        """Return True if modulated audio differs from unmodulated."""
        ch_mod = self._make_channel(target, depth=depth)
        ch_ref = self._make_ref_channel()

        ch_mod.trigger(100)
        ch_ref.trigger(100)

        for _ in range(blocks):
            out_mod = ch_mod.process(BLOCK_SIZE).copy()
            out_ref = ch_ref.process(BLOCK_SIZE).copy()
            if not np.allclose(out_mod, out_ref, atol=1e-7):
                return True
        return False

    def test_osc_frequency(self):
        assert self._audio_differs(ModTarget.OSC_FREQUENCY)

    def test_pitch_semitones(self):
        assert self._audio_differs(ModTarget.PITCH_SEMITONES)

    def test_pitch_mod_amount(self):
        assert self._audio_differs(ModTarget.PITCH_MOD_AMOUNT)

    def test_pitch_mod_rate(self):
        assert self._audio_differs(ModTarget.PITCH_MOD_RATE)

    def test_osc_attack(self):
        assert self._audio_differs(ModTarget.OSC_ATTACK)

    def test_osc_decay(self):
        assert self._audio_differs(ModTarget.OSC_DECAY)

    def test_noise_filter_freq(self):
        assert self._audio_differs(ModTarget.NOISE_FILTER_FREQ)

    def test_noise_filter_q(self):
        assert self._audio_differs(ModTarget.NOISE_FILTER_Q, depth=5.0)

    def test_noise_attack(self):
        assert self._audio_differs(ModTarget.NOISE_ATTACK)

    def test_noise_decay(self):
        assert self._audio_differs(ModTarget.NOISE_DECAY)

    def test_osc_noise_mix(self):
        assert self._audio_differs(ModTarget.OSC_NOISE_MIX)

    def test_level_db(self):
        assert self._audio_differs(ModTarget.LEVEL_DB)

    def test_pan(self):
        assert self._audio_differs(ModTarget.PAN)

    def test_distortion(self):
        assert self._audio_differs(ModTarget.DISTORTION)

    def test_eq_frequency(self):
        assert self._audio_differs(ModTarget.EQ_FREQUENCY)

    def test_eq_gain_db(self):
        assert self._audio_differs(ModTarget.EQ_GAIN_DB)

    def test_vintage_amount(self):
        assert self._audio_differs(ModTarget.VINTAGE_AMOUNT)

    def test_reverb_decay(self):
        assert self._audio_differs(ModTarget.REVERB_DECAY)

    def test_reverb_mix(self):
        assert self._audio_differs(ModTarget.REVERB_MIX)

    def test_reverb_width(self):
        assert self._audio_differs(ModTarget.REVERB_WIDTH)

    def test_delay_feedback(self):
        assert self._audio_differs(ModTarget.DELAY_FEEDBACK)

    def test_delay_mix(self):
        assert self._audio_differs(ModTarget.DELAY_MIX)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def run_tests_standalone():
    """Run tests without pytest."""
    import traceback

    test_classes = [
        TestLFOWaveforms,
        TestLFORetrigger,
        TestLFOSync,
        TestPumpSource,
        TestLFOSerialization,
        TestDrumChannelModulation,
        TestSynthesizerLFOIntegration,
        TestBackwardCompatibility,
        TestModTargetCatalog,
        TestAllModTargetsAudible,
    ]

    total_passed = 0
    total_failed = 0
    failed_tests = []

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        instance = test_class()

        test_methods = sorted(m for m in dir(instance) if m.startswith('test_'))

        for method_name in test_methods:
            try:
                getattr(instance, method_name)()
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
            print(f"  - {cls}.{method}: {error}")
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
