"""
Tests for the preset menu actions: clipboard, initialize and randomize.
"""

import numpy as np

from pythonic.drum_channel import DrumChannel
from pythonic.lfo import ModTarget
from pythonic.pattern_manager import PatternManager
from pythonic.preset_manager import PresetManager
from pythonic.synthesizer import PythonicSynthesizer

SAMPLE_RATE = 44100


# ---------------------------------------------------------------------------
# DrumChannel.reset_to_defaults / randomize
# ---------------------------------------------------------------------------

def test_reset_to_defaults_restores_a_fresh_channel():
    ch = DrumChannel(3, SAMPLE_RATE)
    fresh = ch.get_parameters()
    ch.name = "Snare"
    ch.set_osc_frequency(1234.0)
    ch.set_noise_decay(50.0)
    ch.level_db = -12.0
    ch.lfo1.target = ModTarget.PITCH_SEMITONES

    ch.reset_to_defaults()

    assert ch.get_parameters() == fresh


def test_randomize_changes_the_patch_and_keeps_the_rest():
    ch = DrumChannel(0, SAMPLE_RATE)
    ch.name = "Kick"
    ch.level_db = -6.0
    ch.pan = 25.0
    ch.lfo1.target = ModTarget.PAN
    before = ch.get_parameters()

    ch.randomize(np.random.default_rng(1))
    after = ch.get_parameters()

    assert after['osc_frequency'] != before['osc_frequency']
    assert after['noise_decay'] != before['noise_decay']
    for key in ('name', 'level_db', 'pan', 'lfo1', 'lfo2', 'pump', 'reverb_mix'):
        assert after[key] == before[key], key
    assert 30.0 <= after['osc_frequency'] <= 5000.0
    assert 0.0 <= after['osc_noise_mix'] <= 1.0


def test_randomize_is_reproducible_with_a_seeded_rng():
    a = DrumChannel(0, SAMPLE_RATE)
    b = DrumChannel(0, SAMPLE_RATE)
    a.randomize(np.random.default_rng(7))
    b.randomize(np.random.default_rng(7))
    assert a.get_parameters() == b.get_parameters()


# ---------------------------------------------------------------------------
# PresetManager clipboard dicts
# ---------------------------------------------------------------------------

def test_preset_dict_round_trip_with_patterns():
    synth = PythonicSynthesizer(SAMPLE_RATE)
    pm = PatternManager(num_channels=8, pattern_length=16)
    presets = PresetManager(synth)
    pm.randomize_pattern(0)
    synth.channels[2].set_osc_frequency(321.0)
    clip = presets.export_preset_to_dict(pm)
    expected_patterns = pm.to_dict()

    for ch in synth.channels:
        ch.reset_to_defaults()
    pm.reset_all_patterns()
    presets.import_preset_from_dict(clip, pm)

    assert abs(synth.channels[2].oscillator.frequency - 321.0) < 1e-6
    assert pm.to_dict() == expected_patterns


def test_preset_dict_is_detached_from_live_state():
    synth = PythonicSynthesizer(SAMPLE_RATE)
    presets = PresetManager(synth)
    clip = presets.export_preset_to_dict()
    name = synth.channels[0].name

    presets.import_preset_from_dict(clip)
    synth.channels[0].name = "Changed"
    clip['channels'][1]['name'] = "Edited later"

    assert clip['channels'][0]['name'] == name
    assert synth.channels[1].name != "Edited later"
