"""
Tests for the engine parts of the preset menu actions: initialize and
randomize (the clipboard is the core's, see test_preset_io_core.py; drum WAV
export is the core's, see test_export_core.py).
"""

import numpy as np

from pythonic.drum_channel import DrumChannel
from pythonic.lfo import ModTarget

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
