"""
Tests for the engine parts of the preset menu actions: initialize,
randomize and drum WAV export (the clipboard is the core's, see
test_preset_io_core.py).
"""

import numpy as np
from scipy.io import wavfile

from pythonic.drum_channel import DrumChannel
from pythonic.lfo import ModTarget
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
# WAV export
# ---------------------------------------------------------------------------

def test_export_drum_to_wav_leaves_the_live_channel_alone(tmp_path):
    synth = PythonicSynthesizer(SAMPLE_RATE)
    presets = PresetManager(synth)
    channel = synth.channels[0]

    path = presets.export_drum_to_wav(channel, str(tmp_path / "kick.wav"),
                                      duration_ms=200.0, sample_rate=SAMPLE_RATE)

    rate, audio = wavfile.read(path)
    assert rate == SAMPLE_RATE
    assert audio.shape == (int(0.2 * SAMPLE_RATE), 2)
    assert np.max(np.abs(audio)) > 1000
    assert not channel.is_active


def test_export_all_drums_writes_one_file_per_channel(tmp_path):
    synth = PythonicSynthesizer(SAMPLE_RATE)
    files = PresetManager(synth).export_all_drums_to_wav(
        synth, str(tmp_path), duration_ms=50.0, sample_rate=SAMPLE_RATE)
    assert len(files) == 8
    assert all((tmp_path / f).exists() for f in files)
