"""
Tests for LFO/pump modulation targeting the sound morph.
"""

import numpy as np

from pythonic.lfo import LFOWaveform, ModTarget
from pythonic.morph_manager import MorphManager
from pythonic.synthesizer import PythonicSynthesizer

SAMPLE_RATE = 44100
BLOCK_SIZE = 512


def _morph_synth(lfo_target):
    synth = PythonicSynthesizer(SAMPLE_RATE)
    synth.channels[0].set_osc_frequency(60.0)
    morph = MorphManager(synth)
    synth.channels[0].set_osc_frequency(2000.0)
    morph.capture_endpoint_b()
    synth.set_morph_manager(morph)
    morph.set_position(0.0)

    lfo = synth.channels[0].lfo1
    lfo.enabled = True
    lfo.waveform = LFOWaveform.SQUARE
    lfo.rate_hz = 0.5
    lfo.depth = 1.0
    lfo.target = lfo_target
    return synth, morph


def _render(synth, blocks=8):
    synth.channels[0].trigger(127)
    return np.concatenate([synth.process_audio(BLOCK_SIZE).copy() for _ in range(blocks)])


def test_lfo_targeting_morph_moves_the_sound():
    still, _ = _morph_synth(ModTarget.NONE)
    moved, morph = _morph_synth(ModTarget.MORPH)

    assert not np.allclose(_render(still), _render(moved))
    assert moved.channels[0].lfo1.target == ModTarget.MORPH
    assert morph.position == 0.0


def test_morph_slider_keeps_live_lfo_settings():
    synth, morph = _morph_synth(ModTarget.MORPH)
    morph.set_position(0.7)
    assert synth.channels[0].lfo1.target == ModTarget.MORPH
    assert synth.channels[0].lfo1.depth == 1.0
