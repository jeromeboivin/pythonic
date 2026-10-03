"""
Drum voice engine and sample-accurate event rendering.

The voice runs on an absolute 4-sample block grid, so its output must not
depend on how the audio is split into process() calls (GUI buffer size,
event offsets, WAV export chunks).
"""

import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pythonic import voice_kernel as K
from pythonic.synthesizer import PythonicSynthesizer
from pythonic.voice import DrumVoice, VoiceParams

SR = 44100

PATCHES = {
    # decay pitch mod whose 64-sample latch turns on, then off again
    'decay-mod latch release': VoiceParams(
        wave=0, osc_freq=119.73, osc_decay_ms=349.8, mod_mode=0, mod_rate=210.27,
        mod_amount=-45.84, osc_vel=2.0, mod_vel=2.0, osc_mix=1.0),
    # resonant EQ boost that keeps ringing after both envelopes end
    'eq ring-out': VoiceParams(
        wave=1, osc_freq=180.0, osc_decay_ms=20.0, noise_decay_ms=15.0, osc_mix=0.7,
        eq_freq=400.0, eq_gain_db=30.0, distortion=0.2),
    'noise pitch mod': VoiceParams(
        wave=2, osc_freq=300.0, osc_decay_ms=800.0, mod_mode=2, mod_rate=2000.0,
        mod_amount=12.0, osc_mix=0.6, noise_decay_ms=400.0),
    'stereo noise': VoiceParams(
        osc_mix=0.2, noise_stereo=True, noise_filter=1, noise_freq=3000.0, noise_q=4.0,
        noise_env=2, noise_attack_ms=30.0, noise_decay_ms=300.0, pan=40.0),
    'sine pitch mod, attacks': VoiceParams(
        wave=1, osc_freq=220.0, osc_attack_ms=5.0, osc_decay_ms=600.0, mod_mode=1,
        mod_rate=8.0, mod_amount=7.0, noise_env=1, noise_attack_ms=20.0, noise_decay_ms=200.0),
}


def _render_voice(params, sizes, triggers=(0,)):
    """Render a voice through process() calls of the given sizes, triggering at call indices."""
    v = DrumVoice(SR, seed=3)
    v.set_params(params)
    v.idle(37)  # start off the 64-sample grid
    out = []
    for i, n in enumerate(sizes):
        if i in triggers:
            v.trigger(127)
        out.append(v.process(n))
    return np.concatenate(out), v


def _random_sizes(total, seed):
    r = random.Random(seed)
    sizes = []
    while sum(sizes) < total:
        sizes.append(r.randint(1, 700))
    return sizes


@pytest.mark.parametrize('name', sorted(PATCHES))
def test_voice_is_block_size_independent(name):
    params = PATCHES[name]
    whole, _ = _render_voice(params, [SR])
    sizes = _random_sizes(SR, seed=len(name))
    split, _ = _render_voice(params, sizes)
    np.testing.assert_allclose(split[:SR], whole, atol=1e-6)


def test_retrigger_is_block_size_independent():
    params = PATCHES['noise pitch mod']
    whole, _ = _render_voice(params, [5000, 5000, 5000], triggers=(0, 1, 2))
    split, _ = _render_voice(params, [1000, 4000, 3, 4997, 2500, 2500], triggers=(0, 2, 4))
    np.testing.assert_allclose(split, whole, atol=1e-6)


def test_eq_rings_out_after_the_envelopes():
    v = DrumVoice(SR, seed=1)
    v.set_params(PATCHES['eq ring-out'])
    v.trigger(127)
    v.process(SR // 4)
    # both envelopes are long over: only the EQ tail is left
    assert v._is[K.I_OSTAGE] == 4 and v._is[K.I_NSTAGE] == 4
    tail = v.process(64)
    assert np.max(np.abs(tail)) > 0.0
    assert v.is_active
    for _ in range(200):
        if not v.is_active:
            break
        v.process(1024)
    assert not v.is_active
    assert np.max(np.abs(v.process(256))) == 0.0


def _synth(patches):
    s = PythonicSynthesizer(SR)
    for i, params in enumerate(patches):
        s.channels[i].voice.set_params(params)
        s.channels[i]._voice_params = lambda mod=None, p=params: p
    return s


def test_events_only_split_the_triggered_channel():
    """process_audio_events matches rendering every channel split at every event."""
    patches = [PATCHES['decay-mod latch release'], PATCHES['stereo noise'], PATCHES['eq ring-out']]
    events = [(0, 0, 127), (700, 1, 100), (700, 2, 127), (1500, 0, 64), (3000, 2, 90)]

    a = _synth(patches)
    out_a = np.concatenate([a.process_audio_events(2048, [e for e in events if e[0] < 2048]),
                            a.process_audio_events(2048, [(o - 2048, c, v) for o, c, v in events if o >= 2048])])

    b = _synth(patches)
    out_b = np.empty((4096, 2), dtype=np.float32)
    pos = 0
    for offset, channel, velocity in events:
        if offset > pos:
            out_b[pos:offset] = b.process_audio(offset - pos)
            pos = offset
        b.trigger_drum(channel, velocity)
    out_b[pos:] = b.process_audio(4096 - pos)

    np.testing.assert_allclose(out_a, out_b, atol=1e-6)


def test_process_audio_events_returns_a_new_array():
    s = PythonicSynthesizer(SR)
    first = s.process_audio_events(256, [(0, 0, 127)])
    kept = first.copy()
    s.process_audio_events(256, [(10, 1, 127)])
    np.testing.assert_array_equal(first, kept)


def test_lower_choked_channel_wins_the_same_instant():
    s = PythonicSynthesizer(SR)
    for i in (1, 2):
        s.channels[i].choke_enabled = True
    s.process_audio_events(512, [(100, 2, 127), (100, 1, 127)])
    assert s.channels[1].is_active
    assert not s.channels[2].is_active
