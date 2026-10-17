"""
The hold of the decaying pitch modulation: the start pitch holds for the hold time,
then decays as without it. A patch without a hold renders and saves as before.
"""

import numpy as np
import pytest
import scipy.signal

from pythonic.drum_channel import DrumChannel
from pythonic.preset_manager import (DrumPatchParser, DrumPatchWriter, apply_drum_patch_to_channel,
                                     convert_drum_patch_data)
from pythonic.voice import DrumVoice, VoiceParams

SR = 44100
KICK = dict(wave=0, osc_freq=50.0, osc_decay_ms=400.0, mod_rate=30.0, mod_amount=24.0, osc_mix=1.0)


def render(params, n=SR // 2, chunk=None):
    voice = DrumVoice(SR, seed=1)
    voice.set_params(params)
    voice.trigger(127)
    if chunk is None:
        return voice.process(n)
    return np.concatenate([voice.process(min(chunk, n - i)) for i in range(0, n, chunk)])


def half_cycles(x):
    """(end time in ms, pitch in Hz) of each half cycle."""
    s = scipy.signal.sosfiltfilt(scipy.signal.butter(4, 1500, 'low', fs=SR, output='sos'), x)
    z = np.where(np.diff(np.signbit(s)))[0]
    z = z[np.r_[True, np.diff(z) > 8]]
    return z[1:] / SR * 1000.0, SR / (2.0 * np.diff(z))


def test_the_start_pitch_holds_then_decays():
    t, f = half_cycles(render(VoiceParams(**KICK, mod_hold_ms=7.0))[:, 0])
    start = 50.0 * 2.0 ** (24.0 / 12.0)
    held = f[t < 7.5]
    assert len(held) >= 2 and held == pytest.approx(start, rel=0.03)
    assert f[(t > 12.0) & (t < 20.0)].max() < 0.5 * start


def test_without_a_hold_the_sweep_starts_at_once():
    t, f = half_cycles(render(VoiceParams(**KICK))[:, 0])
    assert f[t < 10.0].max() < 0.5 * 50.0 * 2.0 ** 2.0


def test_the_hold_does_not_depend_on_the_buffer_size():
    p = VoiceParams(**KICK, mod_hold_ms=7.3)
    assert np.allclose(render(p), render(p, chunk=37), atol=1e-8)
    assert np.allclose(render(p), render(p, chunk=256), atol=1e-8)


def test_the_hold_only_acts_on_the_decaying_mode():
    sine = dict(KICK, mod_mode=1, mod_rate=5.0, mod_amount=3.0)
    assert np.array_equal(render(VoiceParams(**sine)), render(VoiceParams(**sine, mod_hold_ms=20.0)))


def test_a_patch_without_a_hold_saves_as_before():
    text = DrumPatchWriter.format_patch(DrumPatchWriter.patch_from_channel(DrumChannel(0)))
    assert 'ModHold' not in text


def test_the_hold_round_trips_through_a_drum_patch():
    channel = DrumChannel(0)
    channel.set_pitch_mod_hold(6.5)
    text = DrumPatchWriter.format_patch(DrumPatchWriter.patch_from_channel(channel))
    assert 'ModHold: "6.5 ms"' in text
    other = DrumChannel(1)
    apply_drum_patch_to_channel(other, convert_drum_patch_data(DrumPatchParser().parse_string(text)))
    assert other.pitch_mod_hold == 6.5
    assert other._voice_params() == channel._voice_params()


def test_loading_a_patch_without_a_hold_clears_it():
    channel = DrumChannel(0)
    channel.set_pitch_mod_hold(10.0)
    plain = DrumPatchWriter.format_patch(DrumPatchWriter.patch_from_channel(DrumChannel(1)))
    apply_drum_patch_to_channel(channel, convert_drum_patch_data(DrumPatchParser().parse_string(plain)))
    assert channel.pitch_mod_hold == 0.0


def test_a_saved_sound_without_a_hold_clears_it():
    from pythonic.synthesizer import PythonicSynthesizer
    old = {k: v for k, v in DrumChannel(1).get_parameters().items() if k != 'pitch_mod_hold'}
    synth = PythonicSynthesizer(SR)
    synth.channels[0].set_pitch_mod_hold(10.0)
    synth.load_preset_data({'channels': [old]})
    assert synth.channels[0].pitch_mod_hold == 0.0


def test_the_hold_is_clamped():
    channel = DrumChannel(0)
    channel.set_pitch_mod_hold(500.0)
    assert channel.pitch_mod_hold == 100.0
    channel.set_pitch_mod_hold(-3.0)
    assert channel.pitch_mod_hold == 0.0
