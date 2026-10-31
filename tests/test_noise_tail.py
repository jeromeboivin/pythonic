"""
The noise tail of the voice: a second envelope on the same filtered noise, decaying from the
hit alongside the noise envelope, as the room after a TR-808 clap's bursts. With its level at
0 it is off, a voice renders exactly as without it, and a drum patch that leaves it off
loads, sounds and saves as before.
"""

import numpy as np
import pytest

from pythonic.drum_channel import DrumChannel
from pythonic.lfo import ModTarget
from pythonic.preset_manager import (DrumPatchParser, DrumPatchWriter, apply_drum_patch_to_channel,
                                     convert_drum_patch_data)
from pythonic.voice import NENV_MOD, TAIL_DEFAULTS, DrumVoice, VoiceParams, tail_on

SR = 44100


def render(params, n=SR, seed=1, velocity=127, chunk=None):
    voice = DrumVoice(SR, seed=seed)
    voice.set_params(params)
    voice.trigger(velocity)
    if chunk is None:
        return voice.process(n).mean(axis=1)
    return np.concatenate([voice.process(min(chunk, n - i)) for i in range(0, n, chunk)]).mean(axis=1)


def clap(**kw):
    """Band-passed noise in bursts over 150 ms, no oscillator."""
    return VoiceParams(**{**dict(osc_mix=0.0, noise_filter=1, noise_freq=1000.0, noise_q=2.0, noise_env=NENV_MOD,
                                 noise_decay_ms=150.0, noise_vel=1.0), **kw})


def db(x, t0, t1):
    return 10.0 * np.log10(np.mean(x[int(t0 * SR):int(t1 * SR)] ** 2) + 1e-30)


def spectrum(x, t0, t1):
    w = x[int(t0 * SR):int(t1 * SR)]
    return np.abs(np.fft.rfft(w * np.hanning(len(w))))


# ---------------------------------------------------------------- the sound
def test_off_renders_exactly_as_without_it():
    assert np.array_equal(render(clap(tail_decay_ms=900.0)), render(clap()))
    assert not tail_on(clap()) and tail_on(clap(tail_level=0.1))


def test_it_keeps_the_noise_sounding_after_the_noise_envelope_at_its_own_decay():
    plain, tail = render(clap()), render(clap(tail_level=0.3, tail_decay_ms=800.0))
    assert db(plain, 0.3, 0.4) < -90.0                       # the noise envelope has ended
    # -60 dB over 800 ms: 15 dB from 0.3-0.4 s to 0.5-0.6 s
    assert db(tail, 0.3, 0.4) - db(tail, 0.5, 0.6) == pytest.approx(15.0, abs=1.5)


def test_it_goes_through_the_noise_filter():
    tail = spectrum(render(clap(tail_level=0.5, tail_decay_ms=800.0)), 0.3, 0.6)
    bins = np.fft.rfftfreq(int(0.3 * SR), 1.0 / SR)
    assert bins[np.argmax(tail)] == pytest.approx(1000.0, rel=0.3)
    assert tail[bins > 8000.0].mean() < tail.max() * 0.05


def test_it_scales_with_velocity_like_the_noise():
    p = clap(tail_level=0.3, tail_decay_ms=800.0)
    loud, soft = render(p), render(p, velocity=40)
    assert db(loud, 0.3, 0.4) - db(soft, 0.3, 0.4) == pytest.approx(db(loud, 0.0, 0.05) - db(soft, 0.0, 0.05), abs=1.0)


def test_it_ends_on_a_choke():
    voice = DrumVoice(SR, seed=1)
    voice.set_params(clap(tail_level=1.0, tail_decay_ms=5000.0))
    voice.trigger(127)
    voice.process(SR // 5)
    voice.choke()
    voice.process(int(0.012 * SR))
    assert np.abs(voice.process(256)).max() < 1e-6
    assert not voice.is_active or np.abs(voice.process(256)).max() == 0.0


def test_the_render_does_not_depend_on_the_buffer_size():
    p = clap(tail_level=0.4, tail_decay_ms=700.0)
    whole = render(p, n=SR // 2)
    for chunk in (64, 333):
        assert np.allclose(render(p, n=SR // 2, chunk=chunk), whole, atol=1e-7)


# ---------------------------------------------------------------- drum patches and presets
def test_a_patch_without_it_saves_as_before():
    text = DrumPatchWriter.format_patch(DrumPatchWriter.patch_from_channel(DrumChannel(0)))
    assert 'NTail' not in text


def test_it_round_trips_through_a_drum_patch():
    channel = DrumChannel(0)
    channel.set_tail('tail_level', 0.25)
    channel.set_tail('tail_decay_ms', 900.0)
    text = DrumPatchWriter.format_patch(DrumPatchWriter.patch_from_channel(channel))
    assert 'NTailLvl: "25.0%"' in text
    other = DrumChannel(1)
    apply_drum_patch_to_channel(other, convert_drum_patch_data(DrumPatchParser().parse_string(text)))
    assert other.tail == pytest.approx(channel.tail)
    assert other._voice_params() == channel._voice_params()


def test_loading_a_plain_patch_turns_it_off():
    channel = DrumChannel(0)
    channel.set_tail('tail_level', 1.0)
    plain = DrumPatchWriter.format_patch(DrumPatchWriter.patch_from_channel(DrumChannel(1)))
    apply_drum_patch_to_channel(channel, convert_drum_patch_data(DrumPatchParser().parse_string(plain)))
    assert channel.tail == TAIL_DEFAULTS


def test_a_saved_sound_without_it_leaves_it_off():
    from pythonic.synthesizer import PythonicSynthesizer
    old = {k: v for k, v in DrumChannel(1).get_parameters().items() if not k.startswith('tail_')}
    synth = PythonicSynthesizer(SR)
    synth.channels[0].set_tail('tail_level', 1.0)
    synth.load_preset_data({'channels': [old]})
    assert synth.channels[0].tail == TAIL_DEFAULTS


def test_the_channel_clamps_it_to_its_ranges():
    channel = DrumChannel(0)
    channel.set_tail('tail_level', 3.0)
    channel.set_tail('tail_decay_ms', 1.0)
    assert channel.tail == {'tail_level': 1.0, 'tail_decay_ms': 10.0}


# ---------------------------------------------------------------- modulation
def test_an_lfo_moves_its_level():
    channel = DrumChannel(0)
    channel.set_tail('tail_level', 0.2)
    assert channel._voice_params({ModTarget.TAIL_LEVEL: 0.3}).tail_level == pytest.approx(0.5)
    assert channel.tail['tail_level'] == 0.2       # the base stays
