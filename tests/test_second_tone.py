"""
The second tone of the voice: a second oscillator at a ratio of the oscillator's pitch,
with its own wave, level and decay, under the oscillator envelope. With its level at 0 it
is off, a voice renders exactly as without it, and a drum patch that leaves it off loads,
sounds and saves as before.
"""

import numpy as np
import pytest

from pythonic.drum_channel import DrumChannel
from pythonic.lfo import ModTarget
from pythonic.preset_manager import (DrumPatchParser, DrumPatchWriter, apply_drum_patch_to_channel,
                                     convert_drum_patch_data)
from pythonic.voice import TONE2_DEFAULTS, DrumVoice, VoiceParams, tone2_on

SR = 44100


def render(params, n=SR // 2, seed=1, velocity=127, chunk=None):
    voice = DrumVoice(SR, seed=seed)
    voice.set_params(params)
    voice.trigger(velocity)
    if chunk is None:
        return voice.process(n).mean(axis=1)
    return np.concatenate([voice.process(min(chunk, n - i)) for i in range(0, n, chunk)]).mean(axis=1)


def drum(**kw):
    """An oscillator alone (no noise): a 200 Hz sine decaying over 600 ms."""
    base = dict(osc_freq=200.0, osc_decay_ms=600.0, osc_mix=1.0)
    base.update(kw)
    return VoiceParams(**base)


def tone2(**kw):
    """The second tone alone: the render with it less the render without it."""
    return render(drum(**kw)) - render(drum(**{k: v for k, v in kw.items() if not k.startswith('tone2_')}))


def peak_hz(x, start=0, n=8192):
    w = x[start:start + n] * np.hanning(n)
    spectrum = np.abs(np.fft.rfft(w, 8 * n))
    return np.argmax(spectrum) * SR / (8 * n)


def rms(x, t0, t1):
    return float(np.sqrt(np.mean(x[int(t0 * SR):int(t1 * SR)] ** 2)))


# ---------------------------------------------------------------- the sound
def test_off_renders_exactly_as_without_it():
    plain = render(drum())
    assert np.array_equal(render(drum(tone2_ratio=3.0, tone2_decay_ms=50.0, tone2_wave=2)), plain)
    assert not tone2_on(drum()) and tone2_on(drum(tone2_level=0.1))


def test_it_sounds_at_its_ratio_of_the_oscillator():
    for ratio in (1.6, 2.5):
        assert peak_hz(tone2(tone2_level=0.5, tone2_ratio=ratio, tone2_decay_ms=600.0)) \
            == pytest.approx(200.0 * ratio, abs=2.0)


def test_it_follows_the_oscillators_pitch_sweep():
    # the oscillator swept down 12 semitones over its first few hundred ms: the second tone
    # stays at its ratio of the oscillator all the way
    sweep = dict(mod_amount=12.0, mod_rate=300.0)
    osc = render(drum(**sweep))
    alone = render(drum(**sweep, tone2_level=0.5, tone2_ratio=1.5, tone2_decay_ms=600.0)) - osc
    early, late = (0, 2048), (int(0.25 * SR), 8192)
    assert peak_hz(osc, *early) > 1.3 * peak_hz(osc, *late)              # it sweeps
    for start, n in (early, late):
        assert peak_hz(alone, start, n) == pytest.approx(1.5 * peak_hz(osc, start, n), rel=0.03)


def test_it_decays_at_its_own_rate_but_never_longer_than_the_oscillator():
    short = tone2(tone2_level=0.5, tone2_ratio=2.5, tone2_decay_ms=100.0)
    # -60 dB over its decay: about 30 dB down from 0-50 ms to 50-100 ms
    drop = 20.0 * np.log10(rms(short, 0.0, 0.05) / rms(short, 0.05, 0.1))
    assert drop == pytest.approx(30.0, abs=6.0)
    longer = tone2(tone2_level=0.5, tone2_ratio=2.5, tone2_decay_ms=5000.0)
    capped = tone2(tone2_level=0.5, tone2_ratio=2.5, tone2_decay_ms=600.0)
    assert np.allclose(longer, capped)


def test_it_scales_with_velocity_like_the_oscillator():
    def ratio(**kw):
        loud = render(drum(osc_vel=1.0, **kw))
        soft = render(drum(osc_vel=1.0, **kw), velocity=40)
        return np.abs(soft).max() / np.abs(loud).max()
    assert ratio(tone2_level=0.8, tone2_ratio=2.5) == pytest.approx(ratio(), rel=0.05)


def test_it_ends_on_a_choke():
    voice = DrumVoice(SR, seed=1)
    voice.set_params(drum(tone2_level=1.0, tone2_ratio=2.5, tone2_decay_ms=600.0))
    voice.trigger(127)
    voice.process(SR // 20)
    voice.choke()
    voice.process(int(0.012 * SR))
    assert np.abs(voice.process(256)).max() < 1e-6


def test_the_render_does_not_depend_on_the_buffer_size():
    p = drum(tone2_level=0.7, tone2_ratio=1.7, tone2_wave=1, mod_amount=6.0, mod_rate=40.0)
    whole = render(p, n=SR // 4)
    for chunk in (64, 333):
        assert np.allclose(render(p, n=SR // 4, chunk=chunk), whole, atol=1e-7)


# ---------------------------------------------------------------- drum patches and presets
def test_a_patch_without_it_saves_as_before():
    text = DrumPatchWriter.format_patch(DrumPatchWriter.patch_from_channel(DrumChannel(0)))
    assert 'Tone2' not in text


def test_it_round_trips_through_a_drum_patch():
    channel = DrumChannel(0)
    for name, value in (('tone2_level', 0.6), ('tone2_ratio', 1.47), ('tone2_decay_ms', 80.0), ('tone2_wave', 2)):
        channel.set_tone2(name, value)
    text = DrumPatchWriter.format_patch(DrumPatchWriter.patch_from_channel(channel))
    assert 'Tone2Lvl: "60.0%"' in text and 'Tone2Wave: "Saw"' in text
    other = DrumChannel(1)
    apply_drum_patch_to_channel(other, convert_drum_patch_data(DrumPatchParser().parse_string(text)))
    assert other.tone2 == pytest.approx(channel.tone2)
    assert other._voice_params() == channel._voice_params()


def test_loading_a_plain_patch_turns_it_off():
    channel = DrumChannel(0)
    channel.set_tone2('tone2_level', 1.0)
    plain = DrumPatchWriter.format_patch(DrumPatchWriter.patch_from_channel(DrumChannel(1)))
    apply_drum_patch_to_channel(channel, convert_drum_patch_data(DrumPatchParser().parse_string(plain)))
    assert channel.tone2 == TONE2_DEFAULTS


def test_a_saved_sound_without_it_leaves_it_off():
    from pythonic.synthesizer import PythonicSynthesizer
    old = {k: v for k, v in DrumChannel(1).get_parameters().items() if not k.startswith('tone2_')}
    synth = PythonicSynthesizer(SR)
    synth.channels[0].set_tone2('tone2_level', 1.0)
    synth.load_preset_data({'channels': [old]})
    assert synth.channels[0].tone2 == TONE2_DEFAULTS


def test_the_channel_clamps_it_to_its_ranges():
    channel = DrumChannel(0)
    channel.set_tone2('tone2_ratio', 100.0)
    channel.set_tone2('tone2_level', -1.0)
    channel.set_tone2('tone2_wave', 7)
    assert channel.tone2['tone2_ratio'] == 8.0 and channel.tone2['tone2_level'] == 0.0
    assert channel.tone2['tone2_wave'] == 2


# ---------------------------------------------------------------- modulation
def test_an_lfo_moves_its_level_and_ratio():
    channel = DrumChannel(0)
    channel.set_tone2('tone2_level', 0.2)
    channel.set_tone2('tone2_ratio', 2.0)
    moved = channel._voice_params({ModTarget.TONE2_LEVEL: 0.3, ModTarget.TONE2_RATIO: -0.5})
    assert moved.tone2_level == pytest.approx(0.5) and moved.tone2_ratio == pytest.approx(1.5)
    assert channel.tone2['tone2_level'] == 0.2       # the base stays
