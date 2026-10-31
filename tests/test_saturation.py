"""
The saturation of the voice: a tanh curve off centre, after the envelopes (of the
oscillators, and of each metal band), so it adds even harmonics that follow the level as
the hit dies, as the analog machines' VCAs do. At 0 it is off, a voice renders exactly as
without it, and a drum patch that leaves it off loads, sounds and saves as before.
"""

import numpy as np
import pytest

from pythonic.drum_channel import DrumChannel
from pythonic.lfo import ModTarget
from pythonic.preset_manager import (DrumPatchParser, DrumPatchWriter, apply_drum_patch_to_channel,
                                     convert_drum_patch_data)
from pythonic.voice import DrumVoice, VoiceParams

SR = 44100


def render(params, n=SR, seed=1, velocity=127, chunk=None):
    voice = DrumVoice(SR, seed=seed)
    voice.set_params(params)
    voice.trigger(velocity)
    if chunk is None:
        return voice.process(n).mean(axis=1)
    return np.concatenate([voice.process(min(chunk, n - i)) for i in range(0, n, chunk)]).mean(axis=1)


def kick(**kw):
    """A 55 Hz sine decaying over 1.5 s, no noise."""
    return VoiceParams(**{**dict(osc_freq=55.0, osc_decay_ms=1500.0, osc_mix=1.0, osc_vel=1.0), **kw})


def cowbell(**kw):
    """Two square notes (540 and 800 Hz) through one band-pass, no oscillator or noise heard."""
    notes = {f'metal_freq{i}': (540.0, 800.0)[i % 2] for i in range(1, 7)}
    return VoiceParams(**{**dict(osc_mix=1.0, osc_freq=10000.0, osc_decay_ms=10.0, **notes,
                                 metal_band1_freq=800.0, metal_band1_q=3.0, metal_band1_level=1.0,
                                 metal_band1_decay_ms=600.0, metal_hp_freq=200.0), **kw})


def level_db(x, hz, t0=0.02, n=8192):
    """dB of the strongest bin within 20 Hz of hz, over n samples from t0 s."""
    w = x[int(t0 * SR):int(t0 * SR) + n] * np.hanning(n)
    spectrum = np.abs(np.fft.rfft(w, 8 * n))
    bins = np.fft.rfftfreq(8 * n, 1.0 / SR)
    near = np.abs(bins - hz) < 20.0
    return 20.0 * np.log10(spectrum[near].max() + 1e-12)


def second_harmonic(x, t0, hz=55.0):
    return level_db(x, 2 * hz, t0) - level_db(x, hz, t0)


# ---------------------------------------------------------------- the sound
def test_off_renders_exactly_as_without_it():
    for p in (kick(distortion=0.3), cowbell(metal_drive=0.5)):
        assert np.array_equal(render(VoiceParams(**{**p.__dict__, 'saturation': 0.0})), render(p))


def test_it_adds_a_second_harmonic_that_fades_with_the_level():
    plain, sat = render(kick()), render(kick(saturation=0.25))
    assert second_harmonic(plain, 0.02) < -55.0
    early, late = second_harmonic(sat, 0.02), second_harmonic(sat, 0.5)
    assert -30.0 < early < -15.0                    # about where a TR-808 kick has it
    drop = level_db(sat, 55.0, 0.02) - level_db(sat, 55.0, 0.5)
    assert early - late == pytest.approx(drop, abs=3.0)     # down as much as the level is


def test_it_is_asymmetric_unlike_the_distortion():
    # where both add the same third harmonic, the distortion (symmetric in its soft range)
    # adds next to no second one
    dist, sat = render(kick(distortion=0.12)), render(kick(saturation=0.25))
    third = lambda x: level_db(x, 165.0) - level_db(x, 55.0)
    assert third(sat) == pytest.approx(third(dist), abs=3.0)
    assert second_harmonic(sat, 0.02) > second_harmonic(dist, 0.02) + 15.0


def test_a_softer_hit_saturates_less():
    loud, soft = render(kick(saturation=0.25)), render(kick(saturation=0.25), velocity=50)
    assert second_harmonic(soft, 0.02) < second_harmonic(loud, 0.02) - 6.0


def test_on_a_two_note_metal_patch_it_adds_their_products_without_a_loud_difference_tone():
    plain, sat = render(cowbell()), render(cowbell(saturation=0.2))
    low = level_db(sat, 540.0)
    assert level_db(sat, 540.0) == pytest.approx(level_db(plain, 540.0), abs=0.5)
    assert level_db(sat, 1340.0) - level_db(plain, 1340.0) > 40.0      # the notes' sum, new
    assert level_db(sat, 260.0) < low - 20.0                             # their difference stays low


def test_it_ends_on_a_choke():
    voice = DrumVoice(SR, seed=1)
    voice.set_params(kick(saturation=0.8))
    voice.trigger(127)
    voice.process(SR // 20)
    voice.choke()
    voice.process(int(0.012 * SR))
    assert np.abs(voice.process(256)).max() < 1e-6


def test_the_render_does_not_depend_on_the_buffer_size():
    for p in (kick(saturation=0.6, mod_amount=12.0, mod_rate=40.0), cowbell(saturation=0.5)):
        whole = render(p, n=SR // 4)
        for chunk in (64, 333):
            assert np.allclose(render(p, n=SR // 4, chunk=chunk), whole, atol=1e-7)


# ---------------------------------------------------------------- drum patches and presets
def test_a_patch_without_it_saves_as_before():
    text = DrumPatchWriter.format_patch(DrumPatchWriter.patch_from_channel(DrumChannel(0)))
    assert 'SatAmt' not in text


def test_it_round_trips_through_a_drum_patch():
    channel = DrumChannel(0)
    channel.set_saturation(0.35)
    text = DrumPatchWriter.format_patch(DrumPatchWriter.patch_from_channel(channel))
    assert 'SatAmt: "35.0%"' in text
    other = DrumChannel(1)
    apply_drum_patch_to_channel(other, convert_drum_patch_data(DrumPatchParser().parse_string(text)))
    assert other.saturation == pytest.approx(0.35)
    assert other._voice_params() == channel._voice_params()


def test_loading_a_plain_patch_turns_it_off():
    channel = DrumChannel(0)
    channel.set_saturation(1.0)
    plain = DrumPatchWriter.format_patch(DrumPatchWriter.patch_from_channel(DrumChannel(1)))
    apply_drum_patch_to_channel(channel, convert_drum_patch_data(DrumPatchParser().parse_string(plain)))
    assert channel.saturation == 0.0


def test_a_saved_sound_without_it_leaves_it_off():
    from pythonic.synthesizer import PythonicSynthesizer
    old = {k: v for k, v in DrumChannel(1).get_parameters().items() if k != 'saturation'}
    synth = PythonicSynthesizer(SR)
    synth.channels[0].set_saturation(1.0)
    synth.load_preset_data({'channels': [old]})
    assert synth.channels[0].saturation == 0.0


def test_the_channel_clamps_it():
    channel = DrumChannel(0)
    channel.set_saturation(3.0)
    assert channel.saturation == 1.0
    channel.set_saturation(-1.0)
    assert channel.saturation == 0.0


# ---------------------------------------------------------------- modulation
def test_an_lfo_moves_it():
    channel = DrumChannel(0)
    channel.set_saturation(0.2)
    assert channel._voice_params({ModTarget.SATURATION: 0.3}).saturation == pytest.approx(0.5)
    assert channel.saturation == 0.2       # the base stays
