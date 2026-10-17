"""
The metal section of the voice: six free-running square waves into two
band-passes, each with its own decay and level, then a high-pass. With both
band levels at 0 it is off, and a voice renders exactly as without it; drum
patches that leave it off load, sound and save as before.
"""

import dataclasses

import numpy as np
import pytest

from pythonic.drum_channel import DrumChannel
from pythonic.preset_manager import (DrumPatchParser, DrumPatchWriter, apply_drum_patch_to_channel,
                                     convert_drum_patch_data)
from pythonic.voice import METAL_DEFAULTS, METAL_PARAMS, DrumVoice, VoiceParams, metal_on

SR = 44100
# The TR-606 metal bank, as found by tools/fit_samples.py in recordings of the machine
BANK_606 = (286.8, 401.5, 480.7, 553.4, 679.8, 971.8)


def render(params, n=SR, seed=3, velocity=127, chunk=None):
    voice = DrumVoice(SR, seed=seed)
    voice.set_params(params)
    voice.trigger(velocity)
    if chunk is None:
        return voice.process(n)
    return np.concatenate([voice.process(min(chunk, n - i)) for i in range(0, n, chunk)])


def metal(**kw):
    """A metal patch: the oscillator silent, a 10 ms noise tick."""
    base = dict(osc_mix=0.0, osc_decay_ms=10.0, osc_freq=20.0, noise_decay_ms=10.0,
                metal_band1_level=1.0, metal_band2_level=0.0, metal_band1_freq=4000.0,
                metal_band1_decay_ms=2000.0, metal_band1_q=0.5, metal_hp_freq=20.0)
    base.update(kw)
    return VoiceParams(**base)


def spectrum(x):
    x = x[:, 0].astype(np.float64)
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.fft.rfftfreq(len(x), 1.0 / SR), spec


# ---------------------------------------------------------------- engine
def test_off_renders_exactly_as_without_the_section():
    plain = VoiceParams(osc_freq=180.0, osc_mix=0.6, noise_filter=1, noise_freq=3000.0,
                        noise_stereo=True, distortion=0.3, eq_gain_db=6.0)
    # every metal control but the levels moved: still off
    moved = dataclasses.replace(plain, metal_freq1=300.0, metal_tune=5.0, metal_band1_freq=900.0,
                                metal_band1_q=7.0, metal_hp_freq=4000.0, metal_vel=1.0)
    assert not metal_on(moved)
    assert np.array_equal(render(plain, chunk=128), render(moved, chunk=128))


def test_the_section_adds_the_odd_harmonics_of_its_squares():
    one = {f'metal_freq{i}': 0.0 for i in range(2, 7)}  # 0 Hz silences an oscillator
    hz, spec = spectrum(render(metal(metal_freq1=500.0, **one), n=SR // 2))
    level = lambda f: spec[np.argmin(np.abs(hz - f))]
    # a near-square (49 % duty): the low even harmonics are weak
    assert level(1500.0) > 3.0 * level(1000.0) and level(1500.0) > 3.0 * level(2000.0)
    assert level(2500.0) > 2.5 * level(2000.0) and level(2500.0) > 2.5 * level(3000.0)


def test_tune_and_pitch_move_every_oscillator():
    one = {f'metal_freq{i}': 0.0 for i in range(2, 7)}
    peak = lambda p: spectrum(render(p, n=SR // 2))[0][np.argmax(spectrum(render(p, n=SR // 2))[1])]
    base = peak(metal(metal_freq1=1000.0, metal_band1_freq=1000.0, metal_band1_q=5.0, **one))
    up = peak(metal(metal_freq1=1000.0, metal_band1_freq=1000.0, metal_band1_q=5.0, metal_tune=12.0, **one))
    assert base == pytest.approx(1000.0, abs=3.0)
    assert up == pytest.approx(2000.0, abs=6.0)


def test_the_band_decay_is_60_db_at_its_time():
    y = render(metal(metal_band1_decay_ms=300.0), n=SR)[:, 0]
    rms = lambda a, b: np.sqrt(np.mean(y[int(a * SR):int(b * SR)] ** 2))
    drop = 20.0 * np.log10(rms(0.29, 0.31) / rms(0.0, 0.02))
    assert drop == pytest.approx(-60.0, abs=4.0)


def test_the_decay_curve_blends_exponential_and_linear():
    y = lambda c: render(metal(metal_band1_decay_ms=500.0, metal_curve=c), n=SR)[:, 0]
    rms = lambda a, t: np.sqrt(np.mean(a[int(t * SR):int(t * SR) + 441] ** 2))
    exp, lin = y(0.0), y(1.0)
    # linear: half the level at half the decay time, silent at it; exponential: -30 dB at half
    assert 20 * np.log10(rms(lin, 0.25) / rms(lin, 0.02)) == pytest.approx(-6.0, abs=2.0)
    assert 20 * np.log10(rms(exp, 0.25) / rms(exp, 0.02)) == pytest.approx(-29.0, abs=4.0)
    assert rms(lin, 0.51) < 1e-6
    half = y(0.5)
    assert rms(lin, 0.25) > rms(half, 0.25) > rms(exp, 0.25)


def test_drive_fills_the_lows_with_the_products_of_the_partials():
    import scipy.signal

    def lows(drive):
        y = render(metal(metal_band1_freq=7000.0, metal_band1_q=6.0, metal_drive=drive))[SR // 20:SR // 2, 0]
        hz, power = scipy.signal.welch(y.astype(np.float64), SR, nperseg=4096)
        low, high = (hz > 100) & (hz < 1500), (hz > 4000) & (hz < 10000)
        flatness = np.exp(np.mean(np.log(power[low]))) / np.mean(power[low])
        return flatness, 10 * np.log10(power[low].sum() / power[high].sum())

    (flat0, level0), (flat1, level1) = lows(0.0), lows(0.6)
    # clean: the squares' own low harmonics, a few lines; driven: the difference tones of
    # the partials (an asymmetric curve), a dense noisy body, louder
    assert flat0 < 0.05 < 0.5 < flat1
    assert level1 > level0 + 5.0


def test_drive_0_leaves_the_metal_clean():
    p = metal(metal_band2_level=0.4)
    assert np.array_equal(render(p), render(dataclasses.replace(p, metal_drive=0.0)))


def test_choke_ends_a_linear_decay_in_10_ms_too():
    voice = DrumVoice(SR, seed=1)
    voice.set_params(metal(metal_curve=0.7))
    voice.trigger(127)
    voice.process(SR // 20)
    voice.choke()
    voice.process(int(0.012 * SR))
    assert np.abs(voice.process(256)).max() < 1e-6


def test_the_voice_rings_on_with_the_metal_after_the_oscillator_and_noise_end():
    voice = DrumVoice(SR, seed=1)
    voice.set_params(metal(metal_band1_decay_ms=1000.0))
    voice.trigger(127)
    voice.process(SR // 10)  # oscillator and noise (10 ms) are long over
    assert voice.is_active
    assert np.abs(voice.process(SR // 10)).max() > 1e-3
    voice.process(3 * SR)
    assert not voice.is_active


def test_the_render_does_not_depend_on_the_buffer_size():
    p = metal(metal_freq1=205.3, metal_band2_level=0.5, metal_band2_freq=7100.0, osc_mix=0.5,
              osc_freq=300.0, osc_decay_ms=200.0, noise_decay_ms=300.0)
    whole = render(p, n=SR // 2)
    # the oscillator and noise paths differ by rounding alone across buffer sizes
    np.testing.assert_allclose(whole, render(p, n=SR // 2, chunk=64), atol=1e-7)
    np.testing.assert_allclose(whole, render(p, n=SR // 2, chunk=1000), atol=1e-7)
    alone = metal(metal_band2_level=0.5, metal_band2_freq=7100.0)
    assert np.array_equal(render(alone, n=SR // 2), render(alone, n=SR // 2, chunk=64))


def test_the_oscillators_run_free_from_hit_to_hit():
    voice = DrumVoice(SR, seed=1)
    voice.set_params(metal())
    voice.trigger(127)
    first = voice.process(2048)
    voice.process(SR)
    voice.trigger(127)
    second = voice.process(2048)
    assert not np.allclose(first, second)  # new phases at every hit, same level
    assert np.abs(second).max() == pytest.approx(np.abs(first).max(), rel=0.5)


def test_choke_ends_the_metal_in_10_ms():
    voice = DrumVoice(SR, seed=1)
    voice.set_params(metal())
    voice.trigger(127)
    voice.process(SR // 20)
    voice.choke()
    voice.process(int(0.012 * SR))
    assert np.abs(voice.process(256)).max() < 1e-6


def test_velocity_scales_the_metal_by_its_sensitivity():
    tail = lambda p, v: np.abs(render(p, n=SR // 4, velocity=v)[SR // 20:]).max()  # past the tick
    assert tail(metal(metal_vel=1.0), 40) < 0.1 * tail(metal(metal_vel=1.0), 127)
    assert tail(metal(metal_vel=0.0), 40) == pytest.approx(tail(metal(metal_vel=0.0), 127))


# ---------------------------------------------------------------- drum patch files
def test_a_patch_without_the_section_saves_as_before(tmp_path):
    channel = DrumChannel(0)
    channel.set_osc_frequency(123.0)
    text = DrumPatchWriter.format_patch(DrumPatchWriter.patch_from_channel(channel))
    assert text.splitlines()[0].endswith('DrumPatchV3: {') and 'Metal' not in text
    assert text.rstrip().splitlines()[-2].strip() == 'ModVel: "0.0%"'


def test_the_section_round_trips_through_a_drum_patch():
    channel = DrumChannel(0)
    for name, value in (('metal_band1_level', 0.7), ('metal_band2_level', 0.25), ('metal_tune', -3.0),
                        ('metal_band1_q', 2.5), ('metal_band2_q', 6.0), ('metal_vel', 0.5), ('metal_curve', 0.4), *zip(
                            [f'metal_freq{i}' for i in range(1, 7)], BANK_606)):
        channel.set_metal(name, value)
    text = DrumPatchWriter.format_patch(DrumPatchWriter.patch_from_channel(channel))
    assert 'MetalB1Lvl: "70.0%"' in text and 'MetalFreq6: "971.8 Hz"' in text
    other = DrumChannel(1)
    apply_drum_patch_to_channel(other, convert_drum_patch_data(DrumPatchParser().parse_string(text)))
    assert other.metal == pytest.approx(channel.metal)
    assert other._voice_params() == channel._voice_params()


def test_loading_a_plain_patch_turns_the_section_off():
    channel = DrumChannel(0)
    channel.set_metal('metal_band1_level', 1.0)
    plain = DrumPatchWriter.format_patch(DrumPatchWriter.patch_from_channel(DrumChannel(1)))
    apply_drum_patch_to_channel(channel, convert_drum_patch_data(DrumPatchParser().parse_string(plain)))
    assert channel.metal == METAL_DEFAULTS


def test_a_saved_sound_without_the_section_leaves_it_off():
    channel = DrumChannel(0)
    channel.set_metal('metal_band2_level', 1.0)
    old = {k: v for k, v in DrumChannel(1).get_parameters().items() if not k.startswith('metal_')}
    from pythonic.synthesizer import PythonicSynthesizer
    synth = PythonicSynthesizer(SR)
    synth.channels[0].set_metal('metal_band2_level', 1.0)
    synth.load_preset_data({'channels': [old]})
    assert synth.channels[0].metal == METAL_DEFAULTS


def test_set_metal_clamps_to_the_ranges():
    channel = DrumChannel(0)
    for name, (_default, _key, _unit, lo, hi) in METAL_PARAMS.items():
        channel.set_metal(name, hi * 10.0 + 1.0)
        assert channel.metal[name] == hi
        channel.set_metal(name, lo - 100.0)
        assert channel.metal[name] == lo
