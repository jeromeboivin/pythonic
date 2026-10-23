"""
The partials section of the voice, for hats and cymbals: decaying partials, each with
its frequency, level, decay and phase, plus white noise in 32 bands, each with its
level and decay, and an optional cut. While its tables are empty it is off and a voice
renders exactly as without it; while it is on, a hit starts it instead of the
oscillator, noise and metal, which keep a classic version of the sound.
"""

import dataclasses
import json

import numpy as np
import pytest

from pythonic.drum_channel import DrumChannel
from pythonic.preset_manager import (DrumPatchParser, DrumPatchWriter, apply_drum_patch_to_channel,
                                     convert_drum_patch_data)
from pythonic.synthesizer import PythonicSynthesizer
from pythonic.voice import (PARTIAL_BANDS, PARTIALS_DEFAULTS, PARTIALS_PARAMS, DrumVoice, VoiceParams,
                            format_bands, format_partials, parse_bands, parse_partials, partial_band_edges,
                            partials_on)

SR = 44100
GAIN = 0.5 * 0.75     # master scale times the centre pan law: a 0 dB section at velocity 127


def render(params, n=SR, seed=3, velocity=127, chunk=None):
    voice = DrumVoice(SR, seed=seed)
    voice.set_params(params)
    voice.trigger(velocity)
    if chunk is None:
        return voice.process(n)
    return np.concatenate([voice.process(min(chunk, n - i)) for i in range(0, n, chunk)])


def partial_rows(*rows):
    """format_partials text of (Hz, amplitude, decay s, phase rad) rows."""
    return format_partials(np.array(rows, dtype=float))


def band_rows(levels, decay=0.2):
    """format_bands text: the given band levels (index -> amplitude), the others silent."""
    rows = np.zeros((PARTIAL_BANDS, 2))
    rows[:, 1] = decay
    for k, a in levels.items():
        rows[k, 0] = a
    return format_bands(rows)


def spectrum(y):
    y = y.astype(np.float64)
    return np.fft.rfftfreq(len(y), 1.0 / SR), np.abs(np.fft.rfft(y * np.hanning(len(y))))


# ---------------------------------------------------------------- engine
def test_off_renders_exactly_as_without_the_section():
    plain = VoiceParams(osc_freq=180.0, osc_mix=0.6, noise_filter=1, noise_freq=3000.0,
                        noise_stereo=True, distortion=0.3, eq_gain_db=6.0)
    # every knob moved, no table: still off
    moved = dataclasses.replace(plain, partials_level_db=6.0, partials_tune=5.0, partials_decay=2.0,
                                partials_vel=1.0, partials_cut_ms=20.0, partials_fade_ms=40.0)
    assert not partials_on(moved)
    assert np.array_equal(render(plain, chunk=128), render(moved, chunk=128))


def test_a_partial_sounds_its_level_decay_and_phase():
    text = partial_rows((1000.0, 0.5, 0.2, 0.7))
    (f, a, d, ph), = parse_partials(text)       # as stored: the level to 0.1 dB
    y = render(VoiceParams(partials=text), n=SR // 2)[:, 0]
    t = np.arange(len(y)) / SR
    want = GAIN * a * np.exp(-t / d) * np.cos(2 * np.pi * f * t + ph)
    assert np.abs(y - want).max() < 1e-5


def test_a_hit_starts_the_section_instead_of_the_classic_sound():
    classic = VoiceParams(osc_freq=200.0, osc_mix=1.0, osc_decay_ms=2000.0)
    y = render(dataclasses.replace(classic, partials=partial_rows((3000.0, 0.5, 0.3, 0.0))), n=SR // 2)
    hz, spec = spectrum(y[:, 0])
    level = lambda f: spec[np.abs(hz - f) < 20.0].max()
    assert level(3000.0) > 1000.0 * level(200.0)


def test_tune_and_pitch_move_every_partial():
    peak = lambda p: (lambda hz, s: hz[np.argmax(s)])(*spectrum(render(p, n=SR // 2)[:, 0]))
    one = partial_rows((1000.0, 0.5, 1.0, 0.0))
    assert peak(VoiceParams(partials=one)) == pytest.approx(1000.0, abs=3.0)
    assert peak(VoiceParams(partials=one, partials_tune=12.0)) == pytest.approx(2000.0, abs=4.0)
    assert peak(VoiceParams(partials=one, pitch_ratio=0.5)) == pytest.approx(500.0, abs=3.0)


def test_a_noise_band_stays_within_its_edges():
    k = 20
    lo, hi = partial_band_edges()[k:k + 2]
    y = render(VoiceParams(partial_bands=band_rows({k: 0.2}, decay=5.0)), n=SR)[:, 0]
    hz, spec = spectrum(y)
    smooth = np.convolve(spec ** 2, np.ones(41) / 41, 'same')
    peak = smooth[(hz >= lo) & (hz < hi)].max()
    assert peak > 10.0 * smooth[(hz < lo / 1.5) | (hz > hi * 1.5)].max()
    # 24 dB an octave each side
    level = lambda f: 10 * np.log10(smooth[np.argmin(np.abs(hz - f))] / peak)
    assert level(lo / 2.0) < -18.0 and level(hi * 2.0) < -18.0
    assert level(lo / 4.0) < -40.0 and level(hi * 4.0) < -40.0


def test_at_48_db_an_octave_a_noise_band_falls_twice_as_steeply():
    k = 20
    lo, hi = partial_band_edges()[k:k + 2]
    levels = {}
    for slope in (24.0, 48.0):
        y = render(VoiceParams(partial_bands=band_rows({k: 0.2}, decay=5.0), partials_slope=slope), n=SR)[:, 0]
        hz, spec = spectrum(y)
        smooth = np.convolve(spec ** 2, np.ones(41) / 41, 'same')
        peak = smooth[(hz >= lo) & (hz < hi)].max()
        levels[slope] = [10 * np.log10(smooth[np.argmin(np.abs(hz - f))] / peak) for f in (lo / 2.0, hi * 2.0)]
    assert max(levels[48.0]) < -40.0
    assert all(b < a - 15.0 for a, b in zip(levels[24.0], levels[48.0]))


def test_a_band_click_is_a_short_burst_at_the_hit_over_its_body():
    k = 22
    body = np.zeros((PARTIAL_BANDS, 4))
    body[:, 1] = 0.5
    body[k, 0] = 0.05
    click = body.copy()
    click[k, 2:] = (0.5, 0.002)
    plain = render(VoiceParams(partial_bands=format_bands(body)), n=SR // 10)[:, 0]
    hit = render(VoiceParams(partial_bands=format_bands(click)), n=SR // 10)[:, 0]
    rms = lambda y, a, b: float(np.sqrt(np.mean(y[int(a * SR):int(b * SR)] ** 2)))
    assert rms(hit, 0.0, 0.002) > 4.0 * rms(plain, 0.0, 0.002)
    # the same noise under it: once the click has died away, only the body is left
    assert np.max(np.abs(hit - plain)[int(0.04 * SR):]) < 1e-3 * rms(plain, 0.04, 0.1)


def test_a_noise_band_has_unit_variance_noise_at_its_level():
    # all bands at one level: white noise through the whole filter bank, about flat
    y = render(VoiceParams(partial_bands=band_rows({k: 0.1 for k in range(PARTIAL_BANDS)}, decay=50.0)),
               n=SR)[:, 0] / GAIN
    total = float(np.mean(y ** 2))
    assert total == pytest.approx(0.1 ** 2, rel=0.5)


def test_decays_are_60_db_at_6_9_time_constants_and_follow_the_decay_knob():
    one = partial_rows((2000.0, 0.5, 0.1, 0.0))
    rms = lambda y, a, b: np.sqrt(np.mean(y[int(a * SR):int(b * SR)] ** 2))
    y = render(VoiceParams(partials=one))[:, 0]
    assert 20 * np.log10(rms(y, 0.2, 0.22) / rms(y, 0.0, 0.02)) == pytest.approx(-8.686 * 2, abs=0.3)
    y = render(VoiceParams(partials=one, partials_decay=2.0))[:, 0]
    assert 20 * np.log10(rms(y, 0.2, 0.22) / rms(y, 0.0, 0.02)) == pytest.approx(-8.686, abs=0.3)


def test_the_cut_fades_the_section_out():
    p = VoiceParams(partials=partial_rows((2000.0, 0.5, 5.0, 0.0)), partials_cut_ms=100.0,
                    partials_fade_ms=20.0)
    voice = DrumVoice(SR, seed=3)
    voice.set_params(p)
    voice.trigger(127)
    y = voice.process(SR // 2)[:, 0]
    rms = lambda a, b: np.sqrt(np.mean(y[int(a * SR):int(b * SR)] ** 2))
    assert rms(0.09, 0.1) > 0.9 * rms(0.0, 0.01)
    # exponential: -30 dB half way, -60 dB at the fade's end, then silence
    assert 20 * np.log10(rms(0.1095, 0.1105) / rms(0.099, 0.1)) == pytest.approx(-30.0, abs=1.5)
    assert np.abs(y[int(0.121 * SR):]).max() == 0.0
    assert not voice.is_active


def test_the_rise_fades_the_section_in():
    text = partial_rows((1000.0, 0.5, 5.0, 0.0))
    at_once = render(VoiceParams(partials=text), n=SR // 10)[:, 0]
    rising = render(VoiceParams(partials=text, partials_rise_ms=10.0), n=SR // 10)[:, 0]
    n = int(0.01 * SR)
    ramp = np.arange(n) / n
    assert np.abs(rising[:n] - at_once[:n] * ramp).max() < 1e-6
    assert np.array_equal(rising[n:], at_once[n:])
    assert np.array_equal(rising, render(VoiceParams(partials=text, partials_rise_ms=10.0), n=SR // 10,
                                         chunk=37)[:, 0])


def test_the_voice_stays_active_while_the_section_sounds():
    voice = DrumVoice(SR, seed=3)
    voice.set_params(VoiceParams(partials=partial_rows((2000.0, 0.5, 0.05, 0.0)),
                                 partial_bands=band_rows({10: 0.05}, decay=0.05)))
    voice.trigger(127)
    voice.process(SR // 10)
    assert voice.is_active
    voice.process(SR)
    assert not voice.is_active


def test_the_render_does_not_depend_on_the_buffer_size():
    p = VoiceParams(partials=partial_rows(*[(500.0 + 377.0 * i, 0.1, 0.05 + 0.03 * i, i) for i in range(20)]),
                    partial_bands=band_rows({k: 0.02 for k in range(5, 30)}, decay=0.1),
                    partials_cut_ms=300.0, partials_fade_ms=30.0)
    whole = render(p, n=SR // 2)
    assert np.array_equal(whole, render(p, n=SR // 2, chunk=37))
    assert np.array_equal(whole, render(p, n=SR // 2, chunk=4096))


def test_choke_ends_the_section_in_10_ms():
    voice = DrumVoice(SR, seed=3)
    voice.set_params(VoiceParams(partials=partial_rows((2000.0, 0.5, 5.0, 0.0))))
    voice.trigger(127)
    voice.process(SR // 10)
    voice.choke()
    y = voice.process(SR // 10)[:, 0]
    assert np.abs(y[int(0.011 * SR):]).max() == 0.0
    assert not voice.is_active


def test_velocity_scales_the_section_by_its_sensitivity():
    one = partial_rows((2000.0, 0.5, 1.0, 0.0))
    peak = lambda p, v: np.abs(render(p, n=SR // 10, velocity=v)).max()
    assert peak(VoiceParams(partials=one), 40) == pytest.approx(peak(VoiceParams(partials=one), 127))
    assert peak(VoiceParams(partials=one, partials_vel=1.0), 40) < 0.5 * peak(VoiceParams(partials=one), 127)


def test_level_and_pan_apply_to_the_section():
    one = partial_rows((2000.0, 0.5, 1.0, 0.0))
    base = render(VoiceParams(partials=one), n=SR // 10)
    louder = render(VoiceParams(partials=one, partials_level_db=6.0), n=SR // 10)
    assert np.abs(louder).max() == pytest.approx(np.abs(base).max() * 10 ** (6 / 20), rel=1e-4)
    left = render(VoiceParams(partials=one, pan=-100.0), n=SR // 10)
    assert np.abs(left[:, 1]).max() == 0.0 and np.abs(left[:, 0]).max() > 0.0


# ---------------------------------------------------------------- tables
def test_the_tables_round_trip_through_their_text():
    rows = np.array([[1234.567, 0.0123, 0.2345, -1.234], [8000.0, 0.5, 1.5, 3.0]])
    back = parse_partials(format_partials(rows))
    assert back[:, 0] == pytest.approx(rows[:, 0], abs=0.006)
    assert 20 * np.log10(back[:, 1]) == pytest.approx(20 * np.log10(rows[:, 1]), abs=0.06)
    assert back[:, 2] == pytest.approx(rows[:, 2], rel=1e-3)
    assert back[:, 3] == pytest.approx(rows[:, 3], abs=0.006)
    bands = np.array([[0.01, 0.1], [0.2, 2.0]])
    assert parse_bands(format_bands(bands)) == pytest.approx(np.c_[bands, np.zeros((2, 2))], rel=1e-2)
    # a band with a click carries its second level and decay; one without stays two numbers
    clicks = np.array([[0.01, 0.1, 0.05, 0.0012], [0.2, 2.0, 0.0, 0.001]])
    text = format_bands(clicks)
    assert [len(e.split()) for e in text.split(',')] == [4, 2]
    assert parse_bands(text) == pytest.approx(np.array([[0.01, 0.1, 0.05, 0.0012], [0.2, 2.0, 0.0, 0.0]]), rel=1e-2)


def test_malformed_entries_are_left_out():
    assert parse_partials('1000 -6 100 0, nonsense, 2000 -6 100').shape == (1, 4)


# ---------------------------------------------------------------- patches
SECTION = dict(partials=partial_rows((3000.0, 0.3, 0.4, 0.5), (7000.0, 0.2, 0.2, -1.0)),
               partial_bands=band_rows({18: 0.05, 25: 0.03}), partials_cut_ms=250.0, partials_vel=0.25)


def test_a_patch_without_the_section_saves_as_before():
    text = DrumPatchWriter.format_patch(DrumPatchWriter.patch_from_channel(DrumChannel(0, SR), 'Plain'))
    assert 'Part' not in text


def test_the_section_round_trips_through_a_drum_patch(tmp_path):
    ch = DrumChannel(0, SR)
    ch.set_parameters(SECTION)
    path = tmp_path / 'hat.mtdrum'
    path.write_text(DrumPatchWriter.format_patch(DrumPatchWriter.patch_from_channel(ch, 'Hat')))
    back = DrumChannel(0, SR)
    apply_drum_patch_to_channel(back, convert_drum_patch_data(DrumPatchParser().parse_file(str(path))))
    assert back.partials == ch.partials
    assert back._voice_params() == ch._voice_params()


def test_loading_a_plain_patch_turns_the_section_off():
    ch = DrumChannel(0, SR)
    ch.set_parameters(SECTION)
    apply_drum_patch_to_channel(ch, convert_drum_patch_data(DrumPatchParser().parse_string(
        'DrumPatchV3: {\n\tName: "Plain"\n\tOscFreq: "200 Hz"\n}')))
    assert ch.partials == PARTIALS_DEFAULTS


def test_a_saved_sound_without_the_section_leaves_it_off():
    synth = PythonicSynthesizer(SR, parallel_channel_processing=False)
    try:
        synth.channels[0].set_parameters(SECTION)
        old = {k: v for k, v in synth.channels[1].get_parameters().items() if k not in PARTIALS_DEFAULTS}
        synth.load_preset_data({'channels': [old]})
        assert synth.channels[0].partials == PARTIALS_DEFAULTS
    finally:
        synth.cleanup()


def test_the_section_survives_a_json_preset():
    ch = DrumChannel(0, SR)
    ch.set_parameters(SECTION)
    back = DrumChannel(0, SR)
    back.set_parameters(json.loads(json.dumps(ch.get_parameters())))
    assert back.partials == ch.partials


def test_set_partials_clamps_to_the_ranges():
    ch = DrumChannel(0, SR)
    for name, (_, _, _, lo, hi) in PARTIALS_PARAMS.items():
        ch.set_partials(name, hi * 10.0 + 10.0)
        assert ch.partials[name] == hi
        ch.set_partials(name, lo - 10.0)
        assert ch.partials[name] == lo
