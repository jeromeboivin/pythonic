"""
The partials section fitter (tools/fit_partials.py): a hat or cymbal is analysed into
noise bands (each with a decay and a click at the hit), its real tones as partials and a
cut, and the patch holds both the section and a classic version. Sounds made by the
section itself are found again.
"""

import numpy as np
import pytest

from pythonic.preset_manager import DrumPatchParser
from pythonic.voice import (PARTIAL_BANDS, DrumVoice, VoiceParams, format_bands, format_partials,
                            parse_bands, parse_partials)

FP = pytest.importorskip('tools.fit_partials')
F = pytest.importorskip('tools.fit_samples')
SR = F.SR
GAIN = 0.375    # the section's output at 0 dB, centre pan, velocity 127

TONES = np.array([[1830.0, 0.08, 0.40, 0.3], [2675.0, 0.06, 0.30, -1.1], [4120.0, 0.07, 0.35, 2.0],
                  [5310.0, 0.05, 0.25, 0.9], [6890.0, 0.04, 0.20, -2.4], [8455.0, 0.03, 0.18, 1.4]])


def bands(levels, decay=0.25, clicks=None):
    """Band rows: the given levels (index -> amplitude), the others silent; clicks: index ->
    (amplitude, decay s)."""
    rows = np.zeros((PARTIAL_BANDS, 4))
    rows[:, 1] = decay
    for k, a in levels.items():
        rows[k, 0] = a
    for k, (a, d) in (clicks or {}).items():
        rows[k, 2:] = (a, d)
    return rows


def render(fields, seconds=0.8, seed=5):
    voice = DrumVoice(SR, seed=seed)
    voice.set_params(VoiceParams(**{'partials_slope': FP.NOISE_SLOPE, **fields}))
    voice.trigger(127)
    return voice.process(int(seconds * SR))[:, 0].astype(np.float64)


def noise(levels, seconds=0.8, **kw):
    return render({'partial_bands': format_bands(bands(levels, **kw))}, seconds)


def found_bands(x):
    """The fitted engine band rows of x, without a gate, over GAIN."""
    rows = FP.engine_bands(FP.click_bands(x, len(x) / SR, 0.005), FP.NOISE_SLOPE)
    return rows / [GAIN, 1.0, GAIN, 1.0]


def test_the_tones_of_a_section_sound_are_found():
    x = render({'partials': format_partials(TONES),
                'partial_bands': format_bands(bands({k: 0.002 for k in range(16, 30)}))})
    rows = FP.real_tones(x)
    for f, a, d, _ in TONES:
        k = np.argmin(np.abs(rows[:, 0] - f))
        assert rows[k, 0] == pytest.approx(f, rel=0.001)
        assert rows[k, 2] == pytest.approx(d, rel=0.2)


def test_noise_has_no_tones():
    # its strongest spectral peaks are noise: as partials they would sound like a frozen hiss
    x = noise({k: 0.01 for k in range(18, 30)})
    assert len(FP.real_tones(x)) == 0


def test_a_short_rim_gets_its_two_tones():
    rng = np.random.default_rng(6)
    t = np.arange(int(0.025 * SR)) / SR
    rim = (np.sin(2 * np.pi * 460.0 * t) * np.exp(-t / 0.005) + 0.8 * np.sin(2 * np.pi * 1750.0 * t) * np.exp(-t / 0.004)
           + 0.05 * rng.standard_normal(len(t)) * np.exp(-t / 0.002))
    rows = FP.short_tones(rim)
    for f0 in (460.0, 1750.0):
        assert np.min(np.abs(rows[:, 0] - f0)) < 25.0
    assert 'tones' in FP.fit(rim, short=True)[1]


def test_a_weak_line_in_the_noise_is_left_out():
    x = render({'partials': format_partials(TONES[:1] * [1, 0.004, 1, 1]),
                'partial_bands': format_bands(bands({k: 0.01 for k in range(18, 30)}))})
    assert len(FP.real_tones(x)) == 0


def test_the_phases_are_random_but_the_same_for_the_same_sample():
    x = render({'partials': format_partials(TONES), 'partial_bands': format_bands(bands({20: 0.001}))})
    a, b = FP.real_tones(x), FP.real_tones(x)
    assert np.array_equal(a, b)
    assert np.std(a[:, 3]) > 1.0


def test_flat_noise_comes_back_flat_in_every_band():
    # no band may take its neighbours' power (a row of peaks instead of a noise), and steady
    # noise has no click
    x = noise({k: 0.01 for k in range(8, 30)}, decay=2.0, seconds=1.0)
    found = found_bands(x)
    level = 20 * np.log10(found[10:28, 0] / 0.01)
    assert np.all(np.abs(level) < 3.0)
    assert np.all(found[10:28, 1] > 0.5)
    assert np.all(found[10:28, 2] < 0.1 * found[10:28, 0])


def test_a_click_at_the_hit_stays_a_click():
    clicks = {k: (0.2, 0.0015) for k in range(18, 28)}
    x = noise({k: 0.01 for k in range(8, 30)}, decay=0.3, clicks=clicks)
    found = found_bands(x)
    hit = found[18:28, 2] > 0.02
    assert hit.sum() >= 8
    assert np.all(found[18:28, 3][hit] < 0.005)
    # the body under it keeps its level
    assert 20 * np.log10(np.median(found[18:28, 0]) / 0.01) == pytest.approx(0.0, abs=3.0)


def test_a_sound_that_holds_then_drops_gets_a_cut_there():
    x = render({'partial_bands': format_bands(bands({k: 0.01 for k in range(20, 30)}, decay=1.0)),
                'partials_cut_ms': 60.0, 'partials_fade_ms': 20.0}, seconds=0.2)
    cut, fade = FP.gate(x)
    assert cut == pytest.approx(0.060, abs=0.005)
    assert fade == pytest.approx(0.020, rel=0.5)
    # a plain decay has none
    assert FP.gate(noise({k: 0.01 for k in range(20, 30)}, decay=0.1)) is None


def test_without_a_gate_the_section_stops_where_the_sample_ends():
    x = noise({k: 0.01 for k in range(20, 30)}, decay=0.1, seconds=0.3)
    fields, how = FP.fit(x)
    assert fields['partials_cut_ms'] == pytest.approx(300.0, abs=1.0)
    assert fields['partials_slope'] == FP.NOISE_SLOPE
    assert how == '0 tones'


def test_the_fit_rings_as_long_as_the_sample():
    x = render({'partials': format_partials(TONES),
                'partial_bands': format_bands(bands({k: 0.004 for k in range(16, 30)}))}, seconds=2.0)
    fields, _ = FP.fit(x)
    y = render(fields, seconds=2.0)

    def time_to(s, db):
        e = np.convolve(s ** 2, np.ones(441) / 441, 'same')
        return np.nonzero(10 * np.log10(e / e.max() + 1e-20) > db)[0][-1] / SR
    assert time_to(y, -40.0) == pytest.approx(time_to(x, -40.0), rel=0.2)


def test_a_quiet_recording_gets_the_same_section():
    x = render({'partials': format_partials(TONES),
                'partial_bands': format_bands(bands({k: 0.004 for k in range(16, 30)}))})
    loud, _ = FP.fit(x)
    quiet, _ = FP.fit(x * 0.01)
    assert parse_partials(quiet['partials']) == pytest.approx(parse_partials(loud['partials']), rel=0.02, abs=1e-4)
    assert parse_bands(quiet['partial_bands']) == pytest.approx(parse_bands(loud['partial_bands']), rel=0.02, abs=1e-5)


def test_a_patch_with_the_section_writes_it_and_its_classic_version():
    p = VoiceParams(osc_freq=8000.0, partials=format_partials(TONES),
                    partial_bands=format_bands(bands({20: 0.01}, clicks={20: (0.05, 0.001)})),
                    partials_cut_ms=120.0, partials_slope=48.0)
    patch = F.patch_dict(p, 'Hat', True)
    parsed = DrumPatchParser().parse_string(F.DrumPatchWriter.format_patch(patch))
    assert parsed['OscFreq'] == 8000.0 and parsed['PartCut'] == 120.0 and parsed['PartSlope'] == 48.0
    rows = parse_bands(parsed['PartBands'])
    assert len(rows) == PARTIAL_BANDS and rows[20, 2] == pytest.approx(0.05, rel=0.01)
    assert 'Partials' not in F.patch_dict(VoiceParams(), 'Plain', False)
