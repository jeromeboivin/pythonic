"""
The step sequencer on per-step velocity and patterns of 1 to 64 steps: which
velocity each hit plays at (accent, velocity, fills, substeps) and where the
steps of a long pattern land, to the sample.
"""

import pytest

from pythonic.pattern_manager import DEFAULT_VELOCITY, PatternManager
from pythonic.sequencer import StepSequencer

SR = 44100


def manager(length=16, tempo=120, step_rate='1/16'):
    pm = PatternManager()
    pm.set_bpm(tempo)
    pm.set_step_rate(step_rate)
    pm.patterns[0].set_length(length)
    return pm


def render(pm, samples, block=512):
    """Run the sequencer for `samples` samples; events as (sample, channel, velocity)."""
    seq = StepSequencer(pm, SR, seed=1)
    pm.start_playback(0)
    seq.start(None, synth_clock=0)
    events, pos = [], 0
    while pos < samples:
        n = min(block, samples - pos)
        events += [(pos + off, ch, vel) for off, ch, vel in seq.advance(n)]
        pos += n
    return events


def step_samples(pm):
    """Length of one step in samples (no swing)."""
    ticks = {'1/8': 960, '1/8T': 640, '1/16': 480, '1/16T': 320, '1/32': 240}[pm.step_rate]
    return ticks * SR / (pm.bpm * 32.0)


def grid(pm):
    """How far a hit may sit from its exact time: the clock advances in whole
    ticks per 64-sample block (up to one tick late) and hits snap to the
    voices' 4-sample trigger grid."""
    return SR / (pm.bpm * 32.0) + 4


def trigger(pm, channel, step, **fields):
    s = pm.patterns[0].channels[channel].steps[step - 1]
    s.trigger = True
    for name, value in fields.items():
        setattr(s, name, value)


# ---------------------------------------------------------------------------
# velocity
# ---------------------------------------------------------------------------

def test_a_new_step_has_the_default_velocity_of_an_unaccented_hit():
    assert DEFAULT_VELOCITY == 64
    pm = manager()
    assert all(s.velocity == 64 for s in pm.patterns[0].channels[0].steps)
    trigger(pm, 0, 1)
    trigger(pm, 1, 1, accent=True)
    events = render(pm, 1000)
    assert sorted((ch, vel) for _, ch, vel in events) == [(0, 64), (1, 127)]


def test_an_unaccented_step_plays_at_its_velocity_an_accented_one_at_127():
    pm = manager()
    trigger(pm, 0, 1, velocity=100)
    trigger(pm, 1, 1, velocity=20, accent=True)
    trigger(pm, 2, 1, velocity=1)
    events = render(pm, 1000)
    assert sorted((ch, vel) for _, ch, vel in events) == [(0, 100), (1, 127), (2, 1)]


def test_fills_start_at_the_step_velocity_and_drop_as_before():
    pm = manager()
    pm.set_fill_rate(4)
    trigger(pm, 0, 1, velocity=100, fill=True)
    trigger(pm, 1, 1, fill=True)  # default velocity: 64, 48, 32, 16 as always
    trigger(pm, 2, 1, accent=True, velocity=10, fill=True)  # accent: 127 -> 64
    events = render(pm, int(step_samples(pm)) - 10)
    by_channel = {ch: [vel for _, c, vel in events if c == ch] for ch in range(3)}
    assert by_channel[0] == [100, 84, 68, 52]
    assert by_channel[1] == [64, 48, 32, 16]
    assert by_channel[2] == [127, 111, 95, 79]


def test_substeps_play_at_the_step_velocity():
    pm = manager()
    trigger(pm, 0, 1, velocity=90, substeps='o-o')
    events = render(pm, int(step_samples(pm)) - 10)
    assert [vel for _, _, vel in events] == [90, 90]


# ---------------------------------------------------------------------------
# long patterns
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('step_rate', ['1/8', '1/16', '1/16T', '1/32'])
def test_steps_of_a_64_step_pattern_land_where_a_16_step_pattern_loops(step_rate):
    """Pattern ticks run on across the loop, so step 17 of a 64-step pattern
    falls on the very sample where a 16-step pattern plays its step 1 again."""
    long_pm = manager(64, tempo=133, step_rate=step_rate)
    short_pm = manager(16, tempo=133, step_rate=step_rate)
    for pm in (long_pm, short_pm):
        trigger(pm, 0, 1)
    for step in (17, 33, 49):
        trigger(long_pm, 0, step)
    samples = int(step_samples(long_pm) * 64) + 100
    long_events = [e for e in render(long_pm, samples)]
    short_events = render(short_pm, samples)
    assert long_events == short_events
    assert len(long_events) == 5  # steps 1, 17, 33, 49, then step 1 of the second pass


def test_the_last_step_of_a_long_pattern_and_the_loop_are_on_the_grid():
    pm = manager(64, tempo=120)
    trigger(pm, 3, 64, velocity=99)
    trigger(pm, 3, 1, velocity=11)
    length = step_samples(pm)  # 5512.5 samples
    events = render(pm, int(length * 64.5))
    assert [(ch, vel) for _, ch, vel in events] == [(3, 11), (3, 99), (3, 11)]
    times = [t for t, _, _ in events]
    assert all(t % 4 == 0 for t in times)  # the voices' 4-sample trigger grid
    assert times[0] == 0
    assert abs(times[1] - 63 * length) <= grid(pm)
    assert abs(times[2] - 64 * length) <= grid(pm)
    assert pm.play_position == 0


@pytest.mark.parametrize('length', [1, 7, 33, 64])
def test_any_length_from_1_to_64_loops_after_its_last_step(length):
    pm = manager(length, tempo=150)
    trigger(pm, 0, 1)
    samples = int(step_samples(pm) * length * 3) - 10
    events = render(pm, samples)
    assert len(events) == 3
    gaps = [b[0] - a[0] for a, b in zip(events, events[1:])]
    for gap in gaps:
        assert abs(gap - step_samples(pm) * length) <= grid(pm)


def test_the_play_position_counts_up_to_64():
    pm = manager(64, tempo=300)
    seen = set()
    seq = StepSequencer(pm, SR)
    pm.start_playback(0)
    seq.start(None)
    for _ in range(int(step_samples(pm) * 64 / 256) + 4):
        seq.advance(256)
        seen.add(pm.play_position)
    assert seen == set(range(64))
