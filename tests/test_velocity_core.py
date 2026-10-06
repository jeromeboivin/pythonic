"""
Per-step velocity and patterns of 1 to 64 steps in the app core (slice 4b):
the ``vel`` step and lane addresses, the pattern length up to 64, pattern ops
on velocities and long patterns, and what the audio callback plays. Every test
drives the core through its interface (get / set / describe / act / poll) and
runs the audio callback by hand on a fake stream.
"""

import time

import numpy as np
import pytest

from pythonic.app import AppCore
from tests.fake_audio import FakeAudioBackend


@pytest.fixture
def backend():
    return FakeAudioBackend()


@pytest.fixture
def make_core(prefs, backend):
    cores = []

    def make(**kwargs):
        kwargs.setdefault('preferences', prefs)
        kwargs.setdefault('audio_backend', backend)
        kwargs.setdefault('stall_timeout', None)
        core = AppCore(**kwargs)
        cores.append(core)
        return core

    yield make
    for core in cores:
        core.close()


def started(core):
    event = core.wait(core.start())
    assert event['status'] == 'done', event
    return core


def run(core, verb, **args):
    event = core.wait(core.act(verb, **args))
    assert event['status'] == 'done', event
    return event.get('result')


def run_live(core, stream, verb, **args):
    action_id = core.act(verb, **args)
    end = time.monotonic() + 5.0
    while time.monotonic() < end:
        stream.pull()
        try:
            event = core.wait(action_id, timeout=0.005)
        except TimeoutError:
            continue
        assert event['status'] == 'done', event
        return event.get('result')
    raise AssertionError(f'{verb} did not finish')


def lane(core, pattern, channel, field):
    return core.get(f'pattern.{pattern}.ch{channel}.{field}')


def record_hits(core):
    """Record what the sequencer plays (step, channel, velocity) from now on."""
    sequencer = core.audio.sequencer
    pm = core.pattern_manager
    hits = []
    advance = sequencer.advance

    def recording(n):
        events = advance(n)
        hits.extend((pm.play_position + 1, ch, vel) for _, ch, vel in events)
        return events
    sequencer.advance = recording
    return hits


# ---------------------------------------------------------------------------
# the vel addresses
# ---------------------------------------------------------------------------

def test_vel_step_address_describes_1_to_127_default_64(make_core):
    core = make_core()
    vel = core.describe('pattern.B.ch2.step5.vel')
    assert (vel['kind'], vel['minimum'], vel['maximum'], vel['default']) == ('int', 1, 127, 64)
    assert not vel['readonly']
    assert core.describe('pattern.L.ch8.step64.vel')['kind'] == 'int'
    assert core.describe('pattern.A.ch1.vel')['kind'] == 'list'
    with pytest.raises(KeyError):
        core.describe('pattern.A.ch1.step65.vel')


def test_vel_set_get_clamp_and_lane(make_core):
    core = make_core()
    assert core.get('pattern.A.ch1.step1.vel') == 64
    assert lane(core, 'A', 1, 'vel') == [64] * 16
    core.set('pattern.A.ch1.step3.vel', 100)
    core.set('pattern.A.ch1.step4.vel', 0)      # clamped to 1
    core.set('pattern.A.ch1.step5.vel', 300)    # clamped to 127
    assert lane(core, 'A', 1, 'vel')[2:5] == [100, 1, 127]
    assert core.pattern_manager.patterns[0].channels[0].steps[2].velocity == 100
    core.set('pattern.A.ch2.vel', [10, 20])
    assert lane(core, 'A', 2, 'vel')[:3] == [10, 20, 64]


def test_velocity_sits_beside_accent_and_survives_the_trigger(make_core):
    core = make_core()
    core.set('pattern.A.ch1.step1.trig', True)
    core.set('pattern.A.ch1.step1.vel', 90)
    core.set('pattern.A.ch1.step1.acc', True)
    assert core.get('pattern.A.ch1.step1.vel') == 90  # accent does not touch it
    core.set('pattern.A.ch1.step1.trig', False)
    assert core.get('pattern.A.ch1.step1.acc') is False
    assert core.get('pattern.A.ch1.step1.vel') == 90  # kept, like probability


def test_vel_sets_are_queued_and_reported(make_core, backend):
    core = started(make_core())
    version = core.poll()['version']
    core.set('pattern.C.ch4.step40.vel', 77)
    core.set('pattern.C.length', 64)
    backend.stream.pull()
    changes = core.poll(version)['changes']
    assert changes['pattern.C.length'] == 64
    assert changes['pattern.C.ch4.vel'][39] == 64  # set before the length grew: ignored
    core.set('pattern.C.ch4.step40.vel', 77)
    backend.stream.pull()
    changes = core.poll(version)['changes']
    assert changes['pattern.C.ch4.step40.vel'] == 77
    assert changes['pattern.C.ch4.vel'][39] == 77


# ---------------------------------------------------------------------------
# length 1..64
# ---------------------------------------------------------------------------

def test_length_runs_from_1_to_64(make_core):
    core = make_core()
    core.set('pattern.A.length', 64)
    assert core.get('pattern.A.length') == 64
    assert all(len(lane(core, 'A', 1, f)) == 64 for f in ('trig', 'acc', 'vel', 'fill', 'prob', 'sub'))
    core.set('pattern.A.ch1.step64.trig', True)
    core.set('pattern.A.ch1.step64.vel', 5)
    assert core.get('pattern.A.ch1.step64.trig') is True
    assert core.get('pattern.A.ch1.step64.vel') == 5
    core.set('pattern.A.length', 100)
    assert core.get('pattern.A.length') == 64
    core.set('pattern.A.length', 0)
    assert core.get('pattern.A.length') == 1
    assert lane(core, 'A', 1, 'vel') == [64]
    core.set('pattern.A.ch1.vel', list(range(1, 65)))  # a list as long as 64 steps is fine
    with pytest.raises(ValueError):
        core.set('pattern.A.ch1.vel', [64] * 65)


# ---------------------------------------------------------------------------
# pattern ops
# ---------------------------------------------------------------------------

def long_pattern(core, name='A'):
    core.set(f'pattern.{name}.length', 64)
    core.set(f'pattern.{name}.ch1.step1.trig', True)
    core.set(f'pattern.{name}.ch1.step1.vel', 11)
    core.set(f'pattern.{name}.ch1.step40.trig', True)
    core.set(f'pattern.{name}.ch1.step40.vel', 99)


def test_shift_and_reverse_move_velocities_across_64_steps(make_core):
    core = make_core()
    long_pattern(core)
    run(core, 'pattern.shift_left', pattern='A')
    vel, trig = lane(core, 'A', 1, 'vel'), lane(core, 'A', 1, 'trig')
    assert trig[63] and vel[63] == 11  # step 1 wrapped round to step 64
    assert trig[38] and vel[38] == 99
    run(core, 'pattern.shift_right', pattern='A')
    run(core, 'pattern.shift_right', pattern='A')
    assert lane(core, 'A', 1, 'vel')[1] == 11 and lane(core, 'A', 1, 'vel')[40] == 99
    run(core, 'pattern.reverse', pattern='A')
    vel = lane(core, 'A', 1, 'vel')
    assert vel[62] == 11 and vel[23] == 99


def test_copy_paste_exchange_and_cut_carry_velocity_and_length(make_core):
    core = make_core()
    long_pattern(core)
    run(core, 'pattern.copy', pattern='A')
    run(core, 'pattern.paste', pattern='B')
    assert core.get('pattern.B.length') == 64
    assert core.get('pattern.B.ch1.step40.vel') == 99

    run(core, 'pattern.cut', pattern='B')
    assert core.get('pattern.B.empty') is True
    assert core.get('pattern.B.ch1.step40.vel') == 64  # cleared steps are back to default
    assert run(core, 'pattern.exchange', pattern='C') == {'exchanged': True}
    assert core.get('pattern.C.length') == 64 and core.get('pattern.C.ch1.step1.vel') == 11


def test_clear_resets_velocities(make_core):
    core = make_core()
    long_pattern(core)
    run(core, 'pattern.clear', pattern='A')
    assert lane(core, 'A', 1, 'vel') == [64] * 64
    assert core.get('pattern.A.length') == 64


def test_randomize_and_alter_cover_long_patterns_and_keep_velocities(make_core):
    core = make_core()
    core.set('pattern.E.length', 64)
    core.set('pattern.E.ch1.vel', [30] * 64)
    np.random.seed(3)
    run(core, 'pattern.randomize', pattern='E')
    triggers = lane(core, 'E', 1, 'trig')
    assert any(triggers[48:])  # the last page is randomized too
    assert lane(core, 'E', 1, 'vel') == [30] * 64
    run(core, 'pattern.randomize_accents_fills', pattern='E')
    run(core, 'pattern.alter', pattern='E')
    assert len(lane(core, 'E', 1, 'trig')) == 64


def test_lane_clipboard_carries_velocities(make_core):
    core = make_core()
    long_pattern(core)
    run(core, 'pattern.copy_lane', pattern='A', channel=1)
    core.set('pattern.D.length', 64)
    assert run(core, 'pattern.paste_lane', pattern='D', channel=3) == {'pasted': True}
    assert core.get('pattern.D.ch3.step40.trig') is True
    assert core.get('pattern.D.ch3.step40.vel') == 99
    # Into a shorter pattern only its steps are pasted
    assert run(core, 'pattern.paste_lane', pattern='F', channel=1) == {'pasted': True}
    assert len(lane(core, 'F', 1, 'vel')) == 16 and core.get('pattern.F.ch1.step1.vel') == 11


def test_undo_snapshots_keep_velocity_and_length(make_core):
    core = make_core()
    long_pattern(core)
    snapshot = core.legacy_snapshot()
    run(core, 'pattern.clear', pattern='A')
    core.set('pattern.A.length', 16)
    run(core, 'legacy.restore_snapshot', snapshot=snapshot)
    assert core.get('pattern.A.length') == 64
    assert core.get('pattern.A.ch1.step40.vel') == 99


# ---------------------------------------------------------------------------
# playback
# ---------------------------------------------------------------------------

def test_playback_plays_step_velocities_and_all_64_steps(make_core, backend):
    core = started(make_core())
    stream = backend.stream
    core.set('global.tempo', 300)
    core.set('pattern.A.length', 64)
    for step, vel in ((1, 64), (17, 100), (33, 1), (64, 127)):
        core.set(f'pattern.A.ch2.step{step}.trig', True)
        core.set(f'pattern.A.ch2.step{step}.vel', vel)
    core.set('pattern.A.ch3.step17.trig', True)
    core.set('pattern.A.ch3.step17.acc', True)
    core.set('pattern.A.ch3.step17.vel', 3)
    hits = record_hits(core)
    run_live(core, stream, 'transport.play', pattern='A')

    positions = set()
    for _ in range(2000):
        stream.pull()
        positions.add(core.poll()['transport']['position'])
        if len(hits) >= 6:
            break
    assert hits[:6] == [(1, 1, 64), (17, 1, 100), (17, 2, 127), (33, 1, 1), (64, 1, 127),
                        (1, 1, 64)]
    assert max(positions) == 63
