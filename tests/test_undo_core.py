"""
The undo journal of the app core (slice 5): sets are journaled as
(address, old, new) entries grouped into steps (one set, a gesture, or a
burst closed after 400 ms idle), bulk changes and pattern ops record a
snapshot, and the ``undo`` / ``redo`` verbs replay them. Every test drives the
core through its interface (set / get / act / poll) and runs the audio
callback by hand on a fake stream where a stream is needed.
"""

import copy
import time

import pytest

from pythonic.app import AppCore
from tests.fake_audio import FakeAudioBackend


# ---------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------

class FakeClock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def backend():
    return FakeAudioBackend()


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def make_core(prefs, backend, clock):
    cores = []

    def make(**kwargs):
        kwargs.setdefault('preferences', prefs)
        kwargs.setdefault('audio_backend', None)  # no stream: sets apply inline
        kwargs.setdefault('stall_timeout', None)
        kwargs.setdefault('clock', clock)
        core = AppCore(**kwargs)
        cores.append(core)
        return core

    yield make
    for core in cores:
        core.close()


def run(core, verb, **args):
    event = core.wait(core.act(verb, **args))
    assert event['status'] == 'done', event
    return event.get('result')


def undo(core):
    return run(core, 'undo')


def redo(core):
    return run(core, 'redo')


def undo_all(core):
    steps = 0
    while undo(core)['done']:
        steps += 1
    return steps


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


# ---------------------------------------------------------------------------
# one set, undo, redo
# ---------------------------------------------------------------------------

def test_a_set_is_one_step_that_undo_and_redo_replay(make_core):
    core = make_core()
    assert core.get('undo.can_undo') is False and core.get('undo.can_redo') is False
    old = core.get('ch2.osc.decay')
    core.set('ch2.osc.decay', 900.0)
    assert core.get('undo.can_undo') is True

    version = core.poll()['version']
    assert undo(core) == {'done': True, 'label': 'ch2.osc.decay'}
    assert core.get('ch2.osc.decay') == old
    changes = core.poll(version)['changes']
    assert changes['ch2.osc.decay'] == old
    assert changes['undo.can_undo'] is False and changes['undo.can_redo'] is True

    version = core.poll()['version']
    assert redo(core)['done'] is True
    assert core.get('ch2.osc.decay') == 900.0
    changes = core.poll(version)['changes']
    assert changes['ch2.osc.decay'] == 900.0 and changes['undo.can_redo'] is False
    assert redo(core) == {'done': False}


def test_undo_with_nothing_to_undo_does_nothing(make_core):
    core = make_core()
    assert undo(core) == {'done': False}


def test_each_set_outside_a_gesture_is_its_own_step(make_core):
    core = make_core()
    core.set('global.tempo', 100)
    core.set('global.tempo', 110)
    core.set('global.swing', 0.3)
    undo(core)
    assert core.get('global.swing') == 0.0 and core.get('global.tempo') == 110
    undo(core)
    assert core.get('global.tempo') == 100
    undo(core)
    assert core.get('global.tempo') == 120


def test_a_new_edit_clears_redo(make_core):
    core = make_core()
    core.set('global.master', -6.0)
    undo(core)
    assert core.get('undo.can_redo') is True
    core.set('global.master', -3.0)
    assert core.get('undo.can_redo') is False
    assert redo(core) == {'done': False}
    assert core.get('global.master') == -3.0


def test_the_journal_keeps_the_last_50_steps(make_core):
    core = make_core()
    for bpm in range(60, 120):  # 60 steps
        core.set('global.tempo', bpm)
    assert undo_all(core) == 50
    assert core.get('global.tempo') == 69


def test_transport_selection_mutes_and_modes_are_not_undone(make_core):
    core = make_core()
    core.set('global.channel', 3)
    core.set('ch1.mute', True)
    core.set('global.edit_all', True)
    core.set('pattern.selected', 'C')
    run(core, 'transport.play')
    assert core.get('undo.can_undo') is False
    assert undo(core) == {'done': False}


def test_ordinary_edits_do_not_deep_copy(make_core, monkeypatch):
    core = make_core()

    def no_deepcopy(*_args, **_kw):
        raise AssertionError('deep copy on an ordinary edit')
    monkeypatch.setattr(copy, 'deepcopy', no_deepcopy)
    core.set('ch1.osc.freq', 300.0)
    core.set('pattern.A.ch1.step3.trig', True)
    core.begin_gesture()
    core.set('ch1.mix.level', -3.0)
    core.end_gesture()
    assert undo_all(core) == 3


def test_a_set_from_a_controller_clock_or_bend_can_skip_the_journal(make_core):
    core = make_core()
    core.set('global.tempo', 99, record=False)
    assert core.get('global.tempo') == 99
    assert core.get('undo.can_undo') is False


# ---------------------------------------------------------------------------
# gestures and bursts
# ---------------------------------------------------------------------------

def test_a_gesture_is_one_step(make_core):
    core = make_core()
    level, pan = core.get('ch1.mix.level'), core.get('ch1.mix.pan')
    core.begin_gesture()
    for value in (-10.0, -5.0, -1.0):
        core.set('ch1.mix.level', value)
    core.set('ch1.mix.pan', 40.0)
    assert core.get('undo.can_undo') is True  # an open gesture can be undone
    core.end_gesture()
    core.set('global.tempo', 130)

    undo(core)
    assert core.get('global.tempo') == 120 and core.get('ch1.mix.level') == -1.0
    undo(core)
    assert core.get('ch1.mix.level') == level and core.get('ch1.mix.pan') == pan
    redo(core)
    assert core.get('ch1.mix.level') == -1.0 and core.get('ch1.mix.pan') == 40.0


def test_a_gesture_that_ends_where_it_started_is_no_step(make_core):
    core = make_core()
    level = core.get('ch1.mix.level')
    core.begin_gesture()
    core.set('ch1.mix.level', -20.0)
    core.set('ch1.mix.level', level)
    core.end_gesture()
    assert core.get('undo.can_undo') is False


def test_undo_during_a_gesture_closes_it_first(make_core):
    core = make_core()
    core.begin_gesture()
    core.set('ch1.mix.level', -20.0)
    undo(core)
    assert core.get('ch1.mix.level') == 0.0
    core.set('ch1.mix.level', -10.0)  # the gesture goes on as a new step
    core.end_gesture()
    undo(core)
    assert core.get('ch1.mix.level') == 0.0
    assert core.get('undo.can_undo') is False


def test_a_burst_is_one_step_closed_after_400_ms_idle(make_core, clock):
    core = make_core()
    for t, value in ((0.0, 200.0), (0.3, 300.0), (0.6, 400.0)):
        clock.t = 100.0 + t
        core.set('ch1.osc.freq', value, burst=True)
    clock.t = 101.2  # 600 ms idle: the next change starts a new burst
    core.set('ch1.osc.freq', 500.0, burst=True)
    clock.t = 101.3
    core.set('ch1.osc.freq', 600.0, burst=True)

    undo(core)
    assert core.get('ch1.osc.freq') == 400.0
    undo(core)
    assert core.get('ch1.osc.freq') != 200.0  # back before the first burst
    assert core.get('undo.can_undo') is False


def test_bursts_are_per_address(make_core, clock):
    core = make_core()
    core.set('ch1.osc.freq', 300.0, burst=True)
    core.set('ch1.noise.freq', 3000.0, burst=True)
    clock.t += 0.2
    core.set('ch1.osc.freq', 310.0, burst=True)
    core.set('ch1.noise.freq', 3100.0, burst=True)
    clock.t += 0.5
    core.poll()  # closes the idle bursts
    assert undo_all(core) == 2


def test_a_burst_that_returns_to_its_start_is_no_step(make_core, clock):
    core = make_core()
    start = core.get('ch1.mix.pan')
    core.set('ch1.mix.pan', 30.0, burst=True)
    core.set('ch1.mix.pan', start, burst=True)
    clock.t += 1.0
    core.poll()
    assert core.get('undo.can_undo') is False


# ---------------------------------------------------------------------------
# Edit all, pattern steps
# ---------------------------------------------------------------------------

def test_an_edit_all_change_is_one_step_for_every_channel(make_core):
    core = make_core()
    olds = [core.get(f'ch{c}.fx.reverb_mix') for c in range(1, 9)]
    core.set('ch4.mute', True)
    core.set('ch1.fx.reverb_mix', 0.8, edit_all=True)
    assert [core.get(f'ch{c}.fx.reverb_mix') for c in range(1, 9)] == \
        [0.8, 0.8, 0.8, olds[3], 0.8, 0.8, 0.8, 0.8]

    undo(core)
    assert [core.get(f'ch{c}.fx.reverb_mix') for c in range(1, 9)] == olds
    core.set('ch4.mute', False)  # mutes are not undone, nor do they redo
    redo(core)
    assert core.get('ch4.fx.reverb_mix') == olds[3]  # it was muted when edited
    assert core.get('ch8.fx.reverb_mix') == 0.8


def test_step_and_lane_edits_are_undone(make_core):
    core = make_core()
    core.set('pattern.B.ch2.step5.trig', True)
    core.set('pattern.B.ch2.step5.acc', True)
    core.set('pattern.B.ch2.step5.vel', 99)
    core.set('pattern.B.ch2.prob', [50] * 16)

    version = core.poll()['version']
    undo(core)
    assert core.get('pattern.B.ch2.prob') == [100] * 16
    assert 'pattern.B.ch2.prob' in core.poll(version)['changes']
    undo(core)
    assert core.get('pattern.B.ch2.step5.vel') == 64
    undo(core)
    undo(core)
    assert core.get('pattern.B.ch2.trig')[4] is False


def test_turning_a_trigger_off_undoes_its_accent_and_fill_too(make_core):
    core = make_core()
    core.set('pattern.A.ch1.step2.trig', True)
    core.set('pattern.A.ch1.step2.acc', True)
    core.set('pattern.A.ch1.step2.fill', True)
    core.set('pattern.A.ch1.step2.trig', False)  # clears accent and fill
    assert core.get('pattern.A.ch1.step2.acc') is False

    undo(core)
    assert core.get('pattern.A.ch1.step2.trig') is True
    assert core.get('pattern.A.ch1.step2.acc') is True
    assert core.get('pattern.A.ch1.step2.fill') is True
    redo(core)
    assert core.get('pattern.A.ch1.step2.acc') is False


def test_shortening_a_pattern_is_undone_with_its_steps(make_core):
    core = make_core()
    core.set('pattern.A.length', 32)
    core.set('pattern.A.ch3.step30.trig', True)
    core.set('pattern.A.ch3.step30.vel', 20)
    core.set('pattern.A.length', 16)

    undo(core)
    assert core.get('pattern.A.length') == 32
    assert core.get('pattern.A.ch3.step30.trig') is True
    assert core.get('pattern.A.ch3.step30.vel') == 20
    redo(core)
    assert core.get('pattern.A.length') == 16


def test_a_paint_stroke_in_a_gesture_is_one_step(make_core):
    core = make_core()
    core.begin_gesture()
    for step in (1, 2, 3, 4):
        core.set(f'pattern.A.ch1.step{step}.trig', True)
    core.end_gesture()
    undo(core)
    assert core.get('pattern.A.ch1.trig')[:4] == [False] * 4
    assert core.get('undo.can_undo') is False


def test_a_pattern_op_is_one_step(make_core):
    core = make_core()
    core.set('pattern.C.ch1.trig', [True, False] * 8)
    core.set('pattern.C.ch5.vel', [30] * 16)
    run(core, 'pattern.clear', pattern='C')
    run(core, 'pattern.shift_right', pattern='A')
    assert core.get('pattern.C.ch1.trig') == [False] * 16

    undo(core)  # the shift of A
    version = core.poll()['version']
    undo(core)
    assert core.get('pattern.C.ch1.trig') == [True, False] * 8
    assert core.get('pattern.C.ch5.vel') == [30] * 16
    assert core.poll(version)['changes']['pattern.C.ch1.trig'] == [True, False] * 8
    redo(core)
    assert core.get('pattern.C.ch1.trig') == [False] * 16


def test_paste_lane_and_chains_are_undone(make_core):
    core = make_core()
    core.set('pattern.A.ch1.trig', [True] * 16)
    run(core, 'pattern.copy_lane', pattern='A', channel=1)
    run(core, 'pattern.paste_lane', pattern='B', channel=2)
    run(core, 'pattern.chain_next', pattern='A')
    assert core.get('pattern.A.chained') is True

    undo(core)
    assert core.get('pattern.A.chained') is False
    undo(core)
    assert core.get('pattern.B.ch2.trig') == [False] * 16


# ---------------------------------------------------------------------------
# bulk changes (snapshots)
# ---------------------------------------------------------------------------

def test_a_bulk_change_is_one_step_reported_by_poll(make_core):
    core = make_core()
    core.set('ch1.osc.decay', 700.0)
    core.set('pattern.A.ch1.step1.trig', True)
    version = core.poll()['version']

    with core.bulk_change('Initialize preset'):
        # code not moved into the core yet writes the engine directly
        for channel in core.synth.channels:
            channel.reset_to_defaults()
        core.pattern_manager.reset_all_patterns()
        core.pattern_manager.set_bpm(90)
        core.set('morph.position', 0.5)  # part of the bulk change, not its own step
    changes = core.poll(version)['changes']
    assert changes['ch1.osc.decay'] == core.get('ch1.osc.decay') != 700.0
    assert changes['pattern.A.ch1.trig'][0] is False
    assert changes['global.tempo'] == 90

    assert undo(core) == {'done': True, 'label': 'Initialize preset'}
    assert core.get('ch1.osc.decay') == 700.0
    assert core.get('pattern.A.ch1.step1.trig') is True
    assert core.get('global.tempo') == 120
    assert core.get('morph.position') == 0.0
    redo(core)
    assert core.get('ch1.osc.decay') != 700.0
    assert core.get('global.tempo') == 90
    assert core.get('pattern.A.ch1.step1.trig') is False
    undo(core)
    undo(core)
    assert core.get('pattern.A.ch1.step1.trig') is False  # the set before the bulk change


def test_a_bulk_change_keeps_mutes_and_selection(make_core):
    core = make_core()
    with core.bulk_change('Load preset'):
        core.synth.mute_channel(2, True)
        core.synth.select_channel(4)
        core.synth.channels[0].set_osc_frequency(1234.0)
    undo(core)
    assert core.get('ch3.mute') is True and core.get('global.channel') == 5
    assert core.get('ch1.osc.freq') != 1234.0


def test_a_bulk_change_restores_programs_and_morph(make_core):
    core = make_core()
    with core.bulk_change('Load preset'):
        core.synth.channels[0].set_osc_decay(55.0)
        core.synth.store_program(3)
        core.morph_manager.capture_endpoint_b()
        core.morph_manager.set_position(0.25)
    undo(core)
    assert core.synth.is_program_occupied(3) is False
    assert core.morph_manager.has_different_endpoints() is False
    assert core.get('morph.position') == 0.0


def test_a_bulk_change_that_fails_is_still_a_step(make_core):
    core = make_core()
    with pytest.raises(RuntimeError):
        with core.bulk_change('Load preset'):
            core.synth.channels[0].set_osc_decay(55.0)
            raise RuntimeError('bad file')
    undo(core)
    assert core.get('ch1.osc.decay') != 55.0


def test_a_bulk_change_on_part_of_the_preset(make_core):
    core = make_core()
    core.set('pattern.A.ch1.step1.trig', True)
    with core.bulk_change('Load drum patch', parts=('channels',)):
        core.synth.channels[2].set_osc_decay(77.0)
    undo(core)
    assert core.get('ch3.osc.decay') != 77.0
    assert core.get('pattern.A.ch1.step1.trig') is True


# ---------------------------------------------------------------------------
# with a running stream: undo applies at block start
# ---------------------------------------------------------------------------

def test_undo_and_redo_apply_at_block_start(make_core, backend):
    core = make_core(audio_backend=backend)
    core.wait(core.start())
    stream = backend.stream
    core.set('ch1.osc.decay', 900.0, edit_all=True)
    stream.pull()
    assert core.get('ch5.osc.decay') == 900.0

    assert run_live(core, stream, 'undo')['done'] is True
    assert core.get('ch5.osc.decay') != 900.0
    with core.bulk_change('Randomize'):
        core.synth.channels[0].set_osc_decay(66.0)
        core.pattern_manager.patterns[0].get_channel(0).set_trigger(0, True)
    assert run_live(core, stream, 'undo')['done'] is True
    assert core.get('pattern.A.ch1.step1.trig') is False
    assert run_live(core, stream, 'redo')['done'] is True
    assert core.get('ch1.osc.decay') == 66.0
    assert core.get('pattern.A.ch1.step1.trig') is True


def test_an_undo_the_stream_does_not_apply_in_time_is_an_error_and_stays(make_core, backend):
    core = make_core(audio_backend=backend, stream_timeout=0.2)
    core.wait(core.start())
    core.set('ch1.osc.decay', 900.0)
    backend.stream.pull()

    event = core.wait(core.act('undo'))  # nobody pulls
    assert event['status'] == 'error'
    backend.stream.pull()  # the late block start must not apply it any more
    assert core.get('ch1.osc.decay') == 900.0
    assert core.get('undo.can_undo') is True and core.get('undo.can_redo') is False
    assert run_live(core, backend.stream, 'undo')['done'] is True
    assert core.get('ch1.osc.decay') != 900.0


def test_undoing_a_whole_preset_refreezes_the_heap(make_core, monkeypatch):
    import gc
    core = make_core()
    with core.bulk_change('Load preset'):
        core.pattern_manager.set_bpm(100)
    freezes = []
    monkeypatch.setattr(gc, 'freeze', lambda: freezes.append(1))
    undo(core)
    assert freezes


def test_a_set_to_the_current_value_is_no_step_and_keeps_redo(make_core):
    core = make_core()
    core.set('global.swing', 0.4)
    undo(core)
    core.set('global.swing', 0.0)  # an echo of the value shown
    assert core.get('undo.can_redo') is True
    assert core.get('undo.can_undo') is False
