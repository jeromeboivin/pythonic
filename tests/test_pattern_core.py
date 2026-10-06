"""
Patterns in the app core (slice 4): step-lane addresses, pattern-op verbs with
the lane clipboard, selection, the queued pattern and chains. Every test drives
the core through its interface (get / set / describe / act / poll) and runs
the audio callback by hand on a fake stream.
"""

import time

import numpy as np
import pytest

from pythonic.app import AppCore
from tests.fake_audio import FakeAudioBackend


# ---------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------

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
    """Run a verb on a core without a stream (applied inline) and return its result."""
    event = core.wait(core.act(verb, **args))
    assert event['status'] == 'done', event
    return event.get('result')


def run_live(core, stream, verb, **args):
    """Run a verb on a started core, pulling audio blocks until it finished."""
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


def play(core, stream, pattern=None):
    """Start playback of a pattern (the selected one by default)."""
    pm = core.pattern_manager
    index = pm.selected_pattern_index if pattern is None else 'ABCDEFGHIJKL'.index(pattern)
    core.audio.submit_call(pm.start_playback, index)
    stream.pull()


def lane(core, pattern, channel, field):
    return core.get(f'pattern.{pattern}.ch{channel}.{field}')


def set_triggers(core, pattern, channel, steps):
    for step in steps:
        core.set(f'pattern.{pattern}.ch{channel}.step{step}.trig', True)


# ---------------------------------------------------------------------------
# step and lane addresses
# ---------------------------------------------------------------------------

def test_step_addresses_describe_every_field(make_core):
    core = make_core()
    trig = core.describe('pattern.B.ch2.step5.trig')
    assert trig['kind'] == 'bool' and trig['default'] is False and not trig['readonly']
    prob = core.describe('pattern.B.ch2.step5.prob')
    assert (prob['kind'], prob['minimum'], prob['maximum'], prob['default'], prob['unit']) == \
        ('int', 0, 100, 100, '%')
    assert core.describe('pattern.L.ch8.step16.acc')['kind'] == 'bool'
    assert core.describe('pattern.A.ch1.step1.fill')['kind'] == 'bool'
    assert core.describe('pattern.A.ch1.step1.sub')['kind'] == 'str'
    assert core.describe('pattern.A.length')['maximum'] == 64
    # Step numbers go up to 64 in the grammar, so longer patterns keep their addresses
    assert core.describe('pattern.A.ch1.step64.trig')['kind'] == 'bool'
    for bad in ('pattern.M.ch1.step1.trig', 'pattern.A.ch9.step1.trig',
                'pattern.A.ch1.step0.trig', 'pattern.A.ch1.step65.trig',
                'pattern.A.ch1.step1.bogus', 'pattern.a.ch1.step1.trig'):
        with pytest.raises(KeyError):
            core.describe(bad)


def test_step_set_and_get(make_core):
    core = make_core()
    core.set('pattern.B.ch2.step5.trig', True)
    core.set('pattern.B.ch2.step5.acc', True)
    core.set('pattern.B.ch2.step5.fill', True)
    core.set('pattern.B.ch2.step5.prob', 140)  # clamped
    core.set('pattern.B.ch2.step5.sub', 'oO-')
    assert core.get('pattern.B.ch2.step5.trig') is True
    assert core.get('pattern.B.ch2.step5.acc') is True
    assert core.get('pattern.B.ch2.step5.fill') is True
    assert core.get('pattern.B.ch2.step5.prob') == 100
    assert core.get('pattern.B.ch2.step5.sub') == 'oo-'
    step = core.pattern_manager.patterns[1].channels[1].steps[4]
    assert (step.trigger, step.accent, step.fill, step.substeps) == (True, True, True, 'oo-')
    # Other steps, channels and patterns are untouched
    assert core.get('pattern.B.ch2.step4.trig') is False
    assert core.get('pattern.B.ch3.step5.trig') is False
    assert core.get('pattern.A.ch2.step5.trig') is False
    with pytest.raises(ValueError):
        core.set('pattern.B.ch2.step5.sub', 'ox')


def test_turning_a_trigger_off_clears_its_accent_and_fill(make_core):
    core = make_core()
    for field in ('trig', 'acc', 'fill'):
        core.set(f'pattern.A.ch1.step3.{field}', True)
    core.set('pattern.A.ch1.step3.trig', False)
    assert core.get('pattern.A.ch1.step3.acc') is False
    assert core.get('pattern.A.ch1.step3.fill') is False


def test_lane_addresses_read_and_write_a_whole_lane(make_core):
    core = make_core()
    set_triggers(core, 'C', 4, (1, 5, 9, 13))
    triggers = lane(core, 'C', 4, 'trig')
    assert len(triggers) == 16
    assert [i + 1 for i, t in enumerate(triggers) if t] == [1, 5, 9, 13]
    assert lane(core, 'C', 4, 'prob') == [100] * 16
    assert core.describe('pattern.C.ch4.trig')['kind'] == 'list'

    core.set('pattern.C.ch4.prob', [50] * 4)  # a shorter list sets the first steps
    assert lane(core, 'C', 4, 'prob') == [50] * 4 + [100] * 12
    with pytest.raises(ValueError):
        core.set('pattern.C.ch4.sub', ['x'])


def test_step_sets_are_queued_and_reported_once_applied(make_core, backend):
    core = started(make_core())
    version = core.poll()['version']
    core.set('pattern.A.ch1.step2.trig', True)
    assert core.get('pattern.A.ch1.step2.trig') is False  # waits for block start
    assert 'pattern.A.ch1.step2.trig' not in core.poll(version)['changes']

    backend.stream.pull()
    changes = core.poll(version)['changes']
    assert changes['pattern.A.ch1.step2.trig'] is True
    # The lane and the pattern summary are reported too
    assert changes['pattern.A.ch1.trig'][1] is True
    assert changes['pattern.A.empty'] is False


def test_length_and_empty(make_core):
    core = make_core()
    assert core.get('pattern.A.length') == 16
    assert core.get('pattern.A.empty') is True
    core.set('pattern.A.length', 12)
    assert core.get('pattern.A.length') == 12
    assert len(lane(core, 'A', 1, 'trig')) == 12
    # Steps past the length read as empty and ignore sets
    assert core.get('pattern.A.ch1.step14.trig') is False
    core.set('pattern.A.ch1.step14.trig', True)
    assert core.get('pattern.A.ch1.step14.trig') is False
    core.set('pattern.A.ch1.step3.trig', True)
    assert core.get('pattern.A.empty') is False
    with pytest.raises(ValueError):
        core.set('pattern.A.empty', True)


# ---------------------------------------------------------------------------
# pattern ops
# ---------------------------------------------------------------------------

def test_copy_paste_cut_and_exchange(make_core):
    core = make_core()
    set_triggers(core, 'A', 1, (1, 2))
    assert run(core, 'pattern.paste', pattern='B') == {'pasted': False}  # empty clipboard

    run(core, 'pattern.copy', pattern='A')
    assert run(core, 'pattern.paste', pattern='B') == {'pasted': True}
    assert lane(core, 'B', 1, 'trig')[:3] == [True, True, False]

    set_triggers(core, 'C', 2, (4,))
    run(core, 'pattern.cut', pattern='C')
    assert core.get('pattern.C.empty') is True
    assert run(core, 'pattern.exchange', pattern='A') == {'exchanged': True}
    assert lane(core, 'A', 2, 'trig')[3] is True and lane(core, 'A', 1, 'trig')[0] is False
    run(core, 'pattern.paste', pattern='D')  # the clipboard now holds the old A
    assert lane(core, 'D', 1, 'trig')[:2] == [True, True]


def test_clear_shift_and_reverse(make_core):
    core = make_core()
    set_triggers(core, 'A', 1, (1,))
    run(core, 'pattern.shift_right', pattern='A')
    assert lane(core, 'A', 1, 'trig')[1] is True
    run(core, 'pattern.shift_left', pattern='A')
    run(core, 'pattern.shift_left', pattern='A')
    assert lane(core, 'A', 1, 'trig')[15] is True  # wraps
    run(core, 'pattern.reverse', pattern='A')
    assert lane(core, 'A', 1, 'trig')[0] is True
    run(core, 'pattern.clear', pattern='A')
    assert core.get('pattern.A.empty') is True


def test_randomize_alter_and_accents_fills(make_core):
    core = make_core()
    np.random.seed(1)
    run(core, 'pattern.randomize', pattern='E')
    assert core.get('pattern.E.empty') is False
    triggers = [lane(core, 'E', c, 'trig') for c in range(1, 9)]
    for c in range(1, 9):
        accents = lane(core, 'E', c, 'acc')
        assert not any(a and not t for a, t in zip(accents, triggers[c - 1]))

    run(core, 'pattern.randomize_accents_fills', pattern='E')
    assert [lane(core, 'E', c, 'trig') for c in range(1, 9)] == triggers  # triggers kept

    count = sum(sum(t) for t in triggers)
    run(core, 'pattern.alter', pattern='E')
    after = sum(sum(lane(core, 'E', c, 'trig')) for c in range(1, 9))
    assert 0 < after <= count


def test_pattern_ops_default_to_the_selected_pattern_and_report_changes(make_core):
    core = make_core()
    run(core, 'pattern.select', pattern='F')
    version = core.poll()['version']
    set_triggers(core, 'F', 3, (1,))
    run(core, 'pattern.shift_right')
    assert lane(core, 'F', 3, 'trig')[:2] == [False, True]
    changes = core.poll(version)['changes']
    assert changes['pattern.F.ch3.trig'][:2] == [False, True]
    assert 'pattern.F.length' in changes and 'pattern.F.empty' in changes


def test_pattern_arguments_take_letters_or_indexes(make_core):
    core = make_core()
    set_triggers(core, 'B', 1, (1,))
    run(core, 'pattern.copy', pattern=1)
    run(core, 'pattern.paste', pattern='c')
    assert lane(core, 'C', 1, 'trig')[0] is True
    event = core.wait(core.act('pattern.copy', pattern='Z'))
    assert event['status'] == 'error'


def test_lane_clipboard_copies_one_channel_without_substeps(make_core):
    core = make_core()
    assert run(core, 'pattern.paste_lane', pattern='A', channel=1) == {'pasted': False}
    set_triggers(core, 'A', 2, (3,))
    core.set('pattern.A.ch2.step3.acc', True)
    core.set('pattern.A.ch2.step3.prob', 40)
    core.set('pattern.A.ch2.step3.sub', 'oo')
    run(core, 'pattern.copy_lane', pattern='A', channel=2)

    core.set('pattern.D.ch5.step3.sub', 'o-o')
    assert run(core, 'pattern.paste_lane', pattern='D', channel=5) == {'pasted': True}
    assert core.get('pattern.D.ch5.step3.trig') is True
    assert core.get('pattern.D.ch5.step3.acc') is True
    assert core.get('pattern.D.ch5.step3.prob') == 40
    assert core.get('pattern.D.ch5.step3.sub') == 'o-o'  # substeps are not copied
    # The pattern clipboard is separate
    assert run(core, 'pattern.paste', pattern='E') == {'pasted': False}


def test_pattern_ops_wait_for_block_start(make_core, backend):
    core = started(make_core())
    stream = backend.stream
    core.set('pattern.A.ch1.step1.trig', True)
    action_id = core.act('pattern.shift_right', pattern='A')
    with pytest.raises(TimeoutError):
        core.wait(action_id, timeout=0.05)
    assert run_live(core, stream, 'pattern.reverse', pattern='A') is None
    assert core.wait(action_id)['status'] == 'done'
    # The queued set landed before the shift, then the reverse: step 2 -> step 15
    assert lane(core, 'A', 1, 'trig')[14] is True


# ---------------------------------------------------------------------------
# selection, queue and chains
# ---------------------------------------------------------------------------

def test_select_while_stopped_switches_and_resets_the_position(make_core):
    core = make_core()
    core.pattern_manager.play_position = 5
    run(core, 'pattern.select', pattern='C')
    transport = core.poll()['transport']
    assert transport['selected_pattern'] == 2 and transport['queued_pattern'] is None
    assert transport['position'] == 0
    assert core.get('pattern.selected') == 'C'


def test_select_while_playing_queues_until_the_pattern_ends(make_core, backend):
    core = started(make_core())
    stream = backend.stream
    core.set('global.tempo', 300)
    set_triggers(core, 'A', 1, range(1, 17))
    play(core, stream)
    run_live(core, stream, 'pattern.select', pattern='B')
    transport = core.poll()['transport']
    assert transport['selected_pattern'] == 1 and transport['queued_pattern'] == 1
    assert transport['playing_pattern'] == 0

    # Selecting the playing pattern cancels the queue
    run_live(core, stream, 'pattern.select', pattern='A')
    assert core.poll()['transport']['queued_pattern'] is None
    run_live(core, stream, 'pattern.queue', pattern='B')
    assert core.poll()['transport']['queued_pattern'] == 1

    seen = []
    for _ in range(400):
        stream.pull()
        transport = core.poll()['transport']
        seen.append((transport['playing_pattern'], transport['position']))
        if transport['playing_pattern'] == 1:
            break
    assert seen[-1] == (1, 0)
    assert max(p for i, p in seen if i == 0) == 15  # A played to its end first
    assert core.poll()['transport']['queued_pattern'] is None


def test_chains_link_neighbours_and_report_the_chain(make_core):
    core = make_core()
    assert run(core, 'pattern.chain_next', pattern='A') == {'chained': True}
    assert run(core, 'pattern.chain_prev', pattern='C') == {'chained': True}
    assert core.get('pattern.A.chained') is True and core.get('pattern.B.chained') is True
    run(core, 'pattern.chain_prev', pattern='A')  # nothing before A
    run(core, 'pattern.chain_next', pattern='L')  # nothing after L
    assert core.get('pattern.L.chained') is False

    version = core.poll()['version']
    assert run(core, 'pattern.chain_next', pattern='B') == {'chained': False}
    assert core.poll(version)['changes']['pattern.B.chained'] is False
    core.set('pattern.B.chained', True)
    assert core.get('pattern.B.chained') is True
    run(core, 'pattern.chain_clear')
    assert not any(core.get(f'pattern.{p}.chained') for p in 'ABCDEFGHIJKL')


def test_chain_advances_loops_and_selection_follows(make_core, backend):
    core = started(make_core())
    stream = backend.stream
    core.set('global.tempo', 300)
    for name in 'ABC':
        core.set(f'pattern.{name}.length', 2)
    run_live(core, stream, 'pattern.chain_next', pattern='A')
    run_live(core, stream, 'pattern.chain_next', pattern='B')
    run_live(core, stream, 'pattern.select', pattern='B')
    play(core, stream)

    order = []
    for _ in range(600):
        stream.pull()
        transport = core.poll()['transport']
        if not order or order[-1] != transport['playing_pattern']:
            order.append(transport['playing_pattern'])
            assert transport['selected_pattern'] == transport['playing_pattern']
            assert transport['chain'] == [0, 1, 2]
        if len(order) >= 5:
            break
    assert order[:5] == [1, 2, 0, 1, 2]


def test_the_queued_pattern_goes_ahead_of_the_chain(make_core, backend):
    core = started(make_core())
    stream = backend.stream
    core.set('global.tempo', 300)
    core.set('pattern.A.length', 2)
    run_live(core, stream, 'pattern.chain_next', pattern='A')
    play(core, stream, 'A')
    run_live(core, stream, 'pattern.queue', pattern='E')
    for _ in range(600):
        stream.pull()
        if core.poll()['transport']['playing_pattern'] != 0:
            break
    assert core.poll()['transport']['playing_pattern'] == 4
