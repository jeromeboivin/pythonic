"""
Transport and pad triggers in the app core (slice 4): play, stop, toggle and
continue verbs, the sequencer triggering steps inside the audio callback, and
pad hits. The audio callback runs by hand on a fake stream.
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
def core(prefs, backend):
    core = AppCore(preferences=prefs, audio_backend=backend, stall_timeout=None)
    event = core.wait(core.start())
    assert event['status'] == 'done', event
    yield core
    core.close()


def run(core, stream, verb, **args):
    """Run a verb, pulling audio blocks until it finished; returns its result."""
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


def peak(stream, blocks=20):
    return max(float(np.abs(stream.pull()).max()) for _ in range(blocks))


def transport(core):
    return core.poll()['transport']


def test_play_and_stop(core, backend):
    stream = backend.stream
    run(core, stream, 'pattern.select', pattern='C')
    assert run(core, stream, 'transport.play') == {'playing': True, 'pattern': 'C'}
    state = transport(core)
    assert state['playing'] and state['playing_pattern'] == 2

    run(core, stream, 'pattern.queue', pattern='D')
    assert run(core, stream, 'transport.stop') == {'playing': False}
    state = transport(core)
    assert not state['playing'] and state['position'] == 0
    assert state['queued_pattern'] is None  # stop drops the queue

    run(core, stream, 'transport.play', pattern='F')
    assert transport(core)['playing_pattern'] == 5


def test_play_while_playing_restarts_from_the_first_step(core, backend):
    stream = backend.stream
    core.set('global.tempo', 300)
    run(core, stream, 'transport.play')
    for _ in range(30):
        stream.pull()
    assert transport(core)['position'] > 0
    run(core, stream, 'transport.play')
    assert transport(core)['position'] == 0


def test_toggle_and_continue(core, backend):
    stream = backend.stream
    assert run(core, stream, 'transport.toggle') == {'playing': True, 'pattern': 'A'}
    assert run(core, stream, 'transport.toggle') == {'playing': False}

    run(core, stream, 'pattern.select', pattern='B')
    assert run(core, stream, 'transport.continue') == {'playing': True, 'pattern': 'A'}
    assert transport(core)['playing_pattern'] == 0  # continues the pattern it played
    assert run(core, stream, 'transport.continue') == {'playing': True, 'pattern': 'A'}


def test_playing_steps_trigger_the_channels(core, backend):
    stream = backend.stream
    core.set('global.tempo', 300)
    assert peak(stream) == 0.0
    for step in (1, 5, 9, 13):
        core.set(f'pattern.A.ch1.step{step}.trig', True)
    run(core, stream, 'transport.play')
    assert peak(stream) > 0.01

    run(core, stream, 'transport.stop')
    for _ in range(400):  # let the hits ring out
        stream.pull()
    assert peak(stream) < 1e-4


def test_steps_of_a_muted_channel_or_with_zero_probability_stay_silent(core, backend):
    stream = backend.stream
    core.set('global.tempo', 300)
    for step in range(1, 17):
        core.set(f'pattern.A.ch2.step{step}.trig', True)
        core.set(f'pattern.A.ch2.step{step}.prob', 0)
    run(core, stream, 'transport.play')
    assert peak(stream, 60) == 0.0

    core.set('pattern.A.ch2.prob', [100] * 16)
    core.set('ch2.mute', True)
    assert peak(stream, 60) == 0.0
    core.set('ch2.mute', False)
    assert peak(stream, 60) > 0.01


def test_pad_trigger_plays_the_channel(core, backend):
    stream = backend.stream
    assert peak(stream, 2) == 0.0
    core.trigger(3, 127)
    assert peak(stream, 4) > 0.01
