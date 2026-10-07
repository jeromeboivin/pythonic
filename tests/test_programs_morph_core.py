"""
Programs and the sound morph in the app core (slice 5): the program bank
(``program.select``, ``program.current``, ``program.occupied``) and the morph
(``morph.position``, ``morph.learn``, ``morph.capture``, ``morph.learning``,
``morph.differs``), with their undo steps and poll reports. Every test drives
the core through its interface and runs the audio callback by hand on a fake
stream where a stream is needed.
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
        kwargs.setdefault('audio_backend', None)  # no stream: changes apply inline
        kwargs.setdefault('stall_timeout', None)
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
# programs
# ---------------------------------------------------------------------------

def test_the_program_bank_is_described(make_core):
    core = make_core()
    assert core.get('program.current') == 1
    assert core.describe('program.current')['minimum'] == 1
    assert core.describe('program.current')['maximum'] == 16
    assert core.get('program.occupied') == [False] * 16


def test_selecting_an_empty_program_copies_the_sounds_into_it(make_core):
    core = make_core()
    core.set('ch1.osc.decay', 500.0)
    version = core.poll()['version']

    assert run(core, 'program.select', program=3) == {'program': 3, 'recalled': False}
    assert core.get('program.current') == 3
    assert core.get('ch1.osc.decay') == 500.0  # the sounds stay
    assert core.get('program.occupied')[:3] == [True, False, True]
    changes = core.poll(version)['changes']
    assert changes['program.current'] == 3
    assert changes['program.occupied'][2] is True


def test_switching_programs_stores_the_old_one_and_recalls_the_new_one(make_core):
    core = make_core()
    core.set('ch1.osc.decay', 500.0)
    run(core, 'program.select', program=2)  # a copy of the sounds
    core.set('ch1.osc.decay', 900.0)
    core.set('ch3.mix.level', -12.0)
    version = core.poll()['version']

    assert run(core, 'program.select', program=1) == {'program': 1, 'recalled': True}
    assert core.get('ch1.osc.decay') == 500.0
    assert core.get('ch3.mix.level') == 0.0
    changes = core.poll(version)['changes']
    assert changes['ch1.osc.decay'] == 500.0 and changes['ch3.mix.level'] == 0.0

    run(core, 'program.select', program=2)  # program 2 kept the edits made on it
    assert core.get('ch1.osc.decay') == 900.0
    assert core.get('ch3.mix.level') == -12.0


def test_selecting_the_current_program_does_nothing(make_core):
    core = make_core()
    assert run(core, 'program.select', program=1) == {'program': 1, 'recalled': False}
    assert core.get('program.occupied') == [False] * 16
    assert core.get('undo.can_undo') is False


def test_a_bad_program_is_an_error(make_core):
    core = make_core()
    for bad in (0, 17, 'A'):
        event = core.wait(core.act('program.select', program=bad))
        assert event['status'] == 'error'


def test_a_program_switch_is_one_undo_step(make_core):
    core = make_core()
    core.set('ch1.osc.decay', 500.0)
    run(core, 'program.select', program=2)
    core.set('ch1.osc.decay', 900.0)
    run(core, 'program.select', program=1)
    assert core.get('ch1.osc.decay') == 500.0

    version = core.poll()['version']
    run(core, 'undo')
    assert core.get('program.current') == 2
    assert core.get('ch1.osc.decay') == 900.0
    changes = core.poll(version)['changes']
    assert changes['program.current'] == 2 and changes['ch1.osc.decay'] == 900.0
    run(core, 'redo')
    assert core.get('program.current') == 1
    assert core.get('ch1.osc.decay') == 500.0


def test_a_program_switch_applies_at_block_start(make_core, backend):
    core = make_core(audio_backend=backend)
    core.wait(core.start())
    core.set('ch1.osc.decay', 500.0)
    assert run_live(core, backend.stream, 'program.select', program=5)['program'] == 5
    assert core.get('program.current') == 5


# ---------------------------------------------------------------------------
# morph
# ---------------------------------------------------------------------------

def make_endpoints(core):
    """Endpoint A with a 100 ms decay on channel 1, endpoint B with 1000 ms."""
    core.set('ch1.osc.decay', 100.0)
    run(core, 'morph.capture', endpoint='a')
    core.set('ch1.osc.decay', 1000.0)
    run(core, 'morph.capture', endpoint='b')


def test_the_morph_is_described(make_core):
    core = make_core()
    assert core.get('morph.learning') == 'off'
    assert core.describe('morph.learning')['labels'] == ['off', 'a', 'b']
    assert core.get('morph.differs') is False
    assert core.describe('morph.position')['unit'] == 'ratio'


def test_capture_stores_the_sounds_as_an_endpoint(make_core):
    core = make_core()
    version = core.poll()['version']
    make_endpoints(core)
    assert core.get('morph.differs') is True
    assert core.poll(version)['changes']['morph.differs'] is True

    core.set('morph.position', 0.0)
    assert core.get('ch1.osc.decay') == pytest.approx(100.0)
    core.set('morph.position', 1.0)
    assert core.get('ch1.osc.decay') == pytest.approx(1000.0)
    core.set('morph.position', 0.5)
    assert 100.0 < core.get('ch1.osc.decay') < 1000.0


def test_a_capture_is_one_undo_step(make_core):
    core = make_core()
    core.set('ch1.osc.decay', 100.0)
    run(core, 'morph.capture', endpoint='b')
    assert core.get('morph.differs') is True
    run(core, 'undo')
    assert core.get('morph.differs') is False
    run(core, 'redo')
    assert core.get('morph.differs') is True


def test_learn_pins_the_sound_to_its_endpoint_and_captures_on_stop(make_core):
    core = make_core()
    make_endpoints(core)
    core.set('morph.position', 0.5)
    version = core.poll()['version']

    assert run(core, 'morph.learn', endpoint='a') == {'learning': 'a'}
    assert core.get('morph.learning') == 'a'
    assert core.get('ch1.osc.decay') == pytest.approx(100.0)  # endpoint A sounds
    changes = core.poll(version)['changes']
    assert changes['morph.learning'] == 'a'
    assert changes['ch1.osc.decay'] == pytest.approx(100.0)

    core.set('ch1.osc.decay', 300.0)  # edits go into endpoint A
    core.set('morph.position', 0.0)   # the position moves, the sound stays on A
    assert core.get('ch1.osc.decay') == 300.0
    assert run(core, 'morph.learn', endpoint=None) == {'learning': 'off'}
    assert core.get('morph.learning') == 'off'
    assert core.get('ch1.osc.decay') == pytest.approx(300.0)  # A at position 0
    core.set('morph.position', 1.0)
    assert core.get('ch1.osc.decay') == pytest.approx(1000.0)


def test_learn_switches_between_endpoints(make_core):
    core = make_core()
    make_endpoints(core)
    run(core, 'morph.learn', endpoint='a')
    core.set('ch1.osc.decay', 200.0)
    assert run(core, 'morph.learn', endpoint='b') == {'learning': 'b'}  # A is captured
    assert core.get('ch1.osc.decay') == pytest.approx(1000.0)
    run(core, 'morph.learn', endpoint=None)
    core.set('morph.position', 0.0)
    assert core.get('ch1.osc.decay') == pytest.approx(200.0)


def test_learning_the_same_endpoint_again_does_nothing(make_core):
    core = make_core()
    run(core, 'morph.learn', endpoint='b')
    assert run(core, 'morph.learn', endpoint='b') == {'learning': 'b'}
    assert core.get('morph.learning') == 'b'


def test_a_bad_endpoint_is_an_error(make_core):
    core = make_core()
    for verb in ('morph.learn', 'morph.capture'):
        event = core.wait(core.act(verb, endpoint='c'))
        assert event['status'] == 'error'


def test_stopping_learn_is_one_undo_step_and_starting_it_none(make_core):
    core = make_core()
    make_endpoints(core)
    run(core, 'morph.learn', endpoint='a')
    core.set('ch1.osc.decay', 300.0)
    run(core, 'morph.learn', endpoint=None)
    assert core.get('ch1.osc.decay') == pytest.approx(300.0)

    run(core, 'undo')  # the capture of A when learn stopped
    assert core.get('ch1.osc.decay') == pytest.approx(100.0)  # old A at position 0
    run(core, 'undo')  # the edit made while learning
    run(core, 'undo')  # the capture of B
    run(core, 'undo')  # the decay set before it
    run(core, 'undo')  # the capture of A
    assert core.get('morph.differs') is False


def test_morph_position_is_undone_and_the_sound_follows(make_core):
    core = make_core()
    make_endpoints(core)
    core.set('morph.position', 0.0)
    core.set('morph.position', 1.0)
    assert core.get('ch1.osc.decay') == pytest.approx(1000.0)
    run(core, 'undo')
    assert core.get('morph.position') == 0.0
    assert core.get('ch1.osc.decay') == pytest.approx(100.0)


def test_an_lfo_on_the_morph_destination_moves_the_sound(make_core, backend):
    core = make_core(audio_backend=backend)
    core.wait(core.start())
    stream = backend.stream
    core.set('ch1.osc.freq', 60.0)
    run_live(core, stream, 'morph.capture', endpoint='a')
    core.set('ch1.osc.freq', 2000.0)
    run_live(core, stream, 'morph.capture', endpoint='b')
    core.set('morph.position', 0.0)
    stream.pull()

    def render():
        core.trigger(0, 127)
        return np.concatenate([stream.pull(512) for _ in range(8)])
    still = render()
    for _ in range(40):
        stream.pull(512)  # let the hit ring out
    core.set('ch1.lfo1.on', True)
    core.set('ch1.lfo1.wave', 'square')
    core.set('ch1.lfo1.rate', 0.5)
    core.set('ch1.lfo1.depth', 100.0)
    core.set('ch1.lfo1.target', 'morph')
    stream.pull(512)
    moved = render()
    assert not np.allclose(still, moved)
    assert core.get('morph.position') == 0.0  # modulation leaves the position alone
