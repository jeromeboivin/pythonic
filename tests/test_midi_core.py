"""
MIDI input in the app core (slice 3): device open/close/rescan, notes placed at
their arrival time, program change, transport, clock tempo, the CC map keyed by
address and normalised by describe(), CC pickup, CC-burst undo steps, MIDI
learn, pitch bend and the migration of old name-keyed preferences. Messages are
injected through a fake MIDI backend; the audio callback runs by hand on a fake
stream.
"""

import math
import time

import pytest

from pythonic.app import AppCore
from tests.fake_audio import FakeAudioBackend
from tests.fake_midi import FakeMidiBackend, cc, message, note_on, pitchwheel, program_change


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
def midi():
    return FakeMidiBackend()


@pytest.fixture
def make_core(prefs, backend, midi):
    cores = []

    def make(**kwargs):
        kwargs.setdefault('preferences', prefs)
        kwargs.setdefault('audio_backend', backend)
        kwargs.setdefault('midi_backend', midi)
        kwargs.setdefault('stall_timeout', None)
        core = AppCore(**kwargs)
        cores.append(core)
        return core

    yield make
    for core in cores:
        core.close()


def done(core, action_id):
    event = core.wait(action_id)
    assert event['status'] == 'done', event
    return event.get('result')


def opened(core, device=None):
    """Open a MIDI input on the core (first fake port by default)."""
    done(core, core.act('midi.open', device=device))
    return core


def send(core, midi, *msgs):
    """Deliver messages to the open port and wait until the core handled them."""
    for msg in msgs:
        midi.port.send(msg)
    core.midi.sync()


def wait_until(predicate, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.002)
    raise AssertionError('condition not reached')


def link(core, midi, control):
    """Sweep a controller from 0 to 127: it crosses any value, so pickup links
    it and the control ends at its maximum."""
    send(core, midi, cc(control, 0), cc(control, 127))


def undo_steps(core):
    """Undo every journaled step; returns how many there were."""
    steps = 0
    while done(core, core.act('undo'))['done']:
        steps += 1
    return steps


# ---------------------------------------------------------------------------
# devices
# ---------------------------------------------------------------------------

def test_open_lists_devices_through_describe_and_saves_the_choice(make_core, midi, prefs):
    core = make_core()
    assert core.get('midi.connected') is False

    result = done(core, core.act('midi.open'))

    assert result['device'] == 'Fake Keys'
    assert core.get('midi.device') == 'Fake Keys'
    assert core.get('midi.connected') is True
    assert core.describe('midi.device')['labels'] == ['Fake Keys', 'Fake Pads']
    assert prefs.get('midi_input_device') is None and prefs.get('midi_enabled') is True
    changes = core.poll()['changes']
    assert changes['midi.device'] == 'Fake Keys' and changes['midi.connected'] is True

    done(core, core.act('midi.open', device='Fake Pads'))
    assert midi.opened[0].closed
    assert core.get('midi.device') == 'Fake Pads'
    assert prefs.get('midi_input_device') == 'Fake Pads'


def test_open_of_a_missing_device_is_an_error(make_core, midi):
    core = make_core()
    event = core.wait(core.act('midi.open', device='Nope'))
    assert event['status'] == 'error' and 'Nope' in event['error']
    assert core.get('midi.connected') is False


def test_rescan_and_close(make_core, midi, prefs):
    core = opened(make_core())
    midi.ports.append('Fake Drums')

    assert done(core, core.act('midi.rescan'))['devices'] == ['Fake Keys', 'Fake Pads',
                                                               'Fake Drums']
    assert core.describe('midi.device')['labels'][-1] == 'Fake Drums'

    done(core, core.act('midi.close'))
    assert midi.port.closed
    assert core.get('midi.device') is None and core.get('midi.connected') is False
    assert prefs.get('midi_enabled') is False


def test_start_connects_the_preferred_device_or_the_first(make_core, midi, prefs):
    prefs.set('midi_enabled', True)
    prefs.set('midi_input_device', 'Fake Pads')
    core = make_core()
    core.wait(core.start())
    wait_until(lambda: core.get('midi.device') == 'Fake Pads')

    prefs.set('midi_input_device', 'Unplugged')
    other = make_core()
    other.wait(other.start())
    wait_until(lambda: other.get('midi.device') == 'Fake Keys')


def test_start_leaves_midi_closed_when_disabled(make_core, midi, prefs):
    prefs.set('midi_enabled', False)
    core = make_core()
    core.wait(core.start())
    core.wait(core.act('midi.rescan'))  # the action thread has passed start-up
    assert midi.opened == []


def test_without_a_midi_backend_open_reports_an_error(make_core):
    core = make_core(midi_backend=None)
    event = core.wait(core.act('midi.open'))
    assert event['status'] == 'error' and 'not available' in event['error']


# ---------------------------------------------------------------------------
# notes, program change, transport, clock
# ---------------------------------------------------------------------------

def test_note_is_triggered_at_the_sample_offset_of_its_arrival(make_core, backend, midi):
    clock = FakeClock(100.0)
    core = opened(make_core(clock=clock))
    core.wait(core.start())
    events_seen = []
    real = core.synth.process_audio_events

    def spy(n, events):
        events_seen.append(list(events))
        return real(n, events)
    core.synth.process_audio_events = spy
    stream = backend.stream
    period = stream.blocksize / stream.samplerate

    stream.pull()                       # block ending at t=100
    clock.t = 100.005                   # the note arrives 5 ms into the next period
    send(core, midi, note_on(36 + 3, 90))
    clock.t = 100.0 + period
    stream.pull()

    assert events_seen[-1] == [(int(0.005 * 44100), 3, 90)]


def test_notes_map_from_the_base_note_and_are_counted_in_poll(make_core, midi, prefs):
    core = opened(make_core())
    activity = core.poll()['midi']['activity']

    send(core, midi, note_on(36), note_on(43, 10), note_on(35), note_on(44), note_on(37, 0))

    state = core.poll()['midi']
    assert state['notes'] == [1, 0, 0, 0, 0, 0, 0, 1]
    assert state['activity'] == activity + 5

    core.set('midi.base_note', 48)
    assert prefs.get('midi_base_note') == 48
    send(core, midi, note_on(48), note_on(36))
    assert core.poll()['midi']['notes'][0] == 2
    core.set('midi.base_note', 127)
    assert core.get('midi.base_note') == 120  # room for eight channels


def test_program_change_selects_or_queues_a_pattern(make_core, midi):
    core = opened(make_core())
    pm = core.pattern_manager

    send(core, midi, program_change(4))
    assert core.poll()['transport']['selected_pattern'] == 4

    send(core, midi, program_change(12))  # only A-L
    assert pm.selected_pattern_index == 4

    pm.start_playback(4)
    send(core, midi, program_change(2))
    transport = core.poll()['transport']
    assert transport['selected_pattern'] == 2 and transport['queued_pattern'] == 2
    send(core, midi, program_change(4))   # the playing pattern cancels the queue
    assert core.poll()['transport']['queued_pattern'] is None


def test_start_stop_and_continue(make_core, midi):
    core = opened(make_core())
    pm = core.pattern_manager
    send(core, midi, program_change(3), message('start'))
    transport = core.poll()['transport']
    assert transport['playing'] and transport['playing_pattern'] == 3

    send(core, midi, message('stop'))
    assert core.poll()['transport']['playing'] is False

    pm.play_position = 5
    send(core, midi, message('continue'))
    assert pm.is_playing and pm.play_position == 5


def test_clock_sets_the_tempo_when_sync_is_on(make_core, midi, prefs):
    clock = FakeClock(10.0)
    core = opened(make_core(clock=clock))
    interval = 60.0 / (93 * 24)
    for _ in range(96):
        clock.t += interval
        send(core, midi, message('clock'))
    assert core.get('global.tempo') == 93
    assert core.get('midi.synced_tempo') == 93

    core.set('midi.clock_sync', False)
    assert prefs.get('midi_clock_sync') is False
    for _ in range(96):
        clock.t += 60.0 / (140 * 24)
        send(core, midi, message('clock'))
    assert core.get('global.tempo') == 93


# ---------------------------------------------------------------------------
# CC map, normalisation, pickup
# ---------------------------------------------------------------------------

def test_cc_sets_the_mapped_address_over_its_describe_range(make_core, midi):
    core = opened(make_core())
    core.set('midi.cc_map', {20: 'selected.mix.level', 21: 'selected.osc.freq',
                             22: 'global.master'})
    for control in (20, 21, 22):
        link(core, midi, control)
    assert core.get('ch1.mix.level') == 10.0
    assert core.get('ch1.osc.freq') == 20000.0

    send(core, midi, cc(20, 64), cc(21, 64), cc(22, 0))

    assert core.get('ch1.mix.level') == pytest.approx(-60.0 + 70.0 * 64 / 127)
    assert core.get('ch1.osc.freq') == pytest.approx(20.0 * 1000.0 ** (64 / 127))  # log curve
    assert core.get('global.master') == -60.0


def test_cc_on_a_sound_parameter_follows_the_selected_channel(make_core, midi):
    core = opened(make_core())
    core.set('midi.cc_map', {20: 'selected.mix.pan'})
    core.set('global.channel', 3)
    link(core, midi, 20)
    assert core.get('ch3.mix.pan') == 100.0
    assert core.get('ch1.mix.pan') == 0.0


def test_unlimited_cc_mappings_are_kept_and_saved(make_core, prefs):
    core = make_core()
    mapping = {n: 'selected.mix.level' for n in range(128)}
    core.set('midi.cc_map', mapping)
    assert core.get('midi.cc_map') == mapping
    assert len(prefs.get('midi_cc_mappings')) == 128
    assert 'midi.cc_map' in core.poll()['changes']
    with pytest.raises(ValueError):
        core.set('midi.cc_map', {1: 'audio.running'})
    with pytest.raises(ValueError):
        core.set('midi.cc_map', {128: 'global.master'})


def test_pickup_waits_until_the_controller_crosses_the_value(make_core, midi):
    core = opened(make_core())
    core.set('midi.cc_map', {20: 'selected.mix.level'})
    core.set('ch1.mix.level', -25.0)  # half way

    send(core, midi, cc(20, 0), cc(20, 40))
    assert core.get('ch1.mix.level') == -25.0
    ghost = core.poll()['midi']['pickup']['ch1.mix.level']
    assert ghost['cc'] == 20 and ghost['linked'] is False
    assert ghost['physical'] == pytest.approx(40 / 127)

    send(core, midi, cc(20, 100))  # crossed: the control follows
    assert core.get('ch1.mix.level') == pytest.approx(-60.0 + 70.0 * 100 / 127)
    assert core.poll()['midi']['pickup']['ch1.mix.level']['linked'] is True

    core.set('ch1.mix.level', -25.0)  # a local edit unlinks it
    send(core, midi, cc(20, 101))
    assert core.get('ch1.mix.level') == -25.0
    assert core.poll()['midi']['pickup']['ch1.mix.level']['linked'] is False

    send(core, midi, cc(20, 50))   # crossed again
    assert core.get('ch1.mix.level') == pytest.approx(-60.0 + 70.0 * 50 / 127)


def test_pickup_links_at_once_on_the_current_value(make_core, midi):
    core = opened(make_core())
    core.set('midi.cc_map', {20: 'selected.mix.pan'})
    send(core, midi, cc(20, 64))  # 64/127 is within half a step of the centre
    assert core.poll()['midi']['pickup']['ch1.mix.pan']['linked'] is True


def test_queued_cc_values_do_not_unlink_pickup(make_core, backend, midi):
    core = opened(make_core())
    core.wait(core.start())
    core.set('midi.cc_map', {20: 'selected.mix.level'})
    link(core, midi, 20)  # nothing applied yet: the sets wait for block start
    send(core, midi, cc(20, 30), cc(20, 20))
    backend.stream.pull()
    assert core.get('ch1.mix.level') == pytest.approx(-60.0 + 70.0 * 20 / 127)


# ---------------------------------------------------------------------------
# CC bursts: one undo step each
# ---------------------------------------------------------------------------

def test_a_cc_burst_is_one_undo_step_closed_after_400_ms_idle(make_core, midi):
    clock = FakeClock(10.0)
    core = opened(make_core(clock=clock))
    core.set('midi.cc_map', {20: 'selected.mix.level', 21: 'selected.mix.pan'})
    link(core, midi, 20)  # 0 -> 10 dB: the first burst
    clock.t = 10.5
    version = core.poll()['version']
    assert core.get('undo.can_undo') is True

    for t, value in ((11.0, 100), (11.3, 90), (11.6, 80)):
        clock.t = t
        send(core, midi, cc(20, value))
    clock.t = 11.95
    core.poll()  # 350 ms idle: still open
    clock.t = 12.05
    core.poll()
    assert core.get('ch1.mix.level') == pytest.approx(-60.0 + 70.0 * 80 / 127)

    done(core, core.act('undo'))
    assert core.get('ch1.mix.level') == 10.0
    assert 'ch1.mix.level' in core.poll(version)['changes']
    done(core, core.act('undo'))
    assert core.get('ch1.mix.level') == 0.0
    assert core.get('undo.can_undo') is False


def test_bursts_are_per_control(make_core, midi):
    clock = FakeClock(10.0)
    core = opened(make_core(clock=clock))
    core.set('midi.cc_map', {20: 'selected.mix.level', 21: 'selected.mix.pan'})
    send(core, midi, cc(20, 0), cc(21, 0))
    clock.t = 10.2
    send(core, midi, cc(20, 127), cc(21, 64))
    clock.t = 10.7
    core.poll()
    assert undo_steps(core) == 2


def test_a_burst_that_returns_to_its_start_is_no_step(make_core, midi):
    clock = FakeClock(10.0)
    core = opened(make_core(clock=clock))
    core.set('midi.cc_map', {21: 'selected.mix.pan'})
    core.set('ch1.mix.pan', -100.0 + 200.0 * 64 / 127, record=False)  # on the CC 64 step
    send(core, midi, cc(21, 64), cc(21, 100), cc(21, 64))
    clock.t = 11.0
    core.poll()
    assert core.get('undo.can_undo') is False


def test_clock_tempo_and_pitch_bend_are_not_undo_steps(make_core, midi):
    clock = FakeClock(10.0)
    core = opened(make_core(clock=clock))
    core.set('midi.pitchbend_target', 'global.master')
    send(core, midi, pitchwheel(4000))
    assert core.get('global.master') != 0.0
    send(core, midi, pitchwheel(0))
    for _ in range(96):
        clock.t += 60.0 / (93 * 24)
        send(core, midi, message('clock'))
    assert core.get('global.tempo') == 93
    assert core.get('undo.can_undo') is False


# ---------------------------------------------------------------------------
# learn
# ---------------------------------------------------------------------------

def test_learn_maps_the_next_cc_and_reports_it_through_poll(make_core, midi, prefs):
    core = opened(make_core())
    decay = core.get('ch1.osc.decay')
    learn = core.act('midi.learn', target='selected.osc.decay')
    wait_until(lambda: core.get('midi.learning') == 'selected.osc.decay')
    assert core.poll()['changes']['midi.learning'] == 'selected.osc.decay'

    send(core, midi, cc(21, 5))

    event = core.wait(learn)
    assert event['status'] == 'done'
    assert event['result'] == {'cc': 21, 'target': 'selected.osc.decay'}
    assert core.get('midi.learning') is None
    assert core.get('midi.cc_map')[21] == 'selected.osc.decay'
    assert prefs.get('midi_cc_mappings')['21'] == 'selected.osc.decay'
    changes = core.poll()['changes']
    assert changes['midi.learning'] is None and changes['midi.cc_map'][21] == 'selected.osc.decay'
    assert core.get('ch1.osc.decay') == decay  # the learning CC is not applied

    # A new mapping for the same control replaces the old one
    learn = core.act('midi.learn', target='selected.osc.decay')
    wait_until(lambda: core.get('midi.learning') is not None)
    send(core, midi, cc(22, 5))
    core.wait(learn)
    assert {c: t for c, t in core.get('midi.cc_map').items()
            if t == 'selected.osc.decay'} == {22: 'selected.osc.decay'}


def test_learn_can_be_cancelled_or_replaced(make_core, midi):
    core = opened(make_core())
    first = core.act('midi.learn', target='global.master')
    wait_until(lambda: core.get('midi.learning') == 'global.master')
    second = core.act('midi.learn', target='morph.position')
    assert core.wait(first)['status'] == 'cancelled'
    done(core, core.act('midi.learn_cancel'))
    assert core.wait(second)['status'] == 'cancelled'
    assert core.get('midi.learning') is None

    bad = core.wait(core.act('midi.learn', target='audio.running'))
    assert bad['status'] == 'error'


# ---------------------------------------------------------------------------
# pitch bend
# ---------------------------------------------------------------------------

def test_pitch_bend_offsets_its_target_and_restores_it_at_centre(make_core, midi, prefs):
    core = opened(make_core())
    core.set('midi.pitchbend_target', 'selected.osc.pitch')
    assert prefs.get('midi_pitchbend_target') == 'selected.osc.pitch'
    core.set('ch1.osc.pitch', 4.0)

    send(core, midi, pitchwheel(-4096))
    assert core.get('ch1.osc.pitch') == pytest.approx(4.0 - 12.0)
    send(core, midi, pitchwheel(8191))
    assert core.get('ch1.osc.pitch') == 24.0  # clamped
    send(core, midi, pitchwheel(0))
    assert core.get('ch1.osc.pitch') == 4.0

    core.set('midi.pitchbend_target', None)
    send(core, midi, pitchwheel(4096))
    assert core.get('ch1.osc.pitch') == 4.0
    with pytest.raises(ValueError):
        core.set('midi.pitchbend_target', 'nowhere')


# ---------------------------------------------------------------------------
# preferences
# ---------------------------------------------------------------------------

def test_old_name_keyed_preferences_are_migrated_on_read(make_core, prefs):
    prefs.set('midi_cc_mappings', {'1': 'osc_freq', '7': 'master_volume', '9': 'sound_morph',
                                   '5': 'bogus', '30': 'selected.osc.decay'})
    prefs.set('midi_pitchbend_target', 'pitch')

    core = make_core()

    assert core.get('midi.cc_map') == {1: 'selected.osc.freq', 7: 'global.master',
                                       9: 'morph.position', 30: 'selected.osc.decay'}
    assert core.get('midi.pitchbend_target') == 'selected.osc.pitch'
    assert prefs.get('midi_cc_mappings') == {'1': 'selected.osc.freq', '7': 'global.master',
                                             '9': 'morph.position', '30': 'selected.osc.decay'}
    assert prefs.get('midi_pitchbend_target') == 'selected.osc.pitch'


def test_default_preferences_map_cc1_and_cc2(make_core):
    core = make_core()  # the default preferences name osc_freq and noise_freq
    assert core.get('midi.cc_map') == {1: 'selected.osc.freq', 2: 'selected.noise.freq'}


def test_normalisation_round_trips_on_every_learnable_curve(make_core):
    core = make_core()
    for address in ('ch1.osc.freq', 'ch1.osc.attack', 'ch1.mix.level', 'global.tempo',
                    'ch1.lfo1.rate', 'ch1.osc.wave', 'ch1.mix.choke'):
        entry = core.registry[address]
        for n in (0.0, 0.25, 0.5, 1.0):
            value = entry.coerce(entry.denormalize(n))
            back = entry.normalize(value)
            assert 0.0 <= back <= 1.0
            assert math.isfinite(back)
    assert core.registry['ch1.osc.attack'].denormalize(0.0) == 0.0
    assert core.registry['ch1.osc.attack'].denormalize(1.0) == 10000.0
