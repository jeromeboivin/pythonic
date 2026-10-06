"""
App core skeleton: composition, audio stream ownership, command queue in and
poll out. Every test drives the core through its interface (get / set /
describe / act / poll / trigger) and runs the audio callback by hand on a fake
stream, so no audio device is needed.
"""

import copy
import gc
import threading
import time

import numpy as np
import pytest

from pythonic.app import AppCore, Address
from pythonic.lfo import ModTarget
from tests.fake_audio import FakeAudioBackend


# ---------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def backend():
    return FakeAudioBackend()


class FakeClock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


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
    """Start the core and wait until the audio stream is open."""
    event = core.wait(core.start())
    assert event['status'] == 'done', event
    return core


def pump_until(core, stream, done, timeout=5.0):
    """Run audio callbacks until done() is true (actions wait for block start)."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        stream.pull()
        if done():
            return
        time.sleep(0.002)
    raise AssertionError('timed out pumping the audio callback')


def action_events(poll, verb=None):
    return [e for e in poll['events'] if e.get('verb') and (verb is None or e['verb'] == verb)]


def error_events(poll):
    return [e for e in poll['events'] if e['status'] == 'error']


# ---------------------------------------------------------------------------
# composition and start-up
# ---------------------------------------------------------------------------

def test_core_builds_the_services_from_preferences(make_core, prefs):
    prefs.set('audio_sample_rate', 48000)
    prefs.set('synth_sample_rate', 96000)
    prefs.set('audio_mono', True)
    prefs.set('param_smoothing_ms', 12.0)

    core = make_core()

    assert core.synth.sr == 48000  # capped to the output rate
    assert core.synth.mono
    assert core.synth._morph_manager is core.morph_manager
    assert core.morph_manager.synth is core.synth
    assert core.preset_manager.synth is core.synth
    assert core.pattern_manager.num_channels == 8
    assert core.preferences is prefs
    assert core.synth._bpm == core.pattern_manager.bpm
    assert core.get('audio.synth_rate') == 48000
    assert core.get('audio.running') is False


def test_live_synth_runs_without_the_channel_thread_pool(make_core):
    core = make_core()
    assert core.synth.parallel_channel_processing is False
    assert core.synth._thread_pool is None


def test_start_freezes_the_gc_heap_and_opens_the_stream(make_core, backend, prefs, monkeypatch):
    calls = []
    monkeypatch.setattr(gc, 'freeze', lambda: calls.append('freeze'))
    prefs.set('audio_buffer_ms', 10.0)
    core = started(make_core())

    assert 'freeze' in calls
    stream = backend.stream
    assert stream.started
    assert stream.samplerate == 44100
    assert stream.blocksize == 441
    assert core.get('audio.running') is True
    assert core.get('audio.block_size') == 441
    poll = core.poll()
    done = action_events(poll, 'audio.start')
    assert done and done[0]['status'] == 'done'
    assert done[0]['result']['sample_rate'] == 44100
    assert poll['changes']['audio.running'] is True


def test_preferred_device_is_resolved_by_name(make_core, backend, prefs):
    prefs.set('audio_output_device', 'Fake Duplex')
    core = started(make_core())
    assert backend.stream.device == 2
    assert core.get('audio.device') == 'Fake Duplex'


def test_unsupported_rate_falls_back_to_the_device_default(make_core, backend, prefs):
    prefs.set('audio_sample_rate', 96000)
    prefs.set('synth_sample_rate', 96000)
    core = started(make_core())
    assert backend.stream.samplerate == 44100
    assert core.get('audio.sample_rate') == 44100


def test_no_device_reports_an_error_through_poll(make_core, backend):
    backend.fail_open = True
    core = make_core()
    event = core.wait(core.start())
    assert event['status'] == 'error'
    assert 'audio' in event['error'].lower()
    assert core.get('audio.running') is False


def test_core_without_an_audio_backend_still_applies_changes(make_core):
    core = make_core(audio_backend=None)
    event = core.wait(core.start())
    assert event['status'] == 'error'
    applied = []
    core.registry.register(Address('test.value', get=lambda: 0, set=applied.append))
    core.set('test.value', 3)
    assert applied == [3]


def test_device_lists(make_core):
    core = make_core()
    assert core.get('audio.output_devices') == ['Fake Out', 'Fake Duplex']
    assert core.get('audio.input_devices') == ['Fake In', 'Fake Duplex']


# ---------------------------------------------------------------------------
# audio callback
# ---------------------------------------------------------------------------

def test_callback_renders_the_synth_into_the_stream(make_core, backend):
    core = started(make_core())
    core.synth.trigger_drum(0, 127)
    block = backend.stream.pull()
    assert block.shape == (backend.stream.blocksize, 2)
    assert not np.isnan(block).any()
    assert np.abs(block).max() > 0


def test_lower_synth_rate_is_upsampled_to_the_output_rate(make_core, backend, prefs):
    prefs.set('synth_sample_rate', 22050)
    core = started(make_core())
    seen = []
    real = core.synth.process_audio_events

    def spy(n, events):
        seen.append(n)
        return real(n, events)
    core.synth.process_audio_events = spy
    block = backend.stream.pull(1024)
    assert seen == [512]
    assert block.shape == (1024, 2)


def test_queued_set_is_applied_by_the_callback_at_block_start(make_core, backend):
    core = started(make_core())
    applied = []
    core.registry.register(Address('test.value', get=lambda: applied[-1] if applied else 0,
                                   set=lambda v: applied.append((v, threading.get_ident()))))
    v0 = core.poll()['version']

    core.set('test.value', 7)
    assert applied == []  # nothing is applied off the audio thread
    backend.stream.pull()

    assert applied == [(7, threading.get_ident())]  # the test thread runs the callback here
    poll = core.poll(since=v0)
    assert poll['changes'] == {'test.value': 7}
    assert core.poll(since=poll['version'])['changes'] == {}


def test_trigger_is_placed_at_the_sample_offset_of_its_arrival(make_core, backend):
    clock = FakeClock(100.0)
    core = started(make_core(clock=clock))
    events_seen = []
    real = core.synth.process_audio_events

    def spy(n, events):
        events_seen.append(list(events))
        return real(n, events)
    core.synth.process_audio_events = spy
    stream = backend.stream
    period = stream.blocksize / stream.samplerate

    stream.pull()                     # block ending at t=100
    core.trigger(3, 100, at=100.005)  # 5 ms into the next buffer period
    clock.t = 100.0 + period
    stream.pull()

    assert events_seen[-1] == [(int(0.005 * 44100), 3, 100)]


def test_late_or_early_triggers_are_clamped_into_the_block(make_core, backend):
    clock = FakeClock(50.0)
    core = started(make_core(clock=clock))
    events_seen = []
    real = core.synth.process_audio_events

    def spy(n, events):
        events_seen.append(list(events))
        return real(n, events)
    core.synth.process_audio_events = spy
    stream = backend.stream
    core.trigger(0, at=10.0)
    core.trigger(1, at=99.0)
    stream.pull()
    offsets = sorted(e[0] for e in events_seen[-1])
    assert offsets == [0, stream.blocksize - 1]


def test_transport_and_play_position_are_reported_by_poll(make_core, backend):
    core = started(make_core())
    pm = core.pattern_manager
    for step in range(16):
        pm.patterns[0].get_channel(0).set_trigger(step, True)
    pm.set_bpm(240)
    pm.start_playback(0)

    poll = core.poll()
    assert poll['transport']['playing'] is True
    assert poll['transport']['playing_pattern'] == 0
    positions = set()
    for _ in range(80):
        backend.stream.pull()
        positions.add(core.poll()['transport']['position'])
    assert len(positions) > 4

    pm.stop_playback()
    backend.stream.pull()
    assert core.poll()['transport']['playing'] is False


def test_modulation_readouts_of_the_selected_channel(make_core, backend):
    core = started(make_core())
    channel = core.synth.channels[2]
    channel.lfo1.enabled = True
    channel.lfo1.target = ModTarget.PAN
    channel.lfo1.depth = 50.0
    core.synth.select_channel(2)
    core.synth.trigger_drum(2, 127)
    for _ in range(5):
        backend.stream.pull()
    mod = core.poll()['modulation']
    assert mod['channel'] == 2
    assert 'pan' in mod['offsets']


def test_callback_failure_outputs_silence_and_is_reported(make_core, backend):
    core = started(make_core())

    def broken(n, events):
        raise RuntimeError('render exploded')
    core.synth.process_audio_events = broken
    v0 = core.poll()['version']
    block = backend.stream.pull()
    assert not block.any()
    errors = error_events(core.poll(since=v0))
    assert errors and 'render exploded' in errors[0]['error']


def test_poll_reports_audio_performance(make_core, backend):
    core = started(make_core())
    backend.stream.pull()
    backend.stream.pull(underflow=True)
    audio = core.poll()['audio']
    assert audio['callbacks'] == 2
    assert audio['underruns'] == 1
    assert audio['running'] is True


# ---------------------------------------------------------------------------
# act verbs: device and rate changes, stream restarts
# ---------------------------------------------------------------------------

def test_audio_apply_saves_preferences_and_restarts_the_stream(make_core, backend, prefs):
    core = started(make_core())
    first = backend.stream

    event = core.wait(core.act('audio.apply', device='Fake Duplex', sample_rate=48000,
                               synth_rate=0, buffer_ms=10.0, mono=True))

    assert event['status'] == 'done', event
    assert first.aborted and first.closed
    stream = backend.stream
    assert stream is not first and stream.started
    assert stream.device == 2 and stream.samplerate == 48000 and stream.blocksize == 480
    assert prefs.get('audio_output_device') == 'Fake Duplex'
    assert prefs.get('audio_sample_rate') == 48000
    assert prefs.get('synth_sample_rate') == 48000
    assert prefs.get('audio_buffer_ms') == 10.0
    assert prefs.get('audio_mono') is True
    assert core.synth.sr == 48000 and core.synth.mono
    assert event['result']['device'] == 'Fake Duplex'


def test_synth_rate_rebuild_keeps_the_sound_and_swaps_in_whole(make_core, backend):
    core = started(make_core())
    old = core.synth
    old.channels[1].set_osc_frequency(321.0)
    old.channels[4].muted = True
    old.store_program(3)
    old.set_bpm(133.0)
    old.select_channel(5)
    old.set_master_volume(-6.0)

    event = core.wait(core.act('audio.apply', device=None, sample_rate=44100,
                               synth_rate=22050, buffer_ms=23.8, mono=False))

    assert event['status'] == 'done', event
    new = core.synth
    assert new is not old and new.sr == 22050
    assert abs(new.channels[1].oscillator.frequency - 321.0) < 1e-6
    assert new.channels[4].muted
    assert new.is_program_occupied(3)
    assert new._bpm == 133.0
    assert new.selected_channel == 5
    assert new.master_volume_db == -6.0
    assert new._master_gain_linear == old._master_gain_linear
    assert new.parallel_channel_processing is False
    assert core.preset_manager.synth is new
    assert core.morph_manager.synth is new
    assert new._morph_manager is core.morph_manager
    # the new stream renders the new synth at the new ratio
    rendered = []
    real = new.process_audio_events
    new.process_audio_events = lambda n, ev: rendered.append(n) or real(n, ev)
    backend.stream.pull(1024)
    assert rendered == [512]


def test_synth_swap_is_applied_at_block_start_of_a_running_stream(make_core, backend):
    """The rebuilt synth is built off the audio thread and installed by the callback."""
    core = started(make_core())
    stream = backend.stream
    installed_on = []
    real_install = core._install_synth

    def spy(args):
        installed_on.append(threading.get_ident())
        real_install(args)
    core._install_synth = spy
    backend.abort_hangs.set()  # keep the old stream alive until the test pulls once
    try:
        aid = core.act('audio.apply', device=None, sample_rate=44100, synth_rate=22050,
                       buffer_ms=23.8, mono=False)
        pump_until(core, stream, lambda: installed_on)
    finally:
        backend.release.set()
    core.wait(aid)
    assert installed_on == [threading.get_ident()]  # the callback ran on this thread


def test_stream_restart_uses_abort_with_a_timeout_and_reports_a_hang(make_core, backend):
    core = started(make_core(stream_timeout=0.2))
    first = backend.stream
    backend.abort_hangs.set()
    v0 = core.poll()['version']
    try:
        event = core.wait(core.act('audio.apply', device=None, sample_rate=44100,
                                   synth_rate=0, buffer_ms=23.8, mono=False))
    finally:
        backend.abort_hangs.clear()
        backend.release.set()
    assert event['status'] == 'done', event  # a new stream was opened anyway
    assert backend.stream is not first and backend.stream.started
    errors = [e for e in error_events(core.poll(since=v0)) if e.get('verb') is None]
    assert errors and 'stop' in errors[0]['error']
    # the abandoned stream's callback no longer touches the engine
    out = first.pull()
    assert not out.any()


def test_stream_start_hang_is_reported(make_core, backend):
    core = make_core(stream_timeout=0.2)
    backend.start_hangs.set()
    try:
        event = core.wait(core.start())
    finally:
        backend.release.set()
    assert event['status'] == 'error'
    assert core.get('audio.running') is False


def test_stalled_stream_is_reported_and_stopped(make_core, backend):
    """A stream that stops calling back (a wedged device) is reported and
    aborted, so queued changes are applied again by their caller."""
    core = started(make_core(stall_timeout=0.3))
    backend.stream.pull()  # a few callbacks, then nothing
    end = time.monotonic() + 5.0
    while core.get('audio.running') and time.monotonic() < end:
        time.sleep(0.05)
    assert core.get('audio.running') is False
    assert backend.stream.aborted
    errors = error_events(core.poll())
    assert errors and 'stalled' in errors[0]['error']
    applied = []
    core.registry.register(Address('test.value', get=lambda: 0, set=applied.append))
    core.set('test.value', 1)
    assert applied == [1]


def test_audio_stop_and_start_verbs(make_core, backend):
    core = started(make_core())
    assert core.wait(core.act('audio.stop'))['status'] == 'done'
    assert backend.stream.aborted
    assert core.get('audio.running') is False
    assert core.wait(core.act('audio.start'))['status'] == 'done'
    assert core.get('audio.running') is True


def test_unknown_verb_is_reported_as_an_error(make_core):
    core = make_core()
    event = core.wait(core.act('no.such.verb'))
    assert event['status'] == 'error'


# ---------------------------------------------------------------------------
# temporary verb: restore an undo snapshot (until the undo journal lands)
# ---------------------------------------------------------------------------

def _snapshot(core):
    return (copy.deepcopy(core.synth.get_preset_data()),
            copy.deepcopy(core.pattern_manager.to_dict()),
            copy.deepcopy(core.morph_manager.to_dict()))


def test_legacy_restore_snapshot_is_applied_at_block_start(make_core, backend, monkeypatch):
    core = started(make_core())
    snap = _snapshot(core)
    core.synth.channels[0].set_osc_frequency(999.0)
    core.pattern_manager.patterns[0].get_channel(0).set_trigger(3, True)
    freezes = []
    monkeypatch.setattr(gc, 'freeze', lambda: freezes.append(1))

    aid = core.act('legacy.restore_snapshot', snapshot=snap)
    pump_until(core, backend.stream, lambda: action_events(core.poll(), 'legacy.restore_snapshot'))
    event = core.wait(aid)

    assert event['status'] == 'done', event
    assert core.synth.channels[0].oscillator.frequency != 999.0
    assert not core.pattern_manager.patterns[0].get_channel(0).get_step(3).trigger
    assert freezes  # bulk swap re-freezes the heap


def test_legacy_restore_that_times_out_is_cancelled(make_core, backend):
    core = started(make_core(stream_timeout=0.2))
    snap = _snapshot(core)
    core.synth.channels[0].set_osc_frequency(999.0)

    event = core.wait(core.act('legacy.restore_snapshot', snapshot=snap))  # nobody pulls
    assert event['status'] == 'error'
    backend.stream.pull()  # the late block start must not apply it any more
    assert core.synth.channels[0].oscillator.frequency == 999.0


def test_legacy_restore_accepts_old_two_part_snapshots(make_core):
    core = make_core(audio_backend=None)
    synth_data, pattern_data, _ = _snapshot(core)
    core.synth.channels[0].set_osc_frequency(999.0)
    event = core.wait(core.act('legacy.restore_snapshot', snapshot=(synth_data, pattern_data)))
    assert event['status'] == 'done', event
    assert core.synth.channels[0].oscillator.frequency != 999.0


# ---------------------------------------------------------------------------
# address registry
# ---------------------------------------------------------------------------

def test_describe_get_and_set_through_the_registry(make_core):
    core = make_core(audio_backend=None)
    store = {'v': 0.5}
    core.registry.register(Address('test.knob', get=lambda: store['v'],
                                   set=lambda v: store.__setitem__('v', v),
                                   kind='float', minimum=0.0, maximum=1.0, default=0.5,
                                   unit='%', curve='linear'))
    info = core.describe('test.knob')
    assert info['minimum'] == 0.0 and info['maximum'] == 1.0
    assert info['unit'] == '%' and info['readonly'] is False
    core.set('test.knob', 0.25)
    assert core.get('test.knob') == 0.25
    assert 'test.knob' in core.registry.names('test.')


def test_read_only_and_unknown_addresses(make_core):
    core = make_core()
    assert core.describe('audio.sample_rate')['readonly'] is True
    with pytest.raises(ValueError):
        core.set('audio.sample_rate', 48000)
    with pytest.raises(KeyError):
        core.get('no.such.address')
    with pytest.raises(KeyError):
        core.describe('no.such.address')


def test_poll_since_returns_only_newer_events(make_core):
    core = make_core(audio_backend=None)
    first = core.wait(core.act('no.such.verb'))
    v = core.poll()['version']
    assert v >= first['version']
    assert core.poll(since=v)['events'] == []
    second = core.wait(core.act('no.such.verb'))
    events = core.poll(since=v)['events']
    assert [e['id'] for e in events] == [second['id']]


def test_close_aborts_the_stream(make_core, backend):
    core = started(make_core())
    core.close()
    assert backend.stream.aborted and backend.stream.closed
    assert core.get('audio.running') is False
