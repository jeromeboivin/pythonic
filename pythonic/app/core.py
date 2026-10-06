"""
The app core: a UI-free owner of the synth, patterns, morph, preferences and
the audio stream, driven through an address-based interface (ADR 0001).

- ``get(addr)`` / ``set(addr, value)`` / ``describe(addr)`` on registered
  addresses. ``set`` is queued and applied by the audio thread at block start.
- ``act(verb, **args)`` returns an action id at once; the verb runs on the
  core's action thread and its result or error arrives through ``poll``.
- ``trigger(channel, velocity, at)`` queues a hit with its arrival time.
- ``poll(since)`` returns everything newer than version ``since``: changed
  addresses, action events, plus the transport, modulation and audio readouts.

The core never calls into a UI thread: front-ends poll on their own timer.
"""

import copy
import gc
import itertools
import queue
import threading
import time

from pythonic.morph_manager import MorphManager
from pythonic.pattern_manager import Pattern, PatternManager
from pythonic.preferences_manager import PreferencesManager
from pythonic.preset_manager import PresetManager
from pythonic.sequencer import StepSequencer
from pythonic.synthesizer import PythonicSynthesizer

from .audio import AudioEngine
from .registry import Address, Registry

_DEFAULT = object()
_MAX_EVENTS = 256
_REPORT_PERIOD_S = 5.0


def _import_sounddevice():
    try:
        import sounddevice
        return sounddevice
    except Exception:  # missing library or PortAudio
        print("Warning: sounddevice not available. Audio playback disabled.")
        return None


class AppCore:
    """UI-free app core. One per process; front-ends drive it via its interface."""

    def __init__(self, preferences=None, audio_backend=_DEFAULT, *,
                 clock=time.perf_counter, stream_timeout=2.0, stall_timeout=2.0):
        self.preferences = preferences if preferences is not None else PreferencesManager()
        if audio_backend is _DEFAULT:
            audio_backend = _import_sounddevice()
        self._clock = clock
        self.stall_timeout = stall_timeout
        prefs = self.preferences

        out_rate = prefs.get('audio_sample_rate', 44100)
        synth_rate = min(prefs.get('synth_sample_rate', 44100), out_rate)

        self.pattern_manager = PatternManager(num_channels=8, pattern_length=16)
        synth = PythonicSynthesizer(synth_rate, parallel_channel_processing=False)
        synth.set_mono(prefs.get('audio_mono', False))
        self._apply_smoothing(synth)
        self.morph_manager = MorphManager(synth)
        synth.set_morph_manager(self.morph_manager)
        self.preset_manager = PresetManager(synth)
        synth.set_bpm(self.pattern_manager.bpm)

        self.audio = AudioEngine(synth, self.pattern_manager, audio_backend,
                                 clock=clock, stream_timeout=stream_timeout)
        self.audio.configure(out_rate, prefs.get('audio_buffer_ms', 23.8))
        self._device_request = prefs.get('audio_output_device')

        # Poll out: versioned changes and events (non-audio threads only)
        self._cond = threading.Condition()
        self._version = 0
        self._changes = {}
        self._events = []
        self._results = {}
        self._action_ids = itertools.count(1)
        self._seen_errors = 0
        self._last_error_report = -1e9
        self._seen_underruns = 0
        self._seen_drops = 0
        self._last_report = time.monotonic()
        self._reported_count = 0
        self._first_underrun_reported = False
        self._stall_count = -1
        self._stall_since = time.monotonic()
        self._stall_reported = False

        self.registry = Registry()
        self._register_addresses()
        self._verbs = {
            'audio.start': self._verb_audio_start,
            'audio.stop': self._verb_audio_stop,
            'audio.apply': self._verb_audio_apply,
            # Temporary until the undo journal lands (slice 5)
            'legacy.restore_snapshot': self._verb_restore_snapshot,
        }

        self._jobs = queue.Queue()
        self._closed = False
        self._worker = threading.Thread(target=self._worker_loop, name='pythonic-core',
                                        daemon=True)
        self._worker.start()

    # ================================================================== services
    @property
    def synth(self):
        """The live synth. It is replaced when the synth rate changes."""
        return self.audio.synth

    def _apply_smoothing(self, synth):
        smoothing_ms = self.preferences.get('param_smoothing_ms', 30.0)
        for channel in synth.channels:
            channel.set_smoothing_time(smoothing_ms)

    # ================================================================== interface
    def register_verb(self, name, handler):
        """Add an act verb; handler(**args) runs on the action thread."""
        if name in self._verbs:
            raise ValueError(f'verb already registered: {name}')
        self._verbs[name] = handler

    def describe(self, address):
        return self.registry[address].describe()

    def get(self, address):
        return self.registry[address].get()

    def set(self, address, value):
        """Queue a change; the audio thread applies it at the next block start."""
        entry = self.registry[address]
        if entry.readonly:
            raise ValueError(f'address is read-only: {address}')
        self.audio.submit_call(entry.set, value)
        self._note_change(address, value)

    def act(self, verb, **args):
        """Start an action; returns its id. The result arrives through poll()."""
        action_id = next(self._action_ids)
        self._jobs.put((action_id, verb, args))
        return action_id

    def trigger(self, channel, velocity=127, at=None):
        """Queue a hit; it plays at the sample offset of its arrival time."""
        if at is None:
            at = self._clock()
        self.audio.submit_trigger(at, channel, velocity)

    def poll(self, since=0):
        """Everything newer than `since`, plus the transport and readouts."""
        self._collect_audio_reports()
        with self._cond:
            version = self._version
            changes = {a: val for a, (v, val) in self._changes.items() if v > since}
            events = [e for e in self._events if e['version'] > since]
        pm = self.pattern_manager
        synth = self.synth
        channel_idx = synth.selected_channel
        offsets = synth.channels[channel_idx]._last_mod_offsets
        return {
            'version': version,
            'changes': changes,
            'events': events,
            'transport': {
                'playing': pm.is_playing,
                'position': self.audio.play_position,
                'playing_pattern': pm.playing_pattern_index,
                'selected_pattern': pm.selected_pattern_index,
                'queued_pattern': pm.queued_pattern_index,
            },
            'modulation': {
                'channel': channel_idx,
                'offsets': {target.value: value for target, value in offsets.items()},
            },
            'audio': self.audio.status(),
        }

    def wait(self, action_id, timeout=10.0):
        """Block until an action has finished and return its event (for tests
        and shutdown paths; front-ends read results from poll)."""
        with self._cond:
            if not self._cond.wait_for(lambda: action_id in self._results, timeout):
                raise TimeoutError(f'action {action_id} did not finish in {timeout} s')
            return self._results[action_id]

    def start(self):
        """Freeze the start-up heap and open the audio stream (returns the action id)."""
        gc.collect()
        gc.freeze()
        return self.act('audio.start')

    def close(self):
        """Stop the action thread and the stream, release the synth."""
        if self._closed:
            return
        self._closed = True
        self._jobs.put(None)
        self._worker.join(timeout=self.audio.stream_timeout + 5.0)
        self.audio.stop()
        self.synth.cleanup()

    # ================================================================== poll bookkeeping
    def _note_change(self, address, value):
        with self._cond:
            self._version += 1
            self._changes[address] = (self._version, value)

    def _note_audio_changes(self):
        self._stall_count = -1  # a new or stopped stream restarts the stall watch
        for name in self.registry.names('audio.'):
            if not name.endswith('_devices'):
                self._note_change(name, self.get(name))

    def _post(self, event):
        with self._cond:
            self._version += 1
            event['version'] = self._version
            self._events.append(event)
            if len(self._events) > _MAX_EVENTS:
                del self._events[:len(self._events) - _MAX_EVENTS]
            if event.get('id') is not None:
                self._results[event['id']] = event
                while len(self._results) > _MAX_EVENTS:
                    del self._results[next(iter(self._results))]
            self._cond.notify_all()

    def _report_error(self, message, source='audio'):
        print(f"ERROR: {message}", flush=True)
        self._post({'id': None, 'verb': None, 'status': 'error', 'source': source,
                    'error': message})

    def _collect_audio_reports(self):
        """Turn the callback's counters into poll events and console reports."""
        audio = self.audio
        now = time.monotonic()
        with self._cond:
            errors = audio.error_count - self._seen_errors
            if errors > 0 and now - self._last_error_report >= 1.0:
                self._seen_errors = audio.error_count
                self._last_error_report = now
            else:
                errors = 0
            underruns = audio.underrun_count
            new_underruns = underruns - self._seen_underruns
            self._seen_underruns = underruns
            drops = audio.dropped_callback_count
            new_drops = drops - self._seen_drops
            self._seen_drops = drops
        if errors > 0:
            times = f' ({errors} times)' if errors > 1 else ''
            self._report_error(f'Audio callback error{times}: {audio.last_error}')
        if new_underruns > 0 and underruns - new_underruns < 5:
            print(f"[Callback #{audio.callback_count}] UNDERRUN! ({underruns} so far)", flush=True)
        if new_drops > 0 and drops - new_drops < 3:
            print(f"[Callback #{audio.callback_count}] DROPPING audio processing "
                  f"to prevent cascade", flush=True)

    def _monitor(self):
        """Periodic work on the action thread: reports, perf print, stall check."""
        self._collect_audio_reports()
        audio = self.audio
        now = time.monotonic()
        count = audio.callback_count
        first_underrun = (audio.underrun_count >= 1 and count > 50
                          and not self._first_underrun_reported)
        if count != self._reported_count and (now - self._last_report > _REPORT_PERIOD_S
                                              or first_underrun):
            if first_underrun:
                self._first_underrun_reported = True
            self._last_report = now
            self._reported_count = count
            report = audio.performance_report()
            if report:
                print(report, flush=True)
        if not audio.live or not self.stall_timeout:
            self._stall_count = -1
            self._stall_reported = False
            return
        if count != self._stall_count:
            self._stall_count = count
            self._stall_since = now
            self._stall_reported = False
        elif not self._stall_reported and now - self._stall_since > self.stall_timeout:
            # A wedged device: nothing is audible and queued changes would
            # never be applied, so abort the stream and let callers apply them.
            self._stall_reported = True
            self._report_error(f'Audio stream stalled: no callback for '
                               f'{self.stall_timeout:g} s; the stream was stopped')
            error = self.audio.stop()
            if error:
                self._report_error(error)
            self._note_audio_changes()

    # ================================================================== action thread
    def _worker_loop(self):
        while True:
            try:
                job = self._jobs.get(timeout=0.25)
            except queue.Empty:
                self._monitor()
                continue
            if job is None:
                return
            action_id, verb, args = job
            handler = self._verbs.get(verb)
            event = {'id': action_id, 'verb': verb}
            if handler is None:
                event.update(status='error', error=f'unknown verb: {verb}')
            else:
                try:
                    event.update(status='done', result=handler(**args))
                except Exception as exc:
                    event.update(status='error', error=str(exc) or type(exc).__name__)
            self._post(event)
            self._monitor()

    # ================================================================== addresses
    def _register_addresses(self):
        audio = self.audio
        reg = self.registry.register
        reg(Address('audio.running', get=lambda: audio.status()['running'], kind='bool'))
        reg(Address('audio.device', get=audio.device_name, kind='str'))
        reg(Address('audio.device_is_default', get=lambda: audio.status()['default_device'],
                    kind='bool'))
        reg(Address('audio.sample_rate', get=lambda: audio.out_rate, kind='int', unit='Hz'))
        reg(Address('audio.synth_rate', get=lambda: audio.synth.sr, kind='int', unit='Hz'))
        reg(Address('audio.block_size', get=lambda: audio.block_size, kind='int',
                    unit='frames'))
        reg(Address('audio.buffer_ms', get=lambda: audio.buffer_ms, kind='float', unit='ms'))
        reg(Address('audio.mono', get=lambda: audio.synth.mono, kind='bool'))
        reg(Address('audio.output_devices', get=audio.output_devices, kind='list'))
        reg(Address('audio.input_devices', get=audio.input_devices, kind='list'))

    # ================================================================== verbs
    def _verb_audio_start(self):
        try:
            return self.audio.start(self._device_request)
        finally:
            self._note_audio_changes()

    def _verb_audio_stop(self):
        error = self.audio.stop()
        self._note_audio_changes()
        if error:
            self._report_error(error)
        return {'running': False}

    def _verb_audio_apply(self, device=None, sample_rate=44100, synth_rate=0,
                          buffer_ms=23.8, mono=False):
        """Save the audio preferences, rebuild the synth if its rate changed,
        and restart the stream on the chosen device."""
        prefs = self.preferences
        effective_synth = min(synth_rate if synth_rate > 0 else sample_rate, sample_rate)
        prefs.set('audio_output_device', device)
        prefs.set('audio_buffer_ms', buffer_ms)
        prefs.set('audio_sample_rate', sample_rate)
        prefs.set('synth_sample_rate', effective_synth)
        prefs.set('audio_mono', bool(mono))

        old = self.synth
        if effective_synth != old.sr:
            new = self._rebuild_synth(old, effective_synth)
            sequencer = StepSequencer(self.pattern_manager, new.sr)
            # Swapped in at the next block start, or when the stream stops below
            self.audio.submit_call(self._install_synth, (new, sequencer))
            print(f"Synth rate changed to {effective_synth} Hz (synth recreated)", flush=True)
        self.audio.submit_call(lambda m: self.synth.set_mono(m), bool(mono))

        error = self.audio.stop()
        if error:
            self._report_error(error)
        if self.synth is not old:
            old.cleanup()
            gc.freeze()
        self.audio.configure(sample_rate, buffer_ms)
        self._device_request = device
        try:
            return self.audio.start(device)
        finally:
            self._note_audio_changes()

    def _rebuild_synth(self, old, rate):
        """Build a synth at a new rate with the old one's whole state (off the audio thread)."""
        new = PythonicSynthesizer(rate, parallel_channel_processing=False)
        new.load_preset_data(copy.deepcopy(old.get_preset_data()))
        new.load_programs_data(copy.deepcopy(old.get_programs_data()))
        new.set_master_volume(old.master_volume_db)  # load_preset_data skips the gain
        new.set_mono(old.mono)
        new.set_bpm(old._bpm)
        new.select_channel(old.selected_channel)
        for src, dst in zip(old.channels, new.channels):
            dst.muted = src.muted
        self._apply_smoothing(new)
        new.set_morph_manager(self.morph_manager)
        return new

    def _install_synth(self, args):
        """Runs on the audio thread at block start (or inline with no stream)."""
        synth, sequencer = args
        self.audio.install(synth, sequencer)
        self.preset_manager.synth = synth
        self.morph_manager.synth = synth

    def _verb_restore_snapshot(self, snapshot):
        """Temporary: restore an undo snapshot of the tkinter GUI (synth preset
        data, pattern dict, morph dict; old snapshots have no morph part)."""
        if len(snapshot) == 3:
            synth_data, pattern_data, morph_data = snapshot
        else:
            synth_data, pattern_data = snapshot
            morph_data = None
        synth_data = copy.deepcopy(synth_data)
        patterns = [Pattern.from_dict(p) for p in pattern_data['patterns']]
        morph_data = copy.deepcopy(morph_data)

        token = [True]  # popped by whichever side wins: block start or a timeout

        def apply(_):
            try:
                token.pop()
            except IndexError:
                return  # cancelled
            self.synth.load_preset_data(synth_data)
            pm = self.pattern_manager
            pm.patterns = patterns
            pm.selected_pattern_index = pattern_data.get('selected_pattern_index', 0)
            pm.playing_pattern_index = pattern_data.get('playing_pattern_index', 0)
            pm.bpm = pattern_data.get('bpm', 120)
            pm.fill_rate = pattern_data.get('fill_rate', 4)
            pm.step_rate = pattern_data.get('step_rate', '1/16')
            pm._update_step_duration()
            if morph_data:
                self.morph_manager.from_dict(morph_data)

        seq = self.audio.submit_call(apply)
        if not self.audio.wait_applied(seq, self.audio.stream_timeout):
            try:
                token.pop()
            except IndexError:  # the audio thread is applying it right now
                self.audio.wait_applied(seq, 60.0)
            else:
                raise TimeoutError('the audio stream did not apply the snapshot in time')
        gc.freeze()
        return {'morph_position': self.morph_manager.position}
