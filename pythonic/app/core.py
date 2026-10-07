"""
The app core: a UI-free owner of the synth, patterns, morph, preferences and
the audio stream, driven through an address-based interface (ADR 0001).

- ``get(addr)`` / ``set(addr, value, edit_all=None)`` / ``describe(addr)`` on
  registered addresses. ``set`` is queued and applied by the audio thread at
  block start; ``poll`` reports the change once it has been applied. With Edit
  all, a sound change of one channel also goes to every unmuted channel.
- ``act(verb, **args)`` returns an action id at once; the verb runs on the
  core's action thread and its result or error arrives through ``poll``.
- ``trigger(channel, velocity, at)`` queues a hit with its arrival time.
- ``poll(since)`` returns everything newer than version ``since``: changed
  addresses, action events, plus the transport, modulation, audio and MIDI
  readouts.
- MIDI input (``midi.*`` addresses and verbs) is routed by ``MidiInput`` on
  the core's MIDI thread (see ``midi.py``).
- Pattern steps, lanes, pattern ops, selection, the queue and chains
  (``pattern.*``) are in ``patterns.py``.
- Programs (``program.*``) are in ``programs.py``, the sound morph
  (``morph.*``) in ``morph.py``.
- Preset and drum-patch files (``preset.*``, ``drum_patch.*``) are in
  ``presets.py``, the preferences (``pref.*``) in ``prefs.py``.
- MIDI and WAV export (``export.*``) is in ``export.py``; WAV renders run on
  the export thread with an offline synth and report progress through poll.
- The AI generators (``ai.*``) are in ``ai.py``; the models run in a
  subprocess (``ai_worker.py``), so the core never imports torch.
- Undo and redo (``undo`` / ``redo`` verbs, ``undo.*`` addresses): every set
  is journaled (``undo.py``); a front-end brackets a drag with
  ``begin_gesture()`` / ``end_gesture()``, a wheel or controller passes
  ``burst=True``, and code not moved into the core yet wraps its direct
  engine writes in ``with core.bulk_change(label):``.

The core never calls into a UI thread: front-ends poll on their own timer.
"""

import copy
import gc
import itertools
import queue
import threading
import time

from pythonic.drum_channel import DrumChannel
from pythonic.morph_manager import MorphManager
from pythonic.pattern_manager import PatternManager
from pythonic.preferences_manager import PreferencesManager
from pythonic.preset_manager import PresetManager
from pythonic.sequencer import StepSequencer
from pythonic.synthesizer import PythonicSynthesizer

from .ai import Ai
from .audio import AudioEngine
from .export import Export
from .midi import MidiInput, import_mido
from .morph import Morph
from .patterns import Patterns
from .prefs import STREAM_SETTINGS, Prefs
from .presets import Presets
from .programs import Programs
from .registry import Address, Registry
from .sound import SOUND_PARAMS, SOUND_SUFFIXES
from .undo import PRESET, Undo

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

    # A verb handler returns this when its action finishes later (MIDI learn);
    # the handler posts the action's event itself.
    DEFERRED = object()

    def __init__(self, preferences=None, audio_backend=_DEFAULT, *, midi_backend=_DEFAULT,
                 clock=time.perf_counter, stream_timeout=2.0, stall_timeout=2.0,
                 ai_worker=_DEFAULT, ai_install=_DEFAULT):
        """``ai_worker``: the command (argv) of the AI worker process (default:
        ``ai_worker.py`` with this Python when torch is installed); None turns
        the AI off. ``ai_install``: the command ``ai.install`` runs."""
        self.preferences = preferences if preferences is not None else PreferencesManager()
        if audio_backend is _DEFAULT:
            audio_backend = _import_sounddevice()
        if midi_backend is _DEFAULT:
            midi_backend = import_mido()
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
        self._pending = []  # (queue seq, addresses, value, related) not yet applied
        self._edit_all = False
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

        self.ai = None  # set below; AppCore.set reads its overlay prefixes
        self.registry = Registry()
        self._register_addresses()
        self.undo = Undo(self, clock)
        self.undo.register(self.registry)
        self.programs = Programs(self)
        self.programs.register(self.registry)
        self.morph = Morph(self)
        self.morph.register(self.registry)
        self.patterns = Patterns(self)
        self.patterns.register(self.registry)
        self.midi = MidiInput(self, midi_backend, clock, prefs)
        self.midi.register(self.registry)
        self.presets = Presets(self)
        self.presets.register(self.registry)
        self.prefs = Prefs(self)
        self.prefs.register(self.registry)
        self.export = Export(self)
        self.ai = Ai(self, **{k: v for k, v in (('worker', ai_worker), ('install', ai_install))
                              if v is not _DEFAULT})
        self.ai.register(self.registry)
        self._verbs = {
            'audio.start': self._verb_audio_start,
            'audio.stop': self._verb_audio_stop,
            'audio.apply': self._verb_audio_apply,
            **self.undo.verbs(),
            **self.programs.verbs(),
            **self.morph.verbs(),
            **self.patterns.verbs(),
            **self.midi.verbs(),
            **self.presets.verbs(),
            **self.prefs.verbs(),
            **self.export.verbs(),
            **self.ai.verbs(),
        }
        self._running_action = None  # id of the verb running on the action thread

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

    def set(self, address, value, *, edit_all=None, burst=False, record=True):
        """Queue a change; the audio thread applies it at the next block start.

        The value is clamped to the address range (ValueError if it is not a
        valid value). For a per-channel sound address, ``edit_all`` (default:
        the core's Edit all mode, ``global.edit_all``) also applies the value
        to the same parameter of every unmuted channel, in the same block.
        Callers restoring state pass ``edit_all=False``.
        The addresses an address names as ``related`` are reported with it.

        The change is one undo step (with what Edit all changed), or part of
        the open gesture; with ``burst=True`` (a wheel, a MIDI controller) the
        changes of this address are one step until 400 ms pass without one.
        ``record=False`` keeps it out of the journal (MIDI clock, pitch bend).
        """
        entry = self.registry[address]
        if entry.readonly:
            raise ValueError(f'address is read-only: {address}')
        value = entry.coerce(value)
        journal = record and entry.undoable and self.undo.journal.recording
        if journal and self.ai is not None and address.startswith(self.ai.unjournaled):
            journal = False  # a tried AI candidate or pattern preview: kept or dropped whole
        if journal:
            # Old values, read before the change is queued (it may apply at once)
            companions = [(name, self._latest(name)) for name in entry.companions]
            olds = {address: self._latest(address)}
        if not entry.queued:
            entry.set(value)
            self._note_change(address, entry.get())
            self.note_changes(entry.related)
            if journal:
                self.undo.record_set(address, (address,), olds, value, companions, 0, burst)
            return
        others = self._edit_all_others(address, edit_all)
        if not others:
            seq = self.audio.submit_call(entry.set, value)
            names = (address,)
        else:
            if journal:
                olds.update((name, self._latest(name)) for _index, name in others)
            # Which channels are unmuted is read at block start, after any
            # mute queued before this change
            names = [address]
            source = entry.set
            channels = tuple((index, name, self.registry[name].set) for index, name in others)
            audio = self.audio

            def apply_all(v):
                source(v)
                synth_channels = audio.synth.channels
                for index, name, setter in channels:
                    if not synth_channels[index].muted:
                        setter(v)
                        names.append(name)
            seq = self.audio.submit_call(apply_all, value)
        with self._cond:
            self._pending.append((seq, names, value, entry.related))
        if journal:
            self.undo.record_set(address, names, olds, value, companions, seq, burst)

    def begin_gesture(self):
        """Start a gesture (a drag, a paint stroke): the sets until
        end_gesture() are one undo step. A new gesture ends an open one."""
        self.undo.journal.begin_gesture()

    def end_gesture(self):
        self.undo.journal.end_gesture()

    def bulk_change(self, label, parts=PRESET):
        """Context manager for code that writes the engine directly (dialogs
        and loaders not moved into the core yet): everything changed inside is
        one undo step, and poll reports every value it may have changed.
        ``parts`` limits the snapshot ('channels', 'globals', 'programs',
        'morph', 'patterns'; default the whole preset)."""
        return self.undo.bulk_change(label, parts)

    def _edit_all_others(self, address, edit_all):
        """(channel index, address) of the same sound parameter on the other
        channels when Edit all applies to this set, else None."""
        if edit_all is None:
            edit_all = self._edit_all
        if not edit_all or not address.startswith('ch'):
            return None
        channel, _, suffix = address.partition('.')
        if suffix not in SOUND_SUFFIXES:
            return None
        return [(i, f'ch{i + 1}.{suffix}') for i in range(PythonicSynthesizer.NUM_CHANNELS)
                if f'ch{i + 1}' != channel]

    def act(self, verb, **args):
        """Start an action; returns its id. The result arrives through poll()."""
        action_id = next(self._action_ids)
        self._jobs.put((action_id, verb, args))
        return action_id

    def call_soon(self, fn):
        """Run fn() on the action thread, after the actions queued before it
        (modules finish deferred actions there: AI replies). It posts no event;
        an exception is reported as an error event."""
        self._jobs.put((None, fn, None))

    def trigger(self, channel, velocity=127, at=None):
        """Queue a hit; it plays at the sample offset of its arrival time."""
        if at is None:
            at = self._clock()
        self.audio.submit_trigger(at, channel, velocity)

    def _latest(self, address):
        """The newest value of an address: a queued set the audio thread has
        not applied yet, else the current value."""
        drained = self.audio.drained_seq
        with self._cond:
            for seq, names, value, _related in reversed(self._pending):
                if seq <= drained:
                    continue  # applied: the current value is newer (an undo, a verb)
                if address in names:
                    return value
        return self.get(address)

    def poll(self, since=0):
        """Everything newer than `since`, plus the transport and readouts.

        ``changes`` maps each address changed since then to its new value;
        a queued set shows up here once the audio thread has applied it.
        ``transport`` holds the playing state, the play position (0-based
        step of the playing pattern), the playing, selected and queued
        pattern indexes (0..11) and the indexes of the chain being played.
        ``midi`` holds the MIDI activity and per-channel note counters and the
        pickup state of each CC-driven control (controller position, linked).
        """
        self.undo.journal.close_idle()
        self._collect_audio_reports()
        self.ai.collect()
        with self._cond:
            self._promote_applied()
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
                'position': pm.play_position,
                'playing_pattern': pm.playing_pattern_index,
                'selected_pattern': pm.selected_pattern_index,
                'queued_pattern': pm.queued_pattern_index,
                'chain': self.patterns.chain(),
            },
            'modulation': {
                'channel': channel_idx,
                'offsets': {target.value: value for target, value in offsets.items()},
            },
            'audio': self.audio.status(),
            'midi': self.midi.readout(),
        }

    def wait(self, action_id, timeout=10.0):
        """Block until an action has finished and return its event (for tests
        and shutdown paths; front-ends read results from poll)."""
        with self._cond:
            if not self._cond.wait_for(lambda: action_id in self._results, timeout):
                raise TimeoutError(f'action {action_id} did not finish in {timeout} s')
            return self._results[action_id]

    def start(self):
        """Freeze the start-up heap, open the audio stream and, when MIDI input
        is enabled, the saved MIDI input (else the first). Returns the id of
        the audio start action."""
        gc.collect()
        gc.freeze()
        action_id = self.act('audio.start')
        prefs = self.preferences
        if self.midi.backend is not None and prefs.get('midi_enabled', True):
            self.act('midi.open', device=prefs.get('midi_input_device'), fallback=True)
        return action_id

    def close(self):
        """Stop the action thread and the stream, release the synth."""
        if self._closed:
            return
        self._closed = True
        self.midi.close()
        self._jobs.put(None)
        self._worker.join(timeout=self.audio.stream_timeout + 5.0)
        self.export.close()
        self.ai.close()
        self.audio.stop()
        self.synth.cleanup()

    # ================================================================== poll bookkeeping
    def _note_change(self, address, value):
        with self._cond:
            self._promote_applied()  # sets the audio thread applied before are older
            self._version += 1
            self._changes[address] = (self._version, value)

    def note_values(self, values):
        """Report addresses with the given values as one change."""
        with self._cond:
            self._promote_applied()
            self._version += 1
            for name, value in values.items():
                self._changes[name] = (self._version, value)

    def note_changes(self, addresses):
        """Report the current values of several addresses as one change."""
        if not addresses:
            return
        values = {name: self.registry[name].get() for name in addresses}
        with self._cond:
            self._promote_applied()
            self._version += 1
            for name, value in values.items():
                self._changes[name] = (self._version, value)

    def submit(self, fn, value=None, names=()):
        """Queue fn(value) for the next block start without waiting; poll
        reports ``names`` with ``value`` once it has been applied."""
        seq = self.audio.submit_call(fn, value)
        if names:
            with self._cond:
                self._pending.append((seq, list(names), value, ()))

    def at_block_start(self, fn, timeout=None):
        """Run fn() on the audio thread at the next block start (at once when
        no stream runs) and return its result, or raise its exception. Called
        from the action thread by verbs that edit engine state. If the stream
        does not get to it within ``timeout`` (the stream timeout), it is
        cancelled and TimeoutError is raised."""
        if timeout is None:
            timeout = self.audio.stream_timeout
        box = {}
        token = [True]  # popped by whichever side wins: block start or a timeout

        def apply(_):
            try:
                token.pop()
            except IndexError:
                return  # cancelled
            try:
                box['result'] = fn()
            except Exception as exc:  # handed to the caller
                box['error'] = exc

        seq = self.audio.submit_call(apply)
        if not self.audio.wait_applied(seq, timeout):
            try:
                token.pop()
            except IndexError:  # the audio thread is applying it right now
                self.audio.wait_applied(seq, 60.0)
            else:
                raise TimeoutError('the audio stream did not apply the change in time')
        if 'error' in box:
            raise box['error']
        return box.get('result')

    def _promote_applied(self):
        """Move queued sets the audio thread has applied into the changes
        (lock held). Version order follows the queue order."""
        drained = self.audio.drained_seq
        ready = [p for p in self._pending if p[0] <= drained]
        if not ready:
            return
        self._pending = [p for p in self._pending if p[0] > drained]
        ready.sort(key=lambda p: p[0])
        for _seq, names, value, related in ready:
            self._version += 1
            for name in names:
                self._changes[name] = (self._version, value)
            for name in related:
                self._changes[name] = (self._version, self.registry[name].get())

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
            if event.get('id') is not None and event['status'] != 'progress':
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
        self.undo.journal.close_idle()
        self._collect_audio_reports()
        self.ai.collect()
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
            if callable(verb):  # call_soon
                try:
                    verb()
                except Exception as exc:
                    self._report_error(f'{type(exc).__name__}: {exc}', source='core')
                self._monitor()
                continue
            handler = self._verbs.get(verb)
            event = {'id': action_id, 'verb': verb}
            if handler is None:
                event.update(status='error', error=f'unknown verb: {verb}')
            else:
                self._running_action = action_id
                try:
                    result = handler(**args)
                except Exception as exc:
                    event.update(status='error', error=str(exc) or type(exc).__name__)
                else:
                    event.update(status='done', result=result)
                finally:
                    self._running_action = None
                if event.get('result') is self.DEFERRED:
                    self._monitor()
                    continue  # the handler posts the event when the action ends
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
        self._register_sound_addresses()
        self._register_global_addresses()

    def _register_sound_addresses(self):
        """ch1..ch8: every sound parameter, the mute and the patch name."""
        reg = self.registry.register
        fresh = DrumChannel(0, 44100)
        self.init_sound = fresh.get_parameters()  # the sound of a new channel
        defaults = {p.suffix: p.get(fresh) for p in SOUND_PARAMS}
        for index in range(PythonicSynthesizer.NUM_CHANNELS):
            prefix = f'ch{index + 1}'

            def channel(i=index):
                return self.synth.channels[i]

            for param in SOUND_PARAMS:
                reg(Address(f'{prefix}.{param.suffix}',
                            get=lambda p=param, c=channel: p.get(c()),
                            set=lambda v, p=param, c=channel: p.set(c(), v),
                            kind=param.kind, minimum=param.minimum, maximum=param.maximum,
                            default=defaults[param.suffix], unit=param.unit,
                            curve=param.curve, labels=param.labels))
            reg(Address(f'{prefix}.mute', get=lambda c=channel: bool(c().muted),
                        set=lambda v, i=index: self.synth.mute_channel(i, v),
                        kind='bool', default=False, undoable=False))
            reg(Address(f'{prefix}.name', get=lambda c=channel: c().name, kind='str'))

    def _register_global_addresses(self):
        """Tempo, swing, step rate, fill rate, master, selection, Edit all."""
        reg = self.registry.register
        pm = self.pattern_manager

        def set_tempo(bpm):
            pm.set_bpm(bpm)
            self.synth.set_bpm(bpm)

        reg(Address('global.tempo', get=lambda: int(pm.bpm), set=set_tempo, kind='int',
                    minimum=1, maximum=300, default=120, unit='BPM'))
        reg(Address('global.swing', get=lambda: float(pm.swing), set=pm.set_swing,
                    minimum=0.0, maximum=1.0, default=0.0, unit='ratio'))
        reg(Address('global.step_rate', get=lambda: pm.step_rate, set=pm.set_step_rate,
                    kind='enum', default='1/16', labels=tuple(pm.STEP_RATES)))
        reg(Address('global.fill_rate', get=lambda: int(pm.fill_rate), set=pm.set_fill_rate,
                    kind='int', minimum=2, maximum=8, default=4, unit='x'))
        reg(Address('global.master', get=lambda: float(self.synth.master_volume_db),
                    set=lambda v: self.synth.set_master_volume(v),
                    minimum=-60.0, maximum=10.0, default=0.0, unit='dB'))
        reg(Address('global.channel', get=lambda: self.synth.selected_channel + 1,
                    set=lambda v: self.synth.select_channel(v - 1), kind='int',
                    minimum=1, maximum=PythonicSynthesizer.NUM_CHANNELS, default=1,
                    undoable=False))
        reg(Address('global.edit_all', get=lambda: self._edit_all,
                    set=lambda v: setattr(self, '_edit_all', v), kind='bool',
                    default=False, queued=False, undoable=False))

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

    def _verb_audio_apply(self, **settings):
        """Apply the saved stream settings (``pref.audio.*``): rebuild the
        synth if its rate changed and restart the stream on the chosen device.
        Settings given here (device, sample_rate, synth_rate with 0 = same
        as output, buffer_ms, mono) are saved first; any left out keep their
        saved value."""
        unknown = set(settings) - {'device', 'sample_rate', 'synth_rate', 'buffer_ms', 'mono'}
        if unknown:
            raise TypeError(f'audio.apply: unknown settings {sorted(unknown)}')
        prefs = self.preferences
        if set(settings) - {'mono'}:
            saved = self.prefs.stream_settings()
            sample_rate = settings.get('sample_rate', saved['pref.audio.sample_rate'])
            self.prefs.set_stream_settings(
                settings.get('device', saved['pref.audio.device']), sample_rate,
                settings.get('synth_rate', self.get('pref.audio.synth_rate')),
                settings.get('buffer_ms', saved['pref.audio.buffer_ms']))
        if 'mono' in settings:
            prefs.set('audio_mono', bool(settings['mono']))
        stream = self.prefs.stream_settings()
        self.prefs.applied = stream
        device = stream['pref.audio.device']
        sample_rate = stream['pref.audio.sample_rate']
        effective_synth = stream['pref.audio.synth_rate']
        buffer_ms = stream['pref.audio.buffer_ms']
        mono = bool(prefs.get('audio_mono', False))

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
            self.note_changes(['pref.audio.pending', 'pref.audio.mono', *STREAM_SETTINGS])

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
