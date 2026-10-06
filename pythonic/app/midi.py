"""
MIDI input of the app core.

The MIDI backend (mido) calls ``MidiInput._receive`` on its own thread as each
message arrives. The core stamps the message with its clock and hands it to
its MIDI thread, which routes it:

- **Note on** (base note .. base note + 7, velocity > 0): ``core.trigger``
  with the arrival time, so the hit lands at the matching sample offset.
- **Program change** 0-11: selects pattern A-L (queued while playing).
- **Start / stop / continue**: the transport. **Clock**: ``global.tempo``
  while clock sync is on.
- **Control change**: the address its CC maps to, at the 0..1 position of the
  CC over the address's ``describe()`` range and curve, with pickup. While MIDI
  learn runs, the next CC is mapped to the learn target instead.
- **Pitch bend**: a temporary offset of its target, +-half the range around
  the value at bend start, restored when the wheel returns to centre.

CC map and pitch-bend targets are addresses (``global.master``), or
``selected.<sound parameter>`` (``selected.osc.decay``) for the sound
parameter of whichever channel is selected when the message arrives.

**Pickup** (soft takeover): a controller moves its control only once it has
crossed the control's current value (or arrives on it). Any other change of
the value (a knob, a preset, the morph) unlinks it until the next crossing.
``poll()['midi']['pickup']`` gives each control's controller position for a
ghost marker.

**CC bursts**: the CC changes of one control are one undo step, closed after
400 ms without a change on that control. Until the undo journal lands, each
burst carries a snapshot of the state before it (``AppCore.legacy_snapshot``)
for the front-end's snapshot undo, reported as a ``cc_burst`` poll event.
"""

import queue
import threading
from typing import List, Optional

from .sound import (
    CHANNEL_CC_PARAMETERS, GLOBAL_CC_PARAMETERS, SELECTED_PREFIX, SOUND_SUFFIXES,
    cc_parameter_target, resolve_target,
)
from .registry import Address

NUM_CHANNELS = 8
NUM_PATTERNS = 12
MAX_BASE_NOTE = 127 - (NUM_CHANNELS - 1)
BURST_IDLE_S = 0.4
PITCHBEND_DEAD_ZONE = 0.02
# A controller within half a CC step of the value is on it
LINK_TOLERANCE = 0.5 / 127 + 1e-9

CC_NAMES = {
    1: "Mod Wheel", 2: "Breath Controller", 4: "Foot Controller", 7: "Volume", 10: "Pan",
    11: "Expression", 12: "Effect Ctrl 1", 13: "Effect Ctrl 2", 16: "General Purpose 1",
    17: "General Purpose 2", 18: "General Purpose 3", 19: "General Purpose 4",
    64: "Sustain Pedal", 65: "Portamento", 71: "Resonance/Timbre", 74: "Brightness/Cutoff",
    91: "Reverb", 93: "Chorus",
}

_NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def cc_name(number):
    """Display name of a CC number, such as ``CC1 (Mod Wheel)``."""
    if number in CC_NAMES:
        return f"CC{number} ({CC_NAMES[number]})"
    return f"CC{number}"


def note_name(note):
    """Name of a MIDI note, such as ``C1`` for 36."""
    return f"{_NOTE_NAMES[note % 12]}{note // 12 - 1}"


def import_mido():
    try:
        import mido
        return mido
    except Exception:  # missing library or MIDI backend
        print("MIDI input not available (mido library not installed)")
        return None


class ClockTempo:
    """Tempo from MIDI clock (24 per quarter note), from message arrival times.

    Averages over the last 192 clocks, checks every 48 clocks, needs 48 clocks
    for a first reading, restarts after 0.5 s without clock, and reports a
    whole BPM in 20..300 when it moves by 1 or more.
    """

    PPQN = 24
    WINDOW = 192
    UPDATE_INTERVAL = 48
    MIN_SAMPLES = 48
    TIMEOUT_S = 0.5
    MIN_BPM = 20.0
    MAX_BPM = 300.0

    def __init__(self):
        self.times: List[float] = []
        self.since_update = 0
        self.reported = 0

    def reset(self, full=False):
        self.times.clear()
        self.since_update = 0
        if full:
            self.reported = 0

    def tick(self, at) -> Optional[int]:
        """Add one clock; returns a new tempo to report, else None."""
        times = self.times
        if times and at - times[-1] > self.TIMEOUT_S:
            self.reset()
        times.append(at)
        self.since_update += 1
        if len(times) > self.WINDOW:
            del times[:len(times) - self.WINDOW]
        if self.since_update < self.UPDATE_INTERVAL:
            return None
        self.since_update = 0
        if len(times) < self.MIN_SAMPLES:
            return None
        span = times[-1] - times[0]
        if span <= 0:
            return None
        bpm = 60.0 / (span / (len(times) - 1) * self.PPQN)
        if not self.MIN_BPM <= bpm <= self.MAX_BPM:
            return None
        bpm = int(round(bpm))
        if self.reported == 0 or abs(bpm - self.reported) >= 1:
            self.reported = bpm
            return bpm
        return None


class _Pickup:
    """Pickup state of one control: where its controller is and whether it follows."""

    __slots__ = ('cc', 'physical', 'linked', 'expected', 'count')

    def __init__(self):
        self.cc = None
        self.physical = None  # last controller position, 0..1
        self.linked = False
        self.expected = None  # the value the controller last wrote
        self.count = 0        # CC messages received (a front-end blinks on change)


class _Burst:
    __slots__ = ('address', 'old', 'new', 'last', 'snapshot')

    def __init__(self, address, old, new, last, snapshot):
        self.address = address
        self.old = old
        self.new = new
        self.last = last
        self.snapshot = snapshot


class MidiInput:
    """The core's MIDI input: device, routing, CC map, learn, pickup, bursts."""

    def __init__(self, core, backend, clock, preferences):
        self._core = core
        self.backend = backend
        self._clock = clock
        self._prefs = preferences
        self._lock = threading.Lock()  # shared with the action and UI threads

        self._port = None
        self._port_name = None
        self._device_entry = None
        self.base_note = max(0, min(MAX_BASE_NOTE, int(preferences.get('midi_base_note', 36))))
        self.clock_sync = bool(preferences.get('midi_clock_sync', True))
        self._tempo = ClockTempo()
        self._cc_map = self._read_cc_map()
        self._pitchbend_target = self._read_pitchbend_target()
        self._learn = None   # (action id, target)
        self._pickup = {}    # address -> _Pickup
        self._bursts = {}    # address -> _Burst
        self._bend = None    # (address, value at bend start)
        self.activity = 0
        self.notes = [0] * NUM_CHANNELS

        self._queue = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name='pythonic-midi', daemon=True)
        self._thread.start()

    # ================================================================== addresses, verbs
    def register(self, registry):
        reg = registry.register
        self._device_entry = reg(Address('midi.device', get=lambda: self._port_name, kind='str'))
        reg(Address('midi.connected', get=lambda: self._port is not None, kind='bool'))
        reg(Address('midi.base_note', get=lambda: self.base_note, set=self._set_base_note,
                    kind='int', minimum=0, maximum=MAX_BASE_NOTE, default=36, queued=False))
        reg(Address('midi.clock_sync', get=lambda: self.clock_sync, set=self._set_clock_sync,
                    kind='bool', default=True, queued=False))
        reg(Address('midi.synced_tempo', get=lambda: self._tempo.reported, kind='int',
                    unit='BPM'))
        reg(Address('midi.cc_map', get=self.cc_map, set=self._set_cc_map, kind='map',
                    queued=False))
        reg(Address('midi.pitchbend_target', get=lambda: self._pitchbend_target,
                    set=self._set_pitchbend_target, kind='str', queued=False))
        reg(Address('midi.learning', get=lambda: self._learn[1] if self._learn else None,
                    kind='str'))

    def verbs(self):
        return {
            'midi.open': self._verb_open,
            'midi.close': self._verb_close,
            'midi.rescan': self._verb_rescan,
            'midi.learn': self._verb_learn,
            'midi.learn_cancel': self._verb_learn_cancel,
        }

    def cc_map(self):
        """CC number -> target (a copy)."""
        with self._lock:
            return dict(self._cc_map)

    def valid_target(self, target):
        """True for an address a controller can drive (any value with a 0..1
        position), or ``selected.<sound parameter>``."""
        if not isinstance(target, str):
            return False
        if target.startswith(SELECTED_PREFIX):
            return target[len(SELECTED_PREFIX):] in SOUND_SUFFIXES
        registry = self._core.registry
        if target.startswith(('midi.', 'audio.')) or target not in registry:
            return False
        entry = registry[target]
        return not entry.readonly and entry.positional

    # ================================================================== settings
    def _set_base_note(self, note):
        self.base_note = note
        self._prefs.set('midi_base_note', note)

    def _set_clock_sync(self, enabled):
        self.clock_sync = enabled
        if not enabled:
            self._tempo.reset()
        self._prefs.set('midi_clock_sync', enabled)

    def _clean_map(self, mapping):
        clean = {}
        for key, target in dict(mapping).items():
            control = int(key)
            if not 0 <= control <= 127:
                raise ValueError(f'midi.cc_map: CC {key} is not 0..127')
            if not self.valid_target(target):
                raise ValueError(f'midi.cc_map: {target!r} is not a learnable control')
            clean[control] = target
        return clean

    def _set_cc_map(self, mapping):
        clean = self._clean_map(mapping)
        with self._lock:
            self._cc_map = clean
        self._save_cc_map(clean)

    def _save_cc_map(self, mapping):
        self._prefs.set('midi_cc_mappings', {str(c): t for c, t in sorted(mapping.items())})

    def _set_pitchbend_target(self, target):
        if target is not None and not self.valid_target(target):
            raise ValueError(f'midi.pitchbend_target: {target!r} is not a learnable control')
        self._pitchbend_target = target
        self._prefs.set('midi_pitchbend_target', target)

    # Saved preferences named controls by their old parameter names
    # ('osc_freq'); they are read as targets and saved back in that form.
    def _migrate(self, name):
        if name in CHANNEL_CC_PARAMETERS or name in GLOBAL_CC_PARAMETERS:
            return cc_parameter_target(name)
        return name if self.valid_target(name) else None

    def _read_cc_map(self):
        raw = self._prefs.get('midi_cc_mappings', {}) or {}
        mapping, changed = {}, False
        for key, name in raw.items():
            try:
                control = int(key)
            except (TypeError, ValueError):
                control = -1
            target = self._migrate(name)
            if target is None or not 0 <= control <= 127:
                print(f"MIDI: dropped the CC mapping {key} -> {name!r} (unknown control)")
                changed = True
                continue
            changed = changed or target != name or str(key) != str(control)
            mapping[control] = target
        if changed:
            self._save_cc_map(mapping)
        return mapping

    def _read_pitchbend_target(self):
        name = self._prefs.get('midi_pitchbend_target', 'pitch')
        if name is None:
            return None
        target = self._migrate(name)
        if target != name:
            self._prefs.set('midi_pitchbend_target', target)
        return target

    # ================================================================== devices
    def _require_backend(self):
        if self.backend is None:
            raise RuntimeError('MIDI input not available (mido is not installed)')

    def _scan(self):
        try:
            ports = list(self.backend.get_input_names())
        except Exception as exc:
            print(f"Error getting MIDI ports: {exc}")
            ports = []
        if self._device_entry is not None:
            self._device_entry.labels = tuple(ports)
        return ports

    def _close_port(self):
        with self._lock:
            port, self._port, self._port_name = self._port, None, None
        if port is not None:
            try:
                port.close()
            except Exception:
                pass
        return port is not None

    def _note_device(self):
        self._core._note_change('midi.device', self._port_name)
        self._core._note_change('midi.connected', self._port is not None)

    def _verb_open(self, device=None, fallback=False):
        """Open an input port: `device`, or the first one when it is None (or,
        with fallback, when it is missing). Saves the choice."""
        self._require_backend()
        ports = self._scan()
        name = device
        if name is None or name not in ports:
            if name is not None and not fallback:
                raise RuntimeError(f"MIDI port '{device}' not found")
            if not ports:
                raise RuntimeError('No MIDI input ports available')
            if name is not None:
                print(f"MIDI port '{device}' not found, using '{ports[0]}'")
            name = ports[0]
        self._close_port()
        try:
            port = self.backend.open_input(name, callback=self._receive)
        finally:
            self._note_device()
        with self._lock:
            self._port, self._port_name = port, name
        self._prefs.set('midi_input_device', device)
        self._prefs.set('midi_enabled', True)
        self._note_device()
        print(f"Connected to MIDI input: {name} (drum notes {note_name(self.base_note)}-"
              f"{note_name(self.base_note + NUM_CHANNELS - 1)})")
        return {'device': name}

    def _verb_close(self):
        self._close_port()
        self._prefs.set('midi_enabled', False)
        self._note_device()
        return {'device': None}

    def _verb_rescan(self):
        self._require_backend()
        return {'devices': self._scan()}

    def close(self):
        """Close the port and stop the MIDI thread (core shutdown)."""
        self._close_port()
        self._queue.put(None)
        self._thread.join(timeout=2.0)

    # ================================================================== learn
    def _verb_learn(self, target):
        """Map the next CC to `target`; the action finishes when it arrives."""
        if not self.valid_target(target):
            raise ValueError(f'not a learnable control: {target!r}')
        action_id = self._core._running_action
        with self._lock:
            old, self._learn = self._learn, (action_id, target)
        if old is not None:
            self._post_learn(old, 'cancelled')
        self._core._note_change('midi.learning', target)
        return self._core.DEFERRED

    def _verb_learn_cancel(self):
        with self._lock:
            old, self._learn = self._learn, None
        if old is not None:
            self._post_learn(old, 'cancelled')
            self._core._note_change('midi.learning', None)
        return {'cancelled': old is not None}

    def _post_learn(self, learn, status, result=None):
        event = {'id': learn[0], 'verb': 'midi.learn', 'status': status}
        if result is not None:
            event['result'] = result
        self._core._post(event)

    # ================================================================== MIDI thread
    def _receive(self, msg):
        """Backend callback (its own thread): stamp the arrival time and queue."""
        self._queue.put((self._clock(), msg))

    def sync(self):
        """Block until every received message has been handled (tests)."""
        self._queue.join()

    def _loop(self):
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                at, msg = item
                try:
                    self.handle(msg, at)
                except Exception as exc:
                    self._core._report_error(f'MIDI message {msg!r} failed: {exc}', source='midi')
            finally:
                self._queue.task_done()

    def handle(self, msg, at):
        """Route one message that arrived at clock time `at`."""
        self.activity += 1
        kind = msg.type
        if kind == 'note_on':
            if msg.velocity > 0:
                channel = msg.note - self.base_note
                if 0 <= channel < NUM_CHANNELS:
                    self._core.trigger(channel, msg.velocity, at)
                    self.notes[channel] += 1
        elif kind == 'control_change':
            self._on_cc(msg.control, msg.value, at)
        elif kind == 'pitchwheel':
            self._on_pitchbend(msg.pitch / 8192.0)
        elif kind == 'program_change':
            if msg.program < NUM_PATTERNS:
                self._core.patterns.select(msg.program)
        elif kind == 'clock':
            if self.clock_sync:
                bpm = self._tempo.tick(at)
                if bpm is not None:
                    self._core.set('global.tempo', bpm)
                    self._core._note_change('midi.synced_tempo', bpm)
        elif kind == 'start':
            self._tempo.reset(full=True)
            self._transport('start')
        elif kind == 'stop':
            self._transport('stop')
        elif kind == 'continue':
            self._transport('continue')

    # ------------------------------------------------------------------ CC
    def _on_cc(self, control, value, at):
        core = self._core
        with self._lock:
            learn = self._learn
            if learn is not None:
                self._learn = None
                target = learn[1]
                mapping = {c: t for c, t in self._cc_map.items() if t != target}
                mapping[control] = target
                self._cc_map = mapping
            else:
                target = self._cc_map.get(control)
        if learn is not None:
            self._save_cc_map(mapping)
            core._note_change('midi.cc_map', dict(mapping))
            core._note_change('midi.learning', None)
            self._post_learn(learn, 'done', {'cc': control, 'target': target})
            print(f"MIDI Learn: {cc_name(control)} -> {target}")
            return
        if target is None:
            return

        address = resolve_target(target, core.get('global.channel'))
        entry = core.registry[address]
        position = value / 127.0
        current = core._latest(address)
        current_position = entry.normalize(current)
        with self._lock:
            state = self._pickup.get(address)
            if state is None:
                state = self._pickup[address] = _Pickup()
            state.cc = control
            state.count += 1
            if (state.linked and state.expected is not None
                    and abs(current_position - entry.normalize(state.expected)) > LINK_TOLERANCE):
                state.linked = False  # changed by something else since
            if not state.linked:
                previous = state.physical
                if (abs(position - current_position) <= LINK_TOLERANCE
                        or (previous is not None
                            and (previous - current_position) * (position - current_position) < 0)):
                    state.linked = True
            state.physical = position
            if not state.linked:
                return
            new = entry.denormalize(position)
            state.expected = new
        self._touch_burst(address, current, new, at)
        core.set(address, new)

    # ------------------------------------------------------------------ bursts
    def _touch_burst(self, address, old, new, at):
        with self._lock:
            burst = self._bursts.get(address)
            closed = None
            if burst is not None and at - burst.last > BURST_IDLE_S:
                closed = self._bursts.pop(address)
                burst = None
            if burst is not None:
                burst.new = new
                burst.last = at
        if closed is not None:
            self._post_burst(closed)
        if burst is None:
            snapshot = self._core.legacy_snapshot()  # the state before the burst
            with self._lock:
                self._bursts[address] = _Burst(address, old, new, at, snapshot)

    def close_idle_bursts(self, now):
        """Report the bursts idle for more than 400 ms as undo steps."""
        with self._lock:
            if not self._bursts:
                return
            closed = [b for b in self._bursts.values() if now - b.last > BURST_IDLE_S]
            for burst in closed:
                del self._bursts[burst.address]
        for burst in closed:
            self._post_burst(burst)

    def _post_burst(self, burst):
        entry = self._core.registry[burst.address]
        if abs(entry.normalize(burst.old) - entry.normalize(burst.new)) < 1e-9:
            return  # back where it started: nothing to undo
        self._core._post({'id': None, 'verb': None, 'status': 'done', 'source': 'midi',
                          'kind': 'cc_burst', 'address': burst.address, 'old': burst.old,
                          'new': burst.new, 'snapshot': burst.snapshot})

    # ------------------------------------------------------------------ pitch bend
    def _on_pitchbend(self, bend):
        core = self._core
        target = self._pitchbend_target
        if abs(bend) < PITCHBEND_DEAD_ZONE:
            if self._bend is not None:
                address, original = self._bend
                self._bend = None
                core.set(address, original)
            return
        if target is None:
            return
        if self._bend is None:
            address = resolve_target(target, core.get('global.channel'))
            self._bend = (address, core._latest(address))
        address, original = self._bend
        entry = core.registry[address]
        core.set(address, entry.denormalize(entry.normalize(original) + bend * 0.5))

    # ------------------------------------------------------------------ transport
    # The transport moves into the core with slice 4b; until then MIDI
    # applies it here, at block start, as the buttons do.
    def _transport(self, command):
        pm = self._core.pattern_manager

        def apply(_):
            if command == 'start':
                selected = pm.selected_pattern_index
                pm.stop_playback()
                pm.start_playback(selected)
            elif command == 'stop':
                pm.stop_playback()
            elif not pm.is_playing:  # continue, from where it stopped
                pm.is_playing = True
        self._core.audio.submit_call(apply)

    # ================================================================== poll
    def readout(self):
        """MIDI part of poll(): activity and note counters, pickup state."""
        with self._lock:
            pickup = {address: {'cc': s.cc, 'physical': s.physical, 'linked': s.linked,
                                'count': s.count}
                      for address, s in self._pickup.items()}
        return {'activity': self.activity, 'notes': list(self.notes), 'pickup': pickup}
