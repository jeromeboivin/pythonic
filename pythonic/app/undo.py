"""
Undo and redo of the app core (ADR 0001, inventory cluster H).

The journal holds up to 50 steps. A step is either

- a list of **entries** ``(address, old, new)``: what ``AppCore.set`` changed.
  One set is one step, unless it joins a **gesture** (a drag or a paint
  stroke a front-end brackets with ``begin_gesture()`` / ``end_gesture()``)
  or a **burst** (``set(..., burst=True)``: a wheel or a MIDI controller; the
  changes of one address are one step, closed after 400 ms without a change
  on it). A step that ends where it started is dropped. Entries are tiny: no
  deep copies on the caller's thread for ordinary edits.
- a **snapshot** of part of the preset, for bulk actions: ``bulk_change()``
  (code that has not moved into the core yet and writes the engine itself:
  preset load, AI apply, PO-32 import, ...) and the verbs that rewrite many
  values (pattern ops, program switch, morph capture).

Covered: every edit saved in the preset (sounds, globals, morph, programs,
patterns). Not covered: transport, selection, mutes and modes (addresses with
``undoable=False``). Morph-derived channel values are not journaled: undo
restores the morph position and the values follow.

Verbs ``undo`` and ``redo`` return ``{'done': bool, 'label': str}``; they apply
at block start. ``undo.can_undo`` / ``undo.can_redo`` are reported by poll.
"""

import collections
import contextlib
import gc
import threading
from typing import Any, Dict, List, Optional, Sequence

from pythonic.pattern_manager import PatternManager

from .registry import Address
from .sound import SOUND_SUFFIXES

DEPTH = 50
BURST_IDLE_S = 0.4
NUM_CHANNELS = 8
PATTERN_NAMES = tuple(PatternManager.PATTERN_NAMES)

# Parts of the preset a snapshot can hold. ``pattern:<index>`` is one pattern.
CHANNELS, GLOBALS, PROGRAMS, MORPH, PATTERNS = 'channels', 'globals', 'programs', 'morph', 'patterns'
PRESET = (CHANNELS, GLOBALS, PROGRAMS, MORPH, PATTERNS)

_SKIP = object()  # an entry's ``new`` when redo leaves it to the entry after it


class Entry:
    """One journaled set: ``names`` (the address, plus the channels Edit all
    reached, filled in by the audio thread when it applies the set), their
    values before it (``olds``) and the value set (``new``). A companion entry
    (``new`` is _SKIP) only restores what the set after it may lose: the
    accent and fill a trigger-off clears, the steps a shorter pattern drops."""

    __slots__ = ('key', 'names', 'olds', 'new', 'seq')

    def __init__(self, key, names, olds, new, seq=0):
        self.key = key
        self.names = names
        self.olds = olds
        self.new = new
        self.seq = seq

    @property
    def companion(self):
        return self.new is _SKIP


class Changes:
    """A step of entries (one set, a gesture or a burst)."""

    __slots__ = ('entries', 'label')

    def __init__(self, entries):
        self.entries = entries
        self.label = next((e.key for e in entries if not e.companion), entries[0].key)


class Snapshot:
    """A step of a bulk change: the parts of the preset before it (after it,
    once undone)."""

    __slots__ = ('parts', 'state', 'label')

    def __init__(self, parts, state, label):
        self.parts = tuple(parts)
        self.state = state
        self.label = label


def _same(a, b):
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(a)))
    except (TypeError, ValueError):
        return False


class Journal:
    """Undo and redo stacks with the open gesture and bursts (any thread but
    the audio thread). ``on_state(flags)`` is called (outside the lock) with
    the flags that changed (``{'can_undo': True}``)."""

    def __init__(self, clock, depth=DEPTH, idle=BURST_IDLE_S, on_state=None):
        self._clock = clock
        self._depth = depth
        self._idle = idle
        self._on_state = on_state
        self._lock = threading.Lock()
        self._undo = collections.deque()
        self._redo: List = []
        self._gesture: Optional[List[Entry]] = None
        self._bursts: Dict[Any, list] = {}  # key -> [entries, last change time]
        self._suppressed = 0
        self._reported = (False, False)

    # ------------------------------------------------------------------ recording
    @property
    def recording(self):
        return self._suppressed == 0

    def record(self, entries, burst=None, step=False):
        """Journal the entries of one set (companions first). ``burst`` is the
        key of the burst it belongs to, if any; ``step=True`` makes them a step
        of their own whatever is open."""
        now = self._clock()
        with self._lock:
            if self._suppressed:
                return
            alone = step or (burst is None and self._gesture is None)
            if alone and self._unchanged(entries):
                return  # a set to the value it has (an echo): no step, redo kept
            self._redo.clear()
            if alone:
                self._commit(list(entries))
            elif burst is not None:
                group = self._bursts.get(burst)
                if group is not None and now - group[1] > self._idle:
                    del self._bursts[burst]
                    self._commit(group[0])
                    group = None
                if group is None:
                    group = self._bursts[burst] = [[], now]
                self._append(group[0], entries)
                group[1] = now
            else:
                self._append(self._gesture, entries)
        self._report()

    @staticmethod
    def _append(group, entries):
        main = entries[-1]
        if group:
            last = group[-1]
            if (last.key == main.key and not last.companion and isinstance(last.names, tuple)
                    and isinstance(main.names, tuple) and last.names == main.names):
                last.new = main.new  # the same control again: keep the first old value
                last.seq = main.seq
                return
        group.extend(entries)

    def _commit(self, entries):
        """Push a finished group of entries (lock held)."""
        if not entries or self._unchanged(entries):
            return
        self._push(Changes(entries))

    @staticmethod
    def _unchanged(entries):
        """True when a group ends where it started (no companions involved)."""
        first, last = {}, {}
        for entry in entries:
            if entry.companion:
                return False
            first.setdefault(entry.key, entry.olds[entry.key])
            last[entry.key] = entry.new
        return all(_same(first[key], last[key]) for key in first)

    def _push(self, step):
        self._undo.append(step)
        while len(self._undo) > self._depth:
            self._undo.popleft()

    def push(self, step):
        """Journal a snapshot step (after the open gesture and bursts)."""
        with self._lock:
            if self._suppressed:
                return
            self._flush()
            self._redo.clear()
            self._push(step)
        self._report()

    def begin_gesture(self):
        with self._lock:
            if self._gesture:
                self._commit(self._gesture)
            self._gesture = []

    def end_gesture(self):
        with self._lock:
            gesture, self._gesture = self._gesture, None
            if gesture:
                self._commit(gesture)
        self._report()

    def close_idle(self):
        """Commit the bursts idle for more than 400 ms."""
        now = self._clock()
        with self._lock:
            if not self._bursts:
                return
            idle = [k for k, (_, last) in self._bursts.items() if now - last > self._idle]
            for key in idle:
                self._commit(self._bursts.pop(key)[0])
        if idle:
            self._report()

    def _flush(self):
        """Commit the open gesture (it goes on as a new step) and every burst."""
        for key in list(self._bursts):
            self._commit(self._bursts.pop(key)[0])
        if self._gesture:
            self._commit(self._gesture)
            self._gesture = []

    @contextlib.contextmanager
    def suppressed(self):
        """Sets inside are not journaled (a bulk change records them as one)."""
        with self._lock:
            self._flush()
            self._suppressed += 1
        try:
            yield
        finally:
            with self._lock:
                self._suppressed -= 1

    # ------------------------------------------------------------------ undo / redo
    def take(self, undo=True):
        """Move the newest undo (or redo) step to the other stack and return it."""
        with self._lock:
            self._flush()
            source, target = (self._undo, self._redo) if undo else (self._redo, self._undo)
            step = source.pop() if source else None
            if step is not None:
                if undo:
                    target.append(step)
                else:
                    self._push(step)
        self._report()
        return step

    def give_back(self, step, undo=True):
        """Put back a step whose undo (or redo) failed."""
        with self._lock:
            source, target = (self._redo, self._undo) if undo else (self._undo, self._redo)
            if source and source[-1] is step:
                source.pop()
                target.append(step)
        self._report()

    def state(self):
        with self._lock:
            return self._flags()

    def _flags(self):
        pending = bool(self._gesture) or any(g[0] for g in self._bursts.values())
        return (bool(self._undo) or pending, bool(self._redo))

    def _report(self):
        with self._lock:
            flags = self._flags()
            old, self._reported = self._reported, flags
        if flags != old and self._on_state is not None:
            self._on_state({name: flag for name, flag, was in
                            zip(('can_undo', 'can_redo'), flags, old) if flag != was})


class Undo:
    """The undo journal of an AppCore: recording, snapshots, verbs."""

    def __init__(self, core, clock):
        self._core = core
        self.journal = Journal(clock, on_state=self._note_state)

    # ================================================================== interface
    def register(self, registry):
        reg = registry.register
        reg(Address('undo.can_undo', get=lambda: self.journal.state()[0], kind='bool'))
        reg(Address('undo.can_redo', get=lambda: self.journal.state()[1], kind='bool'))

    def verbs(self):
        return {'undo': self._verb_undo, 'redo': self._verb_redo}

    def _note_state(self, flags):
        self._core.note_values({f'undo.{name}': flag for name, flag in flags.items()})

    # ================================================================== recording
    def record_set(self, address, names, olds, value, companions, seq, burst):
        """Journal one set from AppCore.set (companions: (name, old) pairs)."""
        entries = [Entry(name, (name,), {name: old}, _SKIP) for name, old in companions]
        entries.append(Entry(address, names, olds, value, seq))
        self.journal.record(entries, burst=address if burst else None)

    def record_changes(self, changes):
        """Journal the changes a verb made (chain toggles) as one step;
        ``changes`` maps addresses to (old, new)."""
        entries = [Entry(a, (a,), {a: old}, new) for a, (old, new) in changes.items()]
        if entries:
            self.journal.record(entries, step=True)

    @contextlib.contextmanager
    def bulk_change(self, label, parts=PRESET):
        """One undo step for everything changed inside, reported by poll.

        The state before is captured on the calling thread; sets made inside
        are part of the step, not steps of their own."""
        parts = tuple(parts)
        before = self.capture(parts)
        try:
            with self.journal.suppressed():
                yield
        finally:
            self.journal.push(Snapshot(parts, before, label))
            self._core.note_changes(self.part_addresses(parts))
            if PATTERNS in parts:
                gc.freeze()

    def snapshot_op(self, parts, label, op):
        """Run op() at block start as one snapshot step (verbs that rewrite
        many values: pattern ops, program switch, morph capture). Returns
        op's result."""
        parts = tuple(parts)
        box = {}

        def apply():
            box['before'] = self.capture(parts)
            return op()
        result = self._core.at_block_start(apply)
        if 'before' in box:
            self.journal.push(Snapshot(parts, box['before'], label))
        return result

    # ================================================================== verbs
    def _verb_undo(self):
        return self._replay(undo=True)

    def _verb_redo(self):
        return self._replay(undo=False)

    def _replay(self, undo):
        step = self.journal.take(undo)
        if step is None:
            return {'done': False}
        try:
            if isinstance(step, Changes):
                self._replay_changes(step, undo)
            else:
                self._replay_snapshot(step)
        except Exception:
            self.journal.give_back(step, undo)
            raise
        return {'done': True, 'label': step.label}

    def _replay_changes(self, step, undo):
        core = self._core
        entries = step.entries
        # The sets of the step have been applied (and Edit all has filled in
        # the channels it reached) once their queue items are drained
        last = max(e.seq for e in entries)
        if last and not core.audio.wait_applied(last, core.audio.stream_timeout):
            raise TimeoutError('the audio stream did not apply the change in time')
        registry = core.registry
        ops = []
        if undo:
            for entry in reversed(entries):
                for name in reversed(list(entry.names)):
                    ops.append((registry[name].set, entry.olds[name]))
        else:
            for entry in entries:
                if not entry.companion:
                    for name in entry.names:
                        ops.append((registry[name].set, entry.new))
        names = []
        for entry in entries:
            for name in entry.names:
                names.append(name)
                names.extend(registry[name].related)

        def apply():
            for setter, value in ops:
                setter(value)
        core.at_block_start(apply)
        core.note_changes(list(dict.fromkeys(names)))

    def _replay_snapshot(self, step):
        core = self._core
        parts = step.parts
        target = step.state
        if PATTERNS in parts:
            # Copying every pattern is too heavy for block start: let the
            # queued sets land, then capture the current state off the audio thread
            core.at_block_start(lambda: None)
            current = self.capture(parts)
            core.at_block_start(lambda: self.restore(parts, target))
            gc.freeze()
        else:
            def swap():
                captured = self.capture(parts)
                self.restore(parts, target)
                return captured
            current = core.at_block_start(swap)
        step.state = current
        core.note_changes(self.part_addresses(parts))

    # ================================================================== snapshots
    def capture(self, parts):
        """The parts of the preset as plain data no live object shares."""
        core = self._core
        synth, pm, morph = core.synth, core.pattern_manager, core.morph_manager
        state = {}
        for part in parts:
            if part == CHANNELS:
                state[part] = [ch.get_parameters() for ch in synth.channels]
            elif part == GLOBALS:
                state[part] = (pm.bpm, pm.swing, pm.step_rate, pm.fill_rate,
                               synth.master_volume_db)
            elif part == PROGRAMS:
                # Program slots are replaced whole, never edited in place
                state[part] = (list(synth._programs), synth._current_program)
            elif part == MORPH:
                state[part] = morph.to_dict()
            elif part == PATTERNS:
                state[part] = [p.copy() for p in pm.patterns]
            else:
                state[part] = pm.patterns[_pattern_part(part)].copy()
        return state

    def restore(self, parts, state):
        """Put captured parts back (at block start). The captured objects are
        installed as they are: a snapshot is restored once."""
        core = self._core
        synth, pm, morph = core.synth, core.pattern_manager, core.morph_manager
        for part in parts:
            data = state[part]
            if part == CHANNELS:
                for channel, params in zip(synth.channels, data):
                    channel.set_parameters(params)
            elif part == GLOBALS:
                bpm, swing, step_rate, fill_rate, master = data
                pm.set_bpm(bpm)
                synth.set_bpm(bpm)
                pm.set_swing(swing)
                pm.set_step_rate(step_rate)
                pm.set_fill_rate(fill_rate)
                synth.set_master_volume(master)
            elif part == PROGRAMS:
                synth._programs, synth._current_program = list(data[0]), data[1]
            elif part == MORPH:
                morph.from_dict(data)
                if CHANNELS not in parts and not morph.is_learning():
                    morph.apply_effective_position()  # the channels follow the morph
            elif part == PATTERNS:
                pm.patterns = data
            else:
                pm.patterns[_pattern_part(part)] = data

    def part_addresses(self, parts):
        """The addresses whose values a snapshot of these parts holds."""
        names = []
        for part in parts:
            if part in (CHANNELS, MORPH):
                for c in range(1, NUM_CHANNELS + 1):
                    names += [f'ch{c}.{suffix}' for suffix in sorted(SOUND_SUFFIXES)]
                    names.append(f'ch{c}.name')
                    if part == CHANNELS:
                        names.append(f'ch{c}.mute')
                if part == MORPH:
                    names += self._core.registry.names('morph.')
            elif part == GLOBALS:
                names += ['global.tempo', 'global.swing', 'global.step_rate',
                          'global.fill_rate', 'global.master']
            elif part == PROGRAMS:
                names += self._core.registry.names('program.')
            elif part == PATTERNS:
                for index in range(len(PATTERN_NAMES)):
                    names += _pattern_addresses(index)
            else:
                names += _pattern_addresses(_pattern_part(part))
        return list(dict.fromkeys(names))


def pattern_part(index):
    """The snapshot part of one pattern (0..11)."""
    return f'pattern:{index}'


def _pattern_part(part):
    return int(part.split(':', 1)[1])


def _pattern_addresses(index):
    from .patterns import lane_addresses
    name = PATTERN_NAMES[index]
    names = lane_addresses(name) + [f'pattern.{name}.length', f'pattern.{name}.empty']
    names += [f'pattern.{PATTERN_NAMES[i]}.chained' for i in (index - 1, index)
              if 0 <= i < len(PATTERN_NAMES)]
    return names
