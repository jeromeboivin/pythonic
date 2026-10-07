"""
The AI generators of the app core (inventory clusters G and the AI part of
Q): drum patch candidates for the eight channels, AI patterns, and the
preview overlay that tries them on the live channels.

The models run in a **subprocess** (``ai_worker.py``, JSON lines over its
stdin and stdout): nothing in the GUI process imports torch. The worker
starts on first use, is restarted on the next request after it fails, and
stops with the core. Generation never blocks the action thread: the verbs
that need the worker return at once and finish when its reply arrives
(their done or error event comes later through ``poll``).

**Trying candidates.** Each channel has a *lane*: a drum type, the
candidates last generated for it and the current candidate. Trying a
candidate puts it on the live channel (the running pattern plays it) as a
preview overlay: the channel's sound before the first try is kept, and
nothing enters the undo journal. ``ai.keep`` commits the tried sounds as one
undo step; ``ai.revert`` puts the kept sounds back.

- Generating a lane tries its candidate 1 at once; ``ai.try`` with
  ``step=±1`` steps through the candidates (the arrows), ``ai.untry`` takes a
  lane back to its old sound.
- **Edits while trying:** a set of a sound address of a channel that is
  trying a candidate changes the tried sound on the live channel. It is not
  an undo step of its own: ``ai.keep`` commits it with the candidate (one
  step), ``ai.revert`` / ``ai.untry`` drop it. Pattern step sets while an AI
  pattern preview runs are handled the same way (dropped when it ends).
- **Anything else that replaces sounds or patterns ends the trial first:**
  a bulk change of the sounds (preset load, paste, initialize, randomize
  all, drum patch load, program select) or the undo / redo of a step that
  touches a trying channel reverts the tried lanes before it applies; a
  change of patterns (pattern ops, AI randomize, preset load, their undo)
  ends the pattern preview first. Front-ends ask "keep or revert" before
  leaving the AI page; when they do not, the core reverts.
- Known limits: the morph position (when the endpoints differ) and LFO or
  pump modulation of the morph set the sounds of every channel, tried lanes
  included; Edit all that reaches a trying channel from another channel
  journals the tried values.

**Patterns.** ``ai.generate_patterns`` makes a bank of 12 patterns (A-L)
for the kit on the face (tried candidates included). ``ai.pattern_try``
previews: ``loop`` plays the AI version of the playing pattern (the selected
one when stopped), ``bank`` plays all 12 chained from A; without a bank (or
with ``bank=False``) they play the preset's own patterns. Stopping the
transport, or ``ai.pattern_try(mode=None)``, ends the preview and puts the
preset's patterns back. ``ai.replace_patterns`` keeps the tried sounds and
replaces all 12 patterns with the bank (one undo step).
``ai.randomize_pattern`` replaces one pattern, or one channel's lane of it,
with an AI pattern at once (one undo step).

Addresses (read-only unless noted):

- ``ai.available``: the ML extras are installed (checked without importing
  torch). ``ai.install_command``: the command that installs them.
  ``ai.installing``.
- ``ai.state``: ``unavailable``, ``idle``, ``installing``, ``loading``
  (a model) or ``generating``.
- ``ai.models``: ``{'patch': {...}, 'pattern': {...}}``, each ``{'path'
  (the checkpoint used: pref.ai.<kind>_model, else the bundled one, else
  None), 'bundled' (a bundled checkpoint exists), 'status' ('missing',
  'unloaded', 'loading', 'loaded', 'error'), 'error', 'sampling'}``.
- ``ai.ch<N>.type`` (settable enum of the 18 drum types): the lane's drum
  type; until set, the channel's drum type, else the slot default (BD, SD,
  CH, OH, TOM, TOM, CLAP, CY).
- ``ai.ch<N>.candidates`` (n), ``ai.ch<N>.candidate`` (i, 1-based; 0 with
  none), ``ai.ch<N>.name`` (the current candidate's name),
  ``ai.ch<N>.trying``, ``ai.ch<N>.generating``, ``ai.ch<N>.error``.
- ``ai.tried``: the channels trying a candidate (1..8).
- ``ai.bank``: ``none``, ``generating`` or ``ready``. ``ai.preview``:
  ``off``, ``loop`` or ``bank``.

Verbs (channels 1..8, candidates 1..n):

- ``ai.load_model`` (kind ``patch`` / ``pattern``, path=None): load a model
  in the worker; an explicit path is saved as ``pref.ai.<kind>_model`` once
  it loads. ``{'kind', 'path', 'sampling'}``.
- ``ai.generate`` (channel=None for all 8, type=None, temperature=None
  (``pref.ai.patch_temperature``), candidates=8 (1..32), seed=None):
  ``{'channels': [...], 'failed': [{'channel', 'error'}]}`` once every lane
  has its candidates (an error when every lane failed).
- ``ai.try`` (channel, candidate=None (the current one), step=0),
  ``ai.untry`` (channel), ``ai.keep`` (channels=None: every trying lane; a
  listed lane not trying is tried first) -> ``{'kept'}``, ``ai.revert`` ->
  ``{'reverted', 'preview'}``, ``ai.clear`` (revert, then forget the
  candidates, the lane types and the bank).
- ``ai.generate_patterns`` (temperature=None (``pref.ai.pattern_temperature``),
  seed=None) -> ``{'patterns': 12}``; ``ai.clear_patterns``.
- ``ai.pattern_try`` (mode ``'loop'``, ``'bank'`` or None to stop,
  bank=True), ``ai.replace_patterns`` (channels=None, as ``ai.keep``).
- ``ai.randomize_pattern`` (pattern=None, channel=None) -> ``{'pattern',
  'channel'}``.
- ``ai.install``: run the install command in the background ->
  ``{'installed', 'output'}``; the worker uses the new packages at once.
"""

import copy
import gc
import importlib
import importlib.util
import itertools
import json
import os
import subprocess
import sys
import threading

from pythonic.drum_generator import DRUM_TYPES, SLOT_MAP, infer_drum_type
from pythonic.pattern_manager import PatternManager
from pythonic.preset_manager import (apply_drum_patch_to_channel, channel_to_raw_patch,
                                     convert_drum_patch_data)

from .patterns import PATTERN_NAMES, channel_index, lane_addresses, pattern_index
from .presets import channel_addresses
from .registry import Address
from .undo import CHANNELS, PATTERNS, PROGRAMS, Snapshot, pattern_part

NUM_CHANNELS = 8
MAX_CANDIDATES = 32
DEFAULT_CANDIDATES = 8
KINDS = ('patch', 'pattern')

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BUNDLED_MODELS = {
    'patch': os.path.join(_ROOT, 'drum_cvae_best.pt'),
    'pattern': os.path.join(_ROOT, 'drum_patterns', 'pattern_cvae_best.pt'),
}
REQUIREMENTS = os.path.join(_ROOT, 'requirements-ml.txt')
WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ai_worker.py')
NO_EXTRAS = ('the ML extras are not installed (torch): install them with '
             'ai.install, or run: {command}')
NO_PATTERN_MODEL = ("No AI pattern model is available. Set one in the AI settings, "
                    "or place 'pattern_cvae_best.pt' in the drum_patterns/ folder.")
NO_PATCH_MODEL = ("No drum patch model is available. Load one, "
                  "or place 'drum_cvae_best.pt' in the program folder.")

# Drum type labels of patch names -> the model's drum types
_LABEL_TYPES = {'BD': 'bd', 'SD': 'sd', 'CH': 'ch', 'OH': 'oh', 'TOM': 'tom',
                'CLAP': 'clap', 'CY': 'cy', 'PERC': 'perc', 'FX': 'fx'}
LANE_FIELDS = ('type', 'candidates', 'candidate', 'name', 'trying', 'generating', 'error')
_DEFAULT = object()


def torch_installed():
    """True when torch can be imported (found without importing it)."""
    try:
        return importlib.util.find_spec('torch') is not None
    except (ImportError, ValueError):
        return False


# ====================================================================== the worker process
class WorkerProcess:
    """The client side of the worker: a subprocess spoken to in JSON lines.

    ``request(op, callback, **args)`` sends a request; ``callback(reply)``
    runs on the reader thread with the reply dict (``ok``, ``result`` or
    ``error``). If the process ends, every request it had not answered gets
    ``{'ok': False, 'error': 'the AI worker stopped ...'}`` and the next
    request starts a new process.
    """

    def __init__(self, command, on_exit=None):
        self.command = list(command)
        self._on_exit = on_exit
        self._lock = threading.Lock()
        self._proc = None
        self._pending = {}  # request id -> (callback, process)
        self._ids = itertools.count(1)
        self._closing = False
        self.starts = 0  # processes started (tests)

    @property
    def pid(self):
        proc = self._proc
        return proc.pid if proc is not None else None

    def _spawn(self):
        proc = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                text=True, encoding='utf-8', bufsize=1)
        self.starts += 1
        threading.Thread(target=self._read, args=(proc,), name='pythonic-ai-reader',
                         daemon=True).start()
        return proc

    def request(self, op, callback, **args):
        with self._lock:
            if self._closing:
                raise RuntimeError('the AI worker is closed')
            if self._proc is None or self._proc.poll() is not None:
                self._proc = self._spawn()
            proc = self._proc
            rid = next(self._ids)
            self._pending[rid] = (callback, proc)
            try:
                proc.stdin.write(json.dumps({'id': rid, 'op': op, **args}) + '\n')
                proc.stdin.flush()
                return rid
            except (OSError, ValueError):
                failed = self._pending.pop(rid, None)
        if failed is not None:  # the reader has not failed it yet
            callback({'id': rid, 'ok': False, 'error': 'the AI worker stopped'})
        return rid

    def _read(self, proc):
        for line in proc.stdout:
            try:
                reply = json.loads(line)
            except ValueError:
                continue
            with self._lock:
                entry = self._pending.pop(reply.get('id'), None)
            if entry is not None:
                entry[0](reply)
        code = proc.wait()
        with self._lock:
            failed = [rid for rid, (_cb, p) in self._pending.items() if p is proc]
            callbacks = [self._pending.pop(rid)[0] for rid in failed]
            if self._proc is proc:
                self._proc = None
            closing = self._closing
        for rid, callback in zip(failed, callbacks):
            callback({'id': rid, 'ok': False,
                      'error': f'the AI worker stopped (exit code {code})'})
        if self._on_exit is not None and not closing:
            self._on_exit(code, bool(callbacks))

    def close(self, timeout=3.0):
        with self._lock:
            self._closing = True
            proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.stdin.write(json.dumps({'id': 0, 'op': 'quit'}) + '\n')
            proc.stdin.flush()
            proc.stdin.close()
        except (OSError, ValueError):
            pass
        try:
            proc.wait(timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout)


# ====================================================================== lanes
class Lane:
    """One channel's candidates and trial."""

    __slots__ = ('type', 'candidates', 'index', 'original', 'generating', 'error')

    def __init__(self):
        self.type = None          # None: follow the channel's drum type
        self.candidates = []      # raw patch dicts
        self.index = 0
        self.original = None      # the channel's parameters before the trial
        self.generating = None    # token of the request in flight
        self.error = None

    @property
    def trying(self):
        return self.original is not None


class Ai:
    """The AI addresses and verbs of an AppCore."""

    def __init__(self, core, worker=_DEFAULT, install=_DEFAULT):
        self._core = core
        if worker is _DEFAULT:
            self._extras = torch_installed
            worker = [sys.executable, '-u', WORKER_SCRIPT]
        else:
            self._extras = lambda: True  # a given worker brings its own
        self._worker = (WorkerProcess(worker, on_exit=self._on_worker_exit)
                        if worker is not None else None)
        if install is _DEFAULT:
            install = [sys.executable, '-m', 'pip', 'install', '-r', REQUIREMENTS]
        self._install_command = list(install)
        self._available = self._worker is not None and self._extras()
        self._lock = threading.RLock()
        self._tokens = itertools.count(1)
        self._lanes = [Lane() for _ in range(NUM_CHANNELS)]
        self._bank = None         # {letter: pattern dict}
        self._bank_token = None   # the bank request in flight
        self._models = {kind: {'loaded': None, 'sampling': None, 'pending': None,
                               'error': None, 'error_path': None} for kind in KINDS}
        self._inflight = {'generate': 0, 'load': 0}
        self._installing = False
        # Preview overlay: swapped on the audio thread only (block start, stop)
        self._sound_prefixes = ()
        self._saved_patterns = None
        self._preview_mode = 'off'
        self._preview_ended = False

    # ================================================================== interface
    @property
    def worker(self):
        return self._worker

    @property
    def unjournaled(self):
        """Address prefixes whose sets stay out of the undo journal: the
        channels trying a candidate, and the patterns during a swapped
        pattern preview."""
        if self._saved_patterns is not None:
            return self._sound_prefixes + ('pattern.',)
        return self._sound_prefixes

    def register(self, registry):
        reg = registry.register
        reg(Address('ai.available', get=lambda: self._available, kind='bool'))
        reg(Address('ai.install_command', get=self.install_command, kind='str'))
        reg(Address('ai.installing', get=lambda: self._installing, kind='bool'))
        reg(Address('ai.state', get=self._state, kind='enum',
                    labels=('unavailable', 'idle', 'installing', 'loading', 'generating')))
        reg(Address('ai.models', get=self._models_state, kind='json'))
        reg(Address('ai.tried', get=lambda: [i + 1 for i, lane in enumerate(self._lanes)
                                             if lane.trying], kind='list'))
        reg(Address('ai.bank', get=self._bank_state, kind='enum',
                    labels=('none', 'generating', 'ready')))
        reg(Address('ai.preview', get=lambda: self._preview_mode, kind='enum',
                    labels=('off', 'loop', 'bank')))
        for i in range(NUM_CHANNELS):
            prefix = f'ai.ch{i + 1}'
            lane = self._lanes[i]
            reg(Address(f'{prefix}.type', get=lambda i=i: self._lane_type(i),
                        set=lambda v, i=i: self._set_type(i, v), kind='enum',
                        labels=tuple(DRUM_TYPES), queued=False, undoable=False))
            reg(Address(f'{prefix}.candidates', get=lambda l=lane: len(l.candidates),
                        kind='int'))
            reg(Address(f'{prefix}.candidate',
                        get=lambda l=lane: l.index + 1 if l.candidates else 0, kind='int'))
            reg(Address(f'{prefix}.name', get=lambda l=lane: _name(l), kind='str'))
            reg(Address(f'{prefix}.trying', get=lambda l=lane: l.trying, kind='bool'))
            reg(Address(f'{prefix}.generating', get=lambda l=lane: l.generating is not None,
                        kind='bool'))
            reg(Address(f'{prefix}.error', get=lambda l=lane: l.error, kind='str'))

    def verbs(self):
        return {
            'ai.load_model': self._verb_load_model,
            'ai.generate': self._verb_generate,
            'ai.try': self._verb_try,
            'ai.untry': self._verb_untry,
            'ai.keep': self._verb_keep,
            'ai.revert': self._verb_revert,
            'ai.clear': self._verb_clear,
            'ai.generate_patterns': self._verb_generate_patterns,
            'ai.clear_patterns': self._verb_clear_patterns,
            'ai.pattern_try': self._verb_pattern_try,
            'ai.replace_patterns': self._verb_replace_patterns,
            'ai.randomize_pattern': self._verb_randomize_pattern,
            'ai.install': self._verb_install,
        }

    def install_command(self):
        return ' '.join(_quote(part) for part in self._install_command)

    def close(self):
        if self._worker is not None:
            self._worker.close()

    # ================================================================== state readers
    def _state(self):
        if self._installing:
            return 'installing'
        if not self._available:
            return 'unavailable'
        if self._inflight['generate']:
            return 'generating'
        if self._inflight['load']:
            return 'loading'
        return 'idle'

    def _bank_state(self):
        if self._bank_token is not None:
            return 'generating'
        return 'ready' if self._bank is not None else 'none'

    def model_path(self, kind):
        """The checkpoint a kind uses: the preference if the file exists,
        else the bundled one, else None."""
        saved = self._core.get(f'pref.ai.{kind}_model')
        if saved and os.path.isfile(saved):
            return saved
        bundled = BUNDLED_MODELS[kind]
        return bundled if os.path.isfile(bundled) else None

    def _models_state(self):
        state = {}
        for kind in KINDS:
            model = self._models[kind]
            path = self.model_path(kind)
            if path is None:
                status = 'missing'
            elif model['loaded'] == path:
                status = 'loaded'
            elif model['pending'] == path:
                status = 'loading'
            elif model['error_path'] == path:
                status = 'error'
            else:
                status = 'unloaded'
            state[kind] = {'path': path, 'bundled': os.path.isfile(BUNDLED_MODELS[kind]),
                           'status': status,
                           'error': model['error'] if status == 'error' else None,
                           'sampling': model['sampling'] if status == 'loaded' else None}
        return state

    def _lane_type(self, index):
        lane = self._lanes[index]
        if lane.type is not None:
            return lane.type
        label = infer_drum_type(self._core.synth.channels[index].name)
        return _LABEL_TYPES.get(label) or SLOT_MAP[index][1][0]

    def _set_type(self, index, value):
        self._lanes[index].type = value

    # ================================================================== poll
    def collect(self):
        """Report what the audio thread changed (a pattern preview ended by a
        stop). Called by poll and the action thread's monitor."""
        if self._preview_ended:
            self._preview_ended = False
            self._note_patterns()

    def _note_patterns(self):
        names = ['ai.preview']
        for index, name in enumerate(PATTERN_NAMES):
            names += lane_addresses(name) + [f'pattern.{name}.length', f'pattern.{name}.empty',
                                             f'pattern.{name}.chained']
        self._core.note_changes(names)

    def _note_lanes(self, indexes, sounds=False):
        names = ['ai.tried', 'ai.state']
        for i in indexes:
            names += [f'ai.ch{i + 1}.{field}' for field in LANE_FIELDS]
            if sounds:
                names += channel_addresses(i + 1)
        self._core.note_changes(names)

    def _update_prefixes(self):
        self._sound_prefixes = tuple(f'ch{i + 1}.' for i, lane in enumerate(self._lanes)
                                     if lane.trying)

    # ================================================================== worker requests
    def _require(self):
        if not self._available:
            raise RuntimeError(NO_EXTRAS.format(command=self.install_command()))

    def _send(self, op, model_kind, path, handler, **args):
        """Send a request; handler(reply) later runs on the action thread."""
        kind = model_kind
        model = self._models[kind]
        busy = 'load' if op == 'load' else 'generate'
        if model['loaded'] != path:
            model['pending'] = path
        self._inflight[busy] += 1
        core = self._core

        def on_reply(reply):
            core.call_soon(lambda: self._on_reply(kind, path, busy, handler, reply))
        try:
            self._worker.request(op, on_reply, path=path, **args)
        except Exception:
            self._inflight[busy] -= 1
            if model['pending'] == path:
                model['pending'] = None
            raise
        finally:
            core.note_changes(['ai.state', 'ai.models'])

    def _on_reply(self, kind, path, busy, handler, reply):
        with self._lock:
            self._inflight[busy] -= 1
            model = self._models[kind]
            if model['pending'] == path:
                model['pending'] = None
            if reply.get('ok'):
                model['loaded'] = path
                model['error'] = model['error_path'] = None
                sampling = reply['result'].get('sampling')
                if sampling is not None or kind == 'pattern':
                    model['sampling'] = sampling
            elif reply.get('load'):
                model['error'], model['error_path'] = reply.get('error'), path
                if model['loaded'] == path:
                    model['loaded'] = None
            self._core.note_changes(['ai.state', 'ai.models'])
            handler(reply)

    def _on_worker_exit(self, code, had_requests):
        """The worker process ended (reader thread): it forgot its models."""
        def forget():
            for model in self._models.values():
                model['loaded'] = None
            self._core.note_changes(['ai.models'])
            if not had_requests:
                self._core._report_error(f'the AI worker stopped (exit code {code})',
                                         source='ai')
        self._core.call_soon(forget)

    def _finish(self, action_id, verb, result=None, error=None):
        event = {'id': action_id, 'verb': verb}
        if error is not None:
            event.update(status='error', error=error)
        else:
            event.update(status='done', result=result)
        self._core._post(event)

    # ================================================================== trying sounds
    def _apply_candidate(self, index, candidate):
        """Put a lane's candidate on the live channel (block start)."""
        lane = self._lanes[index]
        data = convert_drum_patch_data(lane.candidates[candidate])
        core = self._core

        def apply():
            channel = core.synth.channels[index]
            original = lane.original if lane.original is not None else channel.get_parameters()
            apply_drum_patch_to_channel(channel, data)
            return original
        lane.original = core.at_block_start(apply)
        lane.index = candidate
        self._update_prefixes()

    def _revert_lanes(self, indexes):
        """Put the old sounds of trying lanes back (block start)."""
        items = [(i, copy.deepcopy(self._lanes[i].original)) for i in indexes
                 if self._lanes[i].trying]
        if not items:
            return []
        core = self._core

        def apply():
            for i, original in items:
                core.synth.channels[i].set_parameters(original)
        core.at_block_start(apply)
        for i, _ in items:
            self._lanes[i].original = None
        self._update_prefixes()
        reverted = [i for i, _ in items]
        self._note_lanes(reverted, sounds=True)
        return reverted

    def _lane_indexes(self, channels):
        """Lanes for a channels argument: None = every trying lane."""
        if channels is None:
            return [i for i, lane in enumerate(self._lanes) if lane.trying]
        if isinstance(channels, int):
            channels = [channels]
        indexes = []
        for channel in channels:
            i = channel_index(channel)
            if self._lanes[i].candidates and i not in indexes:
                indexes.append(i)
        return indexes

    def _commit_sounds(self, indexes):
        """Try the listed lanes that are not trying; return the channel
        sounds as they are with the old sound of every trying lane (the
        state before the trial), and end the trial of the listed lanes."""
        for i in indexes:
            if not self._lanes[i].trying:
                self._apply_candidate(i, self._lanes[i].index)
        originals = {i: copy.deepcopy(lane.original) for i, lane in enumerate(self._lanes)
                     if lane.trying}
        return originals

    def _end_trial(self, indexes):
        for i in indexes:
            self._lanes[i].original = None
        self._update_prefixes()

    # ================================================================== the release hook
    def release(self, parts=(), names=()):
        """Called by the undo module before a snapshot (``parts``) or the
        replay of journaled sets (``names``): end the trial of the sounds
        and the pattern preview they would overwrite."""
        sounds = any(p in (CHANNELS, PROGRAMS) for p in parts)
        patterns = any(p == PATTERNS or str(p).startswith('pattern:') for p in parts)
        if names:
            prefixes = self._sound_prefixes
            sounds = sounds or (bool(prefixes) and any(n.startswith(prefixes) for n in names))
            patterns = patterns or any(n.startswith('pattern.') for n in names)
        if not (sounds and self._sound_prefixes) and not (patterns and self._saved_patterns):
            return
        with self._lock:
            if sounds:
                self._revert_lanes(range(NUM_CHANNELS))
            if patterns and self._saved_patterns is not None:
                self._core.at_block_start(self._end_preview)
                self.collect()

    # ================================================================== pattern preview
    def on_stop(self):
        """The transport stopped (audio thread): end the pattern preview."""
        if self._preview_mode != 'off':
            self._end_preview()

    def _end_preview(self):
        """Put the preset's patterns back (audio thread)."""
        saved = self._saved_patterns
        if saved is not None:
            self._core.pattern_manager.patterns = saved
            self._saved_patterns = None
        if self._preview_mode != 'off':
            self._preview_mode = 'off'
            self._preview_ended = True

    def _preset_patterns(self):
        """The preset's patterns: the saved ones during a swapped preview."""
        saved = self._saved_patterns
        return saved if saved is not None else self._core.pattern_manager.patterns

    @staticmethod
    def _apply_bank(patterns, bank):
        """Copies of `patterns` with the bank written over them."""
        temp = PatternManager(num_channels=NUM_CHANNELS)
        temp.patterns = [p.copy() for p in patterns]
        temp.apply_pattern_bank(bank)
        return temp.patterns

    # ================================================================== verbs: models
    def _verb_load_model(self, kind, path=None):
        self._require()
        if kind not in KINDS:
            raise ValueError(f'not a model kind: {kind!r} (patch or pattern)')
        explicit = path is not None
        if explicit:
            path = os.path.abspath(os.path.expanduser(str(path)))
            if not os.path.isfile(path):
                raise ValueError(f'no such model file: {path}')
        else:
            path = self.model_path(kind)
            if path is None:
                raise ValueError(NO_PATCH_MODEL if kind == 'patch' else NO_PATTERN_MODEL)
        action_id = self._core._running_action

        def done(reply):
            if not reply.get('ok'):
                self._finish(action_id, 'ai.load_model', error=reply.get('error'))
                return
            if explicit:
                self._core.set(f'pref.ai.{kind}_model', path)
                self._core.note_changes(['ai.models'])
            self._finish(action_id, 'ai.load_model',
                         {'kind': kind, 'path': path,
                          'sampling': reply['result'].get('sampling')})
        with self._lock:
            self._send('load', kind, path, done, kind=kind)
        return self._core.DEFERRED

    # ================================================================== verbs: patches
    def _verb_generate(self, channel=None, type=None, temperature=None,
                       candidates=DEFAULT_CANDIDATES, seed=None):
        self._require()
        if channel is None:
            if type is not None:
                raise ValueError('a drum type needs one channel')
            indexes = list(range(NUM_CHANNELS))
        else:
            indexes = [channel_index(channel)]
        if type is not None and type not in DRUM_TYPES:
            raise ValueError(f'not a drum type: {type!r} ({", ".join(DRUM_TYPES)})')
        if temperature is None:
            temperature = self._core.get('pref.ai.patch_temperature')
        temperature = min(3.0, max(0.1, float(temperature)))
        n = min(MAX_CANDIDATES, max(1, int(candidates)))
        seed = None if seed is None else int(seed)
        path = self.model_path('patch')
        if path is None:
            raise ValueError(NO_PATCH_MODEL)
        action_id = self._core._running_action
        job = {'remaining': len(indexes), 'channels': [], 'failed': []}
        with self._lock:
            if type is not None:
                self._lanes[indexes[0]].type = type
            for i in indexes:
                token = next(self._tokens)
                lane = self._lanes[i]
                lane.generating = token
                lane.error = None
                self._send('patches', 'patch', path,
                           lambda reply, i=i, token=token: self._on_patches(
                               action_id, job, i, token, reply),
                           drum_type=self._lane_type(i), n=n, temperature=temperature,
                           seed=seed)
            self._note_lanes(indexes)
        return self._core.DEFERRED

    def _on_patches(self, action_id, job, index, token, reply):
        lane = self._lanes[index]
        channel = index + 1
        if lane.generating != token:
            job['failed'].append({'channel': channel, 'error': 'superseded'})
        elif not reply.get('ok'):
            lane.generating = None
            lane.error = reply.get('error')
            job['failed'].append({'channel': channel, 'error': lane.error})
        else:
            lane.generating = None
            lane.candidates = list(reply['result']['candidates'])
            lane.index = 0
            try:
                if lane.candidates:
                    self._apply_candidate(index, 0)
                job['channels'].append(channel)
            except Exception as exc:
                lane.error = str(exc) or type(exc).__name__
                job['failed'].append({'channel': channel, 'error': lane.error})
        self._note_lanes([index], sounds=True)
        job['remaining'] -= 1
        if job['remaining'] == 0:
            if job['failed'] and not job['channels']:
                error = '; '.join(f"channel {f['channel']}: {f['error']}" for f in job['failed'])
                self._finish(action_id, 'ai.generate', error=error)
            else:
                self._finish(action_id, 'ai.generate',
                             {'channels': sorted(job['channels']), 'failed': job['failed']})

    def _verb_try(self, channel, candidate=None, step=0):
        index = channel_index(channel)
        with self._lock:
            lane = self._lanes[index]
            if not lane.candidates:
                raise ValueError(f'channel {channel} has no candidates')
            n = len(lane.candidates)
            if candidate is None:
                current = lane.index
            elif isinstance(candidate, int) and not isinstance(candidate, bool) \
                    and 1 <= candidate <= n:
                current = candidate - 1
            else:
                raise ValueError(f'not a candidate: {candidate!r} (1-{n})')
            current = (current + int(step)) % n
            self._apply_candidate(index, current)
            self._note_lanes([index], sounds=True)
            return {'channel': channel, 'candidate': current + 1, 'name': _name(lane)}

    def _verb_untry(self, channel):
        index = channel_index(channel)
        with self._lock:
            reverted = self._revert_lanes([index])
        return {'channel': channel, 'reverted': bool(reverted)}

    def _verb_keep(self, channels=None):
        with self._lock:
            indexes = self._lane_indexes(channels)
            if not indexes:
                return {'kept': []}
            originals = self._commit_sounds(indexes)
            core = self._core
            before = core.at_block_start(lambda: core.undo.capture((CHANNELS,)))
            for i, original in originals.items():
                before[CHANNELS][i] = original
            core.undo.journal.push(Snapshot((CHANNELS,), before, 'keep AI sounds'))
            self._end_trial(indexes)
            self._note_lanes(indexes, sounds=True)
            gc.freeze()
            return {'kept': [i + 1 for i in indexes]}

    def _verb_revert(self):
        with self._lock:
            reverted = self._revert_lanes(range(NUM_CHANNELS))
            preview = self._preview_mode != 'off'
            if preview:
                self._core.at_block_start(self._core.patterns._apply_stop)
                self.collect()
            return {'reverted': [i + 1 for i in reverted], 'preview': preview}

    def _verb_clear(self):
        with self._lock:
            result = self._verb_revert()
            for lane in self._lanes:
                lane.type = None
                lane.candidates = []
                lane.index = 0
                lane.error = None
                lane.generating = None
            self._bank = None
            self._bank_token = None
            self._note_lanes(range(NUM_CHANNELS))
            self._core.note_changes(['ai.bank'])
            return result

    # ================================================================== verbs: patterns
    def _kit(self):
        """The kit on the face and the timing, for the pattern model."""
        core = self._core
        pm = core.pattern_manager

        def read():
            return ([channel_to_raw_patch(ch) for ch in core.synth.channels],
                    float(pm.bpm), float(pm.swing), float(pm.fill_rate), pm.step_rate)
        return core.at_block_start(read)

    def _pattern_model(self):
        self._require()
        path = self.model_path('pattern')
        if path is None:
            raise ValueError(NO_PATTERN_MODEL)
        return path

    def _verb_generate_patterns(self, temperature=None, seed=None):
        path = self._pattern_model()
        if temperature is None:
            temperature = self._core.get('pref.ai.pattern_temperature')
        temperature = min(3.0, max(0.1, float(temperature)))
        raw, tempo, _swing, fill_rate, step_rate = self._kit()
        action_id = self._core._running_action
        with self._lock:
            token = self._bank_token = next(self._tokens)

            def done(reply):
                if self._bank_token != token:
                    self._finish(action_id, 'ai.generate_patterns', error='superseded')
                    return
                self._bank_token = None
                if reply.get('ok'):
                    self._bank = dict(zip(PATTERN_NAMES, reply['result']['patterns']))
                    self._finish(action_id, 'ai.generate_patterns',
                                 {'patterns': len(self._bank)})
                else:
                    self._finish(action_id, 'ai.generate_patterns', error=reply.get('error'))
                self._core.note_changes(['ai.bank'])
            # The bank is generated without swing (as the generator dialog always did)
            self._send('patterns', 'pattern', path, done, raw_patches=raw, tempo=tempo,
                       swing=0.0, fill_rate=fill_rate, step_rate=step_rate,
                       n=len(PATTERN_NAMES), temperature=temperature,
                       seed=None if seed is None else int(seed))
            self._core.note_changes(['ai.bank'])
        return self._core.DEFERRED

    def _verb_clear_patterns(self):
        with self._lock:
            self._bank = None
            self._bank_token = None
            self._core.note_changes(['ai.bank'])
        return {'cleared': True}

    def _verb_pattern_try(self, mode=None, bank=True):
        if mode in (None, 'off'):
            mode = 'off'
        elif mode not in ('loop', 'bank'):
            raise ValueError(f'not a preview mode: {mode!r} (loop, bank or None)')
        core = self._core
        pm = core.pattern_manager
        patterns = core.patterns
        with self._lock:
            if mode == 'off':
                if self._preview_mode != 'off':
                    core.at_block_start(patterns._apply_stop)
                self.collect()
                return {'preview': 'off'}
            use_bank = bool(bank) and self._bank is not None
            core.at_block_start(lambda: None)  # queued sets land first
            base = self._preset_patterns()
            index = pm.playing_pattern_index if pm.is_playing else pm.selected_pattern_index
            new = None
            if mode == 'loop' and use_bank:
                new = list(base)
                temp = PatternManager(num_channels=NUM_CHANNELS)
                temp.patterns = new
                new[index] = base[index].copy()
                temp.apply_single_pattern(index, self._bank[PATTERN_NAMES[index]])
            elif mode == 'bank':
                new = self._apply_bank(base, self._bank) if use_bank else [p.copy() for p in base]
                for i, pattern in enumerate(new):
                    pattern.chained_to_next = i < len(new) - 1
                    pattern.chained_from_prev = i > 0

            def apply():
                if self._saved_patterns is not None:
                    pm.patterns = self._saved_patterns
                    self._saved_patterns = None
                if new is not None:
                    self._saved_patterns = pm.patterns
                    pm.patterns = new
                self._preview_mode = mode
                if mode == 'bank':
                    pm.select_pattern(0)
                    patterns._apply_play(0)
                elif not pm.is_playing:
                    patterns._apply_play(index)
            core.at_block_start(apply)
            self._note_patterns()
            gc.freeze()
            return {'preview': mode, 'pattern': 'A' if mode == 'bank' else PATTERN_NAMES[index],
                    'bank': use_bank}

    def _verb_replace_patterns(self, channels=None):
        core = self._core
        pm = core.pattern_manager
        with self._lock:
            if self._bank is None:
                raise ValueError('no AI patterns: generate them first')
            indexes = self._lane_indexes(channels)
            originals = self._commit_sounds(indexes)
            core.at_block_start(lambda: None)
            new = self._apply_bank(self._preset_patterns(), self._bank)

            def apply():
                before = core.undo.capture((CHANNELS,))
                before[PATTERNS] = self._preset_patterns()  # no longer referenced by pm
                self._saved_patterns = None
                if self._preview_mode != 'off':
                    self._preview_mode = 'off'
                pm.patterns = new
                return before
            before = core.at_block_start(apply)
            for i, original in originals.items():
                before[CHANNELS][i] = original
            core.undo.journal.push(Snapshot((CHANNELS, PATTERNS), before,
                                            'AI sounds and patterns'))
            self._end_trial(indexes)
            self._note_lanes(indexes, sounds=True)
            self._note_patterns()
            gc.freeze()
            return {'kept': [i + 1 for i in indexes], 'patterns': len(PATTERN_NAMES)}

    def _verb_randomize_pattern(self, pattern=None, channel=None):
        core = self._core
        index = pattern_index(pattern, core.pattern_manager.selected_pattern_index)
        ch = None if channel is None else channel_index(channel)
        path = self._pattern_model()
        raw, tempo, swing, fill_rate, step_rate = self._kit()
        temperature = core.get('pref.ai.pattern_temperature')
        action_id = core._running_action
        verb = 'ai.randomize_pattern'

        def done(reply):
            if not reply.get('ok'):
                self._finish(action_id, verb, error=reply.get('error'))
                return
            data = reply['result']['patterns'][0]
            pm = core.pattern_manager
            try:
                if ch is None:
                    core.undo.snapshot_op((pattern_part(index),), 'AI pattern',
                                          lambda: pm.apply_single_pattern(index, data))
                else:
                    core.undo.snapshot_op((pattern_part(index),), 'AI channel',
                                          lambda: pm.apply_single_channel(index, ch, data))
            except Exception as exc:
                self._finish(action_id, verb, error=str(exc) or type(exc).__name__)
                return
            core.patterns._note_pattern(index)
            self._finish(action_id, verb, {'pattern': PATTERN_NAMES[index],
                                           'channel': None if ch is None else ch + 1})
        with self._lock:
            self._send('patterns', 'pattern', path, done, raw_patches=raw, tempo=tempo,
                       swing=swing, fill_rate=fill_rate, step_rate=step_rate, n=1,
                       temperature=temperature, seed=None)
        return core.DEFERRED

    # ================================================================== verbs: install
    def _verb_install(self):
        if self._installing:
            raise RuntimeError('the ML extras are already being installed')
        core = self._core
        action_id = core._running_action
        command = list(self._install_command)
        self._installing = True
        core.note_changes(['ai.installing', 'ai.state'])

        def run():
            try:
                proc = subprocess.run(command, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True)
                code, output = proc.returncode, proc.stdout
            except Exception as exc:  # pip missing, ...
                code, output = -1, f'Install failed: {exc}'
            core.call_soon(lambda: finish(code, output))

        def finish(code, output):
            self._installing = False
            importlib.invalidate_caches()
            self._available = self._worker is not None and self._extras()
            core.note_changes(['ai.installing', 'ai.state', 'ai.available'])
            tail = '\n'.join(output.splitlines()[-10:])
            if code == 0:
                self._finish(action_id, 'ai.install', {'installed': True, 'output': tail})
            else:
                self._finish(action_id, 'ai.install', error=f'the install failed:\n{tail}')
        threading.Thread(target=run, name='pythonic-ai-install', daemon=True).start()
        return core.DEFERRED


def _name(lane):
    if not lane.candidates:
        return ''
    return str(lane.candidates[lane.index].get('Name', ''))


def _quote(part):
    return f'"{part}"' if ' ' in part else part
