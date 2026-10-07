"""
A stand-in app core for page tests, built from the real core's describe()
data so its addresses and metadata cannot drift from the real ones.

``core_table()`` builds a real AppCore once (fake audio, no MIDI, a temporary
preferences folder) and returns ``{'describe': {addr: metadata}, 'values':
{addr: value}}`` for every registered address. ``FakeCore(table)`` serves
get/set/describe/act/poll from it: sets apply at once (coerced as the real
core does) and show in the next poll; verbs are recorded and answered from
``verbs`` (transport verbs flip ``transport``); ``calls`` records every set,
verb and gesture.

Pattern steps and lanes (resolved lazily by the real core, so not in the
table) are kept by ``FakePatterns``: step sets apply the real rules (turning
a trigger off clears accent and fill, steps past the length are ignored) and
report the lanes, as the real core's poll does; ``pattern.<P>.length`` resizes
the lanes. ``pattern.select`` moves ``transport['selected_pattern']``.
"""

import itertools
import re
import tempfile
from unittest import mock

from pythonic.app.registry import Address

PREF_UI = 'pref.ui.'
PATTERN_NAMES = 'ABCDEFGHIJKL'
STEP_ADDRESS = re.compile(
    r'pattern\.([A-L])\.ch([1-8])\.(?:step([1-9][0-9]?)\.)?(trig|acc|vel|fill|prob|sub)$')
STEP_FIELDS = {  # field: (kind, default, minimum, maximum, unit)
    'trig': ('bool', False, None, None, ''), 'acc': ('bool', False, None, None, ''),
    'vel': ('int', 64, 1, 127, ''), 'fill': ('bool', False, None, None, ''),
    'prob': ('int', 100, 0, 100, '%'), 'sub': ('str', '', None, None, ''),
}


def core_table():
    """describe() and get() of every registered address of a real core."""
    from pythonic.app import AppCore
    from pythonic.preferences_manager import PreferencesManager
    from tests.fake_audio import FakeAudioBackend

    with tempfile.TemporaryDirectory() as folder:
        with mock.patch.object(PreferencesManager, '_get_config_dir', lambda self: folder), \
                mock.patch.object(PreferencesManager, '_get_default_preset_folder',
                                  lambda self: f'{folder}/presets'):
            prefs = PreferencesManager()
            prefs.set('midi_enabled', False)
            core = AppCore(preferences=prefs, audio_backend=FakeAudioBackend(),
                           midi_backend=None, stall_timeout=None)
            try:
                names = core.registry.names()
                return {'describe': {name: core.describe(name) for name in names},
                        'values': {name: core.get(name) for name in names}}
            finally:
                core.close()


class _Names:
    def __init__(self, meta):
        self._meta = meta

    def names(self, prefix=''):
        return sorted(name for name in self._meta if name.startswith(prefix))


class FakePatterns:
    """The step lanes of the 12 patterns, with the real core's step rules."""

    def __init__(self, core):
        self.core = core
        self.lanes = {}  # (letter, channel, field) -> list

    def length(self, letter):
        return self.core.values[f'pattern.{letter}.length']

    def lane(self, letter, channel, field):
        key = (letter, channel, field)
        n = self.length(letter)
        lane = self.lanes.setdefault(key, [STEP_FIELDS[field][1]] * n)
        if len(lane) != n:  # the length changed
            lane[:] = (lane + [STEP_FIELDS[field][1]] * n)[:n]
        return lane

    @staticmethod
    def match(address):
        m = STEP_ADDRESS.match(address)
        if m is None:
            return None
        letter, channel, step, field = m.groups()
        if step is not None and not 1 <= int(step) <= 64:
            return None
        return letter, int(channel), None if step is None else int(step), field

    def describe(self, address, parsed):
        _, _, step, field = parsed
        kind, default, lo, hi, unit = STEP_FIELDS[field]
        if step is None:
            kind, default, lo, hi, unit = 'list', None, None, None, ''
        return {'address': address, 'kind': kind, 'minimum': lo, 'maximum': hi, 'default': default,
                'unit': unit, 'curve': 'linear', 'labels': [], 'readonly': False}

    def get(self, parsed):
        letter, channel, step, field = parsed
        lane = self.lane(letter, channel, field)
        if step is None:
            return list(lane)
        return lane[step - 1] if step <= len(lane) else STEP_FIELDS[field][1]

    def set(self, parsed, value):
        letter, channel, step, field = parsed
        if step is None:
            for i, v in enumerate(list(value)[:len(self.lane(letter, channel, field))]):
                self._write(letter, channel, i, field, v)
        elif step <= self.length(letter):
            self._write(letter, channel, step - 1, field, value)
            self.core.post_change(f'pattern.{letter}.ch{channel}.step{step}.{field}', value)
        self.report(letter, [channel])

    def _write(self, letter, channel, index, field, value):
        self.lane(letter, channel, field)[index] = value
        if field == 'trig' and not value:
            self.lane(letter, channel, 'acc')[index] = False
            self.lane(letter, channel, 'fill')[index] = False

    def report(self, letter, channels=range(1, 9)):
        """Post the lanes and empty flag of a pattern (as the real core's poll)."""
        for channel in channels:
            for field in STEP_FIELDS:
                self.core.post_change(f'pattern.{letter}.ch{channel}.{field}',
                                      list(self.lane(letter, channel, field)))
        empty = not any(any(self.lane(letter, c, 'trig')) for c in range(1, 9))
        self.core.post_change(f'pattern.{letter}.empty', empty)


class FakeCore:
    def __init__(self, table):
        self.meta = {name: dict(meta) for name, meta in table['describe'].items()}
        self.values = dict(table['values'])
        self.registry = _Names(self.meta)
        self.calls = []
        self.transport = {'playing': False, 'position': 0, 'playing_pattern': 0,
                          'selected_pattern': 0, 'queued_pattern': None, 'chain': []}
        self.readouts = {
            'modulation': {'channel': 0, 'offsets': {}, 'channels': [{} for _ in range(8)]},
            'audio': {'running': True, 'device': 'Fake Out', 'default_device': True,
                      'sample_rate': 44100, 'synth_rate': 44100, 'block_size': 1050,
                      'latency_ms': 23.8, 'mono': False, 'callbacks': 0, 'underruns': 0,
                      'dropped': 0},
            'midi': {'activity': 0, 'notes': [0] * 8, 'pickup': {}},
            'po32': {'level': 0.0, 'recorded_seconds': 0.0, 'progress': 0.0, 'preview_step': -1},
        }
        self.patterns = FakePatterns(self)
        self.verbs = {
            'transport.play': lambda pattern=None: self._play(True),
            'transport.stop': lambda: self._play(False),
            'transport.toggle': lambda: self._play(not self.transport['playing']),
            'pattern.select': self._select,
        }
        self.closed = False
        self.running = None
        self._version = 0
        self._changes = {}
        self._events = []
        self._ids = itertools.count(1)

    # ---------------------------------------------------------------- interface
    def get(self, address):
        parsed = FakePatterns.match(address)
        if parsed is not None:
            return self.patterns.get(parsed)
        if address.startswith(PREF_UI) and address not in self.values:
            return None  # front-end state, None until set (as the real core)
        if address not in self.values:
            raise KeyError(f'unknown address: {address}')
        return self.values[address]

    def describe(self, address):
        parsed = FakePatterns.match(address)
        if parsed is not None:
            return self.patterns.describe(address, parsed)
        if address.startswith(PREF_UI) and address not in self.meta:
            return {'address': address, 'kind': 'json', 'minimum': None, 'maximum': None,
                    'default': None, 'unit': '', 'curve': 'linear', 'labels': [],
                    'readonly': False}
        if address not in self.meta:
            raise KeyError(f'unknown address: {address}')
        return dict(self.meta[address])

    def set(self, address, value, *, edit_all=None, burst=False, record=True):
        meta = self.describe(address)
        if meta['readonly']:
            raise ValueError(f'address is read-only: {address}')
        value = self._coerce(meta, value)
        self.calls.append(('set', address, value,
                           {'edit_all': edit_all, 'burst': burst, 'record': record}))
        parsed = FakePatterns.match(address)
        if parsed is not None:
            self.patterns.set(parsed, value)
            return
        self.post_change(address, value)
        if address.startswith('pattern.') and address.endswith('.length'):
            self.patterns.report(address.split('.')[1])

    DEFERRED = object()  # a verb handler's result: the test finishes it later (finish)

    def act(self, verb, **args):
        action_id = next(self._ids)
        self.calls.append(('act', verb, args))
        handler = self.verbs.get(verb)
        event = {'id': action_id, 'verb': verb}
        self.running = action_id  # handlers may post progress events for it
        if handler is None:
            event.update(status='done', result=None)
        else:
            try:
                event.update(status='done', result=handler(**args))
            except Exception as exc:
                event.update(status='error', error=str(exc))
        if event.get('result') is self.DEFERRED:
            return action_id
        self._post(event)
        return action_id

    def progress(self, action_id, fraction):
        """Post a progress event of an action (exports)."""
        self._post({'id': action_id, 'verb': None, 'status': 'progress', 'progress': fraction})

    def finish(self, action_id, result=None, error=None):
        """End a deferred action with its result (or an error)."""
        event = {'id': action_id, 'verb': None}
        event.update({'status': 'error', 'error': error} if error else {'status': 'done', 'result': result})
        self._post(event)

    def trigger(self, channel, velocity=127, at=None):
        self.calls.append(('trigger', channel, velocity))

    def begin_gesture(self):
        self.calls.append(('gesture', 'begin'))

    def end_gesture(self):
        self.calls.append(('gesture', 'end'))

    def poll(self, since=0):
        return {
            'version': self._version,
            'changes': {a: v for a, (ver, v) in self._changes.items() if ver > since},
            'events': [e for e in self._events if e['version'] > since],
            'transport': dict(self.transport),
            **{name: dict(value) for name, value in self.readouts.items()},
        }

    def close(self):
        self.closed = True

    # ---------------------------------------------------------------- test controls
    def post_change(self, address, value):
        """A change made by the core itself (a CC, a verb): reported by poll."""
        self.values[address] = value
        self._version += 1
        self._changes[address] = (self._version, value)

    def sets(self):
        return [(call[1], call[2]) for call in self.calls if call[0] == 'set']

    def verbs_called(self):
        return [call[1] for call in self.calls if call[0] == 'act']

    # ---------------------------------------------------------------- internals
    def _post(self, event):
        self._version += 1
        event['version'] = self._version
        self._events.append(event)

    def _select(self, pattern):
        index = PATTERN_NAMES.index(pattern) if isinstance(pattern, str) else pattern
        self.transport['selected_pattern'] = index
        self.post_change('pattern.selected', PATTERN_NAMES[index])
        return {'selected': PATTERN_NAMES[index]}

    def _play(self, playing):
        self.transport['playing'] = playing
        self.transport['position'] = 0
        return None

    @staticmethod
    def _coerce(meta, value):
        """The real core's coercion (clamping, enum names) for this metadata."""
        if meta['kind'] in ('json', 'map', 'list'):
            return value
        if meta['kind'] == 'str':
            return None if value is None else str(value)
        entry = Address(meta['address'], get=lambda: None, kind=meta['kind'],
                        minimum=meta['minimum'], maximum=meta['maximum'],
                        labels=tuple(meta['labels']))
        return entry.coerce(value)
