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
"""

import itertools
import tempfile
from unittest import mock

from pythonic.app.registry import Address

PREF_UI = 'pref.ui.'


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
        }
        self.verbs = {
            'transport.play': lambda pattern=None: self._play(True),
            'transport.stop': lambda: self._play(False),
            'transport.toggle': lambda: self._play(not self.transport['playing']),
        }
        self.closed = False
        self._version = 0
        self._changes = {}
        self._events = []
        self._ids = itertools.count(1)

    # ---------------------------------------------------------------- interface
    def get(self, address):
        if address.startswith(PREF_UI) and address not in self.values:
            return None  # front-end state, None until set (as the real core)
        if address not in self.values:
            raise KeyError(f'unknown address: {address}')
        return self.values[address]

    def describe(self, address):
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
        self.post_change(address, value)

    def act(self, verb, **args):
        action_id = next(self._ids)
        self.calls.append(('act', verb, args))
        handler = self.verbs.get(verb)
        event = {'id': action_id, 'verb': verb}
        if handler is None:
            event.update(status='done', result=None)
        else:
            try:
                event.update(status='done', result=handler(**args))
            except Exception as exc:
                event.update(status='error', error=str(exc))
        self._post(event)
        return action_id

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
        return [(address, value) for kind, address, value, *_ in self.calls if kind == 'set']

    def verbs_called(self):
        return [call[1] for call in self.calls if call[0] == 'act']

    # ---------------------------------------------------------------- internals
    def _post(self, event):
        self._version += 1
        event['version'] = self._version
        self._events.append(event)

    def _play(self, playing):
        self.transport['playing'] = playing
        self.transport['position'] = 0
        return None

    @staticmethod
    def _coerce(meta, value):
        if meta['kind'] in ('json', 'map'):
            return value
        """The real core's coercion (clamping, enum names) for this metadata."""
        entry = Address(meta['address'], get=lambda: None, kind=meta['kind'],
                        minimum=meta['minimum'], maximum=meta['maximum'],
                        labels=tuple(meta['labels']))
        return entry.coerce(value)
