"""
Preset and drum-patch files of the app core (inventory cluster N): one load
and one save path for every front-end.

Paths are absolute, or file names relative to the preset folder
(``pref.preset_folder``). Files are read and written on the action thread;
the loaded state is built there and swapped in at block start.

Addresses:

- ``preset.name`` (read-only): the name of the preset last loaded or saved
  (the ``Name`` of a .mtpreset, else the file name without extension).
- ``preset.path`` (read-only): its path, None before any load or save.
- ``preset.files`` (read-only list): the preset files (``.mtpreset``,
  ``.json``) in the preset folder, sorted by name, for an in-panel list.
- ``preset.clipboard`` (read-only bool): the preset clipboard holds a preset.

Verbs (results and errors arrive through ``poll`` as action events):

- ``preset.load`` (path): load a preset, a .mtpreset or a JSON preset (told
  apart by content). Everything is replaced: the sounds of the eight channels
  (what the file leaves out takes its default), the tempo, swing, step rate,
  fill rate, master, the patterns (patterns missing from the file are empty),
  the programs (a .mtpreset has none: the bank is emptied), the morph (a file
  without morph endpoints gets both endpoints equal to its sounds) and the
  mutes when the file has them. One undo step; ``poll`` reports every value.
  The file becomes the last preset (loaded at the next start-up) and the
  first recent file. Result ``{'path', 'name', 'format'}``.
- ``preset.load_last``: load the last preset if its file still exists;
  ``{'loaded': False}`` otherwise.
- ``preset.save`` (path, overwrite=False): save the whole preset as JSON
  (``.json`` is added to a path without extension). An existing file is only
  replaced with ``overwrite=True``; otherwise nothing is written and the
  result is ``{'saved': False, 'exists': True, 'path'}`` so the front-end can
  ask and save again. Result ``{'saved': True, 'exists': bool, 'path'}``.
- ``drum_patch.load`` (path, channel=None): load a .mtdrum drum patch into a
  channel (1..8, default the selected one); one undo step. Result
  ``{'channel', 'name', 'path'}``.
- ``drum_patch.save`` (path, channel=None, overwrite=False): save a channel's
  drum patch (``.mtdrum`` added when the path has no extension); overwrite
  works as for ``preset.save``.
- ``preset.copy``, ``preset.cut``, ``preset.paste``: the preset clipboard
  (sounds, globals, programs, morph, patterns). Cut is copy then initialize.
- ``preset.initialize``: every channel back to its init sound, every pattern
  empty, both morph endpoints set to the init sounds. One undo step.
- ``preset.randomize_all``: randomize the drum patch of every channel and the
  selected pattern; both morph endpoints take the new sounds. One undo step.
- ``preset.refresh``: rescan the preset folder (``preset.files``).
"""

import copy
import json
import os

from pythonic.pattern_manager import Pattern, PatternManager
from pythonic.preset_manager import (DrumPatchParser, DrumPatchWriter, PythonicPresetParser,
                                     apply_drum_patch_to_channel, convert_drum_patch_data)

from .registry import Address
from .sound import SOUND_SUFFIXES
from .undo import CHANNELS, GLOBALS, MORPH, PATTERNS, PRESET, PROGRAMS, pattern_part

NUM_CHANNELS = 8
NUM_PROGRAMS = 16
PRESET_SUFFIXES = ('.mtpreset', '.json')
DRUM_PATCH_SUFFIX = '.mtdrum'


def channel_addresses(channel):
    """Every address of one channel's drum patch (channel 1..8)."""
    return [f'ch{channel}.{suffix}' for suffix in sorted(SOUND_SUFFIXES)] + [f'ch{channel}.name']


def _with_suffix(path, suffix):
    """The path with ``suffix`` added when it has no extension (a native save
    dialog adds the default suffix only after its own overwrite check)."""
    return path if os.path.splitext(path)[1] else path + suffix


def _write_text(path, text):
    """Write a file in one step: a failed write leaves the old file whole."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    temp = f'{path}.tmp-{os.getpid()}'
    try:
        with open(temp, 'w', encoding='utf-8') as f:
            f.write(text)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.remove(temp)


class Presets:
    """Preset and drum-patch I/O addresses and verbs of an AppCore."""

    def __init__(self, core):
        self._core = core
        self._name = 'Untitled'
        self._path = None
        self._clipboard = None
        self._defaults = core.init_sound  # the sound of a new channel

    @property
    def name(self):
        return self._name

    # ================================================================== interface
    def register(self, registry):
        reg = registry.register
        reg(Address('preset.name', get=lambda: self._name, kind='str'))
        reg(Address('preset.path', get=lambda: self._path, kind='str'))
        reg(Address('preset.files', get=self.files, kind='list'))
        reg(Address('preset.clipboard', get=lambda: self._clipboard is not None, kind='bool'))

    def verbs(self):
        return {
            'preset.load': self._verb_load,
            'preset.load_last': self._verb_load_last,
            'preset.save': self._verb_save,
            'preset.copy': self._verb_copy,
            'preset.cut': self._verb_cut,
            'preset.paste': self._verb_paste,
            'preset.initialize': self._verb_initialize,
            'preset.randomize_all': self._verb_randomize_all,
            'preset.refresh': self._verb_refresh,
            'drum_patch.load': self._verb_load_patch,
            'drum_patch.save': self._verb_save_patch,
        }

    def folder(self):
        return self._core.preferences.get_preset_folder()

    def files(self):
        try:
            names = os.listdir(self.folder())
        except OSError:
            return []
        return sorted(n for n in names if n.lower().endswith(PRESET_SUFFIXES))

    def _resolve(self, path):
        if not path:
            raise ValueError('no file given')
        path = os.path.expanduser(str(path))
        return os.path.abspath(os.path.join(self.folder(), path))

    def _channel_index(self, channel):
        if channel is None:
            return self._core.synth.selected_channel
        if (isinstance(channel, int) and not isinstance(channel, bool)
                and 1 <= channel <= NUM_CHANNELS):
            return channel - 1
        raise ValueError(f'not a channel: {channel!r} (1-{NUM_CHANNELS})')

    def _sound(self, params):
        """A complete channel sound: the init sound with ``params`` over it."""
        sound = copy.deepcopy(self._defaults)
        sound.update(copy.deepcopy(params))
        return sound

    # ================================================================== preset load
    def _verb_load(self, path):
        path = self._resolve(path)
        state, mutes, name, fmt = self.read_preset(path)
        self._install(state, mutes, 'load preset')
        prefs = self._core.preferences
        prefs.add_recent_file(path)
        prefs.set('last_preset', path)
        self._set_document(path, name, extra=('pref.recent_files',))
        return {'path': path, 'name': name, 'format': fmt}

    def _verb_load_last(self):
        path = self._core.preferences.get('last_preset')
        if not path or not os.path.isfile(path):
            return {'loaded': False}
        result = self._verb_load(path)
        result['loaded'] = True
        return result

    def read_preset(self, path):
        """Parse a preset file into (state, mutes, name, format): a state as
        Undo.capture(PRESET) builds it, ready to be installed."""
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            text = f.read()
        stem = os.path.splitext(os.path.basename(path))[0]
        if text.lstrip().startswith('{'):
            try:
                data = json.loads(text)
            except ValueError as exc:
                raise ValueError(f'not a preset file: {exc}') from None
            state, mutes = self._from_json(data)
            return state, mutes, stem, 'json'
        parser = PythonicPresetParser()
        data = parser.convert_to_synth_format(parser.parse_string(text))
        state, mutes = self._from_mtpreset(data)
        return state, mutes, data.get('name') or stem, 'mtpreset'

    def _globals(self, tempo, swing, step_rate, fill_rate, master):
        pm = self._core.pattern_manager
        if step_rate not in pm.STEP_RATES:
            step_rate = pm.step_rate
        return (max(1, min(300, int(round(float(tempo))))), float(swing), step_rate,
                max(2, min(8, int(round(float(fill_rate))))), float(master))

    def _from_mtpreset(self, data):
        drums = data.get('drums') or []
        if not drums:
            raise ValueError('not a preset file: it has no drum patches')
        channels = [self._sound(drums[i] if i < len(drums) else {})
                    for i in range(NUM_CHANNELS)]
        patterns = PatternManager(num_channels=NUM_CHANNELS)
        if data.get('patterns'):
            patterns.load_from_preset_data(data['patterns'])
        position = data.get('morph_position')
        state = {
            CHANNELS: channels,
            GLOBALS: self._globals(data['tempo'], data['swing'], data['step_rate'],
                                   data['fill_rate'], data['master_volume_db']),
            PROGRAMS: ([None] * NUM_PROGRAMS, 0),
            MORPH: self._morph({}, channels, 0.5 if position is None else position),
            PATTERNS: patterns.patterns,
        }
        mutes = [bool(m) for m in (data.get('mutes') or [])][:NUM_CHANNELS]
        return state, mutes or None

    def _from_json(self, data):
        if not isinstance(data, dict) or not data.get('channels'):
            raise ValueError('not a preset file: it has no channels')
        pm = self._core.pattern_manager
        saved = data.get('channels')
        channels = [self._sound(saved[i] if i < len(saved) else {})
                    for i in range(NUM_CHANNELS)]
        patterns_data = data.get('patterns') or {}

        def value(key, pattern_key, current):
            if key in data:
                return data[key]
            return patterns_data.get(pattern_key, current)

        patterns = [Pattern.from_dict(p) for p in patterns_data.get('patterns', [])]
        patterns = patterns[:len(PatternManager.PATTERN_NAMES)]
        for name in PatternManager.PATTERN_NAMES[len(patterns):]:
            patterns.append(Pattern(name, 16, NUM_CHANNELS))
        morph = data.get('morph') or {}
        state = {
            CHANNELS: channels,
            GLOBALS: self._globals(value('tempo', 'bpm', pm.bpm), data.get('swing', pm.swing),
                                   value('step_rate', 'step_rate', pm.step_rate),
                                   value('fill_rate', 'fill_rate', pm.fill_rate),
                                   data.get('master_volume_db',
                                            self._core.synth.master_volume_db)),
            PROGRAMS: self._programs(data.get('programs')),
            MORPH: self._morph(morph, channels, morph.get('position', 0.5) if morph else 0.5),
            PATTERNS: patterns,
        }
        mutes = data.get('mutes')
        mutes = [bool(m) for m in mutes][:NUM_CHANNELS] if isinstance(mutes, list) else None
        return state, mutes or None

    @staticmethod
    def _programs(data):
        slots = [None] * NUM_PROGRAMS
        if not data:
            return slots, 0
        for key, program in (data.get('slots') or {}).items():
            index = int(key)
            if 0 <= index < NUM_PROGRAMS:
                slots[index] = program
        current = int(data.get('current_program', 0))
        return slots, current if 0 <= current < NUM_PROGRAMS else 0

    def _morph(self, data, channels, position):
        """Morph data with both endpoints (missing ones are the sounds)."""
        return {
            'position': max(0.0, min(1.0, float(position))),
            'endpoint_a': copy.deepcopy(data.get('endpoint_a') or channels),
            'endpoint_b': copy.deepcopy(data.get('endpoint_b') or channels),
        }

    def _install(self, state, mutes, label, parts=PRESET):
        """Swap a built state in at block start: one undo step, every value
        reported by poll. Mutes are set too (they are not undone)."""
        core = self._core
        core.at_block_start(lambda: None)  # queued sets land before the snapshot

        def apply():
            core.undo.restore(parts, state)
            for index, muted in enumerate(mutes or ()):
                core.synth.mute_channel(index, muted)
        with core.bulk_change(label, parts):
            core.at_block_start(apply)

    def _set_document(self, path, name, extra=()):
        self._path, self._name = path, name
        self._core.note_changes(['preset.path', 'preset.name', 'preset.files',
                                 *[a for a in extra if a in self._core.registry]])

    # ================================================================== preset save
    def capture(self):
        """The whole preset, taken after the queued sets have landed."""
        core = self._core
        core.at_block_start(lambda: None)
        state = core.undo.capture(PRESET)
        mutes = [bool(ch.muted) for ch in core.synth.channels]
        return state, mutes

    def to_json(self, state, mutes):
        """A preset as the JSON document tkinter has always written (plus mutes)."""
        bpm, swing, step_rate, fill_rate, master = state[GLOBALS]
        slots, current = state[PROGRAMS]
        pm = self._core.pattern_manager
        return {
            'version': '1.0',
            'master_volume_db': master,
            'channels': state[CHANNELS],
            'patterns': {
                'patterns': [p.to_dict() for p in state[PATTERNS]],
                'selected_pattern_index': pm.selected_pattern_index,
                'playing_pattern_index': pm.playing_pattern_index,
                'bpm': bpm,
                'fill_rate': fill_rate,
                'step_rate': step_rate,
            },
            'tempo': bpm,
            'step_rate': step_rate,
            'swing': swing,
            'fill_rate': fill_rate,
            'morph': state[MORPH],
            'programs': {
                'current_program': current,
                'slots': {str(i): p for i, p in enumerate(slots) if p is not None},
            },
            'mutes': mutes,
        }

    def _verb_save(self, path, overwrite=False):
        path = _with_suffix(self._resolve(path), '.json')
        exists = os.path.exists(path)
        if exists and not overwrite:
            return {'saved': False, 'exists': True, 'path': path}
        state, mutes = self.capture()
        _write_text(path, json.dumps(self.to_json(state, mutes), indent=2))
        self._core.preferences.add_recent_file(path)
        self._set_document(path, os.path.splitext(os.path.basename(path))[0],
                           extra=('pref.recent_files',))
        return {'saved': True, 'exists': exists, 'path': path}

    # ================================================================== clipboard, init, randomize
    def _verb_copy(self):
        state, _mutes = self.capture()
        self._clipboard = copy.deepcopy(state)
        self._core.note_changes(['preset.clipboard'])
        return {'copied': True}

    def _verb_cut(self):
        self._verb_copy()
        return self._verb_initialize()

    def _verb_paste(self):
        if self._clipboard is None:
            raise ValueError('the preset clipboard is empty')
        self._install(copy.deepcopy(self._clipboard), None, 'paste preset')
        return {'pasted': True}

    def _verb_initialize(self):
        channels = [self._sound({}) for _ in range(NUM_CHANNELS)]
        state = {
            CHANNELS: channels,
            MORPH: self._morph({}, channels, self._core.morph_manager.position),
            PATTERNS: PatternManager(num_channels=NUM_CHANNELS).patterns,
        }
        self._install(state, None, 'initialize preset', (CHANNELS, MORPH, PATTERNS))
        return {'initialized': True}

    def _verb_randomize_all(self):
        core = self._core
        index = core.pattern_manager.selected_pattern_index
        parts = (CHANNELS, MORPH, pattern_part(index))

        def randomize():
            for channel in core.synth.channels:
                channel.randomize()
            core.pattern_manager.randomize_pattern(index)
            core.morph_manager.capture_endpoint_a()
            core.morph_manager.capture_endpoint_b()
        core.at_block_start(lambda: None)
        with core.bulk_change('randomize all', parts):
            core.at_block_start(randomize)
        return {'pattern': PatternManager.PATTERN_NAMES[index]}

    def _verb_refresh(self):
        self._core.note_changes(['preset.files'])
        return {'files': self.files()}

    # ================================================================== drum patches
    def _verb_load_patch(self, path, channel=None):
        index = self._channel_index(channel)
        path = self._resolve(path)
        patch = DrumPatchParser().parse_file(path)
        # Patches without a Name line are named after the file
        patch.setdefault('Name', os.path.splitext(os.path.basename(path))[0])
        data = convert_drum_patch_data(patch)
        core = self._core
        core.undo.snapshot_op(
            (CHANNELS,), 'load drum patch',
            lambda: apply_drum_patch_to_channel(core.synth.channels[index], data))
        core.note_changes(channel_addresses(index + 1))
        return {'channel': index + 1, 'name': data['name'], 'path': path}

    def _verb_save_patch(self, path, channel=None, overwrite=False):
        index = self._channel_index(channel)
        path = _with_suffix(self._resolve(path), DRUM_PATCH_SUFFIX)
        exists = os.path.exists(path)
        if exists and not overwrite:
            return {'saved': False, 'exists': True, 'path': path}
        core = self._core
        patch = core.at_block_start(
            lambda: DrumPatchWriter.patch_from_channel(core.synth.channels[index]))
        _write_text(path, DrumPatchWriter.format_patch(patch))
        return {'saved': True, 'exists': exists, 'path': path, 'channel': index + 1}
