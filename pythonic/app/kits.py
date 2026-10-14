"""
The kit library of the app core: kits are their own documents, apart from
presets, as on a TR-8S.

A kit is the eight channel sounds under a name. The library holds the factory
kits (read-only, one per drum machine: ``factory:909``) and the user kits,
JSON files in the ``Kits`` folder of the preset folder (``user:My 808`` is
``Kits/My 808.json``). A preset plays one kit: it saves its sounds with the
kit's id, and a preset without one plays its own sounds, the preset kit
``''`` (named after the preset).

Edits mark the current kit as edited (``kit.edited``: its sounds differ from
the stored kit). Switching kits keeps the edits of the kit left for the
session, so going back finds them; ``kit.write`` stores them. A pattern can
recall a kit (``pattern.<P>.kit``): it is switched to when the pattern starts
playing, or when the pattern is selected while stopped.

Addresses:

- ``kit.list`` (read-only list): ``{'id', 'name', 'factory'}`` of every kit,
  the factory kits in machine order, then the user kits by name.
- ``kit.current`` (read-only str): the id of the kit playing, ``''`` for the
  preset kit.
- ``kit.name`` (read-only str): its name (the preset's name for the preset
  kit).
- ``kit.edited`` (read-only bool): the sounds differ from the stored kit.
- ``pattern.<P>.kit`` (str, A..L): the kit the pattern recalls, ``''`` for
  none; a known kit id. One undo step.

Verbs (results and errors arrive through ``poll``):

- ``kit.select`` (kit, browse=False): switch to a kit at block start: the
  edits of the kit left are kept for the session, the new kit plays its
  session edits or its stored sounds. One undo step; with ``browse`` the
  switches in a row are one step. Result ``{'kit', 'name', 'edited'}``.
- ``kit.write`` (name=None, overwrite=False): store the sounds as a user kit:
  over the current user kit without a name, else as ``name`` (an existing
  kit of that name is only replaced with ``overwrite``; otherwise the result
  is ``{'saved': False, 'exists': True, 'name'}``). The written kit becomes
  the current one, not edited. Result ``{'saved': True, 'exists', 'name',
  'kit'}``.
- ``kit.refresh``: rescan the Kits folder.
- ``kit.from_programs``: write the preset's occupied programs as user kits,
  named after the preset and the program. Result ``{'kits': [names]}``.
"""

import copy
import json
import os
import re

from pythonic import factory
from pythonic.drum_channel import DrumChannel
from pythonic.pattern_manager import PatternManager

from .registry import Address
from .undo import CHANNELS, KIT

PRESET_KIT = ''
KITS_FOLDER = 'Kits'
KIT_FORMAT = 'pythonic-kit'
NUM_CHANNELS = 8
PATTERN_NAMES = tuple(PatternManager.PATTERN_NAMES)
_BAD_NAME = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


class Kit:
    """A kit of the library: its name and its eight sounds, as a channel
    reports them (``DrumChannel.get_parameters``), so they compare equal to
    the channels that play them."""

    __slots__ = ('id', 'name', 'factory', 'sounds')

    def __init__(self, kit_id, name, is_factory, sounds):
        self.id, self.name, self.factory, self.sounds = kit_id, name, is_factory, sounds


def kit_name(name):
    """A user kit name as given, checked: no path characters, not empty."""
    name = ' '.join(str(name or '').split())
    if not name or _BAD_NAME.search(name) or name in ('.', '..'):
        raise ValueError(f'not a kit name: {name!r}')
    return name[:60]


class Kits:
    """Kit addresses and verbs of an AppCore."""

    def __init__(self, core):
        self._core = core
        self._scratch = DrumChannel(0, 44100)  # turns stored sounds into channel sounds
        self._factory = [Kit(f'factory:{machine}', machine, True, self._normalize(kit['channels']))
                         for machine, kit in zip(factory.MACHINES, factory.kits())]
        self._library = {}  # id -> Kit, replaced whole on a scan
        self._order = []
        self._scan_user()
        # State the audio thread changes (block start, a pattern start)
        self.current = PRESET_KIT
        self._base = [ch.get_parameters() for ch in core.synth.channels]
        self._preset_sounds = self._base
        self._session = {}  # kit id -> the sounds it was left with, edited
        self._generation = 0  # bumped when a pattern start switches kits
        self._seen_generation = 0
        self._reported_edited = False
        self.touched = False  # the core sets it when channel values are reported

    # ================================================================== interface
    def register(self, registry):
        reg = registry.register
        reg(Address('kit.list', get=self.listing, kind='list'))
        reg(Address('kit.current', get=lambda: self.current, kind='str'))
        reg(Address('kit.name', get=lambda: self.name_of(self.current), kind='str'))
        reg(Address('kit.edited', get=self.edited, kind='bool'))
        pm = self._core.pattern_manager
        for index, letter in enumerate(PATTERN_NAMES):
            def set_kit(value, i=index):
                pm.patterns[i].kit = value
            reg(Address(f'pattern.{letter}.kit', get=lambda i=index: pm.patterns[i].kit,
                        set=set_kit, kind='str', default='', convert=self._pattern_kit))

    def verbs(self):
        return {
            'kit.select': self._verb_select,
            'kit.write': self._verb_write,
            'kit.refresh': self._verb_refresh,
            'kit.from_programs': self._verb_from_programs,
        }

    def folder(self):
        return os.path.join(self._core.preferences.get_preset_folder(), KITS_FOLDER)

    def listing(self):
        return [{'id': k, 'name': self._library[k].name, 'factory': self._library[k].factory}
                for k in self._order]

    def name_of(self, kit_id):
        if kit_id == PRESET_KIT:
            presets = getattr(self._core, 'presets', None)
            return presets.name if presets is not None else 'Untitled'
        kit = self._library.get(kit_id)
        return kit.name if kit is not None else kit_id.partition(':')[2]

    def edited(self):
        """Do the channel sounds differ from the stored kit?"""
        base = self._base
        return any(ch.get_parameters() != sound for ch, sound in zip(self._core.synth.channels, base))

    def _known(self, kit_id):
        kit_id = str(kit_id or '')
        if kit_id != PRESET_KIT and kit_id not in self._library:
            raise ValueError(f'not a kit: {kit_id!r}')
        return kit_id

    def _pattern_kit(self, value):
        value = str(value or '')
        if value and value not in self._library:
            raise ValueError(f'not a kit: {value!r}')
        return value

    # ================================================================== the library
    def _normalize(self, sounds):
        """Stored sounds (any keys missing) as channels report them."""
        result = []
        init = self._core.init_sound
        for i in range(NUM_CHANNELS):
            sound = copy.deepcopy(init)
            if i < len(sounds) and isinstance(sounds[i], dict):
                sound.update(copy.deepcopy(sounds[i]))
            self._scratch.set_parameters(sound)
            result.append(self._scratch.get_parameters())
        return result

    def _scan_user(self):
        user = []
        folder = self.folder()
        try:
            files = sorted(os.listdir(folder), key=str.lower)
        except OSError:
            files = []
        for file in files:
            if not file.lower().endswith('.json') or file.startswith('.'):
                continue
            path = os.path.join(folder, file)
            try:
                with open(path, encoding='utf-8') as f:
                    data = json.load(f)
                sounds = data['channels']
                if not isinstance(sounds, list) or not sounds:
                    raise ValueError('no channels')
            except (OSError, ValueError, KeyError, TypeError) as exc:
                print(f'Kit {path} skipped: {exc}', flush=True)
                continue
            name = os.path.splitext(file)[0]
            user.append(Kit(f'user:{name}', name, False, self._normalize(sounds)))
        kits = self._factory + user
        self._library = {k.id: k for k in kits}
        self._order = [k.id for k in kits]

    def scan(self):
        """Rescan the Kits folder (action thread); poll reports the list."""
        self._scan_user()
        self._core.note_changes(['kit.list', 'kit.name'])

    def _verb_refresh(self):
        self.scan()
        return {'kits': len(self._order)}

    # ================================================================== switching
    def _switch(self, kit_id):
        """Play another kit (audio thread): keep the edits of the one left."""
        channels = self._core.synth.channels
        if self.edited():
            self._session[self.current] = [ch.get_parameters() for ch in channels]
        else:
            self._session.pop(self.current, None)
        base = self._preset_sounds if kit_id == PRESET_KIT else self._library[kit_id].sounds
        for channel, sound in zip(channels, self._session.get(kit_id) or base):
            channel.set_parameters(copy.deepcopy(sound))
        self.current = kit_id
        self._base = base

    def _verb_select(self, kit, browse=False):
        kit = self._known(kit)
        core = self._core
        if kit == self.current:
            return {'kit': kit, 'name': self.name_of(kit), 'edited': self.edited()}
        name = self.name_of(kit)
        core.undo.snapshot_op((CHANNELS, KIT), f'kit {name}', lambda: self._switch(kit),
                              merge='kit' if browse else None)
        core.note_changes(core.undo.part_addresses((CHANNELS, KIT)))
        return {'kit': kit, 'name': name, 'edited': self.edited()}

    def follow(self, index):
        """A pattern starts playing, or is selected while stopped (audio
        thread): switch to the kit it recalls. Not an undo step."""
        kit = self._core.pattern_manager.patterns[index].kit
        if kit and kit != self.current and kit in self._library:
            self._switch(kit)
            self._generation += 1

    def collect(self):
        """Report what the audio thread changed (poll): a kit a pattern
        recalled, the edited mark after channel changes."""
        core = self._core
        if self._generation != self._seen_generation:
            self._seen_generation = self._generation
            core.note_changes(core.undo.part_addresses((CHANNELS, KIT)))
        if self.touched:
            self.touched = False
            edited = self.edited()
            if edited != self._reported_edited:
                self._reported_edited = edited
                core.note_values({'kit.edited': edited})

    # ================================================================== undo and presets
    def capture(self):
        """The kit part of a snapshot (undo.py): the current kit, its stored
        sounds and the preset's own sounds (lists replaced, never edited)."""
        return (self.current, self._base, self._preset_sounds)

    def restore(self, state):
        """Put a kit part back (block start, after the channels). A preset
        load passes ``loaded(kit)``: the channels are the preset's sounds."""
        current, base, preset_sounds = state
        if preset_sounds is None:
            preset_sounds = [ch.get_parameters() for ch in self._core.synth.channels]
            kit = self._library.get(current)
            current, base = (current, kit.sounds) if kit is not None else (PRESET_KIT, preset_sounds)
        self.current, self._base, self._preset_sounds = current, base, preset_sounds

    @staticmethod
    def loaded(kit_id=None):
        """The kit part of a preset read from a file: it plays ``kit_id`` (when
        the library has it) with the sounds the file holds."""
        return (str(kit_id or ''), None, None)

    # ================================================================== writing
    def _write_file(self, name, sounds):
        folder = self.folder()
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f'{name}.json')
        temp = f'{path}.tmp-{os.getpid()}'
        try:
            with open(temp, 'w', encoding='utf-8') as f:
                json.dump({'format': KIT_FORMAT, 'version': 1, 'name': name, 'channels': sounds},
                          f, indent=2)
            os.replace(temp, path)
        finally:
            if os.path.exists(temp):
                os.remove(temp)

    def _path(self, name):
        return os.path.join(self.folder(), f'{name}.json')

    def _verb_write(self, name=None, overwrite=False):
        core = self._core
        if name is None:
            if not self.current.startswith('user:'):
                raise ValueError('a factory kit or a preset kit is written under a new name')
            name, overwrite = self.name_of(self.current), True
        name = kit_name(name)
        exists = os.path.exists(self._path(name))
        if exists and not overwrite:
            return {'saved': False, 'exists': True, 'name': name}
        core.at_block_start(lambda: None)  # queued sets land first
        sounds = core.at_block_start(lambda: [ch.get_parameters() for ch in core.synth.channels])
        self._write_file(name, sounds)
        self._scan_user()
        kit_id = f'user:{name}'
        stored = self._library[kit_id].sounds

        def adopt():
            # The edits went into the written kit: the kit left has none
            self._session.pop(self.current, None)
            self._session.pop(kit_id, None)
            self.current, self._base = kit_id, stored
        core.at_block_start(adopt)
        core.note_changes(['kit.list', 'kit.current', 'kit.name', 'kit.edited'])
        return {'saved': True, 'exists': exists, 'name': name, 'kit': kit_id}

    def _verb_from_programs(self):
        core = self._core
        synth = core.synth
        core.at_block_start(lambda: None)
        current, slots = core.at_block_start(lambda: (
            synth.get_current_program(),
            {int(k): v for k, v in synth.get_programs_data()['slots'].items()}))
        live = [ch.get_parameters() for ch in synth.channels]
        names = []
        preset = _BAD_NAME.sub('-', self.name_of(PRESET_KIT))
        for index in range(16):
            if index == current:
                sounds = live
            elif index in slots and slots[index].get('channels'):
                sounds = slots[index]['channels']
            else:
                continue
            words = core.programs.names()[index]
            base = kit_name(f'{preset} {index + 1}' + (f' {words}' if words else ''))
            name, n = base, 2
            while os.path.exists(self._path(name)):
                name, n = f'{base} ({n})', n + 1
            self._write_file(name, sounds)
            names.append(name)
        if not names:
            raise ValueError('the preset has no programs')
        self.scan()
        return {'kits': names}
