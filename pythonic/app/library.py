"""
The drum patch library of the app core: every drum patch the panel's INST
mode steps through, sorted into categories.

- The factory drum patches (``factory.patches()``), then the ``.mtdrum``
  files of the drum patch folder (``pref.patch_folder``, default ``Drum
  Patches`` in the preset folder) and its subfolders, named after their files.
- Each drum patch has a category guessed from its name (then from its
  folders): ``BD``, ``SD``, ``TOM``, ``RIM``, ``CLAP``, ``CH``, ``OH``,
  ``CYM``, ``PERC``, ``FX`` or ``OTHER``. The channel buttons' drum type
  rules come first; then looser rules read run-together names ("Drybd2",
  "Realoh1", "Lecmidt1"), percussion and effect words, and the short codes
  of sample packs ("HHCD4", "LTAD7").

Addresses:

- ``drum_patch.library`` (read-only list): ``[id, name, category]`` of every
  drum patch, category by category in ``CATEGORIES`` order, the factory ones
  first in machine order, then the folder's by folder and name. An id is
  ``factory:<name>`` or ``user:<path relative to the folder>``.
- ``pref.patch_folder`` (str): the drum patch folder; setting it rescans.

Verbs:

- ``drum_patch.refresh``: rescan the drum patch folder.

``drum_patch.load(library=id)`` (presets.py) loads one into a channel.
"""

import os
import re

from pythonic import factory

from .registry import Address

CATEGORIES = ('BD', 'SD', 'TOM', 'RIM', 'CLAP', 'CH', 'OH', 'CYM', 'PERC', 'FX', 'OTHER')
DRUM_PATCH_SUFFIX = '.mtdrum'
DEFAULT_FOLDER_NAME = 'Drum Patches'


def _word(abbr):
    return re.compile(rf'(?<![A-Z])(?:{abbr})(?![A-Z])', re.I)


# The drum type rules of the panel's channel buttons (drum-type.js), first match wins
_RULES = [
    ('RS', _word('RS|RIM'), re.compile(r'RIM\s*SHOT|SIDE\s*STICK', re.I)),
    ('CP', _word('CP|CLP|HC'), re.compile(r'CLAP', re.I)),
    ('LT', _word('LT'), re.compile(r'(LOW|LO)\s*TOM|TOM\s*(LOW|LO)(?![A-Z])', re.I)),
    ('MT', _word('MT'), re.compile(r'MID\s*TOM|TOM\s*MID', re.I)),
    ('HT', _word('HT'), re.compile(r'(HIGH|HI)\s*TOM|TOM\s*(HIGH|HI)(?![A-Z])', re.I)),
    ('TOM', _word('TOM|TOMS'), None),
    ('BD', _word('BD|BDRUM'), re.compile(r'KICK|BASS\s*DR', re.I)),
    ('SD', _word('SD|SNR'), re.compile(r'SNARE', re.I)),
    ('OH', _word('OH|OHH'), re.compile(r'OPEN\s*H|OP\s*HAT|HAT\s*OPEN', re.I)),
    ('CH', _word('CH|CHH'), re.compile(r'CLOSED\s*H|CL\s*HAT|HAT\s*CLOSED', re.I)),
    ('HH', _word('HH|HAT'), re.compile(r'HI\s*-?\s*HAT', re.I)),
    ('CB', _word('CB'), re.compile(r'COW\s*BELL', re.I)),
    ('RC', _word('RC'), re.compile(r'RIDE', re.I)),
    ('CC', _word('CC'), re.compile(r'CRASH', re.I)),
    ('CY', _word('CY|CYM'), re.compile(r'CYMBAL', re.I)),
    ('CG', _word('CG'), re.compile(r'CONGA|BONGO', re.I)),
    ('TB', _word('TB|TAMB'), re.compile(r'TAMBOURINE', re.I)),
    ('SH', _word('SH'), re.compile(r'SHAKER|CABASA|MARACA', re.I)),
    ('CL', _word('CL'), re.compile(r'CLAVE', re.I)),
    ('PC', _word('PERC'), re.compile(r'PERCUSSION', re.I)),
    ('FX', _word('FX|ZAP|BLIP'), re.compile(r'NOISE|SYNTH|REVERSE|FUZZ', re.I)),
]

_CATEGORY = {'BD': 'BD', 'SD': 'SD', 'LT': 'TOM', 'MT': 'TOM', 'HT': 'TOM', 'TOM': 'TOM',
             'RS': 'RIM', 'CP': 'CLAP', 'CH': 'CH', 'HH': 'CH', 'OH': 'OH', 'RC': 'CYM',
             'CC': 'CYM', 'CY': 'CYM', 'CB': 'PERC', 'CG': 'PERC', 'TB': 'PERC', 'SH': 'PERC',
             'CL': 'PERC', 'PC': 'PERC', 'FX': 'FX'}


# Looser rules for names the drum types miss, first match wins: run-together
# names ending in a drum's abbreviation, then words found anywhere
_LOOSE = [(c, re.compile(rx, re.I)) for c, rx in (
    ('FX', r'sfx|scratch'),
    ('BD', r'(bd|kick|kik|bassdrum)\d*[a-z]?$|^bt[0-9a]|boom'),
    ('SD', r'(sd|sn|snare|snr)\d*[a-z]?$|^st[0-9a]t'),
    ('CLAP', r'clap|clp|snap'),
    ('RIM', r'rim'),
    ('OH', r'(oh|ohh|hhopen)\d*[a-z]?$|^hho'),
    ('CH', r'(ch|chh|ph|hhclose)\d*[a-z]?$|^hhc|opcl|clop|hihat'),
    ('TOM', r'(tom|tm|hito|loto|midt)\d*[a-z]?$|^[hml]tad|tom'),
    ('CYM', r'cym|crash|ride|cshd|china|splash'),
    ('PERC', r'perc|cowbel|guir|agogo|bongo|conga|clave|tamb|gong|shak|cabasa|maraca|woodbl|block'
             r'|tabla|timbal|cuic|tria|whis|hyoshi|bamboo|surd|pand|vibra|slap|metal|bell|brush|ra[t]?chet'),
    ('FX', r'sfx|fx|scratch|blip|zap|noi[sz]|hit\d|voice|vox|vynil|vinyl|loop|laser'),
)]


def drum_type(name):
    """The short drum type label of a drum patch name (BD, SD, LT, ...), or ''."""
    if not name:
        return ''
    for label, abbr, phrase in _RULES:
        if abbr.search(name) or (phrase is not None and phrase.search(name)):
            return label
    return ''


def _loose(name):
    for found, rule in _LOOSE:
        if rule.search(name):
            return found
    return ''


def category(*names):
    """The category of a drum patch: from the first of ``names`` (its name,
    then its folders) that gives one; ``OTHER`` when none does."""
    for name in names:
        label = drum_type(name)
        if label:
            return _CATEGORY[label]
        found = _loose(str(name or ''))
        if found:
            return found
    return 'OTHER'


def _natural(text):
    """A sort key that orders "Bd2" before "Bd10"."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', text)]


def scan_folder(folder):
    """(relative path, name, folders) of every drum patch under ``folder``."""
    found = []
    for root, dirs, files in os.walk(folder):
        dirs.sort()
        for file in files:
            if file.lower().endswith(DRUM_PATCH_SUFFIX) and not file.startswith('.'):
                path = os.path.join(root, file)
                relative = os.path.relpath(path, folder).replace(os.sep, '/')
                folders = relative.split('/')[:-1]
                found.append((relative, os.path.splitext(file)[0], folders))
    found.sort(key=lambda f: (_natural('/'.join(f[2])), _natural(f[1])))
    return found


class Library:
    """The drum patch library of an AppCore."""

    def __init__(self, core):
        self._core = core
        self._entries = []  # [id, name, category], in browse order
        self._factory = [[f'factory:{name}', name, category(name)] for name, _ in factory.patches()]
        self._entries = self._sorted(self._factory)

    # ================================================================== interface
    def register(self, registry):
        reg = registry.register
        reg(Address('drum_patch.library', get=lambda: self._entries, kind='list'))
        reg(Address('pref.patch_folder', get=self.folder, set=self._set_folder, kind='str',
                    queued=False, undoable=False))

    def verbs(self):
        return {'drum_patch.refresh': self._verb_refresh}

    def folder(self):
        prefs = self._core.preferences
        return prefs.get('patch_folder') or os.path.join(prefs.get_preset_folder(),
                                                         DEFAULT_FOLDER_NAME)

    def _set_folder(self, folder):
        folder = os.path.abspath(os.path.expanduser(str(folder)))
        if not os.path.isdir(folder):
            raise ValueError(f'pref.patch_folder: {folder} is not a folder')
        if not self._core.preferences.set('patch_folder', folder):
            raise OSError('the preferences file could not be written (patch_folder)')
        self._core.call_soon(self.scan)

    # ================================================================== the list
    @staticmethod
    def _sorted(entries):
        order = {c: i for i, c in enumerate(CATEGORIES)}
        return sorted(entries, key=lambda e: order[e[2]])  # stable: source order within a category

    def scan(self):
        """Rescan the drum patch folder (action thread); poll reports the list."""
        user = [[f'user:{relative}', name, category(name, *reversed(folders))]
                for relative, name, folders in scan_folder(self.folder())]
        self._entries = self._sorted(self._factory + user)
        self._core.note_changes(['drum_patch.library'])
        return len(user)

    def _verb_refresh(self):
        return {'patches': len(self._entries), 'folder_patches': self.scan()}

    def resolve(self, patch_id):
        """('factory', name) or ('path', absolute path) of a library id."""
        patch_id = str(patch_id)
        kind, _, rest = patch_id.partition(':')
        if kind == 'factory' and rest:
            return 'factory', rest
        if kind == 'user' and rest:
            folder = os.path.realpath(self.folder())
            path = os.path.realpath(os.path.join(folder, rest))
            if os.path.commonpath([path, folder]) == folder:
                return 'path', path
        raise ValueError(f'not a drum patch of the library: {patch_id!r}')
