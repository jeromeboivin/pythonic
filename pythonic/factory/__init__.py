"""
The factory content shipped with Pythonic, read-only.

- ``presets/``: one JSON preset per drum machine (505, 707, 808, 909, DMX,
  LM2, TR-8): that machine's sounds and twelve patterns, with all seven
  machines' sounds in programs 1-7. Built by a local script (not in the
  repository).
- ``drum_patches.json``: the drum patches fitted to TR-8 samples, as channel
  sounds; eight of them are the TR-8 kit.
- the factory kits (``kits()``, those programs) and the factory drum patches
  (``patches()``: every channel sound of every kit, then the other drum
  patches of ``drum_patches.json``, by name).
"""

import json
import os
from pathlib import Path

PRESETS_DIR = Path(__file__).resolve().parent / 'presets'
PATCHES_FILE = Path(__file__).resolve().parent / 'drum_patches.json'
MACHINES = ('505', '707', '808', '909', 'DMX', 'LM2', 'TR-8')


def preset_files():
    """The file names of the factory presets, in machine order."""
    return [f'{m} Beats.json' for m in MACHINES if (PRESETS_DIR / f'{m} Beats.json').is_file()]


def preset_path(name):
    """The path of a factory preset given by file name or name ("808 Beats")."""
    name = str(name)
    for file in preset_files():
        if name in (file, file[:-len('.json')]):
            return PRESETS_DIR / file
    raise ValueError(f'not a factory preset: {name!r}')


def is_factory_path(path):
    """Is ``path`` inside the factory presets folder?"""
    if not path:
        return False
    try:
        return os.path.commonpath([os.path.realpath(path), os.path.realpath(PRESETS_DIR)]) == \
            os.path.realpath(PRESETS_DIR)
    except ValueError:  # another drive (Windows)
        return False


def kits():
    """The factory kits: the stored program of each machine, in machine order
    (programs 1-7 of every factory preset)."""
    files = preset_files()
    if not files:
        return []
    with open(PRESETS_DIR / files[0], encoding='utf-8') as f:
        slots = json.load(f)['programs']['slots']
    return [slots[str(i)] for i in range(len(MACHINES))]


def patches():
    """The factory drum patches: (name, sound) of every channel of every
    factory kit, in machine then channel order ("505 BD", ...), then the
    drum patches of drum_patches.json the kits leave out, in name order."""
    found = [(sound['name'], sound) for kit in kits() for sound in kit['channels']]
    if PATCHES_FILE.is_file():
        with open(PATCHES_FILE, encoding='utf-8') as f:
            more = json.load(f)['patches']
        names = {name for name, _ in found}
        found += sorted(((s['name'], s) for s in more if s['name'] not in names), key=lambda p: p[0])
    return found


def patch(name):
    """The sound of the factory drum patch called ``name``."""
    for patch_name, sound in patches():
        if patch_name == name:
            return sound
    raise ValueError(f'not a factory drum patch: {name!r}')
