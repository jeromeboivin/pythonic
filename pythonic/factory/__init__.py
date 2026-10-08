"""
The factory content shipped with Pythonic, read-only.

- ``presets/``: one JSON preset per drum machine (505, 707, 808, 909, DMX,
  LM2): that machine's sounds and twelve patterns, with all six machines'
  sounds in programs 1-6. Built by a local script (not in the repository).
"""

import json
import os
from pathlib import Path

PRESETS_DIR = Path(__file__).resolve().parent / 'presets'
MACHINES = ('505', '707', '808', '909', 'DMX', 'LM2')


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
    (programs 1-6 of every factory preset)."""
    files = preset_files()
    if not files:
        return []
    with open(PRESETS_DIR / files[0], encoding='utf-8') as f:
        slots = json.load(f)['programs']['slots']
    return [slots[str(i)] for i in range(len(MACHINES))]
