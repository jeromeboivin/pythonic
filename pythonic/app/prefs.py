"""
Preferences of the app core (inventory cluster P): the saved settings as
``pref.*`` addresses. A set saves the preferences file at once (the keys and
values tkinter has always written, so older versions read it) and applies
what applies live. Not undoable; ``set`` runs on the caller's thread.

- ``pref.preset_folder`` (path): the folder of ``preset.files`` and of
  relative preset paths; set to an existing folder only.
- ``pref.recent_files`` (read-only list): recently loaded or saved presets.
"""

import os

from .registry import Address


class Prefs:
    """Preference addresses of an AppCore."""

    def __init__(self, core):
        self._core = core

    @property
    def manager(self):
        return self._core.preferences

    def register(self, registry):
        reg = registry.register
        reg(Address('pref.preset_folder', get=lambda: self.manager.get_preset_folder(),
                    set=self._set_preset_folder, kind='str', queued=False, undoable=False,
                    related=('preset.files',)))
        reg(Address('pref.recent_files', get=lambda: self.manager.get_recent_files(),
                    kind='list'))

    def _set_preset_folder(self, folder):
        folder = os.path.abspath(os.path.expanduser(str(folder)))
        if not os.path.isdir(folder):
            raise ValueError(f'pref.preset_folder: {folder} is not a folder')
        self.manager.set_preset_folder(folder)
