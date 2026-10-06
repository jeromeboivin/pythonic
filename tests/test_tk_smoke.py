"""
Headless smoke test of the tkinter GUI on top of the app core: build
PythonicGUI, run one UI tick, close. Skips when no display is available.
"""

import os
import sys

import pytest

from pythonic.app import AppCore
from tests.fake_audio import FakeAudioBackend


def _display_available():
    try:
        import tkinter as tk
    except ImportError:
        return False
    if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
        return False
    try:
        root = tk.Tk()
        root.destroy()
    except tk.TclError:
        return False
    return True


@pytest.mark.skipif(not _display_available(), reason='no display for tkinter')
def test_tkinter_gui_builds_ticks_and_closes(prefs):
    from gui.main_window import PythonicGUI

    prefs.set('midi_enabled', False)
    backend = FakeAudioBackend()
    core = AppCore(preferences=prefs, audio_backend=backend, stall_timeout=None)
    gui = PythonicGUI(core=core)

    core.wait(gui._audio_start_action)
    assert core.get('audio.running') is True
    play = core.act('transport.play')
    while play not in core._results:  # the verb waits for a block start
        backend.stream.pull()
    assert core.wait(play)['status'] == 'done'
    gui._ui_update_tick()
    assert gui._poll_version > 0

    gui.root.after(120, gui.root.destroy)
    gui.run()

    assert backend.stream.aborted
    assert core.get('audio.running') is False
