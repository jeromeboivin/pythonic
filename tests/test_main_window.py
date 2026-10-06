"""
Tests for main window handlers, run against stubs so no Tk window is needed.
"""

import collections
import types

from pythonic.app import AppCore


def _gui_stub(is_playing):
    from gui.main_window import PythonicGUI

    stub = types.SimpleNamespace()
    stub.scheduled = []
    stub.updates = 0
    stub.button_flash_state = False
    stub._transport = {'playing': is_playing}
    stub.root = types.SimpleNamespace(after=lambda ms, fn: stub.scheduled.append(fn))

    def update():
        stub.updates += 1
    stub._update_pattern_button_states = update
    stub._toggle_button_flash = types.MethodType(PythonicGUI._toggle_button_flash, stub)
    return stub


def test_button_flash_keeps_a_single_timer_chain():
    for playing in (True, False):
        stub = _gui_stub(playing)
        stub.button_flash_state = True
        for _ in range(5):
            stub._toggle_button_flash()
        assert len(stub.scheduled) == 5, playing


def test_pattern_button_update_schedules_no_timer(prefs):
    from gui.main_window import PythonicGUI

    scheduled = []
    core = AppCore(preferences=prefs, audio_backend=None, midi_backend=None)
    btn = types.SimpleNamespace(config=lambda **kw: None)
    stub = types.SimpleNamespace(
        core=core, _transport=core.poll()['transport'], _pattern=0,
        pattern_buttons=[btn] * 12, button_flash_state=False,
        COLORS=collections.defaultdict(str),
        root=types.SimpleNamespace(after=lambda ms, fn: scheduled.append(fn)))
    try:
        PythonicGUI._update_pattern_button_states(stub)
    finally:
        core.close()

    assert scheduled == []
