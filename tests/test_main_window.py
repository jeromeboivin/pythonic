"""
Tests for main window handlers, run against stubs so no Tk window is needed.
"""

import collections
import types

from pythonic.pattern_manager import PatternManager


def test_midi_program_change_selects_like_a_button_click():
    from gui.main_window import PythonicGUI

    selected = []
    stub = types.SimpleNamespace(
        root=types.SimpleNamespace(after=lambda ms, fn: fn()),
        _on_pattern_select=selected.append)

    PythonicGUI._on_midi_pattern_select(stub, 4)
    PythonicGUI._on_midi_pattern_select(stub, 12)

    assert selected == [4]


def _gui_stub(is_playing):
    from gui.main_window import PythonicGUI

    stub = types.SimpleNamespace()
    stub.scheduled = []
    stub.updates = 0
    stub.button_flash_state = False
    stub.pattern_manager = types.SimpleNamespace(is_playing=is_playing)
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


def test_pattern_button_update_schedules_no_timer():
    from gui.main_window import PythonicGUI

    scheduled = []
    pm = PatternManager(num_channels=8, pattern_length=16)
    btn = types.SimpleNamespace(config=lambda **kw: None)
    stub = types.SimpleNamespace(
        pattern_manager=pm, pattern_buttons=[btn] * 12, button_flash_state=False,
        COLORS=collections.defaultdict(str),
        root=types.SimpleNamespace(after=lambda ms, fn: scheduled.append(fn)))

    PythonicGUI._update_pattern_button_states(stub)

    assert scheduled == []
