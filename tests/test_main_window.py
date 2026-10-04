"""
Tests for main window handlers, run against stubs so no Tk window is needed.
"""

import types


def test_midi_program_change_selects_like_a_button_click():
    from gui.main_window import PythonicGUI

    selected = []
    stub = types.SimpleNamespace(
        root=types.SimpleNamespace(after=lambda ms, fn: fn()),
        _on_pattern_select=selected.append)

    PythonicGUI._on_midi_pattern_select(stub, 4)
    PythonicGUI._on_midi_pattern_select(stub, 12)

    assert selected == [4]
