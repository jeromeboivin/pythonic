"""
The tkinter GUI on the core's patterns (slice 4): the pattern buttons, lane
and matrix editors, pattern menu, lane clipboard and chain buttons go through
core.set() and core.act(), and the editors and buttons are refreshed from what
core.poll() reports. Skips without a display.
"""

import pathlib
import time

import pytest

from pythonic.app import AppCore
from tests.fake_audio import FakeAudioBackend
from tests.test_tk_smoke import _display_available

pytestmark = pytest.mark.skipif(not _display_available(), reason='no display for tkinter')

MAIN_WINDOW = pathlib.Path(__file__).resolve().parents[1] / 'gui' / 'main_window.py'


@pytest.fixture
def app(prefs):
    from gui.main_window import PythonicGUI

    prefs.set('midi_enabled', False)
    backend = FakeAudioBackend()
    core = AppCore(preferences=prefs, audio_backend=backend, stall_timeout=None)
    gui = PythonicGUI(core=core)
    core.wait(gui._audio_start_action)
    gui.backend = backend
    yield gui

    def close():
        for job in gui.root.tk.splitlist(gui.root.tk.call('after', 'info')):
            gui.root.after_cancel(job)
        gui.root.destroy()
    gui.root.after(10, close)
    gui.run()


def tick(gui):
    """One audio block (queued sets land), then one UI tick (poll)."""
    gui.backend.stream.pull()
    gui._ui_update_tick()


def settle(gui, action_id=None):
    """Pull blocks and tick until an action (or the newest one) has finished."""
    end = time.monotonic() + 5.0
    while time.monotonic() < end:
        tick(gui)
        if action_id is None or action_id in gui.core._results:
            tick(gui)
            return
        time.sleep(0.002)
    raise AssertionError('action did not finish')


def last_action(gui):
    return max(gui.core._results, default=None)


def run_verb_from(gui, handler, *args):
    """Call a GUI handler that starts a core verb and wait until it is shown."""
    before = gui.core.act('pattern.queue', pattern=None)  # an id to compare with
    settle(gui, before)
    handler(*args)
    settle(gui, before + 1)


def test_no_pattern_handler_writes_the_patterns_directly():
    source = MAIN_WINDOW.read_text()
    for name in ('get_selected_pattern', 'clipboard_data', 'toggle_chain', '.select_pattern(',
                 'shift_pattern', 'set_trigger(', '.cut_pattern(', '.paste_pattern(',
                 'queued_pattern_index', '.randomize_pattern(', 'set_length('):
        assert name not in source, name


def test_pattern_button_selects_and_editors_follow(app):
    app.core.set('pattern.C.ch1.step2.trig', True)
    run_verb_from(app, app._on_pattern_select, 2)
    assert app.core.get('pattern.selected') == 'C'
    assert app._pattern == 2
    assert app.pattern_editors[0].triggers[1] is True
    assert app.pattern_buttons[2].cget('bg') == app.COLORS['highlight']


def test_lane_edits_go_through_the_core(app):
    app._on_pattern_edit(0, 4, 'trig', True)
    app._on_pattern_edit(0, 4, 'acc', True)
    app._on_pattern_edit(0, 4, 'prob', 30)
    app._on_pattern_edit(0, 4, 'sub', 'oo')
    tick(app)
    assert app.core.get('pattern.A.ch1.step5.trig') is True
    assert app.core.get('pattern.A.ch1.step5.acc') is True
    assert app.core.get('pattern.A.ch1.step5.prob') == 30
    assert app.core.get('pattern.A.ch1.step5.sub') == 'oo'
    assert app._undo_stack  # each edit is an undo step

    app._on_pattern_edit_all(0, 'trig', True, set())
    tick(app)
    assert all(app.core.get(f'pattern.A.ch{c}.step1.trig') for c in range(1, 9))
    assert all(editor.triggers[0] for editor in app.pattern_editors)

    app._on_matrix_edit(3, 7, True)
    tick(app)
    assert app.core.get('pattern.A.ch4.step8.trig') is True


def test_a_lane_changed_elsewhere_refreshes_the_editor(app):
    app.core.set('pattern.A.ch3.step9.trig', True)
    tick(app)
    assert app.pattern_editors[2].triggers[8] is True
    # Not while that editor is dragged: it refreshes when the drag ends
    app.pattern_editors[2].dragging = True
    app.core.set('pattern.A.ch3.step10.trig', True)
    tick(app)
    assert app.pattern_editors[2].triggers[9] is False
    app.pattern_editors[2].dragging = False
    tick(app)
    assert app.pattern_editors[2].triggers[9] is True


def test_length_lane_sets_the_pattern_length(app):
    app._on_pattern_length_change(12)
    tick(app)
    assert app.core.get('pattern.A.length') == 12
    assert all(editor.pattern_length == 12 for editor in app.pattern_editors)
    run_verb_from(app, app._on_pattern_select, 1)
    assert all(editor.pattern_length == 16 for editor in app.pattern_editors)


def test_pattern_menu_runs_core_verbs(app):
    app.core.set('pattern.D.ch2.step1.trig', True)
    run_verb_from(app, app._pattern_menu_action, 'shift_right', 3)
    assert app.core.get('pattern.D.ch2.step2.trig') is True
    run_verb_from(app, app._pattern_menu_action, 'copy_pattern', 3)
    run_verb_from(app, app._pattern_menu_action, 'paste_pattern', 0)
    assert app.pattern_editors[1].triggers[1] is True  # A is shown and was refreshed
    assert app.pattern_buttons[0].cget('fg') == app.COLORS['text']  # no longer empty


def test_lane_clipboard_copy_and_paste(app, monkeypatch):
    from tkinter import messagebox
    warnings = []
    monkeypatch.setattr(messagebox, 'showwarning', lambda *a: warnings.append(a))
    run_verb_from(app, app._on_pattern_paste)
    assert warnings  # nothing in the clipboard yet

    app.core.set('pattern.A.ch1.step3.trig', True)
    tick(app)
    run_verb_from(app, app._on_pattern_copy)
    app.core.set('global.channel', 5)
    tick(app)
    run_verb_from(app, app._on_pattern_paste)
    assert app.core.get('pattern.A.ch5.step3.trig') is True
    assert app.pattern_editors[4].triggers[2] is True
    assert len(warnings) == 1


def test_chain_buttons_and_chained_colour(app):
    for name in 'ABC':
        app.core.set(f'pattern.{name}.ch1.step1.trig', True)
    run_verb_from(app, app._on_chain_next)
    assert app.core.get('pattern.A.chained') is True
    for i in (0, 1):
        assert app.pattern_buttons[i].cget('fg') == app.COLORS['highlight']
    assert app.pattern_buttons[2].cget('fg') == app.COLORS['text']
    run_verb_from(app, app._on_pattern_select, 1)
    run_verb_from(app, app._on_chain_previous)
    assert app.core.get('pattern.A.chained') is False


def test_play_and_stop_buttons_go_through_the_core(app):
    import numpy as np

    app.core.set('global.tempo', 300)
    for step in (1, 5, 9, 13):
        app.core.set(f'pattern.A.ch1.step{step}.trig', True)
    run_verb_from(app, app._on_pattern_play)
    assert app.core.poll()['transport']['playing'] is True
    assert app.play_btn.active and not app.stop_btn.active
    peak = 0.0
    positions = set()
    for _ in range(40):
        peak = max(peak, float(np.abs(app.backend.stream.pull()).max()))
        app._ui_update_tick()
        positions.add(app.pattern_editors[0].current_position)
    assert peak > 0.01 and len(positions) > 3

    run_verb_from(app, app._on_pattern_stop)
    assert app.core.poll()['transport']['playing'] is False
    assert app.stop_btn.active and not app.play_btn.active
    assert all(editor.current_position == 0 for editor in app.pattern_editors)


def test_chain_moves_the_shown_pattern(app):
    app.core.set('global.tempo', 300)
    app.core.set('pattern.A.length', 2)
    app.core.set('pattern.B.ch2.step1.trig', True)
    run_verb_from(app, app._on_chain_next)
    run_verb_from(app, app._on_pattern_play)
    for _ in range(200):
        tick(app)
        if app._pattern == 1:
            break
    assert app._pattern == 1
    assert app.pattern_editors[1].triggers[0] is True


def test_number_keys_trigger_the_channel_pads(app):
    import types

    import numpy as np

    app._on_key_press(types.SimpleNamespace(char='3'))
    peak = max(float(np.abs(app.backend.stream.pull()).max()) for _ in range(4))
    assert peak > 0.01


def test_midi_export_writes_a_long_pattern_with_velocities(app, monkeypatch, tmp_path):
    import mido
    from tkinter import filedialog

    path = tmp_path / 'b.mid'
    monkeypatch.setattr(filedialog, 'asksaveasfilename', lambda **kw: str(path))
    app.core.set('pattern.B.length', 64)
    app.core.set('pattern.B.ch1.step50.trig', True)
    app.core.set('pattern.B.ch1.step50.vel', 101)
    tick(app)
    app._export_pattern_to_midi(1)
    now, ons = 0, []
    for message in mido.MidiFile(str(path)).tracks[0]:
        now += message.time
        if message.type == 'note_on':
            ons.append((now, message.velocity))
    assert ons == [(49 * 120, 101)]
