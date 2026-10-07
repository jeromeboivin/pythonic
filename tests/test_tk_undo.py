"""
The tkinter GUI on the core's undo journal (slice 5): the undo and redo
buttons run the core verbs and follow poll, a knob drag or a lane stroke is one
gesture, a wheel turn one burst, and the preset menu's bulk actions are one
step each. Skips without a display.
"""

import pathlib
import time

import pytest

from pythonic.app import AppCore
from tests.fake_audio import FakeAudioBackend
from tests.test_tk_smoke import _display_available

pytestmark = pytest.mark.skipif(not _display_available(), reason='no display for tkinter')

MAIN_WINDOW = pathlib.Path(__file__).resolve().parents[1] / 'gui' / 'main_window.py'


class FakeClock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


@pytest.fixture
def app(prefs):
    from gui.main_window import PythonicGUI

    prefs.set('midi_enabled', False)
    backend = FakeAudioBackend()
    clock = FakeClock()
    core = AppCore(preferences=prefs, audio_backend=backend, clock=clock, stall_timeout=None)
    gui = PythonicGUI(core=core)
    core.wait(gui._audio_start_action)
    gui.backend, gui.clock = backend, clock
    yield gui

    def close():
        for job in gui.root.tk.splitlist(gui.root.tk.call('after', 'info')):
            gui.root.after_cancel(job)
        gui.root.destroy()
    gui.root.after(10, close)
    gui.run()


def tick(gui):
    """One audio block, the Tk idle work (scale commands), one UI tick."""
    gui.backend.stream.pull()
    gui.root.update()
    gui._ui_update_tick()


def press(gui, button):
    """Click undo or redo and wait until the core has applied it."""
    version = gui.core.poll()['version']
    button.invoke()
    end = time.monotonic() + 5.0
    while time.monotonic() < end:
        gui.backend.stream.pull()
        events = gui.core.poll(version)['events']
        if any(e.get('verb') in ('undo', 'redo') for e in events):
            tick(gui)
            return
        time.sleep(0.002)
    raise AssertionError('undo did not finish')


def test_the_old_snapshot_undo_is_gone():
    source = MAIN_WINDOW.read_text()
    for old in ('_undo_stack', '_redo_stack', 'legacy', '_push_undo_state', 'cc_burst'):
        assert old not in source


def test_buttons_follow_the_journal(app):
    assert app.undo_btn.cget('state') == 'disabled'
    assert app.redo_btn.cget('state') == 'disabled'
    app.level_knob.set_value(-12.0)
    tick(app)
    assert app.undo_btn.cget('state') == 'normal'

    press(app, app.undo_btn)
    assert app.core.get('ch1.mix.level') == 0.0
    assert app.level_knob.get_value() == 0.0
    assert app.undo_btn.cget('state') == 'disabled'
    assert app.redo_btn.cget('state') == 'normal'
    press(app, app.redo_btn)
    assert app.level_knob.get_value() == -12.0


def test_a_knob_drag_is_one_step(app):
    knob = app.osc_decay_knob
    old = knob.get_value()
    knob.command_end('start')
    for value in (100.0, 200.0, 300.0):
        knob.set_value(value)
        tick(app)
    knob.command_end('end')
    press(app, app.undo_btn)
    assert app.core.get('ch1.osc.decay') == old
    assert app.core.get('undo.can_undo') is False


def test_a_wheel_turn_is_one_burst(app):
    knob = app.pan_knob
    for _ in range(5):
        knob._wheel_to(knob._value_to_normalized(knob.get_value()) + 0.01)
        app.clock.t += 0.1
    app.clock.t += 0.5
    tick(app)
    assert app.core.get('ch1.mix.pan') != 0.0
    press(app, app.undo_btn)
    assert app.core.get('ch1.mix.pan') == 0.0
    assert app.core.get('undo.can_undo') is False


def test_a_lane_stroke_is_one_step(app):
    editor = app.pattern_editors[0]
    editor.command_end('start')
    for step in range(4):
        app._on_pattern_edit(0, step, 'trig', True)
    editor.command_end('end')
    tick(app)
    assert app.core.get('pattern.A.ch1.trig')[:4] == [True] * 4
    press(app, app.undo_btn)
    assert app.core.get('pattern.A.ch1.trig')[:4] == [False] * 4
    assert editor.triggers[:4] == [False] * 4
    assert app.core.get('undo.can_undo') is False


def test_shift_click_on_every_channel_is_one_step(app):
    app._on_pattern_edit_all(2, 'trig', True, set())
    tick(app)
    press(app, app.undo_btn)
    assert not any(app.core.get(f'pattern.A.ch{c}.step3.trig') for c in range(1, 9))
    assert app.core.get('undo.can_undo') is False


def test_initialize_preset_is_one_step(app):
    app.osc_decay_knob.set_value(1234.0)
    app._on_pattern_edit(0, 0, 'trig', True)
    tick(app)
    app._init_preset()
    tick(app)
    assert app.core.get('pattern.A.ch1.step1.trig') is False

    press(app, app.undo_btn)
    assert app.core.get('ch1.osc.decay') == 1234.0
    assert app.osc_decay_knob.get_value() == 1234.0
    assert app.core.get('pattern.A.ch1.step1.trig') is True
    assert app.pattern_editors[0].triggers[0] is True


def test_a_morph_slider_shown_from_poll_is_no_step(app):
    app.core.synth.channels[0].set_osc_decay(50.0)
    app.morph_manager.capture_endpoint_b()  # endpoints differ: the slider is enabled
    app._update_morph_ui()
    app.core.set('morph.position', 0.4567, record=False)
    for _ in range(3):
        tick(app)
    assert app.morph_slider.get() == 46
    assert app.core.get('morph.position') == 0.4567  # no rounded echo written back
    assert app.core.get('undo.can_undo') is False


def test_the_swing_slider_is_undone(app):
    # The swing slider sits below the visible part of the test window, where
    # Tk holds back its command: call it as Tk would
    app._on_global_swing_change('30')
    tick(app)
    assert app.core.get('global.swing') == 0.3
    press(app, app.undo_btn)
    assert app.core.get('global.swing') == 0.0
    assert app.global_swing_slider.get() == 0
    app._on_global_swing_change('0')  # the slider's echo of the value shown
    tick(app)
    assert app.core.get('undo.can_redo') is True  # the echo wrote nothing


def wait_for(gui, verb):
    """Pull blocks until the core reports an action of `verb`, then tick."""
    version = gui.core.poll()['version']
    end = time.monotonic() + 5.0
    while time.monotonic() < end:
        gui.backend.stream.pull()
        if any(e.get('verb') == verb for e in gui.core.poll(version)['events']):
            tick(gui)
            return
        time.sleep(0.002)
    raise AssertionError(f'{verb} did not finish')


def test_program_switch_goes_through_the_core(app):
    app.osc_decay_knob.set_value(500.0)
    tick(app)
    app.program_var.set('2')
    app._on_program_select()
    wait_for(app, 'program.select')
    app.osc_decay_knob.set_value(900.0)
    tick(app)
    app.program_var.set('1')
    app._on_program_select()
    wait_for(app, 'program.select')
    assert app.core.get('program.current') == 1
    assert app.osc_decay_knob.get_value() == 500.0

    press(app, app.undo_btn)
    assert app.program_var.get() == '2'
    assert app.osc_decay_knob.get_value() == 900.0


def test_morph_learn_buttons_go_through_the_core(app):
    green = '#22cc55'
    app.morph_learn_a_btn.invoke()
    wait_for(app, 'morph.learn')
    assert app.core.get('morph.learning') == 'a'
    assert app.morph_learn_a_btn.cget('bg') == green
    assert app.morph_slider.cget('state') == 'normal'  # enabled while learning

    app.osc_decay_knob.set_value(80.0)
    tick(app)
    app.morph_learn_b_btn.invoke()  # A is captured, B is learned
    wait_for(app, 'morph.learn')
    assert app.core.get('morph.learning') == 'b'
    assert app.morph_learn_a_btn.cget('bg') != green
    assert app.morph_learn_b_btn.cget('bg') == green
    app.morph_learn_b_btn.invoke()
    wait_for(app, 'morph.learn')
    assert app.core.get('morph.learning') == 'off'
    assert app.core.get('morph.differs') is True
    assert app.morph_slider.cget('state') == 'normal'
    assert app.osc_decay_knob.get_value() == app.core.get('ch1.osc.decay') == 80.0
