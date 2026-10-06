"""
The tkinter GUI on the core's MIDI input (slice 3): no MIDI thread or routing
of its own; learn, mappings and pitch bend go through the core, and the LEDs,
channel flashes, pattern buttons and undo steps follow what core.poll()
reports. Skips without a display.
"""

import pathlib
import time

import pytest

from pythonic.app import AppCore
from tests.fake_audio import FakeAudioBackend
from tests.fake_midi import FakeMidiBackend, cc, note_on, program_change
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

    backend = FakeAudioBackend()
    midi = FakeMidiBackend()
    clock = FakeClock()
    core = AppCore(preferences=prefs, audio_backend=backend, midi_backend=midi,
                   clock=clock, stall_timeout=None)
    gui = PythonicGUI(core=core)
    core.wait(gui._audio_start_action)
    assert core.wait(core.act('midi.open'))['status'] == 'done'
    gui.backend, gui.midi, gui.clock = backend, midi, clock
    yield gui

    def close():
        for job in gui.root.tk.splitlist(gui.root.tk.call('after', 'info')):
            gui.root.after_cancel(job)
        gui.root.destroy()
    gui.root.after(10, close)
    gui.run()


def send(gui, *msgs):
    for msg in msgs:
        gui.midi.port.send(msg)
    gui.core.midi.sync()
    gui.backend.stream.pull()
    gui._ui_update_tick()


def test_the_gui_has_no_midi_thread_or_routing_of_its_own():
    source = MAIN_WINDOW.read_text()
    for name in ('MidiManager', 'midi_manager', '_apply_midi_cc', '_apply_midi_pitchbend',
                 '_on_midi_pattern_select', 'trigger_drum'):
        assert name not in source
    assert not (MAIN_WINDOW.parents[1] / 'pythonic' / 'midi_manager.py').exists()


def test_learn_from_the_context_menu_maps_the_next_cc(app):
    app._start_midi_learn('osc_decay', app.osc_decay_knob)
    app.core.wait(app.core.act('midi.rescan'))  # the learn verb has run
    assert app.core.get('midi.learning') == 'selected.osc.decay'
    assert app._midi_learn_flash_id is not None

    send(app, cc(21, 10))

    assert app.core.get('midi.cc_map')[21] == 'selected.osc.decay'
    assert app._midi_learn_action is None and app._midi_learn_flash_id is None


def test_cancel_learn_stops_the_flashing(app):
    app._start_midi_learn('level', app.level_knob)
    app._cancel_midi_learn()
    app.core.wait(app.core.act('midi.rescan'))
    app._ui_update_tick()
    assert app._midi_learn_action is None and app._midi_learn_flash_id is None
    assert app.core.get('midi.learning') is None


def test_mappings_and_pitch_bend_are_core_state(app):
    app.core.set('midi.cc_map', {30: 'selected.osc.decay', 31: 'global.master'})
    app._remove_cc_mapping('osc_decay')
    app._assign_pitchbend('noise_freq')
    assert app.core.get('midi.cc_map') == {31: 'global.master'}
    assert app.core.get('midi.pitchbend_target') == 'selected.noise.freq'
    app._remove_pitchbend_mapping()
    assert app.core.get('midi.pitchbend_target') is None


def test_cc_moves_the_knob_through_poll_after_pickup(app):
    app.core.set('midi.cc_map', {20: 'selected.mix.level'})
    send(app, cc(20, 0), cc(20, 127))  # crosses the value: picked up
    assert app.level_knob.get_value() == 10.0


def test_a_cc_burst_is_one_undo_step(app):
    app.core.set('midi.cc_map', {20: 'selected.mix.level'})
    steps = len(app._undo_stack)
    send(app, cc(20, 0), cc(20, 127), cc(20, 100))
    app.clock.t += 0.5
    app._ui_update_tick()
    assert len(app._undo_stack) == steps + 1

    app._on_undo()
    end = time.monotonic() + 5.0
    while app._restore_pending and time.monotonic() < end:
        app.backend.stream.pull()  # the restore waits for block start
        app._ui_update_tick()
        time.sleep(0.002)
    assert app.core.get('ch1.mix.level') == 0.0
    assert app.level_knob.get_value() == 0.0


def test_notes_flash_the_channel_and_program_change_selects_the_pattern(app):
    send(app, note_on(36 + 2), program_change(5))
    assert app.channel_buttons[2].triggered
    assert app.pattern_buttons[5].cget('bg') == app.COLORS['highlight']
    assert app.pattern_buttons[0].cget('bg') == app.COLORS['bg_light']
