"""
The tkinter GUI on the core's export verbs (slice 7): the pattern menu's MIDI
and audio exports and the preset menu's drum WAV exports keep their dialogs
in tkinter and hand the chosen path (and tail) to the core. Skips without a
display.
"""

import pathlib
import time
import wave
from unittest import mock

import pytest

from tests.test_tk_presets import close, make_app
from tests.test_tk_smoke import _display_available

pytestmark = pytest.mark.skipif(not _display_available(), reason='no display for tkinter')

MAIN_WINDOW = pathlib.Path(__file__).resolve().parent.parent / 'gui' / 'main_window.py'


@pytest.fixture
def app(prefs):
    gui = make_app(prefs)
    yield gui
    close(gui)


def finished(gui, verb, version=None):
    """Pull blocks until the core reports the end of an action of `verb` (newer
    than `version`), then tick."""
    if version is None:
        version = gui.core.poll()['version']
    end = time.monotonic() + 30.0
    while time.monotonic() < end:
        gui.backend.stream.pull()
        events = [e for e in gui.core.poll(version)['events']
                  if e.get('verb') == verb and e['status'] != 'progress']
        if events:
            gui.root.update()
            gui._ui_update_tick()
            return events[0]
        time.sleep(0.002)
    raise AssertionError(f'{verb} did not finish')


def test_the_old_export_paths_are_gone():
    source = MAIN_WINDOW.read_text()
    for old in ('StepSequencer', 'process_audio_events', 'pattern_midi_file', 'wave.open',
                'export_drum_to_wav', 'export_all_drums_to_wav', 'is_playing = True'):
        assert old not in source, old


def test_the_pattern_menu_exports_midi_through_the_core(app, tmp_path):
    app.core.set('pattern.B.ch1.step1.trig', True)
    target = tmp_path / 'b.mid'
    version = app.core.poll()['version']
    with mock.patch('gui.main_window.filedialog.asksaveasfilename', return_value=str(target)):
        app._pattern_menu_action('export_midi', 1)
    event = finished(app, 'export.midi', version)
    assert event['status'] == 'done' and event['result']['pattern'] == 'B'
    assert target.is_file()


def test_the_pattern_menu_exports_audio_with_the_chosen_tail(app, tmp_path):
    import tkinter as tk
    app.core.set('pattern.A.ch1.step1.trig', True)
    target = tmp_path / 'a.wav'

    def choose_loop(dialog):
        radios = []
        stack = [dialog]
        while stack:
            widget = stack.pop()
            radios += [w for w in widget.winfo_children() if isinstance(w, tk.Radiobutton)]
            stack += widget.winfo_children()
        next(r for r in radios if r.cget('text').startswith('Loop')).invoke()
        dialog.destroy()

    version = app.core.poll()['version']
    with mock.patch.object(app.root, 'wait_window', side_effect=choose_loop), \
            mock.patch('gui.main_window.filedialog.asksaveasfilename', return_value=str(target)):
        app._pattern_menu_action('export_audio', 0)
    event = finished(app, 'export.wav', version)
    assert event['status'] == 'done', event
    assert event['result']['tail'] == 'loop'
    with wave.open(str(target)) as f:
        assert f.getnframes() == event['result']['frames']


def test_the_preset_menu_exports_drums_and_asks_before_replacing(app, tmp_path):
    target = tmp_path / 'drum.wav'
    app.core.set('global.channel', 3)
    app.backend.stream.pull()
    app._ui_update_tick()
    version = app.core.poll()['version']
    with mock.patch('gui.main_window.filedialog.asksaveasfilename', return_value=str(target)):
        app._export_current_drum()
    assert finished(app, 'export.drum_wav', version)['result']['channel'] == 3
    assert target.is_file()

    folder = tmp_path / 'all'
    version = app.core.poll()['version']
    with mock.patch('gui.main_window.filedialog.askdirectory', return_value=str(folder)):
        app._export_all_wavs()
    assert finished(app, 'export.drum_wavs', version)['result']['saved']
    assert len(list(folder.glob('*.wav'))) == 8

    with mock.patch('gui.main_window.filedialog.askdirectory', return_value=str(folder)), \
            mock.patch('gui.main_window.messagebox.askyesno', return_value=True) as ask:
        version = app.core.poll()['version']
        app._export_all_wavs()
        refused = finished(app, 'export.drum_wavs', version)  # its tick asks, then exports again
        assert refused['result']['exists'] and not refused['result']['saved']
        assert ask.called
        assert finished(app, 'export.drum_wavs', refused['version'])['result']['saved']
