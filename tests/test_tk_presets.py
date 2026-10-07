"""
The tkinter GUI on the core's preset and drum-patch files and preferences
(slice 6): the preset menu, the preset list, the start-up preset and the
settings dialogs go through the core; the views follow poll. Skips without a
display.
"""

import pathlib
import shutil
import time
from unittest import mock

import pytest

from pythonic.app import AppCore
from tests.fake_audio import FakeAudioBackend
from tests.test_tk_smoke import _display_available

pytestmark = pytest.mark.skipif(not _display_available(), reason='no display for tkinter')

TESTS = pathlib.Path(__file__).resolve().parent
MAIN_WINDOW = TESTS.parent / 'gui' / 'main_window.py'


def make_app(prefs):
    from gui.main_window import PythonicGUI

    backend = FakeAudioBackend()
    core = AppCore(preferences=prefs, audio_backend=backend, stall_timeout=None)
    gui = PythonicGUI(core=core)
    core.wait(gui._audio_start_action)
    gui.backend = backend
    return gui


def close(gui):
    def stop():
        for job in gui.root.tk.splitlist(gui.root.tk.call('after', 'info')):
            gui.root.after_cancel(job)
        gui.root.destroy()
    gui.root.after(10, stop)
    gui.run()


@pytest.fixture
def folder(prefs):
    path = pathlib.Path(prefs.get_preset_folder())
    path.mkdir(parents=True, exist_ok=True)
    shutil.copy(TESTS / '808.mtpreset', path / '808.mtpreset')
    shutil.copy(TESTS / '909.mtpreset', path / '909.mtpreset')
    return path


@pytest.fixture
def app(prefs, folder):
    gui = make_app(prefs)
    yield gui
    close(gui)


def wait_for(gui, verb):
    """Pull blocks until the core reports an action of `verb`, then tick."""
    version = gui.core.poll()['version']
    end = time.monotonic() + 5.0
    while time.monotonic() < end:
        gui.backend.stream.pull()
        events = [e for e in gui.core.poll(version)['events'] if e.get('verb') == verb]
        if events:
            gui.root.update()
            gui._ui_update_tick()
            return events[0]
        time.sleep(0.002)
    raise AssertionError(f'{verb} did not finish')


def test_the_old_preset_paths_are_gone():
    source = MAIN_WINDOW.read_text()
    for old in ('_read_preset_file', 'load_mtpreset', 'json.dump', '_preset_clipboard',
                'load_programs_data', '_init_endpoints', '_after_preset_replaced',
                'get_preset_folder', 'add_recent_file', 'preferences_manager.set(',
                'preferences_manager.get(', 'sd.', 'set_smoothing_time'):
        assert old not in source, old


def test_the_preset_list_loads_through_the_core(app, folder):
    assert list(app.preset_combo['values']) == ['808.mtpreset', '909.mtpreset']
    assert app.preset_combo.get() == ''
    app.preset_combo.set('909.mtpreset')
    app._on_preset_combo_select()
    event = wait_for(app, 'preset.load')
    assert event['status'] == 'done'
    assert app.core.get('preset.path') == str(folder / '909.mtpreset')
    assert app.preset_combo.get() == '909.mtpreset'
    assert app.osc_decay_knob.get_value() == pytest.approx(app.core.get('ch1.osc.decay'))
    assert app.patch_name_label.cget('text') == app.core.get('ch1.name')
    assert app.pattern_editors[0].triggers == app.core.get('pattern.A.ch1.trig')
    assert app.bpm_var.get() == str(app.core.get('global.tempo'))


def test_the_last_preset_loads_at_start_up(prefs, folder):
    prefs.set('last_preset', str(folder / '808.mtpreset'))
    gui = make_app(prefs)
    try:
        # Acted before the stream starts: done once start-up has finished
        events = [e for e in gui.core.poll()['events'] if e.get('verb') == 'preset.load_last']
        assert events[0]['status'] == 'done' and events[0]['result']['loaded'] is True
        gui._ui_update_tick()
        assert gui.core.get('preset.name') == '808 Beats'
        assert gui.preset_combo.get() == '808.mtpreset'
        assert gui.core.get('undo.can_undo') is True
    finally:
        close(gui)


def test_save_and_drum_patch_menu_entries_go_through_the_core(app, folder, tmp_path):
    target = tmp_path / 'saved.json'
    with mock.patch('gui.main_window.filedialog.asksaveasfilename', return_value=str(target)):
        app._save_preset()
    assert wait_for(app, 'preset.save')['result']['saved'] is True
    assert target.is_file()

    patch = tmp_path / 'drum.mtdrum'
    app.core.set('ch1.osc.decay', 432.0)
    with mock.patch('gui.main_window.filedialog.asksaveasfilename', return_value=str(patch)):
        app._save_drum_patch()
    wait_for(app, 'drum_patch.save')
    app.core.set('global.channel', 4)
    app.backend.stream.pull()
    app._ui_update_tick()
    assert app.selected_channel == 3
    with mock.patch('gui.main_window.filedialog.askopenfilename', return_value=str(patch)):
        app._load_drum_patch()
    assert wait_for(app, 'drum_patch.load')['result']['channel'] == 4
    assert app.core.get('ch4.osc.decay') == pytest.approx(432.0)


def test_a_failed_load_shows_the_error(app, tmp_path):
    with mock.patch('gui.main_window.messagebox.showerror') as showerror:
        app._load_preset_file(str(tmp_path / 'missing.json'))
        wait_for(app, 'preset.load')
    assert showerror.called
    assert 'missing.json' in showerror.call_args[0][1]


def test_the_preset_menu_clipboard_follows_the_core(app):
    assert app.core.get('preset.clipboard') is False
    app._copy_preset()
    wait_for(app, 'preset.copy')
    assert app.core.get('preset.clipboard') is True


# ---------------------------------------------------------------------------
# settings dialogs
# ---------------------------------------------------------------------------

def dialog_of(gui):
    import tkinter as tk
    dialogs = [w for w in gui.root.winfo_children() if isinstance(w, tk.Toplevel)]
    assert dialogs
    return dialogs[-1]


def widgets(widget, kind):
    found = []
    for child in widget.winfo_children():
        if isinstance(child, kind):
            found.append(child)
        found += widgets(child, kind)
    return found


def button(dialog, text):
    import tkinter as tk
    return next(b for b in widgets(dialog, tk.Button) if b.cget('text') == text)


def test_the_audio_dialog_saves_through_the_core(app, prefs):
    from tkinter import ttk
    app._show_audio_preferences()
    dialog = dialog_of(app)
    device, input_device, buffer, rate, synth = widgets(dialog, ttk.Combobox)
    end = time.monotonic() + 5.0
    while not list(rate['values'])[0].startswith('48000') and time.monotonic() < end:
        app._ui_update_tick()  # until the core's probe of the device's rates is shown
        time.sleep(0.002)
    assert list(rate['values'])[0].startswith('48000')  # 96000 is not supported
    buffer.set(next(v for v in buffer['values'] if v.startswith('10 ms')))
    input_device.set('Fake In')
    button(dialog, 'OK').invoke()
    assert prefs.get('audio_buffer_ms') == 10.0
    assert prefs.get('audio_input_device') == 'Fake In'
    assert app.core.get('pref.audio.pending') == ['pref.audio.buffer_ms']
    assert app.core.get('audio.block_size') == 1050  # saved for later

    app._show_audio_preferences()
    dialog = dialog_of(app)
    button(dialog, 'Apply Now').invoke()
    event = wait_for(app, 'audio.apply')
    assert event['status'] == 'done'
    assert app.core.get('audio.block_size') == 441
    assert app.core.get('pref.audio.pending') == []
    dialog.destroy()


def test_the_synthesis_and_ai_dialogs_save_through_the_core(app, prefs):
    import tkinter as tk
    app._show_synthesis_preferences()
    dialog = dialog_of(app)
    widgets(dialog, tk.Scale)[0].set(12)
    button(dialog, 'OK').invoke()
    assert prefs.get('param_smoothing_ms') == 12.0
    app.backend.stream.pull()
    assert app.core.synth.channels[5].get_smoothing_time() == 12.0

    app._show_ai_preferences()
    dialog = dialog_of(app)
    entry = widgets(dialog, tk.Entry)[0]
    entry.delete(0, 'end')
    entry.insert(0, '/models/p.pt')
    button(dialog, 'OK').invoke()
    assert prefs.get('drum_generator_pattern_model_path') == '/models/p.pt'
    assert app.core.get('pref.ai.pattern_model') == '/models/p.pt'
