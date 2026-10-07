"""
The tkinter GUI on the core's PO-32 module (slice 9): the transfer and import
dialogs keep their UI and drive po32.* verbs; no audio stream, engine write
or preferences access is left in them. Skips without a display.
"""

import pathlib
import time
from unittest import mock

import numpy as np
import pytest

from pythonic.po32_codec import save_wav
from pythonic.po32_decoder import decode_wav_file
from tests.test_po32_core import CARD_PATTERNS, CARD_SOUNDS, card_signal
from tests.test_tk_presets import close, make_app
from tests.test_tk_smoke import _display_available

pytestmark = pytest.mark.skipif(not _display_available(), reason='no display for tkinter')

GUI = pathlib.Path(__file__).resolve().parent.parent / 'gui'


@pytest.fixture
def app(prefs):
    gui = make_app(prefs)
    yield gui
    close(gui)


@pytest.fixture(scope='module')
def card():
    return card_signal(CARD_SOUNDS, CARD_PATTERNS)


def pump(gui, until, timeout=30.0):
    """Pull output blocks and run the Tk loop until until() is true."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        gui.backend.stream.pull()
        gui.root.update()
        if until():
            return
        time.sleep(0.002)
    raise AssertionError('timed out')


def test_the_old_po32_paths_are_gone():
    for name in ('po32_transfer.py', 'po32_import_dialog.py'):
        source = (GUI / name).read_text(encoding='utf-8')
        for old in ('sounddevice', 'sd.', 'OutputStream', 'InputStream', 'set_parameters',
                    'pattern_manager', 'preferences_manager', 'morph_manager', 'synth',
                    'apply_change', 'threading', 'po32_codec', 'po32_decoder'):
            assert old not in source, (name, old)
    main = (GUI / 'main_window.py').read_text(encoding='utf-8')
    assert 'root.morph_manager' not in main


def test_the_transfer_dialog_sends_through_the_core(app, tmp_path):
    from gui.po32_transfer import PO32TransferDialog

    app.core.set('ch2.mute', True)
    pump(app, lambda: app.core.get('ch2.mute'))
    dialog = PO32TransferDialog(app.root, app.core, when_done=app._when_action_done)
    assert not dialog.channel_vars[1].get()  # the checkboxes start from the face mutes
    pump(app, lambda: 'Ready to transfer' in dialog.status_label.cget('text'))
    assert dialog.seconds > 5.0

    target = tmp_path / 'transfer.wav'
    with mock.patch('gui.po32_transfer.filedialog.asksaveasfilename', return_value=str(target)):
        dialog._on_save_wav()
    pump(app, lambda: 'Saved' in dialog.status_label.cget('text'))
    preset = decode_wav_file(str(target))
    assert preset.error is None
    assert preset.left_patches[1].raw_params == b'\x00\x80' * 21  # unchecked: sent silent

    dialog._on_transfer_click()
    assert app.core.get('po32.transfer') in ('sending', 'ready')
    pump(app, lambda: dialog.status_label.cget('text') == 'Transfer complete!')
    assert app.core.get('po32.transfer') == 'sent'
    dialog._on_close()


def test_the_transfer_dialog_stops_a_send(app):
    from gui.po32_transfer import PO32TransferDialog

    dialog = PO32TransferDialog(app.root, app.core, when_done=app._when_action_done)
    dialog._on_transfer_click()
    pump(app, lambda: app.core.get('po32.progress') > 0.05)
    dialog._on_transfer_click()
    pump(app, lambda: dialog.status_label.cget('text') == 'Transfer stopped')
    assert app.core.get('po32.transfer') == 'stopped'
    dialog._on_close()


def test_the_import_dialog_decodes_picks_and_imports(app, tmp_path, card):
    from gui.po32_import_dialog import PO32ImportDialog

    path = tmp_path / 'card.wav'
    save_wav(card, str(path))
    dialog = PO32ImportDialog(app.root, app.core, when_done=app._when_action_done)
    with mock.patch('gui.po32_import_dialog.filedialog.askopenfilename', return_value=str(path)):
        dialog._on_import_wav()
    pump(app, lambda: 'Decoded PO-32 card' in dialog.source_label.cget('text'))
    assert dialog.pattern_buttons[0].cget('text') == '1→A'
    assert dialog.selection_count_label.cget('text') == '3/12 patterns selected'
    assert dialog.import_btn.cget('text') == 'Import Drums + 3 Patterns'

    dialog._on_pattern_select(1)  # unpick pattern 2
    pump(app, lambda: dialog.selection_count_label.cget('text') == '2/12 patterns selected')
    assert dialog.pattern_buttons[1].cget('text') == '2'

    with mock.patch('gui.po32_import_dialog.messagebox.showinfo') as info:
        dialog._on_import()
        pump(app, lambda: info.called)
    assert app.core.get('ch1.osc.freq') == pytest.approx(60.0, rel=0.02)
    assert app.core.get('pattern.A.ch1.trig')[:4] == [True, False, False, False]
    assert app.core.get('pattern.C.ch4.trig')[:4] == [True, False, False, False]
    assert app.core.get('undo.can_undo')
    assert not dialog.dialog.winfo_exists()


def test_the_import_dialog_records_through_the_core(app, card):
    from gui.po32_import_dialog import PO32ImportDialog

    dialog = PO32ImportDialog(app.root, app.core, when_done=app._when_action_done)
    with mock.patch('gui.po32_import_dialog.messagebox.showerror') as error:
        try:
            record_and_preview(app, dialog, card)
        finally:
            dialog._on_cancel()
    assert not error.called, error.call_args


def record_and_preview(app, dialog, card):
    dialog._on_toggle_record()
    pump(app, lambda: app.core.get('po32.recording'))
    app.backend.input.push(np.concatenate([np.zeros(4096), card * 0.5]))
    pump(app, lambda: dialog.vu_db_label.cget('text') != '-∞ dB')
    dialog._on_toggle_record()
    pump(app, lambda: 'Decoded PO-32 card' in dialog.source_label.cget('text'))
    assert app.backend.input.closed

    dialog._on_toggle_preview()
    pump(app, lambda: dialog.preview_btn.cget('text') == '■ Stop')
    dialog._on_cancel()
    pump(app, lambda: not app.core.get('po32.previewing'))
    assert not app.core.get('po32.listening')
