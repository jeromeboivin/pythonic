"""
The tkinter GUI on the core's AI module (slice 8): the AI drum generator
dialog keeps its UI and drives ai.* verbs, the pattern menu's AI randomize
entries run ai.randomize_pattern, and nothing in the GUI process imports the
generators (the models run in the core's worker process). Runs against the
fake worker. Skips without a display.
"""

import pathlib
import sys
import time

import pytest

from pythonic.app import AppCore
from tests.fake_audio import FakeAudioBackend
from tests.test_tk_presets import close
from tests.test_tk_smoke import _display_available

pytestmark = pytest.mark.skipif(not _display_available(), reason='no display for tkinter')

ROOT = pathlib.Path(__file__).resolve().parent.parent
FAKE_WORKER = str(ROOT / 'tests' / 'fake_ai_worker.py')


@pytest.fixture
def app(prefs, tmp_path):
    from gui.main_window import PythonicGUI

    backend = FakeAudioBackend()
    core = AppCore(preferences=prefs, audio_backend=backend, stall_timeout=None,
                   ai_worker=[sys.executable, FAKE_WORKER])
    for kind in ('patch', 'pattern'):
        model = tmp_path / f'{kind}.pt'
        model.write_bytes(b'x')
        core.set(f'pref.ai.{kind}_model', str(model))
    gui = PythonicGUI(core=core)
    core.wait(gui._audio_start_action)
    gui.backend = backend
    yield gui
    close(gui)


def pump(gui, until, timeout=20.0):
    """Pull blocks and run the UI until until() holds."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        gui.backend.stream.pull()
        gui._ui_update_tick()
        gui.root.update()
        if until():
            return
        time.sleep(0.005)
    raise AssertionError('timed out')


def test_the_old_in_process_generation_paths_are_gone():
    for path in ('gui/main_window.py', 'gui/drum_generator_dialog.py'):
        source = (ROOT / path).read_text()
        for old in ('PatternGenerator', 'PatchGenerator', 'is_torch_available',
                    'install_ml_dependencies', 'apply_drum_patch_to_channel',
                    'channel_to_raw_patch', 'set_parameters', 'trigger_drum',
                    'apply_pattern_bank', 'apply_single_pattern'):
            assert old not in source, (path, old)


def test_the_dialog_generates_tries_keeps_and_closes(app):
    from gui.drum_generator_dialog import DrumGeneratorDialog

    core = app.core
    before = core.get('ch2.name')
    dialog = DrumGeneratorDialog(app.root, core, when_done=app._when_action_done)
    pump(app, lambda: dialog.model_status_label.cget('text').startswith('loaded'))
    dialog.candidates_var.set(3)
    dialog._on_generate_slot(0)
    dialog._on_generate_slot(1)
    pump(app, lambda: dialog.slot_widgets[1]['idx_label'].cget('text') == '1 / 3')
    assert dialog.slot_widgets[0]['name_label'].cget('text') == 'BD 1'
    assert core.get('ch1.name') == 'BD 1' and core.get('ch2.name') == 'SD 1'
    dialog._on_next_candidate(0)
    pump(app, lambda: dialog.slot_widgets[0]['idx_label'].cget('text') == '2 / 3')
    assert core.get('ch1.name') == 'BD 2'
    dialog.slot_widgets[1]['apply_var'].set(False)
    dialog._on_apply_selected()
    pump(app, lambda: core.get('ai.tried') == [2])
    dialog._toggle_preview('loop')
    pump(app, lambda: dialog.preview_loop_btn.cget('text') == 'Stop Loop')
    dialog._on_close()
    pump(app, lambda: core.get('ai.tried') == [] and core.get('ai.preview') == 'off')
    assert core.get('ch1.name') == 'BD 2'  # kept
    assert core.get('ch2.name') == before  # not kept: reverted
    assert core.get('undo.can_undo') is True


def test_the_pattern_menu_randomizes_with_the_core(app):
    core = app.core
    app._pattern_menu_action('ai_randomize_channel', 2)
    pump(app, lambda: core.get('pattern.C.ch1.trig')[:3] == [True, True, True])
    assert core.get('pattern.C.ch2.trig')[:2] == [False, False]  # only the selected channel
    app._pattern_menu_action('ai_randomize_pattern', 2)
    pump(app, lambda: core.get('pattern.C.ch2.trig')[:3] == [True, False, True])
