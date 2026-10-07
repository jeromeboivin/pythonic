"""The `pythonic` command: interface choice, DevTools flag, missing Qt, extras."""

import re
import sys
from pathlib import Path

import pytest

from pythonic import launch
from pythonic.install import EXTRAS, pip_command
from tests.test_tk_smoke import _display_available

ROOT = Path(__file__).resolve().parent.parent


def test_web_is_the_default_interface():
    args = launch.parse_args([])
    assert (args.ui, args.devtools, args.quit_after) == ('web', None, None)
    assert launch.parse_args(['--ui', 'tk']).ui == 'tk'


def test_devtools_sets_the_remote_debugging_port():
    env = {}
    launch.configure_environment(launch.parse_args(['--devtools']), env)
    assert env == {'QTWEBENGINE_REMOTE_DEBUGGING': '9222'}
    launch.configure_environment(launch.parse_args(['--devtools', '9333']), env)
    assert env['QTWEBENGINE_REMOTE_DEBUGGING'] == '9333'
    env = {}
    launch.configure_environment(launch.parse_args([]), env)
    assert env == {}


def test_web_starts_when_qt_is_there(monkeypatch):
    started = []
    monkeypatch.setattr(launch, 'web_unavailable_reason', lambda: None)
    monkeypatch.setattr(launch, 'run_web', lambda args: started.append('web') or 0)
    monkeypatch.setattr(launch, 'run_tk', lambda args, qt_missing=None: started.append('tk'))
    assert launch.main([]) == 0
    assert started == ['web']


def test_missing_qt_starts_tkinter_with_the_explanation(monkeypatch):
    started = []
    monkeypatch.setattr(launch, 'web_unavailable_reason', lambda: 'No module named PySide6')
    monkeypatch.setattr(launch, 'run_web', lambda args: started.append('web'))
    monkeypatch.setattr(launch, 'run_tk',
                        lambda args, qt_missing=None: started.append(('tk', qt_missing)) or 0)
    launch.main(['--ui', 'web'])
    launch.main(['--ui', 'tk'])
    assert started == [('tk', 'No module named PySide6'), ('tk', None)]


def test_install_commands_name_the_extras_packages():
    assert pip_command('qt')[-2:] == EXTRAS['qt']
    assert pip_command('ml')[1:4] == ['-m', 'pip', 'install']


@pytest.mark.skipif(sys.version_info < (3, 11), reason='tomllib needs Python 3.11')
def test_installer_packages_mirror_pyproject():
    import tomllib

    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']
    assert project['optional-dependencies']['ml'] == EXTRAS['ml']
    qt = [d.split(';')[0].strip() for d in project['dependencies'] if d.startswith('PySide6')]
    assert qt == EXTRAS['qt']
    assert project['scripts']['pythonic'] == 'pythonic.launch:main'
    assert re.match(r'>=\s*3\.10', project['requires-python'])


def test_the_explanation_offers_install_except_on_windows_arm64():
    from gui.qt_missing_dialog import explanation

    assert 'pip install PySide6' in explanation('boom', arm64=False)
    text = explanation('boom', arm64=True)
    assert 'pip install' not in text and 'Windows on ARM' in text


@pytest.mark.skipif(not _display_available(), reason='no display for tkinter')
def test_the_qt_missing_dialog_builds_with_and_without_install():
    import tkinter as tk

    from gui.qt_missing_dialog import show_qt_missing

    root = tk.Tk()
    try:
        def buttons(dialog):
            found = []

            def walk(widget):
                for child in widget.winfo_children():
                    if isinstance(child, tk.Button):
                        found.append(child.cget('text'))
                    walk(child)
            walk(dialog)
            return sorted(found)

        assert buttons(show_qt_missing(root, 'boom', arm64=False)) == ['Close', 'Install']
        assert buttons(show_qt_missing(root, 'boom', arm64=True)) == ['Close']
    finally:
        root.destroy()


@pytest.mark.skipif(not _display_available(), reason='no display for tkinter')
@pytest.mark.parametrize('ok', [True, False])
def test_the_install_worker_hands_its_result_to_the_tk_thread(monkeypatch, ok):
    """The pip run happens on a worker thread; only the Tk thread calls tkinter."""
    import threading
    import time
    import tkinter as tk

    from gui import qt_missing_dialog
    from gui.qt_missing_dialog import show_qt_missing

    main = threading.current_thread()
    off_main = []
    after = tk.Misc.after

    def watched_after(self, *args, **kwargs):
        if threading.current_thread() is not main:
            off_main.append(args)
        return after(self, *args, **kwargs)
    monkeypatch.setattr(tk.Misc, 'after', watched_after)

    workers = []

    def fake_install(extra, out):
        workers.append(threading.current_thread())
        out('pip says hello')
        return ok
    monkeypatch.setattr(qt_missing_dialog, 'install_extra', fake_install)
    shown = []
    monkeypatch.setattr(qt_missing_dialog.messagebox, 'showinfo',
                        lambda *a, **k: shown.append(('info', threading.current_thread(), a)))
    monkeypatch.setattr(qt_missing_dialog.messagebox, 'showerror',
                        lambda *a, **k: shown.append(('error', threading.current_thread(), a)))

    root = tk.Tk()
    try:
        dialog = show_qt_missing(root, 'boom', arm64=False)
        install = next(w for w in dialog.winfo_children()[1].winfo_children()
                       if w.cget('text') == 'Install')
        install.invoke()
        deadline = time.monotonic() + 5
        while not shown and time.monotonic() < deadline:
            root.update()
            time.sleep(0.01)
        assert workers and workers[0] is not main
        assert len(shown) == 1 and shown[0][1] is main
        assert shown[0][0] == ('info' if ok else 'error')
        assert install.cget('text') == ('Installed' if ok else 'Install')
        if not ok:
            assert 'pip says hello' in shown[0][2][1]
        assert off_main == []
    finally:
        root.destroy()
