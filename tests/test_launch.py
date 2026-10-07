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
