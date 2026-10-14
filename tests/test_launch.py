"""The `pythonic` command: interface choice, DevTools flag, missing Qt, dependencies."""

import re
import sys
from pathlib import Path

import pytest

from pythonic import launch
from pythonic.install import EXTRAS, pip_command

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
    monkeypatch.setattr(launch, 'run_tk', lambda args: started.append('tk'))
    assert launch.main([]) == 0
    assert started == ['web']


def test_missing_qt_starts_tkinter(monkeypatch):
    started = []
    monkeypatch.setattr(launch, 'web_unavailable_reason', lambda: 'No module named PySide6')
    monkeypatch.setattr(launch, 'run_web', lambda args: started.append('web'))
    monkeypatch.setattr(launch, 'run_tk', lambda args: started.append('tk') or 0)
    launch.main(['--ui', 'web'])
    launch.main(['--ui', 'tk'])
    assert started == ['tk', 'tk']


def test_the_ml_install_command_names_the_extras_packages():
    assert pip_command('ml')[1:4] == ['-m', 'pip', 'install']
    assert pip_command('ml')[4:] == EXTRAS['ml']


@pytest.mark.skipif(sys.version_info < (3, 11), reason='tomllib needs Python 3.11')
def test_the_dependencies_bring_qtwebengine_and_mirror_the_installer():
    import tomllib

    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']
    assert project['optional-dependencies']['ml'] == EXTRAS['ml']
    # From PySide6 6.12 QtWebEngine is its own wheel, only pulled in by the
    # Addons webengine extra: without it the web interface cannot load
    qt = [d.split(';')[0].strip() for d in project['dependencies'] if d.startswith('PySide6')]
    assert 'PySide6-Addons[webengine]>=6.8' in qt
    assert project['scripts']['pythonic'] == 'pythonic.launch:main'
    assert re.match(r'>=\s*3\.10', project['requires-python'])
