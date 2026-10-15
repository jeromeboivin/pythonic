import faulthandler
import os
import sys

import pytest

from pythonic.preferences_manager import PreferencesManager


@pytest.fixture
def prefs(tmp_path, monkeypatch):
    """A real PreferencesManager writing into the test's temporary folder.

    MIDI input is off, so a started core never opens a real MIDI port; tests
    of the MIDI input switch it on with a fake MIDI backend.
    """
    monkeypatch.setattr(PreferencesManager, '_get_config_dir', lambda self: str(tmp_path))
    monkeypatch.setattr(PreferencesManager, '_get_default_preset_folder',
                        lambda self: str(tmp_path / 'presets'))
    manager = PreferencesManager()
    manager.set('midi_enabled', False)
    return manager


# faulthandler_timeout (pyproject.toml) dumps every thread's stack when a test
# hangs, but pytest stops it once a test fails (it gets out of pdb's way): a
# teardown that then hangs, a GUI whose event loop does not end, said nothing.
_DUMP_FD = pytest.StashKey[int]()


def pytest_configure(config):
    config.stash[_DUMP_FD] = os.dup(sys.stderr.fileno())  # the terminal, not a test's capture


def pytest_unconfigure(config):
    os.close(config.stash[_DUMP_FD])


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item):
    """Arm the hang dump again for the teardown (pytest disarms it after a failure)."""
    timeout = float(item.config.getini('faulthandler_timeout') or 0)
    if timeout > 0 and not item.config.getoption('usepdb', False):
        faulthandler.dump_traceback_later(timeout, file=item.config.stash[_DUMP_FD])
    return (yield)
