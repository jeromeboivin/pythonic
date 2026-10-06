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
