import pytest

from pythonic.preferences_manager import PreferencesManager


@pytest.fixture
def prefs(tmp_path, monkeypatch):
    """A real PreferencesManager writing into the test's temporary folder."""
    monkeypatch.setattr(PreferencesManager, '_get_config_dir', lambda self: str(tmp_path))
    monkeypatch.setattr(PreferencesManager, '_get_default_preset_folder',
                        lambda self: str(tmp_path / 'presets'))
    return PreferencesManager()
