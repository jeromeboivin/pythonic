"""
Preferences Manager for Pythonic
Handles persistent user preferences including default preset folder
Cross-platform support using platformdirs
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from platformdirs import user_config_dir, user_data_dir
    PLATFORMDIRS_AVAILABLE = True
except ImportError:
    PLATFORMDIRS_AVAILABLE = False
    print("Warning: platformdirs not available. Using fallback directory paths.")


class PreferencesManager:
    """
    Manages application preferences with persistent storage.
    Preferences are stored in a JSON file in the platform-appropriate location.
    """
    
    APP_NAME = "Pythonic"
    APP_AUTHOR = "Pythonic"
    PREFS_FILENAME = "preferences.json"
    
    DEFAULT_PREFERENCES = {
        'preset_folder': None,  # Will be set to user's Documents/Pythonic Presets
        'last_preset': None,
        'window_width': 1200,
        'window_height': 700,
        'master_volume_db': 0.0,
        'recent_files': [],
        'max_recent_files': 10,
        # Audio settings
        'audio_output_device': None,  # None = system default
        # Parameter smoothing settings
        'param_smoothing_ms': 30.0,  # Smoothing time constant (20-50ms recommended)
        # MIDI settings
        'midi_input_device': None,  # Auto-connect to first if None
        'midi_base_note': 36,  # C1 - default mapping for drum channels
        'midi_enabled': True,  # Whether MIDI input is enabled
        'midi_clock_sync': True,  # Sync BPM to incoming MIDI clock
        # MIDI CC mappings: dict of {cc_number (as string): parameter_name}
        # Default: CC1 (Mod Wheel) -> osc_freq, CC2 (Breath) -> noise_freq
        'midi_cc_mappings': {
            "1": "osc_freq",
            "2": "noise_freq"
        },
    }
    
    def __init__(self):
        """Initialize the preferences manager"""
        self.prefs_dir = self._get_config_dir()
        self.prefs_file = os.path.join(self.prefs_dir, self.PREFS_FILENAME)
        self.preferences = self._load_preferences()
        
        # Initialize default preset folder if not set
        if self.preferences['preset_folder'] is None:
            self.preferences['preset_folder'] = self._get_default_preset_folder()
            self._save_preferences()
    
    def _get_config_dir(self) -> str:
        """Get the configuration directory (cross-platform)"""
        if PLATFORMDIRS_AVAILABLE:
            config_dir = user_config_dir(self.APP_NAME, self.APP_AUTHOR)
        else:
            # Fallback for systems without platformdirs
            home = Path.home()
            if os.name == 'nt':  # Windows
                config_dir = os.path.join(home, 'AppData', 'Roaming', self.APP_NAME)
            elif os.name == 'posix':
                if 'darwin' in os.sys.platform:  # macOS
                    config_dir = os.path.join(home, 'Library', 'Application Support', self.APP_NAME)
                else:  # Linux and other Unix-like
                    config_dir = os.path.join(home, '.config', self.APP_NAME)
            else:
                config_dir = os.path.join(home, '.pythonic')
        
        # Create directory if it doesn't exist
        os.makedirs(config_dir, exist_ok=True)
        return config_dir
    
    def _get_default_preset_folder(self) -> str:
        """Get the default preset folder path (cross-platform)"""
        home = Path.home()
        
        # Try to use Documents folder
        if os.name == 'nt':  # Windows
            documents = os.path.join(home, 'Documents')
        elif os.name == 'posix':
            if 'darwin' in os.sys.platform:  # macOS
                documents = os.path.join(home, 'Documents')
            else:  # Linux
                # Try XDG_DOCUMENTS_DIR first, then fallback to ~/Documents
                xdg_dirs_file = os.path.join(home, '.config', 'user-dirs.dirs')
                documents = os.path.join(home, 'Documents')
                
                if os.path.exists(xdg_dirs_file):
                    try:
                        with open(xdg_dirs_file, 'r') as f:
                            for line in f:
                                if line.startswith('XDG_DOCUMENTS_DIR'):
                                    path = line.split('=')[1].strip().strip('"')
                                    path = path.replace('$HOME', str(home))
                                    if os.path.exists(path):
                                        documents = path
                                        break
                    except Exception:
                        pass
        else:
            documents = str(home)
        
        # Create Pythonic Presets subdirectory
        preset_folder = os.path.join(documents, 'Pythonic Presets')
        os.makedirs(preset_folder, exist_ok=True)
        
        return preset_folder
    
    def _load_preferences(self) -> Dict[str, Any]:
        """Load preferences from file"""
        if os.path.exists(self.prefs_file):
            try:
                with open(self.prefs_file, 'r', encoding='utf-8') as f:
                    loaded_prefs = json.load(f)
                
                # Merge with defaults to ensure all keys exist
                prefs = self.DEFAULT_PREFERENCES.copy()
                prefs.update(loaded_prefs)
                return prefs
            except Exception as e:
                print(f"Error loading preferences: {e}")
                return self.DEFAULT_PREFERENCES.copy()
        else:
            return self.DEFAULT_PREFERENCES.copy()
    
    def _save_preferences(self) -> bool:
        """Save preferences to file"""
        try:
            os.makedirs(os.path.dirname(self.prefs_file), exist_ok=True)
            with open(self.prefs_file, 'w', encoding='utf-8') as f:
                json.dump(self.preferences, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving preferences: {e}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a preference value"""
        return self.preferences.get(key, default)
    
    def set(self, key: str, value: Any) -> bool:
        """Set a preference value and save"""
        self.preferences[key] = value
        return self._save_preferences()
    
    def get_preset_folder(self) -> str:
        """Get the current preset folder path"""
        folder = self.preferences.get('preset_folder')
        if folder and os.path.exists(folder):
            return folder
        else:
            # Reset to default if path doesn't exist
            default_folder = self._get_default_preset_folder()
            self.set('preset_folder', default_folder)
            return default_folder
    
    def set_preset_folder(self, folder_path: str) -> bool:
        """Set the preset folder path"""
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            return self.set('preset_folder', folder_path)
        return False
    
    def add_recent_file(self, filepath: str) -> bool:
        """Add a file to the recent files list"""
        recent = self.preferences.get('recent_files', [])
        
        # Remove if already exists
        if filepath in recent:
            recent.remove(filepath)
        
        # Add to front
        recent.insert(0, filepath)
        
        # Trim to max length
        max_files = self.preferences.get('max_recent_files', 10)
        recent = recent[:max_files]
        
        # Filter out files that no longer exist
        recent = [f for f in recent if os.path.exists(f)]
        
        self.preferences['recent_files'] = recent
        return self._save_preferences()
    
    def get_recent_files(self) -> list:
        """Get the list of recent files"""
        recent = self.preferences.get('recent_files', [])
        # Filter out files that no longer exist
        recent = [f for f in recent if os.path.exists(f)]
        self.preferences['recent_files'] = recent
        return recent
    
    def clear_recent_files(self) -> bool:
        """Clear the recent files list"""
        self.preferences['recent_files'] = []
        return self._save_preferences()
    
    def reset_to_defaults(self) -> bool:
        """Reset all preferences to defaults"""
        self.preferences = self.DEFAULT_PREFERENCES.copy()
        self.preferences['preset_folder'] = self._get_default_preset_folder()
        return self._save_preferences()
    
    def get_all_preferences(self) -> Dict[str, Any]:
        """Get all preferences"""
        return self.preferences.copy()
