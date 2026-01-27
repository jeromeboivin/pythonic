# Quick Start Guide - Preferences System

## What's New? 🎉

Your Pythonic synthesizer now has a **smart preferences system** that remembers your settings!

## First Time Setup

When you run Pythonic for the first time:
1. A config folder is created automatically
2. A `Pythonic Presets` folder is created in your Documents
3. All settings are saved to a `preferences.json` file
4. **That's it!** Everything is automatic.

## Using Presets with the Combo-Box

### The New Preset Header
In the top-left of the window, you'll see:
```
Preset: [    Select a preset...    ] 📁 🔄
```

### Loading a Preset
1. Click the combo-box to open the dropdown
2. Select a preset from the list
3. It loads instantly!

### Changing the Preset Folder
1. Click the **📁 (folder)** button
2. Choose a new folder
3. The list refreshes automatically

### Refreshing the List
1. Click the **🔄 (refresh)** button
2. Useful if you added files to the folder manually

## Saving Presets

When you click **"Save Preset"**:
- The file browser opens in your preset folder
- Save your preset there
- It automatically appears in the combo-box

## Where Are My Preferences Stored?

### Preferences File
| OS | Location |
|---|---|
| Windows | `C:\Users\YourName\AppData\Roaming\Pythonic\preferences.json` |
| macOS | `~/Library/Application Support/Pythonic/preferences.json` |
| Linux | `~/.config/Pythonic/preferences.json` |

### Presets Folder
All platforms: `~/Documents/Pythonic Presets/`

## What Gets Remembered?

✅ Preset folder location  
✅ Last loaded preset  
✅ Recent files (up to 10)  
✅ Master volume  
✅ Window size (coming soon)  

## Supported File Formats

- `.json` - Standard Pythonic format
- `.mtpreset` - Native Pythonic format

## Tips & Tricks

💡 **Organize Presets**: Create subfolders in your presets folder and navigate to them

💡 **Backup**: Copy your `Pythonic Presets` folder to backup your presets

💡 **Share**: Send preset files to others - they'll load in their preset folder

💡 **Recent Files**: The system tracks your 10 most recent presets

## Troubleshooting

**Combo-box is empty?**
- Click 🔄 to refresh
- Check that files are in the preset folder
- Use "Load Preset" button to browse manually

**Changed folder but list didn't update?**
- Click the 🔄 refresh button
- Or close and reopen the app

**Preferences not saving?**
- Check that the config folder is writable
- On Linux: `~/.config/Pythonic/` should exist
- Create it manually if needed: `mkdir -p ~/.config/Pythonic`

**Where are my old presets?**
- They're still in their original location
- Use "Load Preset" button to find them
- Move them to your Pythonic Presets folder
- Refresh with 🔄

## Advanced: Manual Configuration

You can edit `preferences.json` manually for advanced setup:

```json
{
  "preset_folder": "/custom/path/to/presets",
  "master_volume_db": -6.0,
  "max_recent_files": 20
}
```

Just restart Pythonic after editing!

## Need Help?

See `PREFERENCES_README.md` for complete documentation.

Enjoy your Pythonic presets! 🎵
