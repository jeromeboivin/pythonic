# Quick Start

## Install and start

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -e .                                  # add ".[ml]" for the AI drum generator
pythonic                                          # the web interface; pythonic --ui tk for tkinter
```

Optional, for the panel's intended look: `python tools/fetch_fonts.py`
downloads its three OFL fonts into `pythonic/web/static/fonts/` (see
`SOURCES.txt` there); without them it uses system fonts.

## First time

When you run Pythonic for the first time:
1. A config folder is created automatically
2. A `Pythonic Presets` folder is created in your Documents
3. All settings are saved to a `preferences.json` file

Both interfaces read and write the same preferences, presets and patterns:
a preset saved in one loads in the other.

## A first session (web interface)

1. **Load a preset**: click **PRESET** (right column). The menu lists the
   presets of your preset folder (a click loads one) and your recent files;
   **open preset…** browses anywhere. ◀ ▶ step through the folder.
2. **Play**: click **START / STOP** (bottom left). The pads light up as the
   pattern plays; the page bars above them show which 16 steps you see.
3. **Edit steps**: pick a step mode in the left column (**trig**, accent,
   velo, fill, prob, sub) and click or paint along the pads. **last step**
   then a pad sets the pattern's length (up to 64 steps).
4. **Shape the sounds**: drag the strip knobs and faders up and down (Shift
   for fine steps), turn the wheel, double-click to type a value. Click a
   channel button to select it: the edit rack below shows all its
   parameters.
5. **Undo** anything with **undo** (top left).
6. **Save**: PRESET ▸ **save preset as…**.

Right-click any control for reset to default and MIDI learn. **setup**
holds the audio, MIDI (devices, base note, clock, CC mappings), synthesis and
AI settings; **po-32** transfers sounds and patterns to and from a PO-32.

## Presets

- **Preset folder**: PRESET ▸ **preset folder…** (tkinter: the 📁 button
  beside the preset combo-box). The list follows the folder; **refresh list**
  (tkinter: 🔄) picks up files added by hand.
- **Recent files**: the last 10 presets you opened, in the PRESET menu.
- **Reload last preset**: the preset you had open last also loads at start-up.

### Supported File Formats

- `.json` - Pythonic presets (sounds, patterns, tempo, programs)
- `.mtpreset` - imported presets
- `.mtdrum` - single drum patches (PRESET ▸ load / save drum patch, or the edit rack's drum patch ▾)

## Where are my preferences stored?

| OS | Location |
|---|---|
| Windows | `C:\Users\YourName\AppData\Roaming\Pythonic\preferences.json` |
| macOS | `~/Library/Application Support/Pythonic/preferences.json` |
| Linux | `~/.config/Pythonic/preferences.json` (or `$XDG_CONFIG_HOME/Pythonic/`) |

Presets folder, all platforms: `~/Documents/Pythonic Presets/`

What gets remembered: the preset folder, the last preset, recent files,
audio and MIDI settings, CC mappings, AI models and temperatures, and in the
web interface the strip CTRL mode and whether the edit rack is open.

## Troubleshooting

**The preset list is empty?**
- Use **refresh list**, and check that the files are in the preset folder
- Use **open preset…** to browse manually

**Preferences not saving?**
- Check that the config folder is writable
- On Linux: `~/.config/Pythonic/` should exist; create it if needed: `mkdir -p ~/.config/Pythonic`

**Where are my old presets?**
- They're still in their original location: open them with **open preset…**,
  or move them to your Pythonic Presets folder and refresh the list

## Advanced: manual configuration

You can edit `preferences.json` by hand while Pythonic is closed:

```json
{
  "preset_folder": "/custom/path/to/presets",
  "max_recent_files": 20
}
```

See the [README](README.md) for everything else.
