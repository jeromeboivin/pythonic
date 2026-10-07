# Slice W5: preset and pattern menus, exports, pickers, alert sheet

Manual checks of the web panel (`pythonic`), then a short tkinter pass. Use
a preset folder with a few presets (e.g. a copy of `tests/*.mtpreset`), set
through PRESET ▸ preset folder…. The full map of tkinter controls to the
web panel is in [web-parity-matrix.md](web-parity-matrix.md).

## PRESET buttons and menu
- [ ] ◀ ▶ load the previous / next preset of the folder; the display shows its name, the tabs and knobs follow; ◀ is dim on the first file, ▶ on the last.
- [ ] PRESET opens the menu under it: the folder name with ↻, the folder's presets (the current one green), the recent files with their folder; PRESET again (or a click elsewhere) closes it. A click on a preset or a recent file loads it.
- [ ] open preset…: the native open dialog starts in the preset folder (filter `Presets (*.mtpreset *.json)`); cancel does nothing; a `.mtpreset` and a `.json` both load.
- [ ] save preset as…: the native save dialog proposes `<name>.json`; saving over an existing file asks `Replace “x.json”?` on a sheet with a green top edge (cancel: `not saved`; replace: `saved`). After saving, the menu's second entry reads `save “x.json”` and saves over it without asking.
- [ ] A broken preset file (write `{` into `bad.json`): loading it shows a red-edged alert `Could not load the preset` with the reason; the panel keeps playing but ignores clicks until OK; a click beside the alert does not close it.
- [ ] reload last preset loads the last loaded file.
- [ ] copy preset, initialize preset (all channels init, patterns empty), paste preset (everything back; paste is grey before any copy), cut preset; randomize all changes every channel and the selected pattern. Each is one Undo step.
- [ ] load / save drum patch of CHn and export CHn hit as WAV work from the PRESET menu with the edit rack closed.
- [ ] export every drum as WAV…: pick a folder, eight `01_<name>.wav` .. `08_<name>.wav` files appear; again: `Replace 8 files?` lists them.
- [ ] preset folder…: pick another folder; the list shows its presets; restart: the folder is kept.

## Pattern menu exports
- [ ] MENU ▸ export to MIDI…: the save dialog proposes `pythonic_pattern_A.mid`; the file plays in a DAW (channel 10 drum notes, accents at 127); exporting again asks before replacing.
- [ ] MENU ▸ export to audio…: a popover on MENU offers cut / add 2 s / loop +1 pass (the note under them explains); save wav… opens the save dialog; while it renders the display counts `PATTERN A WAV  NN %`, then `saved …wav`. A pattern right-clicked (e.g. C) exports that pattern. The last tail choice is offered next time.
- [ ] Export into a read-only folder: a red alert explains why.

## Pages, sheets, errors
- [ ] PO-32, SETUP, the MIDI LED, right-click a knob ▸ CC mappings…, and PRESET ▸ transfer / import PO-32, AI drum generator, setup each show a `coming soon` alert with a green edge until their slices land (W6-W8).
- [ ] Stop the audio device (unplug, or `pythonic --ui web` with a busy device): `AUDIO ERROR` shows on the display and a red alert explains; OK closes it.
- [ ] Click the selected channel's button: the channel sounds (Ctrl+click louder) and the button flashes; play notes from a MIDI keyboard: the channel buttons flash.

## tkinter interface
- [ ] `pythonic --ui tk`: the preset menu (open, save as, drum patch load / save, export drum / all drums to WAV, cut / copy / paste, initialize, randomize all, select preset folder, refresh) and the pattern menu's MIDI and audio exports still work; a preset saved from the web panel loads in tkinter and the other way round.
