# Slice 6 parity checklist: preset and drum-patch files, preferences in the core

Manual checks in the tkinter GUI after preset and drum-patch I/O and the
preference application moved into the app core (inventory clusters N, P;
sections 1.2, 1.6.1, 1.7.2-1.7.4, 1.11). Start the app with `./run.sh` (or
`python run.py`). Keep a copy of `~/.config/Pythonic/preferences.json` before
you start, and compare it at the end: the same keys must still be there.

## Preset list and start-up (1.2, 1.11)
- [ ] At start-up the last loaded preset is loaded again and its file name is shown in the preset combobox; `↶` is lit (the load is one step).
- [ ] Pick another preset in the combobox: sounds, knobs, pattern lanes, tempo, swing, step rate, fill rate, master, mutes, program and morph slider all change to the file's values.
- [ ] `◀` / `▶` step through the folder's presets in name order.
- [ ] Preset menu > Select Preset Folder...: pick another folder; the combobox lists its `.mtpreset` and `.json` files at once.
- [ ] Copy a preset file into the folder, Preset menu > Refresh Preset List: it appears.

## Open and save (1.6.1)
- [ ] Open Preset... a `.mtpreset` (one from `tests/`): everything listed above follows; the program combobox goes to 1 and the bank is empty (new: a `.mtpreset` replaces the bank).
- [ ] Before loading a `.mtpreset`, turn up reverb mix on channel 1: after the load it is back to its init value (new: the whole sound is replaced).
- [ ] Open Preset... a `.json` preset: programs, morph endpoints and position come back as saved; the master knob and swing slider now follow the file too (new).
- [ ] Open a file that is not a preset (a `.txt` renamed, an empty `.json`): an error box; nothing changes, `↶` does not light.
- [ ] Save Preset As... a new name: the file is written; it shows in the combobox as the current preset if it is in the preset folder.
- [ ] Save Preset As... over an existing file: Tk asks to replace it; Yes replaces it (the saved file now holds the mutes too).
- [ ] Load the saved `.json` in another session: sounds, patterns (also 64-step ones with velocities), programs, morph, mutes, swing and fill rate are as saved.
- [ ] Press `s` (save) and `l` (load) on the keyboard: the same dialogs open.

## Drum patches (1.6.1)
- [ ] Select channel 3, Save Drum Patch (.mtdrum)...: the default file name is the channel's patch name; the file is written.
- [ ] Select channel 6, Load Drum Patch (.mtdrum)... that file: channel 6 takes the sound and name; `Ctrl+Z` brings the old sound back.
- [ ] Load a broken `.mtdrum`: an error box, no change.

## Clipboard, initialize, randomize (1.6.1)
- [ ] Copy Preset, change sounds and patterns, Paste Preset: everything comes back; Paste is greyed out before the first copy.
- [ ] Cut Preset: all channels go to the init sound and all patterns are empty; Paste brings them back; `Ctrl+Z` after Cut undoes the initialize.
- [ ] Initialize Preset, Randomize All: one `Ctrl+Z` each restores everything they changed; Randomize All changes the sounds and the selected pattern only.

## Audio settings (1.7.2)
- [ ] The dialog shows the saved device, input device, buffer, sample rate, synth rate and mono.
- [ ] Pick an output device: the sample-rate list is filtered to the rates it accepts ("Device supports n of 7 ..."), as before.
- [ ] Refresh re-lists the output and input devices.
- [ ] OK with a new buffer size: nothing restarts (saved for the next launch); restart the app: the new buffer is used.
- [ ] OK with Mono ticked: the sound turns mono at once (new: mono applies on OK as on the setup sheet).
- [ ] Apply Now with a new device, rate or synth rate: the stream restarts with them; "Currently using" shows the device.
- [ ] Cancel changes nothing.

## Synthesis and AI settings (1.7.3, 1.7.4)
- [ ] Synthesis Settings: move the smoothing slider, Apply, then OK; the value is kept after a restart.
- [ ] AI Settings: Browse... a `.pt` file, set the temperature, OK; reopen: both are shown; AI Randomize Pattern uses the new model and temperature.
- [ ] AI Settings: Clear, OK: the bundled checkpoint is used again.
- [ ] The AI Drum Generator dialog still shows and saves its model paths and temperatures (it shares the same keys).

## MIDI settings (1.7.5, 1.7.6)
- [ ] MIDI Settings and CC Mappings work as after slice 3 (device, base note, clock sync, mappings saved and restored after a restart).

## Preferences file (1.11)
- [ ] After all of the above, `preferences.json` holds the same keys as before (plus any `ui_*` keys only a new front-end writes).
- [ ] Start the app with an old `preferences.json` that lacks the newer keys (no `synth_sample_rate`, `audio_mono`, AI keys): it starts with their defaults and keeps the keys it had.
