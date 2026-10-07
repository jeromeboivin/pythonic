# Slice 5 parity checklist: undo, programs and the sound morph in the core

Manual checks in the tkinter GUI after undo/redo, the program bank and the
sound morph moved into the app core (inventory clusters H, I, J; sections
1.1, 1.6.1, 1.7.8, 1.7.9, 1.9). Start the app with `./run.sh` (or
`python run.py`), load a preset with different sounds on each channel, and
keep an eye on the undo `↶` and redo `↷` buttons: they light only when there
is something to undo or redo.

## Undo and redo (1.1, 1.9)
- [ ] At start-up after the last preset loaded, `↶` is lit (the load is one step); `Ctrl+Z` goes back to the default sounds, `Ctrl+Y` loads them again.
- [ ] Drag the osc decay knob up and down in one drag, release, `Ctrl+Z`: the knob and the sound go back to where the drag started in one step; `↷` lights; `Ctrl+Y` redoes it.
- [ ] Turn a knob with the mouse wheel for a few notches, pause, `Ctrl+Z`: the whole turn is undone at once.
- [ ] Ctrl+click (or double-click) a knob to reset it, `Ctrl+Z`: the old value comes back.
- [ ] Click a switch (choke, ping-pong, stereo) and a selector (waveform, filter mode, delay time): one `Ctrl+Z` each undoes them.
- [ ] Drag the osc/noise mix slider, the swing slider and the master knob: each drag is one step (they were not undoable before).
- [ ] Change an LFO rate, depth, wave and destination and the pump amount: each is undoable (new).
- [ ] Type a BPM, click a step rate and a fill rate: each is one step (new).
- [ ] With `edit all` on, turn reverb mix on channel 1: every unmuted channel follows; `Ctrl+Z` puts all of them back in one step.
- [ ] Mute a channel, select another channel, toggle `edit all`, play and stop: none of these is an undo step.
- [ ] After `Ctrl+Z`, make a new edit: `↷` goes dark (redo is cleared).
- [ ] More than 50 edits: only the last 50 can be undone.

## Pattern edits
- [ ] Paint four triggers in one drag in the lane editor, release, `Ctrl+Z`: all four go away in one step (before: one step per cell).
- [ ] Drag a step velocity or probability: one step per drag.
- [ ] Shift+click a step (every channel), then `Ctrl+Z`: every channel's step is undone at once.
- [ ] Turn off a trigger that had accent and fill, `Ctrl+Z`: trigger, accent and fill are back.
- [ ] Make a pattern 32 steps long with hits on steps 17-32, shorten it to 16 in the `len` lane, `Ctrl+Z`: the length is 32 again and the hits are still there.
- [ ] Pattern menu: Clear, Shift, Reverse, Randomize, Paste, Paste lane, and the chain buttons: one `Ctrl+Z` each (new for the non-AI entries).
- [ ] Matrix view: a drag across cells is one step.

## Bulk changes (1.6.1, 1.7.8, 1.7.9)
- [ ] Load a preset (`.mtpreset` and `.json`), `Ctrl+Z`: the previous sounds, patterns, tempo, swing, programs and morph come back; mutes and the selected channel stay as they are.
- [ ] Preset menu > Initialize Preset, Randomize All, Cut Preset, Paste Preset: one `Ctrl+Z` each restores everything they changed.
- [ ] Preset menu > Load Drum Patch, `Ctrl+Z`: the channel's old sound is back (new).
- [ ] Pattern menu > AI Randomize Pattern / Channel (with a model): one step each.
- [ ] AI Drum Generator: apply patches, close, `Ctrl+Z`: the sounds from before the Apply are back (before: undo went to the state after it). Previews still sound and are restored on close as before.
- [ ] Import from PO-32: import, `Ctrl+Z`: sounds, patterns and morph endpoints from before the import are back (new); the morph slider goes to 0 after the import as before.

## MIDI (1.10)
- [ ] Turn a mapped controller, pause half a second, `Ctrl+Z`: the move is undone in one step.
- [ ] MIDI clock changing the BPM and the pitch wheel are not undo steps.

## Programs (1.1)
- [ ] Pick program 2 (empty): the sounds stay; change a knob; pick program 1: its sounds come back; pick 2 again: the change is still there.
- [ ] `Ctrl+Z` after a program change: the previous program and its sounds come back (new).
- [ ] Save a JSON preset after using several programs and load it again: the combobox shows the saved program.

## Sound morph (1.1)
- [ ] With equal endpoints the morph slider is greyed out; click `A`: it turns green, the slider is enabled.
- [ ] While learning A, change some knobs; click `B`: A turns grey, B green, the sound jumps to endpoint B; change knobs; click `B` again: learn stops, the slider blends between the two sounds.
- [ ] Move the morph slider: all knobs of the selected channel follow the blend; `Ctrl+Z` puts the slider and the sound back.
- [ ] `Ctrl+Z` right after stopping a learn: the endpoint goes back to what it held before.
- [ ] A loaded preset's morph position is shown on the slider and the slider does not snap to a whole percent by itself.
- [ ] An LFO with destination `Morph` sweeps the sound while the slider stays put.
