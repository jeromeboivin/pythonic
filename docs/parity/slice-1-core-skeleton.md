# Slice 1 parity checklist: core skeleton

Manual checks in the tkinter GUI after the app core (`pythonic/app/`) took over
composition, the audio stream and the UI tick (inventory clusters A, B, C).
Start the app with `./run.sh` (or `python run.py`) and watch the console.

## Start-up and shutdown (A)
- [ ] The window opens at 960×600 and the console prints `Audio stream started on '...' @ ... Hz (buffer: ... samples, ...)`.
- [ ] The last loaded preset comes back on start (preset name, sounds, patterns).
- [ ] With a MIDI device connected and enabled, the console prints `MIDI input connected: ...`.
- [ ] Close the window: the process exits within a second or two, with no traceback.

## Audio stream and callback (B)
- [ ] Press `1`-`8`: each channel sounds at once.
- [ ] Play pattern A with a few steps set: steps sound in time, the playhead moves along the lanes, the playing pattern button flashes.
- [ ] Set a chain A→B and play: when A ends, B plays and the editors follow to B.
- [ ] Preset menu (`▼`) > `Audio Settings...`: the "Currently using: ..." line names the device in use (with `(default)` for the system default).
- [ ] Audio Settings > `Refresh` re-lists the output and input devices.
- [ ] Audio Settings > pick another output device > `Apply Now`: sound continues on the new device and the "Currently using" line updates.
- [ ] Change Output Sample Rate (e.g. 48000) > `Apply Now`: sound continues, pitch is unchanged.
- [ ] Internal Synth Rate 22050 (lo-fi) > `Apply Now` while a pattern plays: playback continues with the lo-fi character; the sounds, mutes, program slots, tempo, selected channel and master volume are unchanged.
- [ ] Buffer size 10 ms > `Apply Now`: sound continues; the console shows the new buffer size.
- [ ] Mono output on > `Apply Now`: pan has no effect any more; off again restores stereo.
- [ ] Audio Settings > `OK` only saves (no stream restart); the values are used on the next launch.

## Performance reporting (C)
- [ ] About every 5 s while audio runs, the console prints the `=== Audio Performance ... ===` block (callbacks, underruns, buffer time, utilization).
- [ ] If the device stops calling back (a wedged device), the console prints `ERROR: Audio stream stalled ...; the stream was stopped`, and the GUI keeps working (silent); `Apply Now` in Audio Settings reopens the stream.

## UI tick and undo
- [ ] Set an LFO on a knob's parameter (e.g. LFO 1 → pan) and trigger the channel: the knob's modulation indicator moves; switching channel shows only that channel's indicators.
- [ ] Turn a knob, then `Ctrl+Z`: the knob and sound go back; `Ctrl+Y` redoes it. The ↶/↷ buttons enable and disable as before.
- [ ] Toggle steps, then undo/redo: the lane editors and the matrix follow.
- [ ] Move the sound morph slider with different endpoints, edit, undo: the slider returns with the sounds.
