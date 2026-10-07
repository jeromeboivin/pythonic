# Slice 9 parity checklist: PO-32 transfer and import in the core

Manual checks in the tkinter GUI after the PO-32 transfer and import moved
into the app core's PO-32 module (inventory cluster Q; sections 1.7.7,
1.7.8). Every stream is now the core's: the transfer and the preview play
through the main output stream, recording uses an input stream the core
opens. Start the app with `./run.sh` (or `python run.py`); a PO-32 (or a
second machine recording/playing WAV files) helps for the real-device checks.

## Transfer (1.7.7)
- [ ] The `PO-32` button and Preset menu > Transfer to PO-32... open the dialog with the preset name, `1 - 8` / `9 - 16`, the chain choices (`A - B` when A is chained to B, else single letters) and the 8 channels with their names.
- [ ] Mute channel 3 on the face, open the dialog: its checkbox starts unchecked.
- [ ] The status shows "Ready to transfer (n.ns audio)" and updates when the bank, the chain or a checkbox changes.
- [ ] Transfer: the status turns "TRANSFERRING...", the progress bar moves, it ends with "Transfer complete!" and then "Ready..." again. The signal comes out of the output device chosen in the Audio settings (new: before, a separate stream on the system default device).
- [ ] Press a pad or have a pattern playing during the transfer: nothing but the modem signal is heard (new: the transfer replaces the synth while it plays). After the transfer the pattern is heard again.
- [ ] A PO-32 in receive mode (hold write, press sound) receives the sounds; unchecked channels arrive silent.
- [ ] Stop during a transfer: "Transfer stopped", the output goes back to the synth at once.
- [ ] Save WAV: the native save dialog proposes "<preset> (PO-32 transfer).wav"; the status shows "Saved: ..."; the file decodes in the import dialog.
- [ ] Stop the audio (Audio settings with a failing device, or unplug): Transfer warns that the audio output is not running; Save WAV still works.
- [ ] Close during a transfer: it stops.

## Import (1.7.8)
- [ ] Preset menu > Import from PO-32... opens the dialog; the input device list is the core's input devices, preselecting the saved input device (Audio settings), else the system default.
- [ ] Monitor: the level meter and dB readout follow the input; Stop closes the input; changing the device while monitoring reopens it on the new device.
- [ ] "Save recorded audio to file (debug)" is remembered across restarts; Open Folder opens `~/Documents/Pythonic Debug Recordings`.
- [ ] Record, play a PO-32 transfer (or a transfer WAV from another device), Stop: the source line shows "Decoded PO-32 card: 16 drums, 16 patterns" (or "Pythonic: 8 drums, n patterns"); with the debug box checked a `po32_recording_*.wav` file appears in the folder and its name is in the source line.
- [ ] Import WAV File...: decodes the file; a file without a modem signal shows the decode error box.
- [ ] Bank 0 / Bank 1 radios are enabled only for decoded banks; the drum patches line and the grid labels follow the bank.
- [ ] Pattern buttons: the first 12 non-empty patterns are picked (`n→A`, ...); clicking toggles a pick (at most 12) and focuses the pattern (grid and summary); the destination menus swap letters on a conflict; Select First 12 and Clear All work; the count reads "n/12 patterns selected".
- [ ] Preview with the main transport playing: the transport stops and the focused pattern loops with the decoded sounds at the panel tempo (new: through the main output stream). Stop: silence, the transport stays stopped. Pressing Play on the main window while previewing ends the preview. Selecting another pattern stops the preview.
- [ ] Import Drums + n Patterns: the knobs of channels 1-8 show the decoded sounds (names unchanged), the picked patterns land on their letters (steps 1-16), the other patterns are empty, the morph slider is at A and moving it reaches the PO-32's morph sounds; the confirmation lists the patterns; the dialog closes.
- [ ] One Undo brings back all sounds, patterns and the morph from before the import.
- [ ] A pattern of length 8 picked as a destination becomes 16 steps long (new: it raised before); a 32-step destination keeps its length with steps 17-32 empty.
- [ ] Closing the dialog with the window's close button stops monitoring, recording and the preview (new: before, the title bar close left the streams open).
