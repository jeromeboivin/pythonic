# Slice 7 parity checklist: export in the core

Manual checks in the tkinter GUI after MIDI and WAV export moved into the app
core (inventory cluster O; sections 1.6.1, 1.6.2, 1.7.1). Start the app with
`./run.sh` (or `python run.py`) and load a preset with busy patterns.

## Pattern MIDI export (1.6.2)
- [ ] Pattern MENU > Export Pattern to MIDI File...: the save dialog proposes `pythonic_pattern_<X>.mid`; the file is written.
- [ ] Open the file in a DAW or MIDI viewer: one note per triggered step on channel 10, GM drum notes, the step velocities (127 on accents), the preset tempo; swing places the off-beat sixteenths as you hear them.
- [ ] A 64-step pattern exports all 64 steps; the file loops cleanly at its end.
- [ ] Right-click another pattern button, export it: that pattern is written, not the selected one.
- [ ] Save over an existing file: Tk asks to replace it; Yes replaces it.

## Pattern audio export (1.7.1)
- [ ] Pattern MENU > Export Pattern to Audio File...: the "Export Audio Options" window offers None (truncate), Append (add silence), Loop (repeat pattern); then the save dialog proposes `pythonic_pattern_<X>.wav`.
- [ ] None: the file lasts one pass of the pattern. Append: 2 s longer, the sounds ring out. Loop: two passes.
- [ ] The file plays the pattern alone, even when it is chained to the next one (new: before, the export followed the live chain and the queued pattern).
- [ ] Export while the sequencer plays: playback goes on without a glitch or a jump, the play position and chain are unaffected (new: before, the export drove the live synth and patterns).
- [ ] The GUI stays responsive during a long export (64 steps at a slow tempo, Loop) (new: the render runs off the Tk thread).
- [ ] With Mono output on (Audio settings), the file is mono; off, stereo. The sample rate is the internal synth rate.
- [ ] Mute a channel before exporting: it is silent in the file.

## Drum WAV export (1.6.1)
- [ ] Preset menu > Export Drum to WAV...: a 2 s file of the selected channel hit at full velocity.
- [ ] Preset menu > Export All Drums to WAV...: pick a folder; it gets `01_<name>.wav` .. `08_<name>.wav`.
- [ ] Export All Drums again into the same folder: a box asks to replace the existing files; No leaves them, Yes rewrites them (new: before, they were replaced without asking).

## Errors
- [ ] Export into a folder you cannot write to: an error box names the problem; nothing else changes.
