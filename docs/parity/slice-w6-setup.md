# Slice W6: setup sheet

Manual checks of the web panel (`pythonic`), then a short tkinter pass. Use
a MIDI controller (or a virtual port) for the MIDI checks. The full map of
tkinter controls to the web panel is in
[web-parity-matrix.md](web-parity-matrix.md).

## The sheet
- [ ] SETUP opens the setup sheet on the audio tab over the dimmed panel; the pattern keeps playing and the playhead moves, but the panel ignores clicks and the wheel. The tabs on top switch audio / midi / synthesis / ai and the sheet resizes (two columns for audio and midi, one for the others). ✕ or a click beside the sheet closes it; SETUP opens it on audio again.
- [ ] The MIDI LED and right-click a panel knob ▸ CC mappings… open the sheet on the midi tab. PRESET ▸ setup… opens it on audio.
- [ ] With the edit rack closed (smaller window) every tab still fits.

## Audio
- [ ] The output device list offers (system default) and the devices; rescan finds a device plugged in since start-up. Picking a device updates the sample rate note ("this device takes n of 7 rates") and the rate list offers only those.
- [ ] Changing the device, buffer, sample rate or synth rate lights a dot beside the field and the restart audio button; the sound carries on unchanged until restart audio, then the stream line shows the new settings and the dots go out. Restart the app: the settings are kept.
- [ ] Mono output applies at once (pan a channel hard left: both speakers sound), no dot.
- [ ] The input device (for the PO-32 import) lists (system default: <name>) and the inputs; the choice is kept after a restart.

## MIDI
- [ ] The device list offers (off), (auto-detect) and the ports; picking a port connects (LED lit, "connected", the MIDI LED on the face blinks on input); (off) disconnects; rescan finds a port plugged in later.
- [ ] Base note: the octave menu, − / + and the wheel change it; "channels 1–8 play on notes …" follows, and the controller's notes hit the matching channels.
- [ ] Follow MIDI clock on: an external clock sets the tempo and the sheet shows "synced tempo n BPM"; off: the tempo stays.
- [ ] CC mappings list every mapping (more than 8 work; past 10 the list scrolls). Type a new CC number (Enter) or turn the wheel on it: the mapping moves to that CC (taking over a CC another row had). Pick another control (section ▸ control); ✕ removes a row; + add makes a row with a free CC that is saved once a control is chosen; clear all asks first.
- [ ] Turning a mapped knob on the controller lights its row and moves the row's bar; the panel control follows with pickup as before.
- [ ] Pitch bend: (none) or a control from the menu; the wheel on the controller bends it.
- [ ] learn on the panel closes the sheet; right-click a control ▸ MIDI learn, move a controller: the new mapping appears in the sheet. A learn started before opening the sheet shows there with cancel.
- [ ] Right-click a control on the sheet (smoothing, temperatures): only reset to default, no MIDI learn or CC mappings.

## Synthesis and AI
- [ ] Smoothing (5–100 ms) applies at once to all channels (short: knob moves click, long: they glide).
- [ ] AI: pattern and drum patch models show "bundled" (or the file name); browse… opens the native dialog for `.pt` files; clear goes back to the bundled one. The temperatures apply to the next pattern menu ▸ randomize (AI) and the next drum patch generation. Without the ML extras a note says so.

## tkinter interface
- [ ] `pythonic --ui tk`: Audio, MIDI, CC mappings, Synthesis and AI settings dialogs still work and show what the web sheet saved (device, buffer, rates, base note, clock, mappings, smoothing, model paths and temperatures), and the other way round.
