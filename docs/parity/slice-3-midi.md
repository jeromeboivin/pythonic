# Slice 3 parity checklist: MIDI input in the core

Manual checks in the tkinter GUI after the MIDI thread, routing, CC map,
learn and pitch bend moved into the app core (inventory cluster M, section
1.10). Start the app with `./run.sh` (or `python run.py`) with a MIDI
controller (keys or pads, knobs, a pitch wheel) plugged in, and load a preset
with different sounds on each channel.

## Device and settings (1.7.5)
- [ ] At start-up the saved MIDI input is connected (else the first port); the console prints `Connected to MIDI input: ...`.
- [ ] Click the MIDI LED (or Preset menu > MIDI Settings...): the device list shows the ports, the status line names the connected one.
- [ ] Plug in another device, click `Refresh Devices`: it appears in the list.
- [ ] Pick another device and `Apply`: the status line shows `connecting...`, then the new device; notes from it play.
- [ ] Change the base note to C3 and `Apply`: notes 60-67 play channels 1-8, 36 no longer plays; restart the app: the base note and device are kept.
- [ ] Untick `Sync BPM to MIDI Clock`, `OK`, send clock: the BPM entry does not move; tick it again: it follows the clock and the dialog shows the synced BPM.

## Notes, patterns, transport (1.10)
- [ ] Play notes base..base+7 softly and hard: channels 1-8 sound with velocity, their buttons flash; notes outside the range do nothing.
- [ ] Hits from a pad stay tight against the playing pattern (no jitter compared with before).
- [ ] Program change 0-11 while stopped: pattern A-L is selected (button and lane editor follow); while playing: the pattern is queued (flashes blue) and takes over at the end of the bar; program change 12+ does nothing.
- [ ] MIDI Start: playback starts from step 1 of the selected pattern, the play button lights; Stop: it stops, the position goes back to step 1; Continue after Stop: playback resumes.
- [ ] MIDI clock at 93 BPM: the BPM entry shows 93 within two beats.
- [ ] The MIDI LED blinks on any incoming message.

## CC map and learn (1.6.3, 1.7.6)
- [ ] Right-click a knob (e.g. osc decay) > `MIDI Learn (CC)`: it flashes orange; move a controller: flashing stops, the console prints `MIDI Learn: CCn -> selected.osc.decay`, and the knob does not jump.
- [ ] Right-click > `Cancel MIDI Learn` during learn: flashing stops, nothing is mapped.
- [ ] Right-click a mapped knob: `Mapped to CCn` and `Remove CC Mapping` are shown; remove it: the controller no longer moves it.
- [ ] Learn a second CC on the same knob: the old CC is no longer mapped (one CC per control).
- [ ] Select another channel and move the controller: the knob of the selected channel moves (sound parameters follow the selected channel); `master` and `sound morph` mappings stay global.
- [ ] Right-click > `MIDI Settings...`: the CC Mappings dialog lists every mapping (more than 8 rows when needed); add a row with any CC 0-127, `Apply`, restart: the mappings are kept.
- [ ] With an old preferences file (names such as `osc_freq`), the default CC1/CC2 mappings still drive osc freq and noise freq of the selected channel.

## Pickup (behaviour change)
- [ ] Set a mapped knob to the middle with the mouse, then move its controller from the bottom: the knob stays put until the controller passes the middle, then follows.
- [ ] Turn the knob with the mouse, move the controller again: it waits for the next crossing.
- [ ] Load a preset or move the morph slider, then move a mapped controller: it waits for a crossing instead of jumping.

## Undo of CC moves (behaviour change)
- [ ] Turn a mapped controller for a while, stop for half a second, `Ctrl+Z`: the whole move is undone in one step (the undo button lights after the pause).
- [ ] Two separate moves with a pause in between: two undo steps.

## Pitch bend
- [ ] Right-click the pitch knob > `Assign Pitch Bend`; bend up and down: the pitch follows (up to half the knob range either way) and returns to the knob's value when the wheel is released.
- [ ] Right-click > `Remove Pitch Bend Mapping`: the wheel does nothing; restart: the choice is kept.
