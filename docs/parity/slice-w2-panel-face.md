# Slice W2: panel face and control behaviour

Manual checks of the web face (`pythonic`), then a short tkinter pass.

## Layout and look
- [ ] The face shows: PYTHONIC, undo / redo, empty step-entry and pattern areas, programs 1-16; top row master, step rate, fill rate, strip ctrl; 8 strips (white name tab, tune, decay, ctrl knobs, lit fader, channel button); right column green display, preset / edit rack / PO-32 / setup placeholders, MUTE, learn A / B, red tempo, tempo and swing knobs, MIDI LED, ringed sound morph knob; START/STOP and 16 pads; the empty edit rack area below.
- [ ] Faders glow red to violet by channel; channel buttons show BD, SD, CH, MT, RS, CB... from the patch names (a name without a type shows the channel number).
- [ ] Load another preset: tabs, buttons, knob values and the display's preset name follow.

## Knobs and faders
- [ ] Drag a decay knob up/down: 200 px covers the range (log feel on ms); hold Shift for fine steps; one Undo restores the whole drag.
- [ ] Wheel over a knob: 1 % per notch; over a fader: 2 %; a quick wheel turn undoes in one step.
- [ ] Click a fader track: the cap jumps there; keep dragging.
- [ ] The value under each control is always printed (ms/s, Hz/kHz, dB, st, %, L/C/R); the green display shows the touched control's name and value for 1.5 s.
- [ ] Double-click a knob: type `1.2 s` (or `750`), Enter applies, Esc cancels.
- [ ] Right-click: Reset to default; MIDI learn (the control pulses blue), move a controller: CC badge appears, the LED blinks on each CC; Remove CC mapping; Assign pitch bend.
- [ ] With a CC mapped and the controller away from the value: the amber ghost marker shows on the ring (fader: a line) until the controller crosses the value.
- [ ] Turn on LFO 1 of a channel aimed at osc decay (tkinter or a preset): a moving cyan band shows on that strip's decay knob; pump on level: a yellow line on the fader.

## Strips and columns
- [ ] Channel buttons select (button lit, tab in the strip colour); MUTE lit: channel buttons toggle mutes (red text, dark fader); MUTE again: selecting.
- [ ] Strip ctrl: pan / reverb mix / delay mix / LFO 1 depth retarget all CTRL knobs; off dims them; user: click a ctrl label to pick its parameter; the choice survives a restart.
- [ ] Step rate buttons, fill rate list, master, tempo knob, wheel over the red tempo, swing change the sound / timing.
- [ ] Programs: click 2, edit, click 1 and 2 again: sounds switch; empty programs are dimmer.
- [ ] Undo / Redo dim when there is nothing to undo / redo; the display names the undone step.
- [ ] Learn A lights; edits are captured; Learn A again stops; right-click Learn B: capture current sounds as B; the morph knob blends A to B (its ring dims while A = B).
- [ ] START/STOP plays the selected pattern from step 1 and stops; the display shows pattern and step.
- [ ] MIDI LED: dim amber when an input is open, flashes on MIDI activity.
- [ ] Unplug the audio device (or stop the stream): "AUDIO ERROR" shows on the display.

## tkinter interface
- [ ] `pythonic --ui tk`: play, turn a knob, LFO modulation indicators on the selected channel still move.
