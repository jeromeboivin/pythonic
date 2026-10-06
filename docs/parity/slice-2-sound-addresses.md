# Slice 2 parity checklist: sound addresses and Edit all

Manual checks in the tkinter GUI after the sound controls, globals, mutes,
channel selection and Edit all moved onto core addresses (inventory clusters
K, L). Start the app with `./run.sh` (or `python run.py`) and load a preset
with different sounds on each channel.

## Drum patch controls (L)
- [ ] Turn each knob and slider of mixing, oscillator, noise, fx and vel on channel 1 while pressing `1`: the sound changes as you drag, the knob does not jump back.
- [ ] Click the waveform icons, pitch mod `Decay`/`Sine`/`Rand`, filter `LP`/`BP`/`HP`, envelope `Exp`/`Lin`/`Mod`, delay time, `choke`, `A`/`B` output, `stereo`, `P.P`: each is heard and stays selected.
- [ ] LFO 1 and 2: `on`, wave, rate, depth, sync, `re`, `uni`, destination (e.g. Pan): the modulation is heard and the knob's orange indicator moves.
- [ ] Pump: `on`, amount, attack, release, curve, destination: the ducking is heard.
- [ ] Ctrl+click or double-click a knob: it resets to its default and the sound follows.
- [ ] Undo/redo (`Ctrl+Z`/`Ctrl+Y`) after a knob drag and after a selector click: knob and sound go back and forth.

## Channel selection and mutes (L)
- [ ] Click channel buttons 1-8: the blue LED, the patch name, the `chN` lane editor and every knob switch to that channel.
- [ ] Click the already selected channel: it sounds (Ctrl+click: louder) and its LED flashes.
- [ ] Keys `1`-`8` sound their channels and flash their LEDs.
- [ ] `m` buttons: the channel LED turns red and the channel goes silent in playback; off again restores it.
- [ ] Load another preset: mutes, sounds and the drum type labels follow.

## Edit all (K)
- [ ] Mute channel 3, turn `edit all` on, select channel 1, turn osc decay, then select channels 2 and 3: channel 2 has the new decay, channel 3 does not.
- [ ] With `edit all` on, also try a selector (filter mode), a toggle (`stereo`) and an LFO destination: they reach every unmuted channel.
- [ ] Mute the selected channel with `edit all` on and turn a knob: the selected channel changes too.
- [ ] Turn `edit all` off: knobs change the selected channel only.
- [ ] Pattern steps are not affected by `edit all` (Shift+click in a lane still edits all 8 channels, muted ones included).

## Globals (L)
- [ ] BPM entry: type 140 and Return: playback speeds up; type 999: shows 300; type text: reverts.
- [ ] Step rate `1/8` … `1/32` and fill rate `2x` … `8x`: the clicked button lights up and playback follows.
- [ ] Swing slider: the shuffle is heard.
- [ ] `master` knob: overall level follows.
- [ ] With MIDI clock sync on and an external clock, the BPM entry follows the clock.

## Sound morph slider and MIDI CC
- [ ] With different A/B endpoints, move the `sound morph` slider: the sound blends and the knobs follow.
- [ ] A learned MIDI CC on a knob (and on `master`, the mix slider, the morph slider) still moves it and the sound.
