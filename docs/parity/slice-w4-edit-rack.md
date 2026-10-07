# Slice W4: edit rack and modulation

Manual checks of the web panel (`pythonic`), then a short tkinter pass. Load
a preset first (e.g. `tests/808.mtpreset`) and start a pattern so the
channels sound.

## The rack
- [ ] Under the face, one row: oscillator, noise, envelopes, mix, velocity, FX, modulation (LFO 1, LFO 2, pump rows in cyan, pink and yellow). No tune, osc decay or level in the rack (they are on the strips).
- [ ] The header shows `CH 1`, the patch name and its drum type; select another channel on the face (or in the matrix): every rack control shows that channel's values and edits it; the display names the touched control (`CH3 NOISE FREQ`).
- [ ] Oscillator: wave sin / tri / saw, pitch mod dec / sine / rand, freq, amount, rate. Noise: LP / BP / HP, stereo, freq, Q. Envelopes: exp / lin / mod over the osc attack, noise attack and noise decay faders. Mix: the osc · noise crossfader (osc left), eq freq, eq gain, distort, pan, out A / B, choke. Velocity: osc, noise, mod faders. FX: vintage, rvb time / mix / wide, delay 1/4 1/8 1/16 1/8T 1/4. with P.P, dly fdbk, dly mix. Each one changes the sound as in tkinter.
- [ ] LFO rows: on, wave list, rate, depth, sync list, re, uni, phase; pump row: on, amount, attack, release, curve, sync. A row dims while it is off.
- [ ] Drag, wheel, double-click (typed value), right-click (reset, MIDI learn, pitch bend) work on rack controls as on the face; a drag is one undo step.
- [ ] A preset with an unusual delay time (e.g. 1/32) shows it lit beside the five delay buttons.

## Edit all
- [ ] edit all in the rack header lights, the note `edits hit every unmuted channel` shows and the other unmuted strips' tabs get a dashed outline; a rack edit then changes every unmuted channel (check another channel's rack); muted channels keep their value.

## Destinations (click to assign)
- [ ] LFO 1 → button: the header shows a cyan bar with off and cancel; the rack controls an LFO can modulate, the selected strip's tune, decay, CTRL and level, master and the sound morph get a dashed cyan outline (the current destination solid).
- [ ] Click the noise freq knob: the button reads `→ noise freq`, the bar goes, the knob did not move. With LFO 1 on and depth up, a cyan band moves on the knob while the channel plays.
- [ ] Arm LFO 2, click the selected strip's level fader: a pink line moves on that fader; arm pump, click the sound morph: the morph knob shows a yellow band while the pump runs.
- [ ] While armed, clicking the wave switch says `not a destination`; another strip's tune says `not on CH1`; cancel (or the → button again) ends it; off clears the destination (`→ off`).
- [ ] The wheel on a → button steps through the targets in engine order; right-click offers assign and clear.
- [ ] Every strip shows its own channel's bands (tune, decay, CTRL, level), not only the selected one's; master and morph show any channel's.

## Drum patch menu
- [ ] drum patch ▾: load a `.mtdrum` into the selected channel (the tab and rack follow); save it (an existing file asks `Replace it`); export the channel's hit as WAV.

## Open / closed
- [ ] edit rack (right column) closes the drawer: the window loses the rack's height and the face keeps its size; again: it comes back. Quit and restart: the state is kept.
- [ ] With the rack closed, ⊞ matrix opens the drawer for the matrix and closing the matrix closes it again; the next start still has the rack closed.
- [ ] A maximized window does not resize: the face letterboxes.

## tkinter interface
- [ ] `pythonic --ui tk`: the drum patch area (mixing, oscillator, noise, FX, velocity, LFO 1 / 2, pump with destination lists) still edits the selected channel; a destination set in the web rack shows in the tkinter LFO destination list for the same preset and the other way round.
