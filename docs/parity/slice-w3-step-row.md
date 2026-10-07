# Slice W3: step row, step entry and patterns

Manual checks of the web panel (`pythonic`), then a short tkinter pass. Load
a preset with patterns first (e.g. `tests/808.mtpreset` through tkinter or
the last preset).

## Pads and step modes
- [ ] The 16 pads show the selected channel's steps of the selected pattern: lit in red / salmon / yellow-green / lavender groups, dimmer for low velocities, a white dot on accents, a stripe on fills, `50%` on steps below 100 % probability, dots for substeps. Selecting another channel shows its lane.
- [ ] trig: click toggles a step; drag along the row paints on (or off, starting from a lit pad); one Undo removes the whole stroke.
- [ ] accent / fill: click and paint only change lit pads; turning a trigger off drops its accent and fill.
- [ ] velo: pads show a bar and the value (ACC on accents); drag a lit pad up/down (200 px = 1-127, Shift fine); an empty pad does nothing; the sound gets softer / louder while playing.
- [ ] prob: drag any pad (0-100 %), the value shows; playback skips hits as expected.
- [ ] sub: click a pad: the menu offers none, the 14 presets and custom… (type `o-o-`, Enter); the dots on the pad follow; the hits split while playing.
- [ ] all ch lit: a click or a stroke hits the same steps on all 8 channels, muted ones included; one Undo takes it back.
- [ ] The green display names each edit (`CH1 STEP 5` / `velocity 96`).

## Length and pages
- [ ] last step (lights red), then click pad 8: the pattern is 8 steps long, pads 9-16 dim, `8]` marks the end in the numbers; last step, page bar 2, pad 24: 24 steps, pages 3-4 dim.
- [ ] Page bars switch the pads (17-32 ...); the display shows `PATTERN A  17-32`; picking a page turns follow off.
- [ ] Play a 32-step pattern with follow on: the pads turn to the playing page, the white playhead outline runs, the playing page bar pulses green.
- [ ] Select another pattern while one plays: no playhead on the pads of the pattern being edited.

## Matrix
- [ ] ⊞ matrix: the drawer shows the 8 channels x 16 steps of the page with names and drum types; cells show triggers, accents, fills, velocity, probability, substeps; the playhead column moves.
- [ ] Click / drag along a row toggles that channel's steps (the step mode applies: accent, velo drag, sub menu ...); click a channel name selects it.
- [ ] ⊞ again (or ◀ edit rack) gives the drawer back.

## Patterns
- [ ] Pattern buttons: the selected one lit, the playing one red, a queued one blinks, empty ones dim, chained ones linked by an amber bar, long patterns show their length.
- [ ] Click B while A plays: B blinks (queued) and the pads show B; A finishes its loop, then B plays.
- [ ] ◀◀ / ▶▶ chain the selected pattern to its neighbours; the display shows the chain (`CHAIN A-B`); chained patterns play in turn.
- [ ] copy then select another channel, paste: the lane (triggers, accents, velocities, fills, probabilities) is copied; paste with nothing copied says so.
- [ ] MENU and right-click on a pattern: cut, copy, paste (into another pattern), exchange, clear, shift left / right, reverse, randomize, alter, randomize accents + fills, copy / paste lane, clear all chains each do their job and undo in one step; play next (while playing) queues without selecting.
- [ ] randomize (AI) / randomize chN (AI) work with the ML extras installed (dim without them).

## tkinter interface
- [ ] `pythonic --ui tk`: edit steps, velocity lane, page selector, pattern buttons, chain and the pattern menu still work; a preset saved from one GUI loads in the other with the same patterns.
