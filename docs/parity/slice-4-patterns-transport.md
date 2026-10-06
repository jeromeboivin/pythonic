# Slice 4 parity checklist: patterns and transport in the core

Manual checks in the tkinter GUI after pattern editing, selection, the queue,
chains and the transport moved into the app core (inventory clusters D, E and
F, sections 1.4, 1.5, 1.6.2, 1.6.4, 1.8.3 and 1.9). Start the app with
`./run.sh` (or `python run.py`) and load a preset with a few patterns filled
in and some patterns left empty.

## Lane editor (1.4, 1.8.3)
- [ ] Click steps in `trig`, `acc` and `fill`: they toggle; accent and fill only toggle on triggered steps; turning a trigger off clears its accent and fill (and they stay cleared after switching pattern and back).
- [ ] Drag along `trig`: every crossed step toggles, the lane does not flicker back while dragging.
- [ ] Ctrl+click in `trig`: trigger and accent toggle together.
- [ ] Shift+click in `trig`/`acc`/`fill`: the same step changes on all 8 channels, muted ones included (check two channels by selecting them).
- [ ] Right-click a step (or click in `sub`): the substeps menu ticks the current value; pick `o-o`, then `Custom...` with `oo--`: both play as substeps.
- [ ] `Prob` on, drag a step up and down: the value follows (0-100), a 0 % step never plays; `Prob` off again.
- [ ] Click in `len` on step 12: the pattern plays 12 steps; select another pattern: its own length shows (behaviour change: the editors used to keep the previous length).
- [ ] `Ctrl+Z` after a step edit: the step goes back.

## Matrix (1.4)
- [ ] `⊞`: the 8×16 trigger grid of the selected pattern shows; click and drag along a row toggles cells; `⊞` again: the lane editor shows the same triggers.

## Pattern buttons, queue and chains (1.4, E)
- [ ] Stopped: click `C`: it turns blue, the editors show C, the playhead is on step 1.
- [ ] Empty patterns have dim letters; chained patterns blue letters.
- [ ] Playing A: click `B`: B flashes blue (queued) and takes over when A reaches its end; the editors follow; click the playing pattern while another is queued: the queue is cancelled.
- [ ] Select B, `▶▶`: B and C are chained (blue letters); `◀◀` on C unchains them; `◀◀` on A and `▶▶` on L do nothing.
- [ ] Chain A→B→C and play A: A, B, C play in turn and start again at A; the selected button and the editors follow the playing pattern.
- [ ] Queue E while a chain plays: E plays next, ahead of the chain.

## Pattern menu and lane clipboard (1.6.2)
- [ ] `Menu` and right-click on a pattern button (which also selects it): Cut, Copy, Paste, Exchange, Shift Left/Right, Reverse, Randomize, Alter Pattern and Randomize Accents/Fills all act on that pattern and the editors update.
- [ ] Paste Pattern into another letter: an exact copy (length, all channels).
- [ ] `Copy` with channel 2 selected, select channel 5, `Paste`: channel 5 gets the triggers, accents, fills and probabilities (substeps stay as they were); `Paste` before any `Copy` in a fresh session warns "Nothing in clipboard".
- [ ] Randomize Pattern (AI) and Randomize Channel (AI) still work (they are not moved yet).

## Transport (1.5, D, 1.10)
- [ ] `▶`: the selected pattern plays from step 1, the play ring lights, the playhead moves over every step (behaviour change: it used to skip steps without hits).
- [ ] `▶` while playing restarts from step 1; `■`: playback stops, the playhead returns to step 1, the stop ring lights.
- [ ] MIDI Start / Stop / Continue and Program Change behave as in the slice 3 checklist.
- [ ] AI Drum Generator while playing: the transport stops while the dialog is open, previews play, and the pattern that was playing restarts when the dialog closes.
- [ ] Export Pattern to Audio File while stopped and while playing: the WAV is right and playback is unaffected.

## Pads (1.9)
- [ ] Keys `1`-`8` play channels 1-8 at full velocity and flash their buttons; clicking the selected channel again plays it (Ctrl+click accented).
