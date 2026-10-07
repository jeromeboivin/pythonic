# Slice 4b parity checklist: per-step velocity and 1-64 step patterns

Manual checks in the tkinter GUI after steps gained a velocity (1-127,
default 64, beside accent) and patterns grew to 64 steps on 4 pages. Start
the app with `./run.sh` (or `python run.py`) and load a preset with a few
patterns filled in.

## Nothing changed for existing patterns
- [ ] Load a factory `.mtpreset` and play it: it sounds as before (unaccented hits at 64, accented at 127, fills fading as before).
- [ ] Load a `.json` preset saved before this slice: it loads, every `vel` bar sits at half length (64).
- [ ] A 16-step pattern: the lane editor looks as before plus the `vel` lane; page `1-16` is highlighted, the other pages dimmed.

## Velocity lane
- [ ] Drag a triggered step up and down in `vel`: the bar grows and shrinks, the number shows while dragging, the hit gets louder/softer (raise `OscVel` on the channel to hear it clearly).
- [ ] Drag an untriggered step in `vel`: nothing changes.
- [ ] Accent a step with a low velocity: it plays at full strength; remove the accent: back to its velocity.
- [ ] Turn a trigger off and on: its velocity is kept.
- [ ] `Ctrl+Z` after a velocity drag: the velocity goes back.
- [ ] Pattern menu `Clear` (and `Cut`): velocities go back to 64. `Shift Left/Right`, `Reverse`, `Copy`/`Paste`, `Exchange`, lane `Copy`/`Paste` carry the velocities.

## Pages and length
- [ ] Page `49-64`, click `len` on step 64: the pattern is 64 steps, all pages light up, step numbers read 49-64.
- [ ] Set triggers on steps 1, 20, 40, 60 and play: each plays once per 64-step pass, on time.
- [ ] Follow on (green): the pages turn with the playhead and the playing page's label is lit; click a page by hand: follow turns off, the page stays while the playhead goes on (its page label stays lit).
- [ ] Page `33-48`, `len` on step 37: steps 38-64 dim and ignore clicks; page `49-64` dims.
- [ ] Try each step rate (1/8 ... 1/32) on a 64-step pattern: it loops after its last step.
- [ ] `⊞` matrix: shows the same page as the lane editor; clicks land on that page's steps; the playhead moves.
- [ ] Chain a 64-step A to a 16-step B: A plays all 4 pages, then B.

## Files and export
- [ ] Save a preset (`S`) with a 64-step pattern and velocities, load it again: length and velocities are back.
- [ ] Pattern menu `Export Pattern to MIDI File...` on a 64-step pattern: in a DAW the notes sit on the step grid (1/16 = a sixteenth), velocities match, swing matches playback, the clip is 64 steps long.
- [ ] `Export Pattern to Audio File...` on a 64-step pattern: the WAV is 64 steps long and the velocities are audible.
