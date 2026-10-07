# Slice W8: the AI drum generator page

Manual checks of the web panel (`pythonic`), then a short tkinter pass. The
ML extras (torch) and the bundled models are needed for most of it; the
first block checks the page without them (a virtual environment without
torch). The full map of tkinter controls to the web panel is in
[web-parity-matrix.md](web-parity-matrix.md).

## Without the ML extras
- [ ] PRESET ▸ AI drum generator…: the drawer shows the page; the header says `ML EXTRAS MISSING`; the middle shows the pip command with copy command and install now; the model lines say `needs the ML extras`; generate all 8 is grey.
- [ ] copy command: the display says `copied`; the command pastes into a terminal.
- [ ] install now asks first (the command on a green sheet); install: the button reads `installing…` while pip runs, the panel keeps playing; at the end `ML extras installed` (or a red alert with pip's last lines), and the lanes appear without a restart.

## Page and lanes
- [ ] The page opens in the drawer (opening a closed edit rack, which closes again with the page); its eight lanes sit exactly under the eight strips, the generate column under the left column, patterns and apply under the right one.
- [ ] The model lines show the bundled models loading, then `drum_cvae_best.pt ✓ (…)` and `pattern_cvae_best.pt ✓`; load… opens the native dialog (`*.pt`); a bad file gives a red alert, a good one is kept for the next start.
- [ ] Each lane's type starts at the channel's drum type (else BD, SD, CH, OH, TOM, TOM, CLAP, CY); its list offers the 18 types.
- [ ] gen on a lane: `generating…` in the lane and the header, then `1/8`, the candidate's name, `trying ✓`; the strip tab shows the name in italics with a dashed outline; while stopped the channel sounds once.
- [ ] ‹ › step through the candidates (wrapping) and each plays on the face; try toggles the lane back to its old sound (the tab goes back too) and on again.
- [ ] Start the transport: the running pattern plays the tried kit; turn a strip knob of a trying channel: the edit stays in the tried sound and UNDO stays grey.
- [ ] candidates ‹ › (1..32, wheel too), seed (type a number; ⟳ picks a random one; blank = random): generate all 8 with the same seed twice gives the same candidates.
- [ ] The patch and pattern temperature knobs move the same values as the setup sheet's ai tab.

## Patterns and apply
- [ ] generate new + generate all 8: the bank note goes `generating patterns…` then `bank of 12 ready`; ▶ loop plays the AI version of the playing (or selected) pattern, ▶ bank A→L plays all 12 chained; a second click or STOP puts the preset's patterns back.
- [ ] keep current: ▶ loop / ▶ bank play the preset's own patterns with the tried kit; replace patterns is grey.
- [ ] Stepping a lane after a bank was made drops it (`(generate all 8 first)`); ↻ makes a new one.
- [ ] keep tried: `kept ch …` on the display, the tabs lose their italics; one UNDO puts every old sound back (edits included).
- [ ] replace patterns: the kept sounds and all 12 patterns change; one UNDO takes both back. revert all: the old sounds return.

## Leaving
- [ ] With tried sounds, ✕ and `◀ ch n edit` ask `Keep the tried sounds?` with cancel / revert / keep; cancel stays on the page.
- [ ] With tried sounds, ⊞ MATRIX, PO-32 or closing the edit rack replaces the page and then asks keep / revert.
- [ ] With tried sounds, loading a preset (menu, ◀ ▶, reload last) asks first; cancel keeps the current preset.
- [ ] Without tried sounds, leaving asks nothing (a running pattern preview stops).

## tkinter interface
- [ ] `pythonic --ui tk`: the AI drum generator dialog still generates, previews, applies and replaces patterns; its temperatures and model paths are the same preferences as the web page's.
