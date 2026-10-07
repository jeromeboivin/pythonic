# Slice W9: integration, parity audit and hardening

Manual checks after W9. Use a scratch preferences folder (Linux:
`XDG_CONFIG_HOME=/tmp/pythonic-check`), never your own.

## Web interface (`pythonic`)

- [ ] Start with `--devtools`: the DevTools console shows no failed loads at start-up, with or without the fonts fetched.
- [ ] Drag a strip knob slowly up and down while the pattern plays: the knob follows the pointer and never steps back.
- [ ] Edit a tempo, a strip decay and a pad, then UNDO three times: the display names each control (`TEMPO`, `CH2 DECAY`, `PATTERN A CH1 STEP3 TRIG`).
- [ ] Trigger a long message (an export error, `edits hit all 8 channels` from ALL CH): the line scrolls to its end and back and stays until read.
- [ ] Close the edit rack, quit, start again: the rack is closed and the window 1280x560; PO-32 and AI pages open it and closing them restores it.
- [ ] With the ML extras: open the AI page while the pattern plays; the models load (about 40 s for the bundled ones) without audible dropouts; gen, ‹ ›, keep tried, UNDO.
- [ ] `pythonic --ui web --quit-after 20` prints `UI frames: ... over the 4 ms budget` and exits 0.

## tkinter interface (`pythonic --ui tk`)

- [ ] Starts, loads the last preset the web interface saved, plays; the AI dialog generates with the lower-priority worker.
