# Slice W7: the PO-32 page

Manual checks of the web panel (`pythonic`), then a short tkinter pass. A
PO-32 (or a WAV of a transfer, e.g. one saved from the transfer tab) and an
audio input are needed for the import. The full map of tkinter controls to
the web panel is in [web-parity-matrix.md](web-parity-matrix.md).

## The page
- [ ] PO-32 opens the page in the edit rack drawer on the transfer tab; the PO-32 button is lit; PO-32 again closes it. The second time it opens on the tab used last. PRESET ▸ transfer to PO-32… / import from PO-32… open their tab.
- [ ] With the edit rack closed, the page opens the drawer; `◀ ch N edit` or ✕ closes the page and the drawer closes again. Channel buttons still select channels while the page shows (the back button follows).
- [ ] Each tab is a numbered flow: the current stage has a green edge, later ones are dim but every control works, done stages show ✓.

## Transfer
- [ ] Mute channel 3 on the face, open the page: channel 3 is unticked, the others ticked. Tick / untick channels: the face mutes do not change.
- [ ] sounds to 1–8 / 9–16, the pattern chain (list of chain groups; link A-B with the chain buttons first): the note says which PO-32 pattern slots are sent empty; the status reads `ready: N s of signal`.
- [ ] On the PO-32: hold write and press sound. ▶ transfer: the bar fills, the status and the display count the percentage, the PO-32 receives the sounds; stages 1-3 ticked at the end. ■ stop during a send stops it (`stopped`); closing the page during a send stops it too.
- [ ] save wav…: the native save dialog proposes `<preset> (PO-32 transfer).wav`; saving again over it asks `Replace …?`; the WAV transfers to the PO-32 when played into it.
- [ ] With the audio output stopped, ▶ transfer shows a red alert that suggests saving the WAV.

## Import
- [ ] The input list shows the input devices (the saved one or the default first); ↻ rescans. monitor lights and the meter moves with the input (dB beside it, a held peak; green, amber, red as it gets louder).
- [ ] ● record, play the PO-32 transfer into the input (the status counts the seconds), ■ stop: the status reads `decoded PO-32 card: 16 sounds, N patterns`; stage 3 lights. keep recordings on: a `po32_recording_*.wav` appears in the folder shown beside it.
- [ ] import wav… decodes a WAV file; a file that is not a transfer shows a red alert `Could not decode the PO-32 signal`.
- [ ] bank 0 / bank 1 are enabled only when decoded; the 8 sound names follow the bank.
- [ ] The 16 pattern buttons (1-based): decoded non-empty patterns are picked `1→A`, `2→B`…; a click unpicks / picks again (first free letter); a 13th pick says `12 patterns picked already`; right-click a pattern: its letter menu, a held letter swaps (`→ A (swaps with 1)`). first 12, clear and the count work; the grid shows the focused pattern's triggers.
- [ ] Start the panel, ▶ preview: the panel stops and the focused pattern loops with the decoded sounds at the panel tempo (the step outlined on the grid); a bank, pattern or pick change ends it; starting the panel ends it.
- [ ] import 8 sounds + N patterns: channels 1-8 get the sounds (names kept), the picked patterns land on their letters, the other patterns are emptied; the display confirms, the page stays open with every stage ticked. One Undo brings everything back.
- [ ] Closing the page ends a preview and closes the input (the meter stops).

## tkinter interface
- [ ] `pythonic --ui tk`: Preset ▼ ▸ Transfer to PO-32 and Import from PO-32 still open their dialogs and work as before (send / stop / save WAV; record / import WAV / bank / picks with letters / preview / import).
