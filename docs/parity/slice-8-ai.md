# Slice 8 parity checklist: AI generators in the core

Manual checks in the tkinter GUI after the AI pattern randomize and the AI
drum generator moved into the app core's AI module, whose models run in a
worker subprocess (inventory clusters G and the AI part of Q; sections 1.6.2,
1.7.4, 1.7.9). Start the app with `./run.sh` (or `python run.py`) with the ML
extras installed and the bundled checkpoints checked out (`git lfs checkout`).

## Responsiveness (new)
- [ ] Start the app, play a busy pattern, open Preset menu > AI Drum Generator...: no audio dropout while the models load (new: they load in the worker process, the status shows "loading..." then "loaded").
- [ ] `ps` shows a second Python process (`ai_worker.py`) once the AI was used; it ends when the app closes.

## Pattern menu, AI entries (1.6.2)
- [ ] Pattern MENU > Randomize Pattern (AI): the selected pattern is replaced by an AI pattern for the current kit; one Undo brings the old pattern back.
- [ ] Pattern MENU > Randomize Channel (AI): only the selected channel's lane changes; one Undo restores it.
- [ ] Right-click another pattern button, use an AI entry: that pattern changes.
- [ ] With no pattern model (AI Settings path to a missing file, no bundled checkpoint): an error box says no AI pattern model is available and how to set one.

## AI Settings (1.7.4)
- [ ] Browse / Clear / temperature, OK: saved; the next AI randomize uses the new path and temperature (no restart needed).
- [ ] The "Bundled checkpoint will be used as fallback." line shows when `drum_patterns/pattern_cvae_best.pt` exists.

## AI Drum Generator (1.7.9)
- [ ] Opening the dialog stops the transport; closing restarts the pattern that was playing.
- [ ] Patch Model / Pattern Model status: "loaded (data-guided)" / "loaded" with the bundled checkpoints (new: the patch model also falls back to the bundled `drum_cvae_best.pt` when no path is saved). Load... picks another checkpoint; it is saved once it loads; a bad file shows "error: ...".
- [ ] Generate on a slot: the lane shows "generating...", then `1 / n` and the candidate name; the candidate is already on the live channel (new: candidates are tried on the live channel when they arrive; the main window's knobs and the channel name follow).
- [ ] `<` / `>` step through the candidates and put each one on the channel (new: before, the arrows only changed the selection).
- [ ] Preview hits the slot's candidate at full velocity.
- [ ] Generate All 8: every lane fills in as its candidates arrive; with "Generate New AI Patterns" the bank status then shows "Bank ready (12 patterns)".
- [ ] Loop Preview plays the playing (or selected) pattern with the tried kit, the AI version of it in generate mode; Stop Loop puts the preset's patterns back. The tried sounds stay until Apply or Close (new: before, stopping the loop also put the sounds back).
- [ ] Preview Bank chains A to L from A; Stop Bank restores the patterns and their chains.
- [ ] Apply Selected keeps the checked slots: one Undo step takes all of them back. Unchecked slots go back to their old sounds on Close.
- [ ] Replace All Patterns From AI (generate mode, bank ready): checked sounds plus all 12 patterns, one Undo step for both.
- [ ] Close without applying: every channel and pattern is as before; nothing was added to the undo history.
- [ ] Without torch installed: the status reads "PyTorch not installed", Install ML Support asks with the pip command, installs in the background; afterwards models load without restarting the app.
