# Release checklist

CI (`.github/workflows/ci.yml`) installs, runs the tests and starts both
interfaces headless on Linux, Windows and macOS. Before a release, check by
hand on each OS (Linux, Windows, macOS) what CI cannot.

Never point a check at your real preferences: start the app with a scratch
config folder (Linux: `XDG_CONFIG_HOME=/tmp/pythonic-check pythonic`; it
creates `Pythonic/preferences.json` there).

## Install and start

- [ ] Fresh venv: `pip install -e ".[dev]"` succeeds (`".[ml]"` too); `pythonic --help` lists `--ui`, `--devtools`, `--quit-after`.
- [ ] `pythonic` starts the web interface: the window opens at 1280x800, letterboxes when resized, cannot shrink below 1280x800 (1280x560 with the edit rack closed).
- [ ] `pythonic --ui web --quit-after 20` exits 0 and prints `UI frames: ... 0 over the 4 ms budget` (a few over is fine on a busy machine); no `[js error]` lines and no Python traceback on stderr.
- [ ] `pythonic --devtools` opens DevTools on `http://127.0.0.1:9222`; the page's console shows no errors at start-up (no failed loads, also without the fonts).
- [ ] `pythonic --ui tk` starts the tkinter interface; presets, patterns and preferences saved by one interface load in the other.
- [ ] Without PySide6 (`pip uninstall PySide6-Addons`), `pythonic` starts tkinter with the explain-and-install dialog; Install works, a restart opens the web interface.
- [ ] Windows ARM64 (if available): installs without PySide6, `pythonic` starts tkinter with the explanation and no Install button.
- [ ] Linux only if needed: on a locked-down distro, container or as root, `QTWEBENGINE_DISABLE_SANDBOX=1` is the documented fallback.

## A session in the web interface

- [ ] **Audio out:** PRESET ▸ a preset from the folder list loads (display: `PRESET` / its name); START/STOP plays it audibly with the playhead on the pads, no underrun messages on stderr while editing; changing the output device and buffer in setup ▸ audio and *restart audio* restarts the stream.
- [ ] **Editing:** a pad click in trig mode, a strip knob drag (the knob follows the pointer without stepping back), a wheel turn, a double-click typed value; **undo** reverts them one by one and the display names each control (`UNDO` / `CH2 DECAY`).
- [ ] **Edit rack:** closing it shrinks the window by its height; restart: it opens closed again. The matrix, PO-32 and AI pages open a closed rack and closing them restores it.
- [ ] **MIDI in:** a connected controller triggers channels (the channel button flashes), learned CCs move their controls (the amber ghost marker until pickup), MIDI clock sync follows.
- [ ] **Modulation:** an LFO aimed at a strip's decay draws a moving arc on the strip and in the rack.
- [ ] **File dialogs:** open and save a preset, load and save a drum patch, choose the preset folder; the dialogs are native (a sheet on macOS; KDE portal where it runs) and the panel keeps playing while they are open.
- [ ] **AI (with `[ml]`):** the AI page loads the bundled models (the stream keeps playing; the worker runs at a lower priority), generate fills a lane and the strip tab shows the tried name; keep tried is one undo step.
- [ ] **PO-32:** a transfer plays through the output; an import from a recording or a WAV decodes and imports in one undo step.
- [ ] **Fonts:** without the fonts the panel uses system fonts (labels condensed, display and tempo monospace) and stays legible at 1280x800; after `python tools/fetch_fonts.py`, labels are in Barlow Condensed, the green display in Doto, the tempo in DSEG7.
- [ ] **Looks:** compare the face with the hardware-look prototype (branch `prototype/hardware-look`, `prototypes/hardware-look.html`): black panel, green accents, lit faders per channel, white patch-name tabs, no third-party names or logos.
