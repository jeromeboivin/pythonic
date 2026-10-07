# Release checklist

CI (`.github/workflows/ci.yml`) installs, runs the tests and starts both
interfaces headless on Linux, Windows and macOS. Before a release, check by
hand on each OS (Linux, Windows, macOS) what CI cannot:

- [ ] Fresh venv: `pip install -e .` succeeds (`".[ml]"` too); `pythonic` starts the web interface.
- [ ] **Audio out:** START/STOP plays the pattern audibly; changing the output device and buffer in setup restarts the stream.
- [ ] **MIDI in:** a connected controller triggers channels, learned CCs move their controls, MIDI clock sync follows.
- [ ] **File dialogs:** open and save a preset, load and save a drum patch, choose the preset folder; the dialogs are native (a sheet on macOS; KDE portal where it runs) and the panel keeps playing while they are open.
- [ ] **Fonts:** labels in Barlow Condensed, the green display in Doto, the tempo in DSEG7 (no system fallbacks).
- [ ] The window opens at 1280x800 and letterboxes when resized; it cannot shrink below 1280x800.
- [ ] `pythonic --devtools` opens DevTools on `http://127.0.0.1:9222`.
- [ ] `pythonic --ui tk` starts the tkinter interface; presets, patterns and preferences saved by one interface load in the other.
- [ ] Without PySide6 (`pip uninstall PySide6-Addons`), `pythonic` starts tkinter with the explain-and-install dialog; Install works, a restart opens the web interface.
- [ ] Windows ARM64 (if available): installs without PySide6, `pythonic` starts tkinter with the explanation and no Install button.
- [ ] Linux only if needed: on a locked-down distro, container or as root, `QTWEBENGINE_DISABLE_SANDBOX=1` is the documented fallback.
