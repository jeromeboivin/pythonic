# Slice W1: web shell, bridge, packaging and test harness

Manual checks after the first web slice (run from a checkout with
`pip install -e ".[dev]"`, or `python run.py ...`).

## Web interface
- [ ] `pythonic` (or `pythonic --ui web`) opens a 1280x800 window with the black panel: PYTHONIC wordmark, green display, red tempo display, START/STOP and 16 pads in four colour groups.
- [ ] Resize wider or taller: the panel keeps its 16:10 shape, scales and stays centred with black bars; the window cannot shrink below 1280x800.
- [ ] Tempo + / - and the mouse wheel over the tempo change it by 1 BPM; the green display shows "TEMPO n BPM" for 1.5 s.
- [ ] START/STOP lights and plays the last preset's pattern (audible); the white outline walks along the pads and line 1 shows the step; START/STOP again stops.
- [ ] A tempo change from a MIDI controller mapped to tempo shows on the tempo display.
- [ ] Closing the window stops the audio and the process exits.
- [ ] `pythonic --devtools` prints the DevTools address; `http://127.0.0.1:9222` lists `app://ui/index.html` and inspects it.

## tkinter interface
- [ ] `pythonic --ui tk` (and `python run.py --ui tk`, `./run.sh --ui tk`) starts the tkinter window as before; play, edit a knob, save and load a preset.
- [ ] AI drum generator dialog: "Install ML Support" offers `pip install torch` (only when torch is missing).
- [ ] In a venv without PySide6, `pythonic` starts tkinter with the "Web interface unavailable" dialog and its Install button.
