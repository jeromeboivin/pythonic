# Pythonic

A modern synthetic drum synthesizer built entirely in Python with algorithmic sound generation—no samples required. It comes with two interfaces on one shared app core: a hardware-style panel (HTML/JS in a Qt window, the default) and the classic tkinter window. Both have MIDI support, a 64-step pattern sequencer with per-step velocity, presets, exports and an optional AI drum generator.

Perfect for producers, sound designers, and developers interested in audio synthesis and digital signal processing.

## 🎵 Features

### Core Synthesis
- **8 Independent Drum Channels** with full parameter control
- **Oscillator Section**
  - Sine, Triangle, and Sawtooth waveforms
  - Pitch modulation: Decay, Sine (FM), and Random modes
  - Attack and decay envelope shaping
- **Noise Generator**
  - Low-pass, Band-pass, High-pass filtering
  - Stereo width control
  - Exponential, Linear, and Modulated envelope modes
- **Advanced Mixing**
  - Oscillator/Noise blend control
  - Soft-clipping distortion
  - Parametric EQ per channel
  - Level and Pan controls
  - Choke groups for realistic hi-hat behavior
- **Modulation** - Two LFOs and a pump per channel, aimed at any sound parameter, the master level or the sound morph

### Effects & Processing
- **Vintage Character** - Analog warmth and saturation
- **Reverb** - Space and ambience
- **Delay** - Rhythmic echoes and timing effects
- **Smoothed Parameters** - Anti-zipper noise on parameter changes

### Workflow Features
- **Two interfaces, one core** - The web panel and the tkinter window share presets, patterns and preferences
- **MIDI Support** - Notes, program changes, clock sync, learned CCs with pickup, pitch bend
- **Pattern Sequencer** - 12 patterns of 1-64 steps with velocity, accents, fills, probability, substeps and chains
- **Sound Morph** - Blend between two learned sets of sounds
- **Undo / Redo** - Every edit, drag and preset change
- **Preset System** - Save and load whole presets and single drum patches
- **Exports** - Patterns to MIDI or audio, drums to WAV, transfers to and from a PO-32
- **Real-time Audio** - Low-latency playback via `sounddevice`

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** (tested up to Python 3.12)
- **pip** package manager
- **MIDI interface** (optional, for MIDI control)

### Installation

1. **Clone or download this repository**

```bash
cd pythonic
```

2. **Create a virtual environment (recommended)**

```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

3. **Install the app**

```bash
pip install -e .            # the app and both interfaces
pip install -e ".[dev]"     # plus the test tools (pytest, pytest-qt, soundfile)
pip install -e ".[ml]"      # plus PyTorch for the AI drum generator
```

The base install brings PySide6 with QtWebEngine for the web interface
(about 650 MB), except on Windows ARM64, where no QtWebEngine exists and the
tkinter interface is used.

> **Note**: If you encounter issues with `mido` or `python-rtmidi`, they're only required for MIDI functionality. The synth works without them.

> **Note**: The drum voice engine is compiled with `numba`. The first launch takes a few extra seconds while it compiles; the result is cached for later runs.

4. **Optional: the panel's fonts**

The web panel is designed for three SIL OFL fonts (Barlow Condensed for
labels, Doto for the green display, DSEG7 for the tempo). They are not in the
repository; without them the panel uses system fonts (DejaVu Sans Condensed
and DejaVu Sans Mono, or the platform's equivalents), which work but look
less like the hardware. To fetch them once:

```bash
python tools/fetch_fonts.py
```

It downloads the four `.woff2` files and their licences into
`pythonic/web/static/fonts/` (sources and licences are listed in
`pythonic/web/static/fonts/SOURCES.txt`); restart the app to use them.

### Run the Application

```bash
pythonic                    # the web interface (default)
pythonic --ui tk            # the tkinter interface
pythonic --devtools [PORT]  # web interface with Chromium DevTools on http://127.0.0.1:9222 (or PORT)
pythonic --quit-after 5     # close after 5 s (start-up checks; the web interface prints its frame timings)
pythonic --help
```

`python run.py`, `python -m pythonic` and `./run.sh` take the same arguments.
One interface runs per launch; no interface preference is stored, only the
flag picks it. If the web interface is asked for but PySide6 with
QtWebEngine is missing, the tkinter interface starts and offers to install it
(on Windows ARM64 it only explains why).

On first launch, Pythonic automatically creates:
- Configuration directory for preferences
- `Pythonic Presets` folder in your Documents
- Default factory presets

## 🎹 Usage

### The web interface

A fixed-size panel scaled to the window (letterboxed when the window's shape
differs; the window cannot shrink below 1280×800):

- **Left column**: undo / redo, the step modes (trig, accent, velo, fill,
  prob, sub), last step, all ch, follow, ⊞ matrix, the patterns A-L with
  chain, menu, lane copy / paste, and the 16 programs.
- **Centre**: master, step rate, fill rate and the strip CTRL mode over
  eight channel strips (patch name, tune, decay, CTRL knob, level fader,
  channel button showing the drum type).
- **Right column**: the green display, PRESET ◀ ▶ and the preset menu, edit
  rack, PO-32, setup, the MUTE latch, morph learn A / B, tempo, swing, the
  MIDI LED and the sound morph knob.
- **Bottom**: START/STOP, page bars and the 16 pads, and under the face the
  **edit rack** of the selected channel (every sound parameter, the LFO and
  pump rows). The edit rack button closes it (the window shrinks; the state
  is remembered). The matrix, the PO-32 page and the AI drum generator open in
  its place; setup opens as a sheet over the panel.

Controls: drag knobs and faders vertically (Shift for fine steps), turn the
mouse wheel over any control, click a fader's track to jump, **double-click**
a knob or fader to type an exact value, **right-click** for reset to
default, MIDI learn, CC mappings and pitch bend. The display shows the
touched control and its value. A click on the selected channel's button hits
it. The web interface has no keyboard shortcuts.

### The tkinter interface

| Key | Action |
|-----|--------|
| `1-8` | Trigger drum channels 1-8 |
| `S` | Save preset |
| `L` | Load preset |
| `Ctrl+Z` / `Ctrl+Y` | Undo / redo |

- **Knobs and sliders**: click and drag vertically (Shift for fine steps), mouse wheel for small steps
- **Ctrl+click or double-click a knob**: reset to default
- **Right-click a control**: MIDI learn

### MIDI

1. Connect your MIDI controller
2. Launch Pythonic—it automatically detects MIDI devices (web: setup ▸ midi)
3. Drum channels follow MIDI notes (default: C2-G2, notes 36-43)
4. Velocity-sensitive response on all channels; program changes 0-11 select patterns A-L
5. Learned CCs move their controls once the controller reaches the value (pickup)

### Presets

**Loading**: web: the PRESET menu lists the preset folder and recent files,
◀ ▶ step through the folder; tkinter: the preset combo-box at the top

**Saving**: save preset as… opens the file browser in your preset folder

**Preset Location**:
- Windows: `%USERPROFILE%\Documents\Pythonic Presets`
- macOS: `~/Documents/Pythonic Presets`
- Linux: `~/Documents/Pythonic Presets`

**Preset Compatibility**:
- Pythonic can import `.mtpreset` presets and load and save `.mtdrum` drum patches
- Load drums and patterns directly from these presets

> **Note**: Pythonic uses its own synthesis engine. While it can read Microtonic presets, the resulting sounds may differ from the original due to differences in DSP implementation. Pythonic is an independent project and is not affiliated with, endorsed by, or sponsored by Sonic Charge or NuEdge Development. Microtonic™ is a trademark of Sonic Charge/NuEdge Development.

### Pattern Sequencer

**Basic Pattern Editing**:
- Click steps to toggle triggers, accents, and fills (web: pick the step mode, then click or paint the pads)
- Per-step velocity (1-127) beside the accent
- Patterns of 1 to 64 steps, shown 16 at a time on 4 pages that can follow the playhead
- Visual step indicator shows current playback position

**Per-Step Probability**:
- Enable probability mode to control the chance of each step triggering (0-100%)
- 100% = always plays, 50% = plays half the time, 0% = never plays
- Adds organic variation and humanization to patterns

**Substeps (Micro-Timing)**:
- Pick a substep pattern for a step (web: the sub step mode; tkinter: right-click a step)
- Define subdivisions using 'o' (play) and '-' (skip) notation
- Examples: "oo-" = play twice then skip, "o-o-" = alternating hits
- Creates flams, rolls, and complex rhythmic variations within a single step

## 📁 Project Structure

```
pythonic/
├── pythonic/              # Core synthesis engine
│   ├── synthesizer.py        # Main 8-channel synthesizer
│   ├── drum_channel.py       # Individual drum channel
│   ├── drum_generator.py     # AI drum generation (CVAE inference)
│   ├── oscillator.py         # Waveform generation
│   ├── noise.py              # Noise generator
│   ├── envelope.py           # ADSR envelope generators
│   ├── filter.py             # State-variable filters
│   ├── reverb.py             # Reverb effect
│   ├── delay.py              # Delay effect
│   ├── vintage.py            # Analog character processing
│   ├── app/                  # App core (UI-free): audio, transport, undo, MIDI, presets, exports, AI, PO-32
│   ├── web/                  # Web interface: Qt window, bridge, static/ (HTML, CSS, JS modules)
│   ├── launch.py             # The `pythonic` command
│   ├── pattern_manager.py    # Step sequencer
│   ├── preset_manager.py     # Preset save/load
│   ├── preferences_manager.py # Settings persistence
│   └── smoothed_parameter.py # Parameter smoothing
├── gui/                   # tkinter interface
│   ├── main_window.py        # Main application window
│   ├── drum_generator_dialog.py # AI drum generator dialog
│   └── widgets.py            # Custom GUI components
├── tests/                 # Unit and integration tests (tests/web: the web interface)
├── docs/                  # Core interface, web front-end, parity lists, release checklist
├── test_patches/          # Test presets and analysis
├── tools/                 # Development utilities (fetch_fonts.py, ...)
├── run.py                 # Application entry point (same arguments as `pythonic`)
├── run.sh                 # Shell launcher script
└── pyproject.toml         # Package, dependencies and extras ([ml], [dev])
```

Developer documentation: [docs/core-interface.md](docs/core-interface.md)
(the app core both interfaces drive), [docs/web-frontend.md](docs/web-frontend.md)
(the web interface), [docs/release-checklist.md](docs/release-checklist.md).

## 🔊 Signal Flow

```
MIDI/Key Input
      ↓
   Trigger
      ↓
   ┌────────────────────┐
   │ Oscillator         │──→ Envelope ──┐
   │ (Sine/Tri/Saw)     │               │
   └────────────────────┘               ├──→ Mix ──→ Distortion ──→ EQ ──→ Vintage ──→ Delay ──→ Reverb ──→ Output
   ┌────────────────────┐               │
   │ Noise Generator    │──→ Filter ──→─┘
   │ (LP/BP/HP)         │    Envelope
   └────────────────────┘
```

## 🎯 Factory Presets

Pythonic includes 8 built-in drum sounds:

1. **Kick** - Deep bass drum with exponential pitch sweep
2. **Snare** - Punchy snare with noise body
3. **Closed Hi-Hat** - Tight, crisp hi-hat
4. **Open Hi-Hat** - Sustaining open hi-hat
5. **High Tom** - Tuned high tom
6. **Low Tom** - Tuned low tom
7. **Clap** - Hand clap with modulated envelope
8. **Rim Shot** - Sharp rimshot click

## 🤖 AI Drum Generator

Pythonic includes an optional AI-powered drum generator that uses a Conditional Variational Autoencoder (CVAE) trained on hundreds of drum patches.

### Optional ML Installation

The generator requires PyTorch, which is **not** included in the base install to keep the app lightweight. It runs in a separate worker process, so the interface never imports PyTorch.

Pre-trained checkpoint files in this repository are stored with Git LFS. If you cloned without LFS enabled, install it and fetch the model payloads before using AI features:

```bash
git lfs install
git lfs pull
```

**Option A — In-app install**: open the AI drum generator (web: PRESET menu → *AI drum generator…*, then *install now*; tkinter: Preset menu → *AI Drum Generator...* → **Install ML Support**). This runs `pip install torch` (the `[ml]` extra) in the current environment after confirmation.

**Option B — Manual install**:

```bash
pip install -e ".[ml]"
```

> For GPU acceleration, install the appropriate CUDA/ROCm PyTorch variant first following [pytorch.org](https://pytorch.org/get-started/locally/). The CPU-only build works fine for inference.

### Models

The repository includes these pre-trained checkpoints, used by default:

| File | Description |
|------|-------------|
| `drum_cvae_best.pt` | Bundled drum-patch CVAE checkpoint used by the AI Drum Generator |
| `drum_patterns/pattern_cvae_best.pt` | Bundled pattern CVAE checkpoint used for AI pattern generation |
| `fit_prior.pt` | Starting-patch model of `tools/fit_samples.py` (see below) |

Other checkpoints can be chosen with **load…** on the AI page or in setup ▸ ai (tkinter: **Load Model...** and Preset menu → **AI Settings**). The chosen paths are saved in preferences.

### Generator Workflow (web interface)

1. Open the PRESET menu and choose **AI drum generator…**: the page opens under the strips, one lane per channel
2. Adjust the patch **temperature** (lower = conservative, higher = experimental), the **candidates** count and an optional **seed**
3. Pick a drum type per lane (it starts from the channel's drum type) and click **gen**, or **generate all 8**
4. The first candidate plays on the face at once; browse candidates with ‹ / ›, edit them with the panel's controls
5. **keep tried** makes them part of the preset (one undo step); **revert all** brings the old sounds back; leaving the page asks which
6. On the right, keep the current patterns or generate new ones, preview them (**▶ loop**, **▶ bank A→L**) and **replace patterns**

In the tkinter dialog, candidates are previewed per slot and **Apply Selected** copies the checked slots into the preset.

### Drum Patches from Samples

`tools/fit_samples.py` turns a folder of one-shot samples (WAV, AIFF or FLAC) into
`.mtdrum` drum patches, one per sample, named after it:

```bash
pip install -e ".[ml,dev]"    # PyTorch for the starting-patch model, soundfile to read samples
python tools/fit_samples.py path/to/samples path/to/patches --wav
```

For each sample it builds starting patches from the partials it finds and from
the predictions of `fit_prior.pt`, then refines them with CMA-ES against the
sample's spectrogram, its pitch below 2 kHz and its envelope. Level is set so
every patch peaks at -10 dBFS (`--peak-db`); hi-hats get choke. `--wav` also writes
`<name>.fit.wav`, the sample on the left and the patch on the right, to compare
them by ear; `fit_report.csv` lists the scores (lower is closer). A sample takes
from one minute (a short rim shot) to half an hour (a long crash) per CPU core;
`--jobs` sets how many are fitted in parallel, `--evals` and `--screen-evals`
trade time for accuracy. One oscillator and one filtered noise match kicks, toms,
snares and other pitched drums closely, cymbals and metallic sounds less so.

`tools/fit_prior.py <patches dir>` retrains the starting-patch model on your own
`.mtdrum` patches (one to two hours on 8 CPU cores for the default 60,000).

## ⚙️ Technical Specifications

- **Sample Rate**: 44.1 kHz (CD quality)
- **Bit Depth**: 32-bit float internal processing
- **Audio Backend**: PortAudio via `sounddevice`
- **Latency**: ~23.8 ms by default (buffer and sample rate in the audio settings; 512 frames, 11.6 ms, is the safe minimum)
- **MIDI Backend**: RtMidi via `python-rtmidi` and `mido`
- **GUI Frameworks**: PySide6 QtWebEngine (HTML/CSS/JS ES modules, no build step) and Tkinter, on one app core

## 🧪 Testing

Install the dev extra (`pip install -e ".[dev]"`), then:

```bash
python -m pytest tests -q          # everything (~2-4 min)
python -m pytest tests/web -q      # the web interface only
./run_tests.sh                     # creates ./venv if needed, installs, runs the tests
```

The web tests run headless (`QT_QPA_PLATFORM=offscreen`, set by
`tests/web/conftest.py`) and need no Node and no display; the tkinter smoke
test needs a display on Linux (it skips without one; use `xvfb-run` on a
server). Pure JS specs can also run with `node --test
pythonic/web/static/test/*.test.js`.

The test suite includes:
- Synthesis accuracy validation and audio output comparison tests
- App core tests through its interface (addresses, verbs, poll) on a fake audio stream
- Web page tests driving the real page over a fake core, and end-to-end tests over the real core
- A parity guard: every core address is bound by a web control or listed as deliberately absent

## 🐛 Troubleshooting

### No Audio Output

1. Verify `sounddevice` installation: `pip install --upgrade sounddevice`
2. Check system audio output device is working
3. Pick another output device or a larger buffer: web: setup ▸ audio (then *restart audio*); tkinter: Preset menu → *Audio Settings...*
4. List available devices: `python -c "import sounddevice; print(sounddevice.query_devices())"`

### MIDI Not Working

1. Ensure your MIDI device is connected before launching (or rescan in setup ▸ midi)
2. Check if `mido` and `python-rtmidi` are installed
3. On Linux, you may need ALSA MIDI permissions
4. Test MIDI: `python test_midi_input.py`

### The web interface does not start

- Check that PySide6 with QtWebEngine imports: `python -c "import PySide6.QtWebEngineWidgets"`; `pythonic --ui tk` always works.
- Linux: on a locked-down distribution, in a container or as root, Chromium's sandbox may refuse to start; `QTWEBENGINE_DISABLE_SANDBOX=1 pythonic` is the fallback.
- `pythonic --devtools` opens the Chromium DevTools on `http://127.0.0.1:9222` to inspect the page.
- Windows: the panel renders without the GPU, because QtWebEngine 6.11's GPU path crashes the app after a minute or two of playing. SETUP ▸ display ▸ use the GPU turns it back on at the next start (to try a newer PySide6, say); turn it off there if the panel crashes on another system.

### High CPU Usage

- Increase audio buffer size (reduces real-time load)
- Disable unused effects (Reverb, Delay)
- Close other audio applications
- Use fewer simultaneous voices

### Installation Issues

**macOS**: May need to install PortAudio: `brew install portaudio`
**Linux**: Install dependencies: `sudo apt install python3-dev portaudio19-dev libasound2-dev libjack-dev`
**Windows**: Usually works out of the box with pip

## 📚 Additional Documentation

- [QUICK_START.md](QUICK_START.md) - First steps, presets and preferences

---

**Built with Python** | **Educational & Open Source** | **No Samples Required**

Contributions are welcome! Areas for improvement:

- [ ] Additional waveforms
- [ ] More effects
- [ ] Packaged installers per OS
