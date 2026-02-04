# Pythonic

A modern synthetic drum synthesizer built entirely in Python with algorithmic sound generation—no samples required. Features a full-featured GUI, MIDI support, pattern sequencer, and advanced DSP effects.

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

### Effects & Processing
- **Vintage Character** - Analog warmth and saturation
- **Reverb** - Space and ambience
- **Delay** - Rhythmic echoes and timing effects
- **Smoothed Parameters** - Anti-zipper noise on parameter changes

### Workflow Features
- **MIDI Support** - Full MIDI input with velocity sensitivity
- **Pattern Manager** - Built-in step sequencer with per-step probability and substeps
- **Preset System** - Save and load complete drum kits
- **Preferences Manager** - Remembers your settings and folder locations
- **Real-time Audio** - Low-latency playback via `sounddevice`

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** (tested up to Python 3.12)
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

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

> **Note**: If you encounter issues with `mido` or `python-rtmidi`, they're only required for MIDI functionality. The synth works without them.

### Run the Application

```bash
python run.py
```

Or use the provided shell script:

```bash
./run.sh
```

On first launch, Pythonic automatically creates:
- Configuration directory for preferences
- `Pythonic Presets` folder in your Documents
- Default factory presets

## 🎹 Usage

### Keyboard Controls

| Key | Action |
|-----|--------|
| `1-8` | Trigger drum channels 1-8 |
| `S` | Save preset |
| `L` | Load preset |

### Mouse Controls

- **Knobs**: Click and drag vertically
- **Sliders**: Click and drag horizontally
- **Buttons**: Click to toggle
- **Double-click knobs**: Reset to default
- **Mouse wheel**: Fine adjustment

### MIDI

1. Connect your MIDI controller
2. Launch Pythonic—it automatically detects MIDI devices
3. Map drum channels to MIDI notes (default: C1-G1)
4. Velocity-sensitive response on all channels

### Presets

**Loading**: Use the preset combo-box at the top of the window, or click the folder icon to browse

**Saving**: Click "Save Preset" button—the file browser opens in your preset folder

**Preset Location**: 
- Windows: `%USERPROFILE%\Documents\Pythonic Presets`
- macOS: `~/Documents/Pythonic Presets`
- Linux: `~/Documents/Pythonic Presets`

**Microtonic Compatibility**:
- Pythonic can import `.mtpreset` files from Soniccharge Microtonic
- Load drums and patterns directly from Microtonic presets
- Provides an open-source alternative workflow for Microtonic users

> **Note**: Pythonic uses its own synthesis engine. While it can read Microtonic presets, the resulting sounds may differ from the original due to differences in DSP implementation. Pythonic is an independent project and is not affiliated with, endorsed by, or sponsored by Sonic Charge or NuEdge Development. Microtonic™ is a trademark of Sonic Charge/NuEdge Development.

### Pattern Sequencer

**Basic Pattern Editing**:
- Click steps to toggle triggers, accents, and fills
- Adjust pattern length per channel
- Visual step indicator shows current playback position

**Per-Step Probability**:
- Enable probability mode to control the chance of each step triggering (0-100%)
- 100% = always plays, 50% = plays half the time, 0% = never plays
- Adds organic variation and humanization to patterns

**Substeps (Micro-Timing)**:
- Right-click any step to access substep menu
- Define subdivisions using 'o' (play) and '-' (skip) notation
- Examples: "oo-" = play twice then skip, "o-o-" = alternating hits
- Creates flams, rolls, and complex rhythmic variations within a single step

## 📁 Project Structure

```
pythonic/
├── pythonic/              # Core synthesis engine
│   ├── synthesizer.py        # Main 8-channel synthesizer
│   ├── drum_channel.py       # Individual drum channel
│   ├── oscillator.py         # Waveform generation
│   ├── noise.py              # Noise generator  
│   ├── envelope.py           # ADSR envelope generators
│   ├── filter.py             # State-variable filters
│   ├── reverb.py             # Reverb effect
│   ├── delay.py              # Delay effect
│   ├── vintage.py            # Analog character processing
│   ├── midi_manager.py       # MIDI input handling
│   ├── pattern_manager.py    # Step sequencer
│   ├── preset_manager.py     # Preset save/load
│   ├── preferences_manager.py # Settings persistence
│   └── smoothed_parameter.py # Parameter smoothing
├── gui/                   # User interface (Tkinter)
│   ├── main_window.py        # Main application window
│   └── widgets.py            # Custom GUI components
├── tests/                 # Unit and integration tests
│   ├── test_synthesis.py     # Synthesis accuracy tests
│   ├── test_patterns.py      # Pattern manager tests
│   └── test_reference_wavs.py # Audio comparison tests
├── test_patches/          # Test presets and analysis
├── tools/                 # Development utilities
├── run.py                 # Application entry point
├── run.sh                 # Shell launcher script
└── requirements.txt       # Python dependencies
```

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

## ⚙️ Technical Specifications

- **Sample Rate**: 44.1 kHz (CD quality)
- **Bit Depth**: 32-bit float internal processing
- **Audio Backend**: PortAudio via `sounddevice`
- **Latency**: ~11.6ms (512-sample buffer, configurable)
- **MIDI Backend**: RtMidi via `python-rtmidi` and `mido`
- **GUI Framework**: Tkinter (cross-platform)

## 🧪 Testing

Run the test suite:

```bash
./run_tests.sh
```

Or manually:

```bash
pytest tests/ -v
```

The test suite includes:
- Synthesis accuracy validation
- Audio output comparison tests
- Pattern manager functionality
- Reference WAV generation

## 🐛 Troubleshooting

### No Audio Output

1. Verify `sounddevice` installation: `pip install --upgrade sounddevice`
2. Check system audio output device is working
3. Try increasing buffer size in [gui/main_window.py](gui/main_window.py)
4. List available devices: `python -c "import sounddevice; print(sounddevice.query_devices())"`

### MIDI Not Working

1. Ensure your MIDI device is connected before launching
2. Check if `mido` and `python-rtmidi` are installed
3. On Linux, you may need ALSA MIDI permissions
4. Test MIDI: `python test_midi_input.py`

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

- [QUICK_START.md](QUICK_START.md) - Detailed guide for new users

---

**Built with Python** | **Educational & Open Source** | **No Samples Required**

Contributions are welcome! Areas for improvement:

- [ ] Pattern sequencer
- [ ] MIDI input support
- [ ] Preset browser
- [ ] Export to WAV
- [ ] Additional waveforms
- [ ] Effects (reverb, delay)
