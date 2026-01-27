# Pythonic

A synthetic drum synthesizer built in Python that generates percussion sounds algorithmically without using samples.

Perfect for producers, sound designers, and developers interested in audio synthesis and digital signal processing.

## Features

- **8 Drum Channels** - Each with independent synthesis parameters
- **Oscillator Section**
  - Sine, Triangle, and Sawtooth waveforms
  - Pitch modulation with Decay, Sine (FM), and Random modes
  - Attack and decay envelope
- **Noise Section**
  - Low-pass, Band-pass, High-pass filters
  - Stereo mode for wide sounds
  - Exponential, Linear, and Modulated envelope modes
- **Mixing Section**
  - Oscillator/Noise mix control
  - Distortion with soft-clipping
  - Parametric EQ
  - Level and Pan controls
  - Choke groups
- **Velocity Sensitivity** - Per-channel velocity response
- **Real-time Audio** - Instant sound response using sounddevice
- **Preset System** - Save and load your drum kits

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setup

1. **Create a virtual environment (recommended):**

```bash
cd pythonic
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

## Running the Application

```bash
python run.py
```

## Controls

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| 1-8 | Trigger drum channels 1-8 |
| S | Save preset |
| L | Load preset |

### Mouse Controls

- **Knobs**: Click and drag vertically to adjust
- **Sliders**: Click and drag to adjust
- **Buttons**: Click to toggle or select
- **Double-click knobs**: Reset to default value
- **Mouse wheel**: Fine adjustment on knobs

## Architecture

```
pythonic-python/
├── pythonic/           # Core synthesis engine
│   ├── synthesizer.py    # Main 8-channel synthesizer
│   ├── drum_channel.py   # Individual drum channel
│   ├── oscillator.py     # Waveform generation
│   ├── noise.py          # Noise generator
│   ├── envelope.py       # ADSR envelopes
│   └── filter.py         # State variable filters
├── gui/                  # User interface
│   ├── main_window.py    # Main application window
│   └── widgets.py        # Custom GUI widgets
├── run.py               # Application entry point
└── requirements.txt     # Python dependencies
```

## Factory Presets

The synth comes initialized with 8 classic drum sounds:

1. **Kick** - Deep bass drum with pitch sweep
2. **Snare** - Punchy snare with noise body
3. **Closed HH** - Tight closed hi-hat
4. **Open HH** - Sustaining open hi-hat
5. **Tom Hi** - High tom-tom
6. **Tom Lo** - Low tom-tom  
7. **Clap** - Handclap with modulated envelope
8. **Rim** - Rimshot click

## Technical Details

### Signal Flow

```
Trigger → Oscillator → Envelope → ╗
                                  ╠═→ Mix → Distortion → EQ → Pan → Level → Output
Trigger → Noise → Filter → Env → ╝
```

### Sample Rate

Default: 44100 Hz (CD quality)

### Audio Latency

Approximately 11.6ms at 512 sample buffer size

## Troubleshooting

### No Audio

1. Make sure `sounddevice` is installed: `pip install sounddevice`
2. Check your system audio output device
3. Try increasing the buffer size in `main_window.py`

### High CPU Usage

- Reduce the number of active channels
- Increase audio buffer size
- Close other audio applications

**Python Implementation** - Educational project

## License

MIT License.

## Contributing

Contributions are welcome! Areas for improvement:

- [ ] Pattern sequencer
- [ ] MIDI input support
- [ ] Preset browser
- [ ] Export to WAV
- [ ] Additional waveforms
- [ ] Effects (reverb, delay)
