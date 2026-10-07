"""
A stand-in for the sounddevice module, so app core tests and the Tk smoke test
run without an audio device. The test calls the audio callback by hand with
FakeStream.pull().
"""

import threading

import numpy as np


class FakeStatus:
    """Callback status flags; falsy unless an underflow is simulated."""

    def __init__(self, underflow=False):
        self.output_underflow = underflow

    def __bool__(self):
        return self.output_underflow


class FakeStream:
    def __init__(self, backend, device=None, channels=2, callback=None,
                 samplerate=44100, blocksize=1024, dtype=None):
        self.backend = backend
        self.device = device
        self.channels = channels
        self.callback = callback
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.started = False
        self.aborted = False
        self.closed = False

    def start(self):
        if self.backend.start_hangs.is_set():
            self.backend.release.wait()
        if self.backend.fail_start:
            raise RuntimeError('fake start failure')
        self.started = True

    def stop(self):
        raise AssertionError('the core must restart streams with abort(), not stop()')

    def abort(self):
        if self.backend.abort_hangs.is_set():
            self.backend.release.wait()
        self.aborted = True
        self.started = False

    def close(self):
        self.closed = True

    def pull(self, frames=None, underflow=False):
        """Run one audio callback, as PortAudio would, and return the block."""
        frames = frames or self.blocksize
        out = np.full((frames, 2), np.nan, dtype=np.float32)
        self.callback(out, frames, None, FakeStatus(underflow))
        return out


class FakeInputStream:
    """An input stream: the test feeds it samples with push(), which runs the
    callback as PortAudio would."""

    def __init__(self, backend, device=None, channels=1, callback=None,
                 samplerate=44100, blocksize=2048, dtype=None):
        self.backend = backend
        self.device = device
        self.channels = channels
        self.callback = callback
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.started = False
        self.aborted = False
        self.closed = False

    def start(self):
        if self.backend.fail_input:
            raise RuntimeError('fake input failure')
        self.started = True

    def stop(self):
        self.started = False

    def abort(self):
        self.aborted = True
        self.started = False

    def close(self):
        self.closed = True

    def push(self, samples):
        """Run the callback over `samples` (mono floats), block by block."""
        samples = np.asarray(samples, dtype=np.float32)
        for start in range(0, len(samples), self.blocksize):
            block = samples[start:start + self.blocksize].reshape(-1, 1)
            block = np.repeat(block, self.channels, axis=1)
            self.callback(block, len(block), None, FakeStatus())


class FakeAudioBackend:
    """Implements the parts of the sounddevice API the app core uses."""

    def __init__(self, devices=None, supported_rates=(44100, 48000, 22050)):
        self.devices = devices if devices is not None else [
            {'name': 'Fake Out', 'max_output_channels': 2, 'max_input_channels': 0,
             'default_samplerate': 44100.0},
            {'name': 'Fake In', 'max_output_channels': 0, 'max_input_channels': 2,
             'default_samplerate': 44100.0},
            {'name': 'Fake Duplex', 'max_output_channels': 2, 'max_input_channels': 2,
             'default_samplerate': 48000.0},
        ]
        self.supported_rates = set(supported_rates)
        self.default = type('Default', (), {'device': [1, 0]})()
        self.streams = []
        self.inputs = []
        self.fail_input = False
        self.fail_open = False
        self.fail_start = False
        self.abort_hangs = threading.Event()
        self.start_hangs = threading.Event()
        self.release = threading.Event()

    @property
    def stream(self):
        return self.streams[-1] if self.streams else None

    def query_devices(self, device=None, kind=None):
        if kind == 'input':
            return self.devices[self.default.device[0]]
        if kind == 'output':
            return self.devices[self.default.device[1]]
        if device is None:
            return list(self.devices)
        return self.devices[device]

    def check_output_settings(self, device=None, channels=2, samplerate=44100, **_kw):
        if samplerate not in self.supported_rates:
            raise ValueError(f'unsupported rate {samplerate}')

    def OutputStream(self, **kwargs):
        if self.fail_open:
            raise RuntimeError('fake open failure')
        stream = FakeStream(self, **kwargs)
        self.streams.append(stream)
        return stream

    def InputStream(self, **kwargs):
        stream = FakeInputStream(self, **kwargs)
        self.inputs.append(stream)
        return stream

    @property
    def input(self):
        return self.inputs[-1] if self.inputs else None
