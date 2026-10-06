"""
A stand-in for the mido module, so app core tests inject MIDI messages without
hardware. ``FakeMidiPort.send(msg)`` calls the core's input callback on the
test thread, as the MIDI backend's own thread would on a real device.
"""

from types import SimpleNamespace


def message(kind, **fields):
    """A MIDI message with mido's attribute names (type, note, velocity, ...)."""
    return SimpleNamespace(type=kind, **fields)


def note_on(note, velocity=100):
    return message('note_on', note=note, velocity=velocity, channel=0)


def cc(control, value):
    return message('control_change', control=control, value=value, channel=0)


def program_change(program):
    return message('program_change', program=program, channel=0)


def pitchwheel(pitch):
    return message('pitchwheel', pitch=pitch, channel=0)


class FakeMidiPort:
    def __init__(self, name, callback):
        self.name = name
        self.callback = callback
        self.closed = False

    def send(self, msg):
        """Deliver one incoming message to the core."""
        assert not self.closed, 'message sent to a closed port'
        self.callback(msg)

    def close(self):
        self.closed = True


class FakeMidiBackend:
    """Implements the parts of the mido API the app core uses."""

    def __init__(self, ports=('Fake Keys', 'Fake Pads')):
        self.ports = list(ports)
        self.opened = []
        self.fail_open = False

    @property
    def port(self):
        """The last port the core opened."""
        return self.opened[-1] if self.opened else None

    def get_input_names(self):
        return list(self.ports)

    def open_input(self, name, callback=None):
        if self.fail_open:
            raise OSError('fake open failure')
        if name not in self.ports:
            raise OSError(f'unknown port {name}')
        port = FakeMidiPort(name, callback)
        self.opened.append(port)
        return port
