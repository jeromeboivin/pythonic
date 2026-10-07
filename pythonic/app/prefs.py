"""
Preferences of the app core (inventory cluster P): the saved settings as
``pref.*`` addresses. A set saves the preferences file at once, with the keys
and values tkinter has always written (older versions read the file, and a
file of an older version is read: missing keys take their defaults), and
applies what applies live. Not undoable; ``set`` runs on the caller's thread.
``describe`` gives the range, unit and default; for a device, rate or buffer
size, ``labels`` lists the values a front-end offers in its menu.

Audio (the setup sheet's audio tab):

- ``pref.audio.device`` (output device name, None for the system default),
  ``pref.audio.buffer_ms``, ``pref.audio.sample_rate`` and
  ``pref.audio.synth_rate`` (the internal synth rate, 0 = same as the output
  rate) are stream settings: they are saved at once and applied by the
  ``audio.apply`` verb, which restarts the stream.
  ``pref.audio.pending`` (read-only) lists those whose saved value waits for
  ``audio.apply``.
- ``pref.audio.mono`` applies at once (``audio.mono`` follows at block start).
- ``pref.audio.input_device`` (input device name or None): the PO-32 import's
  recording input.

Synthesis: ``pref.smoothing_ms`` (5..100 ms), the parameter smoothing time of
every channel, applied at once.

AI (shared by the AI settings and the AI generator): ``pref.ai.pattern_model``
and ``pref.ai.patch_model`` (checkpoint paths, None for the bundled one),
``pref.ai.pattern_temperature`` and ``pref.ai.patch_temperature`` (0.1..3).

Files: ``pref.preset_folder`` (an existing folder: the folder of
``preset.files`` and of relative preset paths) and ``pref.recent_files``
(read-only).

Front-end settings: ``pref.ui.<name>`` (``name`` of lower-case letters,
digits and ``_``) keeps any JSON value for a front-end (saved under the key
``ui_<name>``; None until set).

MIDI settings are the ``midi.*`` addresses and verbs (``midi.base_note``,
``midi.clock_sync``, ``midi.cc_map``, ``midi.pitchbend_target``; the device and
MIDI on/off are saved by ``midi.open`` / ``midi.close``).

Verbs: ``audio.rescan`` lists the audio devices again (``audio.output_devices``,
``audio.input_devices``, ``audio.default_input`` and the device labels);
``audio.rates`` (device=None) returns ``{'device', 'rates'}``, the output
rates of ``pref.audio.sample_rate``'s menu the device accepts (all of them if
the device cannot be probed).
"""

import math
import os
import re

from .registry import Address

BUFFER_SIZES_MS = (2.0, 5.0, 10.0, 15.0, 23.8, 30.0, 50.0, 75.0, 100.0)
SAMPLE_RATES = (96000, 48000, 44100, 32000, 22050, 11025, 8000)
SYNTH_RATES = (0, 22050, 11025, 8000)  # 0: same as the output rate
STREAM_SETTINGS = ('pref.audio.device', 'pref.audio.buffer_ms', 'pref.audio.sample_rate',
                   'pref.audio.synth_rate')
_UI_NAME = re.compile(r'pref\.ui\.([a-z0-9_]+)$')


def effective_synth_rate(synth_rate, sample_rate):
    """The synth rate a stream runs at: the output rate for 0, never above it."""
    return int(min(synth_rate if synth_rate > 0 else sample_rate, sample_rate))


class Prefs:
    """Preference addresses and the audio device verbs of an AppCore."""

    def __init__(self, core):
        self._core = core
        self._entries = {}
        self.applied = self.stream_settings()  # what the stream was built with

    @property
    def manager(self):
        return self._core.preferences

    # ================================================================== registration
    def register(self, registry):
        reg = registry.register
        audio = self._core.audio
        pref = self.manager

        def saved(key, cast=None, default=None):
            def get():
                value = pref.get(key, default)
                return cast(value) if cast is not None and value is not None else value
            return get

        def stream(name, get, set_, **kwargs):
            self._entries[name] = reg(Address(
                name, get=get, set=set_, queued=False, undoable=False,
                related=('pref.audio.pending',), **kwargs))

        stream('pref.audio.device', saved('audio_output_device'),
               lambda v: self._save('audio_output_device', v or None), kind='str',
               labels=tuple(audio.output_devices()))
        stream('pref.audio.buffer_ms', saved('audio_buffer_ms', float, 23.8),
               lambda v: self._save('audio_buffer_ms', v), minimum=BUFFER_SIZES_MS[0],
               maximum=BUFFER_SIZES_MS[-1], default=23.8, unit='ms', labels=BUFFER_SIZES_MS)
        stream('pref.audio.sample_rate', saved('audio_sample_rate', int, 44100),
               self._set_sample_rate, kind='int', minimum=SAMPLE_RATES[-1],
               maximum=SAMPLE_RATES[0], default=44100, unit='Hz', labels=SAMPLE_RATES)
        stream('pref.audio.synth_rate', self._synth_rate, self._set_synth_rate, kind='int',
               minimum=0, maximum=SAMPLE_RATES[0], default=0, unit='Hz', labels=SYNTH_RATES)
        reg(Address('pref.audio.pending', get=self.pending, kind='list'))
        reg(Address('pref.audio.mono', get=saved('audio_mono', bool, False),
                    set=self._set_mono, kind='bool', default=False, queued=False,
                    undoable=False))
        self._entries['pref.audio.input_device'] = reg(Address(
            'pref.audio.input_device', get=saved('audio_input_device'),
            set=lambda v: self._save('audio_input_device', v or None), kind='str',
            queued=False, undoable=False, labels=tuple(audio.input_devices())))
        reg(Address('audio.default_input', get=audio.default_input_name, kind='str'))

        reg(Address('pref.smoothing_ms', get=saved('param_smoothing_ms', float, 30.0),
                    set=self._set_smoothing, minimum=5.0, maximum=100.0, default=30.0,
                    unit='ms', queued=False, undoable=False))

        for name, key, default in (('pattern', 'drum_generator_pattern_model_path', None),
                                   ('patch', 'drum_generator_model_path', None)):
            reg(Address(f'pref.ai.{name}_model', get=saved(key),
                        set=lambda v, k=key: self._save(k, str(v) if v else None),
                        kind='str', queued=False, undoable=False))
        for name, key, default in (('pattern', 'drum_generator_pattern_temperature', 0.7),
                                   ('patch', 'drum_generator_patch_temperature', 1.0)):
            reg(Address(f'pref.ai.{name}_temperature', get=saved(key, float, default),
                        set=lambda v, k=key: self._save(k, v), minimum=0.1, maximum=3.0,
                        default=default, queued=False, undoable=False))

        reg(Address('pref.preset_folder', get=lambda: pref.get_preset_folder(),
                    set=self._set_preset_folder, kind='str', queued=False, undoable=False,
                    related=('preset.files',)))
        reg(Address('pref.recent_files', get=lambda: pref.get_recent_files(), kind='list'))
        registry.add_resolver('pref.ui.', self._resolve_ui)

    def verbs(self):
        return {'audio.rescan': self._verb_rescan, 'audio.rates': self._verb_rates}

    def _resolve_ui(self, name):
        match = _UI_NAME.match(name)
        if not match:
            return None
        key = f'ui_{match.group(1)}'
        return Address(name, get=lambda: self.manager.get(key),
                       set=lambda v: self._save(key, v), kind='json', queued=False,
                       undoable=False)

    # ================================================================== setters
    def _save(self, key, value):
        if not self.manager.set(key, value):
            raise OSError(f'the preferences file could not be written ({key})')

    def _set_sample_rate(self, rate):
        same_as_output = self._synth_rate() == 0
        self._save('audio_sample_rate', rate)
        if same_as_output:
            self._save('synth_sample_rate', rate)

    def _synth_rate(self):
        """0 when the synth runs at the output rate, else its lower rate."""
        out = int(self.manager.get('audio_sample_rate', 44100))
        stored = int(self.manager.get('synth_sample_rate', 44100) or out)
        return stored if stored < out else 0

    def _set_synth_rate(self, rate):
        out = int(self.manager.get('audio_sample_rate', 44100))
        self._save('synth_sample_rate', effective_synth_rate(rate, out))

    def _set_mono(self, mono):
        self._save('audio_mono', bool(mono))
        core = self._core
        core.submit(lambda m: core.synth.set_mono(m), bool(mono), ('audio.mono',))

    def _set_smoothing(self, ms):
        self._save('param_smoothing_ms', ms)
        core = self._core

        def apply(value):
            for channel in core.synth.channels:
                channel.set_smoothing_time(value)
        core.submit(apply, ms)

    def _set_preset_folder(self, folder):
        folder = os.path.abspath(os.path.expanduser(str(folder)))
        if not os.path.isdir(folder):
            raise ValueError(f'pref.preset_folder: {folder} is not a folder')
        self._save('preset_folder', folder)

    # ================================================================== stream settings
    def stream_settings(self):
        """The saved stream settings, as the stream would use them."""
        pref = self.manager
        rate = int(pref.get('audio_sample_rate', 44100))
        return {
            'pref.audio.device': pref.get('audio_output_device'),
            'pref.audio.buffer_ms': float(pref.get('audio_buffer_ms', 23.8)),
            'pref.audio.sample_rate': rate,
            'pref.audio.synth_rate': effective_synth_rate(
                int(pref.get('synth_sample_rate', rate) or rate), rate),
        }

    def pending(self):
        saved = self.stream_settings()
        return [name for name in STREAM_SETTINGS
                if not _same(saved[name], self.applied[name])]

    def set_stream_settings(self, device, sample_rate, synth_rate, buffer_ms):
        """Save stream settings given to audio.apply (synth_rate 0 = same as output)."""
        self._save('audio_output_device', device or None)
        self._save('audio_buffer_ms', float(buffer_ms))
        self._save('audio_sample_rate', int(sample_rate))
        self._save('synth_sample_rate', effective_synth_rate(int(synth_rate), int(sample_rate)))

    # ================================================================== verbs
    def _verb_rescan(self):
        audio = self._core.audio
        outputs, inputs = audio.output_devices(), audio.input_devices()
        self._entries['pref.audio.device'].labels = tuple(outputs)
        self._entries['pref.audio.input_device'].labels = tuple(inputs)
        self._core.note_changes(['audio.output_devices', 'audio.input_devices',
                                 'audio.default_input'])
        return {'output_devices': outputs, 'input_devices': inputs}

    def _verb_rates(self, device=None):
        rates = self._core.audio.supported_rates(device, SAMPLE_RATES)
        return {'device': device, 'rates': rates or list(SAMPLE_RATES)}


def _same(a, b):
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b))
        except (TypeError, ValueError):
            return False
    return a == b
