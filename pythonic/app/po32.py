"""
PO-32 module of the app core (inventory cluster Q): send the eight sounds to
a PO-32 over its audio modem, and import sounds and patterns from one.

Audio. Every stream is the core's, opened through its audio backend:

- **Sending** and **previewing** play a rendered buffer through the live
  output stream (``AudioEngine.play`` with a ``Player``): the buffer is
  rendered off the audio thread at the output rate, and after the synth has
  rendered a block the callback copies the next slice of it into the block.
  A transfer *replaces* the block (pads or a running pattern never reach the
  receiver); a preview is *mixed* in. Both need the stream to run.
- **Listening** and **recording** open a mono 44.1 kHz input stream; its
  callback publishes the block's peak level and, while recording, copies the
  samples into a buffer allocated when the recording starts (up to
  ``MAX_RECORD_SECONDS``; a full buffer ends the recording and decodes it).

Transfer (sounds only: the PO-32 pattern slots of the chain are sent empty,
with the default state block and a default morph patch per sound):

- ``po32.prepare`` (bank=0, chain=None, channels=None): render the modem
  signal from the current sounds. ``bank`` 0 sends to PO-32 sounds 1-8, 1 to
  9-16. ``chain``: one of ``po32.chain_options`` (``'A - B'``, ``'C'``...;
  default the first), a letter or a list of letters/indexes; its length is
  the number of empty pattern slots sent, from PO-32 pattern 1. ``channels``:
  the channels (1..8) whose sounds are sent, the others are sent silent
  (default: the unmuted channels). Result ``{'seconds', 'bank', 'chain',
  'slots', 'channels'}``.
- ``po32.send`` (same arguments): render again from the current sounds and
  play it; finishes when it has played (``{'sent': True, 'seconds'}``), with
  ``progress`` events on the way. ``po32.cancel`` ends it with status
  ``cancelled``; the stream stopping ends it with an error.
- ``po32.save_wav`` (path, overwrite=False, same arguments): the signal as a
  mono 16-bit 44.1 kHz WAV file (``{'saved', 'exists', 'path', 'seconds'}``).

Import:

- ``po32.listen`` (on=True, device=None): open (or close) the input for the
  level meter. ``device`` is an input device name (default
  ``pref.audio.input_device``, else the system default). Closing it ends a
  recording without decoding it.
- ``po32.record`` (device=None): start recording (the input opens when it is
  closed); the previous decode is dropped. ``po32.stop``: end the recording,
  close the input and decode it (as ``po32.decode``). With
  ``pref.po32.save_recordings`` the recording is also saved as a WAV file in
  ``po32.recordings_folder``.
- ``po32.decode`` (path): decode a WAV file. Result ``{'source', 'drums',
  'patterns', 'card', 'banks'}``. Up to 12 non-empty patterns are picked and
  lettered A.. in order, the first is focused, and the first bank that has
  sounds is selected. A failed decode is an error event (and ``po32.error``).
- ``po32.select_bank`` (bank 0/1), ``po32.focus`` (pattern 1..16),
  ``po32.pick`` (pattern, picked=None (toggle), letter=None): at most 12
  picks; a new pick takes the first free letter; giving a letter another pick
  holds swaps their letters. ``po32.pick_first`` (the first 12, non-empty ones
  first) and ``po32.pick_clear``.
- ``po32.preview`` (on=True): stop the panel's transport and loop the focused
  pattern with the bank's decoded sounds at the panel tempo (velocity 100).
  Starting the transport, changing the bank, the focus or the picks, or
  ``on=False`` ends it; the panel stays stopped.
- ``po32.import``: replace the sounds of channels 1-8 with the bank's sounds
  and all 12 patterns: picked ones land on their letters (steps 1-16 of the
  bank's triggers, no accents or fills, probability 100, velocity 64; a
  pattern shorter than 16 steps grows to 16), the others are emptied. The
  morph endpoints take the sounds (B from the PO-32's morph patches) and the
  morph goes to A. One undo step. Result ``{'drums', 'patterns'}``.

State is read through ``po32.*`` addresses (reported by poll after each
verb); the level, recording length, send progress and preview step change
continuously and come in ``poll()['po32']`` every frame.
"""

import copy
import math
import os
import struct
import threading
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

from pythonic.pattern_manager import DEFAULT_VELOCITY, PatternManager
from pythonic.po32_codec import (PARAM_NAMES, TAG_PATCH, TAG_PATTERN,
                                 TAG_STATE, TAG_TRAILER, ModemEncoder, attack_to_norm,
                                 bit_reverse_bytes, decay_to_norm, default_pattern,
                                 default_right_patch, default_state, freq_to_norm,
                                 generate_fsk_signal, level_to_norm, modamt_to_norm,
                                 modrate_decay_to_norm, modrate_noise_to_norm,
                                 modrate_sine_to_norm, save_wav)
from pythonic.po32_decoder import (SAMPLE_RATE, decode_audio_samples, decode_wav_file,
                                   get_pattern_summary, get_pattern_triggers_for_bank,
                                   get_patch_summary)
from pythonic.synthesizer import PythonicSynthesizer

from .audio import PLAYING, DONE, Player
from .presets import _with_suffix
from .registry import Address
from .undo import CHANNELS, MORPH, PATTERNS

NUM_CHANNELS = 8
PATTERN_NAMES = tuple(PatternManager.PATTERN_NAMES)
MAX_PICKS = len(PATTERN_NAMES)
PO32_STEPS = 16
TRANSFER_PEAK = 0.85      # the signal's peak when sent (reliable reception)
INPUT_BLOCK = 2048
MAX_RECORD_SECONDS = 120.0
PREVIEW_VELOCITY = 100
PROGRESS_STEP = 0.05
SILENT_PATCH = b'\x00\x80' * len(PARAM_NAMES)
RECORDINGS_FOLDER = ('Documents', 'Pythonic Debug Recordings')
TRANSFER_STATES = ('none', 'ready', 'sending', 'sent', 'stopped', 'error')
DECODE_STATES = ('none', 'decoding', 'decoded', 'error')

_ENVELOPE_LINEAR = 1
_MOD_MODES = ('Decay', 'Sine', 'Noise')


# ====================================================================== encoding
def sound_to_normalized(params):
    """The 21 PO-32 parameters (0..1) of a channel sound (``get_parameters``)."""
    norm = {}
    norm['OscWave'] = params['osc_waveform'] / 2.0
    norm['ModMode'] = params['pitch_mod_mode'] / 2.0
    norm['NFilMod'] = params['noise_filter_mode'] / 2.0
    norm['NEnvMod'] = params['noise_envelope_mode'] / 2.0
    norm['OscFreq'] = freq_to_norm(params['osc_frequency'])
    norm['NFilFrq'] = freq_to_norm(params['noise_filter_freq'])
    norm['EQFreq'] = freq_to_norm(params['eq_frequency'])
    norm['OscAtk'] = attack_to_norm(params['osc_attack'])
    linear = params['noise_envelope_mode'] == _ENVELOPE_LINEAR
    scale = 1.5 if linear else 1.0  # the receiver's linear envelope runs 1.5x faster
    norm['NEnvAtk'] = attack_to_norm(params['noise_attack'] * scale)
    norm['OscDcy'] = decay_to_norm(params['osc_decay'])
    norm['NEnvDcy'] = decay_to_norm(params['noise_decay'] * scale)
    mode = _MOD_MODES[params['pitch_mod_mode']] if 0 <= params['pitch_mod_mode'] < 3 else 'Decay'
    rate = params['pitch_mod_rate']
    if mode == 'Sine':
        norm['ModRate'] = modrate_sine_to_norm(rate)
    elif mode == 'Noise':
        norm['ModRate'] = modrate_noise_to_norm(rate)
    else:
        norm['ModRate'] = modrate_decay_to_norm(rate)
    norm['ModAmt'] = modamt_to_norm(params['pitch_mod_amount'], mode)
    q = params['noise_filter_q']
    norm['NFilQ'] = 0.0 if q <= 0.1 else max(0.0, min(1.0, math.log(q / 0.1)
                                                       / math.log(10001.0 / 0.1)))
    norm['Mix'] = 1.0 - params['osc_noise_mix']  # the PO-32 mix is the noise share
    norm['DistAmt'] = params['distortion']
    norm['EQGain'] = max(0.0, min(1.0, (params['eq_gain_db'] + 40) / 80))
    norm['Level'] = level_to_norm(params['level_db'])
    norm['OscVel'] = params['osc_vel_sensitivity']
    norm['NVel'] = params['noise_vel_sensitivity']
    norm['ModVel'] = params['mod_vel_sensitivity']
    return norm


def serialize_sound(params):
    """A sound as the 42 bytes of a PO-32 patch (21 uint16, little-endian)."""
    norm = sound_to_normalized(params)
    return b''.join(struct.pack('<H', min(int(round(norm.get(name, 0.0) * 65536)), 65535))
                    for name in PARAM_NAMES)


def transfer_packet(sounds, bank=0, slots=2, silent=None):
    """The modem packet of a transfer: 8 sounds (silent ones as the silent
    patch) with a default morph patch each, ``slots`` empty patterns, the
    default state and the trailer."""
    encoder = ModemEncoder()
    for index, params in enumerate(sounds[:NUM_CHANNELS]):
        drum = index + bank * NUM_CHANNELS
        patch = SILENT_PATCH if silent and silent[index] else serialize_sound(params)
        encoder.append(TAG_PATCH, bytes([0x10 | drum]) + patch)
        encoder.append(TAG_PATCH, bytes([0x20 | drum]) + default_right_patch(drum))
    for slot in range(slots):
        encoder.append(TAG_PATTERN, default_pattern(slot))
    encoder.append(TAG_STATE, default_state())
    encoder.append(TAG_TRAILER, b'')
    return encoder.get_packet()


def transfer_signal(sounds, bank=0, slots=2, silent=None):
    """The FSK signal of a transfer: float64 mono at 44.1 kHz."""
    return generate_fsk_signal(bit_reverse_bytes(transfer_packet(sounds, bank, slots, silent)))


def chain_options(pm):
    """(label, pattern indexes) of each chain group of the patterns, in order."""
    options = []
    i = 0
    while i < len(PATTERN_NAMES):
        start = i
        while i < len(PATTERN_NAMES) - 1 and pm.patterns[i].chained_to_next:
            i += 1
        label = (PATTERN_NAMES[start] if start == i
                 else f'{PATTERN_NAMES[start]} - {PATTERN_NAMES[i]}')
        options.append((label, list(range(start, i + 1))))
        i += 1
    return options


def resample(signal, rate_in, rate_out):
    """Linear resampling of a mono signal (the modem tones stay well below
    the Nyquist frequency of every output rate)."""
    if rate_in == rate_out or not len(signal):
        return np.asarray(signal, dtype=np.float32)
    count = int(round(len(signal) * rate_out / rate_in))
    x = np.arange(count) * (rate_in / rate_out)
    return np.interp(x, np.arange(len(signal)), signal).astype(np.float32)


def render_preview(sounds, triggers, bpm, rate, mono=False):
    """One seamless loop of a 16-step pattern: (frames, 2) float32 at ``rate``.
    Two passes are rendered and the second kept, so the sounds ringing over
    the loop point are in it."""
    synth = PythonicSynthesizer(rate, parallel_channel_processing=False)
    try:
        for channel, params in zip(synth.channels, sounds):
            if params:
                channel.set_parameters(copy.deepcopy(params))
        synth.set_bpm(bpm)
        synth.set_mono(mono)
        step = max(1, int(60.0 / bpm / 4.0 * rate))
        loop = step * PO32_STEPS
        out = np.empty((loop * 2, 2), dtype=np.float32)
        pos = 0
        for index in range(PO32_STEPS * 2):
            events = [(0, d, PREVIEW_VELOCITY) for d in range(NUM_CHANNELS)
                      if triggers[d][index % PO32_STEPS]]
            end = pos + step
            while pos < end:
                n = min(1024, end - pos)
                out[pos:pos + n] = synth.process_audio_events(n, events)
                events = ()
                pos += n
        return np.ascontiguousarray(out[loop:]), step
    finally:
        synth.cleanup()


# ====================================================================== the module
class Po32:
    """PO-32 transfer and import of an AppCore. State changes on the action
    thread only; the stream callbacks write plain attributes."""

    def __init__(self, core):
        self._core = core
        self._lock = threading.Lock()  # the progress and settle bookkeeping of collect
        # Transfer
        self._send = None             # {'id', 'player', 'seconds', 'last'}
        self._transfer = 'none'
        self._seconds = 0.0
        # Input
        self._input = None
        self._input_device = None
        self._recording = False
        self._record_buffer = None
        self._record_pos = 0
        self._record_full = False
        self._full_handled = False
        self.level = 0.0
        # Decode and picks
        self._decoded = None
        self._source = None
        self._decode_state = 'none'
        self._bank = 0
        self._focus = None
        self._picks = {}              # decoded pattern index -> letter
        self._imported = False
        # Preview
        self._preview = None
        self._preview_step_frames = 1
        self._settle_scheduled = False
        self._error = None

    # ================================================================== interface
    def register(self, registry):
        reg = registry.register
        prefs = self._core.preferences

        def ro(name, get, kind, **kwargs):
            reg(Address(name, get=get, kind=kind, **kwargs))

        ro('po32.chain_options', lambda: [label for label, _ in
                                          chain_options(self._core.pattern_manager)], 'list')
        ro('po32.transfer', lambda: self._transfer, 'enum', labels=TRANSFER_STATES)
        ro('po32.transfer_seconds', lambda: self._seconds, 'float', unit='s')
        ro('po32.progress', self.progress, 'float', minimum=0.0, maximum=1.0)
        ro('po32.listening', lambda: self._input is not None, 'bool')
        ro('po32.recording', lambda: self._recording, 'bool')
        ro('po32.input', lambda: self._input_device if self._input is not None else None,
           'str')
        ro('po32.level', lambda: float(self.level), 'float', minimum=0.0, maximum=1.0)
        ro('po32.recorded_seconds', self.recorded_seconds, 'float', unit='s')
        ro('po32.decode', lambda: self._decode_state, 'enum', labels=DECODE_STATES)
        ro('po32.decoded', self._summary, 'json')
        ro('po32.banks', self._banks, 'list')
        ro('po32.bank', lambda: self._bank, 'int', minimum=0, maximum=1)
        ro('po32.sounds', self._sounds, 'list')
        ro('po32.patterns', self._patterns, 'list')
        ro('po32.picks', self._pick_list, 'list')
        ro('po32.focus', lambda: 0 if self._focus is None else self._focus + 1, 'int',
           minimum=0, maximum=16)
        ro('po32.grid', self._grid, 'list')
        ro('po32.previewing', lambda: self._preview is not None, 'bool')
        ro('po32.preview_step', self.preview_step, 'int')
        ro('po32.imported', lambda: self._imported, 'bool')
        ro('po32.error', lambda: self._error, 'str')
        ro('po32.recordings_folder', recordings_folder, 'str')
        reg(Address('pref.po32.save_recordings',
                    get=lambda: bool(prefs.get('po32_debug_save_recordings', False)),
                    set=self._set_save_recordings, kind='bool', default=False,
                    queued=False, undoable=False))

    def _set_save_recordings(self, value):
        if not self._core.preferences.set('po32_debug_save_recordings', bool(value)):
            raise OSError('the preferences file could not be written '
                          '(po32_debug_save_recordings)')

    def verbs(self):
        return {
            'po32.prepare': self._verb_prepare,
            'po32.send': self._verb_send,
            'po32.cancel': self._verb_cancel,
            'po32.save_wav': self._verb_save_wav,
            'po32.listen': self._verb_listen,
            'po32.record': self._verb_record,
            'po32.stop': self._verb_stop,
            'po32.decode': self._verb_decode,
            'po32.select_bank': self._verb_select_bank,
            'po32.focus': self._verb_focus,
            'po32.pick': self._verb_pick,
            'po32.pick_first': self._verb_pick_first,
            'po32.pick_clear': self._verb_pick_clear,
            'po32.preview': self._verb_preview,
            'po32.import': self._verb_import,
        }

    STATE_ADDRESSES = ('po32.transfer', 'po32.transfer_seconds', 'po32.listening',
                       'po32.recording', 'po32.input', 'po32.decode', 'po32.decoded',
                       'po32.banks', 'po32.bank', 'po32.sounds', 'po32.patterns',
                       'po32.picks', 'po32.focus', 'po32.grid', 'po32.previewing',
                       'po32.imported', 'po32.error')

    def _note(self):
        self._core.note_changes(self.STATE_ADDRESSES)

    # ================================================================== readouts
    def progress(self):
        send = self._send
        if send is not None:
            return float(send['player'].progress)
        return 1.0 if self._transfer == 'sent' else 0.0

    def recorded_seconds(self):
        return self._record_pos / SAMPLE_RATE if self._record_buffer is not None else 0.0

    def preview_step(self):
        player = self._preview
        if player is None:
            return -1
        return int(player.pos // self._preview_step_frames) % PO32_STEPS

    def readout(self):
        """The values poll reports every frame."""
        return {'level': float(self.level), 'recorded_seconds': self.recorded_seconds(),
                'progress': self.progress(), 'preview_step': self.preview_step()}

    def collect(self):
        """Report what the stream callbacks did: send progress, the end of a
        send or a preview, a full recording. Called by poll and the action
        thread's monitor; the bookkeeping is finished on the action thread."""
        core = self._core
        settle = False
        with self._lock:
            send = self._send
            if send is not None:
                player = send['player']
                if player.state == PLAYING:
                    progress = player.progress
                    if progress - send['last'] >= PROGRESS_STEP:
                        send['last'] = progress
                        core._post({'id': send['id'], 'verb': 'po32.send',
                                    'status': 'progress', 'progress': progress})
                else:
                    settle = True
            preview = self._preview
            if preview is not None and preview.state != PLAYING:
                settle = True
            if self._record_full and not self._full_handled:
                self._full_handled = True
                settle = True
            if settle and not self._settle_scheduled:
                self._settle_scheduled = True
            else:
                settle = False
        if settle:
            core.call_soon(self._settle)

    def _settle(self):
        """Action thread: finish a send or a preview that ended, decode a
        recording whose buffer is full."""
        with self._lock:
            self._settle_scheduled = False
            send = self._send
            ended = send is not None and send['player'].state != PLAYING
            if ended:
                self._send = None
        if ended:
            player = send['player']
            event = {'id': send['id'], 'verb': 'po32.send'}
            if player.state == DONE:
                self._transfer = 'sent'
                core_post = {'status': 'done', 'result': {'sent': True,
                                                          'seconds': send['seconds']}}
            elif send.get('cancelled'):
                self._transfer = 'stopped'
                core_post = {'status': 'cancelled', 'result': {'sent': False}}
            else:
                self._transfer = 'error'
                self._error = 'the transfer stopped: the audio stream stopped'
                core_post = {'status': 'error', 'error': self._error}
            event.update(core_post)
            self._core._post(event)
        preview = self._preview
        if preview is not None and preview.state != PLAYING:
            self._preview = None
        if self._record_full and self._recording:
            try:
                self._verb_stop()
            except Exception as exc:  # reported through po32.error
                self._core._report_error(f'PO-32 decode: {exc}', source='po32')
        self._note()

    def close(self):
        """Core shutdown: close the input; the players end with the stream."""
        self._recording = False
        self._close_input()

    # ================================================================== transfer
    def _transfer_args(self, bank, chain, channels):
        core = self._core
        if bank not in (0, 1):
            raise ValueError(f'not a PO-32 bank: {bank!r} (0: sounds 1-8, 1: sounds 9-16)')
        options = chain_options(core.pattern_manager)
        if chain is None:
            label, indexes = options[0]
        elif isinstance(chain, str) and any(chain == label for label, _ in options):
            label, indexes = next(o for o in options if o[0] == chain)
        else:
            items = [chain] if isinstance(chain, (str, int)) else list(chain)
            indexes = []
            for item in items:
                if isinstance(item, str) and item.upper() in PATTERN_NAMES:
                    indexes.append(PATTERN_NAMES.index(item.upper()))
                elif isinstance(item, int) and not isinstance(item, bool) \
                        and 0 <= item < len(PATTERN_NAMES):
                    indexes.append(item)
                else:
                    raise ValueError(f'not a pattern chain: {chain!r}')
            if not indexes:
                raise ValueError('the pattern chain is empty')
            label = ' - '.join(PATTERN_NAMES[i] for i in (indexes[0], indexes[-1])) \
                if len(indexes) > 1 else PATTERN_NAMES[indexes[0]]

        def take():
            synth = core.synth
            return ([ch.get_parameters() for ch in synth.channels],
                    [bool(ch.muted) for ch in synth.channels])
        sounds, mutes = core.at_block_start(take)
        if channels is None:
            channels = [i + 1 for i, muted in enumerate(mutes) if not muted]
        channels = sorted({int(c) for c in channels})
        if any(not 1 <= c <= NUM_CHANNELS for c in channels):
            raise ValueError(f'not channels 1-{NUM_CHANNELS}: {channels}')
        silent = [i + 1 not in channels for i in range(NUM_CHANNELS)]
        info = {'bank': bank, 'chain': label, 'slots': list(range(1, len(indexes) + 1)),
                'channels': channels}
        return sounds, silent, len(indexes), info

    def _signal(self, bank, chain, channels):
        sounds, silent, slots, info = self._transfer_args(bank, chain, channels)
        signal = transfer_signal(sounds, bank, slots, silent)
        self._seconds = len(signal) / SAMPLE_RATE
        info['seconds'] = self._seconds
        return signal, info

    def _verb_prepare(self, bank=0, chain=None, channels=None):
        try:
            signal, info = self._signal(bank, chain, channels)
        except Exception as exc:
            self._fail(exc, transfer=True)
            raise
        if self._send is None:
            self._transfer = 'ready'
        self._error = None
        self._note()
        return info

    def _verb_send(self, bank=0, chain=None, channels=None):
        core = self._core
        audio = core.audio
        try:
            if not audio.live:
                raise RuntimeError('the audio output is not running '
                                   '(save the transfer as a WAV file instead)')
            signal, info = self._signal(bank, chain, channels)
        except Exception as exc:
            self._fail(exc, transfer=True)
            raise
        self._end_preview()
        self._cancel_send()
        peak = float(np.max(np.abs(signal))) if len(signal) else 0.0
        if peak > 0:
            signal = signal * (TRANSFER_PEAK / peak)
        player = Player(resample(signal, SAMPLE_RATE, audio.out_rate), replace=True)
        send = {'id': core._running_action, 'player': player, 'seconds': info['seconds'],
                'last': 0.0}
        with self._lock:
            self._send = send
        self._transfer = 'sending'
        self._error = None
        core._post({'id': send['id'], 'verb': 'po32.send', 'status': 'progress',
                    'progress': 0.0})
        audio.play(player)
        self._note()
        return core.DEFERRED

    def _cancel_send(self):
        with self._lock:
            send = self._send
        if send is None:
            return False
        send['cancelled'] = True
        send['player'].stop()
        self._settle()
        return True

    def _verb_cancel(self):
        cancelled = self._cancel_send()
        return {'cancelled': cancelled}

    def _verb_save_wav(self, path, overwrite=False, bank=0, chain=None, channels=None):
        path = _with_suffix(self._core.presets.resolve(path), '.wav')
        exists = os.path.exists(path)
        if exists and not overwrite:
            return {'saved': False, 'exists': True, 'path': path}
        signal, info = self._signal(bank, chain, channels)
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        save_wav(signal, path, SAMPLE_RATE)
        return {'saved': True, 'exists': exists, 'path': path, 'seconds': info['seconds']}

    # ================================================================== input
    def _input_callback(self, indata, frames, time_info, status):
        """The input stream's callback (the backend's input thread)."""
        if not frames:
            return
        column = indata[:, 0]
        high = float(column.max())
        low = float(column.min())
        self.level = high if high > -low else -low
        buffer = self._record_buffer
        if buffer is not None and self._recording:
            pos = self._record_pos
            n = min(frames, len(buffer) - pos)
            if n > 0:
                buffer[pos:pos + n] = column[:n]
                self._record_pos = pos + n
            if n < frames:
                self._record_full = True

    def _open_input(self, device):
        if self._input is not None:
            if device is None or device == self._input_device:
                return
            self._close_input()
        if device is None:
            device = self._core.preferences.get('audio_input_device')
            if device and device not in self._core.audio.input_devices():
                device = None  # the saved device is gone: the system default
        self._input, self._input_device = self._core.audio.open_input(
            device, SAMPLE_RATE, INPUT_BLOCK, self._input_callback)

    def _close_input(self):
        stream, self._input = self._input, None
        self.level = 0.0
        if stream is not None:
            error = self._core.audio.close_input(stream)
            if error:
                self._core._report_error(error, source='po32')

    def _verb_listen(self, on=True, device=None):
        try:
            if on:
                self._open_input(device)
            else:
                self._recording = False
                self._record_buffer = None
                self._close_input()
        except Exception as exc:
            self._fail(exc)
            raise
        self._error = None
        self._note()
        return {'listening': self._input is not None,
                'device': self._input_device if self._input is not None else None}

    def _verb_record(self, device=None):
        try:
            self._open_input(device)
        except Exception as exc:
            self._fail(exc)
            raise
        self._end_preview()
        self._forget_decode()
        self._recording = False
        self._record_pos = 0
        self._record_full = self._full_handled = False
        self._record_buffer = np.empty(int(MAX_RECORD_SECONDS * SAMPLE_RATE), dtype=np.float32)
        self._recording = True
        self._error = None
        self._note()
        return {'recording': True, 'device': self._input_device}

    def _verb_stop(self):
        """End the recording, close the input and decode what was recorded."""
        if not self._recording and self._record_buffer is None:
            raise RuntimeError('not recording')
        self._recording = False
        self._close_input()
        buffer, count = self._record_buffer, self._record_pos
        self._record_buffer = None
        self._record_pos = 0
        if buffer is None or count == 0:
            self._decode_state = 'none'
            self._note()
            return {'decoded': False, 'seconds': 0.0}
        samples = buffer[:count].astype(np.float64)
        source = 'recorded audio'
        if self._core.preferences.get('po32_debug_save_recordings', False):
            try:
                saved = save_recording(samples, SAMPLE_RATE)
            except OSError as exc:
                self._core._report_error(f'PO-32 recording not saved: {exc}', source='po32')
            else:
                source = f'recorded audio ({os.path.basename(saved)})'
        return self._decode(lambda: decode_audio_samples(samples, SAMPLE_RATE), source)

    def _verb_decode(self, path):
        path = self._core.presets.resolve(path)
        self._end_preview()
        return self._decode(lambda: decode_wav_file(path), os.path.basename(path))

    # ================================================================== decode
    def _forget_decode(self):
        self._decoded = None
        self._source = None
        self._decode_state = 'none'
        self._picks = {}
        self._focus = None
        self._imported = False

    def _decode(self, decode, source):
        self._forget_decode()
        self._decode_state = 'decoding'
        self._note()
        try:
            preset = decode()
        except Exception as exc:
            preset = None
            error = f'{type(exc).__name__}: {exc}'
        else:
            error = preset.error
        if error is None and not preset.left_patches and not preset.decoded_patterns:
            error = 'no sounds or patterns in the signal'
        if error is not None:
            self._decode_state = 'error'
            self._error = f'Failed to decode PO-32 data: {error}'
            self._note()
            raise RuntimeError(self._error)
        self._decoded = preset
        self._source = source
        self._decode_state = 'decoded'
        self._error = None
        banks = self._banks()
        self._bank = 0 if banks[0] or not banks[1] else 1
        patterns = preset.decoded_patterns
        non_empty = [i for i, dp in enumerate(patterns) if any(dp.triggers)]
        picked = non_empty[:MAX_PICKS] or list(range(min(MAX_PICKS, len(patterns))))
        self._assign(picked)
        self._focus = picked[0] if picked else None
        self._note()
        return self._summary()

    def _summary(self):
        preset = self._decoded
        if preset is None:
            return None
        drums = len(preset.left_patches)
        patterns = len(preset.decoded_patterns)
        return {'source': self._source, 'drums': drums, 'patterns': patterns,
                'card': drums > NUM_CHANNELS or patterns > 2, 'banks': self._banks()}

    def _banks(self):
        preset = self._decoded
        if preset is None:
            return [False, False]
        drums = preset.left_patches.keys()
        return [any(d < NUM_CHANNELS for d in drums), any(d >= NUM_CHANNELS for d in drums)]

    def _sounds(self):
        """Summaries of the bank's 8 decoded sounds (None where there is none)."""
        preset = self._decoded
        if preset is None:
            return [None] * NUM_CHANNELS
        offset = self._bank * NUM_CHANNELS
        sounds = []
        for d in range(NUM_CHANNELS):
            patch = preset.left_patches.get(d + offset)
            sounds.append(get_patch_summary(patch) if patch is not None else None)
        return sounds

    def _patterns(self):
        preset = self._decoded
        if preset is None:
            return []
        return [{'number': i + 1, 'empty': not any(dp.triggers),
                 'summary': get_pattern_summary(dp)}
                for i, dp in enumerate(preset.decoded_patterns)]

    def _pick_list(self):
        return [{'pattern': i + 1, 'letter': letter} for i, letter in sorted(self._picks.items())]

    def _triggers(self, index):
        """8 lists of 16 bools: the pattern's triggers of the selected bank."""
        dp = self._decoded.decoded_patterns[index]
        triggers = get_pattern_triggers_for_bank(dp, self._bank)
        grid = []
        for d in range(NUM_CHANNELS):
            steps = list(triggers.get(d, ()))[:PO32_STEPS]
            grid.append(steps + [False] * (PO32_STEPS - len(steps)))
        return grid

    def _grid(self):
        if self._decoded is None or self._focus is None:
            return []
        return self._triggers(self._focus)

    # ================================================================== bank and picks
    def _require_decoded(self):
        if self._decoded is None:
            raise RuntimeError('nothing decoded yet: record or import a PO-32 transfer first')
        return self._decoded

    def _pattern(self, pattern):
        preset = self._require_decoded()
        count = len(preset.decoded_patterns)
        if not (isinstance(pattern, int) and not isinstance(pattern, bool)
                and 1 <= pattern <= count):
            raise ValueError(f'not a decoded pattern: {pattern!r} (1-{count})')
        return pattern - 1

    def _assign(self, indexes):
        """Pick these patterns, lettered A.. in pattern order."""
        self._picks = {index: PATTERN_NAMES[n] for n, index in enumerate(sorted(indexes))
                       if n < MAX_PICKS}

    def _verb_select_bank(self, bank):
        if bank not in (0, 1):
            raise ValueError(f'not a PO-32 bank: {bank!r} (0 or 1)')
        if self._decoded is not None and not self._banks()[bank]:
            raise ValueError(f'bank {bank} has no decoded sounds')
        if bank != self._bank:
            self._end_preview()
        self._bank = bank
        self._note()
        return {'bank': bank}

    def _verb_focus(self, pattern):
        index = self._pattern(pattern)
        if index != self._focus:
            self._end_preview()
        self._focus = index
        self._note()
        return {'focus': pattern}

    def _verb_pick(self, pattern, picked=None, letter=None):
        index = self._pattern(pattern)
        picks = dict(self._picks)
        if letter is not None:
            letter = str(letter).upper()
            if letter not in PATTERN_NAMES:
                raise ValueError(f'not a pattern letter: {letter!r} (A-L)')
            picked = True
        if picked is None:
            picked = index not in picks
        if picked and index not in picks:
            if len(picks) >= MAX_PICKS:
                raise ValueError(f'{MAX_PICKS} patterns are picked already')
            used = set(picks.values())
            picks[index] = next(name for name in PATTERN_NAMES if name not in used)
            self._end_preview()
        elif not picked and index in picks:
            del picks[index]
            self._end_preview()
        if letter is not None:
            old = picks[index]
            for other, held in picks.items():
                if other != index and held == letter:
                    picks[other] = old  # a conflict swaps the letters
                    break
            picks[index] = letter
        self._picks = picks
        if index != self._focus:
            self._end_preview()
        self._focus = index
        self._note()
        return {'pattern': pattern, 'picked': index in picks, 'letter': picks.get(index)}

    def _verb_pick_first(self):
        preset = self._require_decoded()
        patterns = preset.decoded_patterns
        chosen = [i for i, dp in enumerate(patterns) if any(dp.triggers)][:MAX_PICKS]
        chosen += [i for i in range(len(patterns)) if i not in chosen][:MAX_PICKS - len(chosen)]
        self._assign(chosen)
        self._end_preview()
        self._note()
        return {'picks': self._pick_list()}

    def _verb_pick_clear(self):
        self._picks = {}
        self._end_preview()
        self._note()
        return {'picks': []}

    # ================================================================== preview
    def _bank_sounds(self, right=False):
        preset = self._decoded
        patches = preset.right_patches if right else preset.left_patches
        offset = self._bank * NUM_CHANNELS
        sounds = []
        for d in range(NUM_CHANNELS):
            patch = patches.get(d + offset)
            sounds.append(patch.synth_params if patch is not None and patch.synth_params
                          else None)
        return sounds

    def _end_preview(self):
        player, self._preview = self._preview, None
        if player is not None:
            player.stop()
            return True
        return False

    def _verb_preview(self, on=True):
        core = self._core
        if not on:
            stopped = self._end_preview()
            self._note()
            return {'previewing': False, 'stopped': stopped}
        self._require_decoded()
        if self._focus is None:
            raise RuntimeError('no pattern to preview')
        if self._send is not None:
            raise RuntimeError('a transfer is being sent')
        audio = core.audio
        if not audio.live:
            raise RuntimeError('the audio output is not running')
        self._end_preview()
        buffer, step = render_preview(self._bank_sounds(), self._triggers(self._focus),
                                      float(core.pattern_manager.bpm), audio.out_rate,
                                      bool(core.synth.mono))
        player = Player(buffer, loop=True, stop_on_play=True)
        core.patterns.stop()  # queued first: the transport is stopped when the preview starts
        self._preview_step_frames = step
        self._preview = player
        audio.play(player)
        core.at_block_start(lambda: None)  # stopped and playing when the verb ends
        self._note()
        return {'previewing': True, 'pattern': self._focus + 1}

    # ================================================================== import
    def _verb_import(self):
        core = self._core
        preset = self._require_decoded()
        self._end_preview()
        bank = self._bank
        core.at_block_start(lambda: None)  # queued sets land first
        current = core.undo.capture((CHANNELS, PATTERNS))
        channels = current[CHANNELS]
        drums = 0
        for d, params in enumerate(self._bank_sounds()):
            if params:
                channels[d].update(copy.deepcopy(params))
                drums += 1
        endpoint_b = copy.deepcopy(channels)
        for d, params in enumerate(self._bank_sounds(right=True)):
            if params:
                endpoint_b[d].update(copy.deepcopy(params))
        morph = {'position': 0.0, 'endpoint_a': copy.deepcopy(channels),
                 'endpoint_b': endpoint_b}
        patterns = current[PATTERNS]
        for pattern in patterns:
            pattern.clear()
        imported = []
        for index, letter in sorted(self._picks.items()):
            if index >= len(preset.decoded_patterns):
                continue
            target = patterns[PATTERN_NAMES.index(letter)]
            if target.length < PO32_STEPS:
                target.set_pattern_length(PO32_STEPS)
            triggers = self._triggers(index)
            for d in range(NUM_CHANNELS):
                steps = target.channels[d].steps
                for s in range(PO32_STEPS):
                    step = steps[s]
                    step.trigger = triggers[d][s]
                    step.accent = False
                    step.fill = False
                    step.probability = 100
                    step.substeps = ''
                    step.velocity = DEFAULT_VELOCITY
            imported.append({'pattern': index + 1, 'letter': letter})
        state = {CHANNELS: channels, MORPH: morph, PATTERNS: patterns}
        core.presets._install(state, None, 'PO-32 import', (CHANNELS, MORPH, PATTERNS))
        self._imported = True
        self._note()
        return {'drums': drums, 'patterns': imported}

    # ================================================================== errors
    def _fail(self, exc, transfer=False):
        self._error = str(exc) or type(exc).__name__
        if transfer and self._send is None:
            self._transfer = 'error'
        self._note()


# ====================================================================== recordings
def recordings_folder():
    """Where recordings are saved with ``pref.po32.save_recordings`` (in the
    Documents folder: Store installs of Python virtualise AppData writes)."""
    return os.path.join(Path.home(), *RECORDINGS_FOLDER)


def save_recording(samples, rate):
    """Save a recording as a mono 16-bit WAV file; returns its path."""
    folder = recordings_folder()
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f'po32_recording_{datetime.now():%Y%m%d_%H%M%S}.wav')
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype('<i2')
    with wave.open(path, 'wb') as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(int(rate))
        f.writeframes(pcm.tobytes())
    return path
