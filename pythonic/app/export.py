"""
Export of the app core (inventory cluster O): pattern MIDI files, pattern
renders and drum renders as WAV files.

Verbs (results and errors arrive through ``poll`` as action events):

- ``export.midi`` (path, pattern=None, overwrite=False): the MIDI file of a
  pattern (default the selected one). One note per triggered step on the
  General MIDI drum channel, at the step's velocity (127 when accented),
  placed as the sequencer plays it: the step rate sets the step length and
  swing moves the second sixteenth of each eighth-note pair the same way.
  Patterns run 1 to 64 steps, and the file ends at the end of the pattern so
  it loops cleanly. Probability, fills and substeps are not written.
  Result ``{'saved', 'exists', 'path', 'pattern'}``.
- ``export.wav`` (path, pattern=None, tail='cut', overwrite=False): one pass
  of a pattern rendered at the synth rate (mono when the output is mono),
  16-bit. ``tail``: ``cut`` ends with the pattern, ``append`` adds 2 s for
  the sounds to ring out, ``loop`` plays the pattern a second time. The
  pattern plays alone (no chain). Result ``{'saved', 'exists', 'path',
  'pattern', 'tail', 'frames', 'sample_rate', 'channels'}``.
- ``export.drum_wav`` (path, channel=None, overwrite=False): 2 s of one
  channel (1..8, default the selected one) hit at velocity 127, peak-limited
  to 0.99. Result ``{'saved', 'exists', 'path', 'channel', 'frames',
  'sample_rate', 'channels'}``.
- ``export.drum_wavs`` (folder, overwrite=False): the same for the eight
  channels, as ``01_<name>.wav`` .. ``08_<name>.wav`` in a folder. Result
  ``{'saved', 'exists', 'folder', 'paths'}``.

Paths are absolute, or names relative to the preset folder; ``.mid`` /
``.wav`` is added to a path without an extension. A save never replaces an
existing file unless ``overwrite=True``: the result is then ``{'saved':
False, 'exists': True, ...}`` (for ``drum_wavs``, ``paths`` lists the files
that exist) and nothing is written.

The state to render is taken when the verb starts (after the queued sets
have landed). WAV renders then run on the export thread with an offline
synth and pattern manager of their own, so the live stream is never touched;
while they run, poll reports ``{'id', 'verb', 'status': 'progress',
'progress': 0..1}`` events, then the usual done or error event. Renders run
one after the other, in the order they were asked for.
"""

import copy
import math
import os
import queue
import re
import threading
import wave
from typing import List, Tuple

import numpy as np

from pythonic.morph_manager import MorphManager
from pythonic.pattern_manager import PatternManager
from pythonic.sequencer import PAIR_TICKS, STEP_TICKS, StepSequencer, TICKS_PER_QUARTER
from pythonic.synthesizer import PythonicSynthesizer

from .patterns import PATTERN_NAMES, pattern_index
from .presets import _with_suffix
from .undo import CHANNELS, GLOBALS, MORPH, pattern_part

GM_DRUM_NOTES = (36, 38, 42, 46, 45, 41, 39, 37)  # kick, snare, CHH, OHH, tom H, tom L, clap, rim
MIDI_DRUM_CHANNEL = 9  # channel 10
TICKS_PER_BEAT = 480
NOTE_TICKS = 10

TAILS = ('cut', 'append', 'loop')
APPEND_SECONDS = 2.0
DRUM_SECONDS = 2.0
DRUM_VELOCITY = 127
RENDER_CHUNK = 1024
PROGRESS_STEP = 0.05  # a progress event at most every 5 %
NUM_CHANNELS = PythonicSynthesizer.NUM_CHANNELS


# ====================================================================== MIDI
def _clock_tick(pattern_tick: int, swing: float) -> float:
    """Swung clock tick (1920 per quarter) of a straight pattern tick, the
    inverse of the sequencer's swing warp."""
    split = int(min(max(swing, 0.0), 1.0) * 240.0 + 480.0 + 0.5)
    pair, pos = divmod(pattern_tick, PAIR_TICKS)
    if pos < 480:
        swung = pos * split / 480.0
    else:
        swung = split + (pos - 480) * (PAIR_TICKS - split) / 480.0
    return pair * PAIR_TICKS + swung


def pattern_notes(pattern, step_rate: str = '1/16', swing: float = 0.0,
                  ticks_per_beat: int = TICKS_PER_BEAT) -> List[Tuple[int, int, int]]:
    """(MIDI tick, channel 0-7, velocity) of every triggered step, in time order."""
    step_ticks = STEP_TICKS.get(step_rate, 480)
    scale = ticks_per_beat / TICKS_PER_QUARTER
    notes = []
    for index in range(pattern.length):
        tick = int(round(_clock_tick(index * step_ticks, swing) * scale))
        for channel_id, channel in enumerate(pattern.channels):
            step = channel.get_step(index)
            if step.trigger:
                notes.append((tick, channel_id, step.hit_velocity()))
    return notes


def pattern_length_ticks(pattern, step_rate: str = '1/16',
                         ticks_per_beat: int = TICKS_PER_BEAT) -> int:
    return int(round(pattern.length * STEP_TICKS.get(step_rate, 480)
                     * ticks_per_beat / TICKS_PER_QUARTER))


def _import_mido():
    try:
        import mido
    except ImportError:
        raise RuntimeError("MIDI export requires the 'mido' library "
                           "(pip install mido)") from None
    return mido


def pattern_midi_file(pattern, bpm: float, step_rate: str = '1/16', swing: float = 0.0):
    """A mido.MidiFile (one track) of a pattern."""
    mido = _import_mido()

    mid = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm), time=0))

    timed = []  # (absolute tick, order, message); note offs go first on a tick
    for tick, channel_id, velocity in pattern_notes(pattern, step_rate, swing):
        note = GM_DRUM_NOTES[channel_id] if channel_id < len(GM_DRUM_NOTES) else 36
        timed.append((tick, 1, mido.Message('note_on', channel=MIDI_DRUM_CHANNEL, note=note,
                                            velocity=velocity)))
        timed.append((tick + NOTE_TICKS, 0, mido.Message('note_off', channel=MIDI_DRUM_CHANNEL,
                                                         note=note, velocity=0)))
    timed.sort(key=lambda item: (item[0], item[1]))

    now = 0
    for tick, _, message in timed:
        track.append(message.copy(time=tick - now))
        now = tick
    end = max(pattern_length_ticks(pattern, step_rate), now)
    track.append(mido.MetaMessage('end_of_track', time=end - now))
    return mid


# ====================================================================== rendering
def pattern_frames(length, step_rate, bpm, rate):
    """Frames of one pass of a pattern, as the sequencer's clock runs
    (1920 ticks per quarter, bpm * 32 ticks per second)."""
    ticks = length * STEP_TICKS.get(step_rate, 480)
    return int(math.ceil(ticks / (bpm * 32.0 / rate)))


def offline_synth(snapshot):
    """A synth of its own with the captured sounds, globals and morph."""
    state = snapshot['state']
    synth = PythonicSynthesizer(snapshot['rate'], parallel_channel_processing=False)
    for channel, params, muted in zip(synth.channels, state[CHANNELS], snapshot['mutes']):
        channel.set_smoothing_time(snapshot['smoothing'])
        channel.set_parameters(copy.deepcopy(params))
        channel.muted = muted
    bpm, _swing, _step_rate, _fill_rate, master = state[GLOBALS]
    synth.set_master_volume(master)
    synth.set_bpm(bpm)
    synth.set_mono(snapshot['mono'])
    if MORPH in state:
        morph = MorphManager(synth)
        morph.from_dict(state[MORPH])
        morph._learn_mode = snapshot['learn']  # LFO morph offsets move from the same place
        synth.set_morph_manager(morph)
    return synth


def render_pattern(snapshot, tail, progress=None):
    """Render one pass of the captured pattern plus its tail: (frames, 2) floats."""
    state = snapshot['state']
    bpm, swing, step_rate, fill_rate, _master = state[GLOBALS]
    rate = snapshot['rate']
    synth = offline_synth(snapshot)
    pm = PatternManager(num_channels=NUM_CHANNELS, pattern_length=16)
    pattern = snapshot['pattern'].copy()
    pattern.chained_to_next = pattern.chained_from_prev = False  # the pattern alone
    pm.patterns[0] = pattern
    pm.set_bpm(bpm)
    pm.set_swing(swing)
    pm.set_step_rate(step_rate)
    pm.set_fill_rate(fill_rate)
    pm.start_playback(0)
    sequencer = StepSequencer(pm, rate)
    sequencer.start(None, synth_clock=synth.sample_clock)

    one_pass = pattern_frames(pattern.length, step_rate, bpm, rate)
    passes = 2 if tail == 'loop' else 1
    extra = int(round(APPEND_SECONDS * rate)) if tail == 'append' else 0
    total = one_pass * passes + extra
    sequenced = one_pass * passes  # then the sounds ring out without new hits
    out = np.empty((total, 2), dtype=np.float32)
    pos = 0
    try:
        while pos < total:
            n = min(RENDER_CHUNK, total - pos)
            if pos < sequenced:
                n = min(n, sequenced - pos)
                events = sequencer.advance(n)
            else:
                events = ()
            out[pos:pos + n] = synth.process_audio_events(n, events)
            pos += n
            if progress is not None:
                progress(pos / total)
    finally:
        synth.cleanup()
    return out


def render_drum(snapshot, index, seconds=DRUM_SECONDS):
    """One channel hit at velocity 127, ``seconds`` long: (frames, 2) floats,
    peak-limited to 0.99 (the channel alone, without the master)."""
    rate = snapshot['rate']
    synth = PythonicSynthesizer(rate, parallel_channel_processing=False)
    channel = synth.channels[index]
    channel.set_smoothing_time(snapshot['smoothing'])
    channel.set_parameters(copy.deepcopy(snapshot['state'][CHANNELS][index]))
    synth.set_bpm(snapshot['state'][GLOBALS][0])  # tempo-synced LFOs and delays
    total = int(seconds * rate)
    out = np.empty((total, 2), dtype=np.float32)
    channel.trigger(DRUM_VELOCITY)
    pos = 0
    while pos < total:
        n = min(RENDER_CHUNK, total - pos)
        out[pos:pos + n] = channel.process(n)
        pos += n
    synth.cleanup()
    peak = float(np.max(np.abs(out))) if total else 0.0
    if peak > 0.99:
        out *= 0.99 / peak
    return out


def write_wav(path, audio, rate, mono):
    """Write 16-bit PCM in one step (a failed write leaves an old file whole).
    Mono takes the average of both sides. Returns the channel count."""
    if mono:
        audio = ((audio[:, 0] + audio[:, 1]) * 0.5).reshape(-1, 1)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype('<i2')
    channels = pcm.shape[1]
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    temp = f'{path}.tmp-{os.getpid()}'
    try:
        with wave.open(temp, 'wb') as f:
            f.setnchannels(channels)
            f.setsampwidth(2)
            f.setframerate(rate)
            f.writeframes(pcm.tobytes())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.remove(temp)
    return channels


def drum_file_name(number, name):
    """``01_Kick.wav``: the channel number and its name made file-safe."""
    safe = re.sub(r'\s+', '_', re.sub(r'[^\w\s-]', '', name or '')).strip('_')
    return f'{number:02d}_{safe or "Drum"}.wav'


# ====================================================================== the module
class Export:
    """Export verbs of an AppCore; WAV renders run on the export thread."""

    def __init__(self, core):
        self._core = core
        self._jobs = queue.Queue()
        self._thread = None
        self._lock = threading.Lock()

    def verbs(self):
        return {
            'export.midi': self._verb_midi,
            'export.wav': self._verb_wav,
            'export.drum_wav': self._verb_drum_wav,
            'export.drum_wavs': self._verb_drum_wavs,
        }

    def close(self, timeout=5.0):
        """Stop the export thread once the renders asked for have finished
        (or the timeout passes: it is a daemon thread)."""
        with self._lock:
            thread, self._thread = self._thread, None
        if thread is not None:
            self._jobs.put(None)
            thread.join(timeout)

    # ------------------------------------------------------------------ helpers
    def _path(self, path, suffix):
        return _with_suffix(self._core.presets.resolve(path), suffix)

    def _channel_index(self, channel):
        return self._core.presets.channel_index(channel)

    def _snapshot(self, parts):
        """The state to render, taken after the queued sets have landed."""
        core = self._core
        core.at_block_start(lambda: None)
        synth = core.synth
        return {
            'state': core.undo.capture(parts),
            'rate': int(synth.sr),
            'mono': bool(synth.mono),
            'mutes': [bool(ch.muted) for ch in synth.channels],
            'learn': core.morph_manager.get_learn_mode(),
            'smoothing': core.preferences.get('param_smoothing_ms', 30.0),
        }

    def _defer(self, verb, work):
        """Run work(progress) on the export thread; its return value is the
        action's result. Returns AppCore.DEFERRED for the verb handler."""
        core = self._core
        action_id = core._running_action
        with self._lock:
            if self._thread is None:
                self._thread = threading.Thread(target=self._loop, name='pythonic-export',
                                                daemon=True)
                self._thread.start()
        self._jobs.put((action_id, verb, work))
        return core.DEFERRED

    def _loop(self):
        core = self._core
        while True:
            job = self._jobs.get()
            if job is None:
                return
            action_id, verb, work = job
            last = [-1.0]

            def progress(fraction, action_id=action_id, verb=verb, last=last):
                if fraction - last[0] >= PROGRESS_STEP or fraction >= 1.0 > last[0]:
                    last[0] = fraction
                    core._post({'id': action_id, 'verb': verb, 'status': 'progress',
                                'progress': min(1.0, float(fraction))})

            event = {'id': action_id, 'verb': verb}
            try:
                progress(0.0)
                result = work(progress)
            except Exception as exc:
                event.update(status='error', error=str(exc) or type(exc).__name__)
            else:
                event.update(status='done', result=result)
            core._post(event)

    # ------------------------------------------------------------------ verbs
    def _verb_midi(self, path, pattern=None, overwrite=False):
        core = self._core
        index = pattern_index(pattern, core.pattern_manager.selected_pattern_index)
        name = PATTERN_NAMES[index]
        path = self._path(path, '.mid')
        exists = os.path.exists(path)
        if exists and not overwrite:
            return {'saved': False, 'exists': True, 'path': path, 'pattern': name}
        snapshot = self._snapshot((GLOBALS, pattern_part(index)))
        bpm, swing, step_rate, _fill_rate, _master = snapshot['state'][GLOBALS]
        mido_file = pattern_midi_file(snapshot['state'][pattern_part(index)], bpm,
                                      step_rate, swing)
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        temp = f'{path}.tmp-{os.getpid()}'
        try:
            mido_file.save(temp)
            os.replace(temp, path)
        finally:
            if os.path.exists(temp):
                os.remove(temp)
        return {'saved': True, 'exists': exists, 'path': path, 'pattern': name}

    def _verb_wav(self, path, pattern=None, tail='cut', overwrite=False):
        core = self._core
        if tail not in TAILS:
            raise ValueError(f'not a tail: {tail!r} ({", ".join(TAILS)})')
        index = pattern_index(pattern, core.pattern_manager.selected_pattern_index)
        name = PATTERN_NAMES[index]
        path = self._path(path, '.wav')
        exists = os.path.exists(path)
        if exists and not overwrite:
            return {'saved': False, 'exists': True, 'path': path, 'pattern': name, 'tail': tail}
        snapshot = self._snapshot((CHANNELS, GLOBALS, MORPH, pattern_part(index)))
        snapshot['pattern'] = snapshot['state'][pattern_part(index)]

        def work(progress):
            audio = render_pattern(snapshot, tail, progress)
            channels = write_wav(path, audio, snapshot['rate'], snapshot['mono'])
            return {'saved': True, 'exists': exists, 'path': path, 'pattern': name,
                    'tail': tail, 'frames': len(audio), 'sample_rate': snapshot['rate'],
                    'channels': channels}
        return self._defer('export.wav', work)

    def _verb_drum_wav(self, path, channel=None, overwrite=False):
        index = self._channel_index(channel)
        path = self._path(path, '.wav')
        exists = os.path.exists(path)
        if exists and not overwrite:
            return {'saved': False, 'exists': True, 'path': path, 'channel': index + 1}
        snapshot = self._snapshot((CHANNELS, GLOBALS))

        def work(progress):
            audio = render_drum(snapshot, index)
            channels = write_wav(path, audio, snapshot['rate'], snapshot['mono'])
            return {'saved': True, 'exists': exists, 'path': path, 'channel': index + 1,
                    'frames': len(audio), 'sample_rate': snapshot['rate'],
                    'channels': channels}
        return self._defer('export.drum_wav', work)

    def _verb_drum_wavs(self, folder, overwrite=False):
        folder = self._core.presets.resolve(folder)
        snapshot = self._snapshot((CHANNELS, GLOBALS))
        names = [params.get('name', '') for params in snapshot['state'][CHANNELS]]
        paths = [os.path.join(folder, drum_file_name(i + 1, name))
                 for i, name in enumerate(names)]
        existing = [p for p in paths if os.path.exists(p)]
        if existing and not overwrite:
            return {'saved': False, 'exists': True, 'folder': folder, 'paths': existing}

        def work(progress):
            for index, path in enumerate(paths):
                write_wav(path, render_drum(snapshot, index), snapshot['rate'],
                          snapshot['mono'])
                progress((index + 1) / len(paths))
            return {'saved': True, 'exists': bool(existing), 'folder': folder,
                    'paths': paths}
        return self._defer('export.drum_wavs', work)
