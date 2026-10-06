"""
Pattern export: the MIDI file of a pattern.

One note per triggered step on the General MIDI drum channel, at the step's
velocity (127 when accented), placed as the sequencer plays it: the step rate
sets the step length and swing moves the second sixteenth of each eighth-note
pair the same way. Patterns run 1 to 64 steps, and the file ends at the end
of the pattern so it loops cleanly. Probability, fills and substeps are not
written (one note per step, as before).

Rendering patterns to WAV runs the sequencer itself, so it needs nothing here.
"""

from typing import List, Tuple

from pythonic.sequencer import PAIR_TICKS, STEP_TICKS, TICKS_PER_QUARTER

GM_DRUM_NOTES = (36, 38, 42, 46, 45, 41, 39, 37)  # kick, snare, CHH, OHH, tom H, tom L, clap, rim
MIDI_DRUM_CHANNEL = 9  # channel 10
TICKS_PER_BEAT = 480
NOTE_TICKS = 10


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


def pattern_midi_file(pattern, bpm: float, step_rate: str = '1/16', swing: float = 0.0):
    """A mido.MidiFile (one track) of a pattern."""
    import mido

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
