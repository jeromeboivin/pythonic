"""
Pattern files and export with per-step velocity and 1-64 step patterns:
JSON presets (pattern dicts) stay readable both ways, .mtpreset files load
with velocity 64 unless they carry velocities, and the MIDI export writes
every step of a long pattern at its velocity.
"""

import json
import pathlib

import mido
import pytest

from pythonic.app.export import GM_DRUM_NOTES, pattern_midi_file, pattern_notes
from pythonic.pattern_manager import Pattern, PatternChannel, PatternManager
from pythonic.preset_manager import PythonicPresetParser

TESTS = pathlib.Path(__file__).resolve().parent
OLD_STEP_KEYS = {'trigger', 'accent', 'fill', 'probability', 'substeps'}


def long_manager():
    pm = PatternManager()
    pattern = pm.patterns[0]
    pattern.set_length(64)
    for index, velocity, accent in ((0, 64, False), (16, 100, False), (40, 5, True), (63, 127, False)):
        step = pattern.channels[1].steps[index]
        step.trigger, step.velocity, step.accent = True, velocity, accent
    return pm


# ---------------------------------------------------------------------------
# JSON presets
# ---------------------------------------------------------------------------

def test_pattern_dicts_add_velocity_beside_the_old_step_keys():
    pm = long_manager()
    data = pm.to_dict()
    steps = data['patterns'][0]['channels'][1]['steps']
    assert data['patterns'][0]['length'] == 64 and len(steps) == 64
    assert set(steps[16]) == OLD_STEP_KEYS | {'velocity'}
    assert steps[16]['velocity'] == 100 and steps[16]['trigger'] is True


def test_old_pattern_dicts_load_with_velocity_64_and_16_steps():
    old = PatternManager().to_dict()
    for pattern in old['patterns']:
        for channel in pattern['channels']:
            for step in channel['steps']:
                del step['velocity']
    old['patterns'][0]['channels'][0]['steps'][2].update(trigger=True, accent=True)
    pm = PatternManager()
    pm.patterns[0].channels[0].steps[2].velocity = 9
    pm.from_dict(json.loads(json.dumps(old)))
    assert pm.patterns[0].length == 16
    assert all(s.velocity == 64 for p in pm.patterns for c in p.channels for s in c.steps)
    assert pm.patterns[0].channels[0].steps[2].accent is True


def test_an_old_reader_reads_new_files():
    """A reader that knows only the old step keys (it reads them by name and
    ignores the rest) gets the triggers of all 64 steps."""
    data = json.loads(json.dumps(long_manager().to_dict()))
    channel = data['patterns'][0]['channels'][1]
    old_steps = [{k: v for k, v in s.items() if k in OLD_STEP_KEYS} for s in channel['steps']]
    triggers = [s['trigger'] for s in old_steps]
    assert [i for i, t in enumerate(triggers) if t] == [0, 16, 40, 63]


def test_json_save_and_load_keep_long_patterns_and_velocities(tmp_path):
    path = tmp_path / 'patterns.json'
    long_manager().save_patterns(str(path))
    pm = PatternManager()
    pm.load_patterns(str(path))
    steps = pm.patterns[0].channels[1].steps
    assert pm.patterns[0].length == 64
    assert [(steps[i].velocity, steps[i].accent) for i in (16, 40, 63)] == \
        [(100, False), (5, True), (127, False)]


def test_velocities_from_a_dict_are_clamped():
    data = {'channel_id': 0, 'steps': [dict(trigger=True, accent=False, fill=False, velocity=v)
                                       for v in (0, 200, 50)]}
    assert [s.velocity for s in PatternChannel.from_dict(data).steps] == [1, 127, 50]


# ---------------------------------------------------------------------------
# .mtpreset
# ---------------------------------------------------------------------------

def load_mtpreset(pm, path):
    parser = PythonicPresetParser()
    data = parser.convert_to_synth_format(parser.parse_file(str(path)))
    pm.load_from_preset_data(data['patterns'])


def test_mtpreset_without_velocities_loads_at_64_over_older_velocities():
    pm = PatternManager()
    for channel in pm.patterns[0].channels:
        for step in channel.steps:
            step.velocity = 3
    load_mtpreset(pm, TESTS / '808.mtpreset')
    assert all(s.velocity == 64 for c in pm.patterns[0].channels for s in c.steps)
    assert any(s.trigger for c in pm.patterns[0].channels for s in c.steps)


def test_mtpreset_velocities_are_read_when_present(tmp_path):
    text = (TESTS / '808.mtpreset').read_text()
    marker = 'Fills:    "----------------"'
    assert marker in text
    text = text.replace(marker, marker + '\n\t\t\t\tVelocities: "10,20,300"', 1)
    path = tmp_path / 'vel.mtpreset'
    path.write_text(text)
    pm = PatternManager()
    load_mtpreset(pm, path)
    velocities = pm.patterns[0].channels[0].get_velocities()
    assert velocities[:4] == [10, 20, 127, 64]


# ---------------------------------------------------------------------------
# MIDI export
# ---------------------------------------------------------------------------

def absolute(track):
    now, out = 0, []
    for message in track:
        now += message.time
        out.append((now, message))
    return out


def test_midi_export_writes_every_step_of_a_long_pattern_at_its_velocity(tmp_path):
    pm = long_manager()
    path = tmp_path / 'a.mid'
    pattern_midi_file(pm.patterns[0], 120, '1/16').save(str(path))
    mid = mido.MidiFile(str(path))
    assert mid.ticks_per_beat == 480
    events = absolute(mid.tracks[0])
    ons = [(t, m.note, m.velocity, m.channel) for t, m in events if m.type == 'note_on']
    snare = GM_DRUM_NOTES[1]
    assert ons == [(0, snare, 64, 9), (16 * 120, snare, 100, 9), (40 * 120, snare, 127, 9),
                   (63 * 120, snare, 127, 9)]
    offs = [t for t, m in events if m.type == 'note_off']
    assert offs == [t + 10 for t, *_ in ons]
    assert events[-1][1].type == 'end_of_track' and events[-1][0] == 64 * 120
    tempo = [m.tempo for _, m in events if m.type == 'set_tempo']
    assert tempo == [mido.bpm2tempo(120)]


@pytest.mark.parametrize('step_rate, ticks', [('1/8', 240), ('1/16T', 80), ('1/32', 60)])
def test_midi_export_follows_the_step_rate(step_rate, ticks):
    pm = long_manager()
    notes = pattern_notes(pm.patterns[0], step_rate)
    assert [t for t, _, _ in notes] == [0, 16 * ticks, 40 * ticks, 63 * ticks]


def test_midi_export_swings_like_the_sequencer():
    pattern = Pattern('A', 4)
    for step in pattern.channels[0].steps:
        step.trigger = True
    assert [t for t, _, _ in pattern_notes(pattern, '1/16', swing=0.0)] == [0, 120, 240, 360]
    # Full swing moves the second sixteenth of each pair from 480 to 720 clock ticks
    assert [t for t, _, _ in pattern_notes(pattern, '1/16', swing=1.0)] == [0, 180, 240, 420]
