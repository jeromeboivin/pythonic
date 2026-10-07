"""
The export module of the app core (slice 7): pattern MIDI files, pattern WAV
renders with their tail choice, and drum WAV renders. Every test drives the
core through its interface (set / act / poll) with temporary folders and
preferences. WAV renders run on the export thread with an offline synth, so
they report progress and their result through poll and never touch the live
stream's state.
"""

import math
import time
import wave

import mido
import numpy as np
import pytest

from pythonic.app import AppCore
from pythonic.sequencer import STEP_TICKS
from tests.fake_audio import FakeAudioBackend


@pytest.fixture
def make_core(prefs):
    cores = []

    def make(**kwargs):
        kwargs.setdefault('preferences', prefs)
        kwargs.setdefault('audio_backend', None)  # no stream: changes apply inline
        kwargs.setdefault('stall_timeout', None)
        core = AppCore(**kwargs)
        cores.append(core)
        return core

    yield make
    for core in cores:
        core.close()


def events_of(core, action_id):
    return [e for e in core.poll()['events'] if e.get('id') == action_id]


def act(core, verb, timeout=30.0, **args):
    """Run a verb and return its final poll event."""
    action_id = core.act(verb, **args)
    event = core.wait(action_id, timeout)
    assert events_of(core, action_id)[-1] is event
    return event


def run(core, verb, **args):
    event = act(core, verb, **args)
    assert event['status'] == 'done', event
    return event['result']


def read_wav(path):
    with wave.open(str(path), 'rb') as f:
        frames = f.readframes(f.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).reshape(-1, f.getnchannels())
        return f.getframerate(), f.getnchannels(), audio


def pattern_frames(core, length):
    """Frames of one pass of a pattern, as the sequencer clock runs."""
    rate = core.get('audio.synth_rate')
    ticks = length * STEP_TICKS[core.get('global.step_rate')]
    return int(math.ceil(ticks / (core.get('global.tempo') * 32.0 / rate)))


def note_ons(path):
    """(absolute tick, note, velocity) of every note on of a MIDI file."""
    mid = mido.MidiFile(str(path))
    now, notes = 0, []
    for message in mid.tracks[0]:
        now += message.time
        if message.type == 'note_on' and message.velocity > 0:
            notes.append((now, message.note, message.velocity))
    return mid, notes


# ---------------------------------------------------------------------------
# MIDI
# ---------------------------------------------------------------------------

def test_midi_export_writes_the_notes_and_velocities_of_a_64_step_pattern(make_core, tmp_path):
    core = make_core()
    core.set('global.tempo', 100)
    core.set('pattern.B.length', 64)
    core.set('pattern.B.ch1.step1.trig', True)               # velocity 64 by default
    core.set('pattern.B.ch2.step5.trig', True)
    core.set('pattern.B.ch2.step5.vel', 100)
    core.set('pattern.B.ch3.step33.trig', True)
    core.set('pattern.B.ch3.step33.acc', True)                # accent plays 127
    core.set('pattern.B.ch8.step64.trig', True)
    core.set('pattern.B.ch8.step64.vel', 7)
    path = tmp_path / 'b.mid'

    result = run(core, 'export.midi', path=str(path), pattern='B')

    assert result == {'saved': True, 'exists': False, 'path': str(path), 'pattern': 'B'}
    mid, notes = note_ons(path)
    assert notes == [(0, 36, 64), (4 * 120, 38, 100), (32 * 120, 42, 127), (63 * 120, 37, 7)]
    assert all(m.channel == 9 for m in mid.tracks[0] if m.type == 'note_on')
    tempo = [m.tempo for m in mid.tracks[0] if m.type == 'set_tempo']
    assert tempo == [mido.bpm2tempo(100)]
    assert sum(m.time for m in mid.tracks[0]) == 64 * 120  # ends where the pattern loops


def test_midi_export_defaults_to_the_selected_pattern_and_adds_the_suffix(make_core, tmp_path):
    core = make_core()
    core.set('pattern.selected', 'C')
    core.set('pattern.C.ch4.step2.trig', True)

    result = run(core, 'export.midi', path=str(tmp_path / 'c'))

    assert result['path'] == str(tmp_path / 'c.mid')
    assert result['pattern'] == 'C'
    assert note_ons(tmp_path / 'c.mid')[1] == [(120, 46, 64)]


def test_midi_export_refuses_to_replace_a_file_unless_asked(make_core, tmp_path):
    core = make_core()
    core.set('pattern.A.ch1.step1.trig', True)
    path = tmp_path / 'a.mid'
    path.write_bytes(b'keep')

    result = run(core, 'export.midi', path=str(path), pattern='A')

    assert result == {'saved': False, 'exists': True, 'path': str(path), 'pattern': 'A'}
    assert path.read_bytes() == b'keep'
    result = run(core, 'export.midi', path=str(path), pattern='A', overwrite=True)
    assert result['saved'] and result['exists']
    assert note_ons(path)[1] == [(0, 36, 64)]


def test_a_bad_pattern_or_path_is_an_error_event(make_core, tmp_path):
    core = make_core()
    event = act(core, 'export.midi', path=str(tmp_path / 'x.mid'), pattern='Z')
    assert event['status'] == 'error'
    assert 'not a pattern' in event['error']

    blocker = tmp_path / 'file'
    blocker.write_text('not a folder')
    event = act(core, 'export.midi', path=str(blocker / 'x.mid'))
    assert event['status'] == 'error'


# ---------------------------------------------------------------------------
# pattern WAV
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('tail', ['cut', 'append', 'loop'])
def test_wav_export_length_is_the_pattern_plus_its_tail(make_core, tmp_path, tail):
    core = make_core()
    core.set('global.tempo', 240)
    core.set('pattern.A.length', 8)
    core.set('pattern.A.ch1.step1.trig', True)
    core.set('pattern.A.ch2.step5.trig', True)
    path = tmp_path / f'{tail}.wav'

    result = run(core, 'export.wav', path=str(path), pattern='A', tail=tail)

    rate = core.get('audio.synth_rate')
    one_pass = pattern_frames(core, 8)
    extra = {'cut': 0, 'append': 2 * rate, 'loop': one_pass}[tail]
    assert result == {'saved': True, 'exists': False, 'path': str(path), 'pattern': 'A',
                      'tail': tail, 'frames': one_pass + extra, 'sample_rate': rate,
                      'channels': 2}
    file_rate, channels, audio = read_wav(path)
    assert (file_rate, channels) == (rate, 2)
    assert len(audio) == one_pass + extra
    assert np.max(np.abs(audio[:one_pass])) > 1000  # not silent
    if tail == 'loop':  # the second pass plays the pattern again
        assert np.max(np.abs(audio[one_pass:one_pass + 2000])) > 1000


def test_wav_export_is_mono_when_the_output_is_mono(make_core, prefs, tmp_path):
    prefs.set('audio_mono', True)
    core = make_core()
    core.set('pattern.A.ch1.step1.trig', True)

    result = run(core, 'export.wav', path=str(tmp_path / 'mono'), tail='cut')

    assert result['channels'] == 1
    assert result['path'] == str(tmp_path / 'mono.wav')
    assert read_wav(tmp_path / 'mono.wav')[1] == 1


def test_wav_export_reports_progress_then_its_result(make_core, tmp_path):
    core = make_core()
    core.set('pattern.A.length', 64)
    core.set('pattern.A.ch1.step1.trig', True)
    version = core.poll()['version']

    action_id = core.act('export.wav', path=str(tmp_path / 'a.wav'), tail='append')
    event = core.wait(action_id, 30.0)

    assert event['status'] == 'done'
    events = [e for e in core.poll(version)['events'] if e.get('id') == action_id]
    progress = [e['progress'] for e in events if e['status'] == 'progress']
    assert len(progress) >= 2
    assert progress == sorted(progress) and 0.0 <= progress[0] and progress[-1] <= 1.0
    assert events[-1] is event and all(e['verb'] == 'export.wav' for e in events)


def test_wav_export_refuses_to_replace_a_file_unless_asked(make_core, tmp_path):
    core = make_core()
    path = tmp_path / 'a.wav'
    path.write_bytes(b'keep')

    result = run(core, 'export.wav', path=str(path), tail='cut')

    assert result == {'saved': False, 'exists': True, 'path': str(path), 'pattern': 'A',
                      'tail': 'cut'}
    assert path.read_bytes() == b'keep'
    assert run(core, 'export.wav', path=str(path), tail='cut', overwrite=True)['saved']
    assert read_wav(path)[0] == core.get('audio.synth_rate')


def test_wav_export_errors_arrive_through_poll(make_core, tmp_path):
    core = make_core()
    event = act(core, 'export.wav', path=str(tmp_path / 'a.wav'), tail='forever')
    assert event['status'] == 'error'
    assert 'tail' in event['error']

    blocker = tmp_path / 'file'
    blocker.write_text('not a folder')
    event = act(core, 'export.wav', path=str(blocker / 'a.wav'), tail='cut')
    assert event['status'] == 'error'  # the write fails on the export thread
    assert not (tmp_path / 'file').is_dir()


def test_wav_export_plays_the_pattern_alone_without_its_chain(make_core, tmp_path):
    core = make_core()
    core.set('pattern.A.ch1.step1.trig', True)
    core.set('pattern.A.chained', True)               # A -> B, and B is empty
    path = tmp_path / 'a.wav'

    run(core, 'export.wav', path=str(path), pattern='A', tail='loop')

    _rate, _channels, audio = read_wav(path)
    one_pass = pattern_frames(core, 16)
    assert np.max(np.abs(audio[one_pass:one_pass + 2000])) > 1000  # A again, not B


def test_offline_render_does_not_disturb_the_live_stream(make_core, prefs, tmp_path):
    """Two identical cores play the same pattern; one exports it meanwhile.
    The live audio of both is the same, block for block."""
    streams, cores = [], []
    for _ in range(2):
        backend = FakeAudioBackend()
        core = make_core(audio_backend=backend)
        core.set('pattern.A.ch1.step1.trig', True)
        core.set('pattern.A.ch2.step3.trig', True)
        core.set('pattern.A.ch3.step5.trig', True)
        assert run(core, 'transport.play', pattern='A')['playing']
        assert core.wait(core.act('audio.start'))['status'] == 'done'
        streams.append(backend.stream)
        cores.append(core)
    exporting, quiet = cores
    blocks = [[], []]

    for _ in range(5):
        for stream, out in zip(streams, blocks):
            out.append(stream.pull().copy())
    action_id = exporting.act('export.wav', path=str(tmp_path / 'live.wav'), tail='append')
    end = time.monotonic() + 30.0
    while action_id not in exporting._results and time.monotonic() < end:
        blocks[0].append(streams[0].pull().copy())
        time.sleep(0.001)
    assert exporting.wait(action_id)['status'] == 'done'
    for _ in range(len(blocks[0]) - len(blocks[1]) + 5):
        blocks[1].append(streams[1].pull().copy())
    for _ in range(5):
        blocks[0].append(streams[0].pull().copy())

    live, reference = np.concatenate(blocks[0]), np.concatenate(blocks[1])
    assert np.max(np.abs(reference)) > 0.01
    np.testing.assert_array_equal(live, reference)
    assert exporting.poll()['transport'] == quiet.poll()['transport']
    assert exporting.synth.sample_clock == quiet.synth.sample_clock
    assert read_wav(tmp_path / 'live.wav')[2].any()


# ---------------------------------------------------------------------------
# drum WAV
# ---------------------------------------------------------------------------

def test_drum_wav_export_renders_two_seconds_of_one_channel(make_core, tmp_path):
    core = make_core()
    path = tmp_path / 'snare.wav'

    result = run(core, 'export.drum_wav', path=str(path), channel=2)

    rate = core.get('audio.synth_rate')
    assert result == {'saved': True, 'exists': False, 'path': str(path), 'channel': 2,
                      'frames': 2 * rate, 'sample_rate': rate, 'channels': 2}
    file_rate, channels, audio = read_wav(path)
    assert (file_rate, channels, len(audio)) == (rate, 2, 2 * rate)
    assert np.max(np.abs(audio)) > 1000
    assert not core.synth.channels[1].is_active  # the live channel did not play


def test_drum_wav_export_defaults_to_the_selected_channel(make_core, tmp_path):
    core = make_core()
    core.set('global.channel', 4)
    result = run(core, 'export.drum_wav', path=str(tmp_path / 'ch4'))
    assert result['channel'] == 4
    assert result['path'] == str(tmp_path / 'ch4.wav')


def test_drum_wav_export_refuses_to_replace_a_file_unless_asked(make_core, tmp_path):
    core = make_core()
    path = tmp_path / 'kick.wav'
    path.write_bytes(b'keep')
    result = run(core, 'export.drum_wav', path=str(path), channel=1)
    assert result == {'saved': False, 'exists': True, 'path': str(path), 'channel': 1}
    assert path.read_bytes() == b'keep'
    assert run(core, 'export.drum_wav', path=str(path), channel=1, overwrite=True)['saved']


def test_drum_wavs_export_writes_every_channel_into_a_folder(make_core, tmp_path):
    core = make_core()
    folder = tmp_path / 'drums'

    result = run(core, 'export.drum_wavs', folder=str(folder))

    names = [core.get(f'ch{c}.name') for c in range(1, 9)]
    assert result['saved'] and result['folder'] == str(folder)
    assert len(result['paths']) == 8
    for number, path in enumerate(result['paths'], 1):
        assert path.startswith(str(folder / f'{number:02d}_'))
        assert read_wav(path)[2].any()
    assert names[0].split()[0] in result['paths'][0]

    again = run(core, 'export.drum_wavs', folder=str(folder))
    assert again == {'saved': False, 'exists': True, 'folder': str(folder),
                     'paths': result['paths']}
    assert run(core, 'export.drum_wavs', folder=str(folder), overwrite=True)['saved']


def test_a_bad_channel_is_an_error_event(make_core, tmp_path):
    core = make_core()
    event = act(core, 'export.drum_wav', path=str(tmp_path / 'x.wav'), channel=9)
    assert event['status'] == 'error'
    assert 'not a channel' in event['error']
