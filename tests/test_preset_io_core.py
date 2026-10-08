"""
Preset and drum-patch files in the app core (slice 6): one load and one save
path for .mtpreset and JSON presets and for .mtdrum drum patches, the preset
clipboard, initialize and randomize all, with their undo steps, poll reports
and errors. Every test drives the core through its interface, with temporary
folders and preferences.
"""

import re
import json
import pathlib
import time

import pytest

from pythonic.app import AppCore
from pythonic.drum_channel import DrumChannel
from pythonic.preset_manager import PythonicPresetParser
from tests.fake_audio import FakeAudioBackend

TESTS = pathlib.Path(__file__).resolve().parent
REFERENCE_PRESETS = sorted(TESTS.glob('*.mtpreset'))


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


def act(core, verb, **args):
    """Run a verb and return its poll event."""
    action_id = core.act(verb, **args)
    core.wait(action_id)
    events = [e for e in core.poll()['events'] if e.get('id') == action_id]
    assert len(events) == 1
    return events[0]


def run(core, verb, **args):
    event = act(core, verb, **args)
    assert event['status'] == 'done', event
    return event['result']


def parsed(path):
    parser = PythonicPresetParser()
    return parser.convert_to_synth_format(parser.parse_file(str(path)))


def sounds(core):
    return [ch.get_parameters() for ch in core.synth.channels]


def patterns(core):
    return [p.to_dict() for p in core.pattern_manager.patterns]


# ---------------------------------------------------------------------------
# .mtpreset
# ---------------------------------------------------------------------------

def test_a_preset_without_a_name_is_named_after_its_file(make_core, tmp_path):
    # Microtonic V3 files carry no preset name (only the drum patches have one)
    text = re.sub(r'^\tName: [^\n]*\n', '', REFERENCE_PRESETS[0].read_text(), count=1, flags=re.M)
    path = tmp_path / 'AC Driver (120).mtpreset'
    path.write_text(text)
    core = make_core()

    result = run(core, 'preset.load', path=str(path))

    assert result['name'] == 'AC Driver (120)'
    assert core.get('preset.name') == 'AC Driver (120)'


@pytest.mark.parametrize('path', REFERENCE_PRESETS, ids=lambda p: p.name)
def test_every_reference_preset_loads_through_the_core(make_core, path):
    core = make_core()
    data = parsed(path)
    version = core.poll()['version']

    result = run(core, 'preset.load', path=str(path))

    name = PythonicPresetParser().parse_file(str(path)).get('Name') or path.stem
    assert result == {'path': str(path), 'name': name, 'format': 'mtpreset'}
    assert core.get('preset.name') == name
    assert core.get('preset.path') == str(path)
    for index, drum in enumerate(data['drums']):
        channel = core.synth.channels[index]
        assert channel.name == drum['name']
        assert channel.oscillator.frequency == pytest.approx(drum['osc_frequency'])
        assert channel.level_db == pytest.approx(drum['level_db'])
    assert core.get('global.tempo') == int(round(data['tempo']))
    assert core.get('global.step_rate') == data['step_rate']
    assert core.get('global.swing') == pytest.approx(data['swing'])
    assert core.get('global.master') == pytest.approx(data['master_volume_db'])
    assert [core.get(f'ch{i}.mute') for i in range(1, 9)] == data['mutes']
    changes = core.poll(version)['changes']
    assert {'ch1.osc.freq', 'ch8.name', 'global.tempo', 'pattern.A.ch1.trig',
            'program.current', 'morph.position', 'preset.name'} <= set(changes)


def test_a_mtpreset_replaces_everything(make_core, tmp_path):
    core = make_core()
    run(core, 'program.select', program=4)  # a program bank from before
    core.set('ch1.fx.reverb_mix', 0.8)  # not in a .mtpreset: back to the init value
    core.set('pattern.L.ch1.step1.trig', True)
    text = (TESTS / '808.mtpreset').read_text()
    start = text.index('\t\tl: {')
    end = text.index('\n\t\t}', start) + 4
    path = tmp_path / 'no-l.mtpreset'
    path.write_text(text[:start] + text[end:])  # pattern L left out

    run(core, 'preset.load', path=str(path))

    assert core.get('program.occupied') == [False] * 16
    assert core.get('program.current') == 1
    assert core.get('ch1.fx.reverb_mix') == core.describe('ch1.fx.reverb_mix')['default']
    assert core.get('pattern.L.ch1.step1.trig') is False  # an empty pattern
    assert core.get('morph.differs') is False
    assert core.get('morph.position') == 0.5


def test_mtpreset_patterns_come_from_the_file(make_core):
    core = make_core()
    run(core, 'preset.load', path=str(TESTS / '909.mtpreset'))
    data = parsed(TESTS / '909.mtpreset')
    for name, pattern in data['patterns'].items():
        assert core.get(f'pattern.{name}.length') == pattern['length']
        for channel, lanes in pattern['channels'].items():
            assert core.get(f'pattern.{name}.ch{channel + 1}.trig') == lanes['triggers']


# ---------------------------------------------------------------------------
# JSON round trip
# ---------------------------------------------------------------------------

def edited_core(make_core):
    core = make_core()
    run(core, 'preset.load', path=str(TESTS / '808.mtpreset'))
    core.set('ch2.osc.decay', 777.0)
    run(core, 'program.select', program=3)
    core.set('ch2.osc.decay', 333.0)
    run(core, 'morph.capture', endpoint='b')
    core.set('morph.position', 0.25)
    core.set('global.swing', 0.4)
    core.set('global.fill_rate', 6)
    core.set('global.master', -3.5)
    core.set('ch3.mute', True)
    core.set('pattern.C.length', 40)
    core.set('pattern.C.ch4.step33.vel', 99)
    core.set('pattern.C.ch4.step33.trig', True)
    run(core, 'pattern.chain_next', pattern='C')
    return core


def test_a_json_preset_round_trips_everything(make_core, tmp_path):
    core = edited_core(make_core)
    before = (sounds(core), patterns(core), core.synth.get_programs_data(),
              core.morph_manager.to_dict(), core.get('global.swing'),
              core.get('global.fill_rate'), core.get('global.master'), core.get('global.tempo'))
    path = tmp_path / 'kit.json'

    result = run(core, 'preset.save', path=str(path))
    assert result == {'saved': True, 'exists': False, 'path': str(path)}
    assert core.get('preset.name') == 'kit'

    other = make_core()
    result = run(other, 'preset.load', path=str(path))
    assert result['format'] == 'json' and result['name'] == 'kit'
    after = (sounds(other), patterns(other), other.synth.get_programs_data(),
             other.morph_manager.to_dict(), other.get('global.swing'),
             other.get('global.fill_rate'), other.get('global.master'), other.get('global.tempo'))
    assert after == before
    assert other.get('ch3.mute') is True and other.get('program.current') == 3
    assert other.get('pattern.C.ch4.step33.vel') == 99


def test_a_saved_json_keeps_the_keys_older_versions_read(make_core, tmp_path):
    core = edited_core(make_core)
    path = tmp_path / 'kit.json'
    run(core, 'preset.save', path=str(path))
    data = json.loads(path.read_text())
    assert {'version', 'master_volume_db', 'channels', 'patterns', 'tempo', 'step_rate',
            'swing', 'fill_rate', 'morph', 'programs'} <= set(data)
    # An older reader: the engine objects' own loaders
    from pythonic.pattern_manager import PatternManager
    from pythonic.synthesizer import PythonicSynthesizer
    synth = PythonicSynthesizer(44100)
    synth.load_preset_data(data)
    synth.load_programs_data(data['programs'])
    pm = PatternManager()
    pm.from_dict(data['patterns'])
    assert [c.get_parameters() for c in synth.channels] == sounds(core)
    assert [p.to_dict() for p in pm.patterns] == patterns(core)
    synth.cleanup()


def test_a_json_written_the_old_way_loads(make_core, tmp_path):
    """The JSON tkinter wrote before this slice (no mutes)."""
    old = edited_core(make_core)
    data = old.synth.get_preset_data()
    data['patterns'] = old.pattern_manager.to_dict()
    data['tempo'] = old.pattern_manager.bpm
    data['step_rate'] = old.pattern_manager.step_rate
    data['swing'] = old.pattern_manager.swing
    data['fill_rate'] = old.pattern_manager.fill_rate
    data['morph'] = old.morph_manager.to_dict()
    data['programs'] = old.synth.get_programs_data()
    path = tmp_path / 'old.json'
    path.write_text(json.dumps(data, indent=2))

    core = make_core()
    core.set('ch5.mute', True)
    run(core, 'preset.load', path=str(path))
    assert sounds(core) == sounds(old)
    assert patterns(core) == patterns(old)
    assert core.synth.get_programs_data() == old.synth.get_programs_data()
    assert core.morph_manager.to_dict() == old.morph_manager.to_dict()
    assert core.get('global.master') == -3.5  # applied with the gain (it was not)
    assert core.get('ch5.mute') is True  # no mutes in the file: left as they are


def test_a_json_without_morph_or_programs_gets_fresh_ones(make_core, tmp_path):
    core = edited_core(make_core)
    data = {'channels': [DrumChannel(i, 44100).get_parameters() for i in range(8)]}
    data['channels'][0]['osc_decay'] = 1500.0
    path = tmp_path / 'bare.json'
    path.write_text(json.dumps(data))
    run(core, 'preset.load', path=str(path))
    assert core.get('ch1.osc.decay') == 1500.0
    assert core.get('program.occupied') == [False] * 16
    assert core.get('morph.differs') is False and core.get('morph.position') == 0.5
    assert core.get('pattern.C.length') == 16
    assert core.get('global.swing') == pytest.approx(0.4)  # not in the file: kept


# ---------------------------------------------------------------------------
# saving: overwrite, suffix, folder
# ---------------------------------------------------------------------------

def test_save_refuses_to_overwrite_unless_asked(make_core, tmp_path):
    core = make_core()
    path = tmp_path / 'kit.json'
    path.write_text('keep me')
    version = core.poll()['version']

    result = run(core, 'preset.save', path=str(path))
    assert result == {'saved': False, 'exists': True, 'path': str(path)}
    assert path.read_text() == 'keep me'
    assert 'preset.path' not in core.poll(version)['changes']

    result = run(core, 'preset.save', path=str(path), overwrite=True)
    assert result == {'saved': True, 'exists': True, 'path': str(path)}
    assert json.loads(path.read_text())['channels']


def test_save_adds_the_suffix_before_checking(make_core, tmp_path):
    core = make_core()
    (tmp_path / 'kit.json').write_text('keep me')
    result = run(core, 'preset.save', path=str(tmp_path / 'kit'))
    assert result['exists'] is True and result['saved'] is False
    assert result['path'] == str(tmp_path / 'kit.json')


def test_names_are_relative_to_the_preset_folder(make_core, prefs):
    core = make_core()
    folder = pathlib.Path(prefs.get_preset_folder())
    version = core.poll()['version']
    run(core, 'preset.save', path='mine')
    assert (folder / 'mine.json').is_file()
    assert core.get('preset.files') == ['mine.json']
    assert core.poll(version)['changes']['preset.files'] == ['mine.json']
    (folder / 'b.mtpreset').write_text((TESTS / '707.mtpreset').read_text())
    (folder / 'notes.txt').write_text('-')
    assert run(core, 'preset.refresh')['files'] == ['b.mtpreset', 'mine.json']
    assert run(core, 'preset.load', path='b.mtpreset')['path'] == str(folder / 'b.mtpreset')


def test_loads_and_saves_update_the_recent_and_last_preset(make_core, prefs, tmp_path):
    core = make_core()
    path = str(TESTS / '707.mtpreset')
    run(core, 'preset.load', path=path)
    assert prefs.get('last_preset') == path
    assert prefs.get_recent_files()[0] == path
    saved = str(tmp_path / 'x.json')
    run(core, 'preset.save', path=saved)
    assert prefs.get_recent_files()[:2] == [saved, path]
    assert prefs.get('last_preset') == path  # a save does not change the start-up preset

    other = make_core()
    assert run(other, 'preset.load_last')['loaded'] is True
    assert other.get('preset.path') == path
    prefs.set('last_preset', str(tmp_path / 'gone.json'))
    assert run(other, 'preset.load_last') == {'loaded': False}


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------

def test_load_errors_arrive_through_poll(make_core, tmp_path):
    core = make_core()
    before = sounds(core)
    event = act(core, 'preset.load', path=str(tmp_path / 'missing.mtpreset'))
    assert event['status'] == 'error' and 'missing.mtpreset' in event['error']

    garbage = tmp_path / 'garbage.mtpreset'
    garbage.write_text('nothing to see')
    event = act(core, 'preset.load', path=str(garbage))
    assert event['status'] == 'error' and 'preset' in event['error'].lower()

    bad_json = tmp_path / 'bad.json'
    bad_json.write_text('{"channels": ')
    assert act(core, 'preset.load', path=str(bad_json))['status'] == 'error'
    patterns_only = tmp_path / 'patterns.json'
    patterns_only.write_text(json.dumps({'patterns': {}}))
    assert act(core, 'preset.load', path=str(patterns_only))['status'] == 'error'

    assert sounds(core) == before
    assert core.get('undo.can_undo') is False
    assert core.get('preset.path') is None


def test_save_errors_arrive_through_poll(make_core, tmp_path):
    core = make_core()
    (tmp_path / 'file').write_text('-')
    event = act(core, 'preset.save', path=str(tmp_path / 'file' / 'kit.json'))
    assert event['status'] == 'error'
    event = act(core, 'drum_patch.save', path=str(tmp_path / 'p.mtdrum'), channel=9)
    assert event['status'] == 'error' and 'channel' in event['error']


# ---------------------------------------------------------------------------
# undo
# ---------------------------------------------------------------------------

def test_a_load_is_one_undo_step(make_core):
    core = edited_core(make_core)
    before = (sounds(core), patterns(core), core.synth.get_programs_data(),
              core.morph_manager.to_dict(), core.get('global.swing'))
    version = core.poll()['version']

    run(core, 'preset.load', path=str(TESTS / 'DMX.mtpreset'))
    loaded = (sounds(core), patterns(core))
    assert core.get('undo.can_undo') is True

    assert run(core, 'undo') == {'done': True, 'label': 'load preset'}
    after = (sounds(core), patterns(core), core.synth.get_programs_data(),
             core.morph_manager.to_dict(), core.get('global.swing'))
    assert after == before
    assert core.get('program.current') == 3
    assert core.get('ch3.mute') is False  # the file's mutes stay: mutes are not undone
    assert 'ch1.osc.decay' in core.poll(version)['changes']

    run(core, 'redo')
    assert (sounds(core), patterns(core)) == loaded


# ---------------------------------------------------------------------------
# drum patches
# ---------------------------------------------------------------------------

def test_a_drum_patch_round_trips_into_another_channel(make_core, tmp_path):
    core = make_core()
    run(core, 'preset.load', path=str(TESTS / '808.mtpreset'))
    core.set('global.channel', 2)
    core.set('ch2.osc.decay', 432.0)
    saved = sounds(core)[1]
    path = tmp_path / 'snare'

    result = run(core, 'drum_patch.save', path=str(path))  # the selected channel
    assert result == {'saved': True, 'exists': False, 'path': str(path) + '.mtdrum', 'channel': 2}
    assert run(core, 'drum_patch.save', path=str(path))['saved'] is False
    old5 = sounds(core)[4]
    version = core.poll()['version']

    result = run(core, 'drum_patch.load', path=str(path) + '.mtdrum', channel=5)
    assert result['channel'] == 5 and result['name'] == saved['name']
    loaded = sounds(core)[4]
    for key in ('osc_frequency', 'osc_decay', 'noise_filter_freq', 'level_db', 'pan',
                'osc_noise_mix', 'name', 'choke_enabled'):
        assert loaded[key] == pytest.approx(saved[key]) if isinstance(saved[key], float) \
            else loaded[key] == saved[key]
    changes = core.poll(version)['changes']
    assert changes['ch5.osc.decay'] == pytest.approx(432.0)
    assert changes['ch5.name'] == saved['name']

    assert run(core, 'undo')['label'] == 'load drum patch'
    assert sounds(core)[4] == old5


# ---------------------------------------------------------------------------
# clipboard, initialize, randomize all
# ---------------------------------------------------------------------------

def test_copy_and_paste_bring_the_preset_back(make_core):
    core = edited_core(make_core)
    before = (sounds(core), patterns(core), core.synth.get_programs_data(),
              core.morph_manager.to_dict())
    assert core.get('preset.clipboard') is False
    with pytest.raises(AssertionError):
        run(core, 'preset.paste')  # nothing to paste: an error event

    run(core, 'preset.copy')
    assert core.get('preset.clipboard') is True
    run(core, 'preset.load', path=str(TESTS / 'LM2.mtpreset'))
    run(core, 'preset.paste')
    assert (sounds(core), patterns(core), core.synth.get_programs_data(),
            core.morph_manager.to_dict()) == before
    run(core, 'preset.load', path=str(TESTS / 'LM2.mtpreset'))
    run(core, 'preset.paste')  # the clipboard can be pasted again
    assert sounds(core) == before[0]


def test_cut_copies_then_initializes(make_core):
    core = edited_core(make_core)
    before = sounds(core)
    run(core, 'preset.cut')
    init = DrumChannel(0, 44100).get_parameters()
    assert all(s == init for s in sounds(core))
    assert all(core.get(f'pattern.{p}.empty') for p in 'ABCDEFGHIJKL')
    assert core.get('morph.differs') is False
    assert core.get('program.current') == 3  # programs and globals stay
    assert run(core, 'undo')['label'] == 'initialize preset'
    assert sounds(core) == before
    run(core, 'preset.paste')
    assert sounds(core) == before


def test_randomize_all_is_one_step(make_core):
    core = edited_core(make_core)
    core.set('pattern.selected', 'C')
    before = (sounds(core), patterns(core))
    run(core, 'preset.randomize_all')
    assert sounds(core) != before[0]
    assert [ch['level_db'] for ch in sounds(core)] == [ch['level_db'] for ch in before[0]]
    assert patterns(core)[2] != before[1][2] and patterns(core)[0] == before[1][0]
    assert core.get('morph.differs') is False
    run(core, 'undo')
    assert (sounds(core), patterns(core)) == before


# ---------------------------------------------------------------------------
# with a running stream
# ---------------------------------------------------------------------------

def test_a_load_is_swapped_in_at_block_start(make_core, tmp_path):
    backend = FakeAudioBackend()
    core = make_core(audio_backend=backend)
    core.wait(core.act('audio.start'))
    stream = backend.stream
    core.set('global.tempo', 99)  # queued, not applied yet: the load comes after it
    action_id = core.act('preset.load', path=str(TESTS / '505.mtpreset'))
    end = time.monotonic() + 5.0
    while action_id not in core._results and time.monotonic() < end:
        out = stream.pull()
        assert not (out != out).any()  # no NaN: every block was rendered
    assert core.wait(action_id)['status'] == 'done'
    assert core.get('global.tempo') == int(round(parsed(TESTS / '505.mtpreset')['tempo']))

    action_id = core.act('undo')
    while action_id not in core._results and time.monotonic() < end:
        stream.pull()
    assert core.get('global.tempo') == 99
