"""
Kits and the drum patch library in the app core (kits.py, library.py): the
kit library apart from presets (factory kits, user kits in the Kits folder),
the current kit and its edited mark, edits kept while away, writing, kits
saved with presets and recalled by patterns, and the library INST mode steps
through. Every test drives the core through its interface.
"""

import json
import time

import pytest

from pythonic.app import AppCore
from pythonic.app.library import category, drum_type
from pythonic.factory import MACHINES, PRESETS_DIR
from tests.fake_audio import FakeAudioBackend
from tests.test_preset_io_core import TESTS


@pytest.fixture
def backend():
    return FakeAudioBackend()


@pytest.fixture
def make_core(prefs, backend):
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


@pytest.fixture
def folder(prefs, tmp_path):
    path = tmp_path / 'presets'
    path.mkdir(exist_ok=True)
    return path


def run(core, verb, **args):
    event = core.wait(core.act(verb, **args))
    assert event['status'] == 'done', event
    return event.get('result')


def fails(core, verb, **args):
    event = core.wait(core.act(verb, **args))
    assert event['status'] == 'error', event
    return event['error']


def run_live(core, stream, verb, **args):
    action_id = core.act(verb, **args)
    end = time.monotonic() + 5.0
    while time.monotonic() < end:
        stream.pull()
        try:
            event = core.wait(action_id, timeout=0.005)
        except TimeoutError:
            continue
        assert event['status'] == 'done', event
        return event.get('result')
    raise AssertionError(f'{verb} did not finish')


def names(core):
    return [core.get(f'ch{n}.name') for n in range(1, 9)]


def write_kit(folder, name, channel_names):
    kits = folder / 'Kits'
    kits.mkdir(exist_ok=True)
    (kits / f'{name}.json').write_text(json.dumps(
        {'format': 'pythonic-kit', 'version': 1, 'name': name,
         'channels': [{'name': n, 'osc_decay': 100.0 + i} for i, n in enumerate(channel_names)]}))


# ---------------------------------------------------------------------------
# the library of kits
# ---------------------------------------------------------------------------

def test_the_factory_kits_come_first_then_the_user_kits_by_name(make_core, folder):
    write_kit(folder, 'zed', ['Z'] * 8)
    write_kit(folder, 'Alpha', ['A'] * 8)
    (folder / 'Kits' / 'broken.json').write_text('{not json')
    core = make_core()
    listing = core.get('kit.list')
    assert [k['id'] for k in listing] == [f'factory:{m}' for m in MACHINES] + ['user:Alpha', 'user:zed']
    assert listing[3] == {'id': 'factory:909', 'name': '909', 'factory': True}
    assert listing[-1] == {'id': 'user:zed', 'name': 'zed', 'factory': False}


def test_a_new_core_plays_the_preset_kit_named_after_the_preset(make_core):
    core = make_core()
    assert core.get('kit.current') == ''
    assert core.get('kit.name') == 'Untitled'
    assert core.get('kit.edited') is False


def test_selecting_a_kit_plays_its_sounds(make_core):
    core = make_core()
    result = run(core, 'kit.select', kit='factory:808')
    assert result == {'kit': 'factory:808', 'name': '808', 'edited': False}
    assert core.get('kit.current') == 'factory:808' and core.get('kit.name') == '808'
    assert all(name.startswith('808 ') for name in names(core))
    assert 'not a kit' in fails(core, 'kit.select', kit='factory:nope')


def test_an_edit_marks_the_kit_edited_and_poll_reports_it(make_core):
    core = make_core()
    run(core, 'kit.select', kit='factory:909')
    version = core.poll()['version']
    core.set('ch2.osc.decay', 1234.0)
    assert core.get('kit.edited') is True
    core.poll()  # the edited mark follows the channel report
    assert core.poll(version)['changes'].get('kit.edited') is True
    core.set('ch2.osc.decay', core.get('ch2.osc.decay'))  # no change: still edited
    run(core, 'undo')
    assert core.get('kit.edited') is False


def test_the_edits_of_a_kit_left_are_kept_for_the_session(make_core):
    core = make_core()
    run(core, 'kit.select', kit='factory:909')
    core.set('ch1.osc.decay', 1500.0)
    assert run(core, 'kit.select', kit='factory:808')['edited'] is False
    assert run(core, 'kit.select', kit='factory:909')['edited'] is True
    assert core.get('ch1.osc.decay') == 1500.0
    core.set('ch1.osc.decay', core.get('ch2.osc.decay'))
    # Edits undone by hand: nothing is kept any more
    run(core, 'undo')
    run(core, 'kit.select', kit='factory:808')
    assert run(core, 'kit.select', kit='factory:909')['edited'] is True


def test_a_kit_switch_is_one_undo_step_and_browsing_one_for_the_whole_browse(make_core):
    core = make_core()
    before = names(core)
    run(core, 'kit.select', kit='factory:505')
    run(core, 'undo')
    assert core.get('kit.current') == '' and names(core) == before

    for kit in ('factory:505', 'factory:707', 'factory:808'):
        run(core, 'kit.select', kit=kit, browse=True)
    assert run(core, 'undo') == {'done': True, 'label': 'kit 505'}
    assert core.get('kit.current') == '' and names(core) == before
    run(core, 'redo')
    assert core.get('kit.current') == 'factory:808'

    # An edit between two browses splits them
    run(core, 'kit.select', kit='factory:909', browse=True)
    core.set('ch1.osc.decay', 900.0)
    run(core, 'kit.select', kit='factory:DMX', browse=True)
    run(core, 'undo')
    assert core.get('kit.current') == 'factory:909' and core.get('ch1.osc.decay') == 900.0


def test_a_kit_switch_applies_at_block_start(make_core, backend):
    core = make_core(audio_backend=backend)
    core.wait(core.start())
    assert run_live(core, backend.stream, 'kit.select', kit='factory:LM2')['kit'] == 'factory:LM2'
    assert core.get('ch1.name').startswith('LM2')


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

def test_writing_a_kit_under_a_name_stores_it_and_plays_it(make_core, folder):
    core = make_core()
    run(core, 'kit.select', kit='factory:909')
    core.set('ch1.osc.decay', 777.0)
    result = run(core, 'kit.write', name='  My   909 ')
    assert result == {'saved': True, 'exists': False, 'name': 'My 909', 'kit': 'user:My 909'}
    saved = json.loads((folder / 'Kits' / 'My 909.json').read_text())
    assert saved['format'] == 'pythonic-kit' and saved['name'] == 'My 909'
    assert saved['channels'][0]['osc_decay'] == 777.0
    assert core.get('kit.current') == 'user:My 909' and core.get('kit.edited') is False
    assert core.get('kit.list')[-1] == {'id': 'user:My 909', 'name': 'My 909', 'factory': False}
    # The factory kit lost its edits to the written kit
    assert run(core, 'kit.select', kit='factory:909')['edited'] is False


def test_writing_over_a_kit_asks_for_overwrite_and_a_user_kit_writes_in_place(make_core, folder):
    write_kit(folder, 'Mine', [f'M{i}' for i in range(8)])
    core = make_core()
    run(core, 'kit.select', kit='factory:808')
    assert run(core, 'kit.write', name='Mine') == {'saved': False, 'exists': True, 'name': 'Mine'}
    assert core.get('kit.current') == 'factory:808'
    assert 'new name' in fails(core, 'kit.write')  # a factory kit is written under a new name

    run(core, 'kit.select', kit='user:Mine')
    core.set('ch3.osc.decay', 432.0)
    assert run(core, 'kit.write')['exists'] is True
    assert json.loads((folder / 'Kits' / 'Mine.json').read_text())['channels'][2]['osc_decay'] == 432.0
    assert core.get('kit.edited') is False
    assert 'not a kit name' in fails(core, 'kit.write', name='a/b')


def test_the_preset_programs_become_user_kits(make_core, folder):
    core = make_core()
    run(core, 'preset.load', factory='808 Beats')
    result = run(core, 'kit.from_programs')
    assert result['kits'][:3] == ['808 Beats 1 505', '808 Beats 2 707', '808 Beats 3 808']
    assert 'user:808 Beats 4 909' in [k['id'] for k in core.get('kit.list')]
    assert run(core, 'kit.from_programs')['kits'][0] == '808 Beats 1 505 (2)'


def test_a_rescan_finds_new_kit_files(make_core, folder):
    core = make_core()
    write_kit(folder, 'Later', ['L'] * 8)
    run(core, 'kit.refresh')
    assert core.get('kit.list')[-1]['id'] == 'user:Later'


# ---------------------------------------------------------------------------
# presets
# ---------------------------------------------------------------------------

def test_a_preset_saves_its_kit_and_loads_it_back(make_core, folder):
    core = make_core()
    run(core, 'kit.select', kit='factory:707')
    run(core, 'preset.save', path='with kit')
    assert json.loads((folder / 'with kit.json').read_text())['kit'] == 'factory:707'
    run(core, 'kit.select', kit='factory:505')
    run(core, 'preset.load', path='with kit.json')
    assert core.get('kit.current') == 'factory:707' and core.get('kit.edited') is False

    core.set('ch1.osc.decay', 1999.0)
    run(core, 'preset.save', path='edited kit', overwrite=True)
    run(core, 'preset.load', path='edited kit.json')
    assert core.get('kit.current') == 'factory:707' and core.get('kit.edited') is True


def test_a_preset_without_a_kit_plays_its_own_sounds(make_core, folder):
    core = make_core()
    run(core, 'kit.select', kit='factory:909')
    run(core, 'preset.load', path=str(TESTS / '808.mtpreset'))
    assert core.get('kit.current') == '' and core.get('kit.name') == core.get('preset.name')
    assert core.get('kit.edited') is False
    sounds = names(core)
    # The preset kit is one more kit to go back to
    run(core, 'kit.select', kit='factory:505')
    run(core, 'kit.select', kit='')
    assert names(core) == sounds
    run(core, 'preset.save', path='plain')
    assert 'kit' not in json.loads((folder / 'plain.json').read_text())
    run(core, 'preset.initialize')
    assert core.get('kit.current') == '' and core.get('kit.edited') is False


def test_a_factory_preset_plays_its_factory_kit(make_core):
    core = make_core()
    for machine in MACHINES:
        assert json.loads((PRESETS_DIR / f'{machine} Beats.json').read_text())['kit'] == f'factory:{machine}'
    run(core, 'preset.load', factory='909 Beats')
    assert core.get('kit.current') == 'factory:909' and core.get('kit.edited') is False


# ---------------------------------------------------------------------------
# patterns recall kits
# ---------------------------------------------------------------------------

def test_a_pattern_kit_is_a_known_kit_and_an_undo_step(make_core):
    core = make_core()
    assert core.get('pattern.C.kit') == ''
    core.set('pattern.C.kit', 'factory:808')
    assert core.get('pattern.C.kit') == 'factory:808'
    with pytest.raises(ValueError):
        core.set('pattern.C.kit', 'factory:nope')
    run(core, 'undo')
    assert core.get('pattern.C.kit') == ''


def test_selecting_a_pattern_while_stopped_recalls_its_kit(make_core):
    core = make_core()
    core.set('pattern.B.kit', 'factory:DMX')
    run(core, 'pattern.select', pattern='B')
    assert core.get('kit.current') == 'factory:DMX' and core.get('ch1.name').startswith('DMX')
    run(core, 'pattern.select', pattern='A')  # no kit: the kit stays
    assert core.get('kit.current') == 'factory:DMX'


def test_the_pattern_kit_is_saved_with_the_preset(make_core, folder):
    core = make_core()
    core.set('pattern.D.kit', 'factory:LM2')
    run(core, 'preset.save', path='kits')
    patterns = json.loads((folder / 'kits.json').read_text())['patterns']['patterns']
    assert patterns[3]['kit'] == 'factory:LM2' and 'kit' not in patterns[0]
    run(core, 'preset.initialize')
    assert core.get('pattern.D.kit') == ''
    run(core, 'preset.load', path='kits.json')
    assert core.get('pattern.D.kit') == 'factory:LM2'


def test_a_pattern_starting_to_play_switches_to_its_kit(make_core, backend):
    core = make_core(audio_backend=backend)
    core.wait(core.start())
    stream = backend.stream
    core.set('global.tempo', 300)
    core.set('pattern.A.length', 2)
    core.set('pattern.B.kit', 'factory:707')
    run_live(core, stream, 'transport.play', pattern='A')
    assert core.get('kit.current') == ''
    run_live(core, stream, 'pattern.queue', pattern='B')
    end = time.monotonic() + 5.0
    while core.poll()['transport']['playing_pattern'] != 1 and time.monotonic() < end:
        stream.pull()
    assert core.get('kit.current') == 'factory:707' and core.get('ch1.name').startswith('707')
    changes = core.poll()['changes']
    assert changes['kit.current'] == 'factory:707' and changes['ch1.name'].startswith('707')


# ---------------------------------------------------------------------------
# the drum patch library
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('name, expected', [
    ('909 BD', 'BD'), ('Bd01', 'BD'), ('BDRUM12', 'BD'), ('TR-8 Kick 01', 'BD'), ('808 SD', 'SD'),
    ('808 LT', 'TOM'), ('TR-8 Tom 05', 'TOM'), ('TR-8 Rim 01', 'RIM'), ('909 Clap', 'CLAP'),
    ('CLAP1', 'CLAP'), ('909 CH', 'CH'), ('808 OH', 'OH'), ('Crash', 'CYM'), ('Ride 2', 'CYM'),
    ('707 Cowbell', 'PERC'), ('LM2 Cabasa', 'PERC'), ('zap01', 'FX'), ('TR-8 Snap', 'CLAP'),
    # run-together names, percussion and effect words, sample pack codes
    ('Drybd2', 'BD'), ('Fingersn', 'SD'), ('Lecmidt1', 'TOM'), ('Tinyrim2', 'RIM'), ('Realoh1', 'OH'),
    ('Roomch', 'CH'), ('Hiagogo', 'PERC'), ('scratch05', 'FX'), ('SFX010', 'FX'),
    ('HHCD4', 'CH'), ('HHOD6', 'OH'), ('LTAD7', 'TOM'), ('BT0A0A7', 'BD'), ('ST3T7SA', 'SD'),
    ('CSHD6', 'CYM'), ('Hat Open Accent Max Decay', 'OH'), ('909-01', 'OTHER'), ('Smp001', 'OTHER'),
])
def test_a_drum_patch_category_comes_from_its_name(name, expected):
    assert category(name) == expected


def test_a_category_falls_back_on_the_folders():
    assert drum_type('X1') == '' and category('X1', 'Snares') == 'SD'


def test_the_library_is_sorted_by_category_factory_first(make_core):
    core = make_core()
    library = core.get('drum_patch.library')
    assert library[0] == ['factory:505 BD', '505 BD', 'BD']
    order = ['BD', 'SD', 'TOM', 'RIM', 'CLAP', 'CH', 'OH', 'CYM', 'PERC', 'FX', 'OTHER']
    seen = [entry[2] for entry in library]
    assert seen == sorted(seen, key=order.index)


def test_the_drum_patch_folder_joins_the_library(make_core, tmp_path):
    core = make_core()
    run(core, 'drum_patch.save', path=str(tmp_path / 'source.mtdrum'), channel=1)
    text = (tmp_path / 'source.mtdrum').read_text()
    patches = tmp_path / 'patches'
    (patches / 'Machine' / 'Snares').mkdir(parents=True)
    for path in ('Machine/Snares/Thing2', 'Machine/Snares/Thing10', 'Kick 1'):
        (patches / f'{path}.mtdrum').write_text(text)
    core.set('pref.patch_folder', str(patches))
    run(core, 'drum_patch.refresh')
    user = [e for e in core.get('drum_patch.library') if e[0].startswith('user:')]
    assert user == [['user:Kick 1.mtdrum', 'Kick 1', 'BD'],
                    ['user:Machine/Snares/Thing2.mtdrum', 'Thing2', 'SD'],
                    ['user:Machine/Snares/Thing10.mtdrum', 'Thing10', 'SD']]
    result = run(core, 'drum_patch.load', library='user:Machine/Snares/Thing10.mtdrum', channel=3)
    assert result['name'] == 'Thing10' and core.get('ch3.name') == 'Thing10'
    assert 'not a drum patch of the library' in fails(core, 'drum_patch.load', library='user:../x.mtdrum')


def test_browsing_the_library_is_one_undo_step_per_channel(make_core):
    core = make_core()
    before = core.get('ch2.name')
    for patch in ('factory:808 SD', 'factory:909 SD', 'factory:DMX SD'):
        run(core, 'drum_patch.load', library=patch, channel=2, browse=True)
    assert core.get('ch2.name') == 'DMX SD'
    run(core, 'drum_patch.load', library='factory:909 BD', channel=1, browse=True)
    run(core, 'undo')
    assert core.get('ch1.name') != '909 BD' and core.get('ch2.name') == 'DMX SD'
    run(core, 'undo')
    assert core.get('ch2.name') == before
    assert 'give a path' in fails(core, 'drum_patch.load', library='factory:909 BD', factory='909 BD')
