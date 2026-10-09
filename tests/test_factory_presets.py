"""
The factory presets (pythonic/factory/presets): each loads through the core
with its machine's drum patches as they are in tests/<machine>.mtpreset (the
TR-8's: eight of the fitted drum patches of pythonic/factory/drum_patches.json),
twelve patterns of its own, and all seven machines in programs 1-7 in the same
order, opening on its own.
"""

import copy
import json

import pytest

from pythonic.factory import MACHINES, PATCHES_FILE, PRESETS_DIR
from pythonic.pattern_manager import PatternManager
from tests.test_preset_io_core import TESTS, make_core, run, sounds  # noqa: F401 (fixture)

FACTORY_PRESETS = [PRESETS_DIR / f'{machine} Beats.json' for machine in MACHINES]
# The TR-8 kit, in the 909's channel layout
TR8_KIT = ['TR-8 Kick 01', 'TR-8 Tom 01', 'TR-8 Tom 02', 'TR-8 Clap 01', 'TR-8 Snare 01',
           'TR-8 Tom 05', 'TR-8 Rim 01', 'TR-8 Closed Hat 01']


def fitted_patches():
    """The drum patches of drum_patches.json, by name."""
    with open(PATCHES_FILE, encoding='utf-8') as f:
        return {sound['name']: sound for sound in json.load(f)['patches']}


@pytest.fixture
def machine_sounds(make_core):
    """The sounds of each machine, loaded from its reference .mtpreset."""
    core = make_core()
    result = {}
    for machine in MACHINES:
        if machine == 'TR-8':
            fitted = fitted_patches()
            for channel, name in zip(core.synth.channels, TR8_KIT):
                channel.set_parameters({**copy.deepcopy(core.init_sound), **copy.deepcopy(fitted[name])})
        else:
            run(core, 'preset.load', path=str(TESTS / f'{machine}.mtpreset'))
        result[machine] = sounds(core)
    return result


def test_the_factory_has_one_preset_per_machine():
    assert sorted(p.name for p in PRESETS_DIR.glob('*.json')) == sorted(p.name for p in FACTORY_PRESETS)


@pytest.mark.parametrize('machine', MACHINES)
def test_a_factory_preset_plays_its_machine(make_core, machine_sounds, machine):
    core = make_core()

    result = run(core, 'preset.load', path=str(PRESETS_DIR / f'{machine} Beats.json'))

    assert result['format'] == 'json' and result['name'] == f'{machine} Beats'
    assert sounds(core) == machine_sounds[machine]
    assert core.get('program.current') == MACHINES.index(machine) + 1
    patterns = core.pattern_manager.patterns
    assert [p.name for p in patterns] == PatternManager.PATTERN_NAMES
    for pattern in patterns:
        assert pattern.length in (16, 32, 64)
        assert any(step.trigger for lane in pattern.channels for step in lane.steps), pattern.name


@pytest.mark.parametrize('machine', MACHINES)
def test_programs_1_to_7_are_the_seven_machines(make_core, machine_sounds, machine):
    core = make_core()
    run(core, 'preset.load', path=str(PRESETS_DIR / f'{machine} Beats.json'))

    assert core.get('program.occupied') == [True] * 7 + [False] * 9
    for program, kit in enumerate(MACHINES, start=1):
        run(core, 'program.select', program=program)
        assert sounds(core) == machine_sounds[kit], (program, kit)


# ---------------------------------------------------------------------------
# the core's factory addresses and verbs
# ---------------------------------------------------------------------------

def act(core, verb, **args):
    action_id = core.act(verb, **args)
    return core.wait(action_id)


def test_the_core_lists_the_factory_presets_in_machine_order(make_core):
    core = make_core()
    assert core.get('factory.presets') == [f'{m} Beats.json' for m in MACHINES]


@pytest.mark.parametrize('name', ['808 Beats', '808 Beats.json'])
def test_a_factory_preset_loads_by_name(make_core, machine_sounds, name):
    core = make_core()
    assert core.get('preset.factory') is False
    version = core.poll()['version']

    result = run(core, 'preset.load', factory=name)

    assert result == {'path': str(PRESETS_DIR / '808 Beats.json'), 'name': '808 Beats', 'format': 'json'}
    assert sounds(core) == machine_sounds['808']
    assert core.get('preset.factory') is True
    assert core.poll(version)['changes']['preset.factory'] is True
    run(core, 'preset.load', path=str(TESTS / '909.mtpreset'))
    assert core.get('preset.factory') is False


def test_a_load_needs_a_path_or_a_known_factory_preset(make_core):
    core = make_core()
    assert 'not a factory preset' in act(core, 'preset.load', factory='606 Beats')['error']
    assert act(core, 'preset.load')['status'] == 'error'
    both = act(core, 'preset.load', path=str(TESTS / '808.mtpreset'), factory='808 Beats')
    assert both['status'] == 'error'


def test_the_factory_presets_are_read_only(make_core):
    core = make_core()
    run(core, 'preset.load', factory='909 Beats')
    path = PRESETS_DIR / '909 Beats.json'
    before = path.read_bytes()

    for target in (path, PRESETS_DIR / 'mine.json'):
        event = act(core, 'preset.save', path=str(target), overwrite=True)
        assert event['status'] == 'error' and 'read-only' in event['error']
    assert path.read_bytes() == before
    assert not (PRESETS_DIR / 'mine.json').exists()


def test_program_names_are_the_kits_and_follow_the_channel_names(make_core, tmp_path):
    core = make_core()
    assert core.get('program.names') == [''] * 16
    run(core, 'preset.load', factory='DMX Beats')
    assert core.get('program.names') == list(MACHINES) + [''] * 9

    # a drum patch from another machine: the current program's channels have nothing in common
    run(core, 'preset.load', path=str(TESTS / '808.mtpreset'))
    run(core, 'drum_patch.save', path=str(tmp_path / 'bd'), channel=1)
    run(core, 'preset.load', factory='DMX Beats')
    version = core.poll()['version']
    run(core, 'drum_patch.load', path=str(tmp_path / 'bd.mtdrum'), channel=1)

    assert core.get('program.names')[4] == ''
    assert core.poll(version)['changes']['program.names'][4] == ''


def test_restoring_the_factory_kits_refills_programs_1_to_7(make_core, machine_sounds):
    core = make_core()
    run(core, 'preset.load', path=str(TESTS / 'LM2.mtpreset'))  # an empty bank
    run(core, 'program.select', program=10)  # a program of the user's own
    lm2 = sounds(core)

    assert run(core, 'program.restore_factory') == {'programs': [1, 2, 3, 4, 5, 6, 7]}

    assert sounds(core) == lm2  # program 10 plays on
    assert core.get('program.occupied')[:7] == [True] * 7
    run(core, 'program.select', program=4)
    assert sounds(core) == machine_sounds['909']


def test_restoring_brings_back_the_current_kit_in_one_undo_step(make_core, machine_sounds):
    core = make_core()
    run(core, 'preset.load', factory='808 Beats')  # opens on program 3
    core.set('ch1.osc.decay', 900.0)
    edited = sounds(core)

    run(core, 'program.restore_factory')
    assert sounds(core) == machine_sounds['808']

    assert run(core, 'undo') == {'done': True, 'label': 'restore factory kits'}
    assert sounds(core) == edited


def test_the_factory_drum_patches_are_every_kit_sound_then_the_fitted_ones(make_core, machine_sounds):
    core = make_core()
    names = core.get('factory.patches')
    kit_names = [sound['name'] for m in MACHINES for sound in machine_sounds[m]]
    others = sorted(set(fitted_patches()) - set(TR8_KIT))
    assert names == kit_names + others
    assert len(names) == 56 + 33 and names[0] == '505 BD' and names[55] == 'TR-8 Closed Hat 01'
    assert all(name.split()[0] in MACHINES for name in names)


def test_a_fitted_drum_patch_outside_the_kit_loads_by_name(make_core):
    core = make_core()
    fitted = fitted_patches()['TR-8 Cowbell']

    assert run(core, 'drum_patch.load', factory='TR-8 Cowbell', channel=6)['name'] == 'TR-8 Cowbell'
    assert sounds(core)[5] == {**core.init_sound, **fitted}


def test_a_factory_drum_patch_loads_into_a_channel_in_one_undo_step(make_core, machine_sounds):
    core = make_core()
    run(core, 'preset.load', factory='808 Beats')
    before = sounds(core)
    version = core.poll()['version']

    result = run(core, 'drum_patch.load', factory='909 BD', channel=2)

    assert result == {'channel': 2, 'name': '909 BD', 'path': None}
    assert sounds(core)[1] == machine_sounds['909'][0]
    assert sounds(core)[0] == before[0]
    changes = core.poll(version)['changes']
    assert changes['ch2.name'] == '909 BD' and changes['program.names'][2] == ''
    assert run(core, 'undo') == {'done': True, 'label': 'load drum patch'}
    assert sounds(core) == before


def test_a_drum_patch_load_needs_a_path_or_a_known_factory_patch(make_core):
    core = make_core()
    assert 'not a factory drum patch' in act(core, 'drum_patch.load', factory='606 BD')['error']
    assert act(core, 'drum_patch.load', channel=1)['status'] == 'error'
