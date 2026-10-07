"""
Sound addresses of the app core (slice 2): every per-channel sound parameter,
the globals (tempo, swing, step rate, fill rate, master), mutes, the selected
channel, the morph position and Edit all, all driven through the core
interface (set / get / describe / poll) with the audio callback run by hand
on a fake stream.
"""

import math
import time

import pytest

from pythonic.app import AppCore
from pythonic.app.sound import (
    CHANNEL_CC_PARAMETERS, GLOBAL_CC_PARAMETERS, SOUND_PARAMS, cc_parameter_address,
)
from pythonic.drum_channel import DrumChannel
from tests.fake_audio import FakeAudioBackend


# ---------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------

def edits(poll):
    """The changes of a poll without the undo journal state."""
    return {a: v for a, v in poll['changes'].items() if not a.startswith('undo.')}


@pytest.fixture
def backend():
    return FakeAudioBackend()


@pytest.fixture
def make_core(prefs, backend):
    cores = []

    def make(**kwargs):
        kwargs.setdefault('preferences', prefs)
        kwargs.setdefault('audio_backend', backend)
        kwargs.setdefault('stall_timeout', None)
        core = AppCore(**kwargs)
        cores.append(core)
        return core

    yield make
    for core in cores:
        core.close()


def started(core):
    event = core.wait(core.start())
    assert event['status'] == 'done', event
    return core


def sound_addresses(core, channel):
    return [f'ch{channel}.{p.suffix}' for p in SOUND_PARAMS]


def flat_engine_keys(params):
    """Flatten DrumChannel.get_parameters() into 'key' / 'lfo1.key' names."""
    keys = {}
    for key, value in params.items():
        if isinstance(value, dict):
            for sub, sub_value in value.items():
                keys[f'{key}.{sub}'] = sub_value
        else:
            keys[key] = value
    return keys


def a_value(info, current=None):
    """A valid value that differs from the address default (and from `current`)."""
    kind = info['kind']
    avoid = info['default'] if current is None else current
    if kind == 'bool':
        return not avoid
    if kind == 'enum':
        labels = info['labels']
        return labels[-1] if labels[-1] != avoid else labels[0]
    lo, hi = info['minimum'], info['maximum']
    if info['curve'] == 'log':
        lo = max(lo, hi / 1000.0)
        value = math.exp(math.log(lo) + (math.log(hi) - math.log(lo)) / 3.0)
    else:
        value = lo + (hi - lo) * 2.0 / 3.0
    if kind == 'int':
        value = int(round(value))
    if value == avoid:
        value = info['maximum']
    return value


# ---------------------------------------------------------------------------
# coverage of the engine
# ---------------------------------------------------------------------------

def test_every_engine_sound_parameter_has_an_address(make_core):
    core = make_core(audio_backend=None)
    engine_keys = set(flat_engine_keys(DrumChannel(0).get_parameters())) - {'name'}
    assert {p.key for p in SOUND_PARAMS} == engine_keys
    for channel in range(1, 9):
        for name in sound_addresses(core, channel):
            assert name in core.registry
            assert core.describe(name)['readonly'] is False
        assert core.describe(f'ch{channel}.name')['readonly'] is True
        assert core.describe(f'ch{channel}.mute')['kind'] == 'bool'


def test_every_cc_learnable_parameter_maps_to_a_settable_address(make_core):
    core = make_core(audio_backend=None)
    assert len(CHANNEL_CC_PARAMETERS) + len(GLOBAL_CC_PARAMETERS) == 35
    for name in list(CHANNEL_CC_PARAMETERS) + list(GLOBAL_CC_PARAMETERS):
        address = cc_parameter_address(name, selected_channel=3)
        assert address in core.registry, name
        info = core.describe(address)
        assert info['readonly'] is False and info['kind'] == 'float', name
    assert cc_parameter_address('osc_decay', 3) == 'ch3.osc.decay'
    assert cc_parameter_address('master_volume', 3) == 'global.master'
    assert cc_parameter_address('sound_morph', 3) == 'morph.position'


# ---------------------------------------------------------------------------
# describe
# ---------------------------------------------------------------------------

def test_describe_gives_engine_units_ranges_curves_and_defaults(make_core):
    core = make_core(audio_backend=None)
    decay = core.describe('ch3.osc.decay')
    assert decay['kind'] == 'float' and decay['unit'] == 'ms' and decay['curve'] == 'log'
    assert (decay['minimum'], decay['maximum']) == (10.0, 10000.0)
    assert decay['default'] == pytest.approx(316.23)

    wave = core.describe('ch1.osc.wave')
    assert wave['kind'] == 'enum'
    assert wave['labels'] == ['sine', 'triangle', 'sawtooth']
    assert wave['default'] == 'sine'

    target = core.describe('ch8.lfo2.target')
    assert target['labels'][0] == 'none' and 'osc_frequency' in target['labels']
    assert core.describe('ch8.pump.target')['default'] == 'level_db'

    assert core.describe('ch2.fx.delay_time')['default'] == 'eighth'
    assert core.describe('ch2.fx.reverb_width')['maximum'] == 2.0
    assert core.describe('ch2.lfo1.retrig')['default'] is True
    assert core.describe('global.tempo') == {
        'address': 'global.tempo', 'kind': 'int', 'minimum': 1, 'maximum': 300,
        'default': 120, 'unit': 'BPM', 'curve': 'linear', 'labels': [], 'readonly': False}
    assert core.describe('global.step_rate')['labels'] == ['1/8', '1/8T', '1/16', '1/16T', '1/32']
    assert core.describe('global.channel')['minimum'] == 1
    assert core.describe('global.channel')['maximum'] == 8


def test_defaults_are_the_values_of_a_fresh_channel(make_core):
    core = make_core(audio_backend=None)
    fresh = DrumChannel(0)
    for param in SOUND_PARAMS:
        assert core.describe(f'ch5.{param.suffix}')['default'] == param.get(fresh), param.suffix


# ---------------------------------------------------------------------------
# get / set round trips and clamping
# ---------------------------------------------------------------------------

def test_every_sound_address_round_trips_and_writes_the_engine(make_core):
    core = make_core(audio_backend=None)
    for channel in (1, 8):
        engine = core.synth.channels[channel - 1]
        for param in SOUND_PARAMS:
            name = f'ch{channel}.{param.suffix}'
            info = core.describe(name)
            before = flat_engine_keys(engine.get_parameters())[param.key]
            value = a_value(info, core.get(name))
            core.set(name, value)
            got = core.get(name)
            if info['kind'] == 'float':
                assert got == pytest.approx(value), name
            else:
                assert got == value, name
            after = flat_engine_keys(engine.get_parameters())[param.key]
            assert after != before, name


def test_global_addresses_round_trip_and_write_the_engine(make_core):
    core = make_core(audio_backend=None)
    synth, pm = core.synth, core.pattern_manager

    core.set('global.tempo', 133)
    assert core.get('global.tempo') == 133 and pm.bpm == 133 and synth._bpm == 133
    core.set('global.swing', 0.4)
    assert core.get('global.swing') == pytest.approx(0.4) and pm.swing == pytest.approx(0.4)
    core.set('global.step_rate', '1/8T')
    assert core.get('global.step_rate') == '1/8T' and pm.step_rate == '1/8T'
    core.set('global.fill_rate', 6)
    assert core.get('global.fill_rate') == 6 and pm.fill_rate == 6
    core.set('global.master', -12.5)
    assert core.get('global.master') == pytest.approx(-12.5)
    assert synth.master_volume_db == pytest.approx(-12.5)
    core.set('global.channel', 4)
    assert core.get('global.channel') == 4 and synth.selected_channel == 3
    core.set('ch6.mute', True)
    assert core.get('ch6.mute') is True and synth.channels[5].muted
    core.set('global.edit_all', True)
    assert core.get('global.edit_all') is True
    assert core.get('ch2.name') == synth.channels[1].name


def test_ranges_clamp_and_bad_values_are_refused(make_core):
    core = make_core(audio_backend=None)
    core.set('ch1.osc.decay', 1e9)
    assert core.get('ch1.osc.decay') == 10000.0
    core.set('ch1.osc.decay', -5)
    assert core.get('ch1.osc.decay') == 10.0
    core.set('global.tempo', 999)
    assert core.get('global.tempo') == 300
    core.set('global.fill_rate', 1)
    assert core.get('global.fill_rate') == 2
    core.set('global.channel', 12)
    assert core.get('global.channel') == 8
    core.set('ch1.osc.wave', 1)  # an enum also takes its index
    assert core.get('ch1.osc.wave') == 'triangle'
    for name, bad in [('ch1.osc.wave', 'square'), ('ch1.osc.freq', float('nan')),
                      ('ch1.osc.freq', 'loud'), ('global.step_rate', '1/64'),
                      ('ch1.osc.wave', 7)]:
        with pytest.raises(ValueError):
            core.set(name, bad)
    with pytest.raises(ValueError):
        core.set('ch1.name', 'Kick')


def test_addresses_follow_a_rebuilt_synth(make_core):
    core = make_core(audio_backend=None)
    core.set('ch2.osc.freq', 321.0)
    new = core._rebuild_synth(core.synth, 22050)
    core._install_synth((new, None))
    assert core.get('ch2.osc.freq') == pytest.approx(321.0)
    core.set('ch2.osc.freq', 654.0)
    assert new.channels[1].oscillator.frequency == pytest.approx(654.0)


# ---------------------------------------------------------------------------
# queue in, poll out
# ---------------------------------------------------------------------------

def test_set_is_applied_at_block_start_and_reported_once_applied(make_core, backend):
    core = started(make_core())
    v0 = core.poll()['version']
    core.set('ch2.osc.freq', 1234)
    assert core.synth.channels[1].oscillator.frequency != 1234.0
    assert edits(core.poll(since=v0)) == {}  # not applied yet

    backend.stream.pull()
    assert core.synth.channels[1].oscillator.frequency == 1234.0
    poll = core.poll(since=v0)
    assert edits(poll) == {'ch2.osc.freq': 1234.0}
    assert poll['version'] > v0
    assert edits(core.poll(since=poll['version'])) == {}


def test_poll_reports_the_clamped_value_with_increasing_versions(make_core):
    core = make_core(audio_backend=None)
    v0 = core.poll()['version']
    core.set('global.tempo', 500)
    v1 = core.poll(since=v0)
    assert edits(v1) == {'global.tempo': 300}
    core.set('global.tempo', 90)
    core.set('global.swing', 0.25)
    v2 = core.poll(since=v1['version'])
    assert edits(v2) == {'global.tempo': 90, 'global.swing': 0.25}
    assert v2['version'] > v1['version']


def test_selection_and_mutes_are_reported_by_poll(make_core):
    core = make_core(audio_backend=None)
    v0 = core.poll()['version']
    core.set('global.channel', 3)
    core.set('ch3.mute', True)
    assert edits(core.poll(since=v0)) == {'global.channel': 3, 'ch3.mute': True}


# ---------------------------------------------------------------------------
# Edit all
# ---------------------------------------------------------------------------

def test_edit_all_applies_a_sound_change_to_every_unmuted_channel(make_core, backend):
    core = started(make_core())
    core.set('ch3.mute', True)
    core.set('global.edit_all', True)
    backend.stream.pull()
    v0 = core.poll()['version']

    core.set('ch2.fx.reverb_mix', 0.7)
    assert all(ch.reverb_mix == 0.0 for ch in core.synth.channels)  # one block, all at once
    backend.stream.pull()

    mixes = [ch.reverb_mix for ch in core.synth.channels]
    assert mixes == [0.7, 0.7, 0.0, 0.7, 0.7, 0.7, 0.7, 0.7]
    changes = edits(core.poll(since=v0))
    assert changes == {f'ch{n}.fx.reverb_mix': 0.7 for n in (1, 2, 4, 5, 6, 7, 8)}


def test_edit_all_honours_a_mute_queued_in_the_same_block(make_core, backend):
    core = started(make_core())
    core.set('global.edit_all', True)
    v0 = core.poll()['version']
    core.set('ch6.mute', True)
    core.set('ch1.fx.delay_mix', 0.4)
    backend.stream.pull()
    assert [ch.delay_mix for ch in core.synth.channels] == [0.4] * 5 + [0.0] + [0.4] * 2
    changes = edits(core.poll(since=v0))
    assert 'ch6.fx.delay_mix' not in changes and changes['ch6.mute'] is True
    assert len(changes) == 8


def test_edit_all_covers_every_sound_address(make_core):
    core = make_core(audio_backend=None)
    core.set('global.edit_all', True)
    for param in SOUND_PARAMS:
        value = a_value(core.describe(f'ch1.{param.suffix}'))
        core.set(f'ch1.{param.suffix}', value)
        for channel in range(2, 9):
            got = core.get(f'ch{channel}.{param.suffix}')
            assert got == pytest.approx(value) if isinstance(value, float) else got == value, \
                param.suffix


def test_edit_all_still_changes_a_muted_source_channel(make_core):
    core = make_core(audio_backend=None)
    core.set('ch2.mute', True)
    core.set('ch5.mute', True)
    core.set('global.edit_all', True)
    core.set('ch2.osc.pitch', 7.0)
    pitches = [core.get(f'ch{n}.osc.pitch') for n in range(1, 9)]
    assert pitches == [7.0, 7.0, 7.0, 7.0, 0.0, 7.0, 7.0, 7.0]


def test_edit_all_set_option_overrides_the_mode(make_core):
    core = make_core(audio_backend=None)
    core.set('ch1.osc.decay', 100.0, edit_all=True)
    assert [core.get(f'ch{n}.osc.decay') for n in range(1, 9)] == [100.0] * 8
    core.set('global.edit_all', True)
    core.set('ch1.osc.decay', 200.0, edit_all=False)
    assert [core.get(f'ch{n}.osc.decay') for n in range(1, 9)] == [200.0] + [100.0] * 7


def test_edit_all_does_not_reach_mutes_or_globals(make_core):
    core = make_core(audio_backend=None)
    core.set('global.edit_all', True)
    core.set('ch1.mute', True)
    assert [core.get(f'ch{n}.mute') for n in range(1, 9)] == [True] + [False] * 7
    core.set('global.master', -6.0)
    assert core.get('global.master') == -6.0


def test_edit_all_mode_takes_effect_at_once(make_core, backend):
    """The mode is core state: a sound change right after turning it on fans out."""
    core = started(make_core())
    core.set('global.edit_all', True)
    core.set('ch1.vel.osc', 1.5)
    backend.stream.pull()
    assert [ch.osc_vel_sensitivity for ch in core.synth.channels] == [1.5] * 8


# ---------------------------------------------------------------------------
# sound morph position
# ---------------------------------------------------------------------------

def test_morph_position_blends_the_endpoints_unless_learning(make_core):
    core = make_core(audio_backend=None)
    morph = core.morph_manager
    core.synth.channels[0].set_osc_decay(100.0)
    morph.capture_endpoint_a()
    core.synth.channels[0].set_osc_decay(300.0)
    morph.capture_endpoint_b()

    core.set('morph.position', 0.5)
    assert core.get('morph.position') == 0.5
    assert 100.0 < core.get('ch1.osc.decay') < 300.0

    morph.start_learn_a()
    morph.apply_effective_position()
    core.set('morph.position', 1.0)
    assert core.get('morph.position') == 1.0
    assert core.get('ch1.osc.decay') == pytest.approx(100.0)  # pinned to A while learning
