"""
Preferences in the app core (slice 6): the ``pref.*`` addresses save the
preferences file with the keys tkinter has always written, apply what applies
live, and leave the stream settings to ``audio.apply``; a file written by an
older version is read. Every test uses a temporary preferences folder.
"""

import json
import time

import pytest

from pythonic.app import AppCore
from pythonic.preferences_manager import PreferencesManager
from tests.fake_audio import FakeAudioBackend


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


def run(core, verb, **args):
    event = core.wait(core.act(verb, **args))
    assert event['status'] == 'done', event
    return event['result']


def saved(prefs):
    """The preferences file as another process would read it."""
    with open(prefs.prefs_file, encoding='utf-8') as f:
        return json.load(f)


def pull_until(core, stream, predicate):
    end = time.monotonic() + 5.0
    while time.monotonic() < end:
        stream.pull()
        if predicate():
            return
    raise AssertionError('condition not reached')


# ---------------------------------------------------------------------------
# describe
# ---------------------------------------------------------------------------

def test_the_setup_values_are_described(make_core):
    core = make_core()
    device = core.describe('pref.audio.device')
    assert device['kind'] == 'str' and device['labels'] == ['Fake Out', 'Fake Duplex']
    assert core.describe('pref.audio.input_device')['labels'] == ['Fake In', 'Fake Duplex']
    buffer = core.describe('pref.audio.buffer_ms')
    assert buffer['unit'] == 'ms' and buffer['default'] == 23.8 and 23.8 in buffer['labels']
    assert core.describe('pref.audio.sample_rate')['labels'] == [
        96000, 48000, 44100, 32000, 22050, 11025, 8000]
    assert core.describe('pref.audio.synth_rate')['labels'] == [0, 22050, 11025, 8000]
    smoothing = core.describe('pref.smoothing_ms')
    assert (smoothing['minimum'], smoothing['maximum'], smoothing['default'],
            smoothing['unit']) == (5.0, 100.0, 30.0, 'ms')
    temperature = core.describe('pref.ai.pattern_temperature')
    assert (temperature['minimum'], temperature['maximum'], temperature['default']) == (0.1, 3.0, 0.7)
    assert core.describe('pref.ai.patch_temperature')['default'] == 1.0
    assert core.get('audio.default_input') == 'Fake In'


def test_preferences_are_not_midi_learnable(make_core):
    core = make_core()
    assert core.midi.valid_target('pref.smoothing_ms') is False
    assert core.midi.valid_target('ch1.osc.decay') is True


# ---------------------------------------------------------------------------
# stream settings wait for audio.apply
# ---------------------------------------------------------------------------

def test_stream_settings_are_saved_and_wait_for_audio_apply(make_core, backend, prefs):
    core = make_core()
    run(core, 'audio.start')
    assert core.get('pref.audio.pending') == []
    version = core.poll()['version']

    core.set('pref.audio.device', 'Fake Duplex')
    core.set('pref.audio.buffer_ms', 10.0)
    core.set('pref.audio.sample_rate', 48000)
    core.set('pref.audio.synth_rate', 22050)

    assert core.get('pref.audio.pending') == [
        'pref.audio.device', 'pref.audio.buffer_ms', 'pref.audio.sample_rate',
        'pref.audio.synth_rate']
    changes = core.poll(version)['changes']
    assert changes['pref.audio.device'] == 'Fake Duplex'
    assert changes['pref.audio.pending'][-1] == 'pref.audio.synth_rate'
    file = saved(prefs)
    assert (file['audio_output_device'], file['audio_buffer_ms'], file['audio_sample_rate'],
            file['synth_sample_rate']) == ('Fake Duplex', 10.0, 48000, 22050)
    assert core.get('audio.sample_rate') == 44100  # the stream is untouched

    result = run(core, 'audio.apply')
    assert result['sample_rate'] == 48000 and result['device'] == 'Fake Duplex'
    assert core.get('audio.synth_rate') == 22050
    assert core.get('audio.block_size') == 480
    assert core.get('pref.audio.pending') == []


def test_same_as_output_follows_the_output_rate(make_core, prefs):
    core = make_core()
    assert core.get('pref.audio.synth_rate') == 0
    core.set('pref.audio.sample_rate', 48000)
    assert saved(prefs)['synth_sample_rate'] == 48000
    assert core.get('pref.audio.synth_rate') == 0
    core.set('pref.audio.synth_rate', 11025)
    core.set('pref.audio.sample_rate', 44100)
    assert core.get('pref.audio.synth_rate') == 11025  # a lower rate stays
    core.set('pref.audio.synth_rate', 0)
    assert saved(prefs)['synth_sample_rate'] == 44100


def test_audio_apply_with_settings_still_saves_them(make_core, prefs):
    core = make_core()
    run(core, 'audio.start')
    run(core, 'audio.apply', device='Fake Duplex', sample_rate=48000, synth_rate=0,
        buffer_ms=50.0, mono=True)
    assert core.get('pref.audio.device') == 'Fake Duplex'
    assert core.get('pref.audio.buffer_ms') == 50.0
    assert core.get('pref.audio.mono') is True and core.get('audio.mono') is True
    assert core.get('pref.audio.pending') == []
    with pytest.raises(AssertionError):
        run(core, 'audio.apply', rate=1)  # an unknown setting is an error event


def test_mono_applies_at_once(make_core, backend, prefs):
    core = make_core()
    run(core, 'audio.start')
    version = core.poll()['version']
    core.set('pref.audio.mono', True)
    assert saved(prefs)['audio_mono'] is True
    pull_until(core, backend.stream, lambda: core.synth.mono)
    assert core.poll(version)['changes']['audio.mono'] is True
    assert core.get('pref.audio.pending') == []


def test_smoothing_applies_to_every_channel(make_core, prefs):
    core = make_core(audio_backend=None)
    core.set('pref.smoothing_ms', 12.0)
    assert [ch.get_smoothing_time() for ch in core.synth.channels] == [12.0] * 8
    assert saved(prefs)['param_smoothing_ms'] == 12.0
    core.set('pref.smoothing_ms', 500)
    assert core.get('pref.smoothing_ms') == 100.0  # clamped to the range


def test_the_input_device_and_ai_settings_are_saved(make_core, prefs):
    core = make_core(audio_backend=None)
    core.set('pref.audio.input_device', 'Fake In')
    core.set('pref.ai.pattern_model', '/models/patterns.pt')
    core.set('pref.ai.pattern_temperature', 1.3)
    core.set('pref.ai.patch_model', '')
    core.set('pref.ai.patch_temperature', 9)
    file = saved(prefs)
    assert file['audio_input_device'] == 'Fake In'
    assert file['drum_generator_pattern_model_path'] == '/models/patterns.pt'
    assert file['drum_generator_pattern_temperature'] == 1.3
    assert file['drum_generator_model_path'] is None
    assert file['drum_generator_patch_temperature'] == 3.0
    core.set('pref.audio.input_device', None)
    assert saved(prefs)['audio_input_device'] is None


# ---------------------------------------------------------------------------
# devices
# ---------------------------------------------------------------------------

def test_supported_rates_of_a_device(make_core, backend):
    core = make_core()
    assert run(core, 'audio.rates', device='Fake Out') == {
        'device': 'Fake Out', 'rates': [48000, 44100, 22050]}
    backend.supported_rates = set()
    assert run(core, 'audio.rates')['rates'] == [96000, 48000, 44100, 32000, 22050, 11025, 8000]


def test_rescan_lists_the_devices_again(make_core, backend):
    core = make_core()
    backend.devices.append({'name': 'New Out', 'max_output_channels': 2,
                            'max_input_channels': 0, 'default_samplerate': 44100.0})
    version = core.poll()['version']
    result = run(core, 'audio.rescan')
    assert result['output_devices'] == ['Fake Out', 'Fake Duplex', 'New Out']
    assert core.describe('pref.audio.device')['labels'][-1] == 'New Out'
    assert core.poll(version)['changes']['audio.output_devices'][-1] == 'New Out'


# ---------------------------------------------------------------------------
# folders, front-end values
# ---------------------------------------------------------------------------

def test_the_preset_folder_must_exist(make_core, prefs, tmp_path):
    core = make_core(audio_backend=None)
    with pytest.raises(ValueError):
        core.set('pref.preset_folder', str(tmp_path / 'nowhere'))
    folder = tmp_path / 'kits'
    folder.mkdir()
    (folder / 'a.json').write_text('{}')
    version = core.poll()['version']
    core.set('pref.preset_folder', str(folder))
    assert saved(prefs)['preset_folder'] == str(folder)
    changes = core.poll(version)['changes']
    assert changes['pref.preset_folder'] == str(folder)
    assert changes['preset.files'] == ['a.json']


def test_front_end_values_are_kept(make_core, prefs):
    core = make_core(audio_backend=None)
    assert core.get('pref.ui.strip_ctrl') is None
    core.set('pref.ui.strip_ctrl', 'reverb_mix')
    core.set('pref.ui.rack', {'open': False})
    assert saved(prefs)['ui_strip_ctrl'] == 'reverb_mix'
    assert core.get('pref.ui.rack') == {'open': False}
    with pytest.raises(KeyError):
        core.get('pref.ui.Bad-Name')


# ---------------------------------------------------------------------------
# file compatibility
# ---------------------------------------------------------------------------

def test_preferences_round_trip_through_the_file(make_core, prefs):
    core = make_core(audio_backend=None)
    core.set('pref.audio.sample_rate', 48000)
    core.set('pref.audio.mono', True)
    core.set('pref.smoothing_ms', 40.0)
    core.set('pref.ai.pattern_temperature', 0.9)
    core.set('midi.base_note', 48)

    reread = PreferencesManager()  # the fixture's folder: what the next launch reads
    assert reread.get('audio_sample_rate') == 48000
    assert reread.get('audio_mono') is True
    assert reread.get('param_smoothing_ms') == 40.0
    assert reread.get('drum_generator_pattern_temperature') == 0.9
    assert reread.get('midi_base_note') == 48
    other = make_core(preferences=reread, audio_backend=None)
    assert other.get('pref.audio.sample_rate') == 48000
    assert other.synth.mono is True and other.get('audio.synth_rate') == 48000
    assert other.synth.channels[0].get_smoothing_time() == 40.0


def test_a_file_from_an_older_version_is_read(make_core, prefs, tmp_path):
    folder = tmp_path / 'kits'
    folder.mkdir()
    old = {  # keys an older version wrote; no synth rate, mono, AI or pitch bend keys
        'preset_folder': str(folder),
        'last_preset': None,
        'window_width': 1200,
        'window_height': 700,
        'recent_files': [],
        'audio_output_device': 'Fake Duplex',
        'audio_buffer_ms': 23.8,
        'audio_sample_rate': 48000,
        'midi_input_device': None,
        'midi_base_note': 40,
        'midi_enabled': False,
        'midi_cc_mappings': {'1': 'osc_freq', '74': 'noise_freq'},
    }
    with open(prefs.prefs_file, 'w', encoding='utf-8') as f:
        json.dump(old, f)
    manager = PreferencesManager()
    core = make_core(preferences=manager, audio_backend=None)

    assert core.get('pref.audio.device') == 'Fake Duplex'
    assert core.get('pref.audio.sample_rate') == 48000
    assert core.get('pref.audio.synth_rate') == 44100  # the old default synth rate
    assert core.get('audio.synth_rate') == 44100
    assert core.get('pref.audio.mono') is False
    assert core.get('pref.smoothing_ms') == 30.0
    assert core.get('pref.ai.pattern_model') is None
    assert core.get('pref.ai.pattern_temperature') == 0.7
    assert core.get('pref.preset_folder') == str(folder)
    assert core.get('midi.base_note') == 40
    assert core.get('midi.cc_map') == {1: 'selected.osc.freq', 74: 'selected.noise.freq'}

    core.set('pref.ai.pattern_temperature', 1.1)
    file = saved(manager)
    assert file['window_width'] == 1200 and file['audio_sample_rate'] == 48000
    assert file['drum_generator_pattern_temperature'] == 1.1
    # The CC map is saved keyed by address since slice 3
    assert file['midi_cc_mappings'] == {'1': 'selected.osc.freq', '74': 'selected.noise.freq'}
