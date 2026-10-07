"""
The AI generators module of the app core (slice 8): drum patch candidates
tried on the live channels as a preview overlay, keep (one undo step) and
revert, AI patterns with loop and bank previews, replace patterns, AI
randomize, model loading, the worker subprocess and its restart.

Every test drives the core through its interface (set / get / act / poll)
with a fake worker (tests/fake_ai_worker.py) speaking the worker's JSON-lines
protocol, so no torch is needed. One end-to-end test runs the real worker
and models and skips without them.
"""

import json
import os
import pathlib
import subprocess
import sys
import time

import pytest

from pythonic.app import AppCore
from pythonic.app.ai import BUNDLED_MODELS, torch_installed

ROOT = pathlib.Path(__file__).resolve().parent.parent
FAKE_WORKER = str(ROOT / 'tests' / 'fake_ai_worker.py')
SOUND = ('osc.freq', 'osc.decay', 'osc.wave', 'noise.freq', 'mix.level', 'mix.osc_noise')


@pytest.fixture
def models(tmp_path):
    folder = tmp_path / 'models'
    folder.mkdir()
    for name in ('patch.pt', 'pattern.pt', 'broken.pt'):
        (folder / name).write_bytes(b'fake')
    return folder


@pytest.fixture
def make_core(prefs, models, tmp_path):
    cores = []

    def make(*extra, worker=True, **kwargs):
        kwargs.setdefault('preferences', prefs)
        kwargs.setdefault('audio_backend', None)  # no stream: changes apply inline
        kwargs.setdefault('stall_timeout', None)
        kwargs.setdefault('midi_backend', None)
        if worker:
            kwargs.setdefault('ai_worker', [sys.executable, FAKE_WORKER, *extra])
        core = AppCore(**kwargs)
        cores.append(core)
        core.set('pref.ai.patch_model', str(models / 'patch.pt'))
        core.set('pref.ai.pattern_model', str(models / 'pattern.pt'))
        return core

    yield make
    for core in cores:
        core.close()


def act(core, verb, timeout=20.0, **args):
    """Run a verb and return its final event."""
    return core.wait(core.act(verb, **args), timeout)


def run(core, verb, **args):
    event = act(core, verb, **args)
    assert event['status'] == 'done', event
    return event['result']


def sound(core, channel):
    return {s: core.get(f'ch{channel}.{s}') for s in SOUND} | {'name': core.get(f'ch{channel}.name')}


def lane(core, channel):
    return {f: core.get(f'ai.ch{channel}.{f}')
            for f in ('candidates', 'candidate', 'name', 'trying', 'generating')}


def triggers(core, pattern, channel=1):
    return core.get(f'pattern.{pattern}.ch{channel}.trig')


def settle(core, timeout=5.0):
    """Wait until the action thread has run what is queued (AI replies)."""
    core.wait(core.act('preset.refresh'), timeout)


# ====================================================================== availability
def test_without_the_ml_extras_the_ai_is_unavailable_and_says_how_to_install(make_core):
    core = make_core(worker=False, ai_worker=None)
    assert core.get('ai.available') is False
    assert core.get('ai.state') == 'unavailable'
    command = core.get('ai.install_command')
    assert 'pip install' in command and 'requirements-ml.txt' in command
    event = act(core, 'ai.generate', channel=1)
    assert event['status'] == 'error' and 'pip install' in event['error']


def test_the_default_worker_is_available_when_torch_is_installed(prefs):
    core = AppCore(preferences=prefs, audio_backend=None, stall_timeout=None, midi_backend=None)
    try:
        assert core.get('ai.available') is torch_installed()
        assert core.get('ai.models')['patch']['bundled'] == os.path.isfile(BUNDLED_MODELS['patch'])
        assert core.ai.worker.pid is None  # started on first use only
    finally:
        core.close()


def test_the_gui_process_never_imports_torch(prefs, tmp_path):
    code = (
        "import sys\n"
        "from pythonic.app import AppCore\n"
        "import gui.main_window, gui.drum_generator_dialog\n"
        "core = AppCore(audio_backend=None, midi_backend=None, stall_timeout=None,\n"
        f"              ai_worker=[sys.executable, {FAKE_WORKER!r}])\n"
        "core.set('pref.ai.patch_model', sys.argv[1])\n"
        "event = core.wait(core.act('ai.generate', channel=1), 20)\n"
        "assert event['status'] == 'done', event\n"
        "core.close()\n"
        "assert 'torch' not in sys.modules\n"
        "print('ok')\n"
    )
    model = tmp_path / 'p.pt'
    model.write_bytes(b'x')
    env = dict(os.environ, HOME=str(tmp_path), XDG_CONFIG_HOME=str(tmp_path))
    out = subprocess.run([sys.executable, '-c', code, str(model)], cwd=ROOT, env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith('ok')


# ====================================================================== generating and trying
def test_generating_a_lane_shows_generating_then_tries_candidate_1(make_core, tmp_path):
    gate = tmp_path / 'gate'
    gate.touch()
    core = make_core('--gate', str(gate))
    before = sound(core, 1)
    action = core.act('ai.generate', channel=1, candidates=4, seed=5)
    deadline = time.monotonic() + 5
    while not core.get('ai.ch1.generating') and time.monotonic() < deadline:
        time.sleep(0.01)
    assert core.get('ai.ch1.generating') is True
    assert core.get('ai.state') == 'generating'
    assert sound(core, 1) == before
    changes = core.poll()['changes']
    assert changes.get('ai.ch1.generating') is True
    gate.unlink()
    event = core.wait(action, 10)
    assert event['status'] == 'done'
    assert event['result'] == {'channels': [1], 'failed': []}
    assert lane(core, 1) == {'candidates': 4, 'candidate': 1, 'name': 'BD 1', 'trying': True,
                             'generating': False}
    assert core.get('ai.state') == 'idle'
    assert core.get('ai.tried') == [1]
    assert core.get('ch1.name') == 'BD 1'
    assert core.get('ch1.osc.freq') == pytest.approx(100.0)
    assert core.get('undo.can_undo') is False  # trying is not an undo step
    changes = core.poll()['changes']
    assert changes['ch1.name'] == 'BD 1' and changes['ai.ch1.trying'] is True


def test_the_lane_type_follows_the_channel_then_sticks_once_set(make_core):
    core = make_core()
    assert core.get('ai.ch1.type') == 'bd'  # the init kit's kick
    assert core.describe('ai.ch1.type')['labels'][:3] == ['bass', 'bd', 'blip']
    assert len(core.describe('ai.ch3.type')['labels']) == 18
    core.set('ai.ch2.type', 'clap')
    assert core.get('ai.ch2.type') == 'clap'
    run(core, 'ai.generate', channel=2)
    assert core.get('ai.ch2.name') == 'CLAP 1'
    run(core, 'ai.generate', channel=3, type='shaker', candidates=2)
    assert core.get('ai.ch3.type') == 'shaker' and core.get('ch3.name') == 'SHAKER 1'


def test_generate_all_eight_tries_candidate_1_on_every_lane(make_core):
    core = make_core()
    result = run(core, 'ai.generate', candidates=3)
    assert result['channels'] == list(range(1, 9))
    assert core.get('ai.tried') == list(range(1, 9))
    for c in range(1, 9):
        assert lane(core, c)['candidates'] == 3
        assert core.get(f'ch{c}.name') == core.get(f'ai.ch{c}.name')


def test_trying_steps_through_candidates_and_untry_restores_the_old_sound(make_core):
    core = make_core()
    before = sound(core, 1)
    params = core.synth.channels[0].get_parameters()
    run(core, 'ai.generate', channel=1, candidates=3)
    assert run(core, 'ai.try', channel=1, candidate=3)['name'] == 'BD 3'
    assert core.get('ch1.osc.freq') == pytest.approx(300.0)
    assert run(core, 'ai.try', channel=1, step=1)['candidate'] == 1  # wraps
    assert run(core, 'ai.try', channel=1, step=-1)['candidate'] == 3
    assert core.get('ai.ch1.candidate') == 3
    run(core, 'ai.untry', channel=1)
    assert core.get('ai.ch1.trying') is False and core.get('ai.ch1.candidate') == 3
    assert sound(core, 1) == before
    assert core.synth.channels[0].get_parameters() == params
    assert run(core, 'ai.try', channel=1)['name'] == 'BD 3'  # the current one again
    assert act(core, 'ai.try', channel=1, candidate=9)['status'] == 'error'
    assert act(core, 'ai.try', channel=2)['status'] == 'error'  # no candidates


# ====================================================================== keep, revert, edits
def test_keep_commits_the_tried_sounds_as_one_undo_step(make_core):
    core = make_core()
    before = [sound(core, c) for c in (1, 2)]
    run(core, 'ai.generate', channel=1)
    run(core, 'ai.generate', channel=2)
    tried = [sound(core, c) for c in (1, 2)]
    assert run(core, 'ai.keep') == {'kept': [1, 2]}
    assert core.get('ai.tried') == []
    assert [sound(core, c) for c in (1, 2)] == tried
    assert core.get('undo.can_undo') is True
    assert run(core, 'undo')['label'] == 'keep AI sounds'
    assert [sound(core, c) for c in (1, 2)] == before
    assert core.get('undo.can_undo') is False  # one step
    run(core, 'redo')
    assert [sound(core, c) for c in (1, 2)] == tried


def test_keep_some_lanes_then_revert_the_others(make_core):
    core = make_core()
    before2 = sound(core, 2)
    run(core, 'ai.generate', channel=1)
    run(core, 'ai.generate', channel=2)
    run(core, 'ai.generate', channel=3)
    run(core, 'ai.untry', channel=3)  # has candidates, not trying
    assert run(core, 'ai.keep', channels=[1, 3])['kept'] == [1, 3]  # 3 is tried first
    assert core.get('ch3.name') == 'CH 1'
    assert core.get('ai.tried') == [2]
    assert run(core, 'ai.revert') == {'reverted': [2], 'preview': False}
    assert sound(core, 2) == before2
    assert core.get('ch1.name') == 'BD 1'
    run(core, 'undo')  # the keep: both kept lanes back, lane 2 untouched
    assert core.get('ch1.name') != 'BD 1' and core.get('ch3.name') != 'CH 1'


def test_revert_restores_the_old_sounds_without_an_undo_step(make_core):
    core = make_core()
    before = [sound(core, c) for c in range(1, 9)]
    run(core, 'ai.generate')
    run(core, 'ai.revert')
    assert [sound(core, c) for c in range(1, 9)] == before
    assert core.get('ai.tried') == []
    assert core.get('undo.can_undo') is False
    assert lane(core, 4)['candidates'] == 8  # the candidates stay


def test_knob_edits_while_trying_go_into_the_tried_sound(make_core):
    core = make_core()
    original = core.get('ch1.osc.decay')
    run(core, 'ai.generate', channel=1)
    core.set('ch1.osc.decay', 777.0)
    assert core.get('ch1.osc.decay') == pytest.approx(777.0)
    assert core.get('undo.can_undo') is False  # not a step of its own
    run(core, 'ai.keep')
    assert core.get('ch1.osc.decay') == pytest.approx(777.0)
    run(core, 'undo')  # keep and the edit are one step
    assert core.get('ch1.osc.decay') == pytest.approx(original)
    assert core.get('undo.can_undo') is False
    # Revert drops an edit made while trying; edits elsewhere stay undoable
    run(core, 'ai.try', channel=1)
    core.set('ch1.osc.decay', 555.0)
    core.set('ch2.osc.decay', 444.0)
    run(core, 'ai.revert')
    assert core.get('ch1.osc.decay') == pytest.approx(original)
    assert core.get('ch2.osc.decay') == pytest.approx(444.0)
    assert run(core, 'undo')['label'] == 'ch2.osc.decay'


def test_a_preset_change_reverts_the_trial_first(make_core):
    core = make_core()
    before = sound(core, 1)
    run(core, 'ai.generate', channel=1)
    run(core, 'preset.initialize')
    assert core.get('ai.tried') == []
    run(core, 'undo')  # back to the sounds before the trial, not the candidate
    assert sound(core, 1) == before
    assert core.get('ai.ch1.trying') is False


def test_undo_of_an_earlier_edit_on_a_trying_channel_reverts_the_trial(make_core):
    core = make_core()
    original = core.get('ch1.osc.decay')
    core.set('ch1.osc.decay', 900.0)
    run(core, 'ai.generate', channel=1)
    run(core, 'undo')
    assert core.get('ai.ch1.trying') is False
    assert core.get('ch1.osc.decay') == pytest.approx(original)
    assert core.get('ch1.name') != 'BD 1'


def test_clear_reverts_and_forgets_the_candidates(make_core):
    core = make_core()
    before = sound(core, 1)
    run(core, 'ai.generate', channel=1, type='sd')
    run(core, 'ai.generate_patterns')
    run(core, 'ai.clear')
    assert sound(core, 1) == before
    assert lane(core, 1)['candidates'] == 0 and core.get('ai.ch1.type') == 'bd'
    assert core.get('ai.bank') == 'none'


# ====================================================================== patterns
def test_generate_patterns_makes_a_bank_of_twelve(make_core, tmp_path):
    log = tmp_path / 'log'
    core = make_core('--log', str(log))
    run(core, 'ai.generate', channel=1)
    core.set('global.tempo', 133)
    assert run(core, 'ai.generate_patterns', temperature=0.5, seed=7) == {'patterns': 12}
    assert core.get('ai.bank') == 'ready'
    request = [json.loads(line) for line in log.read_text().splitlines()][-1]
    assert request['op'] == 'patterns' and request['n'] == 12
    assert request['tempo'] == 133 and request['swing'] == 0.0 and request['seed'] == 7
    assert request['temperature'] == 0.5
    assert request['raw_patches'][0]['Name'] == 'BD 1'  # the tried kit
    run(core, 'ai.clear_patterns')
    assert core.get('ai.bank') == 'none'


def test_loop_preview_plays_the_ai_version_and_stop_restores_the_preset(make_core):
    core = make_core()
    core.set('pattern.A.ch1.step2.trig', True)
    preset = triggers(core, 'A')
    run(core, 'ai.generate_patterns')
    result = run(core, 'ai.pattern_try', mode='loop')
    assert result == {'preview': 'loop', 'pattern': 'A', 'bank': True}
    assert core.get('ai.preview') == 'loop'
    assert core.poll()['transport']['playing'] is True
    assert triggers(core, 'A') == [True] * 16  # bank pattern A, channel 1: every step
    can_undo = core.get('undo.can_undo')
    core.set('pattern.A.ch2.step3.trig', True)  # an edit on the preview
    assert core.get('undo.can_undo') == can_undo
    version = core.poll()['version']
    run(core, 'transport.stop')
    assert core.get('ai.preview') == 'off'
    assert triggers(core, 'A') == preset
    assert core.get('pattern.A.ch2.step3.trig') is False
    changes = core.poll(version)['changes']
    assert changes['ai.preview'] == 'off' and changes['pattern.A.ch1.trig'] == preset


def test_bank_preview_chains_all_twelve_from_a_and_ends_with_revert(make_core):
    core = make_core()
    core.set('pattern.C.ch1.step1.trig', True)
    preset = [triggers(core, p) for p in 'ABCDEFGHIJKL']
    run(core, 'pattern.select', pattern='C')
    run(core, 'ai.generate_patterns')
    run(core, 'ai.pattern_try', mode='bank')
    transport = core.poll()['transport']
    assert transport['playing'] and transport['playing_pattern'] == 0
    assert transport['chain'] == list(range(12))
    assert all(core.get(f'pattern.{p}.chained') for p in 'ABCDEFGHIJK')
    assert run(core, 'ai.revert') == {'reverted': [], 'preview': True}
    assert core.poll()['transport']['playing'] is False
    assert [triggers(core, p) for p in 'ABCDEFGHIJKL'] == preset
    assert not any(core.get(f'pattern.{p}.chained') for p in 'ABCDEFGHIJK')


def test_previews_without_a_bank_play_the_preset_patterns(make_core):
    core = make_core()
    core.set('pattern.A.ch1.step1.trig', True)
    preset = triggers(core, 'A')
    run(core, 'ai.pattern_try', mode='loop')
    assert core.get('ai.preview') == 'loop' and triggers(core, 'A') == preset
    core.set('pattern.A.ch1.step5.trig', True)  # the real pattern: an undoable edit
    assert run(core, 'undo')['label'] == 'pattern.A.ch1.step5.trig'
    run(core, 'ai.pattern_try', mode='bank', bank=False)
    assert core.get('pattern.A.chained') is True
    run(core, 'ai.pattern_try', mode=None)
    assert core.get('ai.preview') == 'off' and core.get('pattern.A.chained') is False
    assert core.poll()['transport']['playing'] is False


def test_replace_patterns_keeps_the_sounds_and_the_bank_as_one_step(make_core):
    core = make_core()
    before = sound(core, 1)
    core.set('pattern.B.ch1.step2.trig', True)
    preset = [triggers(core, p) for p in 'ABCDEFGHIJKL']
    run(core, 'ai.generate', channel=1)
    assert act(core, 'ai.replace_patterns')['status'] == 'error'  # no bank yet
    run(core, 'ai.generate_patterns')
    run(core, 'ai.pattern_try', mode='bank')
    tried = sound(core, 1)
    assert run(core, 'ai.replace_patterns') == {'kept': [1], 'patterns': 12}
    assert core.get('ai.preview') == 'off' and core.get('ai.tried') == []
    assert triggers(core, 'B') == [s % 2 == 0 for s in range(16)]
    assert not core.get('pattern.A.chained')  # the bank's chains, not the preview's
    assert sound(core, 1) == tried
    run(core, 'transport.stop')
    assert triggers(core, 'B') == [s % 2 == 0 for s in range(16)]  # no restore any more
    assert run(core, 'undo')['label'] == 'AI sounds and patterns'
    assert sound(core, 1) == before
    assert [triggers(core, p) for p in 'ABCDEFGHIJKL'] == preset


def test_ai_randomize_replaces_a_pattern_or_a_lane_as_one_step(make_core, tmp_path):
    log = tmp_path / 'log'
    core = make_core('--log', str(log))
    core.set('pattern.D.ch2.step1.trig', True)
    core.set('pattern.D.ch3.step2.trig', True)
    core.set('global.swing', 0.25)
    assert run(core, 'ai.randomize_pattern', pattern='D') == {'pattern': 'D', 'channel': None}
    assert triggers(core, 'D', 2) == [s % 2 == 0 for s in range(16)]
    request = [json.loads(line) for line in log.read_text().splitlines()][-1]
    assert request['n'] == 1 and request['swing'] == 0.25
    assert request['temperature'] == pytest.approx(core.get('pref.ai.pattern_temperature'))
    assert run(core, 'undo')['label'] == 'AI pattern'
    assert core.get('pattern.D.ch3.step2.trig') is True
    run(core, 'ai.randomize_pattern', pattern='D', channel=3)
    assert triggers(core, 'D', 3) == [s % 3 == 0 for s in range(16)]
    assert core.get('pattern.D.ch2.step1.trig') is True  # other lanes stay
    assert run(core, 'undo')['label'] == 'AI channel'


def test_no_pattern_model_is_an_error_with_advice(make_core, models):
    core = make_core()
    core.set('pref.ai.pattern_model', str(models / 'missing.pt'))
    saved = BUNDLED_MODELS['pattern']
    BUNDLED_MODELS['pattern'] = str(models / 'not-bundled.pt')
    try:
        assert core.get('ai.models')['pattern']['status'] == 'missing'
        event = act(core, 'ai.randomize_pattern')
        assert event['status'] == 'error' and 'pattern model' in event['error']
    finally:
        BUNDLED_MODELS['pattern'] = saved


# ====================================================================== models and the worker
def test_load_model_reports_its_status_and_saves_an_explicit_path(make_core, models, tmp_path):
    gate = tmp_path / 'gate'
    gate.touch()
    core = make_core('--gate', str(gate))
    other = models / 'other.pt'
    other.write_bytes(b'x')
    assert core.get('ai.models')['patch']['status'] == 'unloaded'
    action = core.act('ai.load_model', kind='patch', path=str(other))
    deadline = time.monotonic() + 5
    while core.get('ai.state') != 'loading' and time.monotonic() < deadline:
        time.sleep(0.01)
    assert core.get('ai.state') == 'loading'
    gate.unlink()
    result = core.wait(action, 10)['result']
    assert result == {'kind': 'patch', 'path': str(other), 'sampling': 'prior'}
    assert core.get('pref.ai.patch_model') == str(other)
    assert core.get('ai.models')['patch'] == {
        'path': str(other), 'bundled': os.path.isfile(BUNDLED_MODELS['patch']),
        'status': 'loaded', 'error': None, 'sampling': 'prior'}
    event = act(core, 'ai.load_model', kind='pattern', path=str(models / 'broken.pt'))
    assert event['status'] == 'error' and 'broken' in event['error']
    assert core.get('pref.ai.pattern_model') == str(models / 'pattern.pt')  # not saved
    assert act(core, 'ai.load_model', kind='patch', path='/no/such.pt')['status'] == 'error'


def test_a_crashed_worker_fails_its_requests_and_restarts(make_core):
    core = make_core('--crash', 'oh')
    run(core, 'ai.generate', channel=1)
    first = core.ai.worker.pid
    core.set('ai.ch2.type', 'oh')
    event = act(core, 'ai.generate', channel=2)
    assert event['status'] == 'error' and 'stopped' in event['error']
    assert core.get('ai.ch2.error') and core.get('ai.ch2.generating') is False
    assert core.get('ai.state') == 'idle'
    run(core, 'ai.generate', channel=3)  # a new worker
    assert core.ai.worker.pid != first and core.ai.worker.starts == 2
    assert core.get('ai.ch3.name') == 'CH 1'


def test_closing_the_core_stops_the_worker(make_core):
    core = make_core()
    run(core, 'ai.generate', channel=1)
    proc = core.ai.worker._proc
    assert proc.poll() is None
    core.close()
    assert proc.poll() is not None


def test_install_runs_the_command_in_the_background(make_core):
    core = make_core(ai_install=[sys.executable, '-c', 'print("installed fine")'])
    result = run(core, 'ai.install')
    assert result == {'installed': True, 'output': 'installed fine'}
    assert core.get('ai.installing') is False
    failing = make_core(ai_install=[sys.executable, '-c', 'import sys; sys.exit("nope")'])
    event = act(failing, 'ai.install')
    assert event['status'] == 'error' and 'nope' in event['error']


# ====================================================================== the real worker
def _real_models():
    return torch_installed() and all(
        os.path.isfile(p) and os.path.getsize(p) > 1_000_000 for p in BUNDLED_MODELS.values())


@pytest.mark.skipif(not _real_models(), reason='torch or the bundled models are missing')
def test_the_real_worker_generates_candidates_and_patterns(prefs):
    core = AppCore(preferences=prefs, audio_backend=None, stall_timeout=None, midi_backend=None)
    try:
        event = act(core, 'ai.generate', timeout=300, channel=1, candidates=2, seed=1)
        assert event['status'] == 'done', event
        assert core.get('ai.ch1.candidates') == 2 and core.get('ai.ch1.trying') is True
        assert core.get('ch1.name') == core.get('ai.ch1.name') == 'BD'
        assert core.get('ai.models')['patch']['status'] == 'loaded'
        event = act(core, 'ai.randomize_pattern', timeout=300, pattern='B')
        assert event['status'] == 'done', event
        assert core.get('pattern.B.empty') is False
    finally:
        core.close()
