"""The bridge's slots and frames, called from Python over the fake core."""

import json
import re
import sys
from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QWidget

from pythonic.web.bridge import FRAME_BUDGET_MS, Bridge, to_json
from tests.web.page import choose_in_dialog, same_path

ROOT = Path(__file__).resolve().parent.parent.parent


def call(bridge, slot, payload):
    return json.loads(getattr(bridge, slot)(json.dumps(payload)))


def test_get_returns_values_and_errors(fake_core):
    bridge = Bridge(fake_core)
    reply = call(bridge, 'get', ['global.tempo', 'nope.nothing'])
    assert reply['values'] == {'global.tempo': 120}
    assert reply['errors'] == {'nope.nothing': 'unknown address: nope.nothing'}


def test_set_coerces_like_the_core_and_reports_errors(fake_core):
    bridge = Bridge(fake_core)
    reply = call(bridge, 'set', [{'address': 'global.tempo', 'value': 999, 'burst': True},
                                 {'address': 'audio.running', 'value': False},
                                 {'address': 'nope', 'value': 1}])
    assert fake_core.get('global.tempo') == 300  # clamped to the range
    assert fake_core.calls[0] == ('set', 'global.tempo', 300,
                                  {'edit_all': None, 'burst': True, 'record': True})
    assert set(reply['errors']) == {'audio.running', 'nope'}
    assert call(bridge, 'set', {'address': 'global.swing', 'value': 0.5}) == {'errors': {}}


def test_act_returns_an_id_and_the_event_arrives_in_a_frame(fake_core, qtbot):
    bridge = Bridge(fake_core)
    frames = []
    bridge.frame.connect(lambda text: frames.append(json.loads(text)))
    reply = call(bridge, 'act', {'verb': 'transport.toggle'})
    bridge.tick()
    event = frames[-1]['events'][0]
    assert event['id'] == reply['id'] and event['status'] == 'done'
    assert frames[-1]['transport']['playing'] is True
    assert 'error' in call(bridge, 'act', {'args': {}})


def test_describe_by_prefix_or_list(fake_core):
    bridge = Bridge(fake_core)
    tempo = call(bridge, 'describe', ['global.tempo'])['addresses']['global.tempo']
    assert (tempo['kind'], tempo['minimum'], tempo['maximum'], tempo['unit']) == ('int', 1, 300,
                                                                                  'BPM')
    everything = call(bridge, 'describe', '')['addresses']
    assert set(everything) == set(fake_core.meta)
    # device lists come as describe() options
    assert call(bridge, 'describe', 'pref.audio.device')['addresses'][
        'pref.audio.device']['labels'] == ['Fake Out', 'Fake Duplex']


def test_gesture_brackets_reach_the_core(fake_core):
    bridge = Bridge(fake_core)
    bridge.gesture('begin')
    bridge.gesture('end')
    assert fake_core.calls == [('gesture', 'begin'), ('gesture', 'end')]


def test_malformed_requests_answer_an_error(fake_core):
    bridge = Bridge(fake_core)
    for slot in ('get', 'set', 'act', 'describe', 'fileDialog'):
        assert 'error' in json.loads(getattr(bridge, slot)('{not json'))


def test_tick_pushes_changes_once_and_skips_identical_frames(fake_core):
    bridge = Bridge(fake_core)
    frames = []
    bridge.frame.connect(lambda text: frames.append(json.loads(text)))
    bridge.tick()
    fake_core.post_change('global.tempo', 99)
    bridge.tick()
    bridge.tick()  # nothing new: no frame
    assert len(frames) == 2
    assert frames[1]['changes'] == {'global.tempo': 99}
    assert bridge.stats.count == 3
    assert bridge.stats.max_ms < FRAME_BUDGET_MS


def test_timer_pushes_frames_at_the_frame_rate(fake_core, qtbot):
    bridge = Bridge(fake_core)
    frames = []
    bridge.frame.connect(frames.append)
    bridge.start()
    try:
        for i in range(5):
            fake_core.post_change('global.tempo', 100 + i)
            qtbot.wait(20)
    finally:
        bridge.stop()
    assert len(frames) >= 4 and bridge.stats.count >= 5


def test_file_dialog_opens_without_blocking_and_reports_the_path(fake_core, qtbot, tmp_path):
    parent = QWidget()
    qtbot.addWidget(parent)
    bridge = Bridge(fake_core, dialog_parent=parent)
    results = []
    bridge.dialog.connect(lambda text: results.append(json.loads(text)))
    target = tmp_path / 'kit.mtpreset'
    target.write_text('x')
    reply = call(bridge, 'fileDialog', {'mode': 'open', 'title': 'Open preset',
                                        'filters': ['Presets (*.mtpreset *.json)'],
                                        'folder': str(tmp_path)})
    (dialog,) = bridge.open_dialogs()  # open() returned at once: the dialog waits
    assert dialog.fileMode() == QFileDialog.FileMode.ExistingFile
    choose_in_dialog(qtbot, dialog, target)
    qtbot.waitUntil(lambda: len(results) == 1, timeout=3000)
    assert results[0]['id'] == reply['id'] and same_path(results[0]['path'], target)

    reply = call(bridge, 'fileDialog', {'mode': 'save', 'suffix': '.json'})
    (dialog,) = bridge.open_dialogs()
    assert dialog.acceptMode() == QFileDialog.AcceptMode.AcceptSave
    assert dialog.defaultSuffix() == 'json'
    dialog.reject()
    qtbot.waitUntil(lambda: len(results) == 2, timeout=3000)
    assert results[1] == {'id': reply['id'], 'path': None}


def test_resync_resends_unchanged_readouts(fake_core):
    bridge = Bridge(fake_core)
    frames = []
    bridge.frame.connect(frames.append)
    bridge.tick()
    bridge.tick()
    bridge.resync('null')
    bridge.tick()
    assert len(frames) == 2


def test_resize_window_follows_the_stage_height_at_the_panel_scale(open_panel, fake_core, qtbot):
    page = open_panel(fake_core)
    window = page.window
    bridge = window.bridge
    assert call(bridge, 'resizeWindow', {'from': 1000, 'to': 700}) == {'size': [1280, 560]}
    assert (window.width(), window.height()) == (1280, 560)
    assert (window.minimumWidth(), window.minimumHeight()) == (1280, 560)  # still scale 0.8
    assert call(bridge, 'resizeWindow', {'from': 700, 'to': 1000}) == {'size': [1280, 800]}
    assert (window.minimumWidth(), window.minimumHeight()) == (1280, 800)
    window.resize(1600, 1000)  # scale 1: the rack is 300 window pixels
    qtbot.waitUntil(lambda: window.view.height() == 1000)
    assert call(bridge, 'resizeWindow', {'from': 1000, 'to': 700}) == {'size': [1600, 700]}
    assert 'error' in call(bridge, 'resizeWindow', {'from': 'x'})


def test_resize_window_without_a_window_does_nothing(fake_core):
    assert call(Bridge(fake_core), 'resizeWindow', {'from': 1000, 'to': 700}) == {'size': None}


def test_trigger_hits_a_channel_now(fake_core):
    bridge = Bridge(fake_core)
    assert call(bridge, 'trigger', {'channel': 3, 'velocity': 64}) == {}
    assert call(bridge, 'trigger', {'channel': 8}) == {}
    assert [c for c in fake_core.calls if c[0] == 'trigger'] == [('trigger', 2, 64), ('trigger', 7, 127)]
    for bad in ({'channel': 0}, {'channel': 9}, {'channel': 1, 'velocity': 0}, {'velocity': 3},
                {'channel': True}):
        assert 'error' in call(bridge, 'trigger', bad)


def test_to_json_makes_infinity_and_nan_valid_json():
    # A pitch mod rate of "inf" is a real preset value; JSON has no Infinity
    text = to_json({'osc.mod_rate': float('inf'), 'low': -float('inf'), 'nan': float('nan'),
                    'list': [float('inf'), 1.5]})
    assert 'Infinity' not in text and 'NaN' not in text
    data = json.loads(text)
    assert data['osc.mod_rate'] == sys.float_info.max
    assert data['low'] == -sys.float_info.max
    assert data['nan'] is None
    assert data['list'] == [sys.float_info.max, 1.5]


def test_frame_after_loading_a_preset_with_an_infinite_pitch_rate_is_valid_json(real_core, tmp_path):
    text = (ROOT / 'tests' / '505.mtpreset').read_text()
    assert 'ModRate: ' in text
    path = tmp_path / 'inf.mtpreset'
    path.write_text(re.sub(r'ModRate: [^\n]*', 'ModRate: inf ms', text, count=1))
    action = real_core.act('preset.load', path=str(path))
    while True:
        try:
            real_core.wait(action, timeout=0.05)
            break
        except TimeoutError:
            real_core.backend.stream.pull()
    frames = []
    bridge = Bridge(real_core)
    bridge.frame.connect(frames.append)
    bridge.tick()
    assert frames
    assert 'Infinity' not in frames[0]
    state = json.loads(frames[0])
    assert any(v == sys.float_info.max for v in state['changes'].values())
