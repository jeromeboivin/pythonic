"""Page tests of web slice W6 over the fake core: the setup sheet (decision
#22). Tabs on top that size the sheet, SETUP on audio and the MIDI LED on
midi; audio devices, rates and the restart dots; the MIDI input, base note
and clock; the CC mappings (unlimited rows, any CC, add / edit / remove /
clear, live activity, pitch bend), learning left to the panel; smoothing;
the AI models and temperatures."""

import json

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest

from tests.web.page import same_path
from tests.web.test_face_page import wheel_notches
from tests.web.test_menus_page import acts, alert_up, answer_alert, click_item, display, wait_act

SHEET = '.sheet-layer[data-sheet="setup"]'


def open_setup(panel, tab=None):
    if tab is None:
        panel.click('#setup-button')
    else:
        panel.run(f"pythonic.panel.openPage('setup', {{tab: {json.dumps(tab)}}})")
    panel.wait_js(f"!!document.querySelector('{SHEET} .setup')")
    if tab:
        panel.wait_js(f"document.querySelector('.setup').dataset.tab === {json.dumps(tab)}")


def setup_tab(panel):
    return panel.js("document.querySelector('.setup') ? document.querySelector('.setup').dataset.tab : null")


def sets_of(core, address):
    return [v for a, v in core.sets() if a == address]


def wait_set(panel, address, count=1):
    panel.qtbot.waitUntil(lambda: len(sets_of(panel.core, address)) >= count)
    return sets_of(panel.core, address)


def menu_items(panel):
    panel.wait_js("!!document.querySelector('.px-menu .it')")
    return panel.js("[...document.querySelectorAll('.px-menu .it')].map((i) => i.textContent)")


def text(panel, selector):
    return panel.js(f"document.querySelector({json.dumps(selector)}).textContent")


# ---------------------------------------------------------------- the sheet

def test_setup_opens_on_audio_and_sizes_to_its_tab(panel):
    open_setup(panel)
    assert setup_tab(panel) == 'audio'
    assert panel.js("[...document.querySelectorAll('.su-tabs .btn')].map((b) => b.textContent)") == \
        ['audio', 'midi', 'synthesis', 'ai']
    assert panel.js("document.querySelector('.su-tabs .btn.on').dataset.tab") == 'audio'
    wide = panel.js("document.querySelector('.sheet.setup-sheet').offsetWidth")
    panel.click('.su-tabs .btn[data-tab="synthesis"]')
    panel.wait_js("document.querySelector('.setup').dataset.tab === 'synthesis'")
    narrow = panel.js("document.querySelector('.sheet.setup-sheet').offsetWidth")
    assert narrow < wide
    assert panel.js("!!document.querySelector('px-knob[data-address=\"pref.smoothing_ms\"]')")
    panel.click('.su-close')
    panel.wait_js(f"!document.querySelector('{SHEET}')")
    open_setup(panel)  # SETUP always opens on audio
    assert setup_tab(panel) == 'audio'


def test_the_midi_led_opens_the_midi_tab_and_a_click_outside_closes(panel):
    panel.click('#midi-row')
    panel.wait_js("document.querySelector('.setup') && document.querySelector('.setup').dataset.tab === 'midi'")
    assert panel.js("document.querySelector('.su-tabs .btn.on').dataset.tab") == 'midi'
    assert panel.js("!!document.querySelector('.su-ccrows[data-address=\"midi.cc_map\"]')")
    point = panel.press(SHEET, fx=0.02, fy=0.02)  # on the dimmed panel
    QTest.mouseRelease(panel.view.focusProxy(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, point)
    panel.wait_js(f"!document.querySelector('{SHEET}')")


def test_a_control_in_the_sheet_offers_no_learn_and_no_cc_mappings(panel):
    open_setup(panel, 'synthesis')
    panel.right_click('px-knob[data-address="pref.smoothing_ms"]')
    panel.wait_js("!!document.querySelector('.px-menu')")
    items = menu_items(panel)
    assert items == ['Reset to default (30 ms)']
    panel.run("document.querySelector('.px-menu').remove()")


# ---------------------------------------------------------------- audio

def test_output_devices_rescan_and_the_rates_of_a_device(panel):
    core = panel.core
    core.verbs['audio.rates'] = lambda device=None: {
        'device': device, 'rates': [48000, 44100] if device == 'USB Codec' else [96000, 48000, 44100, 32000, 22050, 11025, 8000]}

    def rescan():
        core.post_change('audio.output_devices', ['Fake Out', 'USB Codec'])
        return {'output_devices': ['Fake Out', 'USB Codec'], 'input_devices': ['Fake In']}
    core.verbs['audio.rescan'] = rescan
    open_setup(panel)
    panel.wait_js("document.querySelector('.su-rates').textContent === 'this device takes all 7 rates'")
    device = '.su-choice[data-address="pref.audio.device"]'
    assert text(panel, device) == '(system default)'
    panel.click(device)
    assert menu_items(panel) == ['(system default)', 'Fake Out', 'Fake Duplex']
    panel.run("document.querySelector('.px-menu').remove()")
    panel.click('#su-audio-rescan')
    wait_act(panel, 'audio.rescan')
    panel.wait_js("pythonic.panel.display.text()[0] === 'AUDIO DEVICES'")
    panel.click(device)
    assert menu_items(panel) == ['(system default)', 'Fake Out', 'USB Codec']
    click_item(panel, 'USB Codec')
    assert wait_set(panel, 'pref.audio.device') == ['USB Codec']
    panel.wait_js("document.querySelector('.su-rates').textContent === 'this device takes 2 of 7 rates'")
    assert acts(core, 'audio.rates')[-1] == {'device': 'USB Codec'}
    panel.click('.su-choice[data-address="pref.audio.sample_rate"]')
    assert menu_items(panel) == ['48000 Hz', '44100 Hz']
    click_item(panel, '48000 Hz')
    assert wait_set(panel, 'pref.audio.sample_rate') == [48000]
    panel.click('.su-choice[data-address="pref.audio.synth_rate"]')
    assert menu_items(panel) == ['same as output', '22050 Hz', '11025 Hz', '8000 Hz']
    click_item(panel, '22050 Hz')
    assert wait_set(panel, 'pref.audio.synth_rate') == [22050]
    panel.click('.su-choice[data-address="pref.audio.buffer_ms"]')
    click_item(panel, '10 ms')
    assert wait_set(panel, 'pref.audio.buffer_ms') == [10.0]
    panel.click(device)
    click_item(panel, '(system default)')
    assert wait_set(panel, 'pref.audio.device', 2)[-1] is None


BUFFER_WARNING = '.su-buffer-warning'


def buffer_warning(panel):
    """The buffer warning's text, or None while its row is hidden."""
    return panel.js(f"(() => {{ const n = document.querySelector('{BUFFER_WARNING}');"
                    " return n.closest('.su-field').hidden ? null : n.textContent; })()")


def test_a_buffer_under_512_frames_warns_but_stays_selectable(panel):
    core = panel.core
    core.post_change('pref.audio.sample_rate', 44100)
    core.post_change('pref.audio.buffer_ms', 23.8)
    open_setup(panel)
    panel.wait_js(f"!!document.querySelector('{BUFFER_WARNING}')")
    assert buffer_warning(panel) is None
    panel.click('.su-choice[data-address="pref.audio.buffer_ms"]')
    items = menu_items(panel)
    assert items[:3] == ['2 ms', '5 ms', '10 ms']
    click_item(panel, '10 ms')
    assert wait_set(panel, 'pref.audio.buffer_ms') == [10.0]
    panel.qtbot.waitUntil(lambda: buffer_warning(panel) is not None)
    assert buffer_warning(panel) == '441 frames at 44100 Hz is under the safe 512: expect dropouts'
    # the same 10 ms at 96 kHz is 960 frames: no warning
    core.post_change('pref.audio.sample_rate', 96000)
    panel.qtbot.waitUntil(lambda: buffer_warning(panel) is None)
    core.post_change('pref.audio.sample_rate', 44100)
    panel.qtbot.waitUntil(lambda: buffer_warning(panel) is not None)
    panel.click('.su-choice[data-address="pref.audio.buffer_ms"]')
    click_item(panel, '23.8 ms')
    panel.qtbot.waitUntil(lambda: buffer_warning(panel) is None)


def test_stream_fields_light_their_dot_and_restart_audio(panel):
    core = panel.core
    open_setup(panel)
    panel.wait_js("!!document.querySelector('#su-restart')")
    assert not panel.js("document.querySelector('#su-restart').classList.contains('on')")
    assert panel.js("document.querySelectorAll('.su-dot.on').length") == 0
    core.post_change('pref.audio.pending', ['pref.audio.device', 'pref.audio.buffer_ms'])
    panel.wait_js("document.querySelector('#su-restart').classList.contains('on')")
    lit = panel.js("[...document.querySelectorAll('.su-dot.on')].map((d) => d.dataset.pending)")
    assert lit == ['pref.audio.device', 'pref.audio.buffer_ms']
    assert 'wait for a restart' in text(panel, '.su-restart-note')

    def apply():
        core.post_change('pref.audio.pending', [])
        return {'running': True}
    core.verbs['audio.apply'] = apply
    panel.click('#su-restart')
    wait_act(panel, 'audio.apply')
    panel.wait_js("!document.querySelector('#su-restart').classList.contains('on')")
    assert panel.js("document.querySelectorAll('.su-dot.on').length") == 0
    panel.wait_js("pythonic.panel.display.text()[1] === 'restarted'")


def test_mono_applies_at_once_and_the_input_device_is_saved(panel):
    core = panel.core
    open_setup(panel)
    panel.click('px-toggle[data-address="pref.audio.mono"] .btn')
    assert wait_set(panel, 'pref.audio.mono') == [True]
    inp = '.su-choice[data-address="pref.audio.input_device"]'
    assert text(panel, inp) == '(system default: Fake In)'
    panel.click(inp)
    click_item(panel, 'Fake Duplex')
    assert wait_set(panel, 'pref.audio.input_device') == ['Fake Duplex']
    assert not core.verbs_called() or 'audio.apply' not in core.verbs_called()


def test_the_running_stream_shows_under_the_output(panel):
    core = panel.core
    open_setup(panel)
    panel.wait_js("document.querySelector('.su-stream-text').textContent === 'audio stopped'")
    for address, value in [('audio.device', 'Fake Out'), ('audio.sample_rate', 48000),
                           ('audio.synth_rate', 48000), ('audio.running', True)]:
        core.post_change(address, value)
    panel.wait_js("document.querySelector('.su-stream-text').textContent.startsWith('running: Fake Out · 48000 Hz')")


# ---------------------------------------------------------------- midi

def test_midi_input_devices_open_close_and_rescan(panel):
    core = panel.core
    core.verbs['midi.rescan'] = lambda: {'devices': ['BeatStep', 'Through']}

    def open_(device=None):
        name = device or 'BeatStep'
        core.post_change('midi.device', name)
        core.post_change('midi.connected', True)
        return {'device': name}
    core.verbs['midi.open'] = open_
    core.verbs['midi.close'] = lambda: (core.post_change('midi.device', None),
                                        core.post_change('midi.connected', False)) and {'device': None}
    open_setup(panel, 'midi')
    device = '.su-choice[data-address="midi.device"]'
    assert text(panel, device) == '(off)'
    assert text(panel, '.su-led + .su-note') == 'not connected'
    panel.click('#su-midi-rescan')
    wait_act(panel, 'midi.rescan')
    panel.wait_js("pythonic.panel.display.text()[1] === '2 found'")
    panel.click(device)
    assert menu_items(panel) == ['(off)', '(auto-detect)', 'BeatStep', 'Through']
    click_item(panel, 'Through')
    assert wait_act(panel, 'midi.open') == [{'device': 'Through'}]
    panel.wait_js(f"document.querySelector('{device}').textContent === 'Through'")
    panel.wait_js("document.querySelector('.su-led').classList.contains('connected')")
    panel.click(device)
    click_item(panel, '(auto-detect)')
    assert wait_act(panel, 'midi.open', 2)[1] == {'device': None}
    panel.click(device)
    click_item(panel, '(off)')
    wait_act(panel, 'midi.close')
    panel.wait_js(f"document.querySelector('{device}').textContent === '(off)'")


def test_base_note_steps_and_menu_and_clock_sync(panel):
    core = panel.core
    open_setup(panel, 'midi')
    assert text(panel, '#su-base-note') == 'C2 (36)'
    assert text(panel, '.su-range') == 'channels 1–8 play on notes C2–G2 (36–43)'
    panel.click('#su-base-up')
    assert wait_set(panel, 'midi.base_note') == [37]
    panel.wait_js("document.querySelector('.su-range').textContent.endsWith('C#2–G#2 (37–44)')")
    panel.click('#su-base-note')
    click_item(panel, 'C3 (48)')
    assert wait_set(panel, 'midi.base_note', 2)[-1] == 48
    panel.click('#su-base-down')
    assert wait_set(panel, 'midi.base_note', 3)[-1] == 47
    assert text(panel, '.su-synced') == 'no clock yet'
    core.post_change('midi.synced_tempo', 122)
    panel.wait_js("document.querySelector('.su-synced').textContent === 'synced tempo 122 BPM'")
    panel.click('px-toggle[data-address="midi.clock_sync"] .btn')
    assert wait_set(panel, 'midi.clock_sync') == [False]
    panel.wait_js("document.querySelector('.su-synced').textContent === ''")


def rows(panel):
    return panel.js("[...document.querySelectorAll('.su-ccrow')].map((r) => "
                    "[Number(r.dataset.cc), r.querySelector('.su-target').textContent])")


def test_cc_mappings_list_edit_and_remove(panel):
    core = panel.core
    open_setup(panel, 'midi')
    panel.wait_js("document.querySelectorAll('.su-ccrow').length === 2")
    assert rows(panel) == [[1, 'osc freq sel ch'], [2, 'noise freq sel ch']]
    # a new CC number for the first row
    panel.double_click('.su-ccrow[data-cc="1"] .su-cc')
    panel.run("const i = document.querySelector('.su-ccrow[data-cc=\"1\"] .su-cc'); i.focus(); i.select();")
    panel.type_text('74')
    assert wait_set(panel, 'midi.cc_map') == [{'2': 'selected.noise.freq', '74': 'selected.osc.freq'}]
    panel.wait_js("document.querySelectorAll('.su-ccrow')[1].dataset.cc === '74'")
    # another control for CC 2: section, then control
    panel.click('.su-ccrow[data-cc="2"] .su-target')
    click_item(panel, 'global ▸')
    click_item(panel, 'tempo')
    assert wait_set(panel, 'midi.cc_map', 2)[-1] == {'2': 'global.tempo', '74': 'selected.osc.freq'}
    panel.wait_js("document.querySelector('.su-ccrow[data-cc=\"2\"] .su-target').textContent === 'tempo'")
    # ✕ removes a row
    panel.click('.su-ccrow[data-cc="74"] .su-remove')
    assert wait_set(panel, 'midi.cc_map', 3)[-1] == {'2': 'global.tempo'}
    panel.wait_js("document.querySelectorAll('.su-ccrow').length === 1")


def test_add_a_mapping_takes_a_free_cc_and_needs_a_control(panel):
    core = panel.core
    open_setup(panel, 'midi')
    panel.wait_js("document.querySelectorAll('.su-ccrow').length === 2")
    panel.click('#su-cc-add')
    panel.wait_js("document.querySelectorAll('.su-ccrow.draft').length === 1")
    assert rows(panel)[-1] == [4, 'choose a control']
    assert not sets_of(core, 'midi.cc_map')  # nothing written without a control
    panel.click('.su-ccrow.draft .su-target')
    assert 'fx ▸' in menu_items(panel)
    click_item(panel, 'fx ▸')
    click_item(panel, 'reverb mix')
    assert wait_set(panel, 'midi.cc_map') == [
        {'1': 'selected.osc.freq', '2': 'selected.noise.freq', '4': 'selected.fx.reverb_mix'}]
    panel.wait_js("document.querySelectorAll('.su-ccrow.draft').length === 0 && "
                  "document.querySelectorAll('.su-ccrow').length === 3")
    # a draft row can be dropped too
    panel.click('#su-cc-add')
    panel.wait_js("document.querySelectorAll('.su-ccrow.draft').length === 1")
    panel.click('.su-ccrow.draft .su-remove')
    panel.wait_js("document.querySelectorAll('.su-ccrow.draft').length === 0")


def test_mappings_are_unlimited_and_scroll_after_ten_rows(panel):
    core = panel.core
    many = {str(cc): 'global.tempo' for cc in range(20, 34)}  # 14 mappings
    core.post_change('midi.cc_map', many)
    open_setup(panel, 'midi')
    panel.wait_js("document.querySelectorAll('.su-ccrow').length === 14")
    box = panel.js("(() => { const r = document.querySelector('.su-ccrows'); "
                   "return [r.clientHeight, r.scrollHeight]; })()")
    assert box[1] > box[0]  # it scrolls
    row = panel.js("document.querySelector('.su-ccrow').offsetHeight")
    assert 9.5 * row < box[0] < 11.5 * row  # about ten rows show


def test_clear_all_asks_first(panel):
    open_setup(panel, 'midi')
    panel.wait_js("document.querySelectorAll('.su-ccrow').length === 2")
    panel.click('#su-cc-clear')
    title, body = alert_up(panel, 'ok')
    assert title == 'Clear every CC mapping?'
    assert body == '2 mappings will be removed.'
    answer_alert(panel, primary=False)
    assert not sets_of(panel.core, 'midi.cc_map')
    assert panel.js(f"!!document.querySelector('{SHEET}')")  # the sheet stays under the alert
    panel.click('#su-cc-clear')
    alert_up(panel, 'ok')
    answer_alert(panel)
    assert wait_set(panel, 'midi.cc_map') == [{}]
    panel.wait_js("document.querySelector('.su-empty') !== null")


def test_cc_activity_lights_its_row(panel):
    core = panel.core
    open_setup(panel, 'midi')
    panel.wait_js("document.querySelectorAll('.su-ccrow').length === 2")
    core.readouts['midi'] = {'activity': 1, 'notes': [0] * 8,
                             'pickup': {'ch1.osc.freq': {'cc': 1, 'physical': 0.5, 'linked': True, 'count': 1}}}
    panel.tick()
    panel.wait_js("document.querySelector('.su-ccrow[data-cc=\"1\"] .su-act-bar').style.width === '50%'")
    core.readouts['midi'] = {'activity': 2, 'notes': [0] * 8,
                             'pickup': {'ch1.osc.freq': {'cc': 1, 'physical': 0.75, 'linked': True, 'count': 2}}}
    panel.tick()
    panel.wait_js("document.querySelector('.su-ccrow[data-cc=\"1\"]').classList.contains('cc-hit')")
    assert not panel.js("document.querySelector('.su-ccrow[data-cc=\"2\"]').classList.contains('cc-hit')")


def test_pitch_bend_target_and_learning_on_the_panel(panel):
    core = panel.core
    open_setup(panel, 'midi')
    bend = '.su-bend'
    panel.wait_js(f"document.querySelector('{bend}').textContent === 'tune sel ch'")
    panel.click(bend)
    click_item(panel, '(none)')
    assert wait_set(panel, 'midi.pitchbend_target') == [None]
    panel.wait_js(f"document.querySelector('{bend}').textContent === '(none)'")
    panel.click(bend)
    click_item(panel, 'oscillator ▸')
    click_item(panel, 'osc freq')
    assert wait_set(panel, 'midi.pitchbend_target', 2)[-1] == 'selected.osc.freq'
    # a learn armed on the panel shows here, with cancel
    assert panel.js("document.querySelector('.su-learning').hidden")
    core.post_change('midi.learning', 'ch1.osc.decay')
    panel.wait_js("!document.querySelector('.su-learning').hidden")
    assert 'learning CH1 osc decay' in text(panel, '.su-learning')
    panel.click('.su-learning .btn')
    wait_act(panel, 'midi.learn_cancel')
    # learn on the panel closes the sheet and says how
    panel.click('#su-learn')
    panel.wait_js(f"!document.querySelector('{SHEET}')")
    assert display(panel) == ['MIDI LEARN', 'right-click a control']


def test_the_cc_mappings_entry_of_a_panel_control_opens_the_midi_tab(panel):
    panel.right_click('px-knob[data-address="global.master"]')
    click_item(panel, 'CC mappings…')
    panel.wait_js("document.querySelector('.setup') && document.querySelector('.setup').dataset.tab === 'midi'")


# ---------------------------------------------------------------- ai

def test_ai_models_browse_clear_and_temperatures(panel, tmp_path):
    core = panel.core
    model = tmp_path / 'mine.pt'
    model.write_bytes(b'x')
    core.post_change('ai.available', True)  # the real core's value says whether torch is here
    panel.wait_js("pythonic.store.value('ai.available') === true")
    open_setup(panel, 'ai')
    assert panel.js("document.querySelector('.su-ml').hidden")
    assert panel.js("!!document.querySelector('px-knob[data-address=\"pref.ai.pattern_temperature\"]')")
    assert panel.js("!!document.querySelector('px-knob[data-address=\"pref.ai.patch_temperature\"]')")
    panel.click('.su-browse[data-kind="pattern"]')
    panel.answer_dialog(model)
    (path,) = wait_set(panel, 'pref.ai.pattern_model')
    assert same_path(path, model)
    panel.wait_js("document.querySelector('.su-path[data-address=\"pref.ai.pattern_model\"]').textContent === 'mine.pt'")
    panel.click('.su-clear[data-kind="pattern"]')
    assert wait_set(panel, 'pref.ai.pattern_model', 2)[-1] is None
    wheel_notches(panel, panel.qtbot, 'px-knob[data-address="pref.ai.pattern_temperature"]',
                  'pref.ai.pattern_temperature', 1)
    assert sets_of(core, 'pref.ai.pattern_temperature')[-1] > 0.7


def test_ai_tab_says_when_the_ml_extras_are_missing(panel):
    panel.core.post_change('ai.available', False)
    open_setup(panel, 'ai')
    panel.wait_js("!document.querySelector('.su-ml').hidden")
    assert 'not installed' in text(panel, '.su-ml')
