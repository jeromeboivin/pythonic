"""Page tests of the hardware face (web slice W2) over the fake core: the
knob and fader behaviour (#9), the strips, the top row, the right column and
their wiring to addresses and verbs."""

import pytest
from PySide6.QtCore import Qt

STRIP = '.strip[data-channel="{n}"]'
DECAY3 = '.strip[data-channel="3"] px-knob[data-address="ch3.osc.decay"]'
LEVEL2 = '.strip[data-channel="2"] px-fader'


def set_calls(core):
    return [(a, v, o) for kind, a, v, o in (c for c in core.calls if c[0] == 'set')]


def gestures(core):
    return [c[1] for c in core.calls if c[0] == 'gesture']


def wheel_notches(panel, qtbot, selector, address, notches):
    """Turn the wheel notch by notch (Chromium merges events sent at once)."""
    for _ in range(abs(notches)):
        before = panel.core.values[address]
        panel.wheel(selector, steps=1 if notches > 0 else -1)
        qtbot.waitUntil(lambda: panel.core.values[address] != before)


def text(panel, selector):
    return panel.js(f"document.querySelector({selector!r}).textContent")


# ---------------------------------------------------------------- knobs and faders

def test_a_drag_on_a_strip_knob_is_one_gesture(panel, qtbot):
    core = panel.core
    before = core.values['ch3.osc.decay']
    panel.drag(f'{DECAY3} .dial', dy=-40)  # up 40 view px = 50 design px at 0.8 = +25 %
    qtbot.waitUntil(lambda: gestures(core) == ['begin', 'end'])
    sets = [c for c in set_calls(core) if c[0] == 'ch3.osc.decay']
    assert sets and all(not o['burst'] for _, _, o in sets)
    assert core.values['ch3.osc.decay'] > before
    position = panel.js(f"Number(document.querySelector({DECAY3!r}).dataset.position)")
    assert position == pytest.approx(_position(before) + 0.25, abs=0.03)


def _position(decay_ms):
    import math
    return math.log(decay_ms / 10) / math.log(1000)


def test_shift_drags_ten_times_finer(panel, qtbot):
    core = panel.core
    panel.drag(f'{STRIP.format(n=1)} px-knob[data-address="ch1.osc.pitch"] .dial', dy=-40,
               modifiers=Qt.KeyboardModifier.ShiftModifier)
    qtbot.waitUntil(lambda: gestures(core) == ['begin', 'end'])
    # 50 design px fine = 2.5 % of 48 st
    assert core.values['ch1.osc.pitch'] == pytest.approx(1.2, abs=0.5)


def test_the_wheel_on_a_knob_is_a_burst_of_one_percent(panel, qtbot):
    core = panel.core
    wheel_notches(panel, qtbot, f'{STRIP.format(n=4)} px-knob[data-address="ch4.osc.pitch"]',
                  'ch4.osc.pitch', 2)
    qtbot.waitUntil(lambda: core.values['ch4.osc.pitch'] == pytest.approx(0.96))
    calls = set_calls(core)
    assert {a for a, _, _ in calls} == {'ch4.osc.pitch'}
    assert all(o['burst'] for _, _, o in calls)
    assert gestures(core) == []


def test_a_click_on_the_fader_track_jumps_there(panel, qtbot):
    core = panel.core
    panel.drag(f'{LEVEL2} .track', dy=0, fy=0.95)  # near the bottom of the track
    qtbot.waitUntil(lambda: gestures(core)[-1:] == ['end'])
    assert core.values['ch2.mix.level'] < -50


def test_the_wheel_on_a_fader_moves_two_percent(panel, qtbot):
    panel.wheel(LEVEL2, steps=-1)
    qtbot.waitUntil(lambda: bool(set_calls(panel.core)))
    address, value, options = set_calls(panel.core)[0]
    assert (address, round(value, 6), options['burst']) == ('ch2.mix.level', -1.4, True)


def test_values_are_printed_under_the_controls(panel):
    panel.core.post_change('ch3.osc.decay', 1500.0)
    panel.wait_js(f"document.querySelector('{DECAY3} .val').textContent === '1.50 s'")
    assert text(panel, f'{STRIP.format(n=3)} px-fader .val') == '0.0 dB'
    assert text(panel, 'px-knob[data-address="global.master"] .val') == '0.0 dB'


def test_double_click_types_an_exact_value(panel, qtbot):
    panel.double_click(f'{DECAY3} .dial')
    panel.wait_js("!!(document.activeElement && document.activeElement.closest('.px-entry'))")
    panel.run("document.activeElement.select()")
    panel.type_text('750')
    qtbot.waitUntil(lambda: ('ch3.osc.decay', 750.0) in [(a, v) for a, v, _ in set_calls(panel.core)])
    assert panel.js("document.querySelector('.px-entry')") is None


def test_right_click_resets_to_the_default(panel, qtbot):
    panel.core.post_change('ch3.osc.decay', 2000.0)
    panel.wait_js(f"document.querySelector('{DECAY3} .val').textContent === '2.00 s'")
    panel.right_click(f'{DECAY3} .dial')
    panel.wait_js("!!document.querySelector('.px-menu')")
    items = panel.js("[...document.querySelectorAll('.px-menu .it')].map((i) => i.textContent)")
    assert items[0] == 'Reset to default (316 ms)'
    panel.click('.px-menu .it')
    qtbot.waitUntil(lambda: panel.core.values['ch3.osc.decay'] == pytest.approx(316.23))


def test_midi_learn_from_the_menu_and_the_cc_badge(panel, qtbot):
    core = panel.core
    panel.right_click(f'{DECAY3} .dial')
    panel.wait_js("!!document.querySelector('.px-menu')")
    panel.run("[...document.querySelectorAll('.px-menu .it')]"
              ".find((i) => i.textContent.startsWith('MIDI learn')).click()")
    qtbot.waitUntil(lambda: ('act', 'midi.learn', {'target': 'ch3.osc.decay'}) in core.calls)
    core.post_change('midi.learning', 'ch3.osc.decay')
    panel.wait_js(f"document.querySelector({DECAY3!r}).classList.contains('learning')")
    core.post_change('midi.learning', None)
    core.post_change('midi.cc_map', {'21': 'ch3.osc.decay', '1': 'selected.osc.freq'})
    panel.wait_js(f"document.querySelector('{DECAY3} .cc').textContent === 'CC 21'")
    assert not panel.js(f"document.querySelector({DECAY3!r}).classList.contains('learning')")


def test_incoming_cc_blinks_the_led_and_shows_the_pickup_ghost(panel, qtbot):
    core = panel.core
    core.post_change('midi.cc_map', {'21': 'ch3.osc.decay'})
    pickup = core.readouts['midi']['pickup']
    pickup['ch3.osc.decay'] = {'cc': 21, 'physical': 0.9, 'linked': False, 'count': 1}
    panel.wait_js(f"document.querySelector({DECAY3!r}).dataset.ghost === '0.9000'")
    core.readouts['midi'] = {'activity': 2, 'notes': [0] * 8,
                             'pickup': {'ch3.osc.decay': {'cc': 21, 'physical': 0.9,
                                                          'linked': False, 'count': 2}}}
    panel.wait_js(f"document.querySelector({DECAY3!r}).classList.contains('cc-blink')")
    panel.wait_js("document.querySelector('#midi-led').classList.contains('on')")


def test_the_display_shows_the_touched_control(panel, qtbot):
    panel.wheel(f'{STRIP.format(n=5)} px-knob[data-address="ch5.osc.pitch"]', steps=1)
    panel.wait_js("pythonic.panel.display.text()[0] === 'CH5 TUNE'")
    assert panel.js("pythonic.panel.display.text()[1]") == '+0.5 st'


def test_modulation_draws_a_moving_arc_on_every_strip(panel, qtbot):
    core = panel.core
    core.post_change('ch6.lfo2.on', True)
    core.post_change('ch6.lfo2.target', 'osc_decay')
    channels = [{} for _ in range(8)]
    channels[5] = {'osc_decay': 300.0}
    core.readouts['modulation'] = {'channel': 0, 'offsets': {}, 'channels': channels}
    knob = '.strip[data-channel="6"] px-knob[data-address="ch6.osc.decay"]'
    panel.wait_js(f"document.querySelector({knob!r}).dataset.mod === 'lfo2'")
    first = panel.js(f"document.querySelector('{knob} .mod').getAttribute('d')")
    channels = [{} for _ in range(8)]
    channels[5] = {'osc_decay': 900.0}
    core.readouts['modulation'] = {'channel': 0, 'offsets': {}, 'channels': channels}
    panel.wait_js(f"document.querySelector('{knob} .mod').getAttribute('d') !== {first!r}")
    lfo2 = (255, 95, 168)
    # The middle of the arc on screen
    x, y = panel.js(f"""(() => {{ const p = document.querySelector('{knob} .mod');
      const m = p.getPointAtLength(p.getTotalLength() / 2).matrixTransform(p.getScreenCTM());
      return [m.x, m.y]; }})()""")
    panel.wait_pixels(lambda: panel.close_to(panel.pixel(x, y), lfo2, 60))


# ---------------------------------------------------------------- strips

def test_strips_show_patch_names_and_drum_types(panel):
    names = panel.js("[...document.querySelectorAll('.strip .tab')].map((t) => t.textContent)")
    assert names == [panel.core.values[f'ch{n}.name'] for n in range(1, 9)]
    panel.core.post_change('ch2.name', '808 Rimshot')
    panel.wait_js(f"document.querySelector('{STRIP.format(n=2)} .chb').textContent === 'RS'")
    panel.core.post_change('ch2.name', 'Metal Ping')
    panel.wait_js(f"document.querySelector('{STRIP.format(n=2)} .chb').textContent === '2'")


def test_a_channel_button_selects_and_the_strip_lights(panel, qtbot):
    panel.click(f'{STRIP.format(n=4)} .chb')
    qtbot.waitUntil(lambda: panel.core.values['global.channel'] == 4)
    panel.wait_js(f"document.querySelector({STRIP.format(n=4)!r}).classList.contains('sel')")
    # the selected tab takes the strip's colour (channel 4 green)
    panel.wait_pixels(lambda: panel.close_to(panel.color_at(f'{STRIP.format(n=4)} .tab', 0.1, 0.5),
                                             (45, 255, 122), 60))


def test_the_mute_latch_makes_channel_buttons_mute(panel, qtbot):
    panel.click('#mute-latch')
    panel.wait_js("document.querySelector('#mute-latch').classList.contains('on')")
    panel.click(f'{STRIP.format(n=1)} .chb')
    qtbot.waitUntil(lambda: panel.core.values['ch1.mute'] is True)
    assert panel.core.values['global.channel'] == 1
    track = f'{STRIP.format(n=1)} px-fader .track'
    panel.wait_pixels(lambda: max(panel.color_at(track, 0.02, 0.5)) < 90)  # the fader light is dark
    lit = panel.color_at(f'{STRIP.format(n=2)} px-fader .track', 0.02, 0.5)
    assert lit[0] > 150  # channel 2 still lit (orange)
    panel.click('#mute-latch')
    panel.click(f'{STRIP.format(n=2)} .chb')
    qtbot.waitUntil(lambda: panel.core.values['global.channel'] == 2)
    assert panel.core.values['ch2.mute'] is False


def test_edit_all_marks_the_strips_it_reaches(panel):
    panel.core.post_change('ch5.mute', True)
    panel.core.post_change('global.edit_all', True)
    panel.wait_js("document.querySelector('.stage').classList.contains('edit-all')")
    linked = panel.js("[...document.querySelectorAll('.strip.linked')].map((s) => s.dataset.channel)")
    assert linked == ['2', '3', '4', '6', '7', '8']


def test_the_strip_ctrl_mode_is_a_panel_preference(panel, qtbot):
    ctrl = f'{STRIP.format(n=3)} .ctrl'
    assert panel.js(f"document.querySelector({ctrl!r}).dataset.address") == 'ch3.mix.pan'
    panel.click('#ctrl-mode')
    panel.wait_js("!!document.querySelector('.px-menu')")
    panel.run("[...document.querySelectorAll('.px-menu .it')].find((i) => i.textContent === 'delay mix').click()")
    qtbot.waitUntil(lambda: panel.core.values.get('pref.ui.ctrl_knob', {}).get('mode') == 'delay_mix')
    panel.wait_js(f"document.querySelector({ctrl!r}).dataset.address === 'ch3.fx.delay_mix'")
    panel.wheel(ctrl, steps=1)
    qtbot.waitUntil(lambda: any(a == 'ch3.fx.delay_mix' for a, _, _ in set_calls(panel.core)))


def test_a_user_ctrl_knob_picks_its_parameter_from_its_label(panel, qtbot):
    panel.core.post_change('pref.ui.ctrl_knob', {'mode': 'user', 'user': [None] * 8})
    ctrl = f'{STRIP.format(n=2)} .ctrl'
    panel.wait_js(f"document.querySelector('{ctrl} .lbl').textContent === 'pick ▾'")
    assert panel.js(f"document.querySelector({ctrl!r}).dataset.address") is None
    panel.click(f'{ctrl} .lbl')
    panel.wait_js("!!document.querySelector('.px-menu')")
    panel.run("[...document.querySelectorAll('.px-menu .it')].find((i) => i.textContent.trim() === 'noise freq').click()")
    qtbot.waitUntil(lambda: (panel.core.values['pref.ui.ctrl_knob'] or {}).get('user', [None] * 8)[1]
                    == 'noise.freq')
    panel.wait_js(f"document.querySelector({ctrl!r}).dataset.address === 'ch2.noise.freq'")


# ---------------------------------------------------------------- top row and columns

def test_step_rate_fill_rate_master_and_swing(panel, qtbot):
    core = panel.core
    panel.click('px-switch[data-address="global.step_rate"] .btn[data-value="1/8T"]')
    qtbot.waitUntil(lambda: core.values['global.step_rate'] == '1/8T')
    panel.click('px-list[data-address="global.fill_rate"] .lbox')
    panel.wait_js("!!document.querySelector('.px-menu')")
    panel.run("[...document.querySelectorAll('.px-menu .it')].find((i) => i.textContent === '6x').click()")
    qtbot.waitUntil(lambda: core.values['global.fill_rate'] == 6)
    panel.wheel('px-knob[data-address="global.master"]', steps=-1)
    qtbot.waitUntil(lambda: core.values['global.master'] == pytest.approx(-0.7))
    wheel_notches(panel, qtbot, 'px-knob[data-address="global.swing"]', 'global.swing', 3)
    qtbot.waitUntil(lambda: core.values['global.swing'] == pytest.approx(0.03))


def test_the_tempo_knob_and_display(panel, qtbot):
    panel.wheel('px-knob[data-address="global.tempo"]', steps=1)
    qtbot.waitUntil(lambda: panel.core.values['global.tempo'] == 121)
    panel.wait_js("document.querySelector('#tempo-value').textContent === '121'")
    wheel_notches(panel, qtbot, '#tempo-value', 'global.tempo', -2)
    qtbot.waitUntil(lambda: panel.core.values['global.tempo'] == 119)


def test_programs_select_through_the_verb(panel, qtbot):
    panel.click('#programs .btn[data-program="7"]')
    qtbot.waitUntil(lambda: ('act', 'program.select', {'program': 7}) in panel.core.calls)
    panel.core.post_change('program.current', 7)
    panel.wait_js("document.querySelector('#programs .btn.on').dataset.program === '7'")


def test_undo_and_redo_follow_the_journal(panel, qtbot):
    core = panel.core
    core.verbs['undo'] = lambda: {'done': True, 'label': 'ch1.osc.decay'}
    assert panel.js("document.querySelector('#undo').disabled")
    core.post_change('undo.can_undo', True)
    panel.wait_js("!document.querySelector('#undo').disabled")
    panel.click('#undo')
    qtbot.waitUntil(lambda: 'undo' in core.verbs_called())
    panel.wait_js("pythonic.panel.display.text()[0] === 'UNDO'")


def test_morph_knob_and_learn_buttons(panel, qtbot):
    core = panel.core
    wheel_notches(panel, qtbot, 'px-knob[data-address="morph.position"]', 'morph.position', 5)
    qtbot.waitUntil(lambda: core.values['morph.position'] == pytest.approx(0.05))
    panel.click('#learn-b')
    qtbot.waitUntil(lambda: ('act', 'morph.learn', {'endpoint': 'b'}) in core.calls)
    core.post_change('morph.learning', 'b')
    panel.wait_js("document.querySelector('#learn-b').classList.contains('on')")
    panel.click('#learn-b')
    qtbot.waitUntil(lambda: ('act', 'morph.learn', {'endpoint': None}) in core.calls)


def test_start_stop_runs_the_verb_and_lights_from_the_transport(panel, qtbot):
    assert panel.color_at('#start-stop')[1] < 80  # dark
    panel.click('#start-stop')
    qtbot.waitUntil(lambda: panel.core.verbs_called() == ['transport.toggle'])
    panel.wait_js("document.querySelector('#start-stop').classList.contains('on')")
    panel.wait_pixels(lambda: panel.color_at('#start-stop')[1] > 180)  # lit green


def test_the_midi_led_shows_the_connection(panel):
    assert not panel.js("document.querySelector('#midi-led').classList.contains('connected')")
    panel.core.post_change('midi.connected', True)
    panel.wait_js("document.querySelector('#midi-led').classList.contains('connected')")


def test_stalled_audio_and_errors_show_on_the_display(panel):
    panel.core._post({'id': None, 'verb': None, 'status': 'error', 'source': 'audio',
                      'error': 'Audio stream stalled: no callback for 2 s; the stream was stopped'})
    panel.wait_js("pythonic.panel.display.text()[0] === 'AUDIO ERROR'")
    assert 'stalled' in panel.js("pythonic.panel.display.text()[1]")


def test_the_face_leaves_slots_for_later_slices(panel):
    slots = panel.js("[...document.querySelectorAll('[data-slot]')].map((e) => e.dataset.slot)")
    for name in ('step-entry', 'patterns', 'steps', 'rack', 'preset', 'rack-toggle', 'po32', 'setup'):
        assert name in slots
