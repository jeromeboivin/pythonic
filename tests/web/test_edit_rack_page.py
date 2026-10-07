"""Page tests of the edit rack (web slice W4) over the fake core: the rack
bound to the selected channel (#11), edit all, click to assign destinations
and the modulation bands, the drum patch menu, and the drawer's open / closed
state with the window following it (#7)."""

import pytest
from PySide6.QtCore import QSize

from tests.web.test_face_page import gestures, set_calls, wheel_notches


def rack(suffix):
    return f'.rack [data-suffix="{suffix}"]'


def address(panel, selector):
    return panel.js(f"document.querySelector({selector!r}).dataset.address")


def text(panel, selector):
    return panel.js(f"document.querySelector({selector!r}).textContent")


def select(panel, n):
    panel.core.post_change('global.channel', n)
    panel.wait_js(f"document.querySelector('{rack('osc.freq')}').dataset.address === 'ch{n}.osc.freq'")


def sets_of(core, address):
    return [v for a, v, _ in set_calls(core) if a == address]


# ---------------------------------------------------------------- the rack

def test_the_rack_shows_the_selected_channel_and_follows_a_switch(panel):
    core = panel.core
    core.values['ch3.osc.freq'] = 1234.0
    assert address(panel, rack('osc.freq')) == 'ch1.osc.freq'
    assert text(panel, '.rack .rch') == 'CH 1'
    assert text(panel, '.rack .rname') == core.values['ch1.name']
    select(panel, 3)
    panel.wait_js(f"document.querySelector('{rack('osc.freq')} .val').textContent === '1.23 kHz'")
    assert text(panel, '.rack .rch') == 'CH 3'
    assert text(panel, '.rack .rname') == core.values['ch3.name']
    bound = panel.js("[...document.querySelectorAll('.rack [data-suffix]')].map((e) => e.dataset.address)")
    assert all(a.startswith('ch3.') for a in bound) and len(bound) == 56
    for face_only in ('osc.pitch', 'osc.decay', 'mix.level'):  # on the strips only
        assert f'ch3.{face_only}' not in bound
    assert panel.js(f"document.querySelector({rack('noise.freq')!r}).getAttribute('name')") == 'ch3 noise freq'


def test_a_rack_drag_edits_the_selected_channel_as_one_gesture(panel, qtbot):
    core = panel.core
    select(panel, 2)
    before = core.values['ch2.noise.freq']
    panel.drag(f"{rack('noise.freq')} .dial", dy=40)  # down: lower
    qtbot.waitUntil(lambda: gestures(core) == ['begin', 'end'])
    assert core.values['ch2.noise.freq'] < before
    assert {a for a, _, _ in set_calls(core)} == {'ch2.noise.freq'}
    panel.wait_js("pythonic.panel.display.text()[0] === 'CH2 NOISE FREQ'")


def test_switches_toggles_and_lists_of_the_rack(panel, qtbot):
    core = panel.core
    panel.run(f"[...document.querySelectorAll('{rack('noise.filter')} .btn')]"
              ".find((b) => b.textContent === 'BP').click()")
    panel.click(f"{rack('noise.stereo')} button")
    panel.click(f"{rack('lfo1.wave')} .lbox")
    panel.wait_js("!!document.querySelector('.px-menu')")
    items = panel.js("[...document.querySelectorAll('.px-menu .it')].map((i) => i.textContent)")
    assert items == ['sin', 'tri', 'saw▲', 'saw▼', 'sq', 's&h']
    panel.run("document.querySelectorAll('.px-menu .it')[3].click()")
    qtbot.waitUntil(lambda: core.values['ch1.lfo1.wave'] == 'saw_down')
    assert core.values['ch1.noise.filter'] == 'band_pass'
    assert core.values['ch1.noise.stereo'] is True
    delay = panel.js(f"[...document.querySelectorAll('{rack('fx.delay_time')} .btn')].map((b) => b.textContent)")
    assert delay == ['1/4', '1/8', '1/16', '1/8T', '1/4.']
    core.post_change('ch1.fx.delay_time', 'thirtysecond')  # a delay time only a preset can set
    panel.wait_js(f"document.querySelector('{rack('fx.delay_time')} .other').textContent === '1/32'")


def test_the_osc_noise_crossfader_drags_sideways_towards_noise(panel, qtbot):
    core = panel.core
    before = core.values['ch1.mix.osc_noise']
    panel.drag(f"{rack('mix.osc_noise')} .cap", dy=0, dx=30)
    qtbot.waitUntil(lambda: gestures(core)[-1:] == ['end'])
    assert core.values['ch1.mix.osc_noise'] < before - 0.2


def test_edit_all_in_the_rack_header_links_the_strips(panel, qtbot):
    core = panel.core
    panel.click('#edit-all button')
    qtbot.waitUntil(lambda: core.values['global.edit_all'] is True)
    panel.wait_js("getComputedStyle(document.querySelector('.rack .ea-note')).display !== 'none'")
    panel.wait_js("document.querySelector('.strip[data-channel=\"4\"]').classList.contains('linked')")
    assert not panel.js("document.querySelector('.strip[data-channel=\"1\"]').classList.contains('linked')")


# ---------------------------------------------------------------- click to assign

def arm(panel, source):
    panel.click(f"{rack(source + '.target')} button")
    panel.wait_js("document.querySelector('#stage').classList.contains('assigning')")


def assignable(panel, selector):
    return panel.js(f"document.querySelector({selector!r}).classList.contains('assignable')")


def test_the_arrow_arms_assign_mode_and_a_rack_click_sets_the_destination(panel, qtbot):
    core = panel.core
    arm(panel, 'lfo1')
    assert panel.js("getComputedStyle(document.querySelector('.rack .abar')).display") == 'flex'
    assert assignable(panel, rack('noise.freq'))
    assert assignable(panel, '.strip[data-channel="1"] px-knob[data-address="ch1.osc.pitch"]')
    assert assignable(panel, '.strip[data-channel="1"] px-fader')
    assert assignable(panel, 'px-knob[data-address="global.master"]')
    assert assignable(panel, 'px-knob[data-address="morph.position"]')
    assert not assignable(panel, rack('osc.wave'))
    assert not assignable(panel, '.strip[data-channel="2"] px-knob[data-address="ch2.osc.pitch"]')
    before = core.values['ch1.noise.freq']
    panel.click(f"{rack('noise.freq')} .dial")
    qtbot.waitUntil(lambda: core.values['ch1.lfo1.target'] == 'noise_filter_freq')
    assert core.values['ch1.noise.freq'] == before and gestures(core) == []
    panel.wait_js("!document.querySelector('#stage').classList.contains('assigning')")
    assert text(panel, f"{rack('lfo1.target')} button") == '→ noise freq'
    assert panel.js("pythonic.panel.display.text()") == ['CH1 LFO 1', '→ noise freq (turn it on)']


def test_destinations_on_the_face(panel, qtbot):
    core = panel.core
    arm(panel, 'lfo2')
    panel.click('.strip[data-channel="1"] px-fader .track')
    qtbot.waitUntil(lambda: core.values['ch1.lfo2.target'] == 'level_db')
    arm(panel, 'pump')
    panel.click('px-knob[data-address="morph.position"] .dial')
    qtbot.waitUntil(lambda: core.values['ch1.pump.target'] == 'morph')
    select(panel, 5)
    arm(panel, 'lfo1')
    panel.click('.strip[data-channel="5"] px-knob[data-address="ch5.osc.decay"] .dial')
    qtbot.waitUntil(lambda: core.values['ch5.lfo1.target'] == 'osc_decay')
    assert sets_of(core, 'ch1.mix.level') == [] and sets_of(core, 'morph.position') == []


def test_other_controls_refuse_politely_and_off_or_cancel_end_it(panel, qtbot):
    core = panel.core
    arm(panel, 'lfo1')
    panel.click(f"{rack('osc.wave')} .btn")
    panel.wait_js("pythonic.panel.display.text()[1] === 'not a destination'")
    panel.click('.strip[data-channel="2"] px-knob[data-address="ch2.osc.pitch"] .dial')
    panel.wait_js("pythonic.panel.display.text()[1] === 'not on CH1'")
    assert panel.js("document.querySelector('#stage').classList.contains('assigning')")
    assert sets_of(core, 'ch1.osc.wave') == [] and sets_of(core, 'ch2.osc.pitch') == []
    panel.click('#assign-cancel')
    panel.wait_js("!document.querySelector('#stage').classList.contains('assigning')")
    core.post_change('ch1.pump.target', 'pan')
    arm(panel, 'pump')
    panel.wait_js(f"document.querySelector({rack('mix.pan')!r}).classList.contains('assigned')")
    assert panel.js(f"document.querySelector({rack('mix.pan')!r}).classList.contains('assigned')")
    panel.click('#assign-off')
    qtbot.waitUntil(lambda: core.values['ch1.pump.target'] == 'none')
    arm(panel, 'lfo2')
    panel.click(f"{rack('lfo2.target')} button")  # again: disarms
    panel.wait_js("!document.querySelector('#stage').classList.contains('assigning')")


def test_the_wheel_on_the_arrow_steps_through_the_targets(panel, qtbot):
    wheel_notches(panel, qtbot, f"{rack('lfo1.target')} button", 'ch1.lfo1.target', 1)
    assert panel.core.values['ch1.lfo1.target'] == 'osc_frequency'


def test_modulation_bands_show_on_the_rack_and_on_every_strip(panel):
    core = panel.core
    for source, target in (('lfo1', 'osc_frequency'), ('lfo2', 'pitch_semitones')):
        core.post_change(f'ch1.{source}.on', True)
        core.post_change(f'ch1.{source}.target', target)
    core.post_change('ch4.pump.on', True)
    core.post_change('ch4.pump.target', 'level_db')
    channels = [{} for _ in range(8)]
    channels[0] = {'osc_frequency': 300.0, 'pitch_semitones': 6.0}
    channels[3] = {'level_db': -12.0}
    core.readouts['modulation'] = {'channel': 0, 'offsets': channels[0], 'channels': channels}
    panel.wait_js(f"document.querySelector({rack('osc.freq')!r}).dataset.mod === 'lfo1'")
    assert panel.js("document.querySelector('px-knob[data-address=\"ch1.osc.pitch\"]').dataset.mod") == 'lfo2'
    assert panel.js("document.querySelector('.strip[data-channel=\"4\"] px-fader').dataset.mod") == 'pump'
    knob = 'px-knob[data-address="ch1.osc.pitch"]'
    assert panel.js(f"document.querySelector({knob!r}).dataset.modTo") == '0.6250'  # (0 + 6 + 24) / 48
    channels[0] = {'osc_frequency': 300.0, 'pitch_semitones': -12.0}  # the LFO moves on
    core.readouts['modulation'] = {'channel': 0, 'offsets': channels[0], 'channels': channels}
    panel.wait_js(f"document.querySelector({knob!r}).dataset.modTo === '0.2500'")
    panel.wait_pixels(lambda: panel.close_to(panel.color_at(f'{knob} .dial', 0.19, 0.19), (255, 95, 168), 70))


# ---------------------------------------------------------------- drum patch menu

def test_the_drum_patch_menu_loads_and_saves_the_selected_channel(panel, qtbot, tmp_path):
    core = panel.core
    patch = tmp_path / 'kick.mtdrum'
    patch.write_text('x')
    answers = iter([{'saved': False, 'exists': True, 'path': str(patch)},
                    {'saved': True, 'exists': False, 'path': str(patch)}])
    core.verbs['drum_patch.load'] = lambda path, channel: {'channel': channel, 'name': 'Kick', 'path': path}
    core.verbs['drum_patch.save'] = lambda path, channel, overwrite=False: next(answers)
    select(panel, 2)
    panel.click('#patch-menu')
    panel.wait_js("!!document.querySelector('.px-menu')")
    panel.run("document.querySelectorAll('.px-menu .it')[0].click()")
    panel.answer_dialog(patch)
    qtbot.waitUntil(lambda: any(c[:2] == ('act', 'drum_patch.load') for c in core.calls))
    (load,) = [c[2] for c in core.calls if c[:2] == ('act', 'drum_patch.load')]
    assert load['channel'] == 2 and load['path'].endswith('kick.mtdrum')
    panel.wait_js("pythonic.panel.display.text()[1] === 'KICK'")

    panel.click('#patch-menu')
    panel.wait_js("!!document.querySelector('.px-menu')")
    panel.run("document.querySelectorAll('.px-menu .it')[1].click()")
    panel.answer_dialog(patch)
    panel.wait_js("[...document.querySelectorAll('.px-menu .it')].some((i) => i.textContent === 'Replace it')")
    panel.run("[...document.querySelectorAll('.px-menu .it')].find((i) => i.textContent === 'Replace it').click()")
    qtbot.waitUntil(lambda: len([c for c in core.calls if c[:2] == ('act', 'drum_patch.save')]) == 2)
    saves = [c[2] for c in core.calls if c[:2] == ('act', 'drum_patch.save')]
    assert [s.get('overwrite', False) for s in saves] == [False, True]
    panel.wait_js("pythonic.panel.display.text()[1] === 'saved'")


# ---------------------------------------------------------------- open / closed

def stage_height(panel):
    return panel.js("document.querySelector('#stage').dataset.height || '1000'")


def window_is(panel, qtbot, height):
    """Wait for the window and then the page (it refits on its own resize
    event) to be 1280 wide and `height` high, so clicks land on fresh boxes."""
    qtbot.waitUntil(lambda: panel.window.size() == QSize(1280, height))
    qtbot.waitUntil(lambda: panel.rect('#stage') == pytest.approx((0, 0, 1280, height)))


def test_the_rack_toggle_closes_the_drawer_and_the_window_shrinks(panel, qtbot):
    core = panel.core
    window = panel.window
    assert panel.js("document.querySelector('#rack-toggle').classList.contains('on')")
    panel.click('#rack-toggle')
    window_is(panel, qtbot, 560)
    assert stage_height(panel) == '700'
    qtbot.waitUntil(lambda: core.values.get('pref.ui.rack_open') is False)
    panel.wait_js("getComputedStyle(document.querySelector('.slot-rack')).display === 'none'")
    assert window.minimumSize() == QSize(1280, 560)
    assert not panel.js("document.querySelector('#rack-toggle').classList.contains('on')")
    panel.click('#rack-toggle')
    window_is(panel, qtbot, 800)
    qtbot.waitUntil(lambda: core.values.get('pref.ui.rack_open') is True)
    assert stage_height(panel) == '1000'


def test_a_drawer_page_opens_a_closed_rack_for_as_long_as_it_shows(panel, qtbot):
    core = panel.core
    panel.click('#rack-toggle')
    window_is(panel, qtbot, 560)
    panel.click('#matrix-toggle')
    window_is(panel, qtbot, 800)
    panel.wait_js("!!document.querySelector('.slot-rack .matrix')")
    panel.click('#matrix-toggle')
    window_is(panel, qtbot, 560)
    assert sets_of(core, 'pref.ui.rack_open') == [False]  # the page did not touch the preference


def test_the_rack_starts_closed_when_the_preference_says_so(open_panel, fake_core, qtbot):
    fake_core.values['pref.ui.rack_open'] = False
    page = open_panel(fake_core)
    window_is(page, qtbot, 560)
    assert stage_height(page) == '700'
    assert page.window.minimumSize() == QSize(1280, 560)
