"""Page tests of KIT, INST and WRITE over the fake core (kits.js): the kit
selector and the drum patch selector on the display, stepped by ◀ ▶, the
wheel over the display and the arrow keys (SHIFT jumps a group); the edited
mark; WRITE's menu with the name prompt and the replace question; a
pattern's kit from the pattern menu; the PRESET menu's drum patch folder."""

from PySide6.QtCore import Qt

from tests.web.test_menus_page import (acts, alert_up, answer_alert, click_item, display, items,
                                       wait_act)

KITS = [{'id': 'factory:505', 'name': '505', 'factory': True},
        {'id': 'factory:909', 'name': '909', 'factory': True},
        {'id': 'user:Dub', 'name': 'Dub', 'factory': False}]
LIBRARY = [['factory:505 BD', '505 BD', 'BD'], ['factory:909 BD', '909 BD', 'BD'],
           ['user:Kick 1.mtdrum', 'Kick 1', 'BD'], ['factory:909 SD', '909 SD', 'SD'],
           ['factory:808 CH', '808 CH', 'CH'], ['user:a/Hat.mtdrum', 'Hat', 'CH']]


def kit_state(panel, current='', edited=False, name='Beats'):
    core = panel.core
    core.post_change('kit.list', KITS)
    core.post_change('preset.name', name)
    core.post_change('kit.current', current)
    core.post_change('kit.name', next((k['name'] for k in KITS if k['id'] == current), name))
    core.post_change('kit.edited', edited)
    panel.wait_js(f"pythonic.store.value('kit.current') === {current!r}"
                  f" && pythonic.store.value('kit.edited') === {str(edited).lower()}"
                  " && (pythonic.store.value('kit.list') || []).length === 3")


def inst_state(panel, channel=1, name='909 BD'):
    core = panel.core
    core.post_change('drum_patch.library', LIBRARY)
    core.post_change(f'ch{channel}.name', name)
    core.post_change('global.channel', channel)
    panel.wait_js(f"pythonic.store.value('ch{channel}.name') === {name!r}"
                  " && (pythonic.store.value('drum_patch.library') || []).length === 6"
                  f" && pythonic.store.value('global.channel') === {channel}")


def lit(panel, selector):
    return panel.js(f"document.querySelector({selector!r}).classList.contains('on')")


# ---------------------------------------------------------------- KIT

def test_kit_lights_and_the_display_becomes_the_kit_selector(panel):
    kit_state(panel)
    assert display(panel)[1] == 'BEATS'
    panel.click('#kit-button')
    panel.wait_js("pythonic.panel.display.text()[0] === 'KIT 000  PRESET'")
    assert display(panel) == ['KIT 000  PRESET', 'BEATS']
    assert lit(panel, '#kit-button') and not lit(panel, '#inst-button')
    assert panel.js("document.querySelector('#display').classList.contains('selector')")
    panel.click('#kit-button')
    panel.wait_js("pythonic.panel.display.text()[0].startsWith('PATTERN A')")
    assert not lit(panel, '#kit-button')


def test_value_steps_through_the_kits_and_each_plays_at_once(panel):
    core = panel.core
    kit_state(panel, current='factory:505')
    panel.click('#kit-button')
    panel.wait_js("pythonic.panel.display.text()[0] === 'KIT 001  FACTORY'")
    panel.click('#preset-next')
    assert wait_act(panel, 'kit.select') == [{'kit': 'factory:909', 'browse': True}]
    assert display(panel) == ['KIT 002  FACTORY', '909']
    panel.wheel('#display', -1)  # the wheel down: the next kit
    assert wait_act(panel, 'kit.select', 2)[-1] == {'kit': 'user:Dub', 'browse': True}
    assert display(panel) == ['KIT 003  USER', 'DUB']
    panel.click('#preset-next')  # the last kit: VALUE stops there
    panel.click('#preset-prev', modifiers=Qt.KeyboardModifier.ShiftModifier)  # back to the factory kits
    assert wait_act(panel, 'kit.select', 3)[-1] == {'kit': 'factory:505', 'browse': True}
    panel.press_key(Qt.Key.Key_Left)
    assert wait_act(panel, 'kit.select', 4)[-1] == {'kit': '', 'browse': True}
    assert len(acts(core, 'kit.select')) == 4


def test_the_edited_kit_shows_a_star_and_lights_write(panel):
    kit_state(panel, current='factory:909', edited=True)
    panel.wait_js("pythonic.panel.display.text()[1] === 'KIT 909*'")
    assert panel.js("document.querySelector('#write-button').classList.contains('edited')")
    panel.click('#kit-button')
    panel.wait_js("pythonic.panel.display.text()[0] === 'KIT 002* FACTORY'")
    panel.core.post_change('kit.edited', False)
    panel.wait_js("pythonic.panel.display.text()[0] === 'KIT 002  FACTORY'")
    assert not panel.js("document.querySelector('#write-button').classList.contains('edited')")


def test_the_selector_follows_a_kit_the_core_switched_to(panel):
    kit_state(panel, current='factory:505')
    panel.click('#kit-button')
    panel.wait_js("pythonic.panel.display.text()[0] === 'KIT 001  FACTORY'")
    panel.core.post_change('kit.current', 'user:Dub')  # a pattern recalled it
    panel.wait_js("pythonic.panel.display.text()[0] === 'KIT 003  USER'")
    panel.press_key(Qt.Key.Key_Escape)
    panel.wait_js("!document.querySelector('#kit-button').classList.contains('on')")


# ---------------------------------------------------------------- INST

def test_inst_shows_the_channels_drum_patch_in_its_category(panel):
    inst_state(panel, channel=2, name='Kick 1')
    panel.click('#inst-button')
    panel.wait_js("pythonic.panel.display.text()[0] === 'INST 2 BD 3/3'")
    assert display(panel)[1] == 'KICK 1'
    assert lit(panel, '#inst-button') and not lit(panel, '#kit-button')


def test_value_loads_the_next_drum_patch_and_plays_it_while_stopped(panel):
    core = panel.core
    inst_state(panel, channel=2)
    panel.click('#inst-button')
    panel.wait_js("pythonic.panel.display.text()[0] === 'INST 2 BD 2/3'")
    panel.click('#preset-next')
    assert wait_act(panel, 'drum_patch.load') == [{'library': 'user:Kick 1.mtdrum', 'channel': 2, 'browse': True}]
    panel.qtbot.waitUntil(lambda: ('trigger', 1, 100) in core.calls)  # the core counts channels from 0
    assert display(panel) == ['INST 2 BD 3/3', 'KICK 1']
    # SHIFT: the next category
    panel.click('#preset-next', modifiers=Qt.KeyboardModifier.ShiftModifier)
    assert wait_act(panel, 'drum_patch.load', 2)[-1] == {'library': 'factory:909 SD', 'channel': 2, 'browse': True}
    assert display(panel) == ['INST 2 SD 1/1', '909 SD']
    # A plain step stays in the category: the only SD, so nothing loads
    panel.press_key(Qt.Key.Key_Down)
    panel.wait_js("pythonic.panel.display.text()[1] === 'last SD · shift: next category'")
    panel.wheel('#display', -1)
    assert len(acts(core, 'drum_patch.load')) == 2
    panel.press_key(Qt.Key.Key_Down, Qt.KeyboardModifier.ShiftModifier)
    assert wait_act(panel, 'drum_patch.load', 3)[-1]['library'] == 'factory:808 CH'
    panel.wait_js("pythonic.panel.display.text()[0] === 'INST 2 CH 1/2'")
    panel.press_key(Qt.Key.Key_Up)  # the first CH: stays, the SDs are a SHIFT away
    panel.wait_js("pythonic.panel.display.text()[1] === 'first CH · shift: previous category'")
    assert len(acts(core, 'drum_patch.load')) == 3


def test_a_channel_button_changes_the_channel_inst_chooses_for(panel):
    core = panel.core
    inst_state(panel, channel=1)
    core.post_change('ch4.name', 'Hat')
    panel.click('#inst-button')
    panel.wait_js("pythonic.panel.display.text()[0] === 'INST 1 BD 2/3'")
    panel.click('.strip[data-channel="4"] .chb')
    panel.wait_js("pythonic.panel.display.text()[0] === 'INST 4 CH 2/2'")
    assert display(panel)[1] == 'HAT'
    panel.click('#preset-prev')
    assert wait_act(panel, 'drum_patch.load') == [{'library': 'factory:808 CH', 'channel': 4, 'browse': True}]


# ---------------------------------------------------------------- WRITE

def test_write_as_asks_a_name_and_the_replace_question(panel):
    core = panel.core
    kit_state(panel, current='factory:909', edited=True)
    answers = iter([{'saved': False, 'exists': True, 'name': 'Dub'},
                    {'saved': True, 'exists': True, 'name': 'Dub', 'kit': 'user:Dub'}])
    core.verbs['kit.write'] = lambda name=None, overwrite=False: next(answers)
    panel.click('#write-button')
    panel.wait_js("!!document.querySelector('.px-menu .it')")
    assert items(panel, '.px-menu .it') == ['write', 'kit (a new name: write as)', 'kit as…', 'preset',
                                            'preset as…']
    click_item(panel, 'kit as…')
    alert_up(panel, 'ok')
    assert panel.js("document.querySelector('.alert-input').value") == '909 copy'
    panel.run("document.querySelector('.alert-input').value = 'Dub'")
    answer_alert(panel)
    assert wait_act(panel, 'kit.write') == [{'name': 'Dub'}]
    title, text = alert_up(panel, 'ok')
    assert title == 'Replace the kit?' and 'Dub' in text
    answer_alert(panel)
    assert wait_act(panel, 'kit.write', 2)[-1] == {'name': 'Dub', 'overwrite': True}
    panel.wait_js("pythonic.panel.display.text()[0] === 'WRITE KIT'")


def test_write_over_a_user_kit_asks_first(panel):
    kit_state(panel, current='user:Dub', edited=True)
    panel.click('#write-button')
    click_item(panel, 'kit “Dub”')
    title, _ = alert_up(panel, 'ok')
    assert title == 'Write the kit?'
    answer_alert(panel, primary=False)
    assert acts(panel.core, 'kit.write') == []
    panel.click('#write-button')
    click_item(panel, 'kit “Dub”')
    alert_up(panel, 'ok')
    answer_alert(panel)
    assert wait_act(panel, 'kit.write') == [{}]


def test_write_offers_the_programs_of_an_old_preset_as_kits(panel):
    panel.core.post_change('program.occupied', [True, True] + [False] * 14)
    panel.wait_js("(pythonic.store.value('program.occupied') || [])[0] === true")
    panel.click('#write-button')
    click_item(panel, 'programs as kits')
    assert wait_act(panel, 'kit.from_programs') == [{}]


# ---------------------------------------------------------------- patterns and the preset menu

def test_the_pattern_menu_sets_the_kit_a_pattern_recalls(panel):
    core = panel.core
    kit_state(panel, current='factory:909')
    panel.run("pythonic.panel.patterns.openMenu(1, 300, 300)")
    click_item(panel, 'recall kit “909”')
    panel.qtbot.waitUntil(lambda: ('pattern.B.kit', 'factory:909') in core.sets())
    panel.wait_js("document.querySelector('.pbtn[data-pattern=\"B\"] .kitmark').textContent === 'K'")
    assert panel.js("document.querySelector('.pbtn[data-pattern=\"B\"] .len').textContent") == ''
    panel.run("pythonic.panel.patterns.openMenu(1, 300, 300)")
    click_item(panel, 'recall no kit (now “909”)')
    panel.qtbot.waitUntil(lambda: ('pattern.B.kit', '') in core.sets())
    panel.wait_js("document.querySelector('.pbtn[data-pattern=\"B\"] .kitmark').textContent === ''")


def test_the_preset_menu_sets_the_drum_patch_folder(panel, tmp_path):
    panel.click('#preset-button')
    panel.wait_js("!!document.querySelector('#preset-menu')")
    labels = items(panel, '#preset-menu .pm-cmds .it')
    assert 'drum patch folder…' in labels and 'restore factory kits…' not in labels
    click_item(panel, 'drum patch folder…', '#preset-menu .it')
    panel.answer_dialog(tmp_path)
    panel.qtbot.waitUntil(lambda: ('pref.patch_folder', str(tmp_path)) in panel.core.sets())
