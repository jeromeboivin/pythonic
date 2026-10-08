"""Page tests of the step row, step entry, matrix and pattern controls (web
slice W3, map decision #8) over the fake core, with real pointer input: every
step mode and gesture, one undo gesture per stroke, pages and follow, the
matrix in the drawer, the pattern buttons and menu."""

from PySide6.QtCore import Qt

PAD = '.pads .stp[data-step="{n}"]'


def pad(n):
    return PAD.format(n=n)


def gestures(core):
    return [c[1] for c in core.calls if c[0] == 'gesture']


def step_sets(core):
    return [(a, v) for a, v in core.sets() if a.startswith('pattern.')]


def lane(core, field, channel=1, pattern='A'):
    return core.get(f'pattern.{pattern}.ch{channel}.{field}')


def seed(core, field, steps, value=True, channel=1, pattern='A'):
    """Steps written on the core's side (a preset, a verb): reported by poll."""
    for step in steps:
        core.patterns.set((pattern, channel, step, field), value)


def mode(panel, field):
    panel.click(f'[data-mode="{field}"]')
    panel.wait_js(f"document.querySelector('[data-mode=\"{field}\"]').classList.contains('on')")


def paint(panel, first, last, **kwargs):
    """Press on one pad and drag along the row to another, then release."""
    x0 = panel.center(pad(first)).x()
    x1 = panel.center(pad(last)).x()
    panel.drag(pad(first), dy=0, dx=x1 - x0, steps=max(2, abs(last - first) * 2), **kwargs)


def is_on(panel, n):
    return panel.js(f"document.querySelector('{pad(n)}').classList.contains('on')")


# ---------------------------------------------------------------- trigger, accent, fill

def test_a_pad_click_toggles_its_trigger_as_one_undo_step(panel, qtbot):
    core = panel.core
    panel.click(pad(3))
    qtbot.waitUntil(lambda: gestures(core) == ['begin', 'end'])
    assert step_sets(core) == [('pattern.A.ch1.step3.trig', True)]
    assert lane(core, 'trig')[2] is True
    panel.wait_js(f"document.querySelector('{pad(3)}').classList.contains('on')")
    # Lit in its beat group's colour (steps 1-4: red), the idle pad stays dark
    panel.wait_pixels(lambda: panel.color_at(pad(3), 0.5, 0.85)[0] > 150)
    assert panel.color_at(pad(2), 0.5, 0.85)[0] < 90


def test_a_paint_stroke_along_the_row_is_one_gesture(panel, qtbot):
    core = panel.core
    paint(panel, 2, 6)
    qtbot.waitUntil(lambda: gestures(core) == ['begin', 'end'])
    assert lane(core, 'trig')[:8] == [False, True, True, True, True, True, False, False]
    assert [a for a, _ in step_sets(core)] == [f'pattern.A.ch1.step{n}.trig' for n in range(2, 7)]
    # Painting back over lit pads from a lit one turns them off
    paint(panel, 3, 5)
    qtbot.waitUntil(lambda: gestures(core) == ['begin', 'end'] * 2)
    assert lane(core, 'trig')[:7] == [False, True, False, False, False, True, False]


def test_accent_and_fill_paint_only_triggered_steps(panel, qtbot):
    core = panel.core
    seed(core, 'trig', [1, 3, 5])
    panel.wait_js(f"document.querySelector('{pad(5)}').classList.contains('on')")
    mode(panel, 'acc')
    paint(panel, 1, 5)
    qtbot.waitUntil(lambda: gestures(core) == ['begin', 'end'])
    assert lane(core, 'acc')[:6] == [True, False, True, False, True, False]
    panel.wait_js(f"document.querySelector('{pad(3)}').classList.contains('accent')")
    mode(panel, 'fill')
    panel.click(pad(3))
    qtbot.waitUntil(lambda: lane(core, 'fill')[2] is True)
    panel.click(pad(2))  # no trigger: nothing
    qtbot.wait(100)
    assert lane(core, 'fill')[1] is False


# ---------------------------------------------------------------- velocity, probability

def test_a_velocity_drag_on_a_lit_pad_is_one_gesture(panel, qtbot):
    core = panel.core
    seed(core, 'trig', [5])
    panel.wait_js(f"document.querySelector('{pad(5)}').classList.contains('on')")
    mode(panel, 'vel')
    panel.wait_js(f"document.querySelector('{pad(5)} .pv').textContent === '64'")
    panel.drag(pad(5), dy=-40)  # 40 view px = 50 design px at 0.8 = a quarter of 1-127
    qtbot.waitUntil(lambda: gestures(core) == ['begin', 'end'])
    assert lane(core, 'vel')[4] == 96
    assert {a for a, _ in step_sets(core)} == {'pattern.A.ch1.step5.vel'}
    panel.wait_js(f"document.querySelector('{pad(5)} .pv').textContent === '96'")
    panel.drag(pad(6), dy=-40)  # no trigger: no velocity
    qtbot.wait(100)
    assert gestures(core) == ['begin', 'end']


def test_a_probability_drag_works_on_any_pad_and_shift_is_fine(panel, qtbot):
    core = panel.core
    mode(panel, 'prob')
    panel.drag(pad(7), dy=80)  # down 100 design px = -50 %
    qtbot.waitUntil(lambda: gestures(core) == ['begin', 'end'])
    assert lane(core, 'prob')[6] == 50
    panel.wait_js(f"document.querySelector('{pad(7)} .prob').textContent === '50%'")
    panel.drag(pad(8), dy=80, modifiers=Qt.KeyboardModifier.ShiftModifier)
    qtbot.waitUntil(lambda: gestures(core) == ['begin', 'end'] * 2)
    assert lane(core, 'prob')[7] == 95


# ---------------------------------------------------------------- substeps, last step, all ch

def test_the_substep_popover_sets_a_preset(panel, qtbot):
    core = panel.core
    mode(panel, 'sub')
    panel.click(pad(9))
    panel.wait_js("!!document.querySelector('.px-menu.subs')")
    panel.run("[...document.querySelectorAll('.px-menu .it')].find((i) => i.textContent === 'o-o').click()")
    qtbot.waitUntil(lambda: lane(core, 'sub')[8] == 'o-o')
    panel.wait_js(f"document.querySelectorAll('{pad(9)} .sub i').length === 3")


def test_last_step_then_a_pad_on_another_page_sets_the_length(panel, qtbot):
    core = panel.core
    panel.click('#last-step')
    panel.click('.pg[data-page="1"]')
    panel.wait_js(f"document.querySelector('{pad(17)}') !== null")
    panel.click(pad(24))
    qtbot.waitUntil(lambda: core.values['pattern.A.length'] == 24)
    assert len(lane(core, 'trig')) == 24
    panel.wait_js("document.querySelector('.pnums .end')?.textContent === '24]'")
    assert not panel.js("document.querySelector('#last-step').classList.contains('on')")
    assert panel.js(f"document.querySelector('{pad(25)}').classList.contains('out')")


def test_all_ch_paints_all_eight_channels_in_one_gesture(panel, qtbot):
    core = panel.core
    core.post_change('ch4.mute', True)  # muted channels too
    panel.click('#all-ch')
    paint(panel, 1, 2)
    qtbot.waitUntil(lambda: gestures(core) == ['begin', 'end'])
    for channel in range(1, 9):
        assert lane(core, 'trig', channel)[:3] == [True, True, False]


# ---------------------------------------------------------------- pages, follow, playhead

def test_follow_tracks_the_playhead_and_a_page_pick_stops_it(panel, qtbot):
    core = panel.core
    core.set('pattern.A.length', 64)
    core.transport.update(playing=True, position=40)
    panel.wait_js("document.querySelector('.pads .stp').dataset.step === '33'")
    panel.wait_js(f"document.querySelector('{pad(41)}').classList.contains('ph')")
    assert panel.js("document.querySelector('.pg[data-page=\"2\"]').classList.contains('play')")
    panel.click('.pg[data-page="0"]')
    panel.wait_js("document.querySelector('.pads .stp').dataset.step === '1'")
    assert not panel.js("document.querySelector('#follow').classList.contains('on')")
    core.transport.update(position=50)
    qtbot.wait(100)
    assert panel.js("document.querySelector('.pads .stp').dataset.step") == '1'
    panel.click('#follow')
    panel.wait_js("document.querySelector('.pads .stp').dataset.step === '49'")
    panel.wait_js("pythonic.panel.display.base[0] === 'PATTERN A  49-64'")


# ---------------------------------------------------------------- matrix

def test_the_matrix_opens_in_the_drawer_and_paints_a_row(panel, qtbot):
    core = panel.core
    seed(core, 'trig', [1], channel=6)
    panel.click('#matrix-toggle')
    panel.wait_js("pythonic.panel.drawer.current === 'matrix'")
    cell = '.matrix .stp[data-channel="{c}"][data-step="{s}"]'
    panel.wait_js(f"document.querySelector('{cell.format(c=6, s=1)}').classList.contains('on')")
    x0 = panel.center(cell.format(c=3, s=2)).x()
    x1 = panel.center(cell.format(c=3, s=5)).x()
    panel.drag(cell.format(c=3, s=2), dy=0, dx=x1 - x0, steps=6)
    qtbot.waitUntil(lambda: gestures(core) == ['begin', 'end'])
    assert lane(core, 'trig', 3)[:6] == [False, True, True, True, True, False]
    assert lane(core, 'trig', 1)[:6] == [False] * 6
    panel.click('.mlbl[data-channel="6"]')
    qtbot.waitUntil(lambda: core.values['global.channel'] == 6)
    panel.wait_js(f"document.querySelector('{pad(1)}').classList.contains('on')")  # ch6 on the pads
    panel.click('#matrix-toggle')
    panel.wait_js("pythonic.panel.drawer.current === null")
    assert panel.js("document.querySelector('.matrix')") is None


# ---------------------------------------------------------------- patterns

def test_pattern_buttons_select_and_show_the_transport(panel, qtbot):
    core = panel.core
    panel.click('.pbtn[data-pattern="C"]')
    qtbot.waitUntil(lambda: ('pattern.select', {'pattern': 'C'}) in
                    [(c[1], c[2]) for c in core.calls if c[0] == 'act'])
    panel.wait_js("document.querySelector('.pbtn[data-pattern=\"C\"]').classList.contains('on')")
    seed(core, 'trig', [4], pattern='C')
    panel.wait_js(f"document.querySelector('{pad(4)}').classList.contains('on')")
    panel.click(pad(6))
    qtbot.waitUntil(lambda: lane(core, 'trig', pattern='C')[5] is True)
    core.transport.update(playing=True, playing_pattern=0, queued_pattern=2)
    panel.wait_js("document.querySelector('.pbtn[data-pattern=\"A\"]').classList.contains('playing')")
    assert panel.js("document.querySelector('.pbtn[data-pattern=\"C\"]').classList.contains('queued')")
    assert panel.js("document.querySelectorAll('.pads .ph').length") == 0  # A plays, C is shown


def test_the_pattern_menu_runs_a_verb_and_shows_the_result(panel, qtbot):
    core = panel.core
    core.verbs['pattern.paste'] = lambda pattern=None: {'pasted': False}
    panel.click('#pattern-menu')
    panel.wait_js("!!document.querySelector('.px-menu.pmenu')")
    panel.run("[...document.querySelectorAll('.px-menu .it')].find((i) => i.textContent === 'paste').click()")
    qtbot.waitUntil(lambda: 'pattern.paste' in core.verbs_called())
    panel.wait_js("pythonic.panel.display.text()[1] === 'nothing to paste'")
    panel.right_click('.pbtn[data-pattern="E"]')
    panel.wait_js("document.querySelector('.px-menu .it')?.textContent === 'PATTERN E'")
    panel.run("[...document.querySelectorAll('.px-menu .it')].find((i) => i.textContent === 'shift left').click()")
    qtbot.waitUntil(lambda: ('pattern.shift_left', {'pattern': 'E'}) in
                    [(c[1], c[2]) for c in core.calls if c[0] == 'act'])


def test_a_failed_pattern_verb_alerts_on_the_display(panel, qtbot):
    def fail(pattern=None):
        raise RuntimeError('stream stalled')
    panel.core.verbs['pattern.reverse'] = fail
    panel.click('#pattern-menu')
    panel.wait_js("!!document.querySelector('.px-menu.pmenu')")
    panel.run("[...document.querySelectorAll('.px-menu .it')].find((i) => i.textContent === 'reverse').click()")
    panel.wait_js("pythonic.panel.display.text()[0] === 'PATTERN.REVERSE'")
    assert panel.js("pythonic.panel.display.text()[1]") == 'stream stalled'


def test_chain_and_lane_buttons_run_their_verbs(panel, qtbot):
    core = panel.core
    core.post_change('global.channel', 3)
    panel.wait_js("pythonic.store.value('global.channel') === 3")
    for button in ('#chain-next', '#lane-copy', '#lane-paste'):
        panel.click(button)
    qtbot.waitUntil(lambda: len([c for c in core.calls if c[0] == 'act']) == 3)
    assert [(c[1], c[2]) for c in core.calls if c[0] == 'act'] == [
        ('pattern.chain_next', {'pattern': 'A'}),
        ('pattern.copy_lane', {'pattern': 'A', 'channel': 3}),
        ('pattern.paste_lane', {'pattern': 'A', 'channel': 3})]


# ---------------------------------------------------------------- kit mode

def kit_state(panel, current=3, names=('505', '707', '808', '909', 'DMX', 'LM2')):
    """Programs 1-6 holding the factory kits, ``current`` playing, as the page sees them."""
    core = panel.core
    core.post_change('program.names', list(names) + [''] * (16 - len(names)))
    core.post_change('program.occupied', [True] * len(names) + [False] * (16 - len(names)))
    core.post_change('program.current', current)
    panel.wait_js(f"pythonic.store.value('program.current') === {current}"
                  f" && (pythonic.store.value('program.names') || [])[0] === {names[0]!r}")


def kits(panel, selector='.kits .kit'):
    return panel.js(f"[...document.querySelectorAll('{selector}')].map((k) => k.dataset.program)")


def test_kit_mode_shows_the_programs_on_the_pads(panel, qtbot):
    kit_state(panel)
    panel.click('#kit-mode')
    panel.wait_js("document.querySelector('.slot-steps').dataset.select === 'kit'")
    assert panel.js("getComputedStyle(document.querySelector('.pads')).display") == 'none'
    assert panel.js("[...document.querySelectorAll('.kits .kit .kn')].slice(0, 7).map((k) => k.textContent)") == \
        ['505', '707', '808', '909', 'DMX', 'LM2', '']
    assert kits(panel, '.kits .kit.on') == ['1', '2', '3', '4', '5', '6']
    assert kits(panel, '.kits .kit.cur') == ['3']
    assert panel.js("document.querySelector('#kit-mode').classList.contains('on')")
    assert not panel.js("document.querySelector('[data-mode].on')")


def test_a_kit_pad_switches_program_and_the_pads_follow(panel, qtbot):
    core = panel.core
    kit_state(panel)
    panel.click('#kit-mode')
    panel.wait_js("document.querySelector('.slot-steps').dataset.select === 'kit'")
    panel.click('.kits .kit[data-program="4"]')
    qtbot.waitUntil(lambda: ('act', 'program.select', {'program': 4}) in core.calls)
    panel.wait_js("pythonic.panel.display.text()[1] === '4 909'")
    core.post_change('program.current', 4)
    panel.wait_js("document.querySelector('.kits .kit.cur').dataset.program === '4'")
    # an empty program: a copy of the current sounds
    panel.click('.kits .kit[data-program="9"]')
    qtbot.waitUntil(lambda: ('act', 'program.select', {'program': 9}) in core.calls)
    panel.wait_js("pythonic.panel.display.text()[1] === '9 new: a copy'")
    assert not step_sets(core)


def test_a_step_mode_or_the_kit_button_goes_back_to_the_steps(panel, qtbot):
    kit_state(panel)
    panel.click('#kit-mode')
    panel.wait_js("document.querySelector('.slot-steps').dataset.select === 'kit'")
    panel.click('#kit-mode')
    panel.wait_js("!document.querySelector('.slot-steps').dataset.select")
    assert panel.js("document.querySelector('[data-mode=\"trig\"]').classList.contains('on')")
    panel.click('#kit-mode')
    panel.wait_js("document.querySelector('.slot-steps').dataset.select === 'kit'")
    mode(panel, 'acc')
    assert not panel.js("document.querySelector('.slot-steps').dataset.select")
    panel.core.post_change('pattern.A.length', 32)
    panel.wait_js("pythonic.store.value('pattern.A.length') === 32")
    panel.click('#kit-mode')
    panel.wait_js("document.querySelector('.slot-steps').dataset.select === 'kit'")
    panel.click('.pg[data-page="1"]')
    panel.wait_js("!document.querySelector('.slot-steps').dataset.select && pythonic.panel.steps.state.page === 1")


# ---------------------------------------------------------------- inst mode

FACTORY_PATCHES = ['505 BD', '505 Tom High', '707 BD 1', '707 BD 2', '808 BD', '808 MT', '808 SD', '909 BD',
                   '909 Tom Low', 'DMX BD', 'DMX Tom', 'LM2 BD']


def inst_state(panel, name='808 BD', channel=1, patches=FACTORY_PATCHES):
    core = panel.core
    core.post_change('factory.patches', list(patches))
    core.post_change(f'ch{channel}.name', name)
    core.post_change('global.channel', channel)
    panel.wait_js(f"pythonic.store.value('ch{channel}.name') === {name!r}"
                  f" && (pythonic.store.value('factory.patches') || []).length === {len(patches)}"
                  f" && pythonic.store.value('global.channel') === {channel}")


def inst_pads(panel, selector='.kits .kit.on'):
    return panel.js(f"[...document.querySelectorAll('{selector}')].map((k) => k.dataset.patch)")


def test_inst_mode_offers_the_factory_sounds_of_the_channels_drum_type(panel, qtbot):
    inst_state(panel)
    panel.click('#inst-mode')
    panel.wait_js("document.querySelector('.slot-steps').dataset.select === 'inst'")
    assert inst_pads(panel) == ['505 BD', '707 BD 1', '707 BD 2', '808 BD', '909 BD', 'DMX BD', 'LM2 BD']
    assert inst_pads(panel, '.kits .kit.cur') == ['808 BD']
    assert panel.js("[...document.querySelectorAll('.kits .kit')].slice(0, 2)"
                    ".map((k) => k.querySelector('b').textContent + '|' + k.querySelector('.kn').textContent)") == \
        ['505|BD', '707|BD 1']
    assert panel.js("pythonic.panel.display.text()") == ['INST CH1', 'BD: 7 factory sounds']
    assert panel.js("document.querySelector('#inst-mode').classList.contains('on')")

    # another channel: its own drum family (toms stand in for each other)
    inst_state(panel, name='909 Tom Mid', channel=3)
    panel.wait_js("document.querySelector('.kits .kit.on').dataset.patch === '505 Tom High'")
    assert inst_pads(panel) == ['505 Tom High', '808 MT', '909 Tom Low', 'DMX Tom']


def test_an_inst_pad_loads_the_sound_into_the_channel_and_plays_it(panel, qtbot):
    core = panel.core
    inst_state(panel, channel=2)
    panel.click('#inst-mode')
    panel.wait_js("document.querySelector('.slot-steps').dataset.select === 'inst'")
    panel.click('.kits .kit[data-patch="909 BD"]')
    qtbot.waitUntil(lambda: ('act', 'drum_patch.load', {'factory': '909 BD', 'channel': 2}) in core.calls)
    qtbot.waitUntil(lambda: ('trigger', 1, 100) in core.calls)  # the core counts channels from 0
    panel.wait_js("pythonic.panel.display.text()[1] === '909 BD'")
    assert not step_sets(core)


def test_without_a_drum_type_inst_mode_pages_every_factory_sound(panel, qtbot):
    names = [f'{m} Sound {i}' for m in ('505', '707') for i in range(1, 13)]  # 24 names, no drum type
    inst_state(panel, name='Init', patches=names)
    panel.click('#inst-mode')
    panel.wait_js("document.querySelector('.slot-steps').dataset.select === 'inst'")
    assert len(inst_pads(panel)) == 16
    assert panel.js("[...document.querySelectorAll('.pg.out')].map((b) => b.dataset.page)") == ['2', '3']
    panel.click('.pg[data-page="1"]')
    panel.wait_js("document.querySelector('.kits .kit.on').dataset.patch === '707 Sound 5'")
    assert len(inst_pads(panel)) == 8
    assert panel.js("document.querySelector('.slot-steps').dataset.select") == 'inst'
    panel.click('.pg[data-page="3"]')  # past the sounds: back to the steps
    panel.wait_js("!document.querySelector('.slot-steps').dataset.select")


def test_the_drum_patch_menu_loads_a_factory_sound_by_machine(panel, qtbot):
    core = panel.core
    core.verbs['drum_patch.load'] = lambda factory, channel: {'channel': channel, 'name': factory, 'path': None}
    inst_state(panel, channel=4)
    panel.click('#patch-menu')
    panel.wait_js("[...document.querySelectorAll('.px-menu .it')].some((i) => i.textContent === 'factory drum patch into CH4…')")
    pick(panel, 'factory drum patch into CH4…')
    panel.wait_js("[...document.querySelectorAll('.px-menu .it')].some((i) => i.textContent === 'DMX')")
    assert panel.js("[...document.querySelectorAll('.px-menu .it:not(.dis)')].map((i) => i.textContent)") == \
        ['505', '707', '808', '909', 'DMX', 'LM2']
    pick(panel, '909')
    panel.wait_js("[...document.querySelectorAll('.px-menu .it')].some((i) => i.textContent === '909 Tom Low')")
    assert panel.js("[...document.querySelectorAll('.px-menu .it.cur')].map((i) => i.textContent)") == []
    pick(panel, '909 Tom Low')
    qtbot.waitUntil(lambda: ('act', 'drum_patch.load', {'factory': '909 Tom Low', 'channel': 4}) in core.calls)
    panel.wait_js("pythonic.panel.display.text()[1] === '909 TOM LOW'")


def pick(panel, label):
    panel.run(f"[...document.querySelectorAll('.px-menu .it')].forEach((i) => "
              f"{{ delete i.dataset.pick; if (i.textContent === {label!r}) i.dataset.pick = 'yes'; }})")
    panel.click('.px-menu [data-pick="yes"]')
