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
