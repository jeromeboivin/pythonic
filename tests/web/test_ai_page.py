"""Page tests of the AI generator page (web slice W8) over the fake core: a
drawer page opened from the PRESET menu whose lanes sit under the strips
(#21, #13), every state (ML extras missing and their install, idle, a lane
generating, a new sound playing on the face, an error), one new sound per
lane with old | new, new patterns played on the face with old | new, keep
and revert, and the keep-or-revert question on every way out (the page's
own buttons, another drawer page, a preset load)."""

import pytest

from tests.web.test_edit_rack_page import window_is
from tests.web.test_menus_page import (FILES, alert_up, answer_alert, click_item, display, folder_state,
                                       open_preset_menu)


def acts(core, verb):
    return [c[2] for c in core.calls if c[:2] == ('act', verb)]


def wait_act(panel, verb, count=1):
    panel.qtbot.waitUntil(lambda: len(acts(panel.core, verb)) >= count)
    return acts(panel.core, verb)


def text(panel, selector):
    return panel.js(f"document.querySelector({selector!r}).textContent")


def lane(n, part=''):
    return f'.ai-lane[data-channel="{n}"] {part}'.strip()


def post(core, **values):
    """Core-side changes: keyword names use __ for dots (ai__ch2__trying)."""
    for name, value in values.items():
        core.post_change(name.replace('__', '.'), value)


def tried(core, channel, name, n=8, i=1):
    """The core tried candidate i of n on a channel (its name on the strip)."""
    post(core, **{f'ai__ch{channel}__candidates': n, f'ai__ch{channel}__candidate': i,
                  f'ai__ch{channel}__name': name, f'ai__ch{channel}__trying': True,
                  f'ai__ch{channel}__generating': False, f'ch{channel}__name': name})
    core.post_change('ai.tried', sorted(set(core.values.get('ai.tried') or []) | {channel}))


def open_ai(panel, available=True):
    post(panel.core, ai__available=available, ai__state='idle' if available else 'unavailable')
    panel.wait_js(f"pythonic.store.value('ai.available') === {str(available).lower()}")
    open_preset_menu(panel)
    click_item(panel, 'AI drum generator…')
    panel.wait_js("pythonic.panel.drawer.current === 'ai' && !!document.querySelector('.slot-rack .aipage')")
    painted(panel)


def painted(panel):
    """Wait for two frames: input reaches what the page shows once it is painted."""
    panel.run('window.__painted = false; requestAnimationFrame(() => '
              'requestAnimationFrame(() => { window.__painted = true; }));')
    panel.wait_js('window.__painted')


def buttons(panel):
    return panel.js("[...document.querySelectorAll('.alert-buttons .btn')].map((b) => b.textContent)")


def answer(panel, label):
    panel.run(f"[...document.querySelectorAll('.alert-buttons .btn')].find((b) => b.textContent === {label!r})"
              ".dataset.pick = 'yes'")
    panel.click('.alert-buttons [data-pick="yes"]')
    panel.wait_js("!document.querySelector('.sheet-layer[data-sheet=\"alert\"]')")


# ---------------------------------------------------------------- the page

def test_the_page_opens_in_the_drawer_with_a_lane_under_each_strip(panel):
    open_ai(panel)
    assert text(panel, '#ai-back') == '◀ ch 1 edit'
    for n in range(1, 9):
        sx, _, sw, _ = panel.rect(f'.strip[data-channel="{n}"]')
        lx, _, lw, _ = panel.rect(lane(n))
        assert lx == pytest.approx(sx, abs=1.5) and lw == pytest.approx(sw, abs=1.5)
    # The left and right columns sit under the face's, and nothing spills out of a column
    assert panel.rect('.ai-right')[0] == pytest.approx(panel.rect('.col.right')[0], abs=1.5)
    spill = panel.js("""[...document.querySelectorAll('.ai-sec')].flatMap((sec) => {
        const box = sec.getBoundingClientRect();
        return [...sec.querySelectorAll('button, input, px-knob, px-list, .st, .cname')].filter((e) => {
          const r = e.getBoundingClientRect();
          return r.width && (r.left < box.left - 0.5 || r.right > box.right + 0.5 || r.bottom > box.bottom + 0.5);
        }).map((e) => e.id || e.className || e.tagName);
      })""")
    assert spill == []
    # The saved (or bundled) models load as the page opens (tkinter)
    assert sorted(a['kind'] for a in wait_act(panel, 'ai.load_model', 2)) == ['patch', 'pattern']
    panel.core.post_change('global.channel', 3)
    panel.wait_js("document.querySelector('#ai-back').textContent === '◀ ch 3 edit'")
    panel.click('#ai-close')
    panel.wait_js("pythonic.panel.drawer.current === null && !!document.querySelector('.slot-rack .rack')")


def test_the_page_opens_a_closed_rack_and_closing_it_restores(panel, qtbot):
    panel.click('#rack-toggle')
    window_is(panel, qtbot, 560)
    open_ai(panel)
    window_is(panel, qtbot, 800)
    panel.click('#ai-back')
    window_is(panel, qtbot, 560)


def test_the_settings_are_the_shared_ai_preferences_and_models_open_setup(panel):
    open_ai(panel)
    bound = panel.bound_addresses()
    assert {'pref.ai.patch_temperature', 'pref.ai.pattern_temperature', 'ai.models'} <= bound
    for n in range(1, 9):
        assert {f'ai.ch{n}.{f}' for f in ('type', 'candidates', 'name', 'trying', 'generating')} <= bound
    assert text(panel, '.ai-model-note') == ''
    panel.core.post_change('ai.models', {
        'patch': {'path': '/m/kick.pt', 'bundled': True, 'status': 'loaded', 'error': None, 'sampling': 'prior'},
        'pattern': {'path': '/m/pat.pt', 'bundled': True, 'status': 'error', 'error': 'broken', 'sampling': None}})
    panel.wait_js("document.querySelector('.ai-model-note').textContent === 'pattern model: broken'")
    panel.click('#ai-models')
    panel.wait_js("!!document.querySelector('.sheet-layer[data-sheet=\"setup\"]')")


# ---------------------------------------------------------------- states

def test_without_the_ml_extras_the_lanes_show_the_install_command(panel):
    core = panel.core
    core.post_change('ai.install_command', 'python -m pip install torch')
    open_ai(panel, available=False)
    panel.wait_js("getComputedStyle(document.querySelector('.ai-install')).display === 'flex'")
    assert panel.js("getComputedStyle(document.querySelector('.ai-lanes')).display") == 'none'
    assert text(panel, '.ai-command') == 'python -m pip install torch'
    assert text(panel, '.ai-state') == 'ML extras missing'
    assert panel.js("document.querySelector('#ai-generate-all').disabled")
    assert not acts(core, 'ai.load_model')
    panel.click('#ai-copy')
    panel.wait_js("pythonic.panel.display.text()[0] === 'INSTALL COMMAND'")

    core.verbs['ai.install'] = lambda: core.DEFERRED
    panel.click('#ai-install')
    title, body = alert_up(panel, 'ok')
    assert title == 'Install the ML extras?' and 'python -m pip install torch' in body
    answer_alert(panel)
    wait_act(panel, 'ai.install')
    post(core, ai__installing=True, ai__state='installing')
    panel.wait_js("document.querySelector('#ai-install').disabled")
    assert text(panel, '#ai-install') == 'installing…'
    post(core, ai__installing=False, ai__available=True, ai__state='idle')
    core.finish(core.running, {'installed': True, 'output': 'ok'})
    assert alert_up(panel, 'ok')[0] == 'ML extras installed'
    answer_alert(panel)
    panel.wait_js("getComputedStyle(document.querySelector('.ai-lanes')).display === 'grid'")
    wait_act(panel, 'ai.load_model', 2)  # the models load once the extras are there


def test_the_standalone_app_points_to_the_ai_edition_instead_of_pip(panel):
    core = panel.core
    core.post_change('ai.install_command', '')
    open_ai(panel, available=False)
    panel.wait_js("getComputedStyle(document.querySelector('.ai-install')).display === 'flex'")
    assert text(panel, '.ai-install-title') == 'This edition of Pythonic has no AI generator'
    assert 'Pythonic + AI edition' in text(panel, '.ai-install .note')
    assert panel.js("document.querySelector('#ai-install').offsetParent === null")  # hidden


def test_a_failed_install_says_why(panel):
    core = panel.core
    open_ai(panel, available=False)
    core.verbs['ai.install'] = lambda: (_ for _ in ()).throw(RuntimeError('the install failed:\nno network'))
    panel.click('#ai-install')
    alert_up(panel, 'ok')
    answer_alert(panel)
    assert alert_up(panel, 'error') == ('The install failed', 'the install failed:\nno network')
    answer_alert(panel)


def test_gen_makes_one_new_sound_and_plays_it_on_the_face(panel):
    core = panel.core
    open_ai(panel)
    assert text(panel, lane(2, '.cname')) == ''
    assert panel.js(f"getComputedStyle(document.querySelector('{lane(2, '.ab')}')).visibility") == 'hidden'
    core.verbs['ai.generate'] = lambda **args: core.DEFERRED
    panel.click(lane(2, '.gen'))
    (args,) = wait_act(panel, 'ai.generate')
    assert args == {'channel': 2, 'type': core.values['ai.ch2.type']}  # one sound, a fresh seed
    post(core, ai__ch2__generating=True, ai__state='generating')
    panel.wait_js(f"document.querySelector('{lane(2, '.lnote')}').textContent === 'generating…'")
    assert text(panel, '.ai-state') == 'generating…'
    assert panel.js(f"document.querySelector('{lane(2)}').dataset.state") == 'generating'
    assert panel.js(f"document.querySelector('{lane(2, '.gen')}').disabled")

    tried(core, 2, 'SD 1', n=1)
    post(core, ai__state='idle')
    core.finish(core.running, {'channels': [2], 'failed': []})
    panel.wait_js(f"document.querySelector('{lane(2, '.cname')}').textContent === 'SD 1'")
    assert panel.js(f"document.querySelector('{lane(2, '.new')}').classList.contains('on')")
    assert not panel.js(f"document.querySelector('{lane(2, '.old')}').classList.contains('on')")
    assert text(panel, lane(2, '.lnote')) == 'the new sound plays'
    # The strip tab shows the new sound in italics
    panel.wait_js("document.querySelector('.strip[data-channel=\"2\"] .tab').classList.contains('aitry')")
    assert text(panel, '.strip[data-channel="2"] .tab') == 'SD 1'
    assert panel.js("getComputedStyle(document.querySelector('.strip[data-channel=\"2\"] .tab')).fontStyle") == 'italic'
    assert text(panel, '.ai-tried') == 'new sounds on ch 2'
    assert ('trigger', 1, 127) in core.calls  # heard at once while stopped


def test_old_and_new_switch_a_lane_between_its_sounds(panel):
    core = panel.core
    open_ai(panel)
    tried(core, 4, 'OH 1', n=1)
    panel.wait_js(f"!document.querySelector('{lane(4, '.old')}').disabled")
    panel.click(lane(4, '.new'))  # already playing: nothing to do
    panel.click(lane(4, '.old'))
    assert wait_act(panel, 'ai.untry') == [{'channel': 4}]
    assert not acts(core, 'ai.try')
    post(core, ai__ch4__trying=False, **{'ai.tried': []})
    panel.wait_js(f"document.querySelector('{lane(4, '.old')}').classList.contains('on')")
    assert text(panel, lane(4, '.lnote')) == 'the old sound plays'
    assert not panel.js("document.querySelector('.strip[data-channel=\"4\"] .tab').classList.contains('aitry')")
    core.verbs['ai.try'] = lambda channel, candidate=None, step=0: {'channel': channel, 'candidate': 1, 'name': 'OH 1'}
    panel.click(lane(4, '.new'))
    assert wait_act(panel, 'ai.try') == [{'channel': 4}]
    panel.wait_js("pythonic.panel.display.text()[0] === 'CH4 OH 1'")


def test_a_lane_error_and_a_failed_generation(panel):
    core = panel.core
    open_ai(panel)
    post(core, ai__ch3__error='no drum patch model')
    panel.wait_js(f"document.querySelector('{lane(3, '.lnote')}').textContent === 'error: no drum patch model'")
    assert panel.js(f"document.querySelector('{lane(3, '.lnote')}').classList.contains('bad')")
    core.verbs['ai.generate'] = lambda **args: (_ for _ in ()).throw(ValueError('No drum patch model is available.'))
    panel.click(lane(3, '.gen'))
    assert alert_up(panel, 'error') == ('Generation failed', 'No drum patch model is available.')
    answer_alert(panel)
    assert display(panel)[0] == 'AI.GENERATE'


def test_the_drum_type_of_a_lane_is_any_of_the_18(panel):
    core = panel.core
    open_ai(panel)
    panel.click(lane(5, 'px-list .lbox'))
    panel.wait_js("document.querySelectorAll('.px-menu .it').length === 18")
    click_item(panel, 'perc')
    panel.qtbot.waitUntil(lambda: ('ai.ch5.type', 'perc') in core.sets())
    panel.wait_js(f"document.querySelector('{lane(5, 'px-list .lbox')}').textContent === 'perc'")


def test_generate_8_sounds_makes_one_per_lane(panel):
    core = panel.core
    open_ai(panel)
    core.verbs['ai.generate'] = lambda **args: {'channels': list(range(1, 8)), 'failed': [{'channel': 8, 'error': 'x'}]}
    panel.click('#ai-generate-all')
    assert wait_act(panel, 'ai.generate') == [{}]
    panel.wait_js("pythonic.panel.display.text()[1] === '7 new, 1 failed'")
    assert not acts(core, 'ai.generate_patterns')  # patterns have their own button


def test_generate_patterns_plays_them_and_old_new_switches(panel):
    core = panel.core
    open_ai(panel)
    assert text(panel, '.ai-patnote') == 'make 12 patterns (A–L) for the sounds on the face'
    assert panel.js("document.querySelector('#ai-pat-new').disabled")
    assert panel.js("document.querySelector('#ai-keep').disabled && document.querySelector('#ai-revert').disabled")
    core.verbs['ai.generate_patterns'] = lambda **args: core.DEFERRED
    core.verbs['ai.pattern_try'] = lambda mode=None, bank=True: {'preview': mode or 'off', 'pattern': 'A'}
    panel.click('#ai-gen-patterns')
    assert wait_act(panel, 'ai.generate_patterns') == [{}]
    post(core, ai__bank='generating')
    panel.wait_js("document.querySelector('.ai-patnote').textContent === 'generating patterns…'")
    assert panel.js("document.querySelector('#ai-gen-patterns').disabled")
    post(core, ai__bank='ready')
    core.finish(core.running, {'patterns': 12})
    assert wait_act(panel, 'ai.pattern_try') == [{'mode': 'try'}]  # played at once
    post(core, ai__preview='try')
    panel.wait_js("document.querySelector('#ai-pat-new').classList.contains('on')")
    assert text(panel, '.ai-patnote').startswith('the new patterns play')
    assert text(panel, '.ai-tried') == 'new patterns'
    assert not panel.js("document.querySelector('#ai-keep').disabled")
    panel.wait_js("pythonic.panel.display.text()[1] === '12 new patterns'")

    panel.click('#ai-pat-old')
    assert wait_act(panel, 'ai.pattern_try', 2)[1] == {'mode': None}
    post(core, ai__preview='off')
    panel.wait_js("document.querySelector('#ai-pat-old').classList.contains('on')")
    assert text(panel, '.ai-patnote') == 'the old patterns play'
    panel.click('#ai-pat-new')
    assert wait_act(panel, 'ai.pattern_try', 3)[2] == {'mode': 'try'}


def test_keep_commits_what_plays_then_forgets_the_new_ones(panel):
    core = panel.core
    open_ai(panel)
    tried(core, 1, 'BD 3', n=1)
    tried(core, 3, 'CH 2', n=1)
    panel.wait_js("!document.querySelector('#ai-keep').disabled")
    core.verbs['ai.keep'] = lambda channels=None: {'kept': [1, 3]}
    panel.click('#ai-keep')
    assert wait_act(panel, 'ai.keep') == [{}]
    assert wait_act(panel, 'ai.clear') == [{}]
    panel.wait_js("pythonic.panel.display.text()[1] === 'kept 2 sounds'")

    # With the new patterns playing, keep takes them too (one undo step)
    post(core, ai__bank='ready', ai__preview='try')
    panel.wait_js("pythonic.store.value('ai.preview') === 'try'")
    core.verbs['ai.replace_patterns'] = lambda channels=None: {'kept': [1], 'patterns': 12}
    panel.click('#ai-keep')
    assert wait_act(panel, 'ai.replace_patterns') == [{}]
    assert len(wait_act(panel, 'ai.clear', 2)) == 2
    panel.wait_js("pythonic.panel.display.text()[1] === 'kept ch 1 + 12 patterns'")


def test_revert_puts_the_old_ones_back_and_forgets_the_new_ones(panel):
    core = panel.core
    open_ai(panel)
    tried(core, 5, 'TOM 1', n=1)
    panel.wait_js("!document.querySelector('#ai-revert').disabled")
    panel.click('#ai-revert')
    assert wait_act(panel, 'ai.clear') == [{}]
    assert not acts(core, 'ai.keep')
    panel.wait_js("pythonic.panel.display.text()[1] === 'reverted'")


# ---------------------------------------------------------------- leaving

def test_closing_with_tried_sounds_asks_keep_revert_or_cancel(panel):
    core = panel.core
    open_ai(panel)
    tried(core, 2, 'SD 5')
    panel.wait_js("pythonic.store.value('ai.tried').length === 1")
    panel.click('#ai-close')
    title, body = alert_up(panel, 'ok')
    assert title == 'Keep what is new?' and body.startswith('CH2 SD 5 plays on the face')
    assert buttons(panel) == ['cancel', 'revert', 'keep']
    answer(panel, 'cancel')
    assert panel.js("pythonic.panel.drawer.current") == 'ai'
    assert not acts(core, 'ai.keep') and not acts(core, 'ai.clear')

    core.verbs['ai.keep'] = lambda channels=None: {'kept': [2]}
    panel.click('#ai-close')
    alert_up(panel, 'ok')
    answer(panel, 'keep')
    assert wait_act(panel, 'ai.keep') == [{}]
    panel.wait_js("pythonic.panel.drawer.current === null")


def test_back_to_the_rack_can_revert(panel):
    core = panel.core
    open_ai(panel)
    tried(core, 6, 'TOM 2')
    panel.wait_js("pythonic.store.value('ai.tried').length === 1")
    panel.click('#ai-back')
    alert_up(panel, 'ok')
    answer(panel, 'revert')
    assert wait_act(panel, 'ai.clear') == [{}]
    panel.wait_js("pythonic.panel.drawer.current === null")


def test_leaving_with_new_patterns_asks_too(panel):
    core = panel.core
    open_ai(panel)
    post(core, ai__bank='ready', ai__preview='try')
    panel.wait_js("pythonic.store.value('ai.preview') === 'try'")
    panel.click('#ai-close')
    title, body = alert_up(panel, 'ok')
    assert body.startswith('The 12 new patterns play on the face')
    core.verbs['ai.replace_patterns'] = lambda channels=None: {'kept': [], 'patterns': 12}
    answer(panel, 'keep')
    assert wait_act(panel, 'ai.replace_patterns') == [{}]
    panel.wait_js("pythonic.panel.drawer.current === null")


def test_leaving_without_new_sounds_stops_a_preview_and_asks_nothing(panel):
    core = panel.core
    open_ai(panel)
    post(core, ai__preview='bank')
    panel.wait_js("pythonic.store.value('ai.preview') === 'bank'")
    panel.click('#ai-close')
    assert wait_act(panel, 'ai.pattern_try') == [{'mode': None}]
    panel.wait_js("pythonic.panel.drawer.current === null")
    assert not panel.js("!!document.querySelector('.sheet-layer')")


def test_another_drawer_page_replacing_it_asks_once_it_is_gone(panel):
    core = panel.core
    open_ai(panel)
    tried(core, 1, 'BD 7')
    panel.wait_js("pythonic.store.value('ai.tried').length === 1")
    panel.click('#matrix-toggle')
    panel.wait_js("pythonic.panel.drawer.current === 'matrix'")
    alert_up(panel, 'ok')
    assert buttons(panel) == ['revert', 'keep']
    answer(panel, 'keep')
    assert wait_act(panel, 'ai.keep') == [{}]


def test_a_preset_load_with_tried_sounds_asks_first(panel, tmp_path):
    core = panel.core
    core.verbs['preset.load'] = lambda path: {'path': str(tmp_path / path), 'name': 'Five', 'format': 'mtpreset'}
    folder_state(panel, tmp_path, tmp_path / '808.mtpreset')
    open_ai(panel)
    tried(core, 1, 'BD 2')
    panel.wait_js("pythonic.store.value('ai.tried').length === 1")
    open_preset_menu(panel)
    click_item(panel, '505', '#preset-menu .pm-files .it')
    alert_up(panel, 'ok')
    answer(panel, 'cancel')
    panel.qtbot.wait(100)
    assert not acts(core, 'preset.load')
    open_preset_menu(panel)
    click_item(panel, '505', '#preset-menu .pm-files .it')
    alert_up(panel, 'ok')
    answer(panel, 'revert')
    wait_act(panel, 'ai.clear')
    assert wait_act(panel, 'preset.load') == [{'path': FILES[0]}]
