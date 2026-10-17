"""
The AI generator page over the real app core, with the stand-in AI worker of
the core tests (tests/fake_ai_worker.py: no torch, no models): gen puts one
new sound on the channel, the strip tab shows it, a knob edit goes into it,
keep is one undo step, new patterns play on the face and keep with the
sounds, and leaving with a new sound reverts it when asked to.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
FAKE_WORKER = str(ROOT / 'tests' / 'fake_ai_worker.py')


@pytest.fixture
def ai_core(prefs, tmp_path):
    from pythonic.app import AppCore
    from tests.fake_audio import FakeAudioBackend

    backend = FakeAudioBackend()
    core = AppCore(preferences=prefs, audio_backend=backend, midi_backend=None, stall_timeout=None,
                   ai_worker=[sys.executable, FAKE_WORKER])
    core.backend = backend
    for kind in ('patch', 'pattern'):
        model = tmp_path / f'{kind}.pt'
        model.write_bytes(b'fake')
        core.set(f'pref.ai.{kind}_model', str(model))
    started = core.start()
    assert core.wait(started)['status'] == 'done'
    yield core
    core.close()


def open_ai(page):
    page.run("pythonic.panel.openPage('ai')")
    page.wait_js("pythonic.panel.drawer.current === 'ai'")
    page.run('window.__painted = false; requestAnimationFrame(() => '
             'requestAnimationFrame(() => { window.__painted = true; }));')
    page.wait_js('window.__painted')


def test_generate_try_on_the_face_edit_keep_and_undo_in_one_step(open_panel, ai_core):
    core = ai_core
    page = open_panel(core, owns_core=False)
    pump = core.backend.stream.pull
    before = {a: core.get(a) for a in ('ch1.osc.freq', 'ch1.osc.decay', 'ch1.name')}
    open_ai(page)
    page.wait_js("(pythonic.store.value('ai.models') || {}).patch?.status === 'loaded'", timeout=10000, pump=pump)
    lane_type = core.get('ai.ch1.type')

    page.click('.ai-lane[data-channel="1"] .gen')
    page.wait_js("document.querySelector('.ai-lane[data-channel=\"1\"] .new').classList.contains('on')",
                 timeout=10000, pump=pump)
    assert core.get('ai.ch1.trying') is True and core.get('ai.ch1.candidates') == 1
    assert core.get('ch1.osc.freq') == 100.0  # the stand-in's candidate 1
    tab = '.strip[data-channel="1"] .tab'
    page.wait_js(f"document.querySelector('{tab}').classList.contains('aitry')"
                 f" && document.querySelector('{tab}').textContent === '{lane_type.upper()} 1'", pump=pump)
    assert page.js("document.querySelector('#undo').disabled")

    # old, then new again; a knob edit goes into the new sound
    page.click('.ai-lane[data-channel="1"] .old')
    page.wait_js("pythonic.store.value('ai.ch1.trying') === false", pump=pump)
    assert core.get('ch1.osc.freq') == before['ch1.osc.freq']
    page.click('.ai-lane[data-channel="1"] .new')
    page.wait_js("pythonic.store.value('ai.ch1.trying') === true", pump=pump)
    assert core.get('ch1.osc.freq') == 100.0
    decay = core.get('ch1.osc.decay')
    page.drag('.strip[data-channel="1"] px-knob[data-address="ch1.osc.decay"] .dial', dy=-40)

    def decay_moved():
        pump()
        return core.get('ch1.osc.decay') > decay
    page.qtbot.waitUntil(decay_moved, timeout=5000)
    assert core.get('undo.can_undo') is False  # no undo step of its own

    page.click('#ai-keep')
    page.wait_js("!document.querySelector('#undo').disabled", pump=pump)
    assert core.get('ai.tried') == []
    page.wait_js(f"!document.querySelector('{tab}').classList.contains('aitry')", pump=pump)
    assert core.get('ch1.osc.freq') == 100.0
    page.wait_js("pythonic.store.value('ai.ch1.candidates') === 0", pump=pump)  # forgotten once kept

    page.click('#undo')  # keep was one step: the old sound, edit included, comes back
    page.wait_js("document.querySelector('#undo').disabled", pump=pump)
    assert {a: core.get(a) for a in before} == before


def test_leaving_with_a_tried_sound_and_reverting_puts_the_old_one_back(open_panel, ai_core):
    core = ai_core
    page = open_panel(core, owns_core=False)
    pump = core.backend.stream.pull
    before = (core.get('ch3.osc.freq'), core.get('ch3.name'))
    open_ai(page)
    page.click('.ai-lane[data-channel="3"] .gen')
    page.wait_js("pythonic.store.value('ai.tried').includes(3)", timeout=10000, pump=pump)
    page.click('#ai-close')
    page.wait_js("!!document.querySelector('.alert-sheet')", pump=pump)
    page.run("[...document.querySelectorAll('.alert-buttons .btn')].find((b) => b.textContent === 'revert')"
             ".dataset.pick = 'yes'")
    page.click('.alert-buttons [data-pick="yes"]')
    page.wait_js("pythonic.panel.drawer.current === null && pythonic.store.value('ai.tried').length === 0",
                 pump=pump)
    assert (core.get('ch3.osc.freq'), core.get('ch3.name')) == before
    assert core.get('undo.can_undo') is False


def test_new_patterns_play_on_the_face_and_keep_with_the_sounds(open_panel, ai_core):
    core = ai_core
    page = open_panel(core, owns_core=False)
    pump = core.backend.stream.pull
    preset = [core.get(f'pattern.{p}.ch1.trig') for p in 'ABCDEFGHIJKL']
    before = core.get('ch2.name')
    open_ai(page)
    page.click('.ai-lane[data-channel="2"] .gen')
    page.wait_js("pythonic.store.value('ai.tried').includes(2)", timeout=10000, pump=pump)
    page.click('#ai-gen-patterns')
    page.wait_js("pythonic.store.value('ai.preview') === 'try'", timeout=10000, pump=pump)
    transport = core.poll()['transport']
    assert transport['playing'] is True and transport['chain'] == [4, 5, 6]  # A is in no chain: E-G plays
    page.wait_js("pythonic.panel.display.text()[1] === '12 new, from E'", pump=pump)
    assert core.get('pattern.B.ch1.trig') == [s % 2 == 0 for s in range(16)]  # the stand-in's B
    assert core.get('pattern.E.chained')  # and its chains, named on the page
    page.wait_js("document.querySelector('.ai-patnote').textContent.includes('(chains E–G)')", pump=pump)
    page.click('#ai-pat-old')
    page.wait_js("pythonic.store.value('ai.preview') === 'off'", pump=pump)
    assert [core.get(f'pattern.{p}.ch1.trig') for p in 'ABCDEFGHIJKL'] == preset
    page.click('#ai-pat-new')
    page.wait_js("pythonic.store.value('ai.preview') === 'try'", pump=pump)
    page.click('#ai-keep')
    page.wait_js("pythonic.store.value('ai.bank') === 'none' && pythonic.store.value('ai.tried').length === 0",
                 pump=pump)
    assert core.get('ai.preview') == 'off'
    assert core.get('pattern.B.ch1.trig') == [s % 2 == 0 for s in range(16)]
    page.wait_js("!document.querySelector('#undo').disabled", pump=pump)
    page.click('#undo')  # sounds and patterns were one step

    def restored():
        pump()
        return [core.get(f'pattern.{p}.ch1.trig') for p in 'ABCDEFGHIJKL'] == preset
    page.qtbot.waitUntil(restored, timeout=5000)
    assert core.get('ch2.name') == before
