"""The setup sheet (web slice W6) over the real app core, on a fake audio
backend and temporary preferences: the base note, a new CC mapping and a
restart of the stream reach the core and its preferences file."""

from tests.web.test_menus_page import click_item
from tests.web.test_setup_page import open_setup


def test_a_base_note_step_reaches_the_core_and_its_preferences(open_panel, real_core):
    page = open_panel(real_core, owns_core=False)
    open_setup(page, 'midi')
    page.wait_js("document.querySelector('#su-base-note').textContent === 'C2 (36)'")
    page.click('#su-base-up')
    page.qtbot.waitUntil(lambda: real_core.get('midi.base_note') == 37)
    assert real_core.preferences.get('midi_base_note') == 37
    page.wait_js("document.querySelector('.su-range').textContent.endsWith('C#2–G#2 (37–44)')")


def test_an_added_cc_mapping_reaches_the_cc_map(open_panel, real_core):
    page = open_panel(real_core, owns_core=False)
    before = real_core.get('midi.cc_map')
    open_setup(page, 'midi')
    page.wait_js(f"document.querySelectorAll('.su-ccrow').length === {len(before)}")
    page.click('#su-cc-add')
    page.wait_js("document.querySelectorAll('.su-ccrow.draft').length === 1")
    cc = int(page.js("document.querySelector('.su-ccrow.draft').dataset.cc"))
    assert cc not in before
    page.click('.su-ccrow.draft .su-target')
    click_item(page, 'fx ▸')
    click_item(page, 'reverb mix')
    page.qtbot.waitUntil(lambda: real_core.get('midi.cc_map').get(cc) == 'selected.fx.reverb_mix')
    assert {int(k) for k in real_core.preferences.get('midi_cc_mappings')} == set(before) | {cc}
    page.wait_js(f"document.querySelector('.su-ccrow[data-cc=\"{cc}\"]:not(.draft)') !== null")


def test_a_buffer_change_waits_for_restart_audio(open_panel, real_core):
    page = open_panel(real_core, owns_core=False)
    pump = real_core.backend.stream.pull
    open_setup(page)
    page.click('.su-choice[data-address="pref.audio.buffer_ms"]')
    click_item(page, '10 ms')
    page.qtbot.waitUntil(lambda: real_core.get('pref.audio.pending') == ['pref.audio.buffer_ms'])
    page.wait_js("document.querySelector('#su-restart').classList.contains('on')")
    assert page.js("[...document.querySelectorAll('.su-dot.on')].map((d) => d.dataset.pending)") == \
        ['pref.audio.buffer_ms']
    page.click('#su-restart')
    page.wait_js("!document.querySelector('#su-restart').classList.contains('on')", timeout=10000, pump=pump)
    assert real_core.get('pref.audio.pending') == []
    assert real_core.get('audio.buffer_ms') == 10.0
