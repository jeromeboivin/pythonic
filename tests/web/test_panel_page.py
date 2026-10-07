"""Page tests of the shell: the real app:// page and bridge over the fake core."""

import pytest
from PySide6.QtCore import QSize

from pythonic.web.window import MIN_SIZE
from tests.web.page import same_path


def test_the_page_boots_and_reads_the_tempo(panel):
    assert panel.js("document.querySelector('#tempo-value').textContent") == '120'
    assert panel.js('pythonic.bridge.kind') == 'qt'


def test_the_window_opens_at_the_minimum_size_with_the_stage_at_0_8(panel):
    assert panel.window.minimumSize() == MIN_SIZE
    assert panel.window.size() == QSize(1280, 800)
    assert panel.js("document.querySelector('#stage').dataset.scale") == '0.8'
    assert panel.rect('#stage') == pytest.approx((0, 0, 1280, 800))


def test_a_wider_window_letterboxes_the_stage(panel, qtbot):
    panel.window.resize(1920, 1000)
    qtbot.waitUntil(lambda: panel.js("document.querySelector('#stage').dataset.scale") == '1')
    assert panel.rect('#stage') == pytest.approx((160, 0, 1600, 1000))
    panel.wait_pixels(lambda: panel.pixel(40, 500) == (0, 0, 0))  # letterbox
    stage = panel.pixel(900, 400)
    assert panel.close_to(stage, (11, 11, 12), tolerance=3) and stage != (0, 0, 0)


def test_the_tempo_knob_sets_the_core_and_the_display_follows(panel, qtbot):
    panel.wheel('px-knob[data-address="global.tempo"]', steps=1)
    qtbot.waitUntil(lambda: panel.core.sets() == [('global.tempo', 121)])
    panel.core.post_change('global.tempo', 140)  # e.g. a MIDI controller
    panel.wait_js("document.querySelector('#tempo-value').textContent === '140'")


def test_the_wheel_over_the_tempo_is_a_burst(panel, qtbot):
    panel.wheel('#tempo-value', steps=1)
    qtbot.waitUntil(lambda: bool(panel.core.calls))
    kind, address, value, options = panel.core.calls[0]
    assert (kind, address, value, options['burst']) == ('set', 'global.tempo', 121, True)


def test_start_stop_runs_the_verb_and_lights_from_the_transport(panel, qtbot):
    assert panel.color_at('#start-stop')[1] < 80  # dark
    panel.click('#start-stop')
    qtbot.waitUntil(lambda: panel.core.verbs_called() == ['transport.toggle'])
    panel.wait_js("document.querySelector('#start-stop').classList.contains('on')")
    panel.wait_pixels(lambda: panel.color_at('#start-stop')[1] > 180)  # lit green
    r, g, b = panel.color_at('#start-stop')
    assert g > r and g > b


def test_the_playhead_outlines_its_pad(panel):
    panel.core.transport.update(playing=True, position=5)
    panel.wait_js("document.querySelector('.pad.ph')?.dataset.step === '6'")
    assert panel.js("document.querySelectorAll('.pad.ph').length") == 1


def test_a_file_dialog_from_the_page_returns_the_path(panel, qtbot, tmp_path):
    target = tmp_path / 'kit.mtpreset'
    target.write_text('x')
    panel.run(f"pythonic.client.openFile({{ folder: '{tmp_path}' }})"
              ".then((p) => { window.__path = p; });")
    qtbot.waitUntil(lambda: len(panel.bridge.open_dialogs()) == 1)
    frames = panel.bridge.stats.count
    qtbot.wait(100)
    assert panel.bridge.stats.count > frames  # the frame timer runs while it is open
    panel.answer_dialog(target)
    panel.wait_js('window.__path !== undefined')
    assert same_path(panel.js('window.__path'), target)

    panel.run("pythonic.client.saveFile({}).then((p) => { window.__saved = p; });")
    panel.answer_dialog(None)
    panel.wait_js("window.__saved === null")


def test_closing_the_window_closes_the_core(open_panel, fake_core):
    page = open_panel(fake_core)
    page.window.close()
    assert fake_core.closed and not page.bridge.running


def test_readouts_from_before_the_page_connected_are_shown(open_panel, fake_core):
    fake_core.transport.update(playing=True, position=2)
    page = open_panel(fake_core)
    page.wait_js("document.querySelector('#start-stop').classList.contains('on')")
    assert page.js("document.querySelector('.pad.ph').dataset.step") == '3'


def test_fonts_css_leaves_out_the_fonts_that_are_not_there(tmp_path):
    from pythonic.web.scheme import available_fonts_css
    (tmp_path / 'fonts').mkdir()
    (tmp_path / 'fonts' / 'Here.woff2').write_bytes(b'x')
    css = ('/* bundled */\n'
           '@font-face {\n  font-family: "Here";\n  src: url("../fonts/Here.woff2") format("woff2");\n}\n'
           '@font-face {\n  font-family: "Gone";\n  src: url("../fonts/Gone.woff2") format("woff2");\n}\n')
    out = available_fonts_css(css, tmp_path / 'fonts')
    assert 'Here.woff2' in out and '/* bundled */' in out
    assert 'Gone' not in out


def test_the_page_asks_for_no_file_that_is_not_there(panel):
    # The bundled fonts are optional (tools/fetch_fonts.py): a missing one is
    # not asked for (DevTools would list each failed load as a console error)
    from pythonic.web.scheme import install_handler
    handler = install_handler(panel.page.profile())
    assert handler.missing == []
