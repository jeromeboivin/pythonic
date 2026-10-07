"""
The PO-32 page over the real app core with the fake audio backend: a PO-32
card transfer made with the codec is pushed into the input stream while the
page records, the page decodes, picks and imports it (the pattern lanes
change, one undo step), previews it; the transfer tab prepares, sends
through the pulled output stream and saves the WAV file.
"""

import numpy as np
import pytest

from pythonic.app import po32 as po32_module
from tests.test_po32_core import CARD_PATTERNS, CARD_SOUNDS, RATE, card_signal

QUIET = [False] * 16


@pytest.fixture(scope='module')
def card():
    return card_signal(CARD_SOUNDS, CARD_PATTERNS)


def pumping(core, blocks=1):
    def pump():
        for _ in range(blocks):
            core.backend.stream.pull()
    return pump


def until(page, check, timeout=10000, blocks=1):
    pump = pumping(page.core, blocks)

    def ready():
        pump()
        return check()
    page.qtbot.waitUntil(ready, timeout=timeout)


def open_page(page, tab):
    page.run(f"pythonic.panel.openPage('po32', {{ tab: '{tab}' }})")
    page.wait_js(f"document.querySelector('#po32-page') && pythonic.panel.drawer.current === 'po32'"
                 f" && document.querySelector('#po32-page').dataset.tab === '{tab}'")


def test_record_decode_pick_and_import_as_one_undo_step(open_panel, real_core, card, tmp_path,
                                                        monkeypatch):
    monkeypatch.setattr(po32_module, 'recordings_folder', lambda: str(tmp_path / 'rec'))
    page = open_panel(real_core, owns_core=False)
    pump = pumping(real_core)
    open_page(page, 'import')
    assert page.js("document.querySelector('#po32-input').textContent") == 'Fake In'
    page.click('#po32-keep .btn')  # keep recordings (in the temporary folder)
    until(page, lambda: real_core.get('pref.po32.save_recordings') is True)

    page.click('#po32-record')
    until(page, lambda: real_core.get('po32.recording'))
    page.wait_js("document.querySelector('#po32-record').textContent === '■ stop'")
    stream = real_core.backend.input
    stream.push(np.zeros(RATE // 2))
    stream.push(card * 0.6)
    # The meter and the recording length follow the input
    page.wait_js("parseFloat(document.querySelector('#po32-meter .lvl').style.width) > 30")
    page.wait_js("document.querySelector('#po32-source').textContent.startsWith('recording')")
    stream.push(np.zeros(RATE // 2))

    page.click('#po32-record')  # stop: the core decodes
    page.wait_js("document.querySelector('#po32-source').textContent.startsWith('decoded PO-32 card: 16 sounds, 3 patterns')",
                 timeout=20000)
    assert list((tmp_path / 'rec').glob('po32_recording_*.wav'))
    assert page.js("[...document.querySelectorAll('.po-pat')].map((b) => b.textContent)") == [
        '1→A', '2→B', '3→C', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
    assert page.js("[...document.querySelectorAll('.po-pat')].map((b) => b.disabled)") == [False] * 3 + [True] * 13
    assert page.js("[...document.querySelectorAll('[data-flow=import] .po-stage')].map((s) => s.dataset.state)") == [
        'done', 'done', 'done', 'now']
    assert page.js("document.querySelectorAll('#po32-grid i.on').length") == 8  # pattern 1: drums 1 and 3

    # Pattern 2 (card pattern 3, drum 2) onto A: the letters swap
    page.right_click('.po-pat[data-pattern="2"]')
    page.wait_js("[...document.querySelectorAll('.px-menu .it')].some((i) => i.textContent.startsWith('→ A'))")
    page.run("[...document.querySelectorAll('.px-menu .it')].find((i) => i.textContent.startsWith('→ A'))"
             ".dataset.pick = 'yes'")
    page.click('[data-pick="yes"]')
    until(page, lambda: real_core.get('po32.picks') == [
        {'pattern': 1, 'letter': 'B'}, {'pattern': 2, 'letter': 'A'}, {'pattern': 3, 'letter': 'C'}])
    page.wait_js("document.querySelector('.po-pat[data-pattern=\"1\"]').textContent === '1→B'")
    # Unpick pattern 3
    page.click('.po-pat[data-pattern="3"]')
    until(page, lambda: len(real_core.get('po32.picks')) == 2)
    page.wait_js("document.querySelector('#po32-count').textContent === '2 / 12 picked'")

    # Preview stops the panel's transport; a pick change ends it
    page.click('#start-stop')
    until(page, lambda: real_core.poll()['transport']['playing'])
    page.click('#po32-preview')
    until(page, lambda: real_core.get('po32.previewing'))
    assert not real_core.poll()['transport']['playing']
    page.wait_js("document.querySelector('#po32-preview').textContent === '■ stop'", pump=pump)
    page.click('.po-pat[data-pattern="3"]')  # picked again: the preview ends
    until(page, lambda: not real_core.get('po32.previewing'))
    page.wait_js("document.querySelector('#po32-preview').textContent === '▶ preview'", pump=pump)

    real_core.set('pattern.K.ch1.step3.trig', True)
    until(page, lambda: real_core.get('pattern.K.ch1.trig')[2])
    page.click('#po32-import')
    until(page, lambda: real_core.get('po32.imported'))
    assert real_core.get('pattern.A.ch2.trig')[:16] == [False, True, False, False] * 4
    assert real_core.get('pattern.B.ch1.trig')[:16] == [True, False, False, False] * 4
    assert real_core.get('pattern.C.ch4.trig')[:16] == [True, False, False, False] * 4
    assert real_core.get('pattern.K.empty')
    page.wait_js("pythonic.panel.display.text()[0] === 'PO-32 IMPORT'", pump=pump)
    assert page.js("pythonic.panel.display.text()[1]") == '8 sounds, 3 patterns'
    # The page stays open, stage 4 ticked
    page.wait_js("[...document.querySelectorAll('[data-flow=import] .po-stage')].every((s) => s.dataset.state === 'done')",
                 pump=pump)
    assert page.js("pythonic.panel.drawer.current") == 'po32'

    page.wait_js("!document.querySelector('#undo').disabled", pump=pump)
    page.click('#undo')  # one undo step takes the whole import back
    until(page, lambda: real_core.get('pattern.K.ch1.trig')[2] and not any(real_core.get('pattern.A.ch2.trig')))
    assert real_core.get('pattern.B.ch1.trig')[:16] == QUIET

    page.click('#po32-close')  # closing the page closes the input
    page.wait_js("pythonic.panel.drawer.current === null")
    until(page, lambda: not real_core.get('po32.listening'))


def test_monitor_shows_the_input_level(open_panel, real_core):
    page = open_panel(real_core, owns_core=False)
    open_page(page, 'import')
    page.click('#po32-monitor')
    until(page, lambda: real_core.get('po32.listening'))
    page.wait_js("document.querySelector('#po32-monitor').classList.contains('on')")
    real_core.backend.input.push(0.5 * np.sin(np.arange(4096) * 0.1))
    page.wait_js("document.querySelector('.po-db').textContent === '-6.0 dB'")
    page.click('#po32-monitor')
    until(page, lambda: not real_core.get('po32.listening'))
    page.wait_js("document.querySelector('.po-db').textContent === '-∞ dB'")


def test_a_decoded_wav_file_imports_from_the_dialog(open_panel, real_core, card, tmp_path):
    from pythonic.po32_codec import save_wav
    path = tmp_path / 'card.wav'
    save_wav(card, str(path), RATE)
    page = open_panel(real_core, owns_core=False)
    open_page(page, 'import')
    page.click('#po32-open-wav')
    page.answer_dialog(path)
    page.wait_js("document.querySelector('#po32-source').textContent.endsWith('card.wav')", timeout=20000)
    # Bank 1 holds sounds too (a card has 16)
    page.wait_js("!document.querySelector('[data-bank=\"1\"]').disabled")
    page.click('[data-bank="1"]')
    until(page, lambda: real_core.get('po32.bank') == 1)
    page.wait_js("document.querySelector('[data-bank=\"1\"]').classList.contains('on')")


def test_transfer_prepares_sends_and_saves_the_wav(open_panel, real_core, tmp_path):
    real_core.set('ch3.mute', True)
    until_core = pumping(real_core)
    for _ in range(3):
        until_core()
    assert real_core.get('ch3.mute')
    page = open_panel(real_core, owns_core=False)
    page.click('#po32-button')  # the last tab: transfer, the first time
    page.wait_js("pythonic.panel.drawer.current === 'po32'")
    page.wait_js("document.querySelector('#po32-status').textContent.startsWith('ready:')",
                 pump=pumping(real_core))  # the signal takes the sounds at a block start
    assert page.js("document.querySelector('#po32-slots').textContent") == \
        'PO-32 pattern 1 is sent empty: the transfer carries the sounds only'
    # The channels sent start from the face mutes
    assert page.js("[...document.querySelectorAll('.po-check')].map((c) => c.classList.contains('on'))") == [
        True, True, False, True, True, True, True, True]
    assert page.js("pythonic.panel.drawer.current && document.querySelector('#po32-button').classList.contains('on')")

    page.click('#po32-send')
    page.wait_js("document.querySelector('#po32-send').textContent === '■ stop'")
    page.wait_js("parseFloat(document.querySelector('#po32-progress i').style.width) > 20",
                 pump=pumping(real_core, 20), timeout=20000)
    page.wait_js("pythonic.panel.display.text()[0] === 'PO-32 SEND'")
    page.wait_js("document.querySelector('#po32-status').textContent.startsWith('sent')",
                 pump=pumping(real_core, 20), timeout=30000)
    assert real_core.get('po32.transfer') == 'sent'
    assert page.js("[...document.querySelectorAll('[data-flow=transfer] .po-stage')].map((s) => s.dataset.state)") == [
        'done', 'done', 'done']

    target = tmp_path / 'out.wav'
    page.click('#po32-save-wav')
    page.answer_dialog(target)
    page.wait_js("document.querySelector('#po32-saved').textContent === 'saved out.wav'",
                 pump=pumping(real_core), timeout=10000)
    assert target.read_bytes()[:4] == b'RIFF'
