"""
End-to-end tests of the page over the real app core, without audio: the
fake audio backend's callback is pulled by hand (``pump``), so blocks, the
playhead and verbs waiting for a block start advance under the test's control.
"""

import statistics
from pathlib import Path

from pythonic.web.bridge import FRAME_BUDGET_MS, Bridge

PRESETS = Path(__file__).resolve().parent.parent


def finish(core, action_id, timeout=10.0):
    """Pull audio blocks until an action has finished; returns its event."""
    for _ in range(int(timeout / 0.01)):
        try:
            return core.wait(action_id, timeout=0.01)
        except TimeoutError:
            core.backend.stream.pull()
    raise TimeoutError(f'action {action_id} did not finish')


def test_start_stop_plays_and_the_playhead_comes_back_through_poll(open_panel, real_core):
    page = open_panel(real_core, owns_core=False)
    pump = real_core.backend.stream.pull
    page.click('#start-stop')
    page.wait_js("document.querySelector('#start-stop').classList.contains('on')", pump=pump)
    page.wait_js("pythonic.store.readout('transport').position >= 3", pump=pump)
    for _ in range(3):
        pump()
    position = real_core.poll()['transport']['position']
    page.wait_js(f"pythonic.store.readout('transport').position === {position}")
    assert page.js("document.querySelector('.pad.ph').dataset.step") == str(position % 16 + 1)

    page.click('#start-stop')
    page.wait_js("!document.querySelector('#start-stop').classList.contains('on')", pump=pump)
    assert real_core.poll()['transport']['playing'] is False


def test_a_tempo_edit_round_trips_through_a_preset_file(open_panel, real_core, tmp_path):
    page = open_panel(real_core, owns_core=False)
    loaded = finish(real_core, real_core.act('preset.load', path=str(PRESETS / '808.mtpreset')))
    assert loaded['status'] == 'done'
    tempo = real_core.get('global.tempo')
    page.wait_js(f"document.querySelector('#tempo-value').textContent === '{tempo}'")

    def core_tempo_is(value):
        def check():
            real_core.backend.stream.pull()  # the audio thread applies queued sets
            return real_core.get('global.tempo') == value
        page.qtbot.waitUntil(check, timeout=5000)

    page.click('#tempo-up')
    page.wait_js(f"document.querySelector('#tempo-value').textContent === '{tempo + 1}'")
    core_tempo_is(tempo + 1)

    saved = tmp_path / 'edited.json'
    assert finish(real_core, real_core.act('preset.save', path=str(saved)))['status'] == 'done'
    page.click('#tempo-down')
    page.click('#tempo-down')
    page.wait_js(f"document.querySelector('#tempo-value').textContent === '{tempo - 1}'")
    core_tempo_is(tempo - 1)
    # No poll between the applied set and the load: the load's value still wins
    assert finish(real_core, real_core.act('preset.load', path=str(saved)))['status'] == 'done'
    page.wait_js(f"document.querySelector('#tempo-value').textContent === '{tempo + 1}'",
                 pump=real_core.backend.stream.pull)
    assert real_core.get('global.tempo') == tempo + 1


def test_a_frame_tick_fits_the_frame_budget(real_core):
    bridge = Bridge(real_core)
    finish(real_core, real_core.act('preset.load', path=str(PRESETS / '909.mtpreset')))
    finish(real_core, real_core.act('transport.play'))
    for _ in range(120):
        real_core.backend.stream.pull()
        bridge.tick()
    durations = []
    for _ in range(60):
        real_core.backend.stream.pull()
        before = bridge.stats.total_ms
        bridge.tick()
        durations.append(bridge.stats.total_ms - before)
    assert statistics.median(durations) < FRAME_BUDGET_MS / 2
    assert bridge.stats.count == 180
