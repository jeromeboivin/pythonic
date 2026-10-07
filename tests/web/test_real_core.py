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

    page.wheel('#tempo-value', steps=1)
    page.wait_js(f"document.querySelector('#tempo-value').textContent === '{tempo + 1}'")
    core_tempo_is(tempo + 1)

    saved = tmp_path / 'edited.json'
    assert finish(real_core, real_core.act('preset.save', path=str(saved)))['status'] == 'done'
    page.wheel('#tempo-value', steps=-1)
    page.wait_js(f"document.querySelector('#tempo-value').textContent === '{tempo}'")
    page.wheel('#tempo-value', steps=-1)
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


def test_a_tune_drag_changes_the_core_and_undo_brings_it_back(open_panel, real_core):
    page = open_panel(real_core, owns_core=False)
    pump = real_core.backend.stream.pull
    knob = '.strip[data-channel="2"] px-knob[data-address="ch2.osc.pitch"]'
    before = real_core.get('ch2.osc.pitch')
    page.drag(f'{knob} .dial', dy=-40)  # +25 % of 48 st

    def pitch_moved():
        pump()
        return real_core.get('ch2.osc.pitch') > before + 5
    page.qtbot.waitUntil(pitch_moved, timeout=5000)
    page.wait_js(f"document.querySelector('{knob} .val').textContent.startsWith('+')", pump=pump)
    page.wait_js("!document.querySelector('#undo').disabled", pump=pump)

    page.click('#undo')  # the drag was one gesture: one undo step
    page.wait_js(f"document.querySelector('{knob} .val').textContent === '0.0 st'", pump=pump)
    assert real_core.get('ch2.osc.pitch') == before


def test_a_channel_button_selects_and_mutes_in_the_core(open_panel, real_core):
    page = open_panel(real_core, owns_core=False)
    pump = real_core.backend.stream.pull
    page.click('.strip[data-channel="5"] .chb')
    page.qtbot.waitUntil(lambda: (pump(), real_core.get('global.channel') == 5)[1], timeout=5000)
    page.click('#mute-latch')
    page.click('.strip[data-channel="5"] .chb')
    page.qtbot.waitUntil(lambda: (pump(), real_core.get('ch5.mute') is True)[1], timeout=5000)
    page.wait_js("document.querySelector('.strip[data-channel=\"5\"]').classList.contains('muted')",
                 pump=pump)


def test_the_strip_ctrl_mode_is_saved_as_a_preference(open_panel, real_core):
    page = open_panel(real_core, owns_core=False)
    page.click('#ctrl-mode')
    page.wait_js("!!document.querySelector('.px-menu')")
    page.run("[...document.querySelectorAll('.px-menu .it')].find((i) => i.textContent === 'reverb mix').click()")
    page.qtbot.waitUntil(lambda: (real_core.get('pref.ui.ctrl_knob') or {}).get('mode') == 'reverb_mix')
    assert real_core.preferences.get('ui_ctrl_knob')['mode'] == 'reverb_mix'
    page.wait_js("document.querySelector('.strip[data-channel=\"1\"] .ctrl').dataset.address"
                 " === 'ch1.fx.reverb_mix'")


# ---------------------------------------------------------------- step row (web slice W3)

PAD = '.pads .stp[data-step="{n}"]'


def until(page, check, timeout=5000):
    """Wait for a core-side condition, pulling audio blocks (queued sets apply at block start)."""
    def ready():
        page.core.backend.stream.pull()
        return check()
    page.qtbot.waitUntil(ready, timeout=timeout)


def test_a_paint_stroke_writes_the_lane_and_undo_takes_it_back_whole(open_panel, real_core):
    page = open_panel(real_core, owns_core=False)
    x0, x1 = page.center(PAD.format(n=1)).x(), page.center(PAD.format(n=4)).x()
    page.drag(PAD.format(n=1), dy=0, dx=x1 - x0, steps=6)
    until(page, lambda: real_core.get('pattern.A.ch1.trig')[:5] == [True] * 4 + [False])
    page.wait_js(f"document.querySelector('{PAD.format(n=4)}').classList.contains('on')",
                 pump=real_core.backend.stream.pull)
    page.wait_js("!document.querySelector('#undo').disabled", pump=real_core.backend.stream.pull)
    page.click('#undo')  # the stroke was one gesture: one undo step
    until(page, lambda: real_core.get('pattern.A.ch1.trig')[:4] == [False] * 4)
    page.wait_js(f"!document.querySelector('{PAD.format(n=1)}').classList.contains('on')",
                 pump=real_core.backend.stream.pull)


def test_play_moves_the_playhead_and_follow_turns_the_page(open_panel, real_core):
    page = open_panel(real_core, owns_core=False)
    pump = real_core.backend.stream.pull
    real_core.set('pattern.A.length', 32)
    real_core.set('pattern.A.ch1.step20.trig', True)
    until(page, lambda: real_core.get('pattern.A.length') == 32)
    page.click('#start-stop')
    page.wait_js("pythonic.store.readout('transport').position >= 17", timeout=10000, pump=pump)
    page.wait_js("document.querySelector('.pads .stp').dataset.step === '17'", pump=pump)
    position = page.js("pythonic.store.readout('transport').position")
    assert page.js("document.querySelector('.pads .ph').dataset.step") == str(position + 1)
    assert page.js(f"document.querySelector('{PAD.format(n=20)}').classList.contains('on')")
    assert page.js("document.querySelector('.pg[data-page=\"1\"]').classList.contains('play')")
    page.click('#start-stop')
    page.wait_js("!pythonic.store.readout('transport').playing", pump=pump)


def test_a_lane_copied_from_one_channel_pastes_into_another(open_panel, real_core):
    page = open_panel(real_core, owns_core=False)
    for step in (1, 5, 9):
        real_core.set(f'pattern.A.ch1.step{step}.trig', True)
    real_core.set('pattern.A.ch1.step5.vel', 100)
    until(page, lambda: real_core.get('pattern.A.ch1.step5.vel') == 100)
    page.click('#lane-copy')
    page.wait_js("pythonic.panel.display.text()[1] === 'copied'", pump=real_core.backend.stream.pull)
    page.click('.strip[data-channel="2"] .chb')
    until(page, lambda: real_core.get('global.channel') == 2)
    page.click('#lane-paste')
    until(page, lambda: real_core.get('pattern.A.ch2.trig') == real_core.get('pattern.A.ch1.trig'))
    assert real_core.get('pattern.A.ch2.step5.vel') == 100
    page.wait_js(f"document.querySelector('{PAD.format(n=9)}').classList.contains('on')",
                 pump=real_core.backend.stream.pull)


# ---------------------------------------------------------------- edit rack (web slice W4)

RACK = '.rack [data-suffix="{s}"]'


def test_a_rack_fader_sets_the_noise_attack_of_the_selected_channel(open_panel, real_core):
    page = open_panel(real_core, owns_core=False)
    page.click('.strip[data-channel="3"] .chb')
    until(page, lambda: real_core.get('global.channel') == 3)
    fader = RACK.format(s='noise.attack')
    page.wait_js(f"document.querySelector({fader!r}).dataset.address === 'ch3.noise.attack'",
                 pump=real_core.backend.stream.pull)
    before, ch1 = real_core.get('ch3.noise.attack'), real_core.get('ch1.noise.attack')
    page.drag(f'{fader} .track', dy=0, fy=0.25)  # a track click three quarters up
    until(page, lambda: real_core.get('ch3.noise.attack') > max(before, 50))
    assert real_core.get('ch1.noise.attack') == ch1  # edit all is off
    page.wait_js("!document.querySelector('#undo').disabled", pump=real_core.backend.stream.pull)
    page.click('#undo')
    until(page, lambda: real_core.get('ch3.noise.attack') == before)


def test_click_to_assign_sets_the_lfo_destination_in_the_core(open_panel, real_core):
    page = open_panel(real_core, owns_core=False)
    page.click(f"{RACK.format(s='lfo1.target')} button")
    page.wait_js("document.querySelector('#stage').classList.contains('assigning')")
    page.click('.strip[data-channel="1"] px-knob[data-address="ch1.osc.decay"] .dial')
    until(page, lambda: real_core.get('ch1.lfo1.target') == 'osc_decay')
    page.wait_js(f"document.querySelector('{RACK.format(s='lfo1.target')} button').textContent"
                 " === '→ osc decay'", pump=real_core.backend.stream.pull)


def test_modulation_bands_move_while_the_lfo_runs(open_panel, real_core):
    page = open_panel(real_core, owns_core=False)
    pump = real_core.backend.stream.pull
    for address, value in (('ch2.osc.decay', 10000.0), ('ch2.noise.decay', 10000.0),
                           ('ch2.lfo1.target', 'pitch_semitones'), ('ch2.lfo1.depth', 100.0),
                           ('ch2.lfo1.rate', 9.0), ('ch2.lfo1.on', True)):
        real_core.set(address, value)
    until(page, lambda: real_core.get('ch2.lfo1.on') is True)
    real_core.trigger(1)
    knob = '.strip[data-channel="2"] px-knob[data-address="ch2.osc.pitch"]'
    page.wait_js(f"document.querySelector({knob!r}).dataset.mod === 'lfo1'", pump=pump)
    first = page.js(f"document.querySelector({knob!r}).dataset.modTo")
    page.wait_js(f"document.querySelector({knob!r}).dataset.modTo !== {first!r}", pump=pump)


def test_closing_the_rack_is_saved_as_a_preference(open_panel, real_core):
    page = open_panel(real_core, owns_core=False)
    page.click('#rack-toggle')
    page.qtbot.waitUntil(lambda: real_core.get('pref.ui.rack_open') is False)
    assert real_core.preferences.get('ui_rack_open') is False
    page.qtbot.waitUntil(lambda: page.window.height() == 560)
