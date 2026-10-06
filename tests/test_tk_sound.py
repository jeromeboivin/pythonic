"""
The tkinter GUI on the sound addresses (slice 2): its sound, global, mute,
selection and Edit all handlers write through core.set(), and its widgets are
refreshed from the changes core.poll() reports. Skips without a display.
"""

import pathlib

import pytest

from pythonic.app import AppCore
from pythonic.app.sound import SOUND_SUFFIXES
from tests.fake_audio import FakeAudioBackend
from tests.test_tk_smoke import _display_available

pytestmark = pytest.mark.skipif(not _display_available(), reason='no display for tkinter')

MAIN_WINDOW = pathlib.Path(__file__).resolve().parents[1] / 'gui' / 'main_window.py'


@pytest.fixture
def app(prefs):
    from gui.main_window import PythonicGUI

    prefs.set('midi_enabled', False)
    backend = FakeAudioBackend()
    core = AppCore(preferences=prefs, audio_backend=backend, stall_timeout=None)
    gui = PythonicGUI(core=core)
    core.wait(gui._audio_start_action)
    gui.backend = backend
    yield gui

    def close():
        # Timers of a destroyed window would fire in the next test's event loop
        for job in gui.root.tk.splitlist(gui.root.tk.call('after', 'info')):
            gui.root.after_cancel(job)
        gui.root.destroy()
    gui.root.after(10, close)
    gui.run()


def apply(gui):
    """One audio block (queued sets land), then one UI tick (poll)."""
    gui.backend.stream.pull()
    gui._ui_update_tick()


def test_every_sound_widget_has_an_address(app):
    assert set(app._sound_widgets) <= SOUND_SUFFIXES
    # Not on the tkinter panel: LFO phase offsets and the pump tempo sync
    assert SOUND_SUFFIXES - set(app._sound_widgets) == {'lfo1.phase', 'lfo2.phase', 'pump.sync'}


def test_no_handler_writes_the_engine_directly():
    source = MAIN_WINDOW.read_text()
    assert 'get_selected_channel' not in source
    assert 'edit_all_mode' not in source


def test_knob_writes_through_the_core(app):
    app.osc_decay_knob.set_value(500.0)
    app.distort_knob.set_value(40.0)
    app.lfo1_depth_knob.set_value(25.0)
    app.pump_amount_knob.set_value(60.0)
    app.backend.stream.pull()
    assert app.core.get('ch1.osc.decay') == 500.0
    assert app.core.get('ch1.mix.distortion') == pytest.approx(0.4)
    assert app.core.get('ch1.lfo1.depth') == 25.0
    assert app.core.get('ch1.pump.amount') == pytest.approx(0.6)


def test_selectors_and_lists_write_through_the_core(app):
    app._on_waveform_change(2)  # what a click on the selectors calls
    app._on_delay_time_change(3)
    app._on_output_change(1)
    app.lfo2_wave_var.set('S&H')
    app._on_lfo_waveform('lfo2', app.lfo2_wave_var)
    app.lfo2_dest_var.set('Pan')
    app._on_lfo_dest('lfo2', app.lfo2_dest_var)
    app.pump_dest_var.set('Off')
    app._on_pump_dest()
    app.backend.stream.pull()
    assert app.core.get('ch1.osc.wave') == 'sawtooth'
    assert app.core.get('ch1.fx.delay_time') == 'eighth_t'
    assert app.core.get('ch1.mix.output') == 'B'
    assert app.core.get('ch1.lfo2.wave') == 'sample_and_hold'
    assert app.core.get('ch1.lfo2.target') == 'pan'
    assert app.core.get('ch1.pump.target') == 'none'


def test_widgets_follow_changes_reported_by_poll(app):
    app.core.set('ch1.osc.decay', 777.0)
    app.core.set('ch1.mix.osc_noise', 0.25)
    app.core.set('ch1.noise.stereo', True)
    app.core.set('ch1.lfo1.sync', 'quarter')
    app.core.set('ch1.pump.target', 'pan')
    apply(app)
    assert app.osc_decay_knob.get_value() == 777.0
    assert app.mix_slider.get() == 25
    assert app.stereo_btn.get_value() is True
    assert app.lfo1_sync_var.get() == '1/4'
    assert app.pump_dest_var.get() == 'Pan'


def test_channel_selection_goes_through_the_core(app):
    app.core.set('ch3.osc.decay', 1234.0)
    app._on_channel_select(2)
    assert app.selected_channel == 0  # until the core reports it
    apply(app)
    assert app.core.get('global.channel') == 3
    assert app.selected_channel == 2
    assert app.channel_buttons[2].selected and not app.channel_buttons[0].selected
    assert app.osc_decay_knob.get_value() == 1234.0
    assert app.pattern_channel_label.cget('text') == 'ch3'

    app.osc_decay_knob.set_value(600.0)  # edits now go to channel 3
    app.backend.stream.pull()
    assert app.core.get('ch3.osc.decay') == 600.0
    assert app.core.get('ch1.osc.decay') != 600.0


def test_edit_all_and_mutes_go_through_the_core(app):
    app.mute_buttons[3]._on_click(None)
    app.edit_all_btn._on_click(None)
    assert app.core.get('global.edit_all') is True
    app.reverb_mix_knob.set_value(50.0)
    apply(app)
    assert app.core.get('ch4.mute') is True
    assert app.channel_buttons[3].muted
    mixes = [app.core.get(f'ch{n}.fx.reverb_mix') for n in range(1, 9)]
    assert mixes == [0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.5, 0.5]

    app.core.set('global.edit_all', False)
    app.core.set('ch4.mute', False)
    apply(app)
    assert app.edit_all_btn.get_value() is False
    assert app.mute_buttons[3].get_value() is False
    assert not app.channel_buttons[3].muted


def test_globals_go_through_the_core_and_follow_poll(app):
    app.bpm_var.set('140')
    app._on_bpm_change()
    app._on_step_rate_button('1/32')
    app._on_fill_rate_button(7)
    app._on_global_swing_change('30')  # what the slider calls (Tk only calls it on screen)
    app.master_knob.set_value(-9.0)
    apply(app)
    assert app.core.get('global.tempo') == 140
    assert app.core.get('global.step_rate') == '1/32'
    assert app.core.get('global.fill_rate') == 7
    assert app.core.get('global.swing') == pytest.approx(0.3)
    assert app.core.get('global.master') == -9.0

    app.core.set('global.tempo', 90)
    app.core.set('global.step_rate', '1/8')
    app.core.set('global.fill_rate', 3)
    app.core.set('global.swing', 0.6)
    app.core.set('global.master', -3.0)
    apply(app)
    assert app.bpm_var.get() == '90'
    assert dict(app.step_rate_buttons)['1/8'].cget('bg') == app.COLORS['highlight']
    assert dict(app.step_rate_buttons)['1/32'].cget('bg') == app.COLORS['bg_light']
    assert dict(app.fill_rate_buttons)[3].cget('bg') == app.COLORS['highlight']
    assert app.global_swing_slider.get() == 60
    assert app.master_knob.get_value() == -3.0


def test_bad_tempo_entry_is_reverted(app):
    app.bpm_var.set('fast')
    app._on_bpm_change()
    assert app.bpm_var.get() == str(app.core.get('global.tempo'))


def test_morph_slider_goes_through_the_core(app):
    # The slider is enabled once the endpoints differ
    app.morph_manager.capture_endpoint_a()
    app.core.synth.channels[0].set_osc_decay(50.0)
    app.morph_manager.capture_endpoint_b()
    app._update_morph_ui()
    app.morph_slider.set(40)
    app.root.update()
    apply(app)
    assert app.core.get('morph.position') == pytest.approx(0.4)
    decay_at_40 = app.osc_decay_knob.get_value()
    app.core.set('morph.position', 0.75)
    apply(app)
    assert app.morph_slider.get() == 75
    assert app.osc_decay_knob.get_value() == app.core.get('ch1.osc.decay') != decay_at_40
