"""Page tests of web slice W5 over the fake core: the PRESET buttons and menu
(the folder list, open / save / save as with the replace question, the
clipboard, the drum patch entries, every drum as WAV, the preset folder,
recent files), the pattern menu's exports (MIDI straight to the save
dialog, audio through the tail popover, progress on the display), the alert
sheet (errors that need reading, yes / no questions), the overlay sheet
blocking the panel, the page registry behind PO-32 / SETUP / the MIDI LED,
and a click on the selected channel hitting it."""

import json

from tests.web.page import same_path

FILES = ['505.mtpreset', '808.mtpreset', 'mine.json']


def acts(core, verb):
    return [c[2] for c in core.calls if c[:2] == ('act', verb)]


def wait_act(panel, verb, count=1):
    panel.qtbot.waitUntil(lambda: len(acts(panel.core, verb)) >= count)
    return acts(panel.core, verb)


def folder_state(panel, folder, path=None, files=FILES, recent=()):
    core = panel.core
    core.post_change('pref.preset_folder', str(folder))
    core.post_change('preset.files', list(files))
    core.post_change('preset.path', str(path) if path else None)
    core.post_change('pref.recent_files', [str(p) for p in recent])
    panel.wait_js(f"JSON.stringify(pythonic.store.value('preset.files')) === {json.dumps(json.dumps(list(files), separators=(',', ':')))}")


def open_preset_menu(panel):
    panel.click('#preset-button')
    panel.wait_js("!!document.querySelector('#preset-menu')")


def items(panel, within):
    return panel.js(f"[...document.querySelectorAll({within!r})].map((i) => i.textContent)")


def click_item(panel, label, within='.px-menu .it'):
    """Click the menu row with this text (a real click at its centre)."""
    panel.wait_js(f"[...document.querySelectorAll({within!r})].some((i) => i.textContent === {label!r})")
    panel.run(f"[...document.querySelectorAll({within!r})].forEach((i, n) => "
              f"{{ if (i.textContent === {label!r}) i.dataset.pick = 'yes'; }})")
    panel.click('[data-pick="yes"]')


def alert_up(panel, tone=None):
    tone_sel = f'.tone-{tone}' if tone else ''
    panel.wait_js(f"!!document.querySelector('.sheet-layer[data-sheet=\"alert\"] .alert-sheet{tone_sel}')")
    return (panel.js("document.querySelector('.alert-title').textContent"),
            panel.js("document.querySelector('.alert-text') ? document.querySelector('.alert-text').textContent : ''"))


def answer_alert(panel, primary=True):
    panel.click('.alert-buttons .btn.primary' if primary else '.alert-buttons .btn:not(.primary)')
    panel.wait_js("!document.querySelector('.sheet-layer[data-sheet=\"alert\"]')")


def display(panel):
    return panel.js("pythonic.panel.display.text()")


# ---------------------------------------------------------------- PRESET

def test_the_preset_menu_lists_the_folder_and_loads_a_click(panel, tmp_path):
    core = panel.core
    core.verbs['preset.load'] = lambda path: {'path': str(tmp_path / path), 'name': 'Five O Five', 'format': 'mtpreset'}
    folder_state(panel, tmp_path, tmp_path / '808.mtpreset', recent=[tmp_path / '808.mtpreset'])
    open_preset_menu(panel)
    assert items(panel, '#preset-menu .pm-files .it') == ['505', '808', 'mine']
    assert items(panel, '#preset-menu .pm-files .it.cur') == ['808']
    assert panel.js("document.querySelector('#preset-menu .pm-folder').textContent") == tmp_path.name
    click_item(panel, '505', '#preset-menu .pm-files .it')
    assert wait_act(panel, 'preset.load') == [{'path': '505.mtpreset'}]
    panel.wait_js("pythonic.panel.display.text()[1] === 'FIVE O FIVE'")
    assert not panel.js("!!document.querySelector('#preset-menu')")


def test_prev_and_next_step_through_the_folder_without_wrapping(panel, tmp_path):
    core = panel.core
    core.verbs['preset.load'] = lambda path: {'path': str(tmp_path / path), 'name': path, 'format': 'json'}
    folder_state(panel, tmp_path, tmp_path / '808.mtpreset')
    panel.wait_js("!document.querySelector('#preset-prev').disabled && !document.querySelector('#preset-next').disabled")
    panel.click('#preset-next')
    assert wait_act(panel, 'preset.load') == [{'path': 'mine.json'}]
    panel.click('#preset-prev')
    assert wait_act(panel, 'preset.load', 2)[1] == {'path': '505.mtpreset'}
    core.post_change('preset.path', str(tmp_path / 'mine.json'))
    panel.wait_js("document.querySelector('#preset-next').disabled")
    core.post_change('preset.path', None)  # not in the folder: ▶ starts at the first
    panel.wait_js("!document.querySelector('#preset-next').disabled && document.querySelector('#preset-prev').disabled")


def test_open_preset_goes_through_the_native_dialog(panel, tmp_path):
    core = panel.core
    preset = tmp_path / 'other.mtpreset'
    preset.write_text('x')
    core.verbs['preset.load'] = lambda path: {'path': path, 'name': 'Other', 'format': 'mtpreset'}
    folder_state(panel, tmp_path)
    open_preset_menu(panel)
    click_item(panel, 'open preset…')
    panel.answer_dialog(None)  # cancelled: nothing loads
    open_preset_menu(panel)
    click_item(panel, 'open preset…')
    panel.answer_dialog(preset)
    (load,) = wait_act(panel, 'preset.load')
    assert same_path(load['path'], preset)


def test_save_as_asks_before_replacing_and_saves_again_with_overwrite(panel, qtbot, tmp_path):
    core = panel.core
    target = tmp_path / 'mine.json'
    target.write_text('{}')
    answers = iter([{'saved': False, 'exists': True, 'path': str(target)},
                    {'saved': True, 'exists': True, 'path': str(target)},
                    {'saved': False, 'exists': True, 'path': str(target)}])
    core.verbs['preset.save'] = lambda path, overwrite=False: next(answers)
    folder_state(panel, tmp_path)
    open_preset_menu(panel)
    click_item(panel, 'save preset as…')
    panel.answer_dialog(target)
    title, text = alert_up(panel, 'ok')
    assert title == 'Replace “mine.json”?' and str(tmp_path) in text
    answer_alert(panel, primary=True)
    saves = wait_act(panel, 'preset.save', 2)
    assert [s.get('overwrite', False) for s in saves] == [False, True]
    assert same_path(saves[0]['path'], target)
    panel.wait_js("pythonic.panel.display.text()[1] === 'saved'")

    open_preset_menu(panel)
    click_item(panel, 'save preset as…')
    panel.answer_dialog(target)
    alert_up(panel, 'ok')
    answer_alert(panel, primary=False)  # cancel: not saved, no third call
    panel.wait_js("pythonic.panel.display.text()[1] === 'not saved'")
    assert len(acts(core, 'preset.save')) == 3


def test_save_writes_a_json_preset_over_its_own_file(panel, tmp_path):
    core = panel.core
    core.verbs['preset.save'] = lambda path, overwrite=False: {'saved': True, 'exists': True, 'path': path}
    folder_state(panel, tmp_path, tmp_path / 'mine.json')
    open_preset_menu(panel)
    click_item(panel, 'save “mine.json”')
    assert wait_act(panel, 'preset.save') == [{'path': str(tmp_path / 'mine.json'), 'overwrite': True}]
    panel.wait_js("pythonic.panel.display.text()[1] === 'saved'")
    assert not panel.js("!!document.querySelector('.sheet-layer')")


def test_a_failed_load_shows_on_the_display_and_on_a_red_alert(panel, tmp_path):
    core = panel.core

    def broken(path):
        raise ValueError('not a preset file: line 3')
    core.verbs['preset.load'] = broken
    folder_state(panel, tmp_path, tmp_path / '808.mtpreset')
    panel.click('#preset-next')
    title, text = alert_up(panel, 'error')
    assert (title, text) == ('Could not load the preset', 'not a preset file: line 3')
    assert display(panel) == ['PRESET', 'not a preset file: line 3']
    # the panel behind takes no input while the alert shows
    panel.click('#start-stop')
    panel.qtbot.wait(100)
    assert 'transport.toggle' not in core.verbs_called()
    panel.click('.sheet-layer')  # outside the alert: it stays
    assert panel.js("!!document.querySelector('.sheet-layer')")
    answer_alert(panel)
    panel.click('#start-stop')
    panel.qtbot.waitUntil(lambda: 'transport.toggle' in core.verbs_called())


def test_clipboard_initialize_and_randomize_run_their_verbs(panel, tmp_path):
    core = panel.core
    folder_state(panel, tmp_path)
    open_preset_menu(panel)
    assert panel.js("[...document.querySelectorAll('#preset-menu .it')].find((i) => i.textContent === 'paste preset')"
                    ".classList.contains('dis')")
    panel.click('#preset-button')  # PRESET again closes the menu
    panel.wait_js("!document.querySelector('#preset-menu')")
    for label, verb in [('copy preset', 'preset.copy'), ('cut preset', 'preset.cut'),
                        ('initialize preset', 'preset.initialize'), ('randomize all', 'preset.randomize_all')]:
        open_preset_menu(panel)
        click_item(panel, label)
        wait_act(panel, verb)
    core.post_change('preset.clipboard', True)
    panel.wait_js("pythonic.store.value('preset.clipboard') === true")
    open_preset_menu(panel)
    click_item(panel, 'paste preset')
    wait_act(panel, 'preset.paste')
    panel.wait_js("pythonic.panel.display.text()[1] === 'pasted'")


def test_the_preset_folder_and_recent_files(panel, qtbot, tmp_path):
    core = panel.core
    core.verbs['preset.load'] = lambda path: {'path': path, 'name': 'Old', 'format': 'json'}
    elsewhere = tmp_path / 'kits'
    elsewhere.mkdir()
    recent = tmp_path / 'old' / 'old.json'
    folder_state(panel, tmp_path, recent=[recent])
    open_preset_menu(panel)
    assert items(panel, '#preset-menu .pm-recent .it') == ['old.jsonold']
    click_item(panel, 'old.jsonold', '#preset-menu .pm-recent .it')
    assert wait_act(panel, 'preset.load') == [{'path': str(recent)}]

    open_preset_menu(panel)
    click_item(panel, 'preset folder…')
    panel.answer_dialog(elsewhere)
    qtbot.waitUntil(lambda: any(c[:2] == ('set', 'pref.preset_folder') for c in core.calls))
    (folder,) = [c[2] for c in core.calls if c[:2] == ('set', 'pref.preset_folder')]
    assert same_path(folder, elsewhere)
    panel.wait_js("pythonic.panel.display.text()[0] === 'PRESET FOLDER'")


def test_every_drum_as_wav_asks_before_replacing_the_files(panel, tmp_path):
    core = panel.core
    paths = [str(tmp_path / '01_BD.wav'), str(tmp_path / '02_SD.wav')]
    answers = iter([{'saved': False, 'exists': True, 'folder': str(tmp_path), 'paths': paths},
                    {'saved': True, 'exists': True, 'folder': str(tmp_path), 'paths': paths * 4}])
    core.verbs['export.drum_wavs'] = lambda folder, overwrite=False: next(answers)
    folder_state(panel, tmp_path)
    open_preset_menu(panel)
    click_item(panel, 'export every drum as WAV…')
    panel.answer_dialog(tmp_path)
    title, text = alert_up(panel, 'ok')
    assert title == 'Replace 2 files?' and '01_BD.wav, 02_SD.wav' in text
    answer_alert(panel)
    calls = wait_act(panel, 'export.drum_wavs', 2)
    assert [c.get('overwrite', False) for c in calls] == [False, True]
    panel.wait_js("pythonic.panel.display.text()[1] === '8 files saved'")


def test_the_drum_patch_entries_of_the_preset_menu_use_the_selected_channel(panel, tmp_path):
    core = panel.core
    patch = tmp_path / 'snare.mtdrum'
    patch.write_text('x')
    core.verbs['drum_patch.load'] = lambda path, channel: {'channel': channel, 'name': 'Snare', 'path': path}
    core.post_change('global.channel', 4)
    panel.wait_js("pythonic.store.value('global.channel') === 4")
    open_preset_menu(panel)
    click_item(panel, 'load drum patch into CH4…')
    panel.answer_dialog(patch)
    (load,) = wait_act(panel, 'drum_patch.load')
    assert load['channel'] == 4


# ---------------------------------------------------------------- pattern exports

def open_pattern_menu(panel):
    panel.click('#pattern-menu')
    panel.wait_js("!!document.querySelector('.px-menu.pmenu')")


def test_export_to_midi_goes_straight_to_the_save_dialog(panel, tmp_path):
    core = panel.core
    target = tmp_path / 'pythonic_pattern_A.mid'
    answers = iter([{'saved': False, 'exists': True, 'path': str(target), 'pattern': 'A'},
                    {'saved': True, 'exists': True, 'path': str(target), 'pattern': 'A'}])
    core.verbs['export.midi'] = lambda path, pattern, overwrite=False: next(answers)
    open_pattern_menu(panel)
    click_item(panel, 'export to MIDI…')
    dialog_name = panel.qtbot.waitUntil(lambda: len(panel.bridge.open_dialogs()) == 1)  # noqa: F841
    (dialog,) = panel.bridge.open_dialogs()
    assert dialog.selectedFiles()[0].endswith('pythonic_pattern_A.mid')
    panel.answer_dialog(target)
    alert_up(panel, 'ok')
    answer_alert(panel)
    calls = wait_act(panel, 'export.midi', 2)
    assert [(c['pattern'], c.get('overwrite', False)) for c in calls] == [('A', False), ('A', True)]
    panel.wait_js("pythonic.panel.display.text()[1] === 'saved pythonic_pattern_A.mid'")


def test_export_to_audio_asks_for_the_tail_then_shows_the_render_progress(panel, tmp_path):
    core = panel.core
    target = tmp_path / 'b.wav'
    core.verbs['export.wav'] = lambda path, pattern, tail='cut', overwrite=False: core.DEFERRED
    core.post_change('pattern.selected', 'B')
    core.transport['selected_pattern'] = 1
    panel.wait_js("pythonic.store.value('pattern.selected') === 'B'")
    open_pattern_menu(panel)
    click_item(panel, 'export to audio…')
    panel.wait_js("!!document.querySelector('#tail-pop')")
    assert items(panel, '#tail-pop .tp-opts .btn') == ['cut', 'add 2 s', 'loop +1 pass']
    assert items(panel, '#tail-pop .tp-opts .btn.on') == ['cut']
    panel.click('#tail-pop [data-tail="loop"]')
    panel.wait_js("document.querySelector('#tail-pop [data-tail=\"loop\"]').classList.contains('on')")
    panel.click('#tail-save')
    panel.answer_dialog(target)
    (call,) = wait_act(panel, 'export.wav')
    assert call['pattern'] == 'B' and call['tail'] == 'loop' and same_path(call['path'], target)
    action = core.running
    core.progress(action, 0.4)
    panel.wait_js("pythonic.panel.display.text()[1] === '40 %'")
    core.finish(action, {'saved': True, 'exists': False, 'path': str(target), 'pattern': 'B', 'tail': 'loop'})
    panel.wait_js("pythonic.panel.display.text()[1] === 'saved b.wav'")
    assert panel.js("pythonic.panel.exports.tail") == 'loop'


def test_a_failed_export_says_why_on_a_red_alert(panel, tmp_path):
    core = panel.core

    def full(path, pattern, overwrite=False):
        raise OSError('No space left on device')
    core.verbs['export.midi'] = full
    open_pattern_menu(panel)
    click_item(panel, 'export to MIDI…')
    panel.answer_dialog(tmp_path / 'x.mid')
    assert alert_up(panel, 'error') == ('Could not export the MIDI file', 'No space left on device')
    answer_alert(panel)


# ---------------------------------------------------------------- pages and errors

def test_po32_setup_the_midi_led_and_cc_mappings_open_their_pages(panel):
    panel.run("pythonic.panel.openPage('nothing-yet')")  # an unregistered page
    title, text = alert_up(panel, 'ok')
    assert title == 'nothing-yet' and 'coming soon' in text.lower()
    answer_alert(panel)
    panel.run("window.__opened = []; for (const n of ['po32', 'setup', 'ai'])"
              " pythonic.panel.registerPage(n, (o) => window.__opened.push([n, o]))")
    panel.click('#po32-button')
    panel.click('#setup-button')
    panel.click('#midi-row')
    panel.right_click('#stage px-knob[data-address="global.master"]')
    click_item(panel, 'CC mappings…')
    open_preset_menu(panel)
    click_item(panel, 'import from PO-32…')
    open_preset_menu(panel)
    click_item(panel, 'AI drum generator…')
    panel.wait_js("window.__opened.length === 6")
    assert panel.js("window.__opened") == [
        ['po32', {}], ['setup', {}], ['setup', {'tab': 'midi'}], ['setup', {'tab': 'midi'}],
        ['po32', {'tab': 'import'}], ['ai', {}]]


def test_a_core_error_nobody_waited_for_needs_reading(panel):
    panel.core._post({'id': None, 'verb': None, 'status': 'error', 'source': 'ai',
                      'error': 'the AI worker stopped (exit code 1)'})
    assert alert_up(panel, 'error') == ('The AI generator stopped', 'the AI worker stopped (exit code 1)')
    assert display(panel)[0] == 'AI ERROR'
    answer_alert(panel)


def test_a_click_on_the_selected_channel_hits_it(panel, qtbot):
    core = panel.core
    panel.click('.strip[data-channel="1"] .chb')
    qtbot.waitUntil(lambda: ('trigger', 0, 64) in core.calls)
    panel.wait_js("pythonic.panel.display.text()[1] === 'hit 64'")
