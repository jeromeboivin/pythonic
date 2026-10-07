"""Page tests of web slice W7 over the fake core: the PO-32 page, a drawer
page with a transfer tab (choose › prepare › send) and an import tab (listen
› bank › pick patterns › import), each a numbered stage flow; the PO-32
button and the PRESET menu open it, it remembers its tab, closing it stops
a send, a preview and the input and restores a closed drawer."""

import pytest

from tests.web.page import same_path
from tests.web.test_menus_page import alert_up, answer_alert, click_item, display, open_preset_menu

LETTERS = 'ABCDEFGHIJKL'
SOUNDS = [f'Sine {60 + 20 * d}Hz, decay 300ms' for d in range(8)]


def acts(core, verb):
    return [c[2] for c in core.calls if c[:2] == ('act', verb)]


def wait_act(panel, verb, count=1):
    panel.qtbot.waitUntil(lambda: len(acts(panel.core, verb)) >= count)
    return acts(panel.core, verb)


def states(panel, flow):
    return panel.js(f"[...document.querySelectorAll('[data-flow={flow}] .po-stage')].map((s) => s.dataset.state)")


def texts(panel, selector):
    return panel.js(f"[...document.querySelectorAll({selector!r})].map((e) => e.textContent)")


class FakePo32:
    """The core's po32 verbs, enough for the page: they keep the po32.*
    addresses as the core does (picks lettered in order, a conflict swaps)."""

    def __init__(self, core, patterns=3):
        self.core = core
        self.count = patterns
        self.picks = {}
        self.focus = 0
        self.bank = 0
        core.verbs.update({
            'po32.prepare': self.prepare, 'po32.send': lambda **a: core.DEFERRED,
            'po32.cancel': lambda: {'cancelled': True}, 'po32.listen': self.listen,
            'po32.record': self.record, 'po32.stop': self.stop, 'po32.decode': self.decode,
            'po32.select_bank': self.select_bank, 'po32.focus': self.set_focus, 'po32.pick': self.pick,
            'po32.pick_first': self.pick_first, 'po32.pick_clear': self.pick_clear,
            'po32.preview': self.preview, 'po32.import': self.do_import,
        })

    def post(self, **values):
        for name, value in values.items():
            self.core.post_change(f'po32.{name}', value)

    def prepare(self, bank=0, chain=None, channels=None):
        self.post(transfer='ready', transfer_seconds=6.4)
        slots = [1, 2] if chain == 'A - B' else [1]
        return {'seconds': 6.4, 'bank': bank, 'chain': chain, 'slots': slots, 'channels': channels}

    def listen(self, on=True, device=None):
        self.post(listening=on, input=(device or 'Fake In') if on else None, recording=False)
        return {'listening': on, 'device': device}

    def record(self, device=None):
        self.post(listening=True, recording=True, input=device or 'Fake In', decode='none', decoded=None)
        return {'recording': True, 'device': device}

    def stop(self):
        self.post(recording=False, listening=False)
        return self.decode('recorded audio')

    def decode(self, path):
        summary = {'source': str(path).split('/')[-1], 'drums': 16, 'patterns': self.count, 'card': True,
                   'banks': [True, True]}
        self.picks = {n: LETTERS[n - 1] for n in range(1, min(self.count, 12) + 1)}
        self.focus = 1
        self.post(decode='decoded', decoded=summary, banks=[True, True], bank=0, sounds=SOUNDS,
                  patterns=[{'number': n, 'empty': n > 2, 'summary': f'Pattern {n}: 4/16 steps active'}
                            for n in range(1, self.count + 1)],
                  imported=False, grid=[[s % 4 == 0 for s in range(16)]] + [[False] * 16] * 7)
        self.publish()
        return summary

    def publish(self):
        self.post(picks=[{'pattern': p, 'letter': letter} for p, letter in sorted(self.picks.items())],
                  focus=self.focus)

    def select_bank(self, bank):
        self.bank = bank
        self.post(bank=bank, previewing=False)
        return {'bank': bank}

    def set_focus(self, pattern):
        self.focus = pattern
        self.publish()
        return {'focus': pattern}

    def pick(self, pattern, picked=None, letter=None):
        if letter is not None:
            picked = True
        if picked is None:
            picked = pattern not in self.picks
        if picked and pattern not in self.picks:
            if len(self.picks) >= 12:
                raise ValueError('12 patterns are picked already')
            used = set(self.picks.values())
            self.picks[pattern] = next(x for x in LETTERS if x not in used)
        elif not picked:
            self.picks.pop(pattern, None)
        if letter is not None:
            old = self.picks[pattern]
            for other, held in list(self.picks.items()):
                if other != pattern and held == letter:
                    self.picks[other] = old
            self.picks[pattern] = letter
        self.focus = pattern
        self.post(previewing=False)
        self.publish()
        return {'pattern': pattern, 'picked': pattern in self.picks, 'letter': self.picks.get(pattern)}

    def pick_first(self):
        self.picks = {n: LETTERS[n - 1] for n in range(1, min(self.count, 12) + 1)}
        self.publish()
        return {}

    def pick_clear(self):
        self.picks = {}
        self.publish()
        return {'picks': []}

    def preview(self, on=True):
        self.post(previewing=on)
        if on:
            self.core.transport['playing'] = False
        return {'previewing': on, 'pattern': self.focus}

    def do_import(self):
        self.post(imported=True)
        return {'drums': 8, 'patterns': [{'pattern': p, 'letter': x} for p, x in sorted(self.picks.items())]}


@pytest.fixture
def po(panel):
    return FakePo32(panel.core)


def open_po32(panel, tab=None):
    options = f"{{ tab: '{tab}' }}" if tab else '{}'
    panel.run(f"pythonic.panel.openPage('po32', {options})")
    panel.wait_js("pythonic.panel.drawer.current === 'po32'")
    if tab:
        panel.wait_js(f"document.querySelector('#po32-page').dataset.tab === '{tab}'")


def decoded(panel, po, count=3):
    po.count = count
    po.decode('card.wav')
    panel.wait_js("document.querySelector('#po32-source').textContent.startsWith('decoded')")


# ---------------------------------------------------------------- the page

def test_the_po32_button_opens_the_page_on_its_last_tab_and_toggles_it(panel, po):
    panel.click('#po32-button')
    panel.wait_js("pythonic.panel.drawer.current === 'po32'")
    assert panel.js("document.querySelector('#po32-page').dataset.tab") == 'transfer'
    assert panel.js("document.querySelector('#po32-button').classList.contains('on')")
    panel.click('.po-tabs [data-tab="import"]')
    panel.wait_js("document.querySelector('#po32-page').dataset.tab === 'import'")
    assert panel.js("[...document.querySelectorAll('.po-tabs .btn.on')].map((b) => b.dataset.tab)") == ['import']
    panel.click('#po32-button')  # again: closes
    panel.wait_js("pythonic.panel.drawer.current === null")
    assert not panel.js("document.querySelector('#po32-button').classList.contains('on')")
    panel.click('#po32-button')  # the last tab
    panel.wait_js("pythonic.panel.drawer.current === 'po32' && document.querySelector('#po32-page').dataset.tab === 'import'")


def test_the_preset_menu_opens_the_tab_it_names(panel, po):
    open_preset_menu(panel)
    click_item(panel, 'import from PO-32…')
    panel.wait_js("pythonic.panel.drawer.current === 'po32' && document.querySelector('#po32-page').dataset.tab === 'import'")
    open_preset_menu(panel)
    click_item(panel, 'transfer to PO-32…')
    panel.wait_js("document.querySelector('#po32-page').dataset.tab === 'transfer'")
    assert panel.js("pythonic.panel.drawer.current") == 'po32'


def test_back_follows_the_selected_channel_and_closing_restores_a_closed_rack(panel, po):
    panel.run("pythonic.panel.rack.setOpen(false, { save: false })")
    panel.wait_js("!pythonic.panel.rack.open")
    open_po32(panel)
    panel.wait_js("pythonic.panel.rack.open")  # the page opened the drawer
    panel.click('.strip[data-channel="3"] .chb')  # channels still select
    panel.wait_js("document.querySelector('#po32-back').textContent === '◀ ch 3 edit'")
    assert panel.js("pythonic.panel.drawer.current") == 'po32'
    panel.click('#po32-back')
    panel.wait_js("pythonic.panel.drawer.current === null && !pythonic.panel.rack.open")
    open_po32(panel)
    panel.click('#po32-close')
    panel.wait_js("pythonic.panel.drawer.current === null")


def test_closing_stops_a_send_a_preview_and_the_input(panel, po):
    open_po32(panel, 'import')
    po.post(transfer='sending', previewing=True, listening=True)
    panel.wait_js("document.querySelector('#po32-preview').textContent === '■ stop'")
    panel.click('#po32-close')
    assert wait_act(panel, 'po32.cancel') == [{}]
    assert wait_act(panel, 'po32.preview') == [{'on': False}]
    assert wait_act(panel, 'po32.listen') == [{'on': False}]


# ---------------------------------------------------------------- transfer

def test_choose_starts_from_the_face_mutes_and_prepares_on_every_change(panel, po):
    core = panel.core
    core.post_change('ch2.mute', True)
    core.post_change('ch1.name', 'Kick')
    core.post_change('po32.chain_options', ['A - B', 'C'])
    panel.wait_js("pythonic.store.value('ch2.mute') === true && pythonic.store.value('ch1.name') === 'Kick'")
    open_po32(panel, 'transfer')
    (first,) = wait_act(panel, 'po32.prepare')
    assert first == {'bank': 0, 'chain': 'A - B', 'channels': [1, 3, 4, 5, 6, 7, 8]}
    panel.wait_js("document.querySelector('#po32-slots').textContent.startsWith('PO-32 patterns 1–2 are sent empty')")
    panel.wait_js("document.querySelector('#po32-status').textContent === 'ready: 6.4 s of signal'")
    assert states(panel, 'transfer') == ['now', 'later', 'later']
    assert panel.js("document.querySelector('.po-check[data-channel=\"1\"] .nm').textContent") == 'Kick'
    assert panel.js("[...document.querySelectorAll('.po-check')].map((c) => c.classList.contains('on'))") == [
        True, False, True, True, True, True, True, True]

    panel.click('.po-check[data-channel="2"]')  # independent of MUTE from now on
    assert wait_act(panel, 'po32.prepare', 2)[1]['channels'] == [1, 2, 3, 4, 5, 6, 7, 8]
    panel.click('[data-send-bank="1"]')
    assert wait_act(panel, 'po32.prepare', 3)[2]['bank'] == 1
    panel.click('#po32-chain')
    click_item(panel, 'C', '.px-menu.list .it')
    assert wait_act(panel, 'po32.prepare', 4)[3]['chain'] == 'C'
    panel.wait_js("document.querySelector('#po32-slots').textContent.startsWith('PO-32 pattern 1 is sent empty')")
    assert core.values['ch2.mute'] is True  # the face MUTE is left alone


def test_send_shows_progress_and_stops(panel, po):
    core = panel.core
    open_po32(panel, 'transfer')
    wait_act(panel, 'po32.prepare')
    panel.click('#po32-send')
    (send,) = wait_act(panel, 'po32.send')
    assert send == {'bank': 0, 'chain': 'A', 'channels': [1, 2, 3, 4, 5, 6, 7, 8]}
    action = core.running
    po.post(transfer='sending')
    panel.wait_js("document.querySelector('#po32-send').textContent === '■ stop'")
    assert states(panel, 'transfer') == ['done', 'done', 'now']
    core.progress(action, 0.42)
    core.readouts['po32'] = {'level': 0.0, 'recorded_seconds': 0.0, 'progress': 0.42, 'preview_step': -1}
    panel.wait_js("pythonic.panel.display.text()[0] === 'PO-32 SEND' && pythonic.panel.display.text()[1] === '42 %'")
    panel.wait_js("document.querySelector('#po32-progress i').style.width === '42%'")
    assert panel.js("document.querySelector('#po32-status').textContent") == 'sending… 42 %'
    assert panel.js("document.querySelector('#po32-save-wav').disabled")
    panel.click('#po32-send')  # stop
    wait_act(panel, 'po32.cancel')
    core._post({'id': action, 'verb': 'po32.send', 'status': 'cancelled', 'result': {'sent': False}})
    po.post(transfer='stopped')
    panel.wait_js("pythonic.panel.display.text()[1] === 'stopped'")
    panel.wait_js("document.querySelector('#po32-status').textContent === 'stopped'")


def test_a_finished_send_ticks_every_stage(panel, po):
    core = panel.core
    open_po32(panel, 'transfer')
    panel.click('#po32-send')
    wait_act(panel, 'po32.send')
    core.finish(core.running, {'sent': True, 'seconds': 6.4})
    po.post(transfer='sent')
    panel.wait_js("pythonic.panel.display.text()[1] === 'sent'")
    panel.wait_js("[...document.querySelectorAll('[data-flow=transfer] .po-stage')].every((s) => s.dataset.state === 'done')")
    assert panel.js("document.querySelector('#po32-progress i').style.width") == '100%'


def test_a_failed_send_says_why_on_a_red_alert(panel, po):
    core = panel.core

    def no_stream(**args):
        raise RuntimeError('the audio output is not running (save the transfer as a WAV file instead)')
    core.verbs['po32.send'] = no_stream
    open_po32(panel, 'transfer')
    panel.click('#po32-send')
    assert alert_up(panel, 'error') == ('Could not send to the PO-32',
                                        'the audio output is not running (save the transfer as a WAV file instead)')
    answer_alert(panel)


def test_save_wav_goes_through_the_save_dialog_and_asks_before_replacing(panel, po, tmp_path):
    core = panel.core
    target = tmp_path / 'po.wav'
    answers = iter([{'saved': False, 'exists': True, 'path': str(target)},
                    {'saved': True, 'exists': True, 'path': str(target), 'seconds': 6.4}])
    core.verbs['po32.save_wav'] = lambda **args: next(answers)
    core.post_change('preset.name', 'My Kit')
    open_po32(panel, 'transfer')
    panel.click('#po32-save-wav')
    panel.answer_dialog(target)
    title, _ = alert_up(panel, 'ok')
    assert title == 'Replace “po.wav”?'
    answer_alert(panel)
    first, second = wait_act(panel, 'po32.save_wav', 2)
    assert same_path(first['path'], target) and 'overwrite' not in first
    assert second['overwrite'] is True and second['channels'] == [1, 2, 3, 4, 5, 6, 7, 8]
    panel.wait_js("document.querySelector('#po32-saved').textContent === 'saved po.wav'")
    assert display(panel) == ['PO-32 WAV', 'saved po.wav']


# ---------------------------------------------------------------- import

def test_listen_monitors_records_and_decodes(panel, po):
    core = panel.core
    open_po32(panel, 'import')
    assert states(panel, 'import') == ['now', 'later', 'later', 'later']
    panel.wait_js("document.querySelector('#po32-input').textContent === 'Fake In'")
    panel.click('#po32-monitor')
    assert wait_act(panel, 'po32.listen') == [{'on': True, 'device': 'Fake In'}]
    panel.wait_js("document.querySelector('#po32-monitor').classList.contains('on')")
    core.readouts['po32'] = {'level': 0.5, 'recorded_seconds': 0.0, 'progress': 0.0, 'preview_step': -1}
    panel.wait_js("document.querySelector('.po-db').textContent === '-6.0 dB'")
    assert panel.js("document.querySelector('#po32-meter').dataset.tone") == 'good'

    panel.click('#po32-record')
    assert wait_act(panel, 'po32.record') == [{'device': 'Fake In'}]
    panel.wait_js("document.querySelector('#po32-record').textContent === '■ stop'")
    core.readouts['po32'] = {'level': 0.2, 'recorded_seconds': 3.2, 'progress': 0.0, 'preview_step': -1}
    panel.wait_js("document.querySelector('#po32-source').textContent === 'recording 3.2 s: play the PO-32 transfer now'")
    panel.click('#po32-record')
    wait_act(panel, 'po32.stop')
    panel.wait_js("document.querySelector('#po32-source').textContent === "
                  "'decoded PO-32 card: 16 sounds, 3 patterns · recorded audio'")
    assert states(panel, 'import') == ['done', 'done', 'done', 'now']
    assert panel.js("pythonic.panel.display.text()") == ['PO-32 IMPORT', 'decoded 16 + 3']


def test_choosing_another_input_reopens_the_monitor(panel, po):
    core = panel.core
    open_po32(panel, 'import')
    po.listen(True)
    panel.wait_js("document.querySelector('#po32-monitor').classList.contains('on')")
    panel.click('#po32-input')
    click_item(panel, 'Fake Duplex', '.px-menu.list .it')
    assert wait_act(panel, 'po32.listen') == [{'on': True, 'device': 'Fake Duplex'}]
    panel.click('#po32-rescan')
    wait_act(panel, 'audio.rescan')


def test_an_imported_wav_decodes_and_a_bad_one_says_why(panel, po, tmp_path):
    core = panel.core
    wav = tmp_path / 'card.wav'
    wav.write_bytes(b'RIFF')
    open_po32(panel, 'import')
    panel.click('#po32-open-wav')
    panel.answer_dialog(wav)
    (decode,) = wait_act(panel, 'po32.decode')
    assert same_path(decode['path'], wav)
    panel.wait_js("document.querySelector('#po32-source').textContent.endsWith('card.wav')")

    def bad(path):
        raise RuntimeError('Failed to decode PO-32 data: no sounds or patterns in the signal')
    core.verbs['po32.decode'] = bad
    panel.click('#po32-open-wav')
    panel.answer_dialog(wav)
    assert alert_up(panel, 'error') == ('Could not decode the PO-32 signal',
                                        'Failed to decode PO-32 data: no sounds or patterns in the signal')
    answer_alert(panel)


def test_banks_show_their_sounds_and_only_decoded_ones_work(panel, po):
    open_po32(panel, 'import')
    assert panel.js("[...document.querySelectorAll('[data-bank]')].map((b) => b.disabled)") == [True, True]
    decoded(panel, po)
    po.post(banks=[True, False])
    panel.wait_js("document.querySelector('[data-bank=\"1\"]').disabled")
    assert not panel.js("document.querySelector('[data-bank=\"0\"]').disabled")
    assert texts(panel, '.po-sounds li')[:2] == ['Sine 60Hz', 'Sine 80Hz']
    po.post(banks=[True, True])
    panel.wait_js("!document.querySelector('[data-bank=\"1\"]').disabled")
    panel.click('[data-bank="1"]')
    assert wait_act(panel, 'po32.select_bank') == [{'bank': 1}]
    panel.wait_js("document.querySelector('[data-bank=\"1\"]').classList.contains('on')")


def test_picks_toggle_with_letters_in_order_and_a_letter_swaps(panel, po):
    core = panel.core
    open_po32(panel, 'import')
    decoded(panel, po, count=5)
    assert texts(panel, '.po-pat')[:6] == ['1→A', '2→B', '3→C', '4→D', '5→E', '6']
    assert panel.js("[...document.querySelectorAll('.po-pat')].map((b) => b.disabled)") == [False] * 5 + [True] * 11
    assert panel.js("document.querySelector('.po-pat[data-pattern=\"1\"]').classList.contains('focus')")
    assert panel.js("document.querySelectorAll('#po32-grid i.on').length") == 4
    panel.click('.po-pat[data-pattern="2"]')  # unpick 2
    assert wait_act(panel, 'po32.pick') == [{'pattern': 2}]
    panel.wait_js("document.querySelector('#po32-count').textContent === '4 / 12 picked'")
    panel.click('.po-pat[data-pattern="2"]')  # again: the first free letter
    panel.wait_js("document.querySelector('.po-pat[data-pattern=\"2\"]').textContent === '2→B'")
    # Right-click: the letter menu; a held letter swaps
    panel.right_click('.po-pat[data-pattern="5"]')
    click_item(panel, '→ A  (swaps with 1)', '.px-menu.po-letters .it')
    assert wait_act(panel, 'po32.pick', 3)[2] == {'pattern': 5, 'letter': 'A'}
    panel.wait_js("document.querySelector('.po-pat[data-pattern=\"1\"]').textContent === '1→E'")
    assert panel.js("document.querySelector('.po-pat[data-pattern=\"5\"]').textContent") == '5→A'
    assert panel.js("document.querySelector('#po32-focus').textContent").startswith('pattern 5 → A')
    panel.click('#po32-clear')
    wait_act(panel, 'po32.pick_clear')
    panel.wait_js("document.querySelector('#po32-count').textContent === '0 / 12 picked'")
    assert states(panel, 'import') == ['done', 'done', 'now', 'later']
    panel.click('#po32-first')
    wait_act(panel, 'po32.pick_first')
    panel.wait_js("document.querySelector('#po32-count').textContent === '5 / 12 picked'")
    assert core.values['po32.picks'][0] == {'pattern': 1, 'letter': 'A'}


def test_a_thirteenth_pick_is_refused_on_the_display(panel, po):
    open_po32(panel, 'import')
    decoded(panel, po, count=16)
    panel.wait_js("document.querySelector('#po32-count').textContent === '12 / 12 picked'")
    panel.click('.po-pat[data-pattern="14"]')
    panel.wait_js("pythonic.panel.display.text()[1] === '12 patterns picked already'")
    assert acts(panel.core, 'po32.pick') == []
    assert wait_act(panel, 'po32.focus') == [{'pattern': 14}]


def test_preview_toggles_and_its_errors_need_reading(panel, po):
    core = panel.core
    open_po32(panel, 'import')
    assert panel.js("document.querySelector('#po32-preview').disabled")
    decoded(panel, po)
    core.transport['playing'] = True
    panel.click('#po32-preview')
    assert wait_act(panel, 'po32.preview') == [{'on': True}]
    panel.wait_js("document.querySelector('#po32-preview').textContent === '■ stop'")
    assert panel.js("pythonic.panel.display.text()") == ['PO-32 PREVIEW', 'pattern 1']
    core.readouts['po32'] = {'level': 0.0, 'recorded_seconds': 0.0, 'progress': 0.0, 'preview_step': 4}
    panel.wait_js("[...document.querySelectorAll('#po32-grid i.ph')].every((c) => c.dataset.step === '5')"
                  " && document.querySelectorAll('#po32-grid i.ph').length === 8")
    panel.click('#po32-preview')
    assert wait_act(panel, 'po32.preview', 2)[1] == {'on': False}
    panel.wait_js("document.querySelector('#po32-preview').textContent === '▶ preview'")

    def silent(on=True):
        raise RuntimeError('the audio output is not running')
    core.verbs['po32.preview'] = silent
    panel.click('#po32-preview')
    assert alert_up(panel, 'error') == ('Could not preview', 'the audio output is not running')
    answer_alert(panel)


def test_import_confirms_on_the_display_and_the_page_stays_open(panel, po):
    open_po32(panel, 'import')
    assert panel.js("document.querySelector('#po32-import').disabled")
    decoded(panel, po)
    panel.wait_js("document.querySelector('#po32-import').textContent === 'import 8 sounds + 3 patterns'")
    assert panel.js("document.querySelector('#po32-summary').textContent").startswith(
        'Bank 0 replaces the sounds of channels 1–8 and all 12 patterns: 3 land on A B C')
    panel.click('#po32-import')
    wait_act(panel, 'po32.import')
    panel.wait_js("pythonic.panel.display.text()[1] === '8 sounds, 3 patterns'")
    panel.wait_js("[...document.querySelectorAll('[data-flow=import] .po-stage')].every((s) => s.dataset.state === 'done')")
    assert panel.js("pythonic.panel.drawer.current") == 'po32'


def test_keep_recordings_is_a_preference(panel, po):
    core = panel.core
    open_po32(panel, 'import')
    panel.click('#po32-keep .btn')
    panel.qtbot.waitUntil(lambda: ('pref.po32.save_recordings', True) in core.sets())
    panel.wait_js("document.querySelector('.po-folder').textContent.startsWith('in ')")
