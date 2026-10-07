"""
The PO-32 module of the app core (slice 9): the transfer signal, sending it
through the live stream, listening to and recording the input, decoding a
modem signal made with the codec, bank and pattern picks, the import as one
undo step and the preview. Every test drives the core through its interface
(set / get / act / poll) with the fake audio backend: the output callback is
pulled by hand and the input stream is fed by hand.
"""

import time

import numpy as np
import pytest

from pythonic.app import AppCore
from pythonic.app import po32 as po32_module
from pythonic.drum_channel import DrumChannel
from pythonic.po32_codec import (TAG_PATCH, TAG_PATTERN, TAG_STATE, TAG_TRAILER, ModemEncoder,
                                 bit_reverse_bytes, default_right_patch, default_state,
                                 generate_fsk_signal, save_wav)
from pythonic.po32_decoder import decode_audio_samples, decode_wav_file
from tests.fake_audio import FakeAudioBackend

RATE = 44100


# ====================================================================== helpers
@pytest.fixture
def make_core(prefs):
    cores = []

    def make(backend=None, **kwargs):
        kwargs.setdefault('preferences', prefs)
        kwargs.setdefault('stall_timeout', None)
        core = AppCore(audio_backend=backend, **kwargs)
        cores.append(core)
        if backend is not None:
            event = core.wait(core.start())
            assert event['status'] == 'done', event
        return core

    yield make
    for core in cores:
        core.close()


@pytest.fixture
def backend():
    return FakeAudioBackend()


def finish(core, action_id, stream=None, timeout=20.0):
    """The action's final event, pulling output blocks while it runs."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if stream is not None:
            stream.pull()
        try:
            return core.wait(action_id, timeout=0.002 if stream is not None else 0.2)
        except TimeoutError:
            continue
    raise AssertionError('the action did not finish')


def run(core, verb, stream=None, **args):
    event = finish(core, core.act(verb, **args), stream)
    assert event['status'] == 'done', event
    return event['result']


def fails(core, verb, stream=None, **args):
    event = finish(core, core.act(verb, **args), stream)
    assert event['status'] == 'error', event
    return event['error']


def sound(freq, decay=300.0, mix=0.8):
    params = DrumChannel(0, RATE).get_parameters()
    params.update(osc_frequency=freq, osc_decay=decay, osc_noise_mix=mix)
    return params


def card_signal(sounds, patterns):
    """A PO-32 card transfer made with the codec: 16 sounds (left and morph
    patches) and the given patterns ({number 0-15: 16 trigger bitmasks})."""
    encoder = ModemEncoder()
    for drum, params in enumerate(sounds):
        encoder.append(TAG_PATCH, bytes([0x10 | drum]) + po32_module.serialize_sound(params))
        encoder.append(TAG_PATCH, bytes([0x20 | drum]) + default_right_patch(drum))
    for number, triggers in sorted(patterns.items()):
        encoder.append(TAG_PATTERN, bytes([number]) + bytes(triggers) + bytes(194))
    encoder.append(TAG_STATE, default_state())
    encoder.append(TAG_TRAILER, b'')
    return generate_fsk_signal(bit_reverse_bytes(encoder.get_packet()))


CARD_SOUNDS = [sound(60.0 + 20.0 * d) for d in range(16)]
CARD_PATTERNS = {
    0: [0b1, 0, 0b100, 0] * 4,                 # drums 1 and 3 of bank 0
    2: [0, 0b10, 0, 0] * 4,                    # drum 2 (pattern 3)
    3: [0b1000, 0, 0, 0] * 4,                  # drum 4 (pattern 4)
}  # a step's trigger mask is one byte: patterns hold drums 1-8 only


@pytest.fixture(scope='module')
def card():
    return card_signal(CARD_SOUNDS, CARD_PATTERNS)


@pytest.fixture
def card_wav(tmp_path, card):
    path = tmp_path / 'card.wav'
    save_wav(card, str(path), RATE)
    return path


# ====================================================================== transfer
def test_prepare_renders_the_signal_of_the_current_sounds(make_core, tmp_path):
    core = make_core()
    core.set('ch1.osc.freq', 220.0)
    result = run(core, 'po32.prepare')
    assert result['bank'] == 0 and result['chain'] == 'A' and result['slots'] == [1]
    assert result['channels'] == list(range(1, 9))
    assert 5.0 < result['seconds'] < 15.0
    assert core.get('po32.transfer') == 'ready'
    assert core.get('po32.transfer_seconds') == result['seconds']

    saved = run(core, 'po32.save_wav', path=str(tmp_path / 'out'))
    assert saved['saved'] and saved['path'].endswith('out.wav')
    preset = decode_wav_file(saved['path'])
    assert preset.error is None
    assert sorted(preset.left_patches) == list(range(8))
    assert preset.left_patches[0].synth_params['osc_frequency'] == pytest.approx(220.0, rel=0.02)
    assert len(preset.decoded_patterns) == 1
    # A save never replaces a file unless asked to
    again = run(core, 'po32.save_wav', path=saved['path'])
    assert again == {'saved': False, 'exists': True, 'path': saved['path']}


def test_the_chain_options_follow_the_pattern_chains(make_core, tmp_path):
    core = make_core()
    assert core.get('po32.chain_options')[:3] == ['A', 'B', 'C']
    run(core, 'pattern.chain_next', pattern='A')
    run(core, 'pattern.chain_next', pattern='B')
    options = core.get('po32.chain_options')
    assert options[:2] == ['A - C', 'D']
    result = run(core, 'po32.prepare', chain='A - C', bank=1)
    assert result['slots'] == [1, 2, 3] and result['chain'] == 'A - C'
    saved = run(core, 'po32.save_wav', path=str(tmp_path / 'x.wav'), chain='A - C', bank=1)
    preset = decode_wav_file(saved['path'])
    assert sorted(preset.left_patches) == list(range(8, 16))  # sounds 9-16
    assert len(preset.decoded_patterns) == 3
    assert all(not any(dp.triggers) for dp in preset.decoded_patterns)  # sent empty
    assert 'not a pattern chain' in fails(core, 'po32.prepare', chain='Z')


def test_channels_left_out_are_sent_silent(make_core, tmp_path):
    core = make_core()
    core.set('ch3.mute', True)
    result = run(core, 'po32.prepare')
    assert result['channels'] == [1, 2, 4, 5, 6, 7, 8]  # default: the face mutes
    saved = run(core, 'po32.save_wav', path=str(tmp_path / 'm.wav'), channels=[1, 2])
    preset = decode_wav_file(saved['path'])
    silent = po32_module.SILENT_PATCH
    assert preset.left_patches[0].raw_params != silent
    assert preset.left_patches[1].raw_params != silent
    for drum in range(2, 8):
        assert preset.left_patches[drum].raw_params == silent


def test_send_plays_the_signal_alone_through_the_live_stream(make_core, backend):
    core = make_core(backend)
    stream = backend.stream
    core.set('ch1.osc.freq', 330.0)
    # A pattern playing and a pad hit never reach the receiver
    core.set('pattern.A.ch2.step1.trig', True)
    run(core, 'transport.play', stream)
    version = core.poll()['version']
    action = core.act('po32.send')
    blocks = []
    end = time.monotonic() + 30.0
    event = None
    while time.monotonic() < end:
        blocks.append(stream.pull())
        if len(blocks) == 5:
            core.trigger(0, 127)
        core.poll()  # a front-end's frame: the progress is reported
        try:
            event = core.wait(action, timeout=0.001)
            break
        except TimeoutError:
            continue
    assert event is not None and event['status'] == 'done', event
    assert event['result']['sent'] is True
    progress = [e['progress'] for e in core.poll(version)['events']
                if e['id'] == action and e['status'] == 'progress']
    assert progress[0] == 0.0 and len(progress) >= 10
    assert progress == sorted(progress)
    assert core.get('po32.transfer') == 'sent' and core.get('po32.progress') == 1.0

    out = np.concatenate(blocks)
    assert float(np.abs(out).max()) <= po32_module.TRANSFER_PEAK + 1e-4
    signal = out[:, 0]
    start = int(np.argmax(np.abs(signal) > 0.01))
    preset = decode_audio_samples(signal[start:].astype(np.float64), RATE)
    assert preset.error is None and sorted(preset.left_patches) == list(range(8))
    assert preset.left_patches[0].synth_params['osc_frequency'] == pytest.approx(330.0, rel=0.02)


def test_send_at_another_output_rate_is_resampled(make_core, backend, prefs):
    prefs.set('audio_sample_rate', 48000)
    core = make_core(backend)
    stream = backend.stream
    assert core.get('audio.sample_rate') == 48000
    action = core.act('po32.send', channels=[1])
    blocks = []
    end = time.monotonic() + 30.0
    while time.monotonic() < end:
        blocks.append(stream.pull())
        try:
            event = core.wait(action, timeout=0.001)
            break
        except TimeoutError:
            continue
    assert event['status'] == 'done', event
    out = np.concatenate(blocks)[:, 0].astype(np.float64)
    seconds = np.count_nonzero(np.abs(out) > 1e-6) / 48000
    assert seconds == pytest.approx(event['result']['seconds'], abs=0.1)
    preset = decode_audio_samples(out, 48000)
    assert preset.error is None and len(preset.left_patches) == 8


def test_cancel_ends_the_send(make_core, backend):
    core = make_core(backend)
    stream = backend.stream
    action = core.act('po32.send')
    for _ in range(40):
        stream.pull()
        time.sleep(0.001)
    assert core.get('po32.transfer') == 'sending'
    assert 0.0 < core.get('po32.progress') < 1.0
    assert core.poll()['po32']['progress'] > 0.0
    assert run(core, 'po32.cancel', stream) == {'cancelled': True}
    event = finish(core, action, stream)
    assert event['status'] == 'cancelled'
    assert core.get('po32.transfer') == 'stopped'
    assert float(np.abs(stream.pull()).max()) == 0.0  # the synth again, silent


def test_send_needs_the_output_stream(make_core):
    core = make_core()
    error = fails(core, 'po32.send')
    assert 'not running' in error
    assert core.get('po32.error') == error and core.get('po32.transfer') == 'error'


def test_a_stopped_stream_ends_the_send_with_an_error(make_core, backend):
    core = make_core(backend)
    stream = backend.stream
    action = core.act('po32.send')
    for _ in range(10):
        stream.pull()
        time.sleep(0.001)
    run(core, 'audio.stop')
    event = finish(core, action)
    assert event['status'] == 'error' and 'stream' in event['error']
    assert core.get('po32.transfer') == 'error'


# ====================================================================== input
def test_listen_reports_the_input_level(make_core, backend, prefs):
    core = make_core(backend)
    result = run(core, 'po32.listen')
    assert result == {'listening': True, 'device': 'Fake In'}  # the system default input
    stream = backend.input
    assert stream.started and stream.samplerate == RATE and stream.channels == 1
    stream.push(0.5 * np.sin(np.arange(4096) * 0.1))
    assert core.get('po32.level') == pytest.approx(0.5, abs=0.01)
    assert core.poll()['po32']['level'] == pytest.approx(0.5, abs=0.01)
    assert core.get('po32.listening') and core.get('po32.input') == 'Fake In'
    run(core, 'po32.listen', on=False)
    assert stream.aborted and stream.closed
    assert not core.get('po32.listening') and core.get('po32.level') == 0.0

    # The saved input device is used, an explicit one wins
    core.set('pref.audio.input_device', 'Fake Duplex')
    assert run(core, 'po32.listen')['device'] == 'Fake Duplex'
    assert backend.input.device == 2
    assert run(core, 'po32.listen', device='Fake In')['device'] == 'Fake In'
    assert 'not found' in fails(core, 'po32.listen', device='Nope')
    assert core.get('po32.error')


def test_record_and_stop_decodes_a_modem_signal(make_core, backend, card):
    core = make_core(backend)
    assert run(core, 'po32.record') == {'recording': True, 'device': 'Fake In'}
    assert core.get('po32.recording')
    stream = backend.input
    stream.push(np.zeros(RATE // 2))
    stream.push(card * 0.6)
    stream.push(np.zeros(RATE // 2))
    assert core.get('po32.recorded_seconds') == pytest.approx(len(card) / RATE + 1.0, abs=0.05)
    result = run(core, 'po32.stop')
    assert result == {'source': 'recorded audio', 'drums': 16, 'patterns': 3, 'card': True,
                      'banks': [True, True]}
    assert stream.closed and not core.get('po32.recording') and not core.get('po32.listening')
    assert core.get('po32.decode') == 'decoded'
    assert core.get('po32.decoded') == result
    # The non-empty patterns are picked and lettered, the first is focused
    assert core.get('po32.patterns') == [
        {'number': 1, 'empty': False, 'summary': 'Pattern 1: 8/16 steps active, 2 drums, bank 0'},
        {'number': 2, 'empty': False, 'summary': 'Pattern 3: 4/16 steps active, 1 drums, bank 0'},
        {'number': 3, 'empty': False, 'summary': 'Pattern 4: 4/16 steps active, 1 drums, bank 0'},
    ]
    assert core.get('po32.picks') == [{'pattern': 1, 'letter': 'A'}, {'pattern': 2, 'letter': 'B'},
                                      {'pattern': 3, 'letter': 'C'}]
    assert core.get('po32.focus') == 1 and core.get('po32.bank') == 0
    grid = core.get('po32.grid')
    assert grid[0] == [True, False, False, False] * 4 and grid[2][2] and not any(grid[1])
    sounds = core.get('po32.sounds')
    assert len(sounds) == 8 and sounds[0].startswith('Sine 60Hz')


def test_a_debug_recording_is_saved(make_core, backend, card, tmp_path, monkeypatch):
    monkeypatch.setattr(po32_module, 'recordings_folder', lambda: str(tmp_path / 'rec'))
    core = make_core(backend)
    core.set('pref.po32.save_recordings', True)
    assert core.preferences.get('po32_debug_save_recordings') is True
    run(core, 'po32.record')
    backend.input.push(card)
    result = run(core, 'po32.stop')
    saved = list((tmp_path / 'rec').glob('po32_recording_*.wav'))
    assert len(saved) == 1 and result['source'] == f'recorded audio ({saved[0].name})'


def test_a_full_recording_stops_and_decodes(make_core, backend, card, monkeypatch):
    monkeypatch.setattr(po32_module, 'MAX_RECORD_SECONDS', len(card) / RATE + 0.5)
    core = make_core(backend)
    run(core, 'po32.record')
    backend.input.push(np.concatenate([card, np.zeros(RATE)]))
    end = time.monotonic() + 10.0
    while core.get('po32.decode') != 'decoded' and time.monotonic() < end:
        core.poll()
        time.sleep(0.01)
    assert core.get('po32.decode') == 'decoded'
    assert not core.get('po32.recording') and core.get('po32.decoded')['drums'] == 16


def test_decode_errors_are_reported(make_core, backend, tmp_path):
    core = make_core(backend)
    noise = tmp_path / 'noise.wav'
    save_wav(np.random.default_rng(1).normal(0, 0.1, RATE), str(noise), RATE)
    error = fails(core, 'po32.decode', path=str(noise))
    assert error.startswith('Failed to decode PO-32 data')
    assert core.get('po32.decode') == 'error' and core.get('po32.error') == error
    assert core.get('po32.decoded') is None
    assert 'nothing decoded' in fails(core, 'po32.import')
    assert 'not recording' in fails(core, 'po32.stop')
    backend.fail_input = True
    assert 'Could not open' in fails(core, 'po32.record')


# ====================================================================== picks
def test_picks_bank_and_focus(make_core, card_wav):
    core = make_core()
    result = run(core, 'po32.decode', path=str(card_wav))
    assert result['source'] == 'card.wav'
    # Bank 1: the sounds 9-16 and the patterns' triggers of drums 9-16
    run(core, 'po32.focus', pattern=3)
    assert core.get('po32.grid')[3] == [True, False, False, False] * 4
    run(core, 'po32.select_bank', bank=1)
    assert core.get('po32.bank') == 1
    assert core.get('po32.sounds')[0].startswith('Sine 220Hz')
    assert not any(any(row) for row in core.get('po32.grid'))
    run(core, 'po32.select_bank', bank=0)

    # Toggle off and on: a new pick takes the first free letter
    assert run(core, 'po32.pick', pattern=2) == {'pattern': 2, 'picked': False, 'letter': None}
    assert core.get('po32.focus') == 2
    assert run(core, 'po32.pick', pattern=2)['letter'] == 'B'
    # A letter another pick holds swaps the letters
    run(core, 'po32.pick', pattern=1, letter='C')
    assert core.get('po32.picks') == [{'pattern': 1, 'letter': 'C'}, {'pattern': 2, 'letter': 'B'},
                                      {'pattern': 3, 'letter': 'A'}]
    run(core, 'po32.pick_clear')
    assert core.get('po32.picks') == []
    run(core, 'po32.pick_first')
    assert [p['letter'] for p in core.get('po32.picks')] == ['A', 'B', 'C']
    assert 'not a decoded pattern' in fails(core, 'po32.pick', pattern=9)
    assert 'not a pattern letter' in fails(core, 'po32.pick', pattern=1, letter='M')


def test_at_most_twelve_patterns_are_picked(make_core, tmp_path):
    core = make_core()
    patterns = {n: [1 << (n % 8)] + [0] * 15 for n in range(16)}
    path = tmp_path / 'full.wav'
    save_wav(card_signal(CARD_SOUNDS, patterns), str(path), RATE)
    run(core, 'po32.decode', path=str(path))
    picks = core.get('po32.picks')
    assert [p['pattern'] for p in picks] == list(range(1, 13))
    assert [p['letter'] for p in picks] == list('ABCDEFGHIJKL')
    assert 'picked already' in fails(core, 'po32.pick', pattern=14)
    run(core, 'po32.pick', pattern=5)  # frees E
    assert run(core, 'po32.pick', pattern=14)['letter'] == 'E'


# ====================================================================== import
def test_import_writes_the_sounds_and_patterns_as_one_undo_step(make_core, card_wav):
    core = make_core()
    core.set('pattern.K.ch1.step3.trig', True)
    core.set('pattern.B.length', 8)
    core.set('pattern.A.length', 32)
    core.set('pattern.A.ch4.step20.trig', True)
    core.set('pattern.A.ch1.step1.vel', 100)
    core.set('ch8.osc.freq', 1234.0)
    before = {name: core.get(name) for name in ('ch1.osc.freq', 'ch8.osc.freq',
                                                 'pattern.K.ch1.trig', 'pattern.A.ch4.trig')}
    names = [core.get(f'ch{c}.name') for c in range(1, 9)]
    run(core, 'po32.decode', path=str(card_wav))
    run(core, 'po32.pick', pattern=2, letter='B')  # pattern 3 of the card on B, 1 on A, 4 on C
    version = core.poll()['version']

    result = run(core, 'po32.import')
    assert result == {'drums': 8, 'patterns': [{'pattern': 1, 'letter': 'A'},
                                               {'pattern': 2, 'letter': 'B'},
                                               {'pattern': 3, 'letter': 'C'}]}
    assert core.get('po32.imported')
    for c in range(1, 9):
        assert core.get(f'ch{c}.osc.freq') == pytest.approx(60.0 + 20.0 * (c - 1), rel=0.02)
        assert core.get(f'ch{c}.name') == names[c - 1]  # the PO-32 sends no names
    assert core.get('pattern.A.ch1.trig')[:16] == [True, False, False, False] * 4
    assert core.get('pattern.A.ch3.trig')[:16] == [False, False, True, False] * 4
    assert core.get('pattern.A.length') == 32 and not any(core.get('pattern.A.ch4.trig'))
    assert core.get('pattern.A.ch1.vel')[0] == 64
    assert core.get('pattern.B.length') == 16  # grown to hold the 16 steps
    assert core.get('pattern.B.ch2.trig') == [False, True, False, False] * 4
    assert core.get('pattern.C.ch4.trig') == [True, False, False, False] * 4
    assert core.get('pattern.K.empty')
    assert core.get('morph.position') == 0.0
    changes = core.poll(version)['changes']
    assert 'ch1.osc.freq' in changes and 'pattern.A.ch1.trig' in changes

    assert run(core, 'undo') == {'done': True, 'label': 'PO-32 import'}
    after = {name: core.get(name) for name in before}
    assert after == before
    assert core.get('pattern.B.length') == 8 and core.get('pattern.A.ch1.vel')[0] == 100


def test_import_sets_the_morph_endpoints(make_core, card_wav):
    core = make_core()
    run(core, 'po32.decode', path=str(card_wav))
    run(core, 'po32.import')
    # Endpoint B is the PO-32's morph patch: moving the morph changes the sound
    a = core.get('ch1.osc.freq')
    core.set('morph.position', 1.0)
    assert core.get('morph.differs')
    assert core.get('ch1.osc.freq') != pytest.approx(a)


# ====================================================================== preview
def test_preview_stops_the_transport_and_loops_the_pattern(make_core, backend, card_wav):
    core = make_core(backend)
    stream = backend.stream
    run(core, 'po32.decode', path=str(card_wav), stream=stream)
    run(core, 'transport.play', stream)
    assert core.poll()['transport']['playing']
    assert run(core, 'po32.preview', stream) == {'previewing': True, 'pattern': 1}
    assert not core.poll()['transport']['playing']
    assert core.get('po32.previewing')
    out = np.concatenate([stream.pull() for _ in range(60)])
    assert float(np.abs(out).max()) > 0.05
    assert core.get('po32.preview_step') > 0
    assert run(core, 'po32.preview', stream, on=False)['stopped']
    for _ in range(200):  # the tails die out
        stream.pull()
    assert float(np.abs(stream.pull()).max()) < 1e-3
    assert not core.poll()['transport']['playing'] and not core.get('po32.previewing')
    assert core.get('po32.preview_step') == -1


def test_starting_the_transport_ends_the_preview(make_core, backend, card_wav):
    core = make_core(backend)
    stream = backend.stream
    run(core, 'po32.decode', path=str(card_wav), stream=stream)
    run(core, 'po32.preview', stream)
    run(core, 'transport.play', stream)
    stream.pull()
    end = time.monotonic() + 5.0
    while core.get('po32.previewing') and time.monotonic() < end:
        core.poll()
        stream.pull()
        time.sleep(0.005)
    assert not core.get('po32.previewing')


def test_preview_needs_a_decode_and_the_stream(make_core, card_wav):
    core = make_core()
    assert 'nothing decoded' in fails(core, 'po32.preview')
    run(core, 'po32.decode', path=str(card_wav))
    assert 'not running' in fails(core, 'po32.preview')
