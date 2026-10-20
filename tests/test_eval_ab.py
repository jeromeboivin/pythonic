"""
The A/B listening tool: the record of what was heard, the golden set it keeps,
the comparisons it queues and the page's API.
"""

import json
import os
import pathlib
import threading
import urllib.request

import numpy as np
import pytest
import soundfile as sf

from pythonic.preset_manager import DrumPatchWriter
from pythonic.voice import VoiceParams

E = pytest.importorskip('tools.eval_ab')
A = pytest.importorskip('tools.eval_ab_audio')
F = pytest.importorskip('tools.fit_samples')

SR = A.SR


def tone(freq, decay=0.15, seconds=0.5, level=0.5):
    t = np.arange(int(seconds * SR)) / SR
    return level * np.sin(2 * np.pi * freq * t) * np.exp(-t / decay)


def write_run(samples, fits, names, freq=100.0, score=10.0):
    """Samples and the fits of a run: a decaying sine each, and a .fit.wav a little off."""
    for i, name in enumerate(names):
        (samples / name).parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(samples / f'{name}.wav'), tone(freq + 20 * i), SR, subtype='PCM_24')
        fits.joinpath(name).parent.mkdir(parents=True, exist_ok=True)
        patch = fits / f'{name}.mtdrum'
        patch.write_text(DrumPatchWriter.format_patch(F.patch_dict(VoiceParams(), name, False)), encoding='utf-8')
        wav = fits / f'{name}.fit.wav'
        stereo = np.stack([tone(freq + 20 * i + 5, level=0.3)] * 2, axis=1)
        sf.write(str(wav), stereo, SR, subtype='PCM_24')
        os.utime(wav, (patch.stat().st_mtime + 5,) * 2)
    log = fits.with_name(fits.name + '.log')
    log.write_text(''.join(f'{pathlib.Path(n).name:32s} bd           start {score + 5:6.2f} -> fit {score + i:6.2f}  from partials (9 s)\n'
                           for i, n in enumerate(names)), encoding='utf-8')


@pytest.fixture
def world(tmp_path):
    samples, fits = tmp_path / 'Vintage', tmp_path / 'Vintage-fit'
    names = ['Machine A/BD Kick 01', 'Machine A/SD Snare 02', 'Machine B/BD Kick 03']
    write_run(samples, fits, names)
    store = E.Store(tmp_path / 'eval')
    E.register(store, samples, fits, 'first', log=lambda *_: None)
    return store, samples, fits, names


def make_api(store):
    return E.Api(store)


def test_register_snapshots_audio_and_is_idempotent(world):
    store, samples, fits, names = world
    assert store.one('SELECT count(*) FROM samples')[0] == 3
    assert store.one('SELECT count(*) FROM fits')[0] == 3
    fit = store.one('SELECT * FROM fits ORDER BY id')
    assert store.path(fit['patch_file']).is_file() and store.path(fit['wav_file']).is_file()
    assert fit['score'] == pytest.approx(10.0) and fit['start_score'] == pytest.approx(15.0)
    assert set(json.loads(fit['features'])) >= {'mel', 'pitch_st', 'decay_oct', 'click_db'}
    sample = store.one("SELECT * FROM samples WHERE rel LIKE '%Snare%'")
    assert (sample['machine'], sample['category']) == ('Machine A', 'sd')
    assert store.path(sample['trim_file']).is_file() and store.path(sample['raw_file']).is_file()
    again = E.register(store, samples, fits, 'first', log=lambda *_: None)
    assert (again['added'], again['known']) == (0, 3)
    assert store.one('SELECT count(*) FROM fits')[0] == 3


def test_a_wav_older_than_its_patch_is_rendered_again(tmp_path):
    samples, fits = tmp_path / 's', tmp_path / 'f'
    write_run(samples, fits, ['Kick 01'])
    wav = fits / 'Kick 01.fit.wav'
    os.utime(wav, (1.0, 1.0))
    store = E.Store(tmp_path / 'eval')
    E.register(store, samples, fits, 'r', log=lambda *_: None)
    fit = store.one('SELECT * FROM fits')
    stored, _ = sf.read(str(store.path(fit['wav_file'])))
    assert stored.shape[1] == 2 and len(stored) > SR // 10       # the app's stereo render, not the stale tone
    assert not np.allclose(stored[:SR // 4, 0], sf.read(str(wav))[0][:SR // 4, 0], atol=1e-3)


def test_trial_is_blind_and_worst_score_comes_first(world):
    store, *_ = world
    api = make_api(store)
    trial = api.next({'mode': 'grade'})
    assert set(trial) >= {'token', 'gain', 'name', 'machine'} and 'layout' not in trial
    sample = store.one('SELECT s.* FROM trials t JOIN samples s ON s.id=t.sample_id')
    assert sample['rel'].endswith('Kick 03')                       # score 12 is the worst of 10, 11, 12
    assert max(trial['gain'].values()) <= 0.0
    for slot in ('a', 'b', 'ref'):
        path, ctype = api.media(trial['token'], slot)
        assert path.is_file() and ctype == 'audio/wav'


def test_grade_shuffles_sides_and_reveals_after_the_answer(world):
    store, *_ = world
    api = E.Api(store, rng=__import__('random').Random(1))
    sides = set()
    for _ in range(12):
        t = api.next({'mode': 'grade', 'order': 'random'})
        sides.add(json.loads(store.one('SELECT layout FROM trials WHERE token=?', t['token'])['layout'])['a'])
    assert sides == {'orig', 'fit'}
    reveal = api.answer({'token': t['token'], 'grade': 2, 'tags': ['too dark']})
    assert reveal['orig_side'] in ('A', 'B') and reveal['grade'] == 'Weak' and not reveal['golden']
    with pytest.raises(ValueError):
        api.answer({'token': t['token'], 'grade': 2})


def test_good_grade_makes_the_sample_golden_and_leaves_the_queue(world):
    store, *_ = world
    api = make_api(store)
    t = api.next({'mode': 'grade'})
    sample_id = store.one('SELECT sample_id FROM trials WHERE token=?', t['token'])['sample_id']
    assert api.answer({'token': t['token'], 'grade': 3})['golden']
    assert store.one('SELECT grade FROM golden WHERE sample_id=?', sample_id)['grade'] == 3
    assert api.summary()['todo_grade'] == 2
    with pytest.raises(ValueError):
        api.answer({'token': api.next({'mode': 'grade'})['token'], 'grade': 9})


def new_version(store, sample_rel, freq):
    """A second fit of a sample, as a later run would register it."""
    sample = store.one('SELECT * FROM samples WHERE rel=?', sample_rel)
    old = store.one('SELECT * FROM fits WHERE sample_id=? ORDER BY id', sample['id'])
    wav = store.root / f'new-{freq}.wav'
    sf.write(str(wav), np.stack([tone(freq)] * 2, axis=1), SR, subtype='PCM_24')
    fit_id, _ = E.add_fit(store, sample['id'], store.path(old['patch_file']), wav, 'register')
    return sample['id'], old['id'], fit_id


@pytest.mark.parametrize('outcome, golden_is_new, golden', [
    ('better', True, True), ('same', False, True), ('worse', False, True), ('both_bad', None, False)])
def test_comparison_outcomes_move_the_golden_fit(world, outcome, golden_is_new, golden):
    store, *_ = world
    api = make_api(store)
    sample_id, old_fit, new_fit = new_version(store, 'Machine A/BD Kick 01', 130.0)
    store.run('INSERT INTO grades(sample_id, fit_id, grade, tags, orig_side, ts) VALUES(?,?,?,?,?,?)',
              sample_id, old_fit, 4, '[]', 'a', 1.0)
    store.recompute_golden(sample_id)
    assert store.one('SELECT fit_id FROM golden WHERE sample_id=?', sample_id)['fit_id'] == old_fit
    assert E.queue_comparison(store, sample_id, old_fit, new_fit, 'test')
    assert not E.queue_comparison(store, sample_id, old_fit, new_fit, 'test')      # once
    t = api.next({'mode': 'compare'})
    assert t['kind'] == 'compare' and t['reason'] == 'test'
    reveal = api.answer({'token': t['token'], 'outcome': outcome, 'tags': ['robotic']})
    assert reveal['golden'] is golden
    row = store.one('SELECT fit_id, grade FROM golden WHERE sample_id=?', sample_id)
    if golden_is_new is not None:
        assert row['fit_id'] == (new_fit if golden_is_new else old_fit) and row['grade'] == 4
    assert api.next({'mode': 'compare'}) is None


def test_undo_takes_back_the_last_verdict(world):
    store, *_ = world
    api = make_api(store)
    t = api.next({'mode': 'grade'})
    api.answer({'token': t['token'], 'grade': 4})
    assert store.one('SELECT count(*) FROM golden')[0] == 1
    back = api.undo()
    assert back['kind'] == 'grade'
    assert store.one('SELECT count(*) FROM golden')[0] == 0 and store.one('SELECT count(*) FROM grades')[0] == 0
    assert api.undo() is None


def test_repeats_do_not_touch_the_golden_set(world, monkeypatch):
    store, *_ = world
    monkeypatch.setattr(E, 'REPEAT_AFTER', 1)
    monkeypatch.setattr(E, 'REPEAT_RATE', 1.0)
    api = make_api(store)
    first = api.next({'mode': 'grade'})
    api.answer({'token': first['token'], 'grade': 3})
    again = api.next({'mode': 'grade'})
    row = store.one('SELECT repeat_of FROM trials WHERE token=?', again['token'])
    assert row['repeat_of'] is not None
    api.answer({'token': again['token'], 'grade': 1})
    assert store.one('SELECT grade FROM golden')['grade'] == 3
    out = []
    E.report(store, out.append)
    assert any('1 repeats: same grade 0%' in line for line in out)


def test_compare_runs_pairs_the_golden_fit_with_the_new_one(world, tmp_path):
    store, samples, fits, names = world
    api = make_api(store)
    t = api.next({'mode': 'grade'})
    sample_id = store.one('SELECT sample_id FROM trials WHERE token=?', t['token'])['sample_id']
    api.answer({'token': t['token'], 'grade': 3})
    # a second run that refits the same samples with another sound
    fits2 = tmp_path / 'fits2'
    write_run(samples, fits2, names, freq=140.0)
    E.register(store, samples, fits2, 'second', log=lambda *_: None)
    assert E.compare_runs(store, 'first', 'second', log=lambda *_: None) == 3
    row = store.one('SELECT * FROM comparisons WHERE sample_id=?', sample_id)
    assert row['old_fit'] == store.one('SELECT fit_id FROM golden WHERE sample_id=?', sample_id)['fit_id']
    assert E.compare_runs(store, 'first', 'second', log=lambda *_: None) == 0


def test_distance_is_zero_for_the_same_sound_and_sees_pitch_and_decay():
    x = tone(100.0, seconds=2.0)
    same = A.compare(x, x.copy())
    assert same['distance'] == pytest.approx(0.0, abs=1e-9)
    louder = A.compare(x, x * 2)
    assert louder['distance'] == pytest.approx(0.0, abs=1e-6) and louder['level_db'] == pytest.approx(6.02, abs=0.01)
    up = A.compare(x, tone(200.0, seconds=2.0))
    assert up['pitch_st'] == pytest.approx(12.0, abs=0.5) and up['distance'] > 20
    longer = A.compare(x, tone(100.0, decay=0.3, seconds=2.0))
    assert longer['decay_oct'] > 0.5


def test_check_queues_only_the_golden_patches_that_moved(world):
    store, *_ = world
    api = make_api(store)
    for _ in range(2):
        t = api.next({'mode': 'grade'})
        api.answer({'token': t['token'], 'grade': 3})
    # the stored renders are the stand-in tones, so the engine "moved" for both
    rows = E.check(store, 0.75, queue=False, jobs=1, log=lambda *_: None)
    assert len(rows) == 2 and store.one('SELECT count(*) FROM comparisons')[0] == 0
    # freeze a real render of one of them: nothing moves any more
    gold = store.one('SELECT f.* FROM golden g JOIN fits f ON f.id=g.fit_id ORDER BY g.sample_id')
    E.render_patch(store.path(gold['patch_file']), store.path(gold['wav_file']))
    rows = E.check(store, 0.75, queue=True, jobs=1, log=lambda *_: None)
    moved = {r[1] for r in rows if r[3]['distance'] > 0.75}
    assert gold['id'] not in moved and len(moved) == 1
    assert store.one('SELECT count(*) FROM comparisons')[0] == 1
    pair = store.one('SELECT * FROM comparisons')
    assert pair['old_fit'] in moved and pair['new_fit'] != pair['old_fit'] and pair['distance'] > 0.75
    assert store.one('SELECT source FROM fits WHERE id=?', pair['new_fit'])['source'] == 'check'


def test_http_round_trip(world):
    store, *_ = world
    server = E.make_server(store, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f'http://127.0.0.1:{server.server_address[1]}'
    try:
        get = lambda path: json.load(urllib.request.urlopen(base + path))
        post = lambda path, body: json.load(urllib.request.urlopen(
            urllib.request.Request(base + path, json.dumps(body).encode(), method='POST')))
        assert b'<title>' in urllib.request.urlopen(base + '/').read()
        summary = get('/api/summary')
        assert summary['todo_grade'] == 3 and summary['default_mode'] == 'grade'
        trial = get('/api/next?mode=grade&order=folder')
        assert trial['name'] == 'BD Kick 01'
        audio = urllib.request.urlopen(f'{base}/media/{trial["token"]}/a').read()
        assert audio[:4] == b'RIFF'
        reveal = post('/api/answer', {'token': trial['token'], 'grade': 4, 'tags': [], 'note': 'x'})
        assert reveal['golden']
        assert get('/api/summary')['golden'] == 1
        assert post('/api/rescan', {'run': 'first'})['added'] == 0
    finally:
        server.shutdown()
        server.server_close()


def test_report_lists_tags_and_regressions(world):
    store, *_ = world
    api = make_api(store)
    for grade, tag in ((1, 'too dark'), (3, 'robotic')):
        t = api.next({'mode': 'grade'})
        api.answer({'token': t['token'], 'grade': grade, 'tags': [tag]})
    out = []
    E.report(store, out.append)
    text = '\n'.join(out)
    assert '2 graded fits' in text and 'too dark' in text and 'By category' in text
