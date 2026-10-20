"""
Blind A/B listening page for the sample fits, with a SQLite record of what you heard.

    venv/bin/python tools/eval_ab.py register <samples dir> <fit dir> --label NAME
    venv/bin/python tools/eval_ab.py serve [--port 8767]
    venv/bin/python tools/eval_ab.py compare [--from OLD] --to NEW
    venv/bin/python tools/eval_ab.py check [--threshold 0.75] [--queue]
    venv/bin/python tools/eval_ab.py ladder --only TEXT [--only TEXT ...] [--steps -3,-2,-1,0,1,2,3]
    venv/bin/python tools/eval_ab.py report
    venv/bin/python tools/eval_ab.py features

Everything lives in kits/eval (--root): evals.sqlite and the audio the verdicts
refer to.

register  snapshots a fit run: for each <fit dir>/<path>.mtdrum whose sample
          <samples dir>/<path>.* exists, it copies the patch and its app-rendered
          wav (the .fit.wav next to it, or a new render when that is missing or
          older than the patch, as with a fit still running) into
          versions/<hash>/, and the original (raw, and trimmed as the fitter
          sees it) into originals/. Run it again for the same label to add the
          fits finished since. Scores come from fit_report.csv or <fit dir>.log.
serve     the listening page. Mode 1 grades a fit against its original (Awful,
          Weak, Good, Perfect; Good and Perfect make the sample golden). Mode 2
          compares an old and a new version of a sample: you say which side
          is better (or Same, or Both bad) and the page turns it into Better or
          Worse for the new version. Which side is which is shuffled and hidden until you
          answer, and every sound is played at the same K-weighted loudness.
compare   queues mode 2 pairs for the fits of the NEW run: each against its
          sample's golden fit, else its fit in the OLD run, else (no --from) its
          latest fit registered before.
check     the regression guard after an engine change: re-renders every golden
          patch with the current engine and measures how far it moved from the
          frozen render. It prints how many are over each threshold; --queue
          sends those over --threshold to mode 2 (frozen vs new render).
ladder    renders the current fit of each sample matched by --only with its
          pitch moved by each of --steps semitones (the oscillator with its sweep,
          and a band-passed noise's band);
          the page's Pitch ladder mode plays them shuffled next to the original
          and you pick the closest.
features  measures every fit against its original again, after the measures change.
report    grades by machine and category, defect tags against the fitter score
          and measured features, regressions, and how consistent the ears are.

The golden fit of a sample is replayed from its history: a Good or Perfect
grade makes a fit golden when the sample has none, Better passes the golden on
to the new fit (a new fit better than a non-golden one is graded in mode 1 instead),
Both bad drops the golden, Same and Worse keep it.
"""

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import pathlib
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import eval_ab_audio as audio  # noqa: E402  (next to this script)

DEFAULT_ROOT = ROOT / 'kits' / 'eval'
PAGE = pathlib.Path(__file__).with_name('eval_ab_page.html')
AUDIO_SUFFIXES = ('.wav', '.aif', '.aiff', '.flac')
GRADES = {1: 'Awful', 2: 'Weak', 3: 'Good', 4: 'Perfect'}
GOLDEN_FROM = 3
OUTCOMES = ('better', 'same', 'worse', 'both_bad')
CHOICES = ('a', 'b', 'same', 'both_bad')   # what the listener answers in mode 2
TAGS = ['too dark', 'too bright', 'pitch low', 'pitch high', 'too short', 'too long', 'robotic',
        'weak attack', 'too noisy', 'no body', 'wrong tone', 'tonal artifact', 'tail too quiet']
REPEAT_RATE = 0.05          # share of the grade trials that silently repeat an earlier one
REPEAT_AFTER = 20           # graded fits before repeats start
DEFAULT_THRESHOLD = 0.75    # check: distance over which a re-render goes back to the ear
MAX_SECONDS = 2.5           # length the fitter matches (tools/fit_samples.py)

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(
    id INTEGER PRIMARY KEY, label TEXT UNIQUE, samples_dir TEXT, fit_dir TEXT, git TEXT, created REAL);
CREATE TABLE IF NOT EXISTS samples(
    id INTEGER PRIMARY KEY, set_name TEXT, rel TEXT, machine TEXT, category TEXT, orig_sha TEXT,
    raw_file TEXT, trim_file TEXT, raw_loud REAL, trim_loud REAL, UNIQUE(set_name, rel));
CREATE TABLE IF NOT EXISTS fits(
    id INTEGER PRIMARY KEY, sample_id INTEGER, hash TEXT, source TEXT, patch_file TEXT, wav_file TEXT,
    loud REAL, score REAL, start_score REAL, features TEXT, git TEXT, created REAL, UNIQUE(sample_id, hash));
CREATE TABLE IF NOT EXISTS run_fits(run_id INTEGER, fit_id INTEGER, PRIMARY KEY(run_id, fit_id));
CREATE TABLE IF NOT EXISTS grades(
    id INTEGER PRIMARY KEY, sample_id INTEGER, fit_id INTEGER, grade INTEGER, tags TEXT, note TEXT,
    orig_side TEXT, repeat_of INTEGER, ts REAL);
CREATE TABLE IF NOT EXISTS comparisons(
    id INTEGER PRIMARY KEY, sample_id INTEGER, old_fit INTEGER, new_fit INTEGER, reason TEXT,
    distance REAL, status TEXT DEFAULT 'pending', outcome TEXT, new_side TEXT, tags TEXT, note TEXT,
    ts REAL, UNIQUE(old_fit, new_fit));
CREATE TABLE IF NOT EXISTS golden(sample_id INTEGER PRIMARY KEY, fit_id INTEGER, grade INTEGER, since REAL);
CREATE TABLE IF NOT EXISTS ladders(
    id INTEGER PRIMARY KEY, sample_id INTEGER, base_fit INTEGER, label TEXT, status TEXT DEFAULT 'pending',
    pick_fit INTEGER, pick_shift REAL, tags TEXT, note TEXT, ts REAL);
CREATE TABLE IF NOT EXISTS ladder_items(ladder_id INTEGER, fit_id INTEGER, shift REAL, PRIMARY KEY(ladder_id, fit_id));
CREATE TABLE IF NOT EXISTS trials(
    token TEXT PRIMARY KEY, kind TEXT, sample_id INTEGER, fit_id INTEGER, comparison_id INTEGER,
    layout TEXT, repeat_of INTEGER, created REAL, answered INTEGER DEFAULT 0);
"""


def git_state() -> str:
    """Short commit of the repository, with '+' when files are modified."""
    try:
        head = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return head + ('+' if dirty else '')
    except (OSError, subprocess.CalledProcessError):
        return ''


def sha(*blobs: bytes) -> str:
    h = hashlib.sha1()
    for b in blobs:
        h.update(b)
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- store
class Store:
    """The SQLite file and the audio folders around it."""

    def __init__(self, root=DEFAULT_ROOT):
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.root / 'evals.sqlite'), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        if 'ladder_id' not in {r[1] for r in self.db.execute('PRAGMA table_info(trials)')}:
            self.db.execute('ALTER TABLE trials ADD COLUMN ladder_id INTEGER')
        self.lock = threading.RLock()

    def q(self, sql, *args):
        return self.db.execute(sql, args).fetchall()

    def one(self, sql, *args):
        return self.db.execute(sql, args).fetchone()

    def run(self, sql, *args) -> int:
        cur = self.db.execute(sql, args)
        self.db.commit()
        return cur.lastrowid

    def path(self, rel: str) -> pathlib.Path:
        return self.root / rel

    # -- golden
    def recompute_golden(self, sample_id: int) -> None:
        """Replay the sample's verdicts: see the module text for the rules."""
        events = [(r['ts'], r['id'], 'g', r) for r in self.q(
            'SELECT * FROM grades WHERE sample_id=? AND repeat_of IS NULL', sample_id)]
        events += [(r['ts'], r['id'], 'c', r) for r in self.q(
            "SELECT * FROM comparisons WHERE sample_id=? AND status='done'", sample_id)]
        golden = None   # (fit_id, grade, ts)
        for ts, _, kind, r in sorted(events, key=lambda e: e[:2]):
            if kind == 'g':
                if golden is None and r['grade'] >= GOLDEN_FROM:
                    golden = (r['fit_id'], r['grade'], ts)
            elif r['outcome'] == 'better':
                # better than a golden fit is golden; better than a weak one is graded on its own
                if golden is not None and golden[0] == r['old_fit']:
                    golden = (r['new_fit'], golden[1], ts)
            elif r['outcome'] == 'both_bad':
                golden = None
        self.db.execute('DELETE FROM golden WHERE sample_id=?', (sample_id,))
        if golden:
            self.db.execute('INSERT INTO golden VALUES(?,?,?,?)', (sample_id, *golden))
        self.db.commit()


# --------------------------------------------------------------------------- register
def read_scores(fit_dir: pathlib.Path) -> dict:
    """{stem: (score, start score)} from fit_report.csv, else from the fit log
    (<fit dir>.log, or fit.log inside it)."""
    scores = {}
    report = fit_dir / 'fit_report.csv'
    if report.is_file():
        with open(report, newline='', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                scores[pathlib.Path(r['sample']).stem] = (float(r['score']), float(r['start_score']))
    line = re.compile(r'^(.+?)\s+\S+\s+start\s+([\d.]+) -> fit\s+([\d.]+)')
    for log in (fit_dir.with_name(fit_dir.name + '.log'), fit_dir / 'fit.log'):
        if log.is_file():
            for text in log.read_text(encoding='utf-8', errors='replace').splitlines():
                m = line.match(text)
                if m:
                    scores.setdefault(m.group(1).strip(), (float(m.group(3)), float(m.group(2))))
    return scores


def find_original(samples_dir: pathlib.Path, rel_patch: pathlib.Path):
    for ext in AUDIO_SUFFIXES + tuple(e.upper() for e in AUDIO_SUFFIXES):
        p = samples_dir / rel_patch.with_suffix(ext)
        if p.is_file():
            return p
    return None


def fit_modules():
    import fit_samples
    return fit_samples


def render_patch(patch: pathlib.Path, wav: pathlib.Path) -> None:
    """The patch as the app plays it (the fitter's own preview writer)."""
    fit_modules().write_preview(patch, wav)


def add_sample(store: Store, set_name: str, rel: str, original: pathlib.Path, machine: str,
               category: str) -> int:
    row = store.one('SELECT id FROM samples WHERE set_name=? AND rel=?', set_name, rel)
    blob = original.read_bytes()
    digest = sha(blob)
    if row:
        return row['id']
    raw = f'originals/{digest}{original.suffix.lower()}'
    trim = f'originals/{digest}.trim.wav'
    store.path('originals').mkdir(exist_ok=True)
    if not store.path(raw).exists():
        store.path(raw).write_bytes(blob)
    x = fit_modules().load_target(str(original), MAX_SECONDS)
    sf.write(str(store.path(trim)), x, audio.SR, subtype='PCM_24')
    return store.run('INSERT INTO samples(set_name, rel, machine, category, orig_sha, raw_file, trim_file,'
                     ' raw_loud, trim_loud) VALUES(?,?,?,?,?,?,?,?,?)', set_name, rel, machine, category,
                     digest, raw, trim, audio.loudness_db(store.path(raw)), audio.loudness_db(store.path(trim)))


def add_fit(store: Store, sample_id: int, patch: pathlib.Path, wav: pathlib.Path, source: str,
            score=None, start_score=None) -> tuple:
    """Snapshot a patch and its render; returns (fit id, True when it is new)."""
    patch_blob, wav_blob = patch.read_bytes(), wav.read_bytes()
    digest = sha(patch_blob, wav_blob)
    row = store.one('SELECT id FROM fits WHERE sample_id=? AND hash=?', sample_id, digest)
    if row:
        return row['id'], False
    folder = store.path(f'versions/{digest}')
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'patch.mtdrum').write_bytes(patch_blob)
    (folder / 'fit.wav').write_bytes(wav_blob)
    sample = store.one('SELECT * FROM samples WHERE id=?', sample_id)
    ref, _ = audio.read_mono(store.path(sample['trim_file']))
    out, _ = audio.read_mono(folder / 'fit.wav')
    feats = audio.compare(ref, out)
    fit_id = store.run(
        'INSERT INTO fits(sample_id, hash, source, patch_file, wav_file, loud, score, start_score, features,'
        ' git, created) VALUES(?,?,?,?,?,?,?,?,?,?,?)', sample_id, digest, source,
        f'versions/{digest}/patch.mtdrum', f'versions/{digest}/fit.wav', audio.loudness_db(folder / 'fit.wav'),
        score, start_score, json.dumps(feats), git_state(), time.time())
    return fit_id, True


def refresh_features(store: Store, log=print) -> int:
    """Measure every fit against its original again (after the measures change)."""
    rows = store.q('SELECT f.id, f.wav_file, s.trim_file FROM fits f JOIN samples s ON s.id=f.sample_id')
    for r in rows:
        ref, _ = audio.read_mono(store.path(r['trim_file']))
        out, _ = audio.read_mono(store.path(r['wav_file']))
        store.db.execute('UPDATE fits SET features=? WHERE id=?', (json.dumps(audio.compare(ref, out)), r['id']))
    store.db.commit()
    log(f'{len(rows)} fits measured again')
    return len(rows)


def register(store: Store, samples_dir, fit_dir, label: str, set_name: str = '', only: str = '',
             log=print) -> dict:
    """Snapshot the fits under ``fit_dir`` as run ``label``; see the module text."""
    samples_dir, fit_dir = pathlib.Path(samples_dir).resolve(), pathlib.Path(fit_dir).resolve()
    set_name = set_name or samples_dir.name
    fs = fit_modules()
    run_row = store.one('SELECT id FROM runs WHERE label=?', label)
    run_id = run_row['id'] if run_row else store.run(
        'INSERT INTO runs(label, samples_dir, fit_dir, git, created) VALUES(?,?,?,?,?)',
        label, str(samples_dir), str(fit_dir), git_state(), time.time())
    scores = read_scores(fit_dir)
    added = known = missing = 0
    for patch in sorted(fit_dir.rglob('*.mtdrum')):
        rel = patch.relative_to(fit_dir)
        if only.lower() not in str(rel).lower():
            continue
        original = find_original(samples_dir, rel)
        if original is None:
            missing += 1
            continue
        wav = patch.with_suffix('.fit.wav')
        tmp = None
        if not wav.is_file() or wav.stat().st_mtime < patch.stat().st_mtime:
            tmp = pathlib.Path(tempfile.mkdtemp()) / 'fit.wav'
            render_patch(patch, tmp)
            wav = tmp
        try:
            cats = fs.categories_for(original, samples_dir)
            sample_id = add_sample(store, set_name, str(rel.with_suffix('')), original,
                                   rel.parts[0] if len(rel.parts) > 1 else '', cats[0] if cats else 'other')
            score = scores.get(patch.stem, (None, None))
            fit_id, new = add_fit(store, sample_id, patch, wav, 'register', *score)
            store.run('INSERT OR IGNORE INTO run_fits VALUES(?,?)', run_id, fit_id)
        finally:
            if tmp:
                shutil.rmtree(tmp.parent, ignore_errors=True)
        added += new
        known += not new
    log(f'run {label!r}: {added} new fits, {known} already known, {missing} patches without an original')
    return {'run_id': run_id, 'added': added, 'known': known, 'missing': missing}


# --------------------------------------------------------------------------- compare / check
def queue_comparison(store: Store, sample_id: int, old_fit: int, new_fit: int, reason: str,
                     distance=None) -> bool:
    cur = store.db.execute(
        'INSERT OR IGNORE INTO comparisons(sample_id, old_fit, new_fit, reason, distance) VALUES(?,?,?,?,?)',
        (sample_id, old_fit, new_fit, reason, distance))
    store.db.commit()
    return cur.rowcount > 0


def compare_runs(store: Store, old, new: str, log=print) -> int:
    """Queue old-vs-new pairs for the fits of run ``new``; returns how many are new. The
    old fit of a sample is its golden fit, else its fit in run ``old``, else (``old`` None)
    the latest fit registered before the new one."""
    def run_id(label):
        row = store.one('SELECT id FROM runs WHERE label=?', label)
        if row is None:
            raise SystemExit(f'no run called {label!r}')
        return row['id']
    new_id = run_id(new)
    old_id = run_id(old) if old else None
    queued = same = 0
    for r in store.q('SELECT f.id, f.sample_id FROM fits f JOIN run_fits rf ON rf.fit_id=f.id AND rf.run_id=?',
                     new_id):
        gold = store.one('SELECT fit_id FROM golden WHERE sample_id=?', r['sample_id'])
        if gold:
            old_fit = gold['fit_id']
        elif old_id is not None:
            row = store.one('SELECT f.id FROM fits f JOIN run_fits rf ON rf.fit_id=f.id AND rf.run_id=? '
                            'WHERE f.sample_id=? ORDER BY f.id DESC', old_id, r['sample_id'])
            old_fit = row['id'] if row else None
        else:
            row = store.one("SELECT id FROM fits WHERE sample_id=? AND source='register' AND id<? "
                            'ORDER BY id DESC', r['sample_id'], r['id'])
            old_fit = row['id'] if row else None
        if old_fit is None:
            continue
        if old_fit == r['id']:
            same += 1
            continue
        queued += queue_comparison(store, r['sample_id'], old_fit, r['id'], f'{old or "latest"} -> {new}')
    log(f'{queued} pairs queued, {same} samples unchanged')
    return queued


def current_fit(store: Store, sample_id: int):
    """The fit a sample stands on: its golden fit, else the winner of its latest
    comparison, else its latest registered fit."""
    gold = store.one('SELECT fit_id FROM golden WHERE sample_id=?', sample_id)
    if gold:
        return gold['fit_id']
    c = store.one("SELECT * FROM comparisons WHERE sample_id=? AND status='done' AND outcome!='both_bad' "
                  'ORDER BY ts DESC', sample_id)
    if c:
        return c['new_fit'] if c['outcome'] == 'better' else c['old_fit']
    row = store.one("SELECT id FROM fits WHERE sample_id=? AND source='register' ORDER BY id DESC", sample_id)
    return row['id'] if row else None


OSC_FREQ = re.compile(r'(OscFreq:\s*")([0-9.eE+-]+)(\s*Hz")')
NOISE_FREQ = re.compile(r'(NFilFrq:\s*")([0-9.eE+-]+)(\s*Hz")')


def shift_patch(text: str, semitones: float) -> str:
    """A patch with its pitch moved: the oscillator (its sweep is relative to it) and, when
    the noise goes through a band-pass, that band, which can hold a tom's note."""
    if not OSC_FREQ.search(text):
        raise ValueError('patch has no OscFreq')
    ratio = 2.0 ** (semitones / 12.0)
    move = lambda m: f'{m.group(1)}{float(m.group(2)) * ratio!r}{m.group(3)}'
    text = OSC_FREQ.sub(move, text, count=1)
    if re.search(r'NFilMod:\s*"BP"', text):
        text = NOISE_FREQ.sub(move, text, count=1)
    return text


def make_ladder(store: Store, sample_id: int, steps, label: str) -> int:
    """Queue a pitch ladder of the sample's current fit; returns its id."""
    base = current_fit(store, sample_id)
    fit = store.one('SELECT * FROM fits WHERE id=?', base)
    text = store.path(fit['patch_file']).read_text(encoding='utf-8')
    ladder = store.run('INSERT INTO ladders(sample_id, base_fit, label) VALUES(?,?,?)', sample_id, base, label)
    tmp = pathlib.Path(tempfile.mkdtemp())
    try:
        for st in steps:
            if st == 0:
                fit_id = base
            else:
                patch, wav = tmp / f'{st}.mtdrum', tmp / f'{st}.wav'
                patch.write_text(shift_patch(text, st), encoding='utf-8')
                render_patch(patch, wav)
                fit_id, _ = add_fit(store, sample_id, patch, wav, 'ladder')
            store.run('INSERT OR IGNORE INTO ladder_items VALUES(?,?,?)', ladder, fit_id, float(st))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return ladder


def _rerender(args) -> tuple:
    fit_id, patch, wav = args
    render_patch(pathlib.Path(patch), pathlib.Path(wav))
    return fit_id, wav


CHECK_STEPS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def check(store: Store, threshold: float, queue: bool, label: str = '', jobs: int = 0, log=print) -> list:
    """Re-render the golden patches; returns [(sample id, golden fit id, new wav, measures)]."""
    gold = store.q('SELECT g.sample_id, g.fit_id FROM golden g')
    if not gold:
        log('no golden sample yet')
        return []
    tmp = pathlib.Path(tempfile.mkdtemp())
    try:
        jobs_list = []
        for g in gold:
            fit = store.one('SELECT * FROM fits WHERE id=?', g['fit_id'])
            jobs_list.append((g['fit_id'], str(store.path(fit['patch_file'])), str(tmp / f'{g["fit_id"]}.wav')))
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs or os.cpu_count() or 1) as pool:
            done = list(pool.map(_rerender, jobs_list))
        rows = []
        for (fit_id, wav), g in zip(done, gold):
            fit = store.one('SELECT * FROM fits WHERE id=?', fit_id)
            old, _ = audio.read_mono(store.path(fit['wav_file']))
            new, _ = audio.read_mono(wav)
            rows.append((g['sample_id'], fit_id, wav, audio.compare(old, new)))
        log(f'{len(rows)} golden samples re-rendered; distance to the frozen render:')
        for step in CHECK_STEPS:
            log(f'  over {step:4.2f}: {sum(r[3]["distance"] > step for r in rows)}')
        over = [r for r in rows if r[3]['distance'] > threshold]
        log(f'{len(over)} over the threshold {threshold}' + ('' if queue else ' (--queue sends them to mode 2)'))
        for r in sorted(over, key=lambda r: -r[3]['distance'])[:15]:
            s = store.one('SELECT * FROM samples WHERE id=?', r[0])
            log(f'  {r[3]["distance"]:6.2f}  mel {r[3]["mel"]:.2f} env {r[3]["env"]:.2f}  {s["machine"]}/{s["rel"]}')
        if queue:
            reason = f'check {label or git_state()}'
            for sample_id, fit_id, wav, m in over:
                fit = store.one('SELECT * FROM fits WHERE id=?', fit_id)
                new_id, _ = add_fit(store, sample_id, store.path(fit['patch_file']), pathlib.Path(wav), 'check')
                if new_id != fit_id:
                    queue_comparison(store, sample_id, fit_id, new_id, reason, m['distance'])
        return rows
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- api
def gains_for(*loud) -> list:
    """Gains in dB bringing the sounds to the quietest one's loudness."""
    base = min(loud)
    return [round(base - v, 2) for v in loud]


class Api:
    """What the page asks of the store; every method returns a JSON-able value."""

    def __init__(self, store: Store, rng=None):
        self.store = store
        self.rng = rng or random.Random()
        self.skipped = set()

    # -- state
    def summary(self) -> dict:
        s = self.store
        with s.lock:
            n = lambda sql, *a: s.one(sql, *a)[0]
            dist = {r['grade']: r['n'] for r in s.q(
                'SELECT grade, count(*) n FROM grades WHERE repeat_of IS NULL GROUP BY grade')}
            return {
                'samples': n('SELECT count(*) FROM samples'), 'fits': n('SELECT count(*) FROM fits'),
                'graded': n('SELECT count(*) FROM grades WHERE repeat_of IS NULL'),
                'golden': n('SELECT count(*) FROM golden'),
                'pending_compare': n("SELECT count(*) FROM comparisons WHERE status='pending'"),
                'pending_ladder': n("SELECT count(*) FROM ladders WHERE status='pending'"),
                'todo_grade': len(self._grade_candidates({})),
                'grade_counts': {GRADES[g]: dist.get(g, 0) for g in GRADES},
                'sets': [r[0] for r in s.q('SELECT DISTINCT set_name FROM samples ORDER BY 1')],
                'machines': [r[0] for r in s.q('SELECT DISTINCT machine FROM samples ORDER BY 1')],
                'categories': [r[0] for r in s.q('SELECT DISTINCT category FROM samples ORDER BY 1')],
                'runs': [r[0] for r in s.q('SELECT label FROM runs ORDER BY id')],
                'tags': TAGS + sorted(self._custom_tags()),
                'grades': GRADES, 'outcomes': list(OUTCOMES),
                'default_mode': ('compare' if n("SELECT count(*) FROM comparisons WHERE status='pending'") else
                                 'ladder' if n("SELECT count(*) FROM ladders WHERE status='pending'") else 'grade'),
            }

    def _custom_tags(self) -> set:
        used = set()
        for r in self.store.q('SELECT tags FROM grades UNION ALL SELECT tags FROM comparisons'):
            used.update(json.loads(r[0] or '[]'))
        return used - set(TAGS)

    # -- trials
    def _filters(self, f: dict) -> tuple:
        sql, args = '', []
        for key, col in (('set', 's.set_name'), ('machine', 's.machine'), ('category', 's.category')):
            if f.get(key):
                sql += f' AND {col}=?'
                args.append(f[key])
        if f.get('run'):
            sql += ' AND EXISTS (SELECT 1 FROM run_fits rf JOIN runs r ON r.id=rf.run_id WHERE rf.fit_id=f.id AND r.label=?)'
            args.append(f['run'])
        return sql, args

    def _grade_candidates(self, f: dict) -> list:
        sql, args = self._filters(f)
        return self.store.q(
            "SELECT f.id AS fit_id, f.score, s.id AS sample_id, s.set_name, s.machine, s.rel FROM samples s "
            "JOIN fits f ON f.id=(SELECT max(id) FROM fits WHERE sample_id=s.id AND source='register') "
            'WHERE NOT EXISTS (SELECT 1 FROM golden g WHERE g.sample_id=s.id) '
            'AND NOT EXISTS (SELECT 1 FROM grades gr WHERE gr.fit_id=f.id AND gr.repeat_of IS NULL)' + sql, *args)

    def _layout(self, first: str, second: str) -> dict:
        return {'a': first, 'b': second} if self.rng.random() < 0.5 else {'a': second, 'b': first}

    @staticmethod
    def _heading(sample) -> dict:
        return {'machine': sample['machine'], 'category': sample['category'], 'name': pathlib.Path(sample['rel']).name}

    def _new_trial(self, kind, sample_id, layout, fit_id=None, comparison_id=None, repeat_of=None,
                   ladder_id=None) -> str:
        s = self.store
        s.db.execute('DELETE FROM trials WHERE answered=0')
        token = uuid.uuid4().hex
        s.run('INSERT INTO trials(token, kind, sample_id, fit_id, comparison_id, layout, repeat_of, created, ladder_id)'
              ' VALUES(?,?,?,?,?,?,?,?,?)', token, kind, sample_id, fit_id, comparison_id, json.dumps(layout),
              repeat_of, time.time(), ladder_id)
        return token

    def next(self, f: dict):
        """The next trial of mode ``f['mode']`` (grade, compare or ladder), or None."""
        with self.store.lock:
            if f.get('mode') == 'compare':
                return self._next_compare(f)
            if f.get('mode') == 'ladder':
                return self._next_ladder(f)
            return self._next_grade(f)

    def _next_grade(self, f: dict):
        s = self.store
        graded = s.q('SELECT * FROM grades WHERE repeat_of IS NULL')
        if len(graded) >= REPEAT_AFTER and self.rng.random() < REPEAT_RATE:
            again = {r['repeat_of'] for r in s.q('SELECT repeat_of FROM grades WHERE repeat_of IS NOT NULL')}
            pool = [g for g in graded if g['id'] not in again]
            if pool:
                g = self.rng.choice(pool)
                return self.grade_trial(g['fit_id'], repeat_of=g['id'])
        rows = [r for r in self._grade_candidates(f) if ('grade', r['fit_id']) not in self.skipped]
        if not rows:
            self.skipped.clear()
            rows = list(self._grade_candidates(f))
        if not rows:
            return None
        order = f.get('order', 'score')
        if order == 'random':
            row = self.rng.choice(rows)
        elif order == 'folder':
            row = min(rows, key=lambda r: (r['set_name'], r['machine'], r['rel']))
        else:   # worst score first, unscored last
            row = max(rows, key=lambda r: (r['score'] is not None, r['score'] or 0.0))
        return self.grade_trial(row['fit_id'], remaining=len(rows))

    def grade_trial(self, fit_id: int, repeat_of=None, remaining=None) -> dict:
        s = self.store
        fit = s.one('SELECT * FROM fits WHERE id=?', fit_id)
        sample = s.one('SELECT * FROM samples WHERE id=?', fit['sample_id'])
        layout = self._layout('orig', 'fit')
        token = self._new_trial('grade', sample['id'], layout, fit_id=fit_id, repeat_of=repeat_of)
        gains = dict(zip(('orig', 'fit', 'raw'), gains_for(sample['trim_loud'], fit['loud'], sample['raw_loud'])))
        return {'token': token, 'kind': 'grade', 'remaining': remaining, **self._heading(sample),
                'gain': {'a': gains[layout['a']], 'b': gains[layout['b']], 'ref': gains['raw']},
                'ref_label': 'untrimmed original'}

    def _next_compare(self, f: dict):
        sql, args = self._filters(f)
        rows = self.store.q(
            "SELECT c.id, c.distance FROM comparisons c JOIN samples s ON s.id=c.sample_id "
            "JOIN fits f ON f.id=c.new_fit WHERE c.status='pending'" + sql
            + ' ORDER BY (c.distance IS NULL), c.distance DESC, c.id', *args)
        rows = [r for r in rows if ('compare', r['id']) not in self.skipped] or rows
        return self.compare_trial(rows[0]['id'], remaining=len(rows)) if rows else None

    def compare_trial(self, comparison_id: int, remaining=None) -> dict:
        s = self.store
        c = s.one('SELECT * FROM comparisons WHERE id=?', comparison_id)
        old, new = (s.one('SELECT * FROM fits WHERE id=?', c[k]) for k in ('old_fit', 'new_fit'))
        sample = s.one('SELECT * FROM samples WHERE id=?', c['sample_id'])
        layout = self._layout('old', 'new')
        token = self._new_trial('compare', sample['id'], layout, comparison_id=comparison_id)
        gains = dict(zip(('old', 'new', 'orig'), gains_for(old['loud'], new['loud'], sample['trim_loud'])))
        return {'token': token, 'kind': 'compare', 'remaining': remaining, **self._heading(sample),
                'reason': c['reason'], 'distance': c['distance'],
                'gain': {'a': gains[layout['a']], 'b': gains[layout['b']], 'ref': gains['orig']},
                'ref_label': 'original'}

    def _next_ladder(self, f: dict):
        sql, args = self._filters(f)
        rows = self.store.q("SELECT l.id FROM ladders l JOIN samples s ON s.id=l.sample_id "
                            "JOIN fits f ON f.id=l.base_fit WHERE l.status='pending'" + sql + ' ORDER BY l.id', *args)
        rows = [r for r in rows if ('ladder', r['id']) not in self.skipped] or rows
        return self.ladder_trial(rows[0]['id'], remaining=len(rows)) if rows else None

    def ladder_trial(self, ladder_id: int, remaining=None) -> dict:
        s = self.store
        ladder = s.one('SELECT * FROM ladders WHERE id=?', ladder_id)
        sample = s.one('SELECT * FROM samples WHERE id=?', ladder['sample_id'])
        items = [r['fit_id'] for r in s.q('SELECT fit_id FROM ladder_items WHERE ladder_id=?', ladder_id)]
        self.rng.shuffle(items)
        layout = {str(i + 1): fit_id for i, fit_id in enumerate(items)}
        token = self._new_trial('ladder', sample['id'], layout, ladder_id=ladder_id)
        louds = [s.one('SELECT loud FROM fits WHERE id=?', fit_id)[0] for fit_id in items]
        gains = gains_for(sample['trim_loud'], *louds)
        return {'token': token, 'kind': 'ladder', 'remaining': remaining, **self._heading(sample),
                'slots': len(items), 'gain': {'ref': gains[0], **{str(i + 1): g for i, g in enumerate(gains[1:])}},
                'ref_label': 'original'}

    def media(self, token: str, which: str):
        """(file, content type) of the sound behind a slot of a trial."""
        s = self.store
        with s.lock:
            t = s.one('SELECT * FROM trials WHERE token=?', token)
            if t is None:
                return None
            layout = json.loads(t['layout'])
            sample = s.one('SELECT * FROM samples WHERE id=?', t['sample_id'])
            if t['kind'] == 'ladder':
                if which in ('ref', 'raw'):
                    path = s.path(sample['trim_file' if which == 'ref' else 'raw_file'])
                elif which in layout:
                    path = s.path(s.one('SELECT wav_file FROM fits WHERE id=?', layout[which])[0])
                else:
                    return None
                return path, 'audio/' + ('wav' if path.suffix == '.wav' else path.suffix.lstrip('.'))
            if t['kind'] == 'grade':
                fit = s.one('SELECT * FROM fits WHERE id=?', t['fit_id'])
                files = {'orig': sample['trim_file'], 'fit': fit['wav_file'], 'raw': sample['raw_file']}
                kinds = {**layout, 'ref': 'raw'}
            else:
                c = s.one('SELECT * FROM comparisons WHERE id=?', t['comparison_id'])
                files = {'old': s.one('SELECT wav_file FROM fits WHERE id=?', c['old_fit'])[0],
                         'new': s.one('SELECT wav_file FROM fits WHERE id=?', c['new_fit'])[0],
                         'orig': sample['trim_file']}
                kinds = {**layout, 'ref': 'orig'}
            if which == 'raw' and t['kind'] == 'compare':
                return s.path(sample['raw_file']), 'audio/wav'
            rel = files.get(kinds.get(which, ''))
            if rel is None:
                return None
            path = s.path(rel)
            return path, 'audio/' + ('wav' if path.suffix == '.wav' else path.suffix.lstrip('.'))

    # -- answers
    def answer(self, body: dict) -> dict:
        s = self.store
        with s.lock:
            t = s.one('SELECT * FROM trials WHERE token=?', body.get('token', ''))
            if t is None or t['answered']:
                raise ValueError('unknown or already answered trial')
            layout = json.loads(t['layout'])
            tags = json.dumps([str(x) for x in body.get('tags', [])])
            note = str(body.get('note', ''))
            now = time.time()
            sample = s.one('SELECT * FROM samples WHERE id=?', t['sample_id'])
            if t['kind'] == 'grade':
                grade = int(body['grade'])
                if grade not in GRADES:
                    raise ValueError('grade must be 1-4')
                side = 'a' if layout['a'] == 'orig' else 'b'
                gid = s.run('INSERT INTO grades(sample_id, fit_id, grade, tags, note, orig_side, repeat_of, ts)'
                            ' VALUES(?,?,?,?,?,?,?,?)', t['sample_id'], t['fit_id'], grade, tags, note,
                            side, t['repeat_of'], now)
                s.recompute_golden(t['sample_id'])
                fit = s.one('SELECT * FROM fits WHERE id=?', t['fit_id'])
                reveal = {'orig_side': side.upper(), 'machine': sample['machine'], 'name': pathlib.Path(sample['rel']).name,
                          'score': fit['score'], 'grade': GRADES[grade], 'record': gid,
                          'features': json.loads(fit['features'] or '{}'),
                          'golden': bool(s.one('SELECT 1 FROM golden WHERE sample_id=?', t['sample_id'])),
                          'repeat': t['repeat_of'] is not None}
            elif t['kind'] == 'ladder':
                pick = str(body.get('pick', ''))
                if pick != 'none' and pick not in layout:
                    raise ValueError('pick must be a slot number or none')
                fit_id = layout[pick] if pick != 'none' else None
                shift = s.one('SELECT shift FROM ladder_items WHERE ladder_id=? AND fit_id=?',
                              t['ladder_id'], fit_id)[0] if fit_id else None
                s.run("UPDATE ladders SET status='done', pick_fit=?, pick_shift=?, tags=?, note=?, ts=? WHERE id=?",
                      fit_id, shift, tags, note, now, t['ladder_id'])
                reveal = {'machine': sample['machine'], 'name': pathlib.Path(sample['rel']).name, 'shift': shift,
                          'record': t['ladder_id']}
            else:
                # The listener only knows the sides: the side they prefer is mapped to
                # the new or the old version here
                choice = body.get('choice')
                if choice not in CHOICES:
                    raise ValueError(f'choice must be one of {CHOICES}')
                side = 'a' if layout['a'] == 'new' else 'b'
                outcome = choice if choice in ('same', 'both_bad') else 'better' if choice == side else 'worse'
                s.run("UPDATE comparisons SET status='done', outcome=?, new_side=?, tags=?, note=?, ts=? WHERE id=?",
                      outcome, side, tags, note, now, t['comparison_id'])
                s.recompute_golden(t['sample_id'])
                c = s.one('SELECT * FROM comparisons WHERE id=?', t['comparison_id'])
                reveal = {'new_side': side.upper(), 'machine': sample['machine'], 'name': pathlib.Path(sample['rel']).name,
                          'choice': choice, 'outcome': outcome, 'reason': c['reason'], 'distance': c['distance'], 'record': c['id'],
                          'golden': bool(s.one('SELECT 1 FROM golden WHERE sample_id=?', t['sample_id']))}
            s.run('UPDATE trials SET answered=1 WHERE token=?', t['token'])
            return reveal

    def skip(self, token: str) -> None:
        with self.store.lock:
            t = self.store.one('SELECT * FROM trials WHERE token=?', token)
            if t:
                self.skipped.add(('grade', t['fit_id']) if t['kind'] == 'grade' else
                                 ('ladder', t['ladder_id']) if t['kind'] == 'ladder' else ('compare', t['comparison_id']))

    def undo(self):
        """Take back the latest verdict and return a new trial for the same item."""
        s = self.store
        with s.lock:
            g = s.one('SELECT id, ts, sample_id, fit_id, repeat_of FROM grades ORDER BY ts DESC LIMIT 1')
            c = s.one("SELECT id, ts, sample_id FROM comparisons WHERE status='done' ORDER BY ts DESC LIMIT 1")
            lad = s.one("SELECT id, ts FROM ladders WHERE status='done' ORDER BY ts DESC LIMIT 1")
            latest = max((x for x in (g, c, lad) if x is not None), key=lambda x: x['ts'], default=None)
            if latest is None:
                return None
            if latest is lad:
                s.run("UPDATE ladders SET status='pending', pick_fit=NULL, pick_shift=NULL, tags=NULL, note=NULL,"
                      ' ts=NULL WHERE id=?', lad['id'])
                return self.ladder_trial(lad['id'])
            if latest is g:
                s.run('DELETE FROM grades WHERE id=?', g['id'])
                s.recompute_golden(g['sample_id'])
                return self.grade_trial(g['fit_id'], repeat_of=g['repeat_of'])
            s.run("UPDATE comparisons SET status='pending', outcome=NULL, new_side=NULL, tags=NULL, note=NULL,"
                  ' ts=NULL WHERE id=?', c['id'])
            s.recompute_golden(c['sample_id'])
            return self.compare_trial(c['id'])


# --------------------------------------------------------------------------- report
def mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else None


def fmt(x, nd=2):
    return '-' if x is None else f'{x:.{nd}f}'


FEATURES = ('mel', 'env', 'pitch_st', 'bright_oct', 'decay_oct', 'end_oct', 'click_db', 'level_db')


def report(store: Store, out=print) -> None:
    grades = store.q('SELECT g.*, f.score, f.features, s.machine, s.category, s.rel FROM grades g '
                     'JOIN fits f ON f.id=g.fit_id JOIN samples s ON s.id=g.sample_id WHERE g.repeat_of IS NULL')
    out(f'{len(grades)} graded fits, {store.one("SELECT count(*) FROM golden")[0]} golden, '
        f'{store.one("SELECT count(*) FROM fits")[0]} fits registered, '
        f'{store.one("SELECT count(*) FROM comparisons WHERE status=?", "pending")[0]} comparisons pending')
    if not grades:
        report_versions(store, out)
        return

    def table(title, key):
        out(f'\n{title}: n, mean grade, share Good+ , mean fitter score')
        groups = {}
        for g in grades:
            groups.setdefault(g[key], []).append(g)
        for k, rows in sorted(groups.items(), key=lambda kv: mean([r['grade'] for r in kv[1]])):
            out(f'  {k:28s} {len(rows):4d}  {mean([r["grade"] for r in rows]):.2f}  '
                f'{sum(r["grade"] >= GOLDEN_FROM for r in rows) / len(rows):4.0%}  {fmt(mean([r["score"] for r in rows]))}')

    table('By category', 'category')
    table('By machine', 'machine')

    out('\nDefect tags: n, mean grade, mean fitter score (all graded: '
        f'{fmt(mean([g["score"] for g in grades]))}), mean measures of the fits tagged')
    by_tag = {}
    for g in grades:
        for t in json.loads(g['tags'] or '[]'):
            by_tag.setdefault(t, []).append(g)
    out('  ' + ' ' * 22 + ' n  grade  score  ' + '  '.join(f'{k[:9]:>9s}' for k in FEATURES))
    for t, rows in sorted(by_tag.items(), key=lambda kv: -len(kv[1])):
        feats = [json.loads(r['features'] or '{}') for r in rows]
        out(f'  {t:22s} {len(rows):3d}  {mean([r["grade"] for r in rows]):.2f}  {fmt(mean([r["score"] for r in rows])):>5s}  '
            + '  '.join(f'{fmt(mean([f.get(k) for f in feats])):>9s}' for k in FEATURES))
    out('  (measures are fit minus original: pitch in semitones, brightness and decay in octaves, '
        'click and level in dB; end_oct is how much longer the fit stays within 50 dB of its peak)')

    scored = [(g['score'], g['grade']) for g in grades if g['score'] is not None]
    feats = [json.loads(g['features'] or '{}') for g in grades]
    long = [g for g, f in zip(grades, feats) if (f.get('end_oct') or 0.0) > 0.5]
    if long:
        out(f'\nFits outlasting their original by over 40 % at -50 dB: {len(long)}, mean grade '
            f'{mean([g["grade"] for g in long]):.2f} (others {mean([g["grade"] for g, f in zip(grades, feats) if (f.get("end_oct") or 0.0) <= 0.5]):.2f})')

    if len(scored) > 2 and np.std([a for a, _ in scored]) > 0 and np.std([b for _, b in scored]) > 0:
        out(f'\nCorrelation of the fitter score with your grade: {np.corrcoef(*zip(*scored))[0, 1]:+.2f} '
            f'(-1 means the lower score is the better ear grade, as it should)')

    out('\nYour consistency (silent repeats):')
    reps = store.q('SELECT r.grade AS again, o.grade AS first FROM grades r JOIN grades o ON o.id=r.repeat_of')
    if reps:
        out(f'  {len(reps)} repeats: same grade {sum(r["again"] == r["first"] for r in reps) / len(reps):.0%}, '
            f'within one {sum(abs(r["again"] - r["first"]) <= 1 for r in reps) / len(reps):.0%}')
    else:
        out('  no repeat yet')
    both = store.q('SELECT g.grade, g.orig_side FROM grades g WHERE g.repeat_of IS NULL')
    for side in ('a', 'b'):
        v = [r['grade'] for r in both if r['orig_side'] == side]
        out(f'  mean grade when the original was {side.upper()}: {fmt(mean(v))} (n={len(v)})')

    out('\nGolden samples by category:')
    for r in store.q('SELECT s.category, count(*) n FROM golden g JOIN samples s ON s.id=g.sample_id '
                     'GROUP BY 1 ORDER BY 2 DESC'):
        out(f'  {r["category"]:10s} {r["n"]}')

    report_versions(store, out)


def report_versions(store: Store, out=print) -> None:
    """The pitch ladders and the version comparisons."""
    ladders = store.q("SELECT l.*, s.machine, s.rel FROM ladders l JOIN samples s ON s.id=l.sample_id "
                      "WHERE l.status='done' ORDER BY l.id")
    if ladders:
        out(f'\nPitch ladders: {len(ladders)} answered (semitones you moved the fit to match the original)')
        for r in ladders:
            pick = 'none close' if r['pick_shift'] is None else f'{r["pick_shift"]:+g}'
            out(f'  {pick:>10s}  {r["machine"]}/{pathlib.Path(r["rel"]).name}' + (f'  note: {r["note"]}' if r['note'] else ''))
        picks = [r['pick_shift'] for r in ladders if r['pick_shift'] is not None]
        if picks:
            out(f'  median {float(np.median(picks)):+g}, mean {mean(picks):+.2f} semitones over {len(picks)} picks')

    comps = store.q("SELECT c.*, s.machine, s.rel FROM comparisons c JOIN samples s ON s.id=c.sample_id "
                    "WHERE c.status='done'")
    out(f'\nVersion comparisons: {len(comps)} answered')
    for o in OUTCOMES:
        out(f'  {o:9s} {sum(c["outcome"] == o for c in comps)}')
    for c in comps:
        if c['outcome'] == 'worse':
            out(f'  REGRESSION {c["machine"]}/{pathlib.Path(c["rel"]).name} ({c["reason"]}) tags: {", ".join(json.loads(c["tags"] or "[]"))}'
                + (f' note: {c["note"]}' if c['note'] else ''))
    pref = [c for c in comps if c['outcome'] in ('better', 'worse')]
    if pref:
        out(f'  new version on side A in {sum(c["new_side"] == "a" for c in comps) / len(comps):.0%} of trials '
            f'(position bias check, expect near 50%)')


# --------------------------------------------------------------------------- http
class Handler(BaseHTTPRequestHandler):
    api: Api = None

    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body: bytes, ctype='application/json'):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode())

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path in ('/', '/index.html'):
                self._send(200, PAGE.read_bytes(), 'text/html; charset=utf-8')
            elif url.path == '/api/summary':
                self._json(self.api.summary())
            elif url.path == '/api/next':
                trial = self.api.next(q)
                self._json(trial if trial else {'empty': True})
            elif url.path.startswith('/media/'):
                _, _, token, which = url.path.split('/')
                found = self.api.media(token, which)
                if found is None:
                    return self._json({'error': 'not found'}, 404)
                self._send(200, found[0].read_bytes(), found[1])
            else:
                self._json({'error': 'not found'}, 404)
        except Exception as exc:    # the page shows the message
            self._json({'error': repr(exc)}, 500)

    def do_POST(self):
        url = urlparse(self.path)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get('Content-Length') or 0)) or b'{}')
            if url.path == '/api/answer':
                self._json(self.api.answer(body))
            elif url.path == '/api/skip':
                self.api.skip(body.get('token', ''))
                self._json({})
            elif url.path == '/api/undo':
                trial = self.api.undo()
                self._json(trial if trial else {'empty': True})
            elif url.path == '/api/rescan':
                with self.api.store.lock:
                    run = self.api.store.one('SELECT * FROM runs WHERE label=?', body.get('run', ''))
                    if run is None:
                        return self._json({'error': 'unknown run'}, 404)
                    self._json(register(self.api.store, run['samples_dir'], run['fit_dir'], run['label'],
                                        log=lambda *_: None))
            else:
                self._json({'error': 'not found'}, 404)
        except (ValueError, KeyError) as exc:
            self._json({'error': str(exc)}, 400)
        except Exception as exc:
            self._json({'error': repr(exc)}, 500)


def make_server(store: Store, host='127.0.0.1', port=8767, rng=None) -> ThreadingHTTPServer:
    handler = type('BoundHandler', (Handler,), {'api': Api(store, rng)})
    return ThreadingHTTPServer((host, port), handler)


# --------------------------------------------------------------------------- cli
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip())
    ap.add_argument('--root', default=str(DEFAULT_ROOT), help='folder of the database and the audio')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('register', help='snapshot a fit run')
    p.add_argument('samples')
    p.add_argument('fits')
    p.add_argument('--label', required=True)
    p.add_argument('--set', default='', help='name of the sample set (default: the samples folder name)')
    p.add_argument('--only', default='', help='only the patches whose path contains this text')
    p = sub.add_parser('serve', help='the listening page')
    p.add_argument('--port', type=int, default=8767)
    p.add_argument('--host', default='127.0.0.1')
    p = sub.add_parser('compare', help='queue old-vs-new pairs between two runs')
    p.add_argument('--from', dest='old', default='',
                   help='old run (default: each sample\'s golden, else its latest earlier fit)')
    p.add_argument('--to', dest='new', required=True)
    p = sub.add_parser('check', help='re-render the golden patches with the current engine')
    p.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD)
    p.add_argument('--queue', action='store_true')
    p.add_argument('--label', default='')
    p.add_argument('--jobs', type=int, default=0)
    sub.add_parser('report', help='what the verdicts say')
    sub.add_parser('features', help='measure every fit against its original again')
    p = sub.add_parser('ladder', help='queue pitch ladders of the current fits')
    p.add_argument('--only', action='append', required=True, help='samples whose path contains this text')
    p.add_argument('--steps', default='-3,-2,-1,0,1,2,3', help='semitones, comma separated')
    p.add_argument('--label', default='')
    args = ap.parse_args(argv)

    store = Store(args.root)
    if args.cmd == 'register':
        register(store, args.samples, args.fits, args.label, args.set, args.only)
    elif args.cmd == 'serve':
        server = make_server(store, args.host, args.port)
        print(f'listening on http://{args.host}:{args.port}/  (database {store.root / "evals.sqlite"})', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    elif args.cmd == 'compare':
        compare_runs(store, args.old, args.new)
    elif args.cmd == 'features':
        refresh_features(store)
    elif args.cmd == 'ladder':
        steps = [float(v) for v in args.steps.split(',')]
        for text in args.only:
            rows = store.q('SELECT id, rel FROM samples WHERE rel LIKE ?', f'%{text}%')
            if len(rows) != 1:
                print(f'{text!r} matches {len(rows)} samples: skipped')
                continue
            make_ladder(store, rows[0]['id'], steps, args.label or 'ladder')
            print(f'ladder queued for {rows[0]["rel"]}')
    elif args.cmd == 'check':
        check(store, args.threshold, args.queue, args.label, args.jobs)
    else:
        report(store)


if __name__ == '__main__':
    main()
