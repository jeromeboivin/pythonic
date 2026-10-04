"""Measure the audio callback's timing margin under main-thread and worker-thread load.

Runs the real engine in a real sounddevice OutputStream (output is silenced) while
the main thread runs a UI-like tick of a given cost and/or a worker thread runs a
heavy action. Prints one JSON line per scenario.

Usage (from the repo root):
    venv/bin/python bench_frame_budget.py --seconds 8 > results.jsonl
"""
import argparse
import json
import os
import statistics
import sys
import threading
import time

import numpy as np
import sounddevice as sd

REPO = os.getcwd()
sys.path.insert(0, REPO)

from pythonic.synthesizer import PythonicSynthesizer  # noqa: E402
from pythonic.pattern_manager import PatternManager  # noqa: E402
from pythonic.preset_manager import PresetManager  # noqa: E402
from pythonic.sequencer import StepSequencer  # noqa: E402

SR = 44100
DEVICE = os.environ.get('BENCH_DEVICE') or None
PRESET = os.path.join(REPO, 'tests', '909.mtpreset')


def build_engine(parallel):
    synth = PythonicSynthesizer(SR, parallel_channel_processing=parallel)
    pm = PatternManager(num_channels=8, pattern_length=16)
    data = PresetManager(synth).load_mtpreset(PRESET)
    synth.load_preset_data(data)
    if 'patterns' in data:
        pm.load_from_preset_data(data['patterns'])
    # Worst case: every channel on every 16th, accents on the beats, a fill on step 16.
    pat = pm.patterns[0]
    for ch in pat.channels:
        for i, step in enumerate(ch.steps):
            step.trigger = True
            step.accent = i % 4 == 0
            step.fill = i == 15
    pm.set_bpm(140)
    synth.set_bpm(140)
    seq = StepSequencer(pm, SR)
    seq.start(0, synth_clock=synth.sample_clock)
    return synth, pm, seq


def render_cost(synth, seq, frames, blocks=200):
    """Callback work alone, no contention: per-block render time in ms."""
    ts = []
    for _ in range(blocks):
        t0 = time.perf_counter()
        synth.process_audio_events(frames, seq.advance(frames))
        ts.append((time.perf_counter() - t0) * 1000)
    return ts


# ---------------------------------------------------------------- main-thread loads
def busy_python(ms):
    """Pure-Python work holding the GIL for about `ms` (dict/str churn, like building a frame)."""
    end = time.perf_counter() + ms / 1000.0
    d = {}
    i = 0
    while time.perf_counter() < end:
        d[f'ch{i % 8}.osc.decay'] = i * 0.5
        i += 1
    return i


def realistic_frame(n_changes=120):
    """What one poll + bridge push might serialise: changed addresses, transport, meters, 8x64 steps."""
    frame = {
        'v': 1234,
        'changes': {f'ch{i % 8}.p{i}': i * 0.123 for i in range(n_changes)},
        'transport': {'playing': True, 'step': 5, 'page': 0, 'pattern': 'A', 'queued': None},
        'mod': {f'ch{c}.lfo1': 0.1 * c for c in range(8)},
        'meters': [0.5] * 8,
        'steps': [[[1, 0, 64, 0, 100, ''] for _ in range(64)] for _ in range(8)],
    }
    return json.dumps(frame)


# ---------------------------------------------------------------- worker loads
def worker_preset_load(stop):
    n = 0
    pm_ = None
    while not stop.is_set():
        s = PythonicSynthesizer(SR)
        data = PresetManager(s).load_mtpreset(PRESET)
        s.load_preset_data(data)
        pm_ = PatternManager(8, 16)
        pm_.load_from_preset_data(data['patterns'])
        n += 1
    return n


def worker_wav_render(stop):
    s, _, q = build_engine(False)
    n = 0
    while not stop.is_set():
        for _ in range(int(SR * 2 / 1024)):  # 2 s of audio in 1024-frame blocks
            s.process_audio_events(1024, q.advance(1024))
            if stop.is_set():
                break
        n += 1
    return n


def worker_ai_load(stop):
    from pythonic.drum_generator import PatchGenerator as DrumGenerator
    from pythonic.pattern_generator import PatternGenerator
    n = 0
    while not stop.is_set():
        DrumGenerator().load_model(os.path.join(REPO, 'drum_cvae_best.pt'))
        PatternGenerator().load_model(os.path.join(REPO, 'drum_patterns', 'pattern_cvae_best.pt'))
        n += 1
    return n


WORKERS = {'preset_load': worker_preset_load, 'wav_render': worker_wav_render, 'ai_model_load': worker_ai_load}


# ---------------------------------------------------------------- scenario
def run(frames, parallel, tick_ms, tick_hz, worker, seconds, switch_ms):
    sys.setswitchinterval(switch_ms / 1000.0)
    synth, pm, seq = build_engine(parallel)
    base = render_cost(synth, seq, frames, 100)  # also warms numba caches
    if os.environ.get('BENCH_FREEZE'):
        import gc
        gc.collect(); gc.freeze()
    buf_ms = frames / SR * 1000
    durs, gaps, margins, entry = [], [], [], []
    state = {'underflows': 0, 'last': None, 'n': 0}
    stream_ref = {}

    def cb(outdata, nframes, tinfo, status):
        t0 = time.perf_counter()
        if status.output_underflow:
            state['underflows'] += 1
        if state['last'] is not None:
            gaps.append((t0 - state['last']) * 1000)
        state['last'] = t0
        if tinfo.currentTime > 0:
            entry.append((stream_ref['s'].time - tinfo.currentTime) * 1000)
        synth.process_audio_events(nframes, seq.advance(nframes))
        outdata.fill(0)
        durs.append((time.perf_counter() - t0) * 1000)
        if tinfo.outputBufferDacTime > 0:
            margins.append((tinfo.outputBufferDacTime - stream_ref['s'].time) * 1000)

    stop = threading.Event()
    result = {}
    wt = None
    if worker:
        wt = threading.Thread(target=lambda: result.__setitem__('n', WORKERS[worker](stop)), daemon=True)
    stream = sd.OutputStream(device=DEVICE, channels=2, callback=cb, samplerate=SR, blocksize=frames, dtype=np.float32)
    stream_ref['s'] = stream
    stream.start()
    time.sleep(0.5)
    durs.clear(); gaps.clear(); margins.clear(); entry.clear(); state['underflows'] = 0
    if wt:
        wt.start()
    t_end = time.perf_counter() + seconds
    period = 1.0 / tick_hz
    ticks = 0
    next_t = time.perf_counter()
    while time.perf_counter() < t_end:
        if tick_ms > 0:
            busy_python(tick_ms)
        ticks += 1
        next_t += period
        delay = next_t - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            next_t = time.perf_counter()
    stop.set()
    if wt:
        wt.join(timeout=30)
    stream.abort()
    stream.close()
    synth.cleanup()

    def pct(xs, p):
        return round(float(np.percentile(xs, p)), 3) if xs else None

    over = sum(1 for d in durs if d > buf_ms)
    return {
        'device': DEVICE or 'default', 'gc_frozen': bool(os.environ.get('BENCH_FREEZE')), 'buffer_ms': round(buf_ms, 2), 'frames': frames, 'parallel': parallel,
        'tick_ms': tick_ms, 'tick_hz': tick_hz, 'worker': worker, 'switch_ms': switch_ms,
        'callbacks': len(durs), 'underflows': state['underflows'], 'over_budget': over,
        'render_alone_p50': pct(base, 50), 'render_alone_max': pct(base, 100),
        'cb_p50': pct(durs, 50), 'cb_p99': pct(durs, 99), 'cb_max': pct(durs, 100),
        'gap_max': pct(gaps, 100), 'gap_p99': pct(gaps, 99),
        'entry_delay_p99': pct(entry, 99), 'entry_delay_max': pct(entry, 100),
        'margin_min': pct(margins, 0), 'ticks': ticks, 'worker_iterations': result.get('n'),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=float, default=8)
    ap.add_argument('--suite', default='all')
    a = ap.parse_args()
    if a.suite in ('all', 'frame'):
        t = [realistic_frame() for _ in range(3)]
        ts = []
        for _ in range(200):
            t0 = time.perf_counter(); realistic_frame(); ts.append((time.perf_counter() - t0) * 1000)
        print(json.dumps({'realistic_frame_ms_p50': round(statistics.median(ts), 3), 'max': round(max(ts), 3),
                          'bytes': len(t[0])}), flush=True)
    scen = []
    if a.suite == 'smoke':
        scen = [(1050, True, 0, 60, None, 5), (512, True, 8, 60, None, 5)]
    if a.suite in ('all', 'tick'):
        for frames in (1050, 512, 256):
            for parallel in (True, False):
                for tick in (0, 2, 4, 8, 12, 16):
                    scen.append((frames, parallel, tick, 60, None, 5))
    if a.suite in ('all', 'switch'):
        for frames in (512, 256):
            for tick in (8, 16):
                scen.append((frames, True, tick, 60, None, 1))
    if a.suite in ('all', 'worker'):
        for frames in (1050, 512, 256):
            for w in WORKERS:
                scen.append((frames, True, 2, 60, w, 5))
    if scen:
        run(512, True, 0, 60, None, 2, 5)  # warm-up: the first stream of a process can stall on cold start
    for s in scen:
        print(json.dumps(run(s[0], s[1], s[2], s[3], s[4], a.seconds, s[5])), flush=True)


if __name__ == '__main__':
    main()
