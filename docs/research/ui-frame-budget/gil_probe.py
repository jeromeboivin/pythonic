"""Longest GIL hold per heavy action: a probe thread sleeps 1 ms in a loop and records its lateness."""
import gc, json, os, sys, threading, time
sys.path.insert(0, os.getcwd())
import bench_frame_budget as b

def probe(action, label, reps=1):
    lat, stop = [], threading.Event()
    def p():
        while not stop.is_set():
            t = time.perf_counter(); time.sleep(0.001); lat.append((time.perf_counter() - t - 0.001) * 1000)
    th = threading.Thread(target=p, daemon=True); th.start(); time.sleep(0.2)
    t0 = time.perf_counter()
    for _ in range(reps): action()
    dt = (time.perf_counter() - t0) * 1000 / reps
    stop.set(); th.join()
    lat.sort()
    print(json.dumps({'action': label, 'ms_per_run': round(dt, 1), 'probe_late_max_ms': round(lat[-1], 2), 'p99': round(lat[int(len(lat) * .99)], 2)}), flush=True)

sys.setswitchinterval(0.005)
probe(lambda: time.sleep(0.5), 'idle baseline')
probe(lambda: __import__('torch'), 'import torch (first)')
from pythonic.drum_generator import PatchGenerator
from pythonic.pattern_generator import PatternGenerator
probe(lambda: PatchGenerator().load_model('drum_cvae_best.pt'), 'load patch model (116 MB)', 2)
probe(lambda: PatternGenerator().load_model('drum_patterns/pattern_cvae_best.pt'), 'load pattern model', 2)
st = {}
def preset():
    s = b.PythonicSynthesizer(b.SR); d = b.PresetManager(s).load_mtpreset(b.PRESET); s.load_preset_data(d)
    pm = b.PatternManager(8, 16); pm.load_from_preset_data(d['patterns'])
probe(preset, 'preset load (parse + new synth + patterns)', 5)
probe(lambda: b.PresetManager(b.PythonicSynthesizer(b.SR)).load_mtpreset(b.PRESET), 'parse .mtpreset only', 5)
probe(lambda: b.PythonicSynthesizer(b.SR), 'new PythonicSynthesizer()', 5)
probe(lambda: gc.collect(), 'gc.collect()', 3)
s, _, q = b.build_engine(False)
probe(lambda: [s.process_audio_events(1024, q.advance(1024)) for _ in range(86)], 'WAV render 2 s (worst-case pattern)', 2)
probe(lambda: b.realistic_frame(), 'one realistic UI frame (json)', 50)
