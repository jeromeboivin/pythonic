# UI frame budget that keeps audio safe

Task for [#10](https://github.com/jeromeboivin/pythonic/issues/10), part of the GUI map [#1](https://github.com/jeromeboivin/pythonic/issues/1). This file is measurement only; the recommendations at the end are inputs to the spec, not decisions.

- **Snapshot:** commit `25e9795` on `main`.
- **Machine:** Intel Xeon E3-1245 v5 (4 cores / 8 threads, 3.5 GHz), Linux 6.8, PipeWire 1.0.5. Python 3.12.3, numba 0.68.0, numpy 2.4.1, sounddevice 0.5.5.
- **Caveats:** one machine; another job was running in the background during the runs (load average around 3); each scenario lasted 6 s. The runs used the ALSA `sysdefault` device, because the ALSA `default` device (through PipeWire's pulse layer) stopped delivering callbacks partway through (see "Other observations"). Treat single spikes as noise and look at the trends.
- **Files:** `bench_frame_budget.py` (real engine in a real `sounddevice.OutputStream`, output silenced), `gil_probe.py` (longest GIL hold per action), and the raw results in `results-*.jsonl`, all in this folder.

## Method

- **Engine:** the `909.mtpreset` sounds, pattern A with **every channel triggered on every 16th** (accents on the beats, a fill on step 16), 140 BPM, 44.1 kHz. This is a worst case: all 8 voices are active in every callback.
- **Callback:** exactly what the GUI does: `seq.advance` plus `synth.process_audio_events`. Each callback records:
  - its duration;
  - its entry delay (`stream.time - currentTime`: how long the C callback waited for the GIL before Python ran);
  - the margin to the DAC deadline (`outputBufferDacTime - stream.time` at exit);
  - PortAudio underflows.
- **Main thread:** a 60 Hz tick of pure-Python work holding the GIL for 0, 2, 4, 8, 12 or 16 ms per tick. This stands in for poll plus the bridge push.
- **Worker thread:** loops a heavy action for the whole scenario: preset load, WAV render, or AI model load.
- **Buffers:** 1050 frames (23.8 ms, the default), 512 frames (11.6 ms) and 256 frames (5.8 ms).
- **Variants:**
  - `parallel_channel_processing` on (what `gui/main_window.py:103` sets today) and off;
  - GIL switch interval 5 ms (Python's default) and 1 ms;
  - GC heap frozen after start-up (`gc.freeze()`) or not.

## Q1. Does the render loop release the GIL?

**Yes, the kernel does.** Every numba function in `pythonic/voice_kernel.py` is compiled with `@njit(cache=True, nogil=True)`. Everything around it holds the GIL: the per-channel Python in `DrumChannel.process` (LFO and pump block-rate modulation, vintage, delay, reverb set-up), the mixing, the sequencer and the callback wrapper.

`parallel_channel_processing=True` hands each active channel to an 8-worker `ThreadPoolExecutor`. Only the nogil kernel can actually run in parallel; the Python around it contends for the GIL, and the pool adds thread hand-offs. The synth's own comment (`synthesizer.py:76`) says the pool is meant for non-real-time renderers, but the GUI enables it for the live stream. With the GC not frozen, the pool made things worse: at 23.8 ms it gave 1/0/1/8/12 underflows for ticks of 0/2/4/8/12 ms, against 0/0/0/1/1 with the pool off. With the GC frozen, the two were about equal. It never helped.

## Q2. Callback margin under main-thread load

Python's GIL hand-off bounds how long the callback waits to enter. The **entry delay follows the tick cost up to the 5 ms switch interval**: about 2.2 ms for a 2 ms tick, 4.2 ms for 4 ms, and 5.3-6 ms for 8 ms and above. Without a frozen GC heap, full collections add pauses on top (Q3).

Callback duration with no tick: about 2-5 ms median and 9-17 ms at p99 for the worst-case pattern. The spikes come from render cost and background load, not from the UI.

GC heap frozen, pool off (`results-frozen.jsonl`):

| Buffer | Tick per 60 Hz frame | Underflows / callbacks | Callbacks over budget | Entry delay max | Min margin to deadline |
|---|---|---|---|---|---|
| 23.8 ms | 0 ms | 0 / 253 | 0 | 0.3 ms | 9.6 ms |
| 11.6 ms | 0 ms | 0 / 517 | 0 | 2.6 ms | 24.3 ms |
| 11.6 ms | 4 ms | 0 / 517 | 2 | 4.8 ms | 19.7 ms |
| 11.6 ms | 8 ms | 0 / 517 | 39 | 5.9 ms | 10.6 ms |
| 5.8 ms | 2 ms (1 ms switch) | 0 / 1034 | 18 | 11.2 ms | 23.5 ms |
| 5.8 ms | 4 ms (1 ms switch) | 0 / 1034 | 53 | 10.4 ms | 23.7 ms |

"Over budget" means a callback took longer than its buffer. The host buffer (2-3 periods) absorbed every one of these without an underflow, but each one eats into the headroom.

GC not frozen (`results-sysdefault.jsonl`, the full sweep): the 16 ms tick (main thread 100 % busy) collapses every buffer, with 85-206 underflows in 6 s. Ticks of 8-12 ms already give underflows at 23.8 ms when the pool is on.

**A 1 ms switch interval** brought no clear gain at 11.6 ms (entry delay max 6.8 ms versus 4.8 ms at 5 ms, within noise), so changing it is not worth it.

## Q3. Heavy actions on a worker thread

Longest GIL hold per action, measured by a 1 ms sleeper thread (`results-gil-probe.jsonl`):

| Action | Time per run | Longest GIL hold |
|---|---|---|
| Idle baseline | n/a | 2.6 ms |
| **Full garbage collection** (`gc.collect()`), engine + tkinter + torch loaded | 121 ms | **132 ms** |
| **`import torch`** (first time) | 2.1 s | **381 ms** |
| **Load the patch model** (`drum_cvae_best.pt`, 116 MB) | 9.0 s | **126 ms** |
| Load the pattern model | 0.2 s | 3.0 ms |
| Preset load (parse `.mtpreset`, new synth, patterns) | 81 ms | 5.3 ms |
| WAV render of 2 s, worst-case pattern | 181 ms | 1.8 ms |
| One realistic UI frame (`json.dumps` of 120 changes, transport, meters, 8×64 steps; 15 KB) | 0.4-1.2 ms | (single call) |

How long a **full GC** takes depends on the heap: 31 ms with only the engine modules loaded, 43 ms with tkinter and one synth, 92 ms after `import torch`. **`gc.freeze()` after start-up takes it to about 0 ms**, and it stays about 1 ms after five more preset loads. A full collection can start on any allocating thread, including the audio callback (which allocates every block), and stops every thread until it ends.

Worker scenarios in the real stream, GC not frozen, 2 ms tick:
- **Preset load in a loop:** 1-6 underflows and stalls of 61-123 ms. With the heap frozen this fell to one underflow at 23.8 ms and none at 11.6 ms. That loop runs 12 loads a second, far more than real use.
- **WAV render:** clean at 11.6 and 5.8 ms; one 68 ms gap at 23.8 ms.
- **AI model load:** the first run in the process, which included `import torch`, gave 15 underflows and a 166 ms stall. Once torch was imported, the loads were clean.

## Answers

1. **UI tick budget: at most 4 ms of GIL-holding Python per frame at 60 Hz** (about 25 % of the main thread). Aim for 2 ms or less, and avoid single C calls longer than a few ms. A realistic poll plus serialise frame measured 0.4-1.2 ms, so the budget has about 3× headroom. Frames above 8 ms start eating the audio margin.
2. **Minimum safe buffer: 512 frames (11.6 ms)**, with the GC heap frozen and the channel pool off. It had no underflows up to an 8 ms tick and with a preset load or WAV render running. 256 frames (5.8 ms) had no underflows either, but 18-53 callbacks per 6 s ran over their buffer, so it has no headroom on this machine. Keep 23.8 ms as the default.
3. **Must run in a subprocess, not a thread:** `import torch` and AI patch-model loading (GIL holds of 381 ms and 126 ms, 9 s run). Since the AI pattern and patch generators and their models all need torch, the whole AI feature module should live in a subprocess. **Fine in a thread:** preset and drum-patch load (after `gc.freeze()`), WAV render, pattern-model loading on its own (but it needs torch), and MIDI file export (pure Python, small).
4. **Not measured:** AI inference itself, PO-32 encode/decode, and the QWebChannel push cost on top of serialisation (PySide6 is not installed). Inference and PO-32 run inside the modules recommended for a subprocess or worker anyway.

## Recommendations for the spec

- The core calls `gc.freeze()` after start-up and again after every bulk swap (preset load, AI apply). The automatic thresholds stay as they are, so only young objects are ever scanned.
- The live stream runs with `parallel_channel_processing=False`. The pool stays available for offline renders.
- The AI feature module runs in a subprocess, and nothing in the GUI process imports torch.
- The front-end tick does one `poll` and one push per frame and keeps to the 4 ms budget. Large one-off payloads (a whole pattern bank, the full preset) go out in slices over several frames.

## Other observations

- **Stream restarts on PipeWire:** after a few dozen open/close cycles of `OutputStream` on the ALSA `default` device (PipeWire's pulse layer), new streams delivered about 4 callbacks and then stopped, and `stop()` could hang. `sysdefault` kept working, and plain pulse playback (`pacat`) kept working. The core's stream restart (device or rate change) should use `abort()` rather than `stop()`, with a timeout, and report a failure through `poll` instead of blocking.
- **Cold start:** the first stream opened in a fresh process once stalled for about 5 s before its second callback. The bench now discards a warm-up run.
