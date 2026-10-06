"""
Audio stream ownership and the audio callback of the app core.

Threads never share engine state directly. Any thread queues commands with
``submit_call`` / ``submit_trigger``; the audio callback drains the queue at
the start of each block, so changes land between blocks and triggers land at
the sample offset that matches their arrival time (one buffer of constant
latency). While no stream runs, the submitting thread applies the queue itself.

The callback takes no locks: the queue is a ``deque`` (atomic append and
popleft) and every value it publishes is a plain attribute write.
"""

import threading
import time
from collections import deque

import numpy as np

from pythonic.sequencer import StepSequencer

_TRIGGER = 0
_CALL = 1


class AudioUnavailable(RuntimeError):
    """No audio output could be opened."""


def run_with_timeout(fn, timeout):
    """Run fn() on a helper thread; return (finished, exception)."""
    box = {}

    def target():
        try:
            fn()
        except BaseException as exc:  # reported to the caller
            box['exc'] = exc

    worker = threading.Thread(target=target, name='pythonic-stream-ctl', daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        return False, None
    return True, box.get('exc')


def block_size_for(buffer_ms, rate):
    return max(64, int(round(buffer_ms / 1000.0 * rate)))


class AudioEngine:
    """Owns the output stream, the command queue and the per-block render."""

    def __init__(self, synth, pattern_manager, backend, *, clock=time.perf_counter,
                 stream_timeout=2.0):
        self.synth = synth
        self.pattern_manager = pattern_manager
        self.backend = backend
        self.stream_timeout = stream_timeout
        self._clock = clock

        # Queue in. The lock is only taken by non-audio threads.
        self._queue = deque()
        self._lock = threading.Lock()
        self._seq = 0
        self.drained_seq = 0
        self._live = False
        self._gen = 0
        self._stream = None
        self._triggers = []

        # Stream configuration (changed only while no stream runs)
        self.out_rate = 44100
        self.buffer_ms = 23.8
        self.block_size = 1050
        self.buffer_time_ms = self.block_size / self.out_rate * 1000.0
        self.device_index = None
        self._device_name = None
        self._ratio = 1.0
        self._last_good_audio = np.zeros((self.block_size, 2), dtype=np.float32)
        self._reset_resampler()

        # Sequencer; replaced together with the synth
        self.sequencer = StepSequencer(pattern_manager, synth.sr)
        self._seq_generation = -1
        self._followed_pattern = None  # playing pattern the selection last followed

        # Performance counters (written by the callback only)
        self.enable_sample_dropping = True
        self.callback_times = deque(maxlen=100)
        self.trigger_times = deque(maxlen=100)
        self.process_times = deque(maxlen=100)
        self.callback_count = 0
        self.underrun_count = 0
        self.dropped_callback_count = 0
        self.error_count = 0
        self.last_error = None

    # ------------------------------------------------------------------ queue in
    def submit_call(self, fn, arg=None):
        """Queue fn(arg) to run on the audio thread at the next block start."""
        return self._submit((0, _CALL, fn, arg, None))

    def submit_trigger(self, at, channel, velocity):
        """Queue a trigger that arrived at clock time `at`."""
        return self._submit((0, _TRIGGER, at, channel, velocity))

    def _submit(self, item):
        with self._lock:
            self._seq += 1
            self._queue.append((self._seq,) + item[1:])
            seq = self._seq
            if not self._live:
                self._drain_inline()
        return seq

    def _drain_inline(self):
        """Apply queued calls on the calling thread (no stream; lock held)."""
        q = self._queue
        while q:
            item = q.popleft()
            if item[1] == _CALL:
                try:
                    item[2](item[3])
                except Exception as exc:
                    self.error_count += 1
                    self.last_error = exc
            self.drained_seq = item[0]

    def wait_applied(self, seq, timeout):
        """Wait until queued item `seq` has been applied; False on timeout."""
        end = time.monotonic() + timeout
        while self.drained_seq < seq:
            if not self._live:
                with self._lock:
                    if not self._live:
                        self._drain_inline()
                continue
            if time.monotonic() > end:
                return False
            time.sleep(0.0005)
        return True

    @property
    def live(self):
        return self._live

    # ------------------------------------------------------------------ synth swap
    def install(self, synth, sequencer):
        """Swap in a synth built off the audio thread (runs at block start)."""
        self.synth = synth
        self.sequencer = sequencer
        self._seq_generation = -1
        self._ratio = self.out_rate / synth.sr
        self._reset_resampler()

    # ------------------------------------------------------------------ callback
    def make_callback(self, gen):
        def callback(outdata, frames, time_info, status):
            if gen != self._gen:  # an abandoned stream: never touch the engine
                outdata.fill(0)
                return
            self.process(outdata, frames, time_info, status)
        return callback

    def process(self, outdata, frames, time_info, status):
        """The audio callback body."""
        now = self._clock()
        self.callback_count += 1
        if status:
            self.underrun_count += 1
        try:
            self._render(outdata, frames, now)
        except Exception as exc:
            outdata.fill(0)
            self.error_count += 1
            self.last_error = exc

    def _render(self, outdata, frames, now):
        callback_start = time.perf_counter()
        trigger_time = 0.0

        # Block start: apply queued changes, collect triggers
        triggers = self._triggers
        triggers.clear()
        q = self._queue
        while q:
            try:
                item = q.popleft()
            except IndexError:
                break
            if item[1] == _TRIGGER:
                triggers.append(item)
            else:
                try:
                    item[2](item[3])
                except Exception as exc:
                    self.error_count += 1
                    self.last_error = exc
            self.drained_seq = item[0]

        synth = self.synth
        pm = self.pattern_manager

        # Adaptive: if we're consistently running behind, drop audio processing
        drop_this_callback = False
        times = self.callback_times
        if self.enable_sample_dropping and len(times) > 10:
            recent_total = 0.0
            for i in range(1, 11):
                recent_total += times[-i]
            if recent_total > self.buffer_time_ms * 9.0:
                drop_this_callback = True
                self.dropped_callback_count += 1

        ratio = self._ratio
        synth_frames = frames if ratio == 1.0 else max(1, int(round(frames / ratio)))

        # Sample-accurate pattern sequencing
        events = []
        sequencer = self.sequencer
        if pm.is_playing:
            generation = getattr(pm, 'playback_generation', 0)
            if not sequencer.running or generation != self._seq_generation:
                self._seq_generation = generation
                sequencer.start(None, synth_clock=synth.sample_clock)
                self._followed_pattern = pm.playing_pattern_index
            trigger_start = time.perf_counter()
            events = sequencer.advance(synth_frames)
            playing = pm.playing_pattern_index
            if playing != self._followed_pattern:
                # A chain or the queue moved on: the selection follows it
                self._followed_pattern = playing
                pm.selected_pattern_index = playing
            if drop_this_callback:
                events = []
            if events:
                trigger_time = (time.perf_counter() - trigger_start) * 1000.0
        elif sequencer.running:
            sequencer.stop()

        if drop_this_callback:
            last = self._last_good_audio
            if frames <= len(last):
                outdata[:] = last[:frames]
            else:
                outdata.fill(0)
            self.callback_times.append((time.perf_counter() - callback_start) * 1000.0)
            return

        # Queued triggers: placed at the offset of their arrival in the last buffer period
        if triggers:
            block_start = now - frames / self.out_rate
            sr = synth.sr
            last_offset = synth_frames - 1
            for _seq, _kind, at, channel, velocity in triggers:
                offset = int((at - block_start) * sr)
                if offset < 0:
                    offset = 0
                elif offset > last_offset:
                    offset = last_offset
                events.append((offset, channel, velocity))

        process_start = time.perf_counter()
        if ratio == 1.0:
            audio = synth.process_audio_events(frames, events)
        else:
            audio = self._upsample_linear(synth.process_audio_events(synth_frames, events),
                                          synth_frames, frames)
        process_time = (time.perf_counter() - process_start) * 1000.0

        outdata[:] = audio
        if frames <= len(self._last_good_audio):
            self._last_good_audio[:frames] = audio

        self.callback_times.append((time.perf_counter() - callback_start) * 1000.0)
        if trigger_time > 0:
            self.trigger_times.append(trigger_time)
        self.process_times.append(process_time)

    # ------------------------------------------------------------------ resampling
    def _reset_resampler(self):
        self._resample_out_frames = 0
        self._resample_in_frames = 0
        self._resample_x_out = None
        self._resample_x_in = None
        self._resample_buffer = None

    def _upsample_linear(self, audio, in_frames, out_frames):
        """Upsample stereo audio using linear interpolation (lo-fi preserving)."""
        if in_frames == out_frames:
            return audio
        if out_frames != self._resample_out_frames or in_frames != self._resample_in_frames:
            self._resample_x_out = np.linspace(0, in_frames - 1, out_frames)
            self._resample_x_in = np.arange(in_frames, dtype=np.float64)
            self._resample_buffer = np.empty((out_frames, 2), dtype=np.float32)
            self._resample_out_frames = out_frames
            self._resample_in_frames = in_frames
        self._resample_buffer[:, 0] = np.interp(self._resample_x_out, self._resample_x_in, audio[:, 0])
        self._resample_buffer[:, 1] = np.interp(self._resample_x_out, self._resample_x_in, audio[:, 1])
        return self._resample_buffer

    # ------------------------------------------------------------------ stream lifecycle
    def configure(self, out_rate, buffer_ms):
        """Set the output rate and buffer size (only while no stream runs)."""
        self.out_rate = int(out_rate)
        self.buffer_ms = float(buffer_ms)
        self.block_size = block_size_for(self.buffer_ms, self.out_rate)
        self.buffer_time_ms = self.block_size / self.out_rate * 1000.0
        self._last_good_audio = np.zeros((self.block_size, 2), dtype=np.float32)
        self._ratio = self.out_rate / self.synth.sr
        self._reset_resampler()

    def _find_output_device(self, name):
        if not name:
            return None
        try:
            for i, dev in enumerate(self.backend.query_devices()):
                if dev['max_output_channels'] > 0 and dev['name'] == name:
                    return i
            print(f"Audio device '{name}' not found, using system default", flush=True)
        except Exception as exc:
            print(f"Error querying audio devices: {exc}", flush=True)
        return None

    def start(self, device_name=None):
        """Open and start the output stream, with the old fallbacks.

        Tries (device, rate) -> (device, 44100) -> (default, 44100). Returns a
        dict describing the stream; raises AudioUnavailable when nothing opens.
        """
        if self.backend is None:
            raise AudioUnavailable('Audio output not available: sounddevice is not installed')
        if self._stream is not None:
            return self.status()
        device = self._find_output_device(device_name)

        try:
            self.backend.check_output_settings(device=device, channels=2, samplerate=self.out_rate)
        except Exception:
            default_out = self.backend.default.device[1]
            info = self.backend.query_devices(device if device is not None else default_out)
            fallback = int(info['default_samplerate'])
            print(f"WARNING: {self.out_rate} Hz not supported by device, "
                  f"falling back to {fallback} Hz", flush=True)
            self.configure(fallback, self.buffer_ms)

        attempts = [
            (device, self.out_rate, 'requested'),
            (device, 44100, '44100 Hz fallback'),
            (None, 44100, 'system default @ 44100 Hz'),
        ]
        errors = []
        for dev, rate, desc in attempts:
            if rate != self.out_rate:
                self.configure(rate, self.buffer_ms)
            gen = self._gen + 1
            try:
                stream = self.backend.OutputStream(
                    device=dev, channels=2, callback=self.make_callback(gen),
                    samplerate=self.out_rate, blocksize=self.block_size, dtype=np.float32)
            except Exception as exc:
                print(f"Failed to open audio ({desc}): {exc}", flush=True)
                errors.append(f'{desc}: {exc}')
                continue
            with self._lock:
                self._drain_inline()
                self._gen = gen
                self._live = True
            finished, exc = run_with_timeout(stream.start, self.stream_timeout)
            if finished and exc is None:
                self._stream = stream
                self.device_index = dev
                self._device_name = self._resolve_device_name(dev)
                info = self.status()
                synth_info = (f", synth @ {self.synth.sr} Hz" if self.synth.sr != self.out_rate else "")
                mono_info = ", MONO" if self.synth.mono else ""
                default = " [default]" if info['default_device'] else ""
                print(f"Audio stream started on '{info['device']}'{default} @ {self.out_rate} Hz "
                      f"(buffer: {self.block_size} samples, ~{info['latency_ms']:.1f}ms latency"
                      f"{synth_info}{mono_info})", flush=True)
                return info
            reason = exc if finished else f'start() did not return within {self.stream_timeout:g} s'
            print(f"Failed to open audio ({desc}): {reason}", flush=True)
            errors.append(f'{desc}: {reason}')
            with self._lock:
                self._gen += 1
                self._live = False
                self._drain_inline()
            run_with_timeout(lambda s=stream: (s.abort(), s.close()), self.stream_timeout)
        print("ERROR: Could not open any audio device!", flush=True)
        raise AudioUnavailable('Could not open any audio device (' + '; '.join(errors) + ')')

    def stop(self):
        """Abort the stream with a timeout. Returns an error message or None."""
        stream = self._stream
        error = None
        if stream is not None:
            def abort_and_close():
                stream.abort()
                stream.close()
            finished, exc = run_with_timeout(abort_and_close, self.stream_timeout)
            if not finished:
                error = (f'Audio stream did not stop within {self.stream_timeout:g} s; '
                         f'it was abandoned')
            elif exc is not None:
                error = f'Audio stream failed to stop cleanly: {exc}'
        with self._lock:
            self._gen += 1  # a stream that failed to stop can no longer touch the engine
            self._live = False
            self._stream = None
            self._device_name = None
            self._drain_inline()
        if error:
            print(f"ERROR: {error}", flush=True)
        return error

    # ------------------------------------------------------------------ readouts
    def _resolve_device_name(self, index):
        try:
            if index is None:
                index = self.backend.default.device[1]
            return self.backend.query_devices(index)['name']
        except Exception:
            return None

    def device_name(self):
        """Name of the output device in use (None when no stream runs)."""
        return self._device_name

    def _device_names(self, key):
        if self.backend is None:
            return []
        try:
            return [d['name'] for d in self.backend.query_devices() if d[key] > 0]
        except Exception as exc:
            print(f"Error querying audio devices: {exc}", flush=True)
            return []

    def output_devices(self):
        return self._device_names('max_output_channels')

    def input_devices(self):
        return self._device_names('max_input_channels')

    def status(self):
        running = self._stream is not None
        return {
            'running': running,
            'device': self.device_name(),
            'default_device': running and self.device_index is None,
            'sample_rate': self.out_rate,
            'synth_rate': self.synth.sr,
            'block_size': self.block_size,
            'latency_ms': self.block_size / self.out_rate * 1000.0,
            'mono': self.synth.mono,
            'callbacks': self.callback_count,
            'underruns': self.underrun_count,
            'dropped': self.dropped_callback_count,
        }

    def performance_report(self):
        """The periodic console report, built off the audio thread."""
        times = list(self.callback_times)
        if not times:
            return None
        process = list(self.process_times)
        trig = list(self.trigger_times)
        avg_callback = float(np.mean(times))
        max_callback = float(np.max(times))
        avg_process = float(np.mean(process)) if process else 0.0
        avg_trigger = float(np.mean(trig)) if trig else 0.0
        buffer_time_ms = self.buffer_time_ms
        utilization = avg_callback / buffer_time_ms * 100.0
        synth = self.synth
        active = sum(1 for ch in synth.channels if ch.is_active)
        lines = [
            f"\n=== Audio Performance (last {len(times)} callbacks) ===",
            f"Callbacks: {self.callback_count}, Underruns: {self.underrun_count}, "
            f"Dropped: {self.dropped_callback_count}",
            f"Buffer time available: {buffer_time_ms:.2f}ms",
            f"Active channels: {active}/8",
            f"Sample dropping: {'ENABLED' if self.enable_sample_dropping else 'DISABLED'}",
        ]
        if self._ratio != 1.0:
            lines.append(f"Synth rate: {synth.sr} Hz → output: {self.out_rate} Hz "
                         f"(upsample {self._ratio:.1f}x)")
        lines += [
            f"Mono: {'YES' if synth.mono else 'NO'}",
            f"Avg callback time: {avg_callback:.3f}ms ({utilization:.1f}% utilization)",
            f"Max callback time: {max_callback:.3f}ms",
            f"Avg process_audio: {avg_process:.3f}ms",
            f"Avg trigger_step: {avg_trigger:.3f}ms",
        ]
        if max_callback > buffer_time_ms:
            lines.append(f"WARNING: Max callback time ({max_callback:.3f}ms) exceeds "
                         f"buffer time ({buffer_time_ms:.2f}ms)!")
        if avg_callback > buffer_time_ms * 0.8:
            lines.append(f"WARNING: High CPU utilization ({utilization:.1f}%)")
        lines.append("=" * 60)
        return "\n".join(lines)
