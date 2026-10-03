"""
Sample-accurate pattern sequencer.

The clock runs at 1920 ticks per quarter note and advances once per 64-sample
block (the fractional tick count is carried over).
Swing works on eighth-note pairs (960 ticks): the second sixteenth of a pair
is moved from tick 480 to tick 480 + 240 * swing and the ticks on either side
are stretched linearly.  Events inside a block are placed by linear
interpolation between the block's first and last tick, then snapped down to the
4-sample trigger grid of the voices.

Fills retrigger a step every step_ticks / fill_rate ticks with a velocity
that drops by 64 / fill_rate on each hit (127 -> 64 accented, 64 -> 0 normal).
Probability and sub-steps are Pythonic extensions and use the same clock.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import numpy as np

TICKS_PER_QUARTER = 1920
PAIR_TICKS = 960
BLOCK = 64
STEP_TICKS = {'1/8': 960, '1/8T': 640, '1/16': 480, '1/16T': 320, '1/32': 240}

Event = Tuple[int, int, int]  # (sample offset, channel, velocity)


class StepSequencer:
    """Turns a PatternManager into sample-accurate trigger events."""

    def __init__(self, pattern_manager, sample_rate: int = 44100, seed: Optional[int] = None):
        self.pm = pattern_manager
        self.sr = sample_rate
        self.rng = random.Random(seed)
        self._inv_sr = float(np.float32(1.0 / sample_rate))
        self.running = False
        self._reset_state()

    # ------------------------------------------------------------------ control
    def _reset_state(self):
        self.clock = 0              # swung clock ticks
        self.frac = 0.0
        self.lead = 0               # samples before the first block starts
        self.block_pos = BLOCK      # position inside the current 64-sample block
        self.block_events: List[Event] = []
        self.split = 480
        self.step_index = -1        # step of the playing pattern last fired
        self.step_tick = 0          # pattern tick of that step
        self.next_step_tick = 0
        self.fill_vel = {}          # channel -> current fill velocity (float)
        self.sub_ticks = []         # pending (tick, channel, velocity) sub-step hits

    def start(self, pattern_index: Optional[int] = None, synth_clock: int = 0):
        """Start playback; the first step lands on the next 4-sample boundary."""
        if pattern_index is not None:
            self.pm.start_playback(pattern_index)
        self._reset_state()
        self.lead = (-synth_clock) % 4
        self.running = True

    def stop(self):
        self.running = False
        self._reset_state()

    # ------------------------------------------------------------------ helpers
    def _step_ticks(self) -> int:
        return STEP_TICKS.get(self.pm.step_rate, 480)

    def _fill_ticks(self) -> int:
        return max(1, int(round(self._step_ticks() / max(float(self.pm.fill_rate), 1.0))))

    def _fire_step(self, tick: int, out: list):
        """Advance to the next step at pattern tick `tick` and emit its hits."""
        pm = self.pm
        if self.step_index >= 0:
            self.step_index += 1
            if self.step_index >= pm.get_playing_pattern().length:
                pm.advance_to_next_pattern()
                self.step_index = 0
        else:
            self.step_index = 0
        pm.play_position = self.step_index
        self.step_tick = tick
        pattern = pm.get_playing_pattern()
        step_ticks = self._step_ticks()
        new_fill = {}
        for ch in range(pm.num_channels):
            channel = pattern.get_channel(ch)
            if channel is None:
                continue
            step = channel.get_step(self.step_index)
            if not step.trigger:
                continue
            prob = getattr(step, 'probability', 100)
            if prob < 100 and self.rng.randint(1, 100) > prob:
                continue
            vel = 127 if step.accent else 64
            subs = getattr(step, 'substeps', '') or ''
            if subs:
                n = len(subs)
                for i, mark in enumerate(subs):
                    if mark in 'oO':
                        if i == 0:
                            out.append((ch, vel))
                        else:
                            self.sub_ticks.append((tick + (i * step_ticks) // n, ch, vel))
            else:
                out.append((ch, vel))
            if step.fill:
                new_fill[ch] = float(vel)
        self.fill_vel = new_fill
        self.next_step_tick = tick + step_ticks

    def _events_in(self, p0: int, p1: int):
        """Yield (tick, [(ch, vel)...]) for pattern ticks in [p0, p1)."""
        k = p0
        fill_ticks = self._fill_ticks()
        while k < p1:
            cands = [self.next_step_tick]
            if self.fill_vel:
                rel = k - self.step_tick
                nf = self.step_tick + max(1, -(-rel // fill_ticks)) * fill_ticks
                if nf < self.next_step_tick:
                    cands.append(nf)
            cands.extend(t for t, _, _ in self.sub_ticks)
            t = min(c for c in cands if c >= k) if any(c >= k for c in cands) else p1
            if t >= p1:
                return
            hits = []
            if t == self.next_step_tick:
                self._fire_step(t, hits)
            elif self.fill_vel and (t - self.step_tick) % fill_ticks == 0:
                dec = 64.0 / max(float(self.pm.fill_rate), 1.0)
                for ch in list(self.fill_vel):
                    self.fill_vel[ch] -= dec
                    v = int(self.fill_vel[ch])
                    if v > 0:
                        hits.append((ch, v))
            due = [s for s in self.sub_ticks if s[0] == t]
            if due:
                self.sub_ticks = [s for s in self.sub_ticks if s[0] != t]
                hits.extend((ch, v) for _, ch, v in due)
            if hits:
                yield t, hits
            k = t + 1

    def _make_block(self):
        """Compute the events of the next 64-sample block (offsets within the block)."""
        events: List[Event] = []
        if not (self.running and self.pm.is_playing):
            self.block_events = events
            return
        tempo = float(self.pm.bpm)
        d = self._inv_sr * tempo * 32.0 * BLOCK + self.frac
        nt = int(d)
        self.frac = d - nt
        t0 = self.clock
        t1 = t0 + nt
        self.clock = t1
        if nt <= 0:
            self.block_events = events
            return
        swing = min(max(float(self.pm.swing), 0.0), 1.0)
        t = t0
        s_start = 0
        while t < t1:
            pair = t // PAIR_TICKS
            ps = pair * PAIR_TICKS
            if t == ps:
                v = swing * 240.0 + 480.0
                self.split = int(v + 0.5)
            split = self.split
            pos = t - ps
            remain = t1 - t
            if pos < split:
                seg = min(remain, ps + split - t)
                p0 = ps + (pos * 480) // split
                p1 = ps + ((seg + pos) * 480) // split
            else:
                seg = min(remain, ps + PAIR_TICKS - t)
                p0 = ((pos - split) * 480) // (PAIR_TICKS - split) + 480 + ps
                p1 = ps + 480 + ((seg + pos - split) * 480) // (PAIR_TICKS - split)
            s_cnt = ((seg + t - t0) * BLOCK) // nt - s_start
            if p1 > p0:
                for k, hits in self._events_in(p0, p1):
                    off = s_start + ((k - p0) * s_cnt) // (p1 - p0)
                    off -= off % 4
                    for ch, vel in hits:
                        events.append((off, ch, vel))
            s_start += s_cnt
            t += seg
        self.block_events = events

    # ------------------------------------------------------------------ main
    def advance(self, num_samples: int) -> List[Event]:
        """Advance by `num_samples`; returns events with offsets inside that span."""
        out: List[Event] = []
        if not self.running:
            return out
        pos = 0
        if self.lead:
            k = min(self.lead, num_samples)
            self.lead -= k
            pos = k
        while pos < num_samples:
            if self.block_pos >= BLOCK:
                self._make_block()
                self.block_pos = 0
            take = min(BLOCK - self.block_pos, num_samples - pos)
            for off, ch, vel in self.block_events:
                if self.block_pos <= off < self.block_pos + take:
                    out.append((pos + off - self.block_pos, ch, vel))
            self.block_pos += take
            pos += take
        out.sort()
        return out
