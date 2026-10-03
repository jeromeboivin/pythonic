"""Utilities for rendering .mtpreset files in tests and offline tooling."""

from __future__ import annotations

import math
from typing import List

import numpy as np

from pythonic.voice import DrumVoice
from pythonic.pattern_manager import PatternManager
from pythonic.preset_manager import PythonicPresetParser
from pythonic.sequencer import StepSequencer
from pythonic.synthesizer import PythonicSynthesizer


PATTERN_NAMES = list(PatternManager.PATTERN_NAMES)


def _pattern_index_from_name(name: str) -> int:
    if not name:
        return 0
    normalized = str(name).strip().strip('"').upper()
    if normalized in PATTERN_NAMES:
        return PATTERN_NAMES.index(normalized)
    return 0


def _apply_preset_to_synth(synth: PythonicSynthesizer, preset_data: dict) -> None:
    synth.set_master_volume(float(preset_data.get("master_volume_db", 0.0)))

    drums = preset_data.get("drums", [])
    for index, channel_data in enumerate(drums[: synth.NUM_CHANNELS]):
        synth.channels[index].set_parameters(channel_data, immediate=True)
        synth.channels[index].name = channel_data.get("name", synth.channels[index].name)

    mutes = preset_data.get("mutes", [])
    for index, muted in enumerate(mutes[: synth.NUM_CHANNELS]):
        synth.channels[index].muted = bool(muted)


def _pattern_manager_for(preset_data: dict, num_channels: int = 8) -> PatternManager:
    pm = PatternManager(num_channels=num_channels, pattern_length=16)
    pm.set_bpm(float(preset_data.get("tempo", 120)))
    pm.set_step_rate(preset_data.get("step_rate", "1/16"))
    pm.set_fill_rate(float(preset_data.get("fill_rate", 2)))
    pm.set_swing(float(preset_data.get("swing", 0.0)))
    pm.load_from_preset_data(preset_data.get("patterns"))
    return pm


def _sequence_length_samples(pm: PatternManager, start_index: int, sample_rate: int) -> int:
    """Duration of one pass of the selected pattern (and its chain) in samples."""
    from pythonic.sequencer import STEP_TICKS
    steps = 0
    seen = set()
    idx = start_index
    while 0 <= idx < len(pm.patterns) and idx not in seen:
        seen.add(idx)
        steps += pm.patterns[idx].length
        if not pm.patterns[idx].chained_to_next:
            break
        idx += 1
    ticks = steps * STEP_TICKS.get(pm.step_rate, 480)
    return int(math.ceil(ticks / (pm.bpm * 32.0 / sample_rate)))


def load_and_render_preset(
    preset_path: str,
    duration_seconds: float | None = None,
    sample_rate: int = 44100,
    noise_seed: int | None = None,
):
    """Load an .mtpreset file and render the selected pattern (and chain) once.

    Without a duration the render covers one pass of the sequence; with a
    duration the sequencer keeps looping until that length is reached.
    `noise_seed` reseeds the noise of every channel (default: fixed seeds).
    """
    parser = PythonicPresetParser()
    raw_data = parser.parse_file(preset_path)
    preset_data = parser.convert_to_synth_format(raw_data)

    synth = PythonicSynthesizer(sample_rate)
    if noise_seed is not None:
        for index, channel in enumerate(synth.channels):
            channel.voice = DrumVoice(sample_rate, seed=noise_seed * 100 + index)
    _apply_preset_to_synth(synth, preset_data)

    pm = _pattern_manager_for(preset_data, synth.NUM_CHANNELS)
    start = _pattern_index_from_name(raw_data.get("Pattern", "A"))

    if duration_seconds is None:
        total = _sequence_length_samples(pm, start, sample_rate)
    else:
        total = max(0, int(math.ceil(duration_seconds * sample_rate)))

    seq = StepSequencer(pm, sample_rate, seed=0)
    seq.start(start, synth_clock=synth.sample_clock)
    chunks: List[np.ndarray] = []
    block = 1024
    for pos in range(0, total, block):
        n = min(block, total - pos)
        chunks.append(synth.process_audio_events(n, seq.advance(n)))
    audio = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 2), dtype=np.float32)
    return audio.astype(np.float32, copy=False), synth, pm, preset_data


def get_preset_event_samples(preset_path: str, sample_rate: int = 44100) -> List[int]:
    """Return the scheduled trigger sample positions for one pass of the preset pattern."""
    parser = PythonicPresetParser()
    raw_data = parser.parse_file(preset_path)
    preset_data = parser.convert_to_synth_format(raw_data)
    pm = _pattern_manager_for(preset_data)
    start = _pattern_index_from_name(raw_data.get("Pattern", "A"))
    total = _sequence_length_samples(pm, start, sample_rate)
    seq = StepSequencer(pm, sample_rate, seed=0)
    seq.start(start)
    events = seq.advance(total)
    return sorted(off for off, _ch, _vel in events)
