"""Utilities for rendering .mtpreset files in tests and offline tooling."""

from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple

import numpy as np

from pythonic.pattern_manager import PatternManager
from pythonic.preset_manager import PythonicPresetParser
from pythonic.synthesizer import PythonicSynthesizer


PATTERN_NAMES = list(PatternManager.PATTERN_NAMES)


def _pattern_index_from_name(name: str) -> int:
    if not name:
        return 0
    normalized = str(name).strip().upper()
    if normalized in PATTERN_NAMES:
        return PATTERN_NAMES.index(normalized)
    return 0


def _apply_preset_to_synth(synth: PythonicSynthesizer, preset_data: dict) -> None:
    synth.master_volume_db = preset_data.get("master_volume_db", 0.0)

    drums = preset_data.get("drums", [])
    for index, channel_data in enumerate(drums[: synth.NUM_CHANNELS]):
        synth.channels[index].set_parameters(channel_data, immediate=True)
        synth.channels[index].name = channel_data.get("name", synth.channels[index].name)

    mutes = preset_data.get("mutes", [])
    for index, muted in enumerate(mutes[: synth.NUM_CHANNELS]):
        synth.channels[index].muted = bool(muted)


def _render_step_audio(
    synth: PythonicSynthesizer,
    step_samples: int,
    events: Sequence[Tuple[int, int, int]],
) -> np.ndarray:
    if step_samples <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    output = np.zeros((step_samples, 2), dtype=np.float32)
    cursor = 0

    for offset, channel_id, velocity in sorted(events):
        offset = max(0, min(step_samples, int(offset)))
        if offset > cursor:
            output[cursor:offset] = synth.process_audio(offset - cursor)
            cursor = offset
        synth.trigger_drum(channel_id, velocity)

    if cursor < step_samples:
        output[cursor:step_samples] = synth.process_audio(step_samples - cursor)

    return output


def _collect_step_events(
    pattern_manager: PatternManager,
    step_index: int,
    rng: random.Random,
    sample_rate: int,
) -> List[Tuple[int, int, int]]:
    pattern = pattern_manager.get_playing_pattern()
    step_duration_frames = int(round(pattern_manager.step_duration_ms * 0.001 * sample_rate))
    fill_rate = max(1, int(round(pattern_manager.fill_rate)))
    events: List[Tuple[int, int, int]] = []

    for channel_id in range(pattern_manager.num_channels):
        channel = pattern.get_channel(channel_id)
        if channel is None:
            continue

        step = channel.get_step(step_index)
        if not step.trigger:
            continue

        probability = getattr(step, "probability", 100)
        if probability < 100 and rng.randint(1, 100) > probability:
            continue

        velocity = 127 if step.accent else 64
        substeps = getattr(step, "substeps", "") or ""

        if substeps:
            num_substeps = len(substeps)
            substep_interval = step_duration_frames / max(num_substeps, 1)
            for index, marker in enumerate(substeps):
                if marker.lower() == "o":
                    events.append((int(round(index * substep_interval)), channel_id, velocity))
        else:
            events.append((0, channel_id, velocity))

        if step.fill:
            if step.accent:
                start_velocity = 127
                end_velocity = 64
            else:
                start_velocity = 64
                end_velocity = 0

            fill_interval = step_duration_frames / fill_rate
            for index in range(1, fill_rate):
                progress = index / (fill_rate - 1) if fill_rate > 1 else 0.0
                fill_velocity = int(round(start_velocity - (start_velocity - end_velocity) * progress))
                events.append((int(round(index * fill_interval)), channel_id, max(1, fill_velocity)))

    return events


def _pattern_sequence(pattern_manager: PatternManager, start_index: int) -> List[int]:
    sequence = [start_index]
    current = start_index
    seen = {start_index}

    while 0 <= current < len(pattern_manager.patterns):
        pattern = pattern_manager.patterns[current]
        if not pattern.chained_to_next:
            break
        next_index = current + 1
        if next_index in seen or next_index >= len(pattern_manager.patterns):
            break
        sequence.append(next_index)
        seen.add(next_index)
        current = next_index

    return sequence


def load_and_render_preset(
    preset_path: str,
    duration_seconds: float | None = None,
    sample_rate: int = 44100,
):
    """Load an .mtpreset file and render the selected pattern once plus tail."""

    parser = PythonicPresetParser()
    raw_data = parser.parse_file(preset_path)
    preset_data = parser.convert_to_synth_format(raw_data)

    synth = PythonicSynthesizer(sample_rate)
    _apply_preset_to_synth(synth, preset_data)

    pattern_manager = PatternManager(num_channels=synth.NUM_CHANNELS, pattern_length=16)
    pattern_manager.bpm = max(1, min(300, int(round(float(preset_data.get("tempo", 120))))))
    pattern_manager.step_rate = preset_data.get("step_rate", "1/16")
    pattern_manager.fill_rate = max(1, int(round(float(preset_data.get("fill_rate", 4)))))
    pattern_manager.swing = float(preset_data.get("swing", 0.0))
    pattern_manager._update_step_duration()
    pattern_manager.load_from_preset_data(preset_data.get("patterns"))

    selected_pattern_index = _pattern_index_from_name(raw_data.get("Pattern", "A"))
    sequence = _pattern_sequence(pattern_manager, selected_pattern_index)

    rendered_steps: List[np.ndarray] = []
    rng = random.Random(0)

    for pattern_index in sequence:
        pattern_manager.start_playback(pattern_index)
        pattern = pattern_manager.get_playing_pattern()

        step_start_ms = [pattern_manager.get_step_time_ms(step) for step in range(pattern.length)]
        pattern_duration_ms = pattern.length * pattern_manager.step_duration_ms
        step_start_ms.append(pattern_duration_ms)

        for step in range(pattern.length):
            step_samples = int(round((step_start_ms[step + 1] - step_start_ms[step]) * sample_rate / 1000.0))
            events = _collect_step_events(pattern_manager, step, rng, sample_rate)
            rendered_steps.append(_render_step_audio(synth, step_samples, events))

    audio = np.concatenate(rendered_steps, axis=0) if rendered_steps else np.zeros((0, 2), dtype=np.float32)

    requested_samples = 0
    if duration_seconds is not None:
        requested_samples = max(0, int(math.ceil(duration_seconds * sample_rate)))

    if requested_samples > len(audio):
        tail = synth.process_audio(requested_samples - len(audio))
        audio = np.concatenate([audio, tail], axis=0)
    elif requested_samples and requested_samples < len(audio):
        audio = audio[:requested_samples]

    return audio.astype(np.float32, copy=False), synth, pattern_manager, preset_data


def get_preset_event_samples(preset_path: str, sample_rate: int = 44100) -> List[int]:
    """Return the scheduled trigger sample positions for the selected preset pattern."""

    parser = PythonicPresetParser()
    raw_data = parser.parse_file(preset_path)
    preset_data = parser.convert_to_synth_format(raw_data)

    pattern_manager = PatternManager(num_channels=8, pattern_length=16)
    pattern_manager.bpm = max(1, min(300, int(round(float(preset_data.get("tempo", 120))))))
    pattern_manager.step_rate = preset_data.get("step_rate", "1/16")
    pattern_manager.fill_rate = max(1, int(round(float(preset_data.get("fill_rate", 4)))))
    pattern_manager.swing = float(preset_data.get("swing", 0.0))
    pattern_manager._update_step_duration()
    pattern_manager.load_from_preset_data(preset_data.get("patterns"))

    selected_pattern_index = _pattern_index_from_name(raw_data.get("Pattern", "A"))
    sequence = _pattern_sequence(pattern_manager, selected_pattern_index)

    rng = random.Random(0)
    sample_cursor = 0
    scheduled: List[int] = []

    for pattern_index in sequence:
        pattern_manager.start_playback(pattern_index)
        pattern = pattern_manager.get_playing_pattern()

        step_start_ms = [pattern_manager.get_step_time_ms(step) for step in range(pattern.length)]
        pattern_duration_ms = pattern.length * pattern_manager.step_duration_ms
        step_start_ms.append(pattern_duration_ms)

        for step in range(pattern.length):
            step_samples = int(round((step_start_ms[step + 1] - step_start_ms[step]) * sample_rate / 1000.0))
            for offset, _channel_id, _velocity in _collect_step_events(pattern_manager, step, rng, sample_rate):
                scheduled.append(sample_cursor + offset)
            sample_cursor += step_samples

    return sorted(scheduled)