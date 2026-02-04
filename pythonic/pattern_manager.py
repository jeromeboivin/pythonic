"""
Pattern Manager for Pythonic
Handles pattern storage, manipulation, and playback
"""

import numpy as np
import copy
import json
from typing import List, Dict, Optional, Tuple
from enum import Enum


class PatternStep:
    """Represents a single step in a pattern"""
    def __init__(self):
        self.trigger = False      # Is there a trigger at this step?
        self.accent = False       # Is this step accented?
        self.fill = False         # Does this step have a fill?
        self.probability = 100    # Step probability (0-100%, default 100%)
        self.substeps = ""        # Sub-step pattern: 'o' = play, '-' = don't play (e.g., "oo-" or "o-o-")

    def __repr__(self):
        return f"Step(trig={self.trigger}, acc={self.accent}, fill={self.fill}, prob={self.probability}, sub={self.substeps})"

    def copy(self):
        """Create a deep copy of this step"""
        new_step = PatternStep()
        new_step.trigger = self.trigger
        new_step.accent = self.accent
        new_step.fill = self.fill
        new_step.probability = self.probability
        new_step.substeps = self.substeps
        return new_step


class PatternChannel:
    """Represents a single drum channel's pattern"""
    def __init__(self, channel_id: int, pattern_length: int = 16):
        self.channel_id = channel_id
        self.steps: List[PatternStep] = [PatternStep() for _ in range(pattern_length)]

    def get_step(self, index: int) -> PatternStep:
        """Get a step at index"""
        return self.steps[index % len(self.steps)]

    def set_trigger(self, index: int, value: bool):
        """Set trigger at index"""
        self.steps[index % len(self.steps)].trigger = value

    def set_accent(self, index: int, value: bool):
        """Set accent at index"""
        self.steps[index % len(self.steps)].accent = value

    def set_fill(self, index: int, value: bool):
        """Set fill at index"""
        self.steps[index % len(self.steps)].fill = value

    def set_probability(self, index: int, value: int):
        """Set probability at index (0-100)"""
        self.steps[index % len(self.steps)].probability = max(0, min(100, value))

    def get_triggers(self) -> List[bool]:
        """Get all trigger values as list"""
        return [step.trigger for step in self.steps]
    
    def get_accents(self) -> List[bool]:
        """Get all accent values as list"""
        return [step.accent for step in self.steps]
    
    def get_fills(self) -> List[bool]:
        """Get all fill values as list"""
        return [step.fill for step in self.steps]
    
    def get_probabilities(self) -> List[int]:
        """Get all probability values as list"""
        return [step.probability for step in self.steps]
    
    def set_triggers(self, triggers: List[bool]):
        """Set all trigger values from list"""
        for i, val in enumerate(triggers):
            if i < len(self.steps):
                self.steps[i].trigger = val
    
    def set_accents(self, accents: List[bool]):
        """Set all accent values from list"""
        for i, val in enumerate(accents):
            if i < len(self.steps):
                self.steps[i].accent = val
    
    def set_fills(self, fills: List[bool]):
        """Set all fill values from list"""
        for i, val in enumerate(fills):
            if i < len(self.steps):
                self.steps[i].fill = val
    
    def set_probabilities(self, probabilities: List[int]):
        """Set all probability values from list"""
        for i, val in enumerate(probabilities):
            if i < len(self.steps):
                self.steps[i].probability = max(0, min(100, val))

    def copy(self):
        """Create a deep copy of this channel"""
        new_channel = PatternChannel(self.channel_id, len(self.steps))
        new_channel.steps = [step.copy() for step in self.steps]
        return new_channel

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'channel_id': self.channel_id,
            'steps': [
                {
                    'trigger': step.trigger,
                    'accent': step.accent,
                    'fill': step.fill,
                    'probability': step.probability,
                    'substeps': step.substeps
                }
                for step in self.steps
            ]
        }

    @staticmethod
    def from_dict(data: Dict) -> 'PatternChannel':
        """Create from dictionary"""
        channel = PatternChannel(data['channel_id'], len(data['steps']))
        for i, step_data in enumerate(data['steps']):
            channel.steps[i].trigger = step_data['trigger']
            channel.steps[i].accent = step_data['accent']
            channel.steps[i].fill = step_data['fill']
            channel.steps[i].probability = step_data.get('probability', 100)
            channel.steps[i].substeps = step_data.get('substeps', '')
        return channel


class Pattern:
    """Represents a complete pattern for all 8 channels"""
    def __init__(self, name: str = "", pattern_length: int = 16, num_channels: int = 8):
        self.name = name
        self.length = pattern_length  # Pattern length in steps (16, 32, etc.)
        self.channels: List[PatternChannel] = [
            PatternChannel(i, pattern_length) for i in range(num_channels)
        ]
        self.chained_to_next = False  # Is this pattern chained to the next?
        self.chained_from_prev = False  # Is this pattern chained from the previous?

    def get_channel(self, channel_id: int) -> PatternChannel:
        """Get a channel by ID"""
        if 0 <= channel_id < len(self.channels):
            return self.channels[channel_id]
        return None

    def set_pattern_length(self, length: int):
        """Change the pattern length for all channels"""
        if length == self.length:
            return

        # If growing, add empty steps
        if length > self.length:
            for channel in self.channels:
                for _ in range(length - self.length):
                    channel.steps.append(PatternStep())
        # If shrinking, remove steps
        else:
            for channel in self.channels:
                channel.steps = channel.steps[:length]

        self.length = length
    
    def set_length(self, length: int):
        """Alias for set_pattern_length"""
        self.set_pattern_length(length)

    def copy(self) -> 'Pattern':
        """Create a deep copy of this pattern"""
        new_pattern = Pattern(self.name, self.length, len(self.channels))
        new_pattern.channels = [channel.copy() for channel in self.channels]
        new_pattern.chained_to_next = self.chained_to_next
        new_pattern.chained_from_prev = self.chained_from_prev
        return new_pattern

    def clear(self):
        """Clear all steps in all channels"""
        for channel in self.channels:
            for step in channel.steps:
                step.trigger = False
                step.accent = False
                step.fill = False

    def is_empty(self) -> bool:
        """Check if pattern has any triggers"""
        for channel in self.channels:
            for step in channel.steps:
                if step.trigger:
                    return False
        return True

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'length': self.length,
            'chained_to_next': self.chained_to_next,
            'chained_from_prev': self.chained_from_prev,
            'channels': [channel.to_dict() for channel in self.channels]
        }

    @staticmethod
    def from_dict(data: Dict) -> 'Pattern':
        """Create from dictionary"""
        pattern = Pattern(data.get('name', ''), data['length'], len(data['channels']))
        pattern.channels = [PatternChannel.from_dict(ch) for ch in data['channels']]
        pattern.chained_to_next = data.get('chained_to_next', False)
        pattern.chained_from_prev = data.get('chained_from_prev', False)
        return pattern


class PatternManager:
    """Manages all patterns (A-L)"""

    PATTERN_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']

    def __init__(self, num_channels: int = 8, pattern_length: int = 16):
        self.num_channels = num_channels
        self.pattern_length = pattern_length

        # Initialize 12 patterns (A-L)
        self.patterns: List[Pattern] = [
            Pattern(name, pattern_length, num_channels)
            for name in self.PATTERN_NAMES
        ]

        # Current selection
        self.selected_pattern_index = 0  # Currently selected for editing
        self.playing_pattern_index = 0   # Currently playing

        # Playback state
        self.is_playing = False
        self.play_position = 0  # 0-15 (or more depending on length)
        self.current_step = 0

        # Clipboard for copy/paste operations
        self.clipboard_pattern: Optional[Pattern] = None
        self.clipboard_channel: Optional[PatternChannel] = None

        # Fill rate (fills per step): 2-8
        self.fill_rate = 4

        # Tempo settings
        self.bpm = 120
        self.step_duration_ms = 250  # milliseconds per step at 120 BPM
        
        # Step rate: note subdivision for each pattern step
        # Values: '1/8', '1/8T', '1/16', '1/16T', '1/32'
        # Divisor is how many steps fit in one beat (quarter note)
        self.STEP_RATES = {
            '1/8': 2,      # 2 steps per beat (eighth notes)
            '1/8T': 3,     # 3 steps per beat (eighth note triplets)
            '1/16': 4,     # 4 steps per beat (sixteenth notes) - default
            '1/16T': 6,    # 6 steps per beat (sixteenth note triplets)
            '1/32': 8,     # 8 steps per beat (thirty-second notes)
        }
        self.step_rate = '1/16'  # Default to 1/16 notes
        
        # Swing: 0.0 = no swing (equal timing), ~0.67 = classic swing, range 0-1
        # 0% = no swing, 100% = max swing
        # Affects even-numbered steps (1, 3, 5, etc. in 0-indexed)
        self.swing = 0.0

        self._update_step_duration()

    def _update_step_duration(self):
        """Calculate step duration based on BPM and step rate"""
        # 60000 ms / bpm = duration of one beat (quarter note) in ms
        # Divide by step rate divisor to get duration of each step
        beat_duration_ms = 60000 / self.bpm
        step_divisor = self.STEP_RATES.get(self.step_rate, 4)
        self.step_duration_ms = beat_duration_ms / step_divisor

    def set_bpm(self, bpm: int):
        """Set the tempo in BPM"""
        self.bpm = max(1, min(300, bpm))  # Clamp 1-300
        self._update_step_duration()

    def set_step_rate(self, rate: str):
        """Set the step rate (note subdivision)"""
        if rate in self.STEP_RATES:
            self.step_rate = rate
            self._update_step_duration()

    def set_fill_rate(self, rate: int):
        """Set fill rate (2-8 times per step)"""
        self.fill_rate = max(2, min(8, rate))
    
    def set_swing(self, swing: float):
        """Set swing amount (0.0-1.0, where 0.5 = no swing)
        
        Swing delays even-indexed steps (1, 3, 5, ...) relative to the beat.
        At swing=0.5: even timing (no swing)
        At swing=0.67: classic shuffle/swing feel
        At swing=1.0: even steps pushed to just before the next odd step
        """
        self.swing = max(0.0, min(1.0, swing))
    
    def get_step_time_ms(self, step_index: int) -> float:
        """Get the time offset for a step in milliseconds, accounting for swing.
        
        Swing affects pairs of steps. In each pair (0,1), (2,3), (4,5)...:
        - First step (0, 2, 4, ...) plays at the expected time
        - Second step (1, 3, 5, ...) is delayed by swing amount
        
        Swing formula:
        - At swing=0: second step at 50% of pair duration (no swing)
        - At swing=1: second step at 75% of pair duration (max swing, triplet feel)
        
        Returns time offset from pattern start in ms.
        """
        pair_index = step_index // 2
        is_swung_step = (step_index % 2) == 1
        
        # Base time for the start of this pair
        pair_duration_ms = self.step_duration_ms * 2
        pair_start_ms = pair_index * pair_duration_ms
        
        if is_swung_step:
            # Swung step: delayed within the pair
            # swing=0 -> at 50% of pair duration (normal 1/16 position)
            # swing=1 -> at 90% of pair duration (maximum shuffle)
            # Linear interpolation: position = 0.5 + swing * 0.40
            # Calibrated from "Pedinner Guranodous (129)" reference recording:
            #   Step 1 (swing 0.6335) peaks at ~174ms = position 0.747
            #   Step 3 peaks at ~410ms = position 0.762 within pair
            swing_position = 0.5 + self.swing * 0.40
            return pair_start_ms + pair_duration_ms * swing_position
        else:
            # Non-swung step: at start of pair
            return pair_start_ms

    def get_pattern(self, index: int) -> Optional[Pattern]:
        """Get pattern by index (0-11)"""
        if 0 <= index < len(self.patterns):
            return self.patterns[index]
        return None

    def get_pattern_by_name(self, name: str) -> Optional[Pattern]:
        """Get pattern by name (A-L)"""
        try:
            index = self.PATTERN_NAMES.index(name)
            return self.get_pattern(index)
        except ValueError:
            return None

    def select_pattern(self, index: int):
        """Select a pattern for editing"""
        if 0 <= index < len(self.patterns):
            self.selected_pattern_index = index

    def get_selected_pattern(self) -> Pattern:
        """Get the currently selected pattern"""
        return self.patterns[self.selected_pattern_index]

    def get_playing_pattern(self) -> Pattern:
        """Get the currently playing pattern"""
        return self.patterns[self.playing_pattern_index]

    # ============ Pattern Operations ============

    def cut_pattern(self, pattern_index: int):
        """Cut pattern to clipboard"""
        self.clipboard_pattern = self.patterns[pattern_index].copy()
        self.patterns[pattern_index].clear()

    def cut_channel(self, pattern_index: int, channel_id: int):
        """Cut a channel to clipboard"""
        self.clipboard_channel = self.patterns[pattern_index].get_channel(channel_id).copy()
        self.patterns[pattern_index].get_channel(channel_id).steps = [
            PatternStep() for _ in range(self.pattern_length)
        ]

    def copy_pattern(self, pattern_index: int):
        """Copy pattern to clipboard"""
        if 0 <= pattern_index < len(self.patterns):
            self.clipboard_pattern = self.patterns[pattern_index].copy()

    def copy_channel(self, pattern_index: int, channel_id: int):
        """Copy a channel to clipboard"""
        channel = self.patterns[pattern_index].get_channel(channel_id)
        if channel:
            self.clipboard_channel = channel.copy()

    def paste_pattern(self, pattern_index: int) -> bool:
        """Paste pattern from clipboard, replacing current pattern"""
        if self.clipboard_pattern and 0 <= pattern_index < len(self.patterns):
            self.patterns[pattern_index] = self.clipboard_pattern.copy()
            self.patterns[pattern_index].name = self.PATTERN_NAMES[pattern_index]
            return True
        return False

    def paste_channel(self, pattern_index: int, channel_id: int) -> bool:
        """Paste a channel from clipboard"""
        if self.clipboard_channel and 0 <= pattern_index < len(self.patterns):
            new_channel = self.clipboard_channel.copy()
            new_channel.channel_id = channel_id
            self.patterns[pattern_index].channels[channel_id] = new_channel
            return True
        return False

    def exchange_pattern(self, pattern_index: int) -> bool:
        """Swap pattern with clipboard"""
        if self.clipboard_pattern and 0 <= pattern_index < len(self.patterns):
            temp = self.patterns[pattern_index].copy()
            self.patterns[pattern_index] = self.clipboard_pattern.copy()
            self.clipboard_pattern = temp
            return True
        return False

    def exchange_channel(self, pattern_index: int, channel_id: int) -> bool:
        """Swap channel with clipboard"""
        if self.clipboard_channel and 0 <= pattern_index < len(self.patterns):
            temp = self.patterns[pattern_index].get_channel(channel_id).copy()
            self.paste_channel(pattern_index, channel_id)
            self.clipboard_channel = temp
            return True
        return False

    def shift_pattern_left(self, pattern_index: int):
        """Shift entire pattern left by one step (with wrap)"""
        for channel in self.patterns[pattern_index].channels:
            first = channel.steps[0].copy()
            for i in range(len(channel.steps) - 1):
                channel.steps[i] = channel.steps[i + 1].copy()
            channel.steps[-1] = first

    def shift_pattern_right(self, pattern_index: int):
        """Shift entire pattern right by one step (with wrap)"""
        for channel in self.patterns[pattern_index].channels:
            last = channel.steps[-1].copy()
            for i in range(len(channel.steps) - 1, 0, -1):
                channel.steps[i] = channel.steps[i - 1].copy()
            channel.steps[0] = last

    def shift_channel_left(self, pattern_index: int, channel_id: int):
        """Shift channel left by one step"""
        channel = self.patterns[pattern_index].get_channel(channel_id)
        if channel:
            first = channel.steps[0].copy()
            for i in range(len(channel.steps) - 1):
                channel.steps[i] = channel.steps[i + 1].copy()
            channel.steps[-1] = first

    def shift_channel_right(self, pattern_index: int, channel_id: int):
        """Shift channel right by one step"""
        channel = self.patterns[pattern_index].get_channel(channel_id)
        if channel:
            last = channel.steps[-1].copy()
            for i in range(len(channel.steps) - 1, 0, -1):
                channel.steps[i] = channel.steps[i - 1].copy()
            channel.steps[0] = last

    def reverse_pattern(self, pattern_index: int):
        """Reverse entire pattern"""
        for channel in self.patterns[pattern_index].channels:
            channel.steps.reverse()

    def reverse_channel(self, pattern_index: int, channel_id: int):
        """Reverse a channel"""
        channel = self.patterns[pattern_index].get_channel(channel_id)
        if channel:
            channel.steps.reverse()

    def randomize_pattern(self, pattern_index: int, trigger_density: float = 0.5):
        """Randomize entire pattern"""
        pattern = self.patterns[pattern_index]
        for channel in pattern.channels:
            for step in channel.steps:
                step.trigger = np.random.random() < trigger_density
                step.accent = np.random.random() < 0.3 if step.trigger else False
                step.fill = np.random.random() < 0.2 if step.trigger else False

    def randomize_channel(self, pattern_index: int, channel_id: int, 
                         trigger_density: float = 0.5):
        """Randomize a channel"""
        channel = self.patterns[pattern_index].get_channel(channel_id)
        if channel:
            for step in channel.steps:
                step.trigger = np.random.random() < trigger_density
                step.accent = np.random.random() < 0.3 if step.trigger else False
                step.fill = np.random.random() < 0.2 if step.trigger else False

    def randomize_accents_fills(self, pattern_index: int):
        """Randomize only accents and fills"""
        pattern = self.patterns[pattern_index]
        for channel in pattern.channels:
            for step in channel.steps:
                if step.trigger:
                    step.accent = np.random.random() < 0.3
                    step.fill = np.random.random() < 0.2

    def randomize_channel_accents_fills(self, pattern_index: int, channel_id: int):
        """Randomize accents and fills for a channel"""
        channel = self.patterns[pattern_index].get_channel(channel_id)
        if channel:
            for step in channel.steps:
                if step.trigger:
                    step.accent = np.random.random() < 0.3
                    step.fill = np.random.random() < 0.2

    def alter_pattern(self, pattern_index: int, amount: float = 0.3):
        """Create variations by shuffling pattern parts"""
        pattern = self.patterns[pattern_index]
        if pattern.is_empty():
            return

        # Find all trigger positions
        for channel in pattern.channels:
            trigger_indices = [i for i, step in enumerate(channel.steps) if step.trigger]
            if not trigger_indices:
                continue

            # Randomly move some triggers
            num_to_move = max(1, int(len(trigger_indices) * amount))
            indices_to_move = np.random.choice(trigger_indices, num_to_move, replace=False)

            for idx in indices_to_move:
                # Clear old position
                channel.steps[idx].trigger = False
                # Find a new random position
                new_idx = np.random.randint(0, len(channel.steps))
                if not channel.steps[new_idx].trigger:
                    channel.steps[new_idx].trigger = True
                    channel.steps[new_idx].accent = np.random.random() < 0.3
                    channel.steps[new_idx].fill = np.random.random() < 0.2

    # ============ Playback ============

    def update_playback(self, frame_count: int, sample_rate: int):
        """Update playback position"""
        if not self.is_playing:
            return

        frames_per_step = (sample_rate * self.step_duration_ms) / 1000.0
        self.current_step += frame_count / frames_per_step

        while self.current_step >= 1.0:
            self.current_step -= 1.0
            self.play_position = (self.play_position + 1) % self.patterns[self.playing_pattern_index].length

    def get_triggers_at_step(self, step: int, pattern_index: Optional[int] = None) -> Dict[int, Tuple[bool, bool, bool]]:
        """
        Get all triggers at a step
        Returns dict of channel_id -> (trigger, accent, fill)
        """
        if pattern_index is None:
            pattern_index = self.playing_pattern_index

        pattern = self.patterns[pattern_index]
        result = {}

        for channel_id, channel in enumerate(pattern.channels):
            step_data = channel.get_step(step)
            result[channel_id] = (step_data.trigger, step_data.accent, step_data.fill)

        return result

    def start_playback(self, pattern_index: int = 0):
        """Start playback"""
        self.playing_pattern_index = pattern_index
        self.play_position = 0
        self.current_step = 0
        self.is_playing = True

    def stop_playback(self):
        """Stop playback"""
        self.is_playing = False
        self.play_position = 0
        self.current_step = 0

    def toggle_playback(self, pattern_index: int = 0):
        """Toggle playback on/off"""
        if self.is_playing:
            self.stop_playback()
        else:
            self.start_playback(pattern_index)

    def set_play_position(self, position: int):
        """Set the current playback position"""
        pattern = self.get_playing_pattern()
        self.play_position = position % pattern.length

    # ============ Pattern Chaining ============

    def chain_patterns(self, from_index: int, to_index: int):
        """
        Chain pattern at from_index to pattern at to_index.
        Patterns are always chained in alphabetical order (A->B, B->C, etc.).
        The 'to_index' must be from_index + 1.
        """
        if to_index != from_index + 1:
            return False
        if not (0 <= from_index < len(self.patterns) - 1):
            return False
        
        self.patterns[from_index].chained_to_next = True
        self.patterns[to_index].chained_from_prev = True
        return True

    def unchain_patterns(self, from_index: int, to_index: int):
        """
        Remove chain between pattern at from_index and pattern at to_index.
        """
        if to_index != from_index + 1:
            return False
        if not (0 <= from_index < len(self.patterns) - 1):
            return False
        
        self.patterns[from_index].chained_to_next = False
        self.patterns[to_index].chained_from_prev = False
        return True

    def toggle_chain_to_next(self, pattern_index: int) -> bool:
        """
        Toggle chain from pattern at pattern_index to the next pattern.
        Returns the new chain state.
        """
        if not (0 <= pattern_index < len(self.patterns) - 1):
            return False
        
        pattern = self.patterns[pattern_index]
        next_pattern = self.patterns[pattern_index + 1]
        
        if pattern.chained_to_next:
            pattern.chained_to_next = False
            next_pattern.chained_from_prev = False
        else:
            pattern.chained_to_next = True
            next_pattern.chained_from_prev = True
        
        return pattern.chained_to_next

    def toggle_chain_from_prev(self, pattern_index: int) -> bool:
        """
        Toggle chain from previous pattern to pattern at pattern_index.
        Returns the new chain state.
        """
        if not (0 < pattern_index < len(self.patterns)):
            return False
        
        pattern = self.patterns[pattern_index]
        prev_pattern = self.patterns[pattern_index - 1]
        
        if pattern.chained_from_prev:
            pattern.chained_from_prev = False
            prev_pattern.chained_to_next = False
        else:
            pattern.chained_from_prev = True
            prev_pattern.chained_to_next = True
        
        return pattern.chained_from_prev

    def get_chain_start(self, pattern_index: int) -> int:
        """
        Find the first pattern in the chain containing pattern at pattern_index.
        Returns the index of the first pattern in the chain.
        """
        current = pattern_index
        while current > 0 and self.patterns[current].chained_from_prev:
            current -= 1
        return current

    def get_chain_end(self, pattern_index: int) -> int:
        """
        Find the last pattern in the chain containing pattern at pattern_index.
        Returns the index of the last pattern in the chain.
        """
        current = pattern_index
        while current < len(self.patterns) - 1 and self.patterns[current].chained_to_next:
            current += 1
        return current

    def get_chain_patterns(self, pattern_index: int) -> List[int]:
        """
        Get all pattern indices in the chain containing pattern at pattern_index.
        Returns list of pattern indices in order.
        """
        start = self.get_chain_start(pattern_index)
        end = self.get_chain_end(pattern_index)
        return list(range(start, end + 1))

    def is_in_chain(self, pattern_index: int) -> bool:
        """Check if pattern at pattern_index is part of a chain."""
        pattern = self.patterns[pattern_index]
        return pattern.chained_to_next or pattern.chained_from_prev

    def get_next_pattern_in_chain(self, pattern_index: int) -> Optional[int]:
        """
        Get the next pattern in the chain after pattern at pattern_index.
        If at the end of a chain, returns the chain start (loop).
        If not in a chain, returns None (no chaining).
        
        Returns:
            Next pattern index to play, or None if pattern should not chain.
        """
        pattern = self.patterns[pattern_index]
        
        if pattern.chained_to_next and pattern_index < len(self.patterns) - 1:
            # Continue to next pattern in chain
            return pattern_index + 1
        elif self.is_in_chain(pattern_index):
            # At end of chain, loop back to start
            return self.get_chain_start(pattern_index)
        else:
            # Not in a chain, no automatic progression
            return None

    def advance_to_next_pattern(self) -> bool:
        """
        Advance playback to the next pattern in the chain.
        Call this when the current pattern has finished playing.
        
        Returns:
            True if advanced to a new pattern, False if staying on current or stopped.
        """
        if not self.is_playing:
            return False
        
        next_pattern = self.get_next_pattern_in_chain(self.playing_pattern_index)
        
        if next_pattern is not None:
            self.playing_pattern_index = next_pattern
            self.play_position = 0
            return True
        else:
            # Not in a chain - just loop current pattern
            self.play_position = 0
            return False

    # ============ Serialization ============

    def to_dict(self) -> Dict:
        """Serialize all patterns to dictionary"""
        return {
            'patterns': [p.to_dict() for p in self.patterns],
            'selected_pattern_index': self.selected_pattern_index,
            'playing_pattern_index': self.playing_pattern_index,
            'bpm': self.bpm,
            'fill_rate': self.fill_rate,
            'step_rate': self.step_rate,
        }

    def from_dict(self, data: Dict):
        """Load patterns from dictionary"""
        self.patterns = [Pattern.from_dict(p) for p in data['patterns']]
        self.selected_pattern_index = data.get('selected_pattern_index', 0)
        self.playing_pattern_index = data.get('playing_pattern_index', 0)
        self.bpm = data.get('bpm', 120)
        self.fill_rate = data.get('fill_rate', 4)
        self.step_rate = data.get('step_rate', '1/16')
        self._update_step_duration()

    def save_patterns(self, filepath: str):
        """Save all patterns to a JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def load_patterns(self, filepath: str):
        """Load patterns from a JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.from_dict(data)

    def load_from_preset_data(self, patterns_data: Dict):
        """Load patterns from preset data (parsed from .mtpreset file)"""
        if not patterns_data:
            return
        
        # patterns_data is a dict with keys 'A'-'L'
        pattern_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
        
        # First pass: load pattern data and set chained_to_next
        for i, pattern_name in enumerate(pattern_names):
            if pattern_name not in patterns_data:
                continue
            
            pattern_info = patterns_data[pattern_name]
            pattern = self.patterns[i]
            
            # Set pattern properties
            length = pattern_info.get('length', 16)
            pattern.set_length(length)
            pattern.chained_to_next = pattern_info.get('chained', False)
            pattern.chained_from_prev = False  # Reset, will be set in second pass
            
            # Load channel data
            channels_data = pattern_info.get('channels', {})
            for channel_id, channel_info in channels_data.items():
                if 0 <= channel_id < len(pattern.channels):
                    channel = pattern.channels[channel_id]
                    channel.set_triggers(channel_info['triggers'])
                    channel.set_accents(channel_info['accents'])
                    channel.set_fills(channel_info['fills'])
                    # Load probabilities if available (default to 100)
                    if 'probabilities' in channel_info:
                        channel.set_probabilities(channel_info['probabilities'])
                    # Load substeps if available (default to empty)
                    if 'substeps' in channel_info:
                        for i, substep_pattern in enumerate(channel_info['substeps']):
                            if i < len(channel.steps):
                                channel.steps[i].substeps = substep_pattern
        
        # Second pass: set chained_from_prev based on previous pattern's chained_to_next
        for i in range(1, len(self.patterns)):
            if self.patterns[i - 1].chained_to_next:
                self.patterns[i].chained_from_prev = True
