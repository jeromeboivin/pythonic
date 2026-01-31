"""
Stereo Delay/Echo Effect for Pythonic
Tempo-synced delay with standard musical timing options

Features:
- Tempo-synchronized delay times (1/1, 1/2, 1/4, 1/8, 1/16, 1/32, triplets, dotted)
- Stereo ping-pong mode
- Feedback control for repeating echoes
- Highly optimized using NumPy vectorization
"""

import numpy as np
from enum import IntEnum


class DelayTime(IntEnum):
    """Delay time presets based on musical divisions"""
    WHOLE = 0        # 1/1 - whole note
    HALF = 1         # 1/2 - half note
    QUARTER = 2      # 1/4 - quarter note
    EIGHTH = 3       # 1/8 - eighth note
    SIXTEENTH = 4    # 1/16 - sixteenth note
    THIRTYSECOND = 5 # 1/32 - thirty-second note
    HALF_T = 6       # 1/2T - half note triplet
    QUARTER_T = 7    # 1/4T - quarter note triplet
    EIGHTH_T = 8     # 1/8T - eighth note triplet
    SIXTEENTH_T = 9  # 1/16T - sixteenth note triplet
    HALF_D = 10      # 1/2. - dotted half note
    QUARTER_D = 11   # 1/4. - dotted quarter note
    EIGHTH_D = 12    # 1/8. - dotted eighth note
    SIXTEENTH_D = 13 # 1/16. - dotted sixteenth note


# Delay time multipliers relative to one beat (quarter note)
# Value represents number of beats for each delay time
DELAY_TIME_BEATS = {
    DelayTime.WHOLE: 4.0,
    DelayTime.HALF: 2.0,
    DelayTime.QUARTER: 1.0,
    DelayTime.EIGHTH: 0.5,
    DelayTime.SIXTEENTH: 0.25,
    DelayTime.THIRTYSECOND: 0.125,
    DelayTime.HALF_T: 4.0 / 3.0,      # Triplet = 2/3 of normal
    DelayTime.QUARTER_T: 2.0 / 3.0,
    DelayTime.EIGHTH_T: 1.0 / 3.0,
    DelayTime.SIXTEENTH_T: 0.5 / 3.0,
    DelayTime.HALF_D: 3.0,            # Dotted = 1.5x normal
    DelayTime.QUARTER_D: 1.5,
    DelayTime.EIGHTH_D: 0.75,
    DelayTime.SIXTEENTH_D: 0.375,
}

# Human-readable labels for UI
DELAY_TIME_LABELS = {
    DelayTime.WHOLE: "1/1",
    DelayTime.HALF: "1/2",
    DelayTime.QUARTER: "1/4",
    DelayTime.EIGHTH: "1/8",
    DelayTime.SIXTEENTH: "1/16",
    DelayTime.THIRTYSECOND: "1/32",
    DelayTime.HALF_T: "1/2T",
    DelayTime.QUARTER_T: "1/4T",
    DelayTime.EIGHTH_T: "1/8T",
    DelayTime.SIXTEENTH_T: "1/16T",
    DelayTime.HALF_D: "1/2.",
    DelayTime.QUARTER_D: "1/4.",
    DelayTime.EIGHTH_D: "1/8.",
    DelayTime.SIXTEENTH_D: "1/16.",
}


class StereoDelay:
    """
    Tempo-synced stereo delay effect
    
    Features:
    - Musical timing presets (1/4, 1/8, triplets, dotted, etc.)
    - Feedback for repeating echoes
    - Optional ping-pong stereo mode
    - Low CPU usage suitable for real-time synthesis
    """
    
    # Maximum delay time in seconds (for buffer allocation)
    MAX_DELAY_SECONDS = 4.0  # Supports up to 4 seconds delay
    
    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        
        # Parameters
        self._bpm = 120.0
        self._delay_time = DelayTime.EIGHTH
        self._feedback = 0.3      # 0.0 to 0.95 (capped to prevent runaway)
        self._mix = 0.0           # 0.0 = dry, 1.0 = fully wet
        self._ping_pong = False   # True for stereo ping-pong mode
        
        # Calculate max buffer size
        self._max_delay_samples = int(self.MAX_DELAY_SECONDS * sample_rate)
        
        # Delay buffers (circular)
        self._buffer_l = np.zeros(self._max_delay_samples, dtype=np.float32)
        self._buffer_r = np.zeros(self._max_delay_samples, dtype=np.float32)
        self._write_pos = 0
        
        # Current delay in samples (calculated from BPM and time setting)
        self._delay_samples = self._calculate_delay_samples()
    
    def _calculate_delay_samples(self) -> int:
        """Calculate delay time in samples based on BPM and time setting"""
        # Beats for this delay time setting
        beats = DELAY_TIME_BEATS.get(self._delay_time, 1.0)
        
        # Seconds per beat at current BPM
        seconds_per_beat = 60.0 / self._bpm
        
        # Total delay time in seconds
        delay_seconds = beats * seconds_per_beat
        
        # Convert to samples, capped to buffer size
        delay_samples = int(delay_seconds * self.sr)
        return min(delay_samples, self._max_delay_samples - 1)
    
    def set_bpm(self, bpm: float):
        """Set tempo in BPM"""
        self._bpm = max(20.0, min(300.0, bpm))
        self._delay_samples = self._calculate_delay_samples()
    
    def set_delay_time(self, delay_time: DelayTime):
        """Set delay time preset"""
        self._delay_time = delay_time
        self._delay_samples = self._calculate_delay_samples()
    
    def set_delay_time_index(self, index: int):
        """Set delay time by index (for UI)"""
        if 0 <= index < len(DelayTime):
            self._delay_time = DelayTime(index)
            self._delay_samples = self._calculate_delay_samples()
    
    def set_feedback(self, feedback: float):
        """Set feedback amount (0.0 to 0.95)"""
        self._feedback = np.clip(feedback, 0.0, 0.95)
    
    def set_mix(self, mix: float):
        """Set dry/wet mix (0.0 = dry, 1.0 = wet)"""
        self._mix = np.clip(mix, 0.0, 1.0)
    
    def set_ping_pong(self, enabled: bool):
        """Enable/disable ping-pong stereo mode"""
        self._ping_pong = enabled
    
    def reset(self):
        """Reset delay buffers"""
        self._buffer_l.fill(0)
        self._buffer_r.fill(0)
        self._write_pos = 0
    
    def get_delay_time_label(self) -> str:
        """Get human-readable label for current delay time"""
        return DELAY_TIME_LABELS.get(self._delay_time, "1/8")
    
    def process(self, input_signal: np.ndarray) -> np.ndarray:
        """
        Process stereo signal through delay
        
        Args:
            input_signal: Stereo input array [num_samples, 2]
            
        Returns:
            Processed stereo array [num_samples, 2]
        """
        if self._mix < 0.001:
            return input_signal
        
        num_samples = len(input_signal)
        in_l = input_signal[:, 0]
        in_r = input_signal[:, 1]
        
        # Pre-allocate output
        wet_l = np.zeros(num_samples, dtype=np.float32)
        wet_r = np.zeros(num_samples, dtype=np.float32)
        
        delay = self._delay_samples
        feedback = self._feedback
        buffer_size = self._max_delay_samples
        
        if self._ping_pong:
            # Ping-pong mode: L echoes to R, R echoes to L
            # Process sample by sample for feedback
            for n in range(num_samples):
                # Read from delay buffers
                read_pos = (self._write_pos - delay) % buffer_size
                delayed_l = self._buffer_l[read_pos]
                delayed_r = self._buffer_r[read_pos]
                
                # Output delayed signals
                wet_l[n] = delayed_r  # Right -> Left (ping-pong)
                wet_r[n] = delayed_l  # Left -> Right (ping-pong)
                
                # Write input + feedback to buffers (cross-feed for ping-pong)
                self._buffer_l[self._write_pos] = in_l[n] + delayed_r * feedback
                self._buffer_r[self._write_pos] = in_r[n] + delayed_l * feedback
                
                self._write_pos = (self._write_pos + 1) % buffer_size
        else:
            # Standard stereo delay: L->L, R->R
            for n in range(num_samples):
                read_pos = (self._write_pos - delay) % buffer_size
                delayed_l = self._buffer_l[read_pos]
                delayed_r = self._buffer_r[read_pos]
                
                wet_l[n] = delayed_l
                wet_r[n] = delayed_r
                
                # Write input + feedback
                self._buffer_l[self._write_pos] = in_l[n] + delayed_l * feedback
                self._buffer_r[self._write_pos] = in_r[n] + delayed_r * feedback
                
                self._write_pos = (self._write_pos + 1) % buffer_size
        
        # Mix dry and wet
        dry_gain = 1.0 - self._mix * 0.5  # Keep some dry even at 100% wet
        wet_gain = self._mix
        
        output = np.empty_like(input_signal)
        output[:, 0] = in_l * dry_gain + wet_l * wet_gain
        output[:, 1] = in_r * dry_gain + wet_r * wet_gain
        
        return output


class FastStereoDelay:
    """
    Optimized stereo delay using vectorized operations
    
    Uses chunk-based processing for better performance.
    Slightly less accurate feedback but much faster.
    """
    
    MAX_DELAY_SECONDS = 4.0
    
    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        
        # Parameters
        self._bpm = 120.0
        self._delay_time = DelayTime.EIGHTH
        self._feedback = 0.3
        self._mix = 0.0
        self._ping_pong = False
        
        # Buffer
        self._max_delay_samples = int(self.MAX_DELAY_SECONDS * sample_rate)
        self._buffer_l = np.zeros(self._max_delay_samples, dtype=np.float32)
        self._buffer_r = np.zeros(self._max_delay_samples, dtype=np.float32)
        self._write_pos = 0
        
        self._delay_samples = self._calculate_delay_samples()
    
    def _calculate_delay_samples(self) -> int:
        beats = DELAY_TIME_BEATS.get(self._delay_time, 1.0)
        seconds_per_beat = 60.0 / self._bpm
        delay_seconds = beats * seconds_per_beat
        delay_samples = int(delay_seconds * self.sr)
        return min(delay_samples, self._max_delay_samples - 1)
    
    def set_bpm(self, bpm: float):
        self._bpm = max(20.0, min(300.0, bpm))
        self._delay_samples = self._calculate_delay_samples()
    
    def set_delay_time(self, delay_time: DelayTime):
        self._delay_time = delay_time
        self._delay_samples = self._calculate_delay_samples()
    
    def set_delay_time_index(self, index: int):
        if 0 <= index < len(DelayTime):
            self._delay_time = DelayTime(index)
            self._delay_samples = self._calculate_delay_samples()
    
    def set_feedback(self, feedback: float):
        self._feedback = np.clip(feedback, 0.0, 0.95)
    
    def set_mix(self, mix: float):
        self._mix = np.clip(mix, 0.0, 1.0)
    
    def set_ping_pong(self, enabled: bool):
        self._ping_pong = enabled
    
    def reset(self):
        self._buffer_l.fill(0)
        self._buffer_r.fill(0)
        self._write_pos = 0
    
    def get_delay_time_label(self) -> str:
        return DELAY_TIME_LABELS.get(self._delay_time, "1/8")
    
    def process(self, input_signal: np.ndarray) -> np.ndarray:
        """Vectorized delay processing"""
        if self._mix < 0.001:
            return input_signal
        
        num_samples = len(input_signal)
        in_l = input_signal[:, 0]
        in_r = input_signal[:, 1]
        
        delay = self._delay_samples
        buffer_size = self._max_delay_samples
        
        # Calculate read indices for all samples at once
        read_indices = (self._write_pos + np.arange(num_samples) - delay) % buffer_size
        write_indices = (self._write_pos + np.arange(num_samples)) % buffer_size
        
        # Read delayed samples
        delayed_l = self._buffer_l[read_indices]
        delayed_r = self._buffer_r[read_indices]
        
        if self._ping_pong:
            wet_l = delayed_r
            wet_r = delayed_l
            # Write with cross-feedback (approximated - single pass)
            self._buffer_l[write_indices] = in_l + delayed_r * self._feedback
            self._buffer_r[write_indices] = in_r + delayed_l * self._feedback
        else:
            wet_l = delayed_l
            wet_r = delayed_r
            self._buffer_l[write_indices] = in_l + delayed_l * self._feedback
            self._buffer_r[write_indices] = in_r + delayed_r * self._feedback
        
        # Update write position
        self._write_pos = (self._write_pos + num_samples) % buffer_size
        
        # Mix
        dry_gain = 1.0 - self._mix * 0.5
        wet_gain = self._mix
        
        output = np.empty_like(input_signal)
        output[:, 0] = in_l * dry_gain + wet_l * wet_gain
        output[:, 1] = in_r * dry_gain + wet_r * wet_gain
        
        return output
