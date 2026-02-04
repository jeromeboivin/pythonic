"""
MIDI Input Manager for Pythonic
Handles MIDI input for drum triggering and pattern control
"""

import threading
import time
from typing import Optional, Callable, List, Dict, Any
from enum import Enum

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False

try:
    import rtmidi
    RTMIDI_AVAILABLE = True
except ImportError:
    RTMIDI_AVAILABLE = False


class MidiManager:
    """
    Manages MIDI input for Pythonic.
    
    Features:
    - Note-on messages trigger individual drum channels (C1-G1 by default = notes 36-43)
    - MIDI Start/Stop/Continue control pattern playback
    - Program Change selects patterns A-L (0-11)
    - MIDI Clock sync for BPM (24 pulses per quarter note)
    - Configurable base note for drum mapping
    - Auto-connects to first available MIDI input device
    """
    
    # Default MIDI note mapping: C1 (36) is channel 0, up to G1 (43) for channel 7
    DEFAULT_BASE_NOTE = 36  # C1 in MIDI
    NUM_CHANNELS = 8
    NUM_PATTERNS = 12  # A-L
    MIDI_CLOCK_PPQN = 24  # Pulses per quarter note
    MAX_CC_MAPPINGS = 8  # Maximum number of CC mappings
    
    # Common CC numbers
    CC_MOD_WHEEL = 1
    CC_BREATH = 2
    CC_FOOT = 4
    CC_EXPRESSION = 11
    CC_EFFECT1 = 12
    CC_EFFECT2 = 13
    CC_GENERAL1 = 16
    CC_GENERAL2 = 17
    
    def __init__(self):
        self.enabled = MIDO_AVAILABLE
        self.base_note = self.DEFAULT_BASE_NOTE
        
        # MIDI input port
        self._input_port: Optional[Any] = None
        self._port_name: Optional[str] = None
        
        # Callbacks
        self._on_drum_trigger: Optional[Callable[[int, int], None]] = None  # (channel, velocity)
        self._on_pattern_select: Optional[Callable[[int], None]] = None  # (pattern_index)
        self._on_transport_start: Optional[Callable[[], None]] = None
        self._on_transport_stop: Optional[Callable[[], None]] = None
        self._on_transport_continue: Optional[Callable[[], None]] = None
        self._on_midi_activity: Optional[Callable[[], None]] = None  # For visual feedback
        self._on_bpm_change: Optional[Callable[[float], None]] = None  # For BPM sync
        self._on_cc_change: Optional[Callable[[int, int], None]] = None  # (cc_number, value)
        self._on_pitchbend_change: Optional[Callable[[float], None]] = None  # (normalized -1.0 to 1.0)
        
        # Pitch bend state
        self._pitchbend_enabled = True
        self._pitchbend_target: Optional[str] = None  # Parameter name to control with pitch bend
        self._current_pitchbend = 0.0  # Current normalized pitch bend value
        
        # Thread for polling MIDI input
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # Activity tracking
        self._last_activity_time = 0
        
        # MIDI Clock sync state
        self._clock_sync_enabled = True
        self._clock_count = 0
        self._last_clock_time = 0.0
        self._clock_times: List[float] = []  # Rolling window of clock timestamps
        self._clock_window_size = 192  # Average over 8 quarter notes for stability
        self._current_bpm = 0.0
        self._reported_bpm = 0.0  # The BPM we last reported to callback (with hysteresis)
        self._min_bpm = 20.0   # Ignore unrealistic BPM values
        self._max_bpm = 300.0
        self._clock_timeout = 0.5  # Reset clock buffer if no clock for 500ms
        self._bpm_hysteresis = 1.0  # BPM must change by 1+ to trigger update (allows 0.1 precision after rounding)
        self._clock_update_interval = 48  # Check BPM every 48 clocks (2 beats)
        self._clocks_since_update = 0
        
        # CC Mapping state
        self._cc_mappings: Dict[int, str] = {}  # CC number -> parameter name
        self._midi_learn_active = False
        self._midi_learn_callback: Optional[Callable[[int], None]] = None  # Called when CC learned
        
    @property
    def is_connected(self) -> bool:
        """Check if a MIDI input is connected"""
        return self._input_port is not None
    
    @property
    def port_name(self) -> Optional[str]:
        """Get the name of the connected MIDI port"""
        return self._port_name
    
    def get_available_ports(self) -> List[str]:
        """Get list of available MIDI input ports"""
        if not MIDO_AVAILABLE:
            return []
        try:
            return mido.get_input_names()
        except Exception as e:
            print(f"Error getting MIDI ports: {e}")
            return []
    
    def connect(self, port_name: Optional[str] = None) -> bool:
        """
        Connect to a MIDI input port.
        
        Args:
            port_name: Name of the port to connect to. If None, connects to first available.
            
        Returns:
            True if connection successful, False otherwise.
        """
        if not MIDO_AVAILABLE:
            print("MIDI not available (mido library not installed)")
            return False
        
        # Disconnect existing port
        self.disconnect()
        
        try:
            available_ports = self.get_available_ports()
            if not available_ports:
                print("No MIDI input ports available")
                return False
            
            # Select port
            if port_name is None:
                port_name = available_ports[0]
            elif port_name not in available_ports:
                print(f"MIDI port '{port_name}' not found")
                return False
            
            # Open the port
            self._input_port = mido.open_input(port_name)
            self._port_name = port_name
            
            # Start the polling thread
            self._running = True
            self._thread = threading.Thread(target=self._midi_loop, daemon=True)
            self._thread.start()
            
            print(f"Connected to MIDI input: {port_name}")
            return True
            
        except Exception as e:
            print(f"Failed to connect to MIDI input: {e}")
            self._input_port = None
            self._port_name = None
            return False
    
    def disconnect(self):
        """Disconnect from the current MIDI input port"""
        self._running = False
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        
        if self._input_port:
            try:
                self._input_port.close()
            except Exception:
                pass
            self._input_port = None
            self._port_name = None
    
    def auto_connect(self) -> bool:
        """Auto-connect to the first available MIDI input port"""
        return self.connect(None)
    
    def set_base_note(self, base_note: int):
        """
        Set the base MIDI note for drum channel mapping.
        
        Args:
            base_note: MIDI note number (0-127) for channel 0.
                       Channels 0-7 will be mapped to base_note through base_note+7.
        """
        self.base_note = max(0, min(120, base_note))  # Ensure room for 8 channels
    
    def get_base_note(self) -> int:
        """Get the current base note"""
        return self.base_note
    
    def note_to_channel(self, note: int) -> Optional[int]:
        """
        Convert a MIDI note to a drum channel index.
        
        Returns:
            Channel index (0-7) or None if note is outside the mapped range.
        """
        channel = note - self.base_note
        if 0 <= channel < self.NUM_CHANNELS:
            return channel
        return None
    
    def channel_to_note(self, channel: int) -> int:
        """Convert a drum channel index to its MIDI note"""
        return self.base_note + channel
    
    # Callback setters
    def set_drum_trigger_callback(self, callback: Callable[[int, int], None]):
        """Set callback for drum triggers: callback(channel_index, velocity)"""
        self._on_drum_trigger = callback
    
    def set_pattern_select_callback(self, callback: Callable[[int], None]):
        """Set callback for pattern selection: callback(pattern_index)"""
        self._on_pattern_select = callback
    
    def set_transport_callbacks(self, 
                                on_start: Callable[[], None],
                                on_stop: Callable[[], None],
                                on_continue: Optional[Callable[[], None]] = None):
        """Set callbacks for transport control"""
        self._on_transport_start = on_start
        self._on_transport_stop = on_stop
        self._on_transport_continue = on_continue or on_start
    
    def set_activity_callback(self, callback: Callable[[], None]):
        """Set callback for MIDI activity indication"""
        self._on_midi_activity = callback
    
    def set_bpm_callback(self, callback: Callable[[float], None]):
        """Set callback for BPM changes from MIDI clock sync: callback(bpm)"""
        self._on_bpm_change = callback
    
    def set_cc_callback(self, callback: Callable[[int, int], None]):
        """Set callback for CC changes: callback(cc_number, value)"""
        self._on_cc_change = callback
    
    def set_pitchbend_callback(self, callback: Callable[[float], None]):
        """Set callback for pitch bend changes: callback(normalized_value)
        
        Args:
            callback: Called with normalized pitch bend value (-1.0 to 1.0)
        """
        self._on_pitchbend_change = callback
    
    def set_pitchbend_enabled(self, enabled: bool):
        """Enable or disable pitch bend processing"""
        self._pitchbend_enabled = enabled
    
    def is_pitchbend_enabled(self) -> bool:
        """Check if pitch bend is enabled"""
        return self._pitchbend_enabled
    
    def set_pitchbend_target(self, parameter_name: Optional[str]):
        """Set the parameter to control with pitch bend.
        
        Args:
            parameter_name: Name of the parameter to control, or None to disable
        """
        self._pitchbend_target = parameter_name
    
    def get_pitchbend_target(self) -> Optional[str]:
        """Get the parameter name controlled by pitch bend"""
        return self._pitchbend_target
    
    def get_current_pitchbend(self) -> float:
        """Get the current pitch bend value (-1.0 to 1.0)"""
        return self._current_pitchbend
    
    def set_clock_sync_enabled(self, enabled: bool):
        """Enable or disable MIDI clock sync"""
        self._clock_sync_enabled = enabled
        if not enabled:
            self._reset_clock_sync()
    
    def is_clock_sync_enabled(self) -> bool:
        """Check if clock sync is enabled"""
        return self._clock_sync_enabled
    
    def get_synced_bpm(self) -> float:
        """Get the current synced BPM (0 if not synced)"""
        return self._reported_bpm
    
    def _reset_clock_sync(self, full_reset: bool = False):
        """Reset clock sync state.
        
        Args:
            full_reset: If True, also reset BPM values to force re-sync.
        """
        self._clock_count = 0
        self._last_clock_time = 0.0
        self._clock_times.clear()
        self._clocks_since_update = 0
        if full_reset:
            self._current_bpm = 0.0
            self._reported_bpm = 0.0
    
    # ============ CC Mapping Methods ============
    
    def set_cc_mappings(self, mappings: Dict[int, str]):
        """
        Set CC mappings from preferences.
        
        Args:
            mappings: Dictionary of {cc_number: parameter_name}
        """
        self._cc_mappings = mappings.copy()
    
    def get_cc_mappings(self) -> Dict[int, str]:
        """Get current CC mappings"""
        return self._cc_mappings.copy()
    
    def add_cc_mapping(self, cc_number: int, parameter_name: str) -> bool:
        """
        Add a CC mapping.
        
        Returns:
            True if mapping was added, False if max mappings reached.
        """
        if len(self._cc_mappings) >= self.MAX_CC_MAPPINGS and cc_number not in self._cc_mappings:
            return False
        self._cc_mappings[cc_number] = parameter_name
        return True
    
    def remove_cc_mapping(self, cc_number: int):
        """Remove a CC mapping"""
        if cc_number in self._cc_mappings:
            del self._cc_mappings[cc_number]
    
    def clear_cc_mappings(self):
        """Clear all CC mappings"""
        self._cc_mappings.clear()
    
    def get_parameter_for_cc(self, cc_number: int) -> Optional[str]:
        """Get the parameter name mapped to a CC number"""
        return self._cc_mappings.get(cc_number)
    
    def get_cc_for_parameter(self, parameter_name: str) -> Optional[int]:
        """Get the CC number mapped to a parameter"""
        for cc, param in self._cc_mappings.items():
            if param == parameter_name:
                return cc
        return None
    
    # ============ MIDI Learn Methods ============
    
    def start_midi_learn(self, callback: Callable[[int], None]):
        """
        Start MIDI learn mode.
        
        Args:
            callback: Called with the CC number when a CC message is received.
        """
        self._midi_learn_active = True
        self._midi_learn_callback = callback
    
    def stop_midi_learn(self):
        """Stop MIDI learn mode"""
        self._midi_learn_active = False
        self._midi_learn_callback = None
    
    def is_midi_learn_active(self) -> bool:
        """Check if MIDI learn is active"""
        return self._midi_learn_active

    def _midi_loop(self):
        """Background thread that polls for MIDI messages"""
        while self._running and self._input_port:
            try:
                # Non-blocking poll for messages
                for msg in self._input_port.iter_pending():
                    self._handle_message(msg)
                
                # Small sleep to prevent CPU spinning
                time.sleep(0.001)  # 1ms polling interval
                
            except Exception as e:
                print(f"MIDI loop error: {e}")
                time.sleep(0.1)
    
    def _handle_message(self, msg):
        """Handle an incoming MIDI message"""
        # Trigger activity callback for visual feedback
        self._last_activity_time = time.time()
        if self._on_midi_activity:
            self._on_midi_activity()
        
        # Note On -> Drum trigger
        if msg.type == 'note_on' and msg.velocity > 0:
            channel = self.note_to_channel(msg.note)
            if channel is not None and self._on_drum_trigger:
                self._on_drum_trigger(channel, msg.velocity)
        
        # Note Off with velocity (some controllers send this)
        elif msg.type == 'note_off':
            pass  # Drums are one-shot, no need to handle note off
        
        # Program Change -> Pattern selection
        elif msg.type == 'program_change':
            if msg.program < self.NUM_PATTERNS and self._on_pattern_select:
                self._on_pattern_select(msg.program)
        
        # Transport controls (MIDI realtime messages)
        elif msg.type == 'start':
            self._reset_clock_sync(full_reset=True)  # Full reset to re-sync BPM
            if self._on_transport_start:
                self._on_transport_start()
        
        elif msg.type == 'stop':
            if self._on_transport_stop:
                self._on_transport_stop()
        
        elif msg.type == 'continue':
            if self._on_transport_continue:
                self._on_transport_continue()
        
        # MIDI Clock -> BPM sync
        elif msg.type == 'clock':
            self._handle_clock_message()
        
        # Control Change (CC) -> Parameter control or MIDI Learn
        elif msg.type == 'control_change':
            self._handle_cc_message(msg.control, msg.value)
        
        # Pitch Bend -> Parameter control
        elif msg.type == 'pitchwheel':
            self._handle_pitchbend_message(msg.pitch)
    
    def _handle_pitchbend_message(self, pitch: int):
        """
        Handle a MIDI Pitch Bend message.
        
        Args:
            pitch: Raw pitch bend value (-8192 to 8191)
        """
        if not self._pitchbend_enabled:
            return
        
        # Normalize to -1.0 to 1.0 range
        # Note: pitch range is -8192 to 8191, so we divide by 8192 for symmetry
        self._current_pitchbend = pitch / 8192.0
        
        # Call the pitch bend callback if set
        if self._on_pitchbend_change:
            self._on_pitchbend_change(self._current_pitchbend)
    
    def _handle_cc_message(self, cc_number: int, value: int):
        """
        Handle a MIDI Control Change message.
        
        Args:
            cc_number: CC number (0-127)
            value: CC value (0-127)
        """
        # If MIDI Learn is active, capture this CC and stop learning
        if self._midi_learn_active and self._midi_learn_callback:
            self._midi_learn_callback(cc_number)
            self._midi_learn_active = False
            self._midi_learn_callback = None
            return
        
        # Check if this CC is mapped to a parameter
        if cc_number in self._cc_mappings:
            if self._on_cc_change:
                self._on_cc_change(cc_number, value)
    
    def _handle_clock_message(self):
        """
        Handle MIDI clock message for BPM sync.
        
        MIDI clock sends 24 pulses per quarter note (PPQN).
        We calculate BPM by measuring the time between clock messages.
        Uses a large window for accuracy but can report initial BPM faster
        with a smaller minimum sample size.
        """
        if not self._clock_sync_enabled:
            return
        
        current_time = time.perf_counter()  # High-resolution timer
        
        # Check for clock timeout - if too much time has passed since last clock,
        # reset the buffer to avoid stale timestamps causing instability
        if self._clock_times and (current_time - self._clock_times[-1]) > self._clock_timeout:
            self._clock_times.clear()
            self._clocks_since_update = 0
        
        # Store timestamp
        self._clock_times.append(current_time)
        self._clocks_since_update += 1
        
        # Keep only the last N clock times for averaging
        while len(self._clock_times) > self._clock_window_size:
            self._clock_times.pop(0)
        
        # Only calculate BPM periodically to reduce overhead
        if self._clocks_since_update < self._clock_update_interval:
            return
        self._clocks_since_update = 0
        
        # Need minimum 48 clocks (2 beats) for initial reading, prefer full buffer for accuracy
        min_samples = 48
        if len(self._clock_times) >= min_samples:
            # Calculate average time between clocks
            time_span = self._clock_times[-1] - self._clock_times[0]
            num_intervals = len(self._clock_times) - 1
            
            if time_span > 0 and num_intervals > 0:
                avg_clock_interval = time_span / num_intervals
                
                # Convert to BPM: 24 clocks per quarter note
                # BPM = 60 / (avg_clock_interval * 24)
                raw_bpm = 60.0 / (avg_clock_interval * self.MIDI_CLOCK_PPQN)
                
                # Validate BPM is in reasonable range
                if self._min_bpm <= raw_bpm <= self._max_bpm:
                    # Round to nearest integer
                    rounded_bpm = round(raw_bpm)
                    self._current_bpm = rounded_bpm
                    
                    # Use hysteresis: only report if BPM changed significantly from last REPORTED value
                    if self._reported_bpm == 0 or abs(rounded_bpm - self._reported_bpm) >= self._bpm_hysteresis:
                        self._reported_bpm = rounded_bpm
                        
                        # Call callback if set
                        if self._on_bpm_change:
                            self._on_bpm_change(self._reported_bpm)
    
    def get_note_name(self, note: int) -> str:
        """Convert MIDI note number to note name (e.g., 36 -> 'C1')"""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (note // 12) - 1
        name = note_names[note % 12]
        return f"{name}{octave}"
    
    def get_mapping_description(self) -> str:
        """Get a human-readable description of the current note mapping"""
        start_note = self.get_note_name(self.base_note)
        end_note = self.get_note_name(self.base_note + self.NUM_CHANNELS - 1)
        return f"{start_note} - {end_note} (notes {self.base_note}-{self.base_note + self.NUM_CHANNELS - 1})"
    
    def cleanup(self):
        """Clean up resources"""
        self.disconnect()


# Common CC names for display
CC_NAMES = {
    1: "Mod Wheel",
    2: "Breath Controller",
    4: "Foot Controller",
    7: "Volume",
    10: "Pan",
    11: "Expression",
    12: "Effect Ctrl 1",
    13: "Effect Ctrl 2",
    16: "General Purpose 1",
    17: "General Purpose 2",
    18: "General Purpose 3",
    19: "General Purpose 4",
    64: "Sustain Pedal",
    65: "Portamento",
    71: "Resonance/Timbre",
    74: "Brightness/Cutoff",
    91: "Reverb",
    93: "Chorus",
}


def get_cc_name(cc_number: int) -> str:
    """Get the name of a CC number"""
    if cc_number in CC_NAMES:
        return f"CC{cc_number} ({CC_NAMES[cc_number]})"
    return f"CC{cc_number}"


# Common base note presets
BASE_NOTE_PRESETS = {
    'C1 (GM Kick)': 36,
    'C2': 48,
    'C3 (Middle C)': 60,
    'C4': 72,
}


def get_base_note_options() -> List[tuple]:
    """
    Get list of base note options for UI.
    
    Returns:
        List of (display_name, midi_note) tuples
    """
    options = []
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Generate options for octaves -1 through 8
    for octave in range(-1, 9):
        for i, name in enumerate(note_names):
            midi_note = (octave + 1) * 12 + i
            if 0 <= midi_note <= 120:  # Leave room for 8 channels
                display = f"{name}{octave} (note {midi_note})"
                options.append((display, midi_note))
    
    return options
