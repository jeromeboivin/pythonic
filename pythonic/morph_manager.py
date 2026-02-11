"""
Morph Manager for Pythonic

Handles smooth interpolation between two drum patch endpoint states (A and B)
across all eight channels simultaneously using a single morph slider.

Morph behavior:
- Slider at 0.0 = fully endpoint A, slider at 1.0 = fully endpoint B
- Moving the slider interpolates all numeric drum parameters between endpoints
- Discrete parameters (waveform, filter mode, etc.) snap at the midpoint
- Editing a patch adjusts both endpoints proportionally based on slider position:
  - Slider at 0.0: only A endpoint is changed
  - Slider at 1.0: only B endpoint is changed
  - Slider at 0.5: both endpoints change equally
- Morph can be MIDI-controlled and automated
"""

import copy
import numpy as np
from typing import Dict, List, Any, Optional


# Parameters that can be linearly interpolated
INTERPOLATABLE_PARAMS = {
    'osc_frequency', 'pitch_mod_amount', 'pitch_mod_rate',
    'osc_attack', 'osc_decay',
    'noise_filter_freq', 'noise_filter_q',
    'noise_attack', 'noise_decay',
    'osc_noise_mix', 'distortion',
    'eq_frequency', 'eq_gain_db',
    'level_db', 'pan',
    'osc_vel_sensitivity', 'noise_vel_sensitivity', 'mod_vel_sensitivity',
    'vintage_amount',
    'reverb_decay', 'reverb_mix', 'reverb_width',
    'delay_feedback', 'delay_mix',
}

# Parameters that use logarithmic interpolation (frequencies)
LOG_INTERPOLATABLE_PARAMS = {
    'osc_frequency', 'noise_filter_freq', 'eq_frequency',
    'pitch_mod_rate',
}

# Discrete params: use nearest-endpoint value (snap at midpoint)
DISCRETE_PARAMS = {
    'osc_waveform', 'pitch_mod_mode', 'noise_filter_mode',
    'noise_envelope_mode', 'noise_stereo', 'choke_enabled',
    'output_pair', 'delay_time', 'delay_ping_pong',
}

# Parameters to NOT interpolate (identity/metadata)
EXCLUDED_PARAMS = {'name'}


class MorphManager:
    """
    Manages morph interpolation between two endpoint states for all 8 drum channels.
    
    Usage:
        morph = MorphManager(synthesizer)
        morph.capture_endpoint_a()  # Store current state as endpoint A
        morph.capture_endpoint_b()  # Store current state as endpoint B
        morph.set_position(0.5)     # Interpolate halfway
    """
    
    NUM_CHANNELS = 8
    
    def __init__(self, synthesizer):
        """
        Args:
            synthesizer: PythonicSynthesizer instance
        """
        self.synth = synthesizer
        
        # Morph position: 0.0 = fully A, 1.0 = fully B
        self._position = 0.0
        
        # Endpoint states: list of 8 channel parameter dicts each
        self._endpoint_a: Optional[List[Dict]] = None
        self._endpoint_b: Optional[List[Dict]] = None
        
        # Learn mode: 'a', 'b', or None
        self._learn_mode: Optional[str] = None
        
        # Flag to prevent re-entrant updates
        self._updating = False
        
        # Initialize endpoints from current state
        self._init_endpoints()
    
    def _init_endpoints(self):
        """Initialize both endpoints from current synthesizer state."""
        current = self._capture_all_channels()
        self._endpoint_a = copy.deepcopy(current)
        self._endpoint_b = copy.deepcopy(current)
    
    def _capture_all_channels(self) -> List[Dict]:
        """Capture parameters from all 8 channels."""
        return [ch.get_parameters() for ch in self.synth.channels]
    
    # ============ Learn Mode ============
    
    def start_learn_a(self):
        """Start learning mode for endpoint A.
        All parameter changes will be captured as endpoint A."""
        self._learn_mode = 'a'
    
    def start_learn_b(self):
        """Start learning mode for endpoint B.
        All parameter changes will be captured as endpoint B."""
        self._learn_mode = 'b'
    
    def stop_learn(self):
        """Stop learning mode."""
        if self._learn_mode == 'a':
            self.capture_endpoint_a()
        elif self._learn_mode == 'b':
            self.capture_endpoint_b()
        self._learn_mode = None
    
    def get_learn_mode(self) -> Optional[str]:
        """Get current learn mode ('a', 'b', or None)."""
        return self._learn_mode
    
    def is_learning(self) -> bool:
        """Check if any learn mode is active."""
        return self._learn_mode is not None
    
    # ============ Endpoint Capture ============
    
    def capture_endpoint_a(self):
        """Store the current synthesizer state as endpoint A."""
        self._endpoint_a = self._capture_all_channels()
    
    def capture_endpoint_b(self):
        """Store the current synthesizer state as endpoint B."""
        self._endpoint_b = self._capture_all_channels()
    
    def get_endpoint_a(self) -> Optional[List[Dict]]:
        """Get endpoint A data."""
        return self._endpoint_a
    
    def get_endpoint_b(self) -> Optional[List[Dict]]:
        """Get endpoint B data."""
        return self._endpoint_b
    
    def set_endpoint_a(self, data: List[Dict]):
        """Set endpoint A data directly (for loading presets)."""
        self._endpoint_a = copy.deepcopy(data)
    
    def set_endpoint_b(self, data: List[Dict]):
        """Set endpoint B data directly (for loading presets)."""
        self._endpoint_b = copy.deepcopy(data)
    
    # ============ Position & Interpolation ============
    
    @property
    def position(self) -> float:
        """Get current morph position (0.0 to 1.0)."""
        return self._position
    
    @position.setter
    def position(self, value: float):
        """Set morph position and apply interpolation."""
        self.set_position(value)
    
    def get_effective_position(self) -> float:
        """Get the effective morph position, accounting for learn mode.
        
        During learn mode the effective position is pinned to the endpoint
        being edited so the user hears only that endpoint:
        - Learning A → 0.0
        - Learning B → 1.0
        - No learn  → actual slider position
        """
        if self._learn_mode == 'a':
            return 0.0
        elif self._learn_mode == 'b':
            return 1.0
        return self._position
    
    def set_position(self, value: float):
        """
        Set the morph slider position and apply interpolation to all channels.
        
        Args:
            value: Morph position 0.0 (fully A) to 1.0 (fully B)
        """
        self._position = max(0.0, min(1.0, value))
        self._apply_interpolation()
    
    def apply_effective_position(self):
        """Apply the effective position (learn-aware) to all channels.
        
        Call this when entering or leaving learn mode to immediately
        update the synth output to match the target endpoint.
        """
        self._apply_interpolation()
    
    def _apply_interpolation(self):
        """Apply the effective morph position to all channels by interpolating
        between endpoint A and endpoint B."""
        if self._endpoint_a is None or self._endpoint_b is None:
            return
        if self._updating:
            return
        
        self._updating = True
        try:
            t = self.get_effective_position()
            
            for ch_idx in range(self.NUM_CHANNELS):
                if ch_idx >= len(self._endpoint_a) or ch_idx >= len(self._endpoint_b):
                    continue
                
                a_params = self._endpoint_a[ch_idx]
                b_params = self._endpoint_b[ch_idx]
                interpolated = self._interpolate_params(a_params, b_params, t)
                
                # Apply to channel
                self.synth.channels[ch_idx].set_parameters(interpolated, immediate=True)
        finally:
            self._updating = False
    
    def _interpolate_params(self, a: Dict, b: Dict, t: float) -> Dict:
        """
        Interpolate between two parameter dictionaries.
        
        Args:
            a: Endpoint A parameters
            b: Endpoint B parameters
            t: Interpolation factor (0.0 = fully A, 1.0 = fully B)
            
        Returns:
            Interpolated parameter dictionary
        """
        result = {}
        
        # Gather all keys
        all_keys = set(a.keys()) | set(b.keys())
        
        for key in all_keys:
            if key in EXCLUDED_PARAMS:
                # Use A endpoint name when closer to A, B when closer to B
                result[key] = a.get(key) if t < 0.5 else b.get(key)
                continue
            
            a_val = a.get(key)
            b_val = b.get(key)
            
            if a_val is None:
                result[key] = b_val
                continue
            if b_val is None:
                result[key] = a_val
                continue
            
            if key in INTERPOLATABLE_PARAMS:
                if key in LOG_INTERPOLATABLE_PARAMS:
                    # Logarithmic interpolation for frequency params
                    result[key] = self._log_lerp(float(a_val), float(b_val), t)
                else:
                    # Linear interpolation
                    result[key] = float(a_val) + (float(b_val) - float(a_val)) * t
            elif key in DISCRETE_PARAMS:
                # Snap at midpoint
                result[key] = a_val if t < 0.5 else b_val
            else:
                # Unknown param - snap at midpoint
                result[key] = a_val if t < 0.5 else b_val
        
        return result
    
    @staticmethod
    def _log_lerp(a: float, b: float, t: float) -> float:
        """Logarithmic interpolation between two positive values."""
        if a <= 0 or b <= 0:
            # Fallback to linear if values aren't positive
            return a + (b - a) * t
        log_a = np.log(a)
        log_b = np.log(b)
        return float(np.exp(log_a + (log_b - log_a) * t))
    
    # ============ Edit Integration ============
    
    def on_parameter_edited(self, channel_idx: int, param_name: str, new_value):
        """
        Called when a parameter is edited via the GUI while morph is active.
        
        In learn mode the edit goes exclusively to the target endpoint.
        Otherwise adjusts both endpoints proportionally based on slider position:
        - At position 0.0: only endpoint A is changed
        - At position 1.0: only endpoint B is changed
        - At position 0.5: both change equally
        
        Args:
            channel_idx: Channel index (0-7)
            param_name: Parameter name
            new_value: The new parameter value set by the user
        """
        if self._updating:
            return
        if self._endpoint_a is None or self._endpoint_b is None:
            return
        if channel_idx >= len(self._endpoint_a) or channel_idx >= len(self._endpoint_b):
            return
        
        # In learn mode, edits go directly to the target endpoint only
        if self._learn_mode == 'a':
            self._endpoint_a[channel_idx][param_name] = new_value
            return
        elif self._learn_mode == 'b':
            self._endpoint_b[channel_idx][param_name] = new_value
            return
        
        t = self._position
        
        a_params = self._endpoint_a[channel_idx]
        b_params = self._endpoint_b[channel_idx]
        
        if param_name in EXCLUDED_PARAMS:
            a_params[param_name] = new_value
            b_params[param_name] = new_value
            return
        
        if param_name in DISCRETE_PARAMS:
            # For discrete params, update the endpoint(s) based on position
            if t < 0.25:
                a_params[param_name] = new_value
            elif t > 0.75:
                b_params[param_name] = new_value
            else:
                a_params[param_name] = new_value
                b_params[param_name] = new_value
            return
        
        if param_name in INTERPOLATABLE_PARAMS:
            # Get old interpolated value and compute delta
            a_val = float(a_params.get(param_name, 0))
            b_val = float(b_params.get(param_name, 0))
            
            if param_name in LOG_INTERPOLATABLE_PARAMS:
                old_interp = self._log_lerp(a_val, b_val, t)
            else:
                old_interp = a_val + (b_val - a_val) * t
            
            new_val = float(new_value)
            delta = new_val - old_interp
            
            # Distribute delta to endpoints based on position
            # Weight to A = (1 - t), weight to B = t
            weight_a = 1.0 - t
            weight_b = t
            
            # Prevent division by zero: if at one extreme, all goes to that endpoint
            if weight_a < 0.001:
                b_params[param_name] = new_val
            elif weight_b < 0.001:
                a_params[param_name] = new_val
            else:
                if param_name in LOG_INTERPOLATABLE_PARAMS and a_val > 0 and b_val > 0:
                    # For log params, apply ratio-based adjustment
                    if old_interp > 0 and new_val > 0:
                        ratio = new_val / old_interp
                        a_params[param_name] = a_val * (ratio ** weight_a)
                        b_params[param_name] = b_val * (ratio ** weight_b)
                    else:
                        a_params[param_name] = a_val + delta * weight_a
                        b_params[param_name] = b_val + delta * weight_b
                else:
                    a_params[param_name] = a_val + delta
                    b_params[param_name] = b_val + delta
        else:
            # Unknown param - just update both
            a_params[param_name] = new_value
            b_params[param_name] = new_value
    
    # ============ Serialization ============
    
    def to_dict(self) -> Dict:
        """Serialize morph state for saving."""
        return {
            'position': self._position,
            'endpoint_a': copy.deepcopy(self._endpoint_a) if self._endpoint_a else None,
            'endpoint_b': copy.deepcopy(self._endpoint_b) if self._endpoint_b else None,
        }
    
    def from_dict(self, data: Dict):
        """Restore morph state from saved data."""
        if not data:
            return
        self._position = data.get('position', 0.0)
        if data.get('endpoint_a'):
            self._endpoint_a = copy.deepcopy(data['endpoint_a'])
        if data.get('endpoint_b'):
            self._endpoint_b = copy.deepcopy(data['endpoint_b'])
    
    def has_different_endpoints(self) -> bool:
        """Check if endpoints A and B are different (morph is meaningful)."""
        if self._endpoint_a is None or self._endpoint_b is None:
            return False
        return self._endpoint_a != self._endpoint_b
