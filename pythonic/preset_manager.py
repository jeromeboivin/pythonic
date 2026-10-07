"""
Preset Manager for Pythonic
Handles loading/saving of .mtpreset files and WAV export
"""

import re
import json
import os
import numpy as np
from scipy.io import wavfile
from typing import Dict, Any, List, Optional
from .pattern_manager import DEFAULT_VELOCITY, clamp_velocity


# ─────────────────────────────────────────────
# Shared enum maps (used by parsers, converters, and generator bridge)
# ─────────────────────────────────────────────

WAVEFORM_MAP = {'Sine': 0, 'Triangle': 1, 'Saw': 2}
PITCH_MOD_MAP = {'Decay': 0, 'Sine': 1, 'Noise': 2}
FILTER_MODE_MAP = {'LP': 0, 'BP': 1, 'HP': 2}
ENV_MODE_MAP = {'Exp': 0, 'Linear': 1, 'Mod': 2}


# ─────────────────────────────────────────────
# Public helpers: raw patch dict → channel format → apply
# ─────────────────────────────────────────────

def convert_drum_patch_data(patch: Dict) -> Dict:
    """Convert a raw patch dict (OscFreq / OscWave style) to internal channel format.

    Works identically for file-parsed .mtdrum dicts and model-generated dicts.
    """
    mix_value = patch.get('Mix', 50.0)
    if isinstance(mix_value, tuple):
        osc_mix = mix_value[0] / 100.0
    else:
        osc_mix = mix_value / 100.0

    return {
        'name': patch.get('Name', 'Untitled'),
        'osc_frequency': patch.get('OscFreq', 440.0),
        'osc_waveform': WAVEFORM_MAP.get(patch.get('OscWave', 'Sine'), 0),
        'osc_attack': patch.get('OscAtk', 0.0) / 1000.0,
        'osc_decay': patch.get('OscDcy', 316.0) / 1000.0,
        'pitch_mod_mode': PITCH_MOD_MAP.get(patch.get('ModMode', 'Decay'), 0),
        'pitch_mod_amount': patch.get('ModAmt', 0.0),
        'pitch_mod_rate': (patch.get('ModRate', 100.0) / 1000.0
                           if patch.get('ModMode', 'Decay') == 'Decay'
                           else patch.get('ModRate', 100.0)),
        'noise_filter_mode': FILTER_MODE_MAP.get(patch.get('NFilMod', 'LP'), 0),
        'noise_filter_freq': patch.get('NFilFrq', 20000.0),
        'noise_filter_q': patch.get('NFilQ', 0.707),
        'noise_stereo': patch.get('NStereo', False),
        'noise_envelope_mode': ENV_MODE_MAP.get(patch.get('NEnvMod', 'Exp'), 0),
        'noise_attack': patch.get('NEnvAtk', 0.0) / 1000.0,
        'noise_decay': patch.get('NEnvDcy', 316.0) / 1000.0,
        'osc_noise_mix': osc_mix,
        'distortion': patch.get('DistAmt', 0.0) / 100.0,
        'eq_frequency': patch.get('EQFreq', 1000.0),
        'eq_gain_db': patch.get('EQGain', 0.0),
        'level_db': patch.get('Level', 0.0),
        'pan': patch.get('Pan', 0.0),
        'output_pair': patch.get('Output', 'A'),
        'osc_vel_sensitivity': patch.get('OscVel', 0.0) / 100.0,
        'noise_vel_sensitivity': patch.get('NVel', 0.0) / 100.0,
        'mod_vel_sensitivity': patch.get('ModVel', 0.0) / 100.0,
        **({'choke_enabled': bool(patch['Choke'])} if 'Choke' in patch else {}),
    }


def channel_to_raw_patch(channel) -> Dict:
    """Convert a live DrumChannel to a raw patch dict (OscFreq, OscWave, etc.).

    This is the inverse of convert_drum_patch_data + apply_drum_patch_to_channel.
    Useful for building kit-fingerprint condition vectors from the current synth state.
    """
    waveform_names = ['Sine', 'Triangle', 'Saw']
    mod_mode_names = ['Decay', 'Sine', 'Noise']
    filter_mode_names = ['LP', 'BP', 'HP']
    env_mode_names = ['Exp', 'Linear', 'Mod']

    return {
        'Name': channel.name,
        'OscWave': waveform_names[channel.oscillator.waveform.value],
        'OscFreq': channel.oscillator.frequency,
        'OscDcy': channel.osc_envelope.decay_ms,
        'ModMode': mod_mode_names[channel.oscillator.pitch_mod_mode.value],
        # The channel stores the rate in display units (ms for Decay, Hz otherwise)
        'ModRate': channel.oscillator.pitch_mod_rate,
        'ModAmt': channel.oscillator.pitch_mod_amount,
        'NFilMod': filter_mode_names[channel.noise_gen.filter_mode.value],
        'NFilFrq': channel.noise_gen.filter_frequency,
        'NFilQ': channel.noise_gen.filter_q,
        'NStereo': channel.noise_gen.stereo,
        'NEnvMod': env_mode_names[channel.noise_gen.envelope_mode.value],
        'NEnvAtk': channel.noise_gen.attack_ms,
        'NEnvDcy': channel.noise_gen.decay_ms,
        'Mix': channel.osc_noise_mix * 100.0,
        'DistAmt': channel.distortion * 100.0,
        'EQFreq': channel.eq_frequency,
        'EQGain': channel.eq_gain_db,
        'Level': channel.level_db,
        'Pan': channel.pan,
        'Output': channel.output_pair,
        'OscVel': channel.osc_vel_sensitivity * 100.0,
        'NVel': channel.noise_vel_sensitivity * 100.0,
        'ModVel': channel.mod_vel_sensitivity * 100.0,
    }


def apply_drum_patch_to_channel(channel, data: Dict):
    """Apply a converted channel-format dict to a DrumChannel."""
    from .oscillator import WaveformType, PitchModMode
    from .noise import NoiseFilterMode, NoiseEnvelopeMode

    channel.name = data.get('name', 'Untitled')
    channel.set_pitch_semitones(data.get('pitch_semitones', 0.0))

    channel.set_osc_frequency(data.get('osc_frequency', 440.0))
    channel.set_osc_waveform(WaveformType(data.get('osc_waveform', 0)))
    channel.set_osc_attack(data.get('osc_attack', 0.0) * 1000.0)
    channel.set_osc_decay(data.get('osc_decay', 0.316) * 1000.0)

    pitch_mod_int = data.get('pitch_mod_mode', 0)
    channel.set_pitch_mod_mode(PitchModMode(pitch_mod_int))
    channel.set_pitch_mod_amount(data.get('pitch_mod_amount', 0.0))
    # convert_drum_patch_data gives the Decay rate in seconds; the channel uses ms
    rate = data.get('pitch_mod_rate', 0.1)
    if pitch_mod_int == 0:
        rate = rate * 1000.0
    channel.set_pitch_mod_rate(rate)

    channel.set_noise_filter_mode(NoiseFilterMode(data.get('noise_filter_mode', 0)))
    channel.set_noise_filter_freq(data.get('noise_filter_freq', 20000.0))
    channel.set_noise_filter_q(data.get('noise_filter_q', 0.707))
    channel.set_noise_stereo(data.get('noise_stereo', False))
    channel.set_noise_envelope_mode(NoiseEnvelopeMode(data.get('noise_envelope_mode', 0)))
    channel.set_noise_attack(data.get('noise_attack', 0.0) * 1000.0)
    channel.set_noise_decay(data.get('noise_decay', 0.316) * 1000.0)

    channel.set_osc_noise_mix(data.get('osc_noise_mix', 0.5))
    channel.set_distortion(data.get('distortion', 0.0))
    channel.set_eq_frequency(data.get('eq_frequency', 1000.0))
    channel.set_eq_gain(data.get('eq_gain_db', 0.0))
    channel.level_db = data.get('level_db', 0.0)
    channel.pan = data.get('pan', 0.0)
    channel.output_pair = data.get('output_pair', 'A')

    channel.osc_vel_sensitivity = data.get('osc_vel_sensitivity', 0.0)
    channel.noise_vel_sensitivity = data.get('noise_vel_sensitivity', 0.0)
    channel.mod_vel_sensitivity = data.get('mod_vel_sensitivity', 0.0)
    if 'choke_enabled' in data:
        channel.choke_enabled = bool(data['choke_enabled'])

class PythonicPresetParser:
    """
    Parser for Pythonic .mtpreset files
    """

    # Delegate to module-level maps for backward compat with code that
    # references parser.WAVEFORM_MAP etc.
    WAVEFORM_MAP = WAVEFORM_MAP
    PITCH_MOD_MAP = PITCH_MOD_MAP
    FILTER_MODE_MAP = FILTER_MODE_MAP
    ENV_MODE_MAP = ENV_MODE_MAP

    def __init__(self):
        self.content = ""
        self.pos = 0

    def parse_file(self, filepath: str) -> Dict[str, Any]:
        with open(filepath, 'r', encoding='utf-8') as f:
            return self.parse_string(f.read())

    def parse_string(self, content: str) -> Dict[str, Any]:
        self.content = content
        self.pos = 0
        return self._parse_root()

    def _skip_whitespace(self):
        while self.pos < len(self.content):
            if self.content[self.pos].isspace():
                self.pos += 1
            elif self.content[self.pos:self.pos+2] == '//':
                # Single-line comment
                while self.pos < len(self.content) and self.content[self.pos] != '\n':
                    self.pos += 1
            elif self.content[self.pos:self.pos+2] == '/*':
                # Block comment
                self.pos += 2
                while self.pos < len(self.content) - 1:
                    if self.content[self.pos:self.pos+2] == '*/':
                        self.pos += 2
                        break
                    self.pos += 1
            else:
                break

    def _parse_root(self) -> Dict[str, Any]:
        self._skip_whitespace()
        # Find "PresetV3:" in the content (case-insensitive)
        remaining = self.content[self.pos:].lower()
        preset_v3_pos = remaining.find('presetv3:')
        if preset_v3_pos == -1:
            raise ValueError("Invalid preset format: expected PresetV3")
        self.pos += preset_v3_pos + len('presetv3:')
        self._skip_whitespace()
        return self._parse_block()

    def _parse_block(self) -> Dict[str, Any]:
        self._skip_whitespace()
        if self.content[self.pos] != '{':
            raise ValueError(f"Expected '{{' at position {self.pos}")
        self.pos += 1
        result = {}
        while True:
            self._skip_whitespace()
            if self.pos >= len(self.content):
                break
            if self.content[self.pos] == '}':
                self.pos += 1
                break
            key = self._parse_key()
            self._skip_whitespace()
            if self.content[self.pos] == ':':
                self.pos += 1
                self._skip_whitespace()
                if self.content[self.pos] == '{':
                    result[key] = self._parse_brace_content()
                else:
                    result[key] = self._parse_value()
            elif self.content[self.pos] == '=':
                self.pos += 1
                self._skip_whitespace()
                result[key] = self._parse_value()
            else:
                raise ValueError(f"Expected ':' or '=' after key '{key}' at position {self.pos}")
        return result

    def _parse_brace_content(self) -> Any:
        self._skip_whitespace()
        if self.content[self.pos] != '{':
            raise ValueError(f"Expected '{{' at position {self.pos}")
        saved_pos = self.pos
        self.pos += 1
        self._skip_whitespace()
        if self.content[self.pos] == '}':
            self.pos += 1
            return {}
        
        # Peek ahead to determine if this is a block (key: value) or a list
        first_token_start = self.pos
        
        if self.content[self.pos] == '"':
            # Quoted token — skip the quoted string to peek at what follows
            self.pos += 1
            while self.pos < len(self.content) and self.content[self.pos] != '"':
                if self.content[self.pos] == '\\':
                    self.pos += 2
                else:
                    self.pos += 1
            if self.pos < len(self.content):
                self.pos += 1  # skip closing quote
        else:
            # Unquoted token
            while self.pos < len(self.content) and (self.content[self.pos].isalnum() or self.content[self.pos] in '_-'):
                self.pos += 1
        
        self._skip_whitespace()
        next_char = self.content[self.pos] if self.pos < len(self.content) else ''
        self.pos = saved_pos
        if next_char in (':', '='):
            return self._parse_block()
        else:
            return self._parse_list()

    def _parse_list(self) -> List:
        self._skip_whitespace()
        if self.content[self.pos] != '{':
            raise ValueError(f"Expected '{{' at position {self.pos}")
        self.pos += 1
        items = []
        while True:
            self._skip_whitespace()
            if self.pos >= len(self.content):
                break
            if self.content[self.pos] == '}':
                self.pos += 1
                break
            items.append(self._parse_value())
            self._skip_whitespace()
            if self.pos < len(self.content) and self.content[self.pos] == ',':
                self.pos += 1
        return items

    def _parse_key(self) -> str:
        self._skip_whitespace()
        if self.pos < len(self.content) and self.content[self.pos] == '"':
            # Quoted key (e.g., "1", "2", etc.)
            return self._parse_string()
        start = self.pos
        while self.pos < len(self.content) and (self.content[self.pos].isalnum() or self.content[self.pos] in '_-'):
            self.pos += 1
        return self.content[start:self.pos]

    def _parse_value(self) -> Any:
        self._skip_whitespace()
        if self.content[self.pos] == '"':
            return self._parse_string()
        elif self.content[self.pos] == '{':
            return self._parse_brace_content()
        elif self.content[self.pos:self.pos+2] == 'On' and (self.pos + 2 >= len(self.content) or not self.content[self.pos+2].isalnum()):
            self.pos += 2
            return True
        elif self.content[self.pos:self.pos+3] == 'Off' and (self.pos + 3 >= len(self.content) or not self.content[self.pos+3].isalnum()):
            self.pos += 3
            return False
        elif self.content[self.pos:self.pos+4] == 'true' and (self.pos + 4 >= len(self.content) or not self.content[self.pos+4].isalnum()):
            self.pos += 4
            return True
        elif self.content[self.pos:self.pos+5] == 'false' and (self.pos + 5 >= len(self.content) or not self.content[self.pos+5].isalnum()):
            self.pos += 5
            return False
        elif self.content[self.pos:self.pos+3] == 'inf' and (self.pos + 3 >= len(self.content) or not self.content[self.pos+3].isalnum()):
            self.pos += 3
            # Skip any trailing unit (like 'ms')
            while self.pos < len(self.content) and self.content[self.pos] in ' \t':
                self.pos += 1
            while self.pos < len(self.content) and (self.content[self.pos].isalpha() or self.content[self.pos] in '%x'):
                self.pos += 1
            return float('inf')
        else:
            return self._parse_number_or_identifier()

    def _parse_string(self) -> str:
        if self.content[self.pos] != '"':
            raise ValueError(f'Expected \'"\' at position {self.pos}')
        self.pos += 1
        start = self.pos
        while self.pos < len(self.content) and self.content[self.pos] != '"':
            if self.content[self.pos] == '\\':
                self.pos += 2
            else:
                self.pos += 1
        result = self.content[start:self.pos]
        self.pos += 1
        return result

    def _parse_number_or_identifier(self) -> Any:
        self._skip_whitespace()
        start = self.pos
        
        # Handle negative sign
        if self.pos < len(self.content) and self.content[self.pos] in '-+':
            self.pos += 1
        
        # Parse the numeric part
        while self.pos < len(self.content) and (self.content[self.pos].isdigit() or self.content[self.pos] == '.'):
            self.pos += 1
        
        # Check for exponent (scientific notation)
        if self.pos < len(self.content) and self.content[self.pos].lower() == 'e':
            self.pos += 1
            if self.pos < len(self.content) and self.content[self.pos] in '-+':
                self.pos += 1
            while self.pos < len(self.content) and self.content[self.pos].isdigit():
                self.pos += 1
        
        value_str = self.content[start:self.pos].strip()
        
        # Skip inline whitespace (spaces/tabs) but NOT newlines before checking for units
        # Units like "bpm", "Hz" appear on the same line as the number
        while self.pos < len(self.content) and self.content[self.pos] in ' \t':
            self.pos += 1
        
        # Check for special case: "number / number" format (e.g., Mix: 50.00 / 50.00)
        # Or step rate format like "1/16"
        if self.pos < len(self.content) and self.content[self.pos] == '/':
            # Check if this looks like a step rate (1/8, 1/16, 1/32, etc.)
            saved_pos = self.pos
            self.pos += 1  # skip '/'
            
            # Skip whitespace after '/'
            while self.pos < len(self.content) and self.content[self.pos] in ' \t':
                self.pos += 1
            
            # Parse second number
            second_start = self.pos
            if self.pos < len(self.content) and self.content[self.pos] in '-+':
                self.pos += 1
            while self.pos < len(self.content) and (self.content[self.pos].isdigit() or self.content[self.pos] == '.'):
                self.pos += 1
            
            second_value_str = self.content[second_start:self.pos].strip()
            
            # Check for 'T' suffix (triplet, e.g., 1/16T)
            if self.pos < len(self.content) and self.content[self.pos] == 'T':
                self.pos += 1
                second_value_str += 'T'
            
            # If second part is a common step rate divisor, return as string
            if second_value_str in ['8', '8T', '16', '16T', '32']:
                return f"{value_str}/{second_value_str}"
            
            # Otherwise it's a Mix-style fraction, return just the first number
            try:
                return float(value_str) if '.' in value_str else int(value_str)
            except ValueError:
                return value_str
        
        # Check if there's a unit suffix
        if self.pos < len(self.content):
            # Common unit patterns
            unit_start = self.pos
            # Parse potential unit (letters, %, x, but NOT /)
            while self.pos < len(self.content) and (self.content[self.pos].isalpha() or self.content[self.pos] in '%x'):
                self.pos += 1
            
            # If we found a unit, we ignore it (it's just metadata)
            # But we need to check if this is actually a unit or the start of a new key
            potential_unit = self.content[unit_start:self.pos]
            
            # Common units in presets
            known_units = ['bpm', 'Hz', 'ms', 'dB', 'sm', '%', 'x']
            
            # If it's not a known unit and has content, it might be a pattern like "1/16"
            if potential_unit and potential_unit not in known_units and not self.content[unit_start-1:unit_start] in [' ', '\t', '\n', '\r']:
                # This might be part of an identifier, reset
                self.pos = unit_start
        
        # Try to convert to number
        try:
            if '.' in value_str:
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            # If it's not a number, treat as identifier
            # Reset and parse as identifier
            self.pos = start
            while self.pos < len(self.content) and (self.content[self.pos].isalnum() or self.content[self.pos] in '._/-#'):
                self.pos += 1
            return self.content[start:self.pos].strip()

    def convert_to_synth_format(self, preset_data: Dict) -> Dict:
        result = {
            'name': preset_data.get('Name', 'Untitled'),
            'master_volume_db': self._to_float(preset_data.get('MastVol', 0.0)),
            'tempo': self._to_float(preset_data.get('Tempo', 120), 120.0),
            'step_rate': preset_data.get('StepRate', '1/16'),
            'swing': self._to_swing_value(preset_data.get('Swing', 0.0)),
            'fill_rate': self._to_float(preset_data.get('FillRate', 4.0), 4.0),
            'drums': [],
            'mutes': [self._to_bool(mute) for mute in preset_data.get('Mutes', [False] * 8)],
            'patterns': None,
            'morph_position': None,
        }
        
        # Parse Morph block if present
        if 'Morph' in preset_data:
            morph_block = preset_data['Morph']
            result['morph_position'] = self._to_float(morph_block.get('Time', 0.5))
        if 'DrumPatches' in preset_data:
            for i in range(1, 9):
                key = str(i)
                if key in preset_data['DrumPatches']:
                    patch = preset_data['DrumPatches'][key]
                    channel_data = self._convert_drum_patch(patch)
                    result['drums'].append(channel_data)
                else:
                    result['drums'].append(self._default_channel())
        
        # Parse patterns if available
        if 'Patterns' in preset_data:
            result['patterns'] = self._convert_patterns(preset_data['Patterns'])
        
        return result

    def _convert_drum_patch(self, patch: Dict) -> Dict:
        # Handle Mix value: can be a float (first part of "X / Y"), a string "X / Y", or already numeric
        mix_raw = patch.get('Mix', 50.0)
        if isinstance(mix_raw, str) and '/' in mix_raw:
            try:
                mix_raw = float(mix_raw.split('/')[0].strip())
            except ValueError:
                mix_raw = 50.0
        elif isinstance(mix_raw, str):
            try:
                mix_raw = float(mix_raw)
            except ValueError:
                mix_raw = 50.0
        
        return {
            'name': patch.get('Name', 'Untitled'),
            'osc_frequency': self._to_float(patch.get('OscFreq', 440.0)),
            'osc_waveform': self.WAVEFORM_MAP.get(patch.get('OscWave', 'Sine'), 0),
            'osc_attack': self._to_float(patch.get('OscAtk', 0.0)),
            'osc_decay': self._to_float(patch.get('OscDcy', 316.0)),
            'pitch_mod_mode': self.PITCH_MOD_MAP.get(patch.get('ModMode', 'Decay'), 0),
            'pitch_mod_amount': self._to_float(patch.get('ModAmt', 0.0)),
            'pitch_mod_rate': self._to_float(patch.get('ModRate', 100.0)),
            'noise_filter_mode': self.FILTER_MODE_MAP.get(patch.get('NFilMod', 'LP'), 0),
            'noise_filter_freq': self._to_float(patch.get('NFilFrq', 20000.0)),
            'noise_filter_q': self._to_float(patch.get('NFilQ', 0.707)),
            'noise_stereo': self._to_bool(patch.get('NStereo', False)),
            'noise_envelope_mode': self.ENV_MODE_MAP.get(patch.get('NEnvMod', 'Exp'), 0),
            'noise_attack': self._to_float(patch.get('NEnvAtk', 0.0)),
            'noise_decay': self._to_float(patch.get('NEnvDcy', 316.0)),
            'osc_noise_mix': float(mix_raw) / 100.0,
            'distortion': self._to_float(patch.get('DistAmt', 0.0)) / 100.0,
            'eq_frequency': self._to_float(patch.get('EQFreq', 1000.0)),
            'eq_gain_db': self._to_float(patch.get('EQGain', 0.0)),
            'level_db': self._to_float(patch.get('Level', 0.0)),
            'pan': self._to_float(patch.get('Pan', 0.0)),
            'output_pair': patch.get('Output', 'A'),
            'osc_vel_sensitivity': self._to_float(patch.get('OscVel', 0.0)) / 100.0,
            'noise_vel_sensitivity': self._to_float(patch.get('NVel', 0.0)) / 100.0,
            'mod_vel_sensitivity': self._to_float(patch.get('ModVel', 0.0)) / 100.0,
            'choke_enabled': self._to_bool(patch.get('Choke', False)),
        }

    @staticmethod
    def _to_float(val, default=0.0) -> float:
        """Safely convert a value to float, handling string representations."""
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            # Strip unit suffixes (Hz, ms, dB, sm, %, x, bpm)
            import re
            match = re.match(r'^([+-]?\d*\.?\d+)', val.strip())
            if match:
                return float(match.group(1))
        return float(default)

    @classmethod
    def _to_swing_value(cls, val, default=0.0) -> float:
        """Convert preset/GUI swing values to PatternManager's 0-1 range."""
        if isinstance(val, str) and '%' in val:
            return max(0.0, min(1.0, cls._to_float(val, default) / 100.0))

        numeric = cls._to_float(val, default)
        if numeric > 1.0:
            numeric /= 100.0
        return max(0.0, min(1.0, numeric))

    @staticmethod
    def _to_bool(val, default=False) -> bool:
        """Convert On/Off style fields to bool."""
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            normalized = val.strip().lower()
            if normalized in {'on', 'true', 'yes', '1'}:
                return True
            if normalized in {'off', 'false', 'no', '0'}:
                return False
        return bool(default)

    def _default_channel(self) -> Dict:
        return {
            'name': 'Empty',
            'osc_frequency': 440.0,
            'osc_waveform': 0,
            'osc_attack': 0.0,
            'osc_decay': 316.0,
            'pitch_mod_mode': 0,
            'pitch_mod_amount': 0.0,
            'pitch_mod_rate': 100.0,
            'noise_filter_mode': 0,
            'noise_filter_freq': 20000.0,
            'noise_filter_q': 0.707,
            'noise_stereo': False,
            'noise_envelope_mode': 0,
            'noise_attack': 0.0,
            'noise_decay': 316.0,
            'osc_noise_mix': 0.5,
            'distortion': 0.0,
            'eq_frequency': 1000.0,
            'eq_gain_db': 0.0,
            'level_db': 0.0,
            'pan': 0.0,
            'output_pair': 'A',
            'osc_vel_sensitivity': 0.0,
            'noise_vel_sensitivity': 0.0,
            'mod_vel_sensitivity': 0.0,
        }

    def _convert_patterns(self, patterns_data: Dict) -> Dict:
        """Convert pattern data from preset format to internal format"""
        patterns = {}
        
        # Pattern names from preset file are lowercase (a-l)
        pattern_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
        
        for pattern_name in pattern_names:
            if pattern_name not in patterns_data:
                continue
                
            pattern_block = patterns_data[pattern_name]
            
            # Get pattern properties
            length = pattern_block.get('Length', 16)
            chained = pattern_block.get('Chained', False)
            
            # Parse channels (1-8)
            channels = {}
            for channel_id in range(1, 9):
                channel_key = str(channel_id)
                if channel_key in pattern_block:
                    channel_data = pattern_block[channel_key]
                    channels[channel_id - 1] = self._parse_pattern_channel(
                        channel_data, length
                    )
            
            patterns[pattern_name.upper()] = {
                'length': length,
                'chained': chained,
                'channels': channels
            }
        
        return patterns
    
    def _parse_pattern_channel(self, channel_data, expected_length: int) -> Dict:
        """Parse a single channel's pattern data"""
        # Handle list format (e.g., [8888, 8888, 8888, 8888]) - silent pattern
        if isinstance(channel_data, list):
            return {
                'triggers': [False] * expected_length,
                'accents': [False] * expected_length,
                'fills': [False] * expected_length,
                'probabilities': [100] * expected_length,
                'substeps': [''] * expected_length,
                'velocities': [DEFAULT_VELOCITY] * expected_length,
            }
        
        triggers_str = channel_data.get('Triggers', '')
        accents_str = channel_data.get('Accents', '')
        fills_str = channel_data.get('Fills', '')
        probs_str = channel_data.get('Probabilities', '')
        substeps_str = channel_data.get('Substeps', '')  # New: parse substeps
        velocities_str = channel_data.get('Velocities', '')  # Pythonic: per-step velocity
        
        # Convert pattern strings to boolean lists
        # '#' means on, '-' means off
        triggers = [c == '#' for c in triggers_str]
        accents = [c == '#' for c in accents_str]
        fills = [c == '#' for c in fills_str]
        
        # Parse probabilities - format is comma-separated integers (e.g., "100,75,50,100")
        # Default to 100 if not present
        if probs_str:
            try:
                probabilities = [int(p) for p in probs_str.split(',')]
            except ValueError:
                probabilities = [100] * expected_length
        else:
            probabilities = [100] * expected_length
        
        # Parse velocities - comma-separated integers 1-127 like probabilities;
        # files without them play their unaccented steps at 64 as always
        try:
            velocities = [clamp_velocity(v) for v in velocities_str.split(',')] \
                if velocities_str else []
        except ValueError:
            velocities = []

        # Parse substeps - format is comma-separated patterns (e.g., "oo-,o-o-,,oo-")
        # Empty means no substeps for that step
        if substeps_str:
            substeps = substeps_str.split(',')
        else:
            substeps = [''] * expected_length
        
        # Ensure all lists are the expected length
        triggers = triggers[:expected_length] + [False] * max(0, expected_length - len(triggers))
        accents = accents[:expected_length] + [False] * max(0, expected_length - len(accents))
        fills = fills[:expected_length] + [False] * max(0, expected_length - len(fills))
        probabilities = probabilities[:expected_length] + [100] * max(0, expected_length - len(probabilities))
        substeps = substeps[:expected_length] + [''] * max(0, expected_length - len(substeps))
        velocities = velocities[:expected_length] + \
            [DEFAULT_VELOCITY] * max(0, expected_length - len(velocities))
        
        return {
            'triggers': triggers,
            'accents': accents,
            'fills': fills,
            'probabilities': probabilities,
            'substeps': substeps,
            'velocities': velocities,
        }


class PresetManager:
    def __init__(self, synthesizer):
        self.synth = synthesizer
        self.parser = PythonicPresetParser()
        self.drum_parser = DrumPatchParser()
        self.current_preset_name = "Untitled"

    def load_mtpreset(self, filepath: str) -> Dict:
        raw_data = self.parser.parse_file(filepath)
        preset_data = self.parser.convert_to_synth_format(raw_data)
        self.current_preset_name = preset_data.get('name', 'Untitled')
        return preset_data

    def load_drum_patch(self, filepath: str, channel_index: int):
        """
        Load a .mtdrum file into a specific drum channel
        
        Args:
            filepath: Path to the .mtdrum file
            channel_index: Channel index (0-7) to load the patch into
        """
        if not 0 <= channel_index < 8:
            raise ValueError(f"Invalid channel index: {channel_index}. Must be 0-7")
        
        # Parse the drum patch file
        drum_data = self.drum_parser.parse_file(filepath)
        # Patches without a Name line are named after the file
        drum_data.setdefault('Name', os.path.splitext(os.path.basename(filepath))[0])
        
        # Convert to channel format
        channel_data = self._convert_drum_patch_data(drum_data)
        
        # Apply to the channel
        channel = self.synth.channels[channel_index]
        self._apply_drum_patch_to_channel(channel, channel_data)
        
        return channel_data

    def save_drum_patch(self, channel_index: int, filepath: str, name: Optional[str] = None):
        """
        Save a drum channel to a .mtdrum file
        
        Args:
            channel_index: Channel index (0-7) to save
            filepath: Path where to save the .mtdrum file
            name: Optional name for the drum patch (defaults to channel name)
        """
        if not 0 <= channel_index < 8:
            raise ValueError(f"Invalid channel index: {channel_index}. Must be 0-7")
        
        channel = self.synth.channels[channel_index]
        DrumPatchWriter.write_drum_patch(filepath, channel, name)
        
        return filepath

    def _convert_drum_patch_data(self, patch: Dict) -> Dict:
        """Convert parsed .mtdrum data to internal channel format"""
        return convert_drum_patch_data(patch)

    def _apply_drum_patch_to_channel(self, channel, data: Dict):
        """Apply drum patch data to a channel"""
        apply_drum_patch_to_channel(channel, data)


    def export_drum_to_wav(self, channel, filepath: str, duration_ms: float = 2000.0, velocity: int = 127, sample_rate: int = 44100, bit_depth: int = 16, mono: bool = False):
        from .drum_channel import DrumChannel

        # Render a copy so the live voice (and the audio thread using it) is untouched
        render = DrumChannel(channel.channel_id, sample_rate)
        render.set_parameters(channel.get_parameters())
        render._synthesizer = channel._synthesizer  # tempo-synced LFOs read its BPM
        num_samples = int(duration_ms * sample_rate / 1000.0)
        render.trigger(velocity)
        audio = render.process(num_samples)
        if mono:
            audio = ((audio[:, 0] + audio[:, 1]) * 0.5).reshape(-1, 1)
        max_val = np.max(np.abs(audio))
        if max_val > 0.99:
            audio = audio * (0.99 / max_val)
        if bit_depth == 16:
            audio_int = (audio * 32767).astype(np.int16)
        else:
            audio_int = audio.astype(np.float32)
        wavfile.write(filepath, sample_rate, audio_int)
        return filepath

    def export_all_drums_to_wav(self, synth, output_dir: str, duration_ms: float = 2000.0, velocity: int = 127, sample_rate: int = 44100, bit_depth: int = 16, mono: bool = False) -> List[str]:
        os.makedirs(output_dir, exist_ok=True)
        exported_files = []
        for i in range(8):
            channel = synth.channels[i]
            safe_name = re.sub(r'[^\w\s-]', '', channel.name)
            safe_name = re.sub(r'\s+', '_', safe_name)
            filename = f"{i+1:02d}_{safe_name}.wav"
            filepath = os.path.join(output_dir, filename)
            self.export_drum_to_wav(channel=channel, filepath=filepath, duration_ms=duration_ms, velocity=velocity, sample_rate=sample_rate, bit_depth=bit_depth, mono=mono)
            exported_files.append(filepath)
        return exported_files


class DrumPatchParser:
    """
    Parser for .mtdrum drum patch files.

    Reads every .mtdrum format version: V1 ``{ Key=value }``, V2 ``{ Key = value }``
    and V3 ``{ Key: value }``, with bare or quoted values. Numbers (with or without a unit, including
    ``inf``) become floats, Mix becomes an (osc, noise) tuple and On/Off,
    true/false become booleans.
    """

    WAVEFORM_MAP = WAVEFORM_MAP
    PITCH_MOD_MAP = PITCH_MOD_MAP
    FILTER_MODE_MAP = FILTER_MODE_MAP
    ENV_MODE_MAP = ENV_MODE_MAP

    TEXT_KEYS = {'Name', 'Path'}
    _HEADER = re.compile(r'DrumPatchV\d\s*[=:]\s*\{', re.IGNORECASE)
    _ENTRY = re.compile(r'([A-Za-z][\w-]*)\s*[=:]\s*(.*?)\s*$')
    _NUMBER = re.compile(r'([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?|[+-]?inf)\s*(?:Hz|ms|dB|sm|%)?$',
                         re.IGNORECASE)

    def parse_file(self, filepath: str) -> Dict[str, Any]:
        """Parse a .mtdrum file and return the drum patch parameters"""
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            return self.parse_string(f.read())

    def parse_string(self, content: str) -> Dict[str, Any]:
        header = self._HEADER.search(content)
        if not header:
            raise ValueError("Invalid drum patch format: expected DrumPatchV1/V2/V3")
        result = {}
        for line in content[header.end():].splitlines():
            line = line.strip()
            if line.startswith('}'):
                break
            if not line or line.startswith('//'):
                continue
            entry = self._ENTRY.match(line)
            if not entry:
                raise ValueError(f"Invalid drum patch line: {line!r}")
            key, text = entry.groups()
            result[key] = self._parse_value(key, text)
        return result

    def _parse_value(self, key: str, text: str) -> Any:
        if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
            text = text[1:-1]
        if key in self.TEXT_KEYS:
            return text
        lowered = text.lower()
        if lowered in ('on', 'true'):
            return True
        if lowered in ('off', 'false'):
            return False
        if '/' in text:
            osc, noise = text.split('/', 1)
            try:
                return (float(osc), float(noise))
            except ValueError:
                return text
        number = self._NUMBER.match(text)
        return float(number.group(1)) if number else text


class DrumPatchWriter:
    """
    Writer for .mtdrum drum patch files (format version 3, quoted values at
    full precision).
    """

    UNITS = {
        'OscFreq': ' Hz', 'OscAtk': ' ms', 'OscDcy': ' ms', 'ModAmt': ' sm',
        'NFilFrq': ' Hz', 'NEnvAtk': ' ms', 'NEnvDcy': ' ms', 'EQFreq': ' Hz',
        'EQGain': ' dB', 'Level': ' dB', 'OscVel': '%', 'NVel': '%', 'ModVel': '%',
    }

    @staticmethod
    def patch_from_channel(channel, name: Optional[str] = None) -> Dict[str, Any]:
        """Display values of a drum channel, keyed and ordered as in a .mtdrum file."""
        osc, noise = channel.oscillator, channel.noise_gen
        return {
            'Name': channel.name if name is None else name,
            'Modified': False,
            'OscWave': ['Sine', 'Triangle', 'Saw'][osc.waveform.value],
            'OscFreq': osc.frequency,
            'OscAtk': channel.osc_envelope.attack_ms,
            'OscDcy': channel.osc_envelope.decay_ms,
            'ModMode': ['Decay', 'Sine', 'Noise'][osc.pitch_mod_mode.value],
            'ModRate': osc.pitch_mod_rate,  # ms (Decay) or Hz, as displayed
            'ModAmt': osc.pitch_mod_amount,
            'NFilMod': ['LP', 'BP', 'HP'][noise.filter_mode.value],
            'NFilFrq': noise.filter_frequency,
            'NFilQ': noise.filter_q,
            'NStereo': noise.stereo,
            'NEnvMod': ['Exp', 'Linear', 'Mod'][noise.envelope_mode.value],
            'NEnvAtk': noise.attack_ms,
            'NEnvDcy': noise.decay_ms,
            'Mix': (channel.osc_noise_mix * 100.0, (1.0 - channel.osc_noise_mix) * 100.0),
            'DistAmt': channel.distortion * 100.0,
            'EQFreq': channel.eq_frequency,
            'EQGain': channel.eq_gain_db,
            'Level': channel.level_db,
            'Pan': channel.pan,
            'Output': channel.output_pair,
            'Choke': channel.choke_enabled,
            'OscVel': channel.osc_vel_sensitivity * 100.0,
            'NVel': channel.noise_vel_sensitivity * 100.0,
            'ModVel': channel.mod_vel_sensitivity * 100.0,
        }

    @staticmethod
    def _number(value) -> str:
        """Shortest round-trip decimal."""
        value = float(value)
        if np.isinf(value):
            return 'inf' if value > 0 else '-inf'
        return np.format_float_positional(value, unique=True, trim='0')

    @classmethod
    def format_value(cls, key: str, value: Any, patch: Optional[Dict[str, Any]] = None) -> str:
        """Format one patch value (without the surrounding quotes)."""
        if isinstance(value, bool):
            return 'On' if value else 'Off'
        if isinstance(value, str):
            return value.replace('"', "'")
        if isinstance(value, (tuple, list)):
            return f'{cls._number(value[0])} / {cls._number(value[1])}'
        if key == 'ModRate':
            decay = (patch or {}).get('ModMode', 'Decay') == 'Decay'
            unit = ' ms' if decay or np.isinf(float(value)) else ' Hz'
            return cls._number(value) + unit
        return cls._number(value) + cls.UNITS.get(key, '')

    @classmethod
    def format_patch(cls, patch: Dict[str, Any]) -> str:
        lines = ['MicrotonicDrumPatchV3: {']
        for key, value in patch.items():
            if key == 'Modified':
                lines.append(f'\t{key}: {"true" if value else "false"}')
            else:
                lines.append(f'\t{key}: "{cls.format_value(key, value, patch)}"')
        lines.append('}')
        return '\n'.join(lines) + '\n'

    @classmethod
    def write_drum_patch(cls, filepath: str, channel, name: Optional[str] = None):
        """Write a drum channel to a .mtdrum file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(cls.format_patch(cls.patch_from_channel(channel, name)))
