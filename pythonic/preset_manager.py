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


class PythonicPresetParser:
    """
    Parser for Pythonic .mtpreset files
    """

    WAVEFORM_MAP = {
        'Sine': 0,
        'Triangle': 1,
        'Saw': 2,
    }

    PITCH_MOD_MAP = {
        'Decay': 0,
        'Sine': 1,
        'Noise': 2,
    }

    FILTER_MODE_MAP = {
        'LP': 0,
        'BP': 1,
        'HP': 2,
    }

    ENV_MODE_MAP = {
        'Exp': 0,
        'Linear': 1,
        'Mod': 2,
    }

    def __init__(self):
        self.content = ""
        self.pos = 0

    def parse_file(self, filepath: str) -> Dict[str, Any]:
        with open(filepath, 'r', encoding='utf-8') as f:
            self.content = f.read()
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
        if not self.content[self.pos:].startswith('MicroTonicPresetV3:'):
            raise ValueError("Invalid preset format: expected MicroTonicPresetV3")
        self.pos += len('MicroTonicPresetV3:')
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
        first_token_start = self.pos
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
        if self.pos < len(self.content) and self.content[self.pos] == '-':
            self.pos += 1
        while self.pos < len(self.content) and (self.content[self.pos].isalnum() or self.content[self.pos] in '._/-#'):
            self.pos += 1
        value_str = self.content[start:self.pos].strip()
        try:
            if '.' in value_str:
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            return value_str

    def convert_to_synth_format(self, preset_data: Dict) -> Dict:
        result = {
            'name': preset_data.get('Name', 'Untitled'),
            'master_volume_db': preset_data.get('MastVol', 0.0),
            'tempo': preset_data.get('Tempo', 120),
            'drums': [],
            'mutes': preset_data.get('Mutes', [False] * 8),
        }
        if 'DrumPatches' in preset_data:
            for i in range(1, 9):
                key = str(i)
                if key in preset_data['DrumPatches']:
                    patch = preset_data['DrumPatches'][key]
                    channel_data = self._convert_drum_patch(patch)
                    result['drums'].append(channel_data)
                else:
                    result['drums'].append(self._default_channel())
        return result

    def _convert_drum_patch(self, patch: Dict) -> Dict:
        return {
            'name': patch.get('Name', 'Untitled'),
            'osc_frequency': patch.get('OscFreq', 440.0),
            'osc_waveform': self.WAVEFORM_MAP.get(patch.get('OscWave', 'Sine'), 0),
            'osc_attack': patch.get('OscAtk', 0.0),
            'osc_decay': patch.get('OscDcy', 316.0),
            'pitch_mod_mode': self.PITCH_MOD_MAP.get(patch.get('ModMode', 'Decay'), 0),
            'pitch_mod_amount': patch.get('ModAmt', 0.0),
            'pitch_mod_rate': patch.get('ModRate', 100.0),
            'noise_filter_mode': self.FILTER_MODE_MAP.get(patch.get('NFilMod', 'LP'), 0),
            'noise_filter_freq': patch.get('NFilFrq', 20000.0),
            'noise_filter_q': patch.get('NFilQ', 0.707),
            'noise_stereo': patch.get('NStereo', False),
            'noise_envelope_mode': self.ENV_MODE_MAP.get(patch.get('NEnvMod', 'Exp'), 0),
            'noise_attack': patch.get('NEnvAtk', 0.0),
            'noise_decay': patch.get('NEnvDcy', 316.0),
            'osc_noise_mix': patch.get('Mix', 50.0) / 100.0,
            'distortion': patch.get('DistAmt', 0.0) / 100.0,
            'eq_frequency': patch.get('EQFreq', 1000.0),
            'eq_gain_db': patch.get('EQGain', 0.0),
            'level_db': patch.get('Level', 0.0),
            'pan': patch.get('Pan', 0.0),
            'output_pair': patch.get('Output', 'A'),
            'osc_vel_sensitivity': patch.get('OscVel', 0.0) / 100.0,
            'noise_vel_sensitivity': patch.get('NVel', 0.0) / 100.0,
            'mod_vel_sensitivity': patch.get('ModVel', 0.0) / 100.0,
        }

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


class PresetManager:
    def __init__(self, synthesizer):
        self.synth = synthesizer
        self.parser = PythonicPresetParser()
        self.current_preset_name = "Untitled"

    def load_mtpreset(self, filepath: str) -> Dict:
        raw_data = self.parser.parse_file(filepath)
        preset_data = self.parser.convert_to_synth_format(raw_data)
        self.current_preset_name = preset_data.get('name', 'Untitled')
        return preset_data

    def export_drum_to_wav(self, channel, filepath: str, duration_ms: float = 2000.0, velocity: int = 127, sample_rate: int = 44100, bit_depth: int = 16):
        num_samples = int(duration_ms * sample_rate / 1000.0)
        channel.trigger(velocity)
        audio = channel.process(num_samples)
        max_val = np.max(np.abs(audio))
        if max_val > 0.99:
            audio = audio * (0.99 / max_val)
        if bit_depth == 16:
            audio_int = (audio * 32767).astype(np.int16)
        else:
            audio_int = audio.astype(np.float32)
        wavfile.write(filepath, sample_rate, audio_int)
        return filepath

    def export_all_drums_to_wav(self, synth, output_dir: str, duration_ms: float = 2000.0, velocity: int = 127, sample_rate: int = 44100, bit_depth: int = 16) -> List[str]:
        os.makedirs(output_dir, exist_ok=True)
        exported_files = []
        for i in range(8):
            channel = synth.channels[i]
            safe_name = re.sub(r'[^\w\s-]', '', channel.name)
            safe_name = re.sub(r'\s+', '_', safe_name)
            filename = f"{i+1:02d}_{safe_name}.wav"
            filepath = os.path.join(output_dir, filename)
            self.export_drum_to_wav(channel=channel, filepath=filepath, duration_ms=duration_ms, velocity=velocity, sample_rate=sample_rate, bit_depth=bit_depth)
            exported_files.append(filepath)
        return exported_files
