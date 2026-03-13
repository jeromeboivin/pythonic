import os
import re
import hashlib
import json
from pythonic.preset_manager import PythonicPresetParser

# List of drum type keywords (lowercase)
DRUM_TYPES = [
    "bd", "sd", "ch", "oh", "fx", "bass", "reverse", "perc", "blip", "zap",
    "fuzz", "synth", "cowbell", "tom", "clap", "cy", "shaker", "other"
]

def find_mtpreset_files(root_dir):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.mtpreset'):
                yield os.path.join(dirpath, filename)

def detect_drum_type(name):
    lname = name.lower()
    for drum_type in DRUM_TYPES:
        if re.search(r'\b' + re.escape(drum_type) + r'\b', lname):
            return drum_type
    return "other"

# Mock classes to mimic the channel structure for DrumPatchWriter
class MockWaveform:
    def __init__(self, value):
        self.value = value

class MockMode:
    def __init__(self, value):
        self.value = value

class MockOscillator:
    def __init__(self, data):
        self.frequency = data['osc_frequency']
        self.waveform = MockWaveform(data['osc_waveform'])
        self.pitch_mod_mode = MockMode(data['pitch_mod_mode'])
        self.pitch_mod_amount = data['pitch_mod_amount']
        self.pitch_mod_rate = data['pitch_mod_rate']

class MockEnvelope:
    def __init__(self, data):
        self.decay_ms = data['osc_decay'] * 1000.0  # Convert seconds to ms

class MockNoiseGen:
    def __init__(self, data):
        self.filter_mode = MockMode(data['noise_filter_mode'])
        self.filter_frequency = data['noise_filter_freq']
        self.filter_q = data['noise_filter_q']
        self.stereo = data['noise_stereo']
        self.envelope_mode = MockMode(data['noise_envelope_mode'])
        self.attack_ms = data['noise_attack'] * 1000.0  # Convert seconds to ms
        self.decay_ms = data['noise_decay'] * 1000.0  # Convert seconds to ms

class MockChannel:
    def __init__(self, data):
        self.name = data['name']
        self.oscillator = MockOscillator(data)
        self.osc_envelope = MockEnvelope(data)
        self.noise_gen = MockNoiseGen(data)
        self.osc_noise_mix = data['osc_noise_mix']
        self.distortion = data['distortion']
        self.eq_frequency = data['eq_frequency']
        self.eq_gain_db = data['eq_gain_db']
        self.level_db = data['level_db']
        self.pan = data['pan']
        self.output_pair = data['output_pair']
        self.osc_vel_sensitivity = data['osc_vel_sensitivity']
        self.noise_vel_sensitivity = data['noise_vel_sensitivity']
        self.mod_vel_sensitivity = data['mod_vel_sensitivity']

def main(root_dir, target_dir):
    from pythonic.preset_manager import DrumPatchWriter
    parser = PythonicPresetParser()
    os.makedirs(target_dir, exist_ok=True)
    for drum_type in DRUM_TYPES:
        if drum_type != "other":
            os.makedirs(os.path.join(target_dir, drum_type), exist_ok=True)

    seen_hashes = set()

    for mtpreset_path in find_mtpreset_files(root_dir):
        try:
            preset_data = parser.parse_file(mtpreset_path)
            synth_data = parser.convert_to_synth_format(preset_data)
            drums = synth_data.get("drums", [])
            for idx, drum in enumerate(drums):
                patch_name = drum.get("name", f"drum_{idx+1}")
                drum_type = detect_drum_type(patch_name)
                if drum_type == "other":
                    continue
                # Compute hash excluding name
                drum_copy = {k: v for k, v in drum.items() if k != 'name'}
                drum_str = json.dumps(drum_copy, sort_keys=True)
                drum_hash = hashlib.md5(drum_str.encode()).hexdigest()
                if drum_hash in seen_hashes:
                    continue
                seen_hashes.add(drum_hash)
                safe_name = re.sub(r'[^\w\s-]', '', patch_name).strip().replace(' ', '_')
                mtdrum_filename = f"{safe_name or f'drum_{idx+1}'}.mtdrum"
                mtdrum_path = os.path.join(target_dir, drum_type, mtdrum_filename)
                # Create mock channel from drum data
                mock_channel = MockChannel(drum)
                # Write drum patch as .mtdrum file
                DrumPatchWriter.write_drum_patch(mtdrum_path, mock_channel, patch_name)
                # print(f"Extracted: {mtdrum_path}")
        except Exception as e:
            print(f"Failed to process {mtpreset_path}: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python extract_drum_patches.py <root_folder> <target_folder>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])