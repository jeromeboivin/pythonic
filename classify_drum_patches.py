#!/usr/bin/env python3
"""
KNN-based drum patch classifier.

Learns from a labelled reference library of .mtdrum files,
then classifies every drum channel extracted from a folder of .mtpreset files
and writes labelled .mtdrum outputs into per-type folders.

Usage:
    python classify_drum_patches.py \
        --reference  "/path/to/reference_patches" \
        --input      "/path/to/patterns" \
        --output     "./drum_patches_knn" \
        --k 5
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

from pythonic.preset_manager import (
    DrumPatchParser,
    DrumPatchWriter,
    PythonicPresetParser,
    WAVEFORM_MAP,
    PITCH_MOD_MAP,
    FILTER_MODE_MAP,
    ENV_MODE_MAP,
)


# ─────────────────────────────────────────────
# Constants (mirrored from train.py)
# ─────────────────────────────────────────────

# KNN features. OscAtk is left out on purpose: most reference patches use the
# V1 format, which has no OscAtk, so it would read as 0 for all of them.

CONTINUOUS_PARAMS = [
    "OscFreq", "OscDcy", "ModAmt", "ModRate",
    "NFilFrq", "NFilQ", "NEnvAtk", "NEnvDcy",
    "Mix", "DistAmt", "EQFreq", "EQGain",
    "Level", "OscVel", "NVel", "ModVel",
]

LOG_PARAMS = {
    "OscFreq", "OscDcy", "ModRate", "NFilFrq",
    "NFilQ", "NEnvAtk", "NEnvDcy", "EQFreq",
}

PARAM_CLAMP = {
    "OscFreq": (20.0, 20_000.0),
    "OscDcy":  (1.0, 10_000_000.0),
    "ModAmt":  (-96.0, 96.0),
    "ModRate": (0.001, 100_000.0),
    "NFilFrq": (20.0, 20_000.0),
    "NFilQ":   (0.5, 100.0),
    "NEnvAtk": (0.001, 100_000.0),
    "NEnvDcy": (1.0, 10_000_000.0),
    "Mix":     (0.0, 100.0),
    "DistAmt": (0.0, 100.0),
    "EQFreq":  (20.0, 20_000.0),
    "EQGain":  (-40.0, 40.0),
    "Level":   (-50.0, 20.0),
    "OscVel":  (0.0, 200.0),
    "NVel":    (0.0, 200.0),
    "ModVel":  (0.0, 200.0),
}

CATEGORICAL_PARAMS = {
    "OscWave": ["Sine", "Triangle", "Saw"],
    "ModMode": ["Decay", "Sine", "Noise"],
    "NFilMod": ["LP", "BP", "HP"],
    "NEnvMod": ["Exp", "Linear", "Mod"],
}

DRUM_TYPES = [
    "bass", "bd", "blip", "ch", "clap", "cowbell", "cy", "fuzz", "fx",
    "oh", "perc", "reverse", "sd", "shaker", "synth", "tom", "zap", "other",
]

# ─────────────────────────────────────────────
# Reference patch name → DRUM_TYPES mapping
# ─────────────────────────────────────────────
# Maps substrings found in reference patch names to DRUM_TYPES labels.
# Checked in order; first match wins.

_REFERENCE_LABEL_RULES: List[Tuple[re.Pattern, str]] = [
    # Kick / bass drum
    (re.compile(r'\bBD\b|\bKICK\b|BASS\s*DR', re.I), "bd"),
    # Snare drum
    (re.compile(r'\bSD\b|\bSNARE\b|\bSNR\b', re.I), "sd"),
    # Closed hihat / closed sounds
    (re.compile(r'\bCH\b|\bHH\b(?!.*Long)(?!.*Med)|\bCL\b(?!aves|ap)|\bSH\b(?!aker)', re.I), "ch"),
    # HH damped / choked → ch
    (re.compile(r'HH Dampen|HH.*Choked', re.I), "ch"),
    # Open hihat / open sounds
    (re.compile(r'\bOH\b|HH Long|HH Med', re.I), "oh"),
    # Cymbal family: crash, ride, CY, triangle
    (re.compile(r'\bCY\b|Crash|Ride|\bTriangle\b|DidgCymbal', re.I), "cy"),
    # Clap and rimshot
    (re.compile(r'Clap|\bCP\b|\bRS\b|Rimshot', re.I), "clap"),
    # Tom (all pitches)
    (re.compile(r'Tom|\bHT\b|\bMT\b|\bLT\b|Hi Drum|Lo Drum', re.I), "tom"),
    # Cowbell
    (re.compile(r'Cowbell|\bCB\b', re.I),   "cowbell"),
    # Shaker / maracas / cabasa / tambourine / guiro
    (re.compile(r'Shaker|Maracas|\bMA\b|Cabasa|Tambourine|Tamb\b|Guiro|MetalGuiro', re.I), "shaker"),
    # Percussion: conga, bongo, timbale, claves, blocks, quijada, click
    (re.compile(r'Conga|Bongo|Timbale|Claves|Blocks|Quijada|\bHC\b|\bLC\b|\bMC\b|\bClick\b|\bPerc\b', re.I), "perc"),
    # Metal / misc percussion
    (re.compile(r'Metal(?!Guiro)', re.I),    "perc"),
    # FX / synth / noise / bass synth / alarm / special
    (re.compile(r'\bFX\b|\bZAP\b|Alarm|SpecialFx|SteamClap|VinylScratch|Noise|DigitFX|Looney|Wobbler|Rattler|Chopper|Blipper|AcidDrip', re.I), "fx"),
    # Bass synth sounds
    (re.compile(r'Bass(?!\s*DR)|Stomp|Boot|SubSonar|SubAlarm|Coconut', re.I), "bass"),
]


def map_reference_name_to_label(name) -> Optional[str]:
    """Map a reference patch name to a DRUM_TYPES label, or None if unmapped."""
    if not isinstance(name, str):
        return None
    for pattern, label in _REFERENCE_LABEL_RULES:
        if pattern.search(name):
            return label
    return None


# ─────────────────────────────────────────────
# Feature encoding (same logic as train.py PatchPreprocessor)
# ─────────────────────────────────────────────

def _extract_continuous(patch: Dict[str, Any]) -> np.ndarray:
    """Extract, clamp, and log-transform continuous parameters from a raw patch dict."""
    vals = []
    for p in CONTINUOUS_PARAMS:
        v = patch.get(p, 0.0)
        if isinstance(v, (tuple, list)):
            v = v[0]
        try:
            v = float(v)
        except Exception:
            v = 0.0
        if p in PARAM_CLAMP:
            lo, hi = PARAM_CLAMP[p]
            v = max(lo, min(hi, v))
        if p in LOG_PARAMS:
            v = np.log(v)
        vals.append(v)
    return np.array(vals, dtype=np.float32)


def _extract_categorical_onehot(patch: Dict[str, Any]) -> np.ndarray:
    """One-hot encode categorical parameters."""
    cat_to_idx = {
        k: {v: i for i, v in enumerate(vals)}
        for k, vals in CATEGORICAL_PARAMS.items()
    }
    vecs = []
    for param, vals in CATEGORICAL_PARAMS.items():
        val = patch.get(param, vals[0])
        idx = cat_to_idx[param].get(val, 0)
        oh = np.zeros(len(vals), dtype=np.float32)
        oh[idx] = 1.0
        vecs.append(oh)
    return np.concatenate(vecs)


def encode_patch(patch: Dict[str, Any], scaler: MinMaxScaler) -> np.ndarray:
    """Encode a raw patch dict into a normalised feature vector."""
    cont = _extract_continuous(patch)
    cont_norm = scaler.transform(cont.reshape(1, -1))[0]
    cat_oh = _extract_categorical_onehot(patch)
    return np.concatenate([cont_norm, cat_oh])


# ─────────────────────────────────────────────
# Internal channel format → raw patch dict
# ─────────────────────────────────────────────

_WAVEFORM_NAMES = ['Sine', 'Triangle', 'Saw']
_MOD_MODE_NAMES = ['Decay', 'Sine', 'Noise']
_FILTER_MODE_NAMES = ['LP', 'BP', 'HP']
_ENV_MODE_NAMES = ['Exp', 'Linear', 'Mod']


def _internal_to_raw(drum: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an internal channel-format dict (from convert_to_synth_format)
    back to the raw patch dict format expected by PatchPreprocessor / DrumPatchWriter."""
    osc_mix = drum.get('osc_noise_mix', 0.5)
    mod_mode_idx = drum.get('pitch_mod_mode', 0)

    return {
        'Name':    drum.get('name', 'Untitled'),
        'OscWave': _WAVEFORM_NAMES[drum.get('osc_waveform', 0)],
        'OscFreq': drum.get('osc_frequency', 440.0),
        'OscAtk':  drum.get('osc_attack', 0.0),
        'OscDcy':  drum.get('osc_decay', 316.0),
        'ModMode': _MOD_MODE_NAMES[mod_mode_idx],
        'ModAmt':  drum.get('pitch_mod_amount', 0.0),
        'ModRate': drum.get('pitch_mod_rate', 100.0),
        'NFilMod': _FILTER_MODE_NAMES[drum.get('noise_filter_mode', 0)],
        'NFilFrq': drum.get('noise_filter_freq', 20000.0),
        'NFilQ':   drum.get('noise_filter_q', 0.707),
        'NStereo': drum.get('noise_stereo', False),
        'NEnvMod': _ENV_MODE_NAMES[drum.get('noise_envelope_mode', 0)],
        'NEnvAtk': drum.get('noise_attack', 0.0),
        'NEnvDcy': drum.get('noise_decay', 316.0),
        'Mix':     (osc_mix * 100.0, (1.0 - osc_mix) * 100.0),
        'DistAmt': drum.get('distortion', 0.0) * 100.0,
        'EQFreq':  drum.get('eq_frequency', 1000.0),
        'EQGain':  drum.get('eq_gain_db', 0.0),
        'Level':   drum.get('level_db', 0.0),
        'Pan':     drum.get('pan', 0.0),
        'Output':  drum.get('output_pair', 'A'),
        'Choke':   drum.get('choke_enabled', False),
        'OscVel':  drum.get('osc_vel_sensitivity', 0.0) * 100.0,
        'NVel':    drum.get('noise_vel_sensitivity', 0.0) * 100.0,
        'ModVel':  drum.get('mod_vel_sensitivity', 0.0) * 100.0,
    }


# ─────────────────────────────────────────────
# Reference dataset builder
# ─────────────────────────────────────────────

def load_reference_set(
    reference_dir: str,
) -> Tuple[List[Dict], List[str], List[str]]:
    """Load labelled .mtdrum files from the reference directory.

    Returns (patches, labels, source_files) — only patches whose names
    could be mapped via the manual reference rules are included.
    """
    parser = DrumPatchParser()
    patches: List[Dict] = []
    labels: List[str] = []
    sources: List[str] = []
    skipped: List[str] = []

    # Recursively find all .mtdrum files under reference_dir
    mtdrum_files = []
    for dirpath, _, filenames in os.walk(reference_dir):
        for fname in sorted(filenames):
            if fname.lower().endswith('.mtdrum'):
                mtdrum_files.append((os.path.join(dirpath, fname), fname))
    mtdrum_files.sort(key=lambda x: x[1])

    for fpath, fname in mtdrum_files:
        try:
            patch = parser.parse_file(fpath)
        except Exception as exc:
            print(f"  [ref] skip parse error: {fname}: {exc}")
            continue

        name = patch.get('Name', fname)
        label = map_reference_name_to_label(name)
        if label is None:
            label = map_reference_name_to_label(fname)
        if label is None:
            skipped.append(fname)
            continue

        patches.append(patch)
        labels.append(label)
        sources.append(fname)

    if skipped:
        print(f"  [ref] skipped {len(skipped)} unmapped patches: {skipped}")
    print(f"  [ref] loaded {len(patches)} labelled reference patches")
    return patches, labels, sources


# ─────────────────────────────────────────────
# Target preset scanner
# ─────────────────────────────────────────────

def _find_mtpreset_files(root_dir: str):
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith('.mtpreset'):
                yield os.path.join(dirpath, fname)


def _content_hash(raw_patch: Dict) -> str:
    """Stable hash of patch parameters, excluding name and non-sonic fields."""
    d = {k: v for k, v in raw_patch.items() if k not in ('Name', 'Modified')}
    # Normalise tuples to lists for stable JSON
    for k, v in d.items():
        if isinstance(v, tuple):
            d[k] = list(v)
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def extract_target_drums(
    input_dir: str,
) -> List[Dict]:
    """Extract raw patch dicts from all .mtpreset files under input_dir.

    Each returned dict has extra keys:
      _source_preset, _source_gen, _drum_index, _content_hash
    """
    parser = PythonicPresetParser()
    seen_hashes: set = set()
    results: List[Dict] = []
    preset_count = 0
    dup_count = 0

    for mtpreset_path in _find_mtpreset_files(input_dir):
        try:
            preset_data = parser.parse_file(mtpreset_path)
            synth_data = parser.convert_to_synth_format(preset_data)
        except Exception as exc:
            print(f"  [target] skip parse error: {mtpreset_path}: {exc}")
            continue

        preset_count += 1
        gen_folder = os.path.basename(os.path.dirname(mtpreset_path))

        for idx, drum in enumerate(synth_data.get('drums', [])):
            raw = _internal_to_raw(drum)
            h = _content_hash(raw)
            if h in seen_hashes:
                dup_count += 1
                continue
            seen_hashes.add(h)

            raw['_source_preset'] = os.path.basename(mtpreset_path)
            raw['_source_gen'] = gen_folder
            raw['_drum_index'] = idx
            raw['_content_hash'] = h
            results.append(raw)

    print(f"  [target] scanned {preset_count} presets, "
          f"extracted {len(results)} unique drums "
          f"({dup_count} duplicates removed)")
    return results


# ─────────────────────────────────────────────
# KNN classifier
# ─────────────────────────────────────────────

def build_classifier(
    ref_patches: List[Dict],
    ref_labels: List[str],
    k: int = 5,
) -> Tuple[KNeighborsClassifier, MinMaxScaler]:
    """Fit a KNN classifier on the reference set.

    Returns (knn, fitted_scaler).
    """
    # Fit the scaler on reference continuous features
    cont_data = np.stack([_extract_continuous(p) for p in ref_patches])
    scaler = MinMaxScaler()
    scaler.fit(cont_data)

    # Encode all reference patches
    X_ref = np.stack([encode_patch(p, scaler) for p in ref_patches])
    y_ref = np.array(ref_labels)

    knn = KNeighborsClassifier(n_neighbors=min(k, len(ref_patches)), weights='distance')
    knn.fit(X_ref, y_ref)

    print(f"  [knn] trained with k={knn.n_neighbors} on {len(ref_patches)} samples, "
          f"{len(set(ref_labels))} classes")
    return knn, scaler


def classify_drums(
    knn: KNeighborsClassifier,
    scaler: MinMaxScaler,
    target_patches: List[Dict],
    ref_sources: List[str],
    ref_labels: List[str],
) -> List[Dict]:
    """Classify target patches and return enriched result dicts."""
    if not target_patches:
        return []

    X_target = np.stack([encode_patch(p, scaler) for p in target_patches])
    predictions = knn.predict(X_target)
    distances, indices = knn.kneighbors(X_target)

    results = []
    for i, patch in enumerate(target_patches):
        neighbor_labels = [ref_labels[j] for j in indices[i]]
        neighbor_sources = [ref_sources[j] for j in indices[i]]
        neighbor_dists = distances[i].tolist()

        results.append({
            'patch': patch,
            'predicted_label': predictions[i],
            'neighbor_labels': neighbor_labels,
            'neighbor_sources': neighbor_sources,
            'neighbor_distances': neighbor_dists,
        })
    return results


# ─────────────────────────────────────────────
# Holdout evaluation
# ─────────────────────────────────────────────

def evaluate_holdout(
    ref_patches: List[Dict],
    ref_labels: List[str],
    ref_sources: List[str],
    k: int = 5,
):
    """Leave-one-out evaluation on the reference set."""
    from collections import Counter

    cont_data = np.stack([_extract_continuous(p) for p in ref_patches])
    scaler = MinMaxScaler()
    scaler.fit(cont_data)
    X = np.stack([encode_patch(p, scaler) for p in ref_patches])
    y = np.array(ref_labels)

    # Use k+1 neighbours so we can skip self
    knn = KNeighborsClassifier(n_neighbors=min(k + 1, len(ref_patches)), weights='distance')
    knn.fit(X, y)

    correct = 0
    confusion: Dict[str, Counter] = {lbl: Counter() for lbl in set(ref_labels)}

    for i in range(len(ref_patches)):
        dists, idxs = knn.kneighbors(X[i:i+1])
        # skip self (distance ~0)
        neighbor_labels = []
        for j_pos, j in enumerate(idxs[0]):
            if j == i:
                continue
            neighbor_labels.append(y[j])
            if len(neighbor_labels) == k:
                break
        if not neighbor_labels:
            neighbor_labels = [y[idxs[0][1]]] if len(idxs[0]) > 1 else [y[0]]

        # Distance-weighted vote
        pred = Counter(neighbor_labels).most_common(1)[0][0]
        confusion[y[i]][pred] += 1
        if pred == y[i]:
            correct += 1

    accuracy = correct / len(ref_patches) * 100.0
    print(f"\n  [eval] Leave-one-out accuracy: {correct}/{len(ref_patches)} = {accuracy:.1f}%")
    print(f"  [eval] Confusion (true → predicted counts):")
    for true_label in sorted(confusion):
        preds = confusion[true_label]
        total = sum(preds.values())
        parts = ", ".join(f"{pl}:{pc}" for pl, pc in preds.most_common())
        print(f"    {true_label:>10s} ({total:3d}): {parts}")
    print()


# ─────────────────────────────────────────────
# Mock channel for DrumPatchWriter
# ─────────────────────────────────────────────

class _Val:
    def __init__(self, v): self.value = v

class _MockOsc:
    def __init__(self, raw):
        self.frequency = raw.get('OscFreq', 440.0)
        self.waveform = _Val(WAVEFORM_MAP.get(raw.get('OscWave', 'Sine'), 0))
        self.pitch_mod_mode = _Val(PITCH_MOD_MAP.get(raw.get('ModMode', 'Decay'), 0))
        self.pitch_mod_amount = raw.get('ModAmt', 0.0)
        self.pitch_mod_rate = raw.get('ModRate', 100.0)

class _MockEnv:
    def __init__(self, raw):
        self.attack_ms = raw.get('OscAtk', 0.0)
        self.decay_ms = raw.get('OscDcy', 316.0)

class _MockNoise:
    def __init__(self, raw):
        self.filter_mode = _Val(FILTER_MODE_MAP.get(raw.get('NFilMod', 'LP'), 0))
        self.filter_frequency = raw.get('NFilFrq', 20000.0)
        self.filter_q = raw.get('NFilQ', 0.707)
        self.stereo = raw.get('NStereo', False)
        self.envelope_mode = _Val(ENV_MODE_MAP.get(raw.get('NEnvMod', 'Exp'), 0))
        self.attack_ms = raw.get('NEnvAtk', 0.0)
        self.decay_ms = raw.get('NEnvDcy', 316.0)

class _MockChannel:
    def __init__(self, raw: Dict):
        self.name = raw.get('Name', 'Untitled')
        self.oscillator = _MockOsc(raw)
        self.osc_envelope = _MockEnv(raw)
        self.noise_gen = _MockNoise(raw)
        mix = raw.get('Mix', 50.0)
        if isinstance(mix, (tuple, list)):
            self.osc_noise_mix = mix[0] / 100.0
        else:
            self.osc_noise_mix = float(mix) / 100.0
        self.distortion = raw.get('DistAmt', 0.0) / 100.0
        self.eq_frequency = raw.get('EQFreq', 1000.0)
        self.eq_gain_db = raw.get('EQGain', 0.0)
        self.level_db = raw.get('Level', 0.0)
        self.pan = raw.get('Pan', 0.0)
        self.output_pair = raw.get('Output', 'A')
        self.choke_enabled = raw.get('Choke', False)
        self.osc_vel_sensitivity = raw.get('OscVel', 0.0) / 100.0
        self.noise_vel_sensitivity = raw.get('NVel', 0.0) / 100.0
        self.mod_vel_sensitivity = raw.get('ModVel', 0.0) / 100.0


# ─────────────────────────────────────────────
# Output writer
# ─────────────────────────────────────────────

def write_outputs(
    results: List[Dict],
    output_dir: str,
    manifest_path: str,
):
    """Write labelled .mtdrum files and a CSV manifest."""
    # Create label folders
    for dt in DRUM_TYPES:
        os.makedirs(os.path.join(output_dir, dt), exist_ok=True)

    written = 0
    manifest_rows = []

    for r in results:
        patch = r['patch']
        label = r['predicted_label']
        name = patch.get('Name', 'drum')
        safe_name = re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')
        if not safe_name:
            safe_name = f"drum_{patch.get('_drum_index', 0)}"

        # Avoid filename collisions by appending hash suffix
        h = patch.get('_content_hash', '')[:8]
        mtdrum_filename = f"{safe_name}_{h}.mtdrum"
        mtdrum_path = os.path.join(output_dir, label, mtdrum_filename)

        try:
            mock_ch = _MockChannel(patch)
            DrumPatchWriter.write_drum_patch(mtdrum_path, mock_ch, name)
            written += 1
        except Exception as exc:
            print(f"  [write] failed: {mtdrum_path}: {exc}")
            mtdrum_path = ""

        manifest_rows.append({
            'source_preset': patch.get('_source_preset', ''),
            'source_gen': patch.get('_source_gen', ''),
            'drum_index': patch.get('_drum_index', ''),
            'original_name': name,
            'content_hash': patch.get('_content_hash', ''),
            'predicted_label': label,
            'neighbor_labels': ';'.join(r['neighbor_labels']),
            'neighbor_distances': ';'.join(f"{d:.4f}" for d in r['neighbor_distances']),
            'neighbor_sources': ';'.join(r['neighbor_sources']),
            'output_path': mtdrum_path,
        })

    # Write CSV manifest
    if manifest_rows:
        fieldnames = list(manifest_rows[0].keys())
        with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)

    print(f"  [write] wrote {written} .mtdrum files to {output_dir}")
    print(f"  [write] manifest: {manifest_path} ({len(manifest_rows)} rows)")

    # Summary by label
    from collections import Counter
    label_counts = Counter(r['predicted_label'] for r in results)
    print(f"\n  Distribution by label:")
    for lbl in sorted(label_counts):
        print(f"    {lbl:>10s}: {label_counts[lbl]}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="KNN drum patch classifier: learns from a labelled .mtdrum "
                    "reference library, classifies drums extracted from .mtpreset files."
    )
    ap.add_argument("--reference", required=True,
                    help="Directory containing labelled .mtdrum reference patches "
                         "(patch names carry the drum type, e.g. \"BD 808\")")
    ap.add_argument("--input", required=True,
                    help="Root directory of .mtpreset files to classify")
    ap.add_argument("--output", default="./drum_patches_knn",
                    help="Output directory for labelled .mtdrum files (default: ./drum_patches_knn)")
    ap.add_argument("--manifest", default=None,
                    help="Path for CSV manifest (default: <output>/manifest.csv)")
    ap.add_argument("--k", type=int, default=5,
                    help="Number of neighbours for KNN (default: 5)")
    ap.add_argument("--eval-only", action="store_true",
                    help="Only run leave-one-out evaluation on the reference set, then exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="Extract and classify but do not write output files")
    args = ap.parse_args()

    manifest_path = args.manifest or os.path.join(args.output, "manifest.csv")

    # ── Phase 1: Load reference set ──
    print("Loading reference set…")
    ref_patches, ref_labels, ref_sources = load_reference_set(args.reference)
    if not ref_patches:
        print("ERROR: No labelled reference patches found. Check --reference path "
              "and the label mapping table.", file=sys.stderr)
        sys.exit(1)

    # ── Phase 2: Holdout evaluation ──
    print("Running leave-one-out evaluation on reference set…")
    evaluate_holdout(ref_patches, ref_labels, ref_sources, k=args.k)

    if args.eval_only:
        return

    # ── Phase 3: Build classifier ──
    print("Building KNN classifier…")
    knn, scaler = build_classifier(ref_patches, ref_labels, k=args.k)

    # ── Phase 4: Extract target drums ──
    print("Extracting drums from target presets…")
    target_patches = extract_target_drums(args.input)
    if not target_patches:
        print("ERROR: No drums extracted from target presets. Check --input path.",
              file=sys.stderr)
        sys.exit(1)

    # ── Phase 5: Classify ──
    print("Classifying target drums…")
    results = classify_drums(knn, scaler, target_patches, ref_sources, ref_labels)

    # ── Phase 6: Write outputs ──
    if args.dry_run:
        print("[dry-run] Skipping file output.")
        from collections import Counter
        label_counts = Counter(r['predicted_label'] for r in results)
        print(f"\n  Distribution by label:")
        for lbl in sorted(label_counts):
            print(f"    {lbl:>10s}: {label_counts[lbl]}")
    else:
        os.makedirs(args.output, exist_ok=True)
        write_outputs(results, args.output, manifest_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
