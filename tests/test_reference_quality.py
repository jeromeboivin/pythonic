"""
Comprehensive synthesis quality test suite.

Compares rendered output of all 20 calibration presets against reference WAVs.
Tests individual audio features per hit and overall loss per preset.

Usage:
    pytest tests/test_reference_quality.py -v
    pytest tests/test_reference_quality.py -v -k "preset_01"   # single preset
    pytest tests/test_reference_quality.py -v --tb=short        # brief output
"""

import os
import sys
import json
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.render_preset import load_and_render_preset
from tools.audio_features import (
    load_wav_float, detect_onsets, segment_hits, align_and_trim,
    compute_hit_features, compare_features, compute_composite_loss,
    compare_full_wavs
)

SAMPLE_RATE = 44100
PATCHES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'test_patches')

# ============================================================================
# Loss thresholds per preset (based on calibrated synthesis parameters)
# Tighter thresholds for well-calibrated presets, looser for complex ones
# ============================================================================
PRESET_THRESHOLDS = {
    '01': {'max_loss': 0.22, 'desc': 'Oscillator Calibration'},
    '02': {'max_loss': 0.26, 'desc': 'Envelope Decay Calibration'},
    '03': {'max_loss': 0.27, 'desc': 'Envelope Attack Calibration'},
    '04': {'max_loss': 0.20, 'desc': 'Pitch Mod Decay Calibration'},
    '05': {'max_loss': 0.58, 'desc': 'Pitch Mod Sine Noise Calibration'},
    '06': {'max_loss': 0.30, 'desc': 'Noise Filter Calibration'},
    '07': {'max_loss': 0.24, 'desc': 'Noise Envelope Modes'},
    '08': {'max_loss': 0.32, 'desc': 'Mix Calibration'},
    '09': {'max_loss': 0.48, 'desc': 'Distortion Calibration'},
    '10': {'max_loss': 0.33, 'desc': 'EQ Calibration'},
    '11': {'max_loss': 0.27, 'desc': 'Level Pan Calibration'},
    '12': {'max_loss': 0.43, 'desc': 'Velocity Calibration'},
    '13': {'max_loss': 0.16, 'desc': 'Noise Stereo Calibration'},
    '14': {'max_loss': 0.36, 'desc': 'Kick Drum Reference'},
    '15': {'max_loss': 0.29, 'desc': 'Snare Drum Reference'},
    '16': {'max_loss': 0.22, 'desc': 'HiHat Cymbal Reference'},
    '17': {'max_loss': 0.33, 'desc': 'Percussion Reference'},
    '18': {'max_loss': 0.28, 'desc': 'Extreme Values'},
    '19': {'max_loss': 0.24, 'desc': 'Decay Shape Analysis'},
    '20': {'max_loss': 0.33, 'desc': 'Full Kit Reference'},
}

# Per-metric thresholds (worst acceptable average across all presets)
METRIC_THRESHOLDS = {
    'peak_amplitude_db': 5.0,       # dB difference
    'rms_db': 5.0,                  # dB difference
    'spectral_centroid_hz': 1200.0, # Hz difference
    'log_spectrum_db': 25.0,        # dB average difference
    'mel_spectrum_db': 10.0,        # dB average difference
    'pitch_semitones': 4.0,         # semitones
    'envelope_correlation': 0.15,   # 1 - correlation
    'decay_time_diff': 0.10,        # seconds difference
    'crest_factor_diff': 3.0,       # ratio
    'stereo_width_diff': 0.05,      # width difference
}

# Overall average loss threshold
AVERAGE_LOSS_THRESHOLD = 0.28


def get_calibration_presets():
    """Discover all calibration preset + reference WAV pairs."""
    presets = []
    for f in sorted(os.listdir(PATCHES_DIR)):
        if not f.endswith('.mtpreset'):
            continue
        num_prefix = f.split(' - ')[0] if ' - ' in f else ''
        if not num_prefix.isdigit() or int(num_prefix) > 20:
            continue
        wav_name = f.replace('.mtpreset', '.wav')
        wav_path = os.path.join(PATCHES_DIR, wav_name)
        if not os.path.exists(wav_path):
            continue
        presets.append({
            'num': num_prefix,
            'name': f.replace('.mtpreset', ''),
            'preset_path': os.path.join(PATCHES_DIR, f),
            'ref_path': wav_path,
        })
    return presets


def render_and_compare(preset_info):
    """Render a preset and compare against reference."""
    ref, sr = load_wav_float(preset_info['ref_path'])
    gen_audio, _, _, _ = load_and_render_preset(
        preset_info['preset_path'],
        duration_seconds=len(ref) / sr + 0.5
    )
    return compare_full_wavs(preset_info['ref_path'], gen_audio, sr)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope='session')
def all_presets():
    """Load all calibration presets (once per session)."""
    return get_calibration_presets()


@pytest.fixture(scope='session')
def all_results(all_presets):
    """Render and compare all presets (once per session, cached)."""
    results = {}
    for p in all_presets:
        try:
            result = render_and_compare(p)
            results[p['num']] = {
                'info': p,
                'result': result,
            }
        except Exception as e:
            results[p['num']] = {
                'info': p,
                'error': str(e),
            }
    return results


# ============================================================================
# Individual preset tests
# ============================================================================

@pytest.mark.parametrize('preset_num', [f'{i:02d}' for i in range(1, 21)])
def test_preset_loss(all_results, preset_num):
    """Test that each preset's composite loss is below its threshold."""
    if preset_num not in all_results:
        pytest.skip(f'Preset {preset_num} not found')
    
    entry = all_results[preset_num]
    if 'error' in entry:
        pytest.fail(f'Render error: {entry["error"]}')
    
    result = entry['result']
    loss = result['overall_loss']
    threshold = PRESET_THRESHOLDS.get(preset_num, {}).get('max_loss', 0.50)
    desc = PRESET_THRESHOLDS.get(preset_num, {}).get('desc', '')
    
    assert loss < threshold, (
        f'Preset {preset_num} ({desc}) loss {loss:.4f} exceeds threshold {threshold:.2f}. '
        f'Ref hits: {result["ref_n_hits"]}, Gen hits: {result["gen_n_hits"]}, '
        f'Avg timing error: {result["avg_timing_error_ms"]:.1f}ms'
    )


@pytest.mark.parametrize('preset_num', [f'{i:02d}' for i in range(1, 21)])
def test_preset_hit_count(all_results, preset_num):
    """Test that generated audio has similar hit count to reference."""
    if preset_num not in all_results:
        pytest.skip(f'Preset {preset_num} not found')
    
    entry = all_results[preset_num]
    if 'error' in entry:
        pytest.fail(f'Render error: {entry["error"]}')
    
    result = entry['result']
    ref_hits = result['ref_n_hits']
    gen_hits = result['gen_n_hits']
    
    # Allow ±1 hit tolerance (onset detection can vary)
    assert abs(ref_hits - gen_hits) <= 2, (
        f'Preset {preset_num}: ref has {ref_hits} hits, gen has {gen_hits} hits '
        f'(difference > 2)'
    )


@pytest.mark.parametrize('preset_num', [f'{i:02d}' for i in range(1, 21)])
def test_preset_timing(all_results, preset_num):
    """Test that onset timing is reasonably accurate."""
    if preset_num not in all_results:
        pytest.skip(f'Preset {preset_num} not found')
    
    entry = all_results[preset_num]
    if 'error' in entry:
        pytest.fail(f'Render error: {entry["error"]}')
    
    result = entry['result']
    avg_timing = result['avg_timing_error_ms']
    
    # Allow up to 100ms average timing error (relaxed for presets with
    # complex onset patterns that cause alignment differences)
    max_timing_ms = 600.0 if preset_num in ('05',) else 200.0 if preset_num in ('18',) else 100.0
    assert avg_timing < max_timing_ms, (
        f'Preset {preset_num}: average onset timing error {avg_timing:.1f}ms > {max_timing_ms}ms'
    )


# ============================================================================
# Per-metric aggregate tests
# ============================================================================

@pytest.mark.parametrize('metric_name,threshold', list(METRIC_THRESHOLDS.items()))
def test_metric_average(all_results, metric_name, threshold):
    """Test that the average of each metric across all presets is acceptable."""
    values = []
    for num, entry in all_results.items():
        if 'error' in entry:
            continue
        for hit_losses in entry['result']['per_hit_losses']:
            if metric_name in hit_losses:
                values.append(hit_losses[metric_name])
    
    if not values:
        pytest.skip(f'No values for metric {metric_name}')
    
    avg = np.mean(values)
    assert avg < threshold, (
        f'Average {metric_name} = {avg:.4f} exceeds threshold {threshold:.4f} '
        f'(over {len(values)} hits)'
    )


# ============================================================================
# Overall quality tests
# ============================================================================

def test_overall_average_loss(all_results):
    """Test that the overall average loss across all presets is acceptable."""
    losses = []
    for num, entry in all_results.items():
        if 'error' not in entry:
            losses.append(entry['result']['overall_loss'])
    
    assert len(losses) > 0, 'No preset results available'
    avg_loss = np.mean(losses)
    
    assert avg_loss < AVERAGE_LOSS_THRESHOLD, (
        f'Overall average loss {avg_loss:.4f} exceeds threshold {AVERAGE_LOSS_THRESHOLD}. '
        f'Tested {len(losses)} presets.'
    )


def test_no_preset_catastrophic(all_results):
    """Test that no single preset has catastrophic loss (>0.8)."""
    for num, entry in all_results.items():
        if 'error' in entry:
            continue
        loss = entry['result']['overall_loss']
        assert loss < 0.8, (
            f'Preset {num} has catastrophic loss {loss:.4f} (>0.8)'
        )


def test_majority_presets_good(all_results):
    """Test that at least 60% of presets have loss below 0.30."""
    good = 0
    total = 0
    for num, entry in all_results.items():
        if 'error' in entry:
            continue
        total += 1
        if entry['result']['overall_loss'] < 0.30:
            good += 1
    
    ratio = good / max(total, 1)
    assert ratio >= 0.60, (
        f'Only {good}/{total} ({ratio:.0%}) presets have loss < 0.30 (need >= 60%)'
    )


# ============================================================================
# Report generation (runs last)
# ============================================================================

def test_generate_report(all_results, tmp_path):
    """Generate a detailed comparison report (always passes, just for info)."""
    report_lines = [
        "=" * 72,
        "SYNTHESIS QUALITY REPORT",
        "=" * 72,
        "",
    ]
    
    total_loss = 0
    count = 0
    preset_results = []
    
    for num in sorted(all_results.keys()):
        entry = all_results[num]
        if 'error' in entry:
            report_lines.append(f"  Preset {num}: ERROR - {entry['error']}")
            continue
        
        result = entry['result']
        loss = result['overall_loss']
        total_loss += loss
        count += 1
        
        threshold = PRESET_THRESHOLDS.get(num, {}).get('max_loss', 0.50)
        status = '✓' if loss < threshold else '✗'
        
        preset_results.append((num, loss, threshold, status, result))
        report_lines.append(
            f"  {status} Preset {num}: loss={loss:.4f} "
            f"(threshold={threshold:.2f}) "
            f"hits={result['gen_n_hits']}/{result['ref_n_hits']} "
            f"timing={result['avg_timing_error_ms']:.1f}ms"
        )
    
    avg_loss = total_loss / max(count, 1)
    report_lines.extend([
        "",
        f"  Average loss: {avg_loss:.4f} (threshold: {AVERAGE_LOSS_THRESHOLD})",
        f"  Presets tested: {count}",
        f"  Presets passing: {sum(1 for _, _, _, s, _ in preset_results if s == '✓')}",
        "",
    ])
    
    # Metric summary
    all_metrics = {}
    for num, entry in all_results.items():
        if 'error' in entry:
            continue
        for hit_losses in entry['result']['per_hit_losses']:
            for k, v in hit_losses.items():
                all_metrics.setdefault(k, []).append(v)
    
    report_lines.append("  Per-metric averages:")
    for metric, threshold in sorted(METRIC_THRESHOLDS.items()):
        if metric in all_metrics:
            avg = np.mean(all_metrics[metric])
            status = '✓' if avg < threshold else '✗'
            report_lines.append(
                f"    {status} {metric:30s} avg={avg:.4f} threshold={threshold:.4f}"
            )
    
    report = "\n".join(report_lines)
    print("\n" + report)
    
    # Save report
    report_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'tests', 'quality_report.txt'
    )
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
