"""
Regression tests against the reference renders.

* Single hits from test_patches/ (oscillator-only patches are deterministic and
  must match sample for sample, to the 16-bit quantisation floor).
* The beat loops in tests/ (<machine>.mtpreset + <machine>.wav), rendered
  through the app's own preset renderer. Noise is random, so loops are checked
  on spectrum, envelope, energy and waveform correlation.
"""

import glob
import os

import numpy as np
import pytest
import soundfile as sf

from pythonic.voice import DrumVoice, params_from_patch
from pythonic.preset_manager import PythonicPresetParser
from tools.render_preset import load_and_render_preset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATCH_DIR = os.path.join(ROOT, 'test_patches')
TEST_DIR = os.path.join(ROOT, 'tests')


def _calibration_patches():
    patches = {}
    for f in glob.glob(os.path.join(PATCH_DIR, '*.mtpreset')):
        for d in PythonicPresetParser().parse_file(f)['DrumPatches'].values():
            patches[d['Name']] = d
    return patches


# Deterministic hits (no noise in the mix) and the velocity they were played at.
DETERMINISTIC_HITS = {
    'TEST Sine Pure': 127, 'TEST Tri Pure': 127, 'TEST Saw Pure': 127,
    'TEST Pitch Mod': 127, 'TEST ModRate 50': 127, 'TEST Atk 20ms': 127,
    'TEST Dist 50': 127, 'TEST Dist 100': 127, 'TEST EQ Boost': 127, 'TEST EQ Cut': 127,
    'TEST Mix 100-0 Simple': 127,
    'ATK 5ms': 127, 'ATK 100ms': 127, 'DCY 100ms': 127, 'DCY 1000ms': 127,
    'VEL 50': 64, 'VEL 100': 64, 'MOD VEL': 64,
}


@pytest.fixture(scope='module')
def calibration_patches():
    if not os.path.isdir(PATCH_DIR):
        pytest.skip('test_patches/ not available')
    return _calibration_patches()


def _render_hit(patch, velocity, n, sr, seed=0):
    voice = DrumVoice(sr, seed=seed)
    voice.set_params(params_from_patch(patch))
    voice.trigger(velocity)
    return np.concatenate([voice.process(min(512, n - i)) for i in range(0, n, 512)]).astype(float)


@pytest.mark.parametrize('name', sorted(DETERMINISTIC_HITS))
def test_deterministic_hit_matches_sample_for_sample(calibration_patches, name):
    path = os.path.join(PATCH_DIR, f'{name}.wav')
    if not os.path.exists(path) or name not in calibration_patches:
        pytest.skip(f'{name} not available')
    ref, sr = sf.read(path)
    n = len(ref)
    gen = _render_hit(calibration_patches[name], DETERMINISTIC_HITS[name], n, sr)
    # the recordings start up to a few samples late (4-sample trigger grid)
    best = min(
        np.linalg.norm(ref[max(l, 0):n + min(l, 0)] - gen[max(-l, 0):n - max(l, 0)])
        for l in range(-6, 7)
    )
    resid_db = 20 * np.log10(best / np.linalg.norm(ref))
    assert resid_db < -45.0, f'{name}: residual {resid_db:.1f} dB'


@pytest.mark.parametrize('name, velocity', [
    ('TEST Noise LP', 127), ('TEST Noise BP', 127), ('TEST Noise HP', 127),
    ('NOISE EXP', 127), ('TEST Mix 50-50', 127),
    ('SC SD Elec Long', 64), ('SC CY Elec', 64), ('SC Chop Elec L', 64),
])
def test_noise_hit_energy(calibration_patches, name, velocity):
    path = os.path.join(PATCH_DIR, f'{name}.wav')
    if not os.path.exists(path) or name not in calibration_patches:
        pytest.skip(f'{name} not available')
    ref, sr = sf.read(path)
    energies = [np.sum(_render_hit(calibration_patches[name], velocity, len(ref), sr, seed) ** 2)
                for seed in range(6)]
    diff_db = 10 * np.log10(np.sum(ref ** 2) / np.mean(energies))
    assert abs(diff_db) < 1.5, f'{name}: energy differs by {diff_db:+.2f} dB'


# --------------------------------------------------------------------------- loops
LOOPS = ['505', '707', '808', '909', 'DMX', 'LM2']


def _bands_db(x, sr):
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / sr)
    centres = 31.25 * 2 ** (np.arange(0, 30) / 3)
    return np.array([10 * np.log10(spec[(f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6))].sum() + 1e-12)
                     for c in centres])


def _env_db(x, sr, ms=10):
    h = int(sr * ms / 1000)
    n = len(x) // h
    return 20 * np.log10(np.sqrt(np.mean(x[:n * h].reshape(n, h) ** 2, axis=1)) + 1e-9)


@pytest.mark.parametrize('machine', LOOPS)
def test_loop_matches_reference(machine):
    preset, wav = f'{machine}.mtpreset', f'{machine}.wav'
    ref, sr = sf.read(os.path.join(TEST_DIR, wav))
    n = len(ref)
    gen = load_and_render_preset(os.path.join(TEST_DIR, preset), duration_seconds=n / sr,
                                 sample_rate=sr)[0][:n].astype(float)

    corr = np.sum(ref * gen) / np.sqrt(np.sum(ref ** 2) * np.sum(gen ** 2))
    energy_db = 10 * np.log10(np.sum(ref ** 2) / np.sum(gen ** 2))
    m_ref, m_gen = ref.mean(1), gen.mean(1)
    br, bg = _bands_db(m_ref, sr), _bands_db(m_gen, sr)
    keep = br > br.max() - 40
    band_err = np.mean(np.abs(br[keep] - bg[keep]))
    er, eg = _env_db(m_ref, sr), _env_db(m_gen, sr)
    keep = er > er.max() - 40
    env_err = np.mean(np.abs(er[keep] - eg[keep]))

    # Thresholds sit just above the noise-limited values of the current engine
    # (corr 0.87-0.95, band err 0.15-0.4 dB, env err 0.1-1.1 dB).
    assert abs(energy_db) < 0.5, f'{wav}: energy {energy_db:+.2f} dB'
    assert band_err < 0.8, f'{wav}: band error {band_err:.2f} dB'
    assert env_err < 1.6, f'{wav}: envelope error {env_err:.2f} dB'
    assert corr > 0.85, f'{wav}: correlation {corr:.3f}'
