"""
The sample fitter's metal analysis: the square-wave bank behind a metallic
sample is found from its partials, shared across the samples of a machine,
and never found in a drum without one.
"""

import numpy as np
import pytest

from pythonic.voice import DrumVoice, VoiceParams

F = pytest.importorskip('tools.fit_samples')

SR = F.SR
BANK_606 = (286.8, 401.5, 480.7, 553.4, 679.8, 971.8)


def render(params, seconds):
    voice = DrumVoice(SR, seed=5)
    voice.set_params(params)
    voice.trigger(127)
    return voice.process(int(seconds * SR))[:, 0].astype(np.float64)


def cymbal(seconds=1.5, bank=BANK_606):
    return render(VoiceParams(osc_mix=0.0, noise_decay_ms=10.0, osc_decay_ms=10.0,
                              **{f'metal_freq{i}': f for i, f in enumerate(bank, 1)},
                              metal_band1_freq=7000.0, metal_band1_decay_ms=1500.0, metal_band1_level=1.0,
                              metal_band2_freq=3500.0, metal_band2_decay_ms=800.0, metal_band2_level=0.5,
                              metal_band1_q=1.5, metal_band2_q=1.5, metal_hp_freq=2000.0), seconds)


def test_the_bank_of_a_metal_sound_is_found():
    found = F.metal_fundamentals(cymbal())
    assert found == pytest.approx(BANK_606, rel=0.002)


def test_a_short_hit_takes_the_bank_of_its_machine():
    other = (205.3, 304.4, 369.6, 522.7, 540.0, 800.0)
    hit = cymbal(seconds=0.12)
    assert F.metal_fundamentals(hit, banks=[other, BANK_606]) == BANK_606


def test_drums_without_a_bank_have_none():
    kick = render(VoiceParams(osc_freq=55.0, osc_decay_ms=400.0, osc_mix=1.0), 0.6)
    snare = render(VoiceParams(osc_freq=180.0, osc_mix=0.4, noise_filter=2, noise_freq=2000.0,
                               noise_decay_ms=200.0), 0.4)
    assert F.metal_fundamentals(kick) == () and F.metal_fundamentals(snare) == ()


def test_the_metal_controls_ride_on_the_vector():
    p = F.metal_starts(VoiceParams(), cymbal(), BANK_606)[0]
    u = F.encode(p, metal=True)
    assert len(u) == len(F.CONTINUOUS) + len(F.METAL_CONTINUOUS)
    q = F.decode(u, F.switches_of(p), BANK_606)
    assert q.metal_freq6 == BANK_606[5]
    for name, *_ in F.METAL_CONTINUOUS:
        assert getattr(q, name) == pytest.approx(getattr(p, name), rel=1e-6)
    assert F.decode(u[:len(F.CONTINUOUS)], F.switches_of(p)).metal_band1_level == 0.0


def test_partials_above_2_khz_count_only_where_there_is_energy_there():
    t = np.arange(int(0.5 * SR)) / SR
    kick = np.sin(2 * np.pi * 55.0 * t) * np.exp(-t / 0.12)
    target = F.Target(kick)
    y = np.concatenate([np.sin(2 * np.pi * 60.0 * t) * np.exp(-t / 0.1), np.zeros(target.n - len(t))])
    *_, high_err, tonal_err, _balance, _partials = target.loss(y, parts=True)
    assert target.high_share < 1e-3
    assert F.HIGH_WEIGHT * high_err + F.TONAL_WEIGHT * tonal_err < 1e-2


def test_a_metal_fit_leaves_the_eq_flat():
    p = F.metal_starts(VoiceParams(eq_gain_db=12.0), cymbal(), BANK_606)[0]
    assert p.eq_gain_db == 0.0
    u = F.encode(VoiceParams(eq_gain_db=20.0, metal_band1_level=0.5), metal=True)
    assert F.decode(u, (0, 0, 0, 0), BANK_606).eq_gain_db == 0.0
    assert F.decode(u[:len(F.CONTINUOUS)], (0, 0, 0, 0)).eq_gain_db == pytest.approx(20.0)


def test_an_acoustic_cymbal_gets_no_bank():
    # the modes of a real cymbal: many partials at unrelated frequencies, some noise
    rng = np.random.default_rng(3)
    t = np.arange(int(1.5 * SR)) / SR
    modes = rng.uniform(2000.0, 14000.0, 80)
    x = sum(np.sin(2 * np.pi * f * t + rng.uniform(0, 6.3)) * rng.uniform(0.2, 1.0) for f in modes)
    x = (x + 2.0 * rng.standard_normal(len(t))) * np.exp(-t / 0.6)
    assert F.tonality(F.spectra(x)[1]) > 0.3  # tonal enough to be looked at
    assert F.metal_fundamentals(x) == ()


def test_the_patches_mirror_the_sample_folders(tmp_path, monkeypatch):
    import soundfile as sf
    samples, out = tmp_path / 'samples', tmp_path / 'patches'
    kick = render(VoiceParams(osc_freq=60.0, osc_decay_ms=150.0, osc_mix=1.0), 0.25)
    for sub in ('Kicks', 'Kicks/Deep', 'Other'):
        (samples / sub).mkdir(parents=True)
        sf.write(samples / sub / 'Hit.wav', kick, SR)
    monkeypatch.setattr('sys.argv', ['fit_samples.py', str(samples), str(out), '--wav', '--prior', '',
                                     '--screen', '1', '--screen-evals', '4', '--starts', '1',
                                     '--evals', '8', '--restarts', '0', '--jobs', '1'])
    F.main()
    for sub in ('Kicks', 'Kicks/Deep', 'Other'):
        assert (out / sub / 'Hit.mtdrum').is_file() and (out / sub / 'Hit.fit.wav').is_file()
    report = (out / 'fit_report.csv').read_text()
    assert 'Kicks/Deep/Hit.mtdrum' in report.replace('\\', '/')
