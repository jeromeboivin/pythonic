"""
The sample fitter's metal analysis: the square-wave bank behind a metallic
sample is found from its partials, shared across the samples of a machine,
and never found in a drum without one.
"""

import math

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
    _, _, _, high_err, tonal_err, *_ = target.loss(y, parts=True)
    assert target.high_share < 1e-3
    assert F.HIGH_WEIGHT * high_err + F.TONAL_WEIGHT * tonal_err < 1e-2


def test_a_patch_ringing_past_a_cut_short_sample_is_penalised():
    t = np.arange(int(1.0 * SR)) / SR
    tom = np.sin(2 * np.pi * 120.0 * t) * np.exp(-t / 0.4)
    cut = tom[:int(0.15 * SR)]                     # a ROM sample stopping 3 dB under its peak
    target = F.Target(cut)
    assert target.n >= len(cut) + int(F.TAIL_SECONDS * SR)
    stops = np.concatenate([cut, np.zeros(target.n - len(cut))])
    assert target.loss(tom[:target.n]) > target.loss(stops) + 1.0
    # cut by --max-seconds, the sample goes on: compared only up to the cut
    assert F.Target(cut, ended=False).n == len(cut)


def test_a_band_passed_noise_is_never_narrow_enough_to_ring():
    from pythonic.voice import FILT_BP, FILT_LP
    u = F.encode(VoiceParams(noise_q=20.0, noise_filter=FILT_BP))
    assert F.decode(u, F.switches_of(VoiceParams(noise_filter=FILT_BP))).noise_q == F.NOISE_BP_Q_MAX
    assert F.decode(u, F.switches_of(VoiceParams(noise_filter=FILT_LP))).noise_q == pytest.approx(20.0)
    # toms keep the full range: their narrow noise is their note
    assert F.decode(u, F.switches_of(VoiceParams(noise_filter=FILT_BP)), bp_q_max=math.inf).noise_q == pytest.approx(20.0)


def test_a_recording_off_zero_is_brought_back(tmp_path):
    import soundfile as sf
    t = np.arange(int(0.2 * SR)) / SR
    hat = np.random.default_rng(1).standard_normal(len(t)) * np.exp(-t / 0.02) * 0.5
    sf.write(tmp_path / 'hat.wav', hat - 0.03, SR, subtype='FLOAT')
    x = F.load_target(str(tmp_path / 'hat.wav'), 2.5)
    assert abs(np.median(x[-200:])) < 1e-4
    # a lopsided hit ending on zero keeps its shape
    kick = np.sin(2 * np.pi * 50.0 * t) * np.exp(-t / 0.03) + 0.2 * np.exp(-t / 0.01)
    assert abs(F.dc_offset(kick)) < 0.002          # its ring, far under a capture's offset


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


def test_a_fit_keeps_to_the_waves_it_is_given(tmp_path, monkeypatch):
    import soundfile as sf
    from pythonic.preset_manager import DrumPatchParser
    samples, out = tmp_path / 'samples', tmp_path / 'patches'
    samples.mkdir()
    tom = render(VoiceParams(wave=1, osc_freq=120.0, osc_decay_ms=250.0, osc_mix=1.0), 0.4)
    sf.write(samples / 'Tom.wav', tom, SR)
    monkeypatch.setattr('sys.argv', ['fit_samples.py', str(samples), str(out), '--prior', '', '--waves', 'sine,saw',
                                     '--screen', '1', '--screen-evals', '4', '--starts', '1',
                                     '--evals', '8', '--restarts', '0', '--jobs', '1'])
    F.main()
    assert DrumPatchParser().parse_file(str(out / 'Tom.mtdrum'))['OscWave'] != 'Triangle'


def test_only_snares_and_cymbals_get_a_triangle_by_default():
    triangle = F.WAVE_NAMES.index('triangle')
    assert triangle in F.waves_for(['sd']) and triangle in F.waves_for(['cy'])
    for cats in (['bd'], ['tom'], ['clap'], ['perc', 'sd'], ['cowbell']):
        assert triangle not in F.waves_for(cats)


def test_a_spiky_render_of_a_kick_loses_its_punch():
    # A squared sine with a fast sweep: as loud in its body as at its strike. The same
    # sound with a click on top has nearly its spectra, but its level set by the peak
    # leaves the body some 12 dB quieter, and only the punch term hears it.
    t = np.arange(int(0.4 * SR)) / SR
    phase = 2 * np.pi * np.cumsum(50.0 + 150.0 * np.exp(-t / 0.02)) / SR
    kick = np.tanh(3.0 * np.sin(phase)) * np.exp(-t / 0.3)
    target = F.Target(kick)
    pad = lambda y: np.concatenate([y, np.zeros(target.n - len(y))])
    spiky = kick.copy()
    spiky[200] += 3.0
    same = target.loss(pad(kick), parts=True)
    worse = target.loss(pad(spiky), parts=True)
    assert same[7] < 0.1
    assert worse[7] > 6.0
    spectral = lambda p: p[0] + F.MAG_WEIGHT * p[1]
    assert spectral(worse) - spectral(same) < F.PUNCH_WEIGHT * worse[7]


def test_the_punch_of_a_short_sample_skips_the_windows_past_its_end():
    d = F.density(np.ones(int(0.04 * SR)))
    assert np.allclose(d[:3], 0.0) and np.isnan(d[3:]).all()


def test_the_hiss_under_a_quiet_tail_is_not_asked_of_the_patch():
    # A kick recorded with hiss 70 dB under its peak: a clean render matches its octave
    # balance, so the fit has no reason to add noise to the tail
    t = np.arange(int(0.8 * SR)) / SR
    kick = np.sin(2 * np.pi * np.cumsum(50.0 + 150.0 * np.exp(-t / 0.02)) / SR) * np.exp(-t / 0.08)
    hiss = np.random.default_rng(0).standard_normal(len(t)) * 10.0 ** (-70.0 / 20.0)
    target = F.Target(kick + hiss)
    balance = target.loss(np.concatenate([kick, np.zeros(target.n - len(t))]), parts=True)[5]
    assert balance < 0.5


def test_a_strike_that_starts_too_high_is_heard():
    # A kick holding 240 Hz for 7 ms before its sweep: the same sweep without the hold
    # starts higher and falls sooner, and the strike term scores it worse
    held = render(VoiceParams(wave=0, osc_freq=60.0, osc_decay_ms=400.0, mod_rate=60.0, mod_amount=24.0,
                              mod_hold_ms=7.0, osc_mix=1.0), 0.5)
    target = F.Target(held)
    assert target.strike is not None
    pad = lambda y: np.concatenate([y, np.zeros(target.n - len(y))])
    same = target.loss(pad(held), parts=True)[8]
    swept = render(VoiceParams(wave=0, osc_freq=60.0, osc_decay_ms=400.0, mod_rate=60.0, mod_amount=30.0,
                               osc_mix=1.0), 0.5)
    assert same < 0.1
    assert target.loss(pad(swept), parts=True)[8] > 2.0


def test_a_strike_the_render_comes_in_too_late_for_is_missed():
    # The same kick swelling in over 12 ms has no pitch where the sample strikes: those
    # milliseconds count as missed, not as left out
    kick = dict(wave=0, osc_freq=60.0, osc_decay_ms=400.0, mod_rate=60.0, mod_amount=24.0, osc_mix=1.0)
    target = F.Target(render(VoiceParams(**kick), 0.5))
    late = render(VoiceParams(**kick, osc_attack_ms=12.0), 0.5)
    assert target._strike_err(np.concatenate([late, np.zeros(target.n - len(late))])) > 4.0


def test_one_stray_half_cycle_does_not_unpitch_a_sweep(monkeypatch):
    sweep = np.log2([157.0, 120.0, 97.0, 81.0, 72.0, 48.0, 64.0, 67.0])   # the 707's Kick 01
    monkeypatch.setattr(F, '_half_cycles', lambda x: (np.arange(8.0), sweep))
    assert F.strike_jitter(None) > F.STRIKE_JITTER > F.strike_jitter(None, strays=False)


def test_a_clap_keeps_the_burst_envelope():
    from pythonic.voice import NENV_MOD
    assert F.noise_envs_for(['clap']) == (NENV_MOD,)
    assert len(F.noise_envs_for(['sd'])) == 3


def test_noise_has_no_strike_pitch():
    noise = np.random.default_rng(1).uniform(-1.0, 1.0, int(0.3 * SR)) * np.exp(-np.arange(int(0.3 * SR)) / (0.05 * SR))
    assert F.Target(noise).strike is None


def test_a_fitted_hold_is_written_to_the_patch():
    from pythonic.preset_manager import DrumPatchParser, DrumPatchWriter
    from pythonic.voice import params_from_patch
    p = VoiceParams(mod_hold_ms=6.0)
    text = DrumPatchWriter.format_patch(F.patch_dict(p, 'kick', False))
    assert params_from_patch(DrumPatchParser().parse_string(text)).mod_hold_ms == 6.0
    assert 'ModHold' not in DrumPatchWriter.format_patch(F.patch_dict(VoiceParams(), 'kick', False))
