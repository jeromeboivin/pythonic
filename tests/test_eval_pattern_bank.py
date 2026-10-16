"""
tools/eval_pattern_bank.py: the bank statistics on banks whose answers are
known (chains, runs, lengths, density, coherence), near copies found by
the novelty check, and a report over a tiny trained model and the legacy
single-pattern model on the presets of the bank cache tests.
"""

import json
import sys

import numpy as np
import pytest

torch = pytest.importorskip('torch')

import train_pattern_bank as tpb  # noqa: E402
from tests.test_pattern_bank_cache import TINY, dataset, preset  # noqa: E402,F401
from tools import eval_pattern_bank as ev  # noqa: E402


def banks(n=4):
    grid = np.zeros((n, 12, 8, 16, 3), bool)
    length = np.full((n, 12), 16)
    chain = np.zeros((n, 11), bool)
    return grid, length, chain


def test_chains_runs_and_lengths():
    grid, length, chain = banks(4)
    chain[0, [0, 2, 3]] = True  # a-b, c-e
    chain[1, :] = True          # a-l
    length[2, :6] = 8
    stats = ev.bank_stats(grid, length, chain, np.random.default_rng(0))
    assert stats['chain_rate'] == pytest.approx(14 / 44)
    assert stats['banks_with_chain'] == 0.5
    assert stats['chain_rate_by_link'][0] == 0.5 and stats['chain_rate_by_link'][1] == 0.25
    assert stats['chain_runs'] == {'2': 1 / 3, '3': 1 / 3, '4': 0.0, '5-11': 0.0, '12': 1 / 3}
    assert stats['lengths'] == {'16': 42 / 48, '8': 6 / 48, '12': 0.0, 'other': 0.0}


def test_density_counts_playing_steps_and_coherence_separates_banks():
    grid, length, chain = banks(6)
    rng = np.random.default_rng(3)
    for b in range(6):  # every pattern of a bank is the same; banks differ
        grid[b, :, :, :, 0] = rng.random((8, 16)) < 0.4
    grid[:, :, 2, :, 1] = True  # channel 3 accents every step
    length[:, 0] = 8
    grid[:, 0, :, 8:] = False
    stats = ev.bank_stats(grid, length, chain, np.random.default_rng(0))
    assert stats['density']['accent'] == pytest.approx(1 / 8)
    assert stats['jaccard_within_bank'] < 1.0  # pattern A plays half its steps
    assert stats['jaccard_within_bank'] > stats['jaccard_across_banks']
    assert stats['same_lane_adjacent'] > 0.5
    assert len(stats['density_by_channel']) == 8


def test_jaccard_and_novelty_find_near_copies():
    a = np.array([[1, 1, 0, 0], [0, 0, 0, 0]], bool)
    assert ev.jaccard(a, np.array([[1, 0, 0, 0], [0, 0, 0, 0]], bool)).tolist() == [0.5, 1.0]
    rng = np.random.default_rng(0)
    train = rng.random((50, 128)) < 0.3
    copies = train[:10].copy()
    fresh = ~train[10:20]
    assert ev.novelty(copies, train)['near_copies'] == 1.0
    result = ev.novelty(fresh, train)
    assert result['near_copies'] == 0.0 and result['closest_train_jaccard'] < 0.6


def test_a_report_over_a_tiny_model_and_the_legacy_model(dataset, tmp_path, monkeypatch, capsys):  # noqa: F811
    cache_path = tmp_path / 'cache.pt'
    cache = tpb.build_bank_cache(str(dataset), str(cache_path), workers=1)
    train, valid = tpb.split_indexes(cache, valid_gens=('gen_002',))
    best = tmp_path / 'best.pt'
    tpb.train_bank_model(cache, train, valid, epochs=2, device='cpu', best_path=str(best),
                         final_path=str(tmp_path / 'final.pt'), model_kwargs=TINY, batch_size=2)

    from pythonic.pattern_generator import _build_pattern_model
    legacy = _build_pattern_model(384, 48, 8, 32)
    legacy_path = tmp_path / 'legacy.pt'
    torch.save({'model_state': legacy.state_dict(), 'scaler': cache['scaler'],
                'model_config': {'pattern_dim': 384, 'cond_dim': 48, 'latent_dim': 8, 'hidden_dim': 32}},
               legacy_path)

    samplers = [ev.BankSampler(best, length_mode='sample', chain_mode='bernoulli'), ev.LegacySampler(legacy_path)]
    report = ev.evaluate(cache, samplers, novelty_banks=5, valid_gens=('gen_002',))
    assert set(report) == {'data', 'bank', 'legacy'}
    assert report['data']['banks'] == 1 and report['data']['chain_rate_by_link'][10] == 1.0
    assert report['legacy']['chain_rate'] == 0.0 and report['legacy']['lengths']['16'] == 1.0
    for name in ('bank', 'legacy'):
        assert {'condition_fidelity', 'kit_sensitivity', 'novelty'} <= set(report[name])

    # The command line writes the report without the per-bank arrays
    out = tmp_path / 'report.json'
    monkeypatch.setattr(sys, 'argv', ['eval', '--cache', str(cache_path), '--bank', str(best),
                                      '--legacy', str(legacy_path), '--valid-gens', 'gen_002',
                                      '--json', str(out)])
    ev.main()
    written = json.loads(out.read_text())
    assert set(written) == {'data', 'bank', 'legacy'} and '_per_channel' not in written['bank']
    assert 'jaccard within bank' in capsys.readouterr().out
