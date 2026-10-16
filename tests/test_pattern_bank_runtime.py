"""
The pattern bank model in the app's runtime (pythonic/pattern_generator.py)
and its worker (pythonic/app/ai_worker.py), over a tiny checkpoint trained
by train_pattern_bank.py: the runtime's copy of the model gives what the
training script's gives, generation makes whole banks with lengths and
chains, a re-roll redraws one slot in the bank's style and keeps its chain,
and a single-pattern checkpoint still loads and generates.
"""

import importlib.util
import pathlib

import numpy as np
import pytest

torch = pytest.importorskip('torch')

import train_pattern_bank as tpb  # noqa: E402
from pythonic import pattern_generator as pg  # noqa: E402
from tests.test_pattern_bank_cache import TINY, dataset, preset  # noqa: E402,F401

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = [{'OscFreq': 50.0 * (c + 1), 'OscDcy': 300.0, 'Mix': 60.0, 'NFilFrq': 2500.0, 'DistAmt': 5.0}
       for c in range(8)]


@pytest.fixture
def checkpoint(dataset, tmp_path):  # noqa: F811
    """A tiny bank checkpoint with its kit prior, and the cache."""
    cache = tpb.build_bank_cache(str(dataset), str(tmp_path / 'cache.pt'), workers=1)
    train, valid = tpb.split_indexes(cache, valid_gens=('gen_002',))
    best = tmp_path / 'bank.pt'
    model, _ = tpb.train_bank_model(cache, train, valid, epochs=2, device='cpu', best_path=str(best),
                                    final_path=str(tmp_path / 'final.pt'), model_kwargs=TINY, batch_size=2)
    prior, stats = tpb.fit_kit_prior(model, cache, train, epochs=2, hidden=16, device='cpu')
    tpb.save_kit_prior(str(best), prior, stats)
    return best, cache


def loaded(path):
    generator = pg.PatternGenerator()
    generator.load_model(str(path))
    return generator


def test_the_runtime_model_matches_the_training_script(checkpoint):
    path, cache = checkpoint
    runtime = loaded(path)
    trained, ckpt = tpb.load_bank_model(str(path))
    prior = tpb.load_kit_prior(ckpt)
    assert runtime.is_bank and runtime.is_loaded and runtime.sampling_summary == 'bank'

    torch.manual_seed(0)
    z_bank, z_slot, cond = torch.randn(3, 8), torch.randn(3, 12, 4), cache['cond']
    grid, length, chain, _ = next(tpb.BankDataset(cache).batches(3))
    with torch.no_grad():
        for mine, theirs in zip(runtime._bank.decode(z_bank, z_slot, cond), trained.decode(z_bank, z_slot, cond)):
            assert torch.allclose(mine, theirs, atol=1e-5)
        bank_mu, slot_mu = runtime._bank.encode(grid, length, chain, cond)
        expected = trained.encode(grid, length, chain, cond)
        assert torch.allclose(bank_mu, expected[0], atol=1e-5) and torch.allclose(slot_mu, expected[2], atol=1e-5)
        noise = torch.randn(3, 8 + 48)
        assert torch.allclose(runtime._prior.net(cond) + noise @ runtime._prior.scale_tril.T,
                              prior(cond, noise), atol=1e-5)

    # The runtime encodes pattern dicts as the cache does
    banks = tpb.unpack_grid(cache['grid'].numpy())
    dicts = tpb.decode_bank(banks[0], cache['length'][0].numpy(), cache['chain'][0].numpy())
    g, l, c = pg.encode_bank_patterns(dicts)
    assert (g == grid[0].numpy()).all() and (l == length[0].numpy()).all() and (c == chain[0].numpy()).all()


def test_generation_makes_whole_banks_with_lengths_and_chains(checkpoint):
    path, _ = checkpoint
    generator = loaded(path)
    bank = generator.generate(RAW, tempo=128, n=12, temperature=1.0, seed=4)
    assert len(bank) == 12 and bank == generator.generate(RAW, tempo=128, n=12, temperature=1.0, seed=4)
    assert bank[11]['Chained'] is False
    for pattern in bank:
        assert 1 <= pattern['Length'] <= 16 and isinstance(pattern['Chained'], bool)
        assert all(len(pattern[str(ch)]['Triggers']) == 16 for ch in range(1, 9))
    assert len(generator.generate(RAW, n=1, seed=1)) == 1
    assert len(generator.generate(RAW, n=13, seed=1)) == 13  # two banks, cut
    named = generator.generate_bank(RAW, tempo=128, temperature=1.0, seed=4)
    assert list(named) == list('ABCDEFGHIJKL') and named['A'] == bank[0]


def test_a_reroll_redraws_one_slot_and_keeps_its_chain(checkpoint):
    path, _ = checkpoint
    generator = loaded(path)
    bank = generator.generate(RAW, n=12, seed=4)
    bank[3]['Chained'] = True
    pattern = generator.reroll(RAW, bank, 3, seed=9)
    assert pattern == generator.reroll(RAW, bank, 3, seed=9)
    assert pattern['Chained'] is True and 1 <= pattern['Length'] <= 16
    assert generator.reroll(RAW, bank, 11, seed=9)['Chained'] is False
    with pytest.raises(ValueError):
        generator.reroll(RAW, bank, 12)
    with pytest.raises(ValueError):
        generator.reroll(RAW, bank[:11], 3)


def test_a_bank_without_kit_prior_samples_from_the_normal(checkpoint, tmp_path):
    path, _ = checkpoint
    ckpt = torch.load(path, weights_only=False)
    del ckpt['kit_prior']
    bare = tmp_path / 'bare.pt'
    torch.save(ckpt, bare)
    generator = loaded(bare)
    assert generator.sampling_summary == 'bank, no kit prior'
    assert len(generator.generate(RAW, n=12, seed=1)) == 12
    assert generator.reroll(RAW, generator.generate(RAW, n=12, seed=1), 0, seed=2)['Length'] >= 1


def test_a_single_pattern_checkpoint_still_loads(checkpoint, tmp_path):
    _, cache = checkpoint
    legacy = pg._build_pattern_model(384, 48, 8, 32)
    path = tmp_path / 'single.pt'
    torch.save({'model_state': legacy.state_dict(), 'scaler': cache['scaler'],
                'model_config': {'pattern_dim': 384, 'cond_dim': 48, 'latent_dim': 8, 'hidden_dim': 32}}, path)
    generator = loaded(path)
    assert not generator.is_bank and generator.sampling_summary is None
    bank = generator.generate(RAW, n=12, seed=1)
    assert len(bank) == 12 and not any(p['Chained'] for p in bank)
    assert generator.reroll(RAW, bank, 5, seed=3) == generator.generate(RAW, n=1, seed=3)[0]


def test_the_worker_rerolls_with_the_face_bank(checkpoint):
    path, _ = checkpoint
    spec = importlib.util.spec_from_file_location('ai_worker_test', ROOT / 'pythonic' / 'app' / 'ai_worker.py')
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    models = worker.Models()
    assert models.handle({'op': 'load', 'kind': 'pattern', 'path': str(path)})['sampling'] == 'bank'
    request = {'op': 'patterns', 'path': str(path), 'raw_patches': RAW, 'tempo': 120, 'swing': 0,
               'fill_rate': 4, 'step_rate': '1/16', 'n': 12, 'temperature': 1.0, 'seed': 1}
    reply = models.handle(request)
    bank = reply['patterns']
    assert len(bank) == 12 and reply['sampling'] == 'bank'  # the model line keeps saying so
    rerolled = models.handle({**request, 'n': 1, 'bank': bank, 'slot': 2})['patterns']
    assert len(rerolled) == 1 and rerolled[0]['Chained'] == bank[2]['Chained']
    assert np.isfinite(len(rerolled[0]['1']['Triggers']))
