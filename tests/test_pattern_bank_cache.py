"""
train_pattern_bank.py. The bank cache: one preset is one bank of 12
patterns with their lengths and chain flags, packed and unpacked without
loss, steps past a pattern's length masked, the condition as the app's
runtime builds it, and the round-trip check against the parsed presets.
The model: a tiny bank CVAE trains, saves and loads, ignores steps past a
pattern's length, and samples valid banks. The presets are written here,
in the dataset's plain-number form.
"""

import numpy as np
import pytest

torch = pytest.importorskip('torch')

import train_pattern_bank as tpb  # noqa: E402

KIT = {'OscFreq': 55.0, 'OscDcy': 400.0, 'Mix': 70.0, 'NFilFrq': 3000.0, 'DistAmt': 10.0}


def lane(seed, channel, slot):
    """A 16-step string of # and -, different per preset, channel and slot."""
    rng = np.random.default_rng(seed * 1000 + channel * 20 + slot)
    return ''.join('#' if on else '-' for on in rng.random(16) < 0.35)


def preset(seed, chains=(), lengths=None, tempo=120):
    """A preset with 8 drum patches and 12 patterns (A-L). chains: the slots
    (0 = A) chained to the next; lengths: {slot: length}. Pattern L has no
    Chained key, as in the dataset."""
    lengths = lengths or {}
    out = ['PresetV3: {', f'\tName = "test {seed}"', f'\tTempo = {tempo}', '\tSwing = 12.5',
           '\tFillRate = 2', '\tStepRate = 1/16', '\tPattern = a', '\tDrumPatches: {']
    for ch in range(1, 9):
        out.append(f'\t\t{ch}: {{')
        out += [f'\t\t\t{k} = {v * (1 + 0.1 * ch + 0.01 * seed)}' for k, v in KIT.items()]
        out.append('\t\t}')
    out += ['\t}', '\tPatterns: {']
    for slot, key in enumerate(tpb.SLOT_KEYS):
        out.append(f'\t\t{key}: {{')
        if slot < 11:
            out.append(f'\t\tChained = {"true" if slot in chains else "false"}')
        out.append(f'\t\tLength = {lengths.get(slot, 16)}')
        for ch in range(1, 9):
            out += [f'\t\t\t{ch}: {{', f'\t\t\t\tAccents = {lane(seed + 50, ch, slot)}',
                    f'\t\t\t\tFills = {lane(seed + 90, ch, slot)}',
                    f'\t\t\t\tTriggers = {lane(seed, ch, slot)}', '\t\t\t}']
        out.append('\t\t}')
    out += ['\t}', '}', '']
    return '\n'.join(out)


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / 'patterns'
    for gen, name, text in [
        ('gen_001', 'plain.mtpreset', preset(1)),
        ('gen_001', 'chains.mtpreset', preset(2, chains=(0, 2, 3), tempo=95)),
        ('gen_002', 'short.mtpreset', preset(3, chains=(10,), lengths={1: 8, 4: 12})),
    ]:
        (root / gen).mkdir(parents=True, exist_ok=True)
        (root / gen / name).write_text(text, encoding='utf-8')
    (root / 'gen_002' / 'broken.mtpreset').write_text('not a preset', encoding='utf-8')
    return root


def test_the_cache_holds_one_bank_per_preset(dataset, tmp_path):
    cache = tpb.build_bank_cache(str(dataset), str(tmp_path / 'cache.pt'), workers=1)
    cache = tpb.load_bank_cache(str(tmp_path / 'cache.pt'))
    assert cache['names'] == ['gen_001/chains.mtpreset', 'gen_001/plain.mtpreset', 'gen_002/short.mtpreset']
    assert [path for path, _ in cache['skipped']] == ['gen_002/broken.mtpreset']
    assert cache['grid'].shape == (3, 12, 48) and cache['cond'].shape == (3, 48)
    assert cache['gens'] == ['gen_001', 'gen_002'] and cache['gen'].tolist() == [0, 0, 1]

    chain = cache['chain'].numpy()
    assert chain.shape == (3, 11)
    assert np.flatnonzero(chain[0]).tolist() == [0, 2, 3]  # A→B, C→D, D→E
    assert not chain[1].any()
    assert np.flatnonzero(chain[2]).tolist() == [10]  # K→L
    assert cache['length'][2].tolist() == [16, 8, 16, 16, 12] + [16] * 7

    # Steps past a pattern's length never play: they are zero in the grid
    grid = tpb.unpack_grid(cache['grid'].numpy())
    assert '#' in lane(3, 1, 1)[8:]  # the file has triggers there
    assert not grid[2, 1, :, 8:].any()
    assert ''.join('#' if on else '-' for on in grid[2, 1, 0, :8, 0]) == lane(3, 1, 1)[:8]

    assert tpb.round_trip_check(cache, str(dataset), samples=3) == 0


def test_pack_and_decode_are_lossless_and_mask_past_the_length():
    rng = np.random.default_rng(1)
    grid = rng.integers(0, 2, (5, 12, 8, 16, 3), dtype=np.uint8)
    assert (tpb.unpack_grid(tpb.pack_grid(grid)) == grid).all()
    length = np.full(12, 16, np.uint8)
    length[3] = 4
    chain = np.zeros(11, np.uint8)
    chain[5] = 1
    bank = tpb.decode_bank(grid[0], length, chain)
    assert [p['Chained'] for p in bank] == [s == 5 for s in range(12)]
    assert bank[3]['Length'] == 4 and set(bank[3]['1']['Triggers'][4:]) <= {'-'}
    back, back_length, back_chain = tpb.encode_bank(dict(zip(tpb.SLOT_KEYS, bank)))
    assert (back_length == length).all() and (back_chain == chain).all()
    expected = grid[0].copy()
    expected[3, :, 4:] = 0
    assert (back == expected).all()


def test_the_condition_matches_the_app_runtime(dataset, tmp_path):
    """The cache's condition is what pythonic/pattern_generator.py builds
    from the same kit and timing with the cache's scaler."""
    from pythonic.pattern_generator import PatternPreprocessor
    from pythonic.preset_manager import PythonicPresetParser

    cache = tpb.build_bank_cache(str(dataset), str(tmp_path / 'cache.pt'), workers=1)
    data = PythonicPresetParser().parse_file(str(dataset / 'gen_001' / 'plain.mtpreset'))
    runtime = PatternPreprocessor()
    runtime.load_scaler_state(cache['scaler'])
    raw = [data['DrumPatches'][str(c)] for c in range(1, 9)]
    cond = runtime.encode_condition(raw, float(data['Tempo']), float(data['Swing']),
                                    float(data['FillRate']), str(data['StepRate']))
    assert np.allclose(cond, cache['cond'][1].numpy(), atol=1e-5)


def test_the_split_holds_out_whole_gen_folders(dataset, tmp_path):
    cache = tpb.build_bank_cache(str(dataset), str(tmp_path / 'cache.pt'), workers=1)
    train, valid = tpb.split_indexes(cache, valid_gens=('gen_002',))
    assert train.tolist() == [0, 1] and valid.tolist() == [2]
    items = tpb.BankDataset(cache, valid)
    grid, length, chain, cond = items[0]
    assert grid.shape == (12, 384) and grid.dtype == torch.float32
    assert length.tolist()[:5] == [15, 7, 15, 15, 11]  # classes: length - 1
    assert chain.shape == (11,) and cond.shape == (48,)


# ---------------------------------------------------------------- the model

TINY = {'d_model': 32, 'n_heads': 2, 'enc_layers': 1, 'dec_layers': 1, 'bank_latent': 8,
        'slot_latent': 4, 'dropout': 0.0}


def batch(cache):
    items = tpb.BankDataset(cache)
    return next(items.batches(len(items)))


def test_the_model_outputs_a_bank_and_its_loss_ignores_steps_past_a_length(dataset, tmp_path):
    cache = tpb.build_bank_cache(str(dataset), str(tmp_path / 'cache.pt'), workers=1)
    grid, length, chain, cond = batch(cache)
    torch.manual_seed(0)
    model = tpb.BankCVAE(**TINY)
    out = model(grid, length, chain, cond)
    assert out[0].shape == (3, 12, 384) and out[1].shape == (3, 12, 16) and out[2].shape == (3, 11)
    assert out[3].shape == (3, 8) and out[5].shape == (3, 12, 4)

    logits = out[0].detach().requires_grad_()
    loss, parts = tpb.bank_loss((logits, *out[1:]), grid, length, chain, beta=0.5)
    assert torch.isfinite(loss) and set(parts) >= {'grid', 'length', 'chain', 'kl_bank', 'kl_slot', 'objective'}
    loss.backward()
    steps = logits.grad.reshape(3, 12, 8, 16, 3)
    assert not steps[2, 1, :, 8:].any()  # pattern B of the short preset plays 8 steps
    assert steps[2, 1, :, :8].abs().sum() > 0 and steps[1, 1, :, 8:].abs().sum() > 0


def test_a_short_training_run_saves_a_checkpoint_the_sampler_loads(dataset, tmp_path):
    cache = tpb.build_bank_cache(str(dataset), str(tmp_path / 'cache.pt'), workers=1)
    train, valid = tpb.split_indexes(cache, valid_gens=('gen_002',))
    best, final = tmp_path / 'best.pt', tmp_path / 'final.pt'
    model, history = tpb.train_bank_model(cache, train, valid, epochs=4, device='cpu', best_path=str(best),
                                          final_path=str(final), model_kwargs=TINY, batch_size=2)
    assert len(history) == 4 and all(np.isfinite(h['train_loss']) for h in history)
    assert {'val_f1_trigger', 'val_length_acc', 'val_chain_acc', 'val_au_bank'} <= set(history[-1])
    assert best.exists() and final.exists()

    loaded, ckpt = tpb.load_bank_model(str(best))
    assert ckpt['model_config']['kind'] == 'pattern_bank' and ckpt['scaler'] == cache['scaler']
    assert set(ckpt) >= {'model_state', 'model_config', 'scaler'}  # what the app's runtime reads
    # Saved and loaded, the decoder gives the same output
    model.eval()
    z_bank, z_slot = torch.randn(2, 8), torch.randn(2, 12, 4)
    again, _ = tpb.load_bank_model(str(final))
    with torch.no_grad():
        assert torch.allclose(model.decode(z_bank, z_slot, cache['cond'][:2])[0],
                              again.decode(z_bank, z_slot, cache['cond'][:2])[0], atol=1e-5)

    banks = tpb.sample_banks(loaded, cache['cond'][:3], generator=torch.Generator().manual_seed(1))
    assert len(banks) == 3
    for bank in banks:
        assert len(bank) == 12 and bank[11]['Chained'] is False
        for pattern in bank:
            assert 1 <= pattern['Length'] <= 16 and isinstance(pattern['Chained'], bool)
            for ch in range(1, 9):
                assert all(len(pattern[str(ch)][f]) == 16 for f in ('Triggers', 'Accents', 'Fills'))
                assert set(pattern[str(ch)]['Triggers'][pattern['Length']:]) <= {'-'}
        assert isinstance(tpb.chain_text(bank), str)


def test_chain_text_names_the_runs():
    bank = [{'Chained': s in (0, 2, 3, 8, 9, 10)} for s in range(12)]
    assert tpb.chain_text(bank) == 'a-b c-e i-l'
    assert tpb.chain_text([{'Chained': False}] * 12) == '-'
