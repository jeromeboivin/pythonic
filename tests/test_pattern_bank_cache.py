"""
The bank cache of train_pattern_bank.py: one preset is one bank of 12
patterns with their lengths and chain flags, packed and unpacked without
loss, steps past a pattern's length masked, the condition as the app's
runtime builds it, and the round-trip check against the parsed presets.
The presets are written here, in the dataset's plain-number form.
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
