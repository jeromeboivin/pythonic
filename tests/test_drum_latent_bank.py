"""
The drum patch model's data-guided latent bank (pythonic/drum_generator.py),
over a tiny checkpoint and dataset cache written here: a model trained with
a KL term loads without building the bank (auto mode samples its prior) and
builds it on the first data-guided request; once built, the bank is saved
next to the checkpoint and read back by the next load, which then gives the
same candidates; a changed checkpoint or dataset cache rebuilds it; a folder
that cannot be written only costs the rebuild.
"""

import os

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from pythonic import drum_generator as dg  # noqa: E402

TYPES = ['bd', 'sd', 'ch', 'clap']


def write_model(folder, beta, rows=64, seed=0):
    """A tiny checkpoint and the dataset cache next to it."""
    torch.manual_seed(seed)
    pre = dg.PatchPreprocessor()
    model = dg._build_model(pre.param_dim, pre.type_dim, pre.cont_dim, pre.cat_dim, latent_dim=4,
                            hidden_dim=16, dropout=0.0)
    path = folder / 'model.pt'
    torch.save({
        'model_state': model.state_dict(),
        'model_config': {'param_dim': pre.param_dim, 'type_dim': pre.type_dim, 'cont_dim': pre.cont_dim,
                         'cat_dim': pre.cat_dim, 'latent_dim': 4, 'hidden_dim': 16, 'dropout': 0.0},
        'continuous_params': pre.continuous_params,
        'training_config': {'beta': beta},
        'scaler': {'scale_': [1.0] * pre.cont_dim, 'min_': [0.0] * pre.cont_dim,
                   'data_min_': [0.0] * pre.cont_dim, 'data_max_': [1.0] * pre.cont_dim},
    }, path)
    write_cache(folder, rows)
    return path


def write_cache(folder, rows):
    pre = dg.PatchPreprocessor()
    rng = np.random.default_rng(rows)
    types = [TYPES[i % len(TYPES)] for i in range(rows)]
    torch.save({'train_x': torch.tensor(rng.random((rows, pre.param_dim)), dtype=torch.float32),
                'train_c': torch.tensor(np.stack([pre.encode_type(t) for t in types])),
                'train_types': types}, folder / dg.DEFAULT_DATASET_CACHE)


@pytest.fixture
def fresh(monkeypatch):
    """A new process: the in-process bank cache is empty. Calling the
    fixture's value empties it again."""
    def reset():
        monkeypatch.setattr(dg.PatchGenerator, '_EMPIRICAL_BANK_CACHE', {})
    reset()
    return reset


def loaded(path):
    generator = dg.PatchGenerator()
    generator.load_model(str(path))
    return generator


def encoder_runs(generator, monkeypatch):
    """Make the encoder count its calls."""
    runs = []
    real = generator._model.encoder.forward

    def forward(*args):
        runs.append(1)
        return real(*args)
    monkeypatch.setattr(generator._model.encoder, 'forward', forward)
    return runs


def test_a_kl_model_loads_without_the_bank_and_builds_it_on_request(tmp_path, monkeypatch, fresh):
    path = write_model(tmp_path, beta=0.5)
    generator = loaded(path)
    runs = encoder_runs(generator, monkeypatch)
    assert generator._latent_bank == {}
    patches = generator.generate('bd', n=2, seed=3)  # auto: the prior, no bank
    assert len(patches) == 2 and runs == [] and generator.sampling_summary == 'prior'
    assert not (tmp_path / 'model.latents.pt').exists()

    assert len(generator.generate('sd', n=2, seed=3, sampling_mode='empirical')) == 2
    assert runs and set(generator._latent_bank) == set(TYPES)
    assert (tmp_path / 'model.latents.pt').exists()


def test_the_saved_bank_is_read_back_and_gives_the_same_candidates(tmp_path, monkeypatch, fresh):
    path = write_model(tmp_path, beta=0.0)
    first = loaded(path)  # a model without KL builds the bank at load
    assert first.sampling_summary == 'data-guided' and (tmp_path / 'model.latents.pt').exists()
    expected = first.generate('clap', n=3, seed=5)

    fresh()
    again = dg.PatchGenerator()
    read = []
    real_read = again._load_saved_latent_bank
    monkeypatch.setattr(again, '_load_saved_latent_bank', lambda key: read.append(real_read(key)) or read[-1])
    again.load_model(str(path))
    assert read == [True] and again.has_empirical_bank  # read back, not encoded again
    for drum_type in TYPES:
        assert torch.equal(again._latent_bank[drum_type], first._latent_bank[drum_type])
    assert again.generate('clap', n=3, seed=5) == expected


def test_a_changed_cache_or_checkpoint_rebuilds_the_bank(tmp_path, monkeypatch, fresh):
    path = write_model(tmp_path, beta=0.0, rows=64)
    rows64 = loaded(path)._latent_bank['bd'].shape[0]

    fresh()
    write_cache(tmp_path, rows=96)  # a new dataset cache
    assert loaded(path)._latent_bank['bd'].shape[0] == 96 // len(TYPES) != rows64

    fresh()
    saved = torch.load(tmp_path / 'model.latents.pt', weights_only=False)
    write_model(tmp_path, beta=0.0, rows=96, seed=1)  # a retrained checkpoint
    rebuilt = loaded(path)
    assert torch.load(tmp_path / 'model.latents.pt', weights_only=False)['key'] != saved['key']
    assert not torch.equal(rebuilt._latent_bank['bd'], saved['bank']['bd'])


def test_a_folder_that_cannot_be_written_only_costs_the_rebuild(tmp_path, monkeypatch, fresh):
    path = write_model(tmp_path, beta=0.0)

    def refuse(*args, **kwargs):
        raise PermissionError('read-only')
    monkeypatch.setattr(os, 'replace', refuse)
    generator = loaded(path)
    assert generator.has_empirical_bank and generator.sampling_summary == 'data-guided'
    assert not (tmp_path / 'model.latents.pt').exists()
    assert not [name for name in os.listdir(tmp_path) if name.endswith('.tmp')]
