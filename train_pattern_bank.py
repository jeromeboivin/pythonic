# %% [markdown]
# # Drum Pattern Bank CVAE — Training Script
#
# Trains a model that generates a whole **bank** at a time: the 12 patterns
# A-L of a preset, each with its length (1-16 steps) and the chain flags
# A→B … K→L, conditioned on the kit fingerprint and the timing (the same
# 48-dim condition as the single-pattern model, train_patterns.py).
#
# One preset of the pattern dataset is one sample: the patterns of a preset
# share a style, and the chains lay the bank out in blocks (pairs, A-D / E-H
# / I-L), which a model of single patterns cannot learn.
#
# **Usage:**
#   PATTERNARIUM_DIR=<patterns folder> python train_pattern_bank.py
#   (builds the cache once; on Colab only the cache is uploaded)
#
#   Environment overrides for short runs: BANK_EPOCHS, BANK_SUBSET (train on
#   the first N banks), BANK_BETA, BANK_TRAIN=0 (cache only), DRIVE_DIR.
#
# **Outputs (in DRIVE_DIR):**
#   - pattern_bank_cache.pt  — preprocessed tensors (built locally, uploaded)
#   - pattern_bank_best.pt   — best checkpoint (validation ELBO)
#   - pattern_bank_final.pt  — last checkpoint

# %% [markdown]
# ## Cell 1 — Configuration

# %%
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
# PATTERNARIUM_DIR: the dataset's "patterns" folder (contains gen_0*/)
PATTERNARIUM_DIR = os.path.expanduser(os.environ.get("PATTERNARIUM_DIR", "./patternarium/patterns"))
DRIVE_DIR       = os.path.expanduser(os.environ.get("DRIVE_DIR", "./drum_patterns"))
BANK_CACHE_PATH = os.path.join(DRIVE_DIR, "pattern_bank_cache.pt")
BEST_CKPT       = os.path.join(DRIVE_DIR, "pattern_bank_best.pt")
FINAL_CKPT      = os.path.join(DRIVE_DIR, "pattern_bank_final.pt")

# ── Split ─────────────────────────────────────────────────────────────────────
# A whole gen folder is held out: validation runs on presets of a batch the
# model never saw
VALID_GENS = ("gen_029",)

# ── Round-trip check ──────────────────────────────────────────────────────────
ROUND_TRIP_SAMPLES = 50

# ── Model ─────────────────────────────────────────────────────────────────────
D_MODEL      = 256
N_HEADS      = 4
ENC_LAYERS   = 4
DEC_LAYERS   = 4
BANK_LATENT  = 64           # the bank's shared style
SLOT_LATENT  = 16           # each pattern's variation
DROPOUT      = 0.1

# ── Training ──────────────────────────────────────────────────────────────────
TRAIN            = os.environ.get("BANK_TRAIN", "1") != "0"
EPOCHS           = int(os.environ.get("BANK_EPOCHS", 400))
TRAIN_SUBSET     = int(os.environ.get("BANK_SUBSET", 0))  # 0: every training bank
BATCH_SIZE       = 256
LR               = 3e-4
WEIGHT_DECAY     = 0.01
WARMUP_EPOCHS    = 5        # learning-rate warm-up, then cosine decay
BETA             = float(os.environ.get("BANK_BETA", 0.2))  # KL weight (per-bank ELBO: 1.0 is the plain VAE)
KL_WARMUP_EPOCHS = 40       # β rises linearly from 0
KL_FREE_BITS     = 0.1      # nats per latent dim, so the slot latents are not ignored
LENGTH_WEIGHT    = 4.0      # the 12 length and 11 chain terms are small next to the
CHAIN_WEIGHT     = 8.0      # 4,608 grid terms of a bank: weigh them up
GRAD_CLIP        = 1.0
LOG_EVERY        = 5
EARLY_STOPPING_PATIENCE = 60  # epochs without a better validation ELBO (after the KL warm-up)
SEED             = 0

# %% [markdown]
# ## Cell 2 — Imports

# %%
import glob
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from tqdm.auto import tqdm
except ImportError:  # the local venv has no tqdm: no progress bars
    def tqdm(iterable, **_kwargs):
        return iterable

# %% [markdown]
# ## Cell 3 — Constants

# %%

# ── Bank grid dimensions ──────────────────────────────────────────────────────
NUM_SLOTS     = 12          # patterns A-L
SLOT_KEYS     = [chr(ord("a") + i) for i in range(NUM_SLOTS)]
NUM_CHAINS    = NUM_SLOTS - 1  # A→B … K→L (L chains to nothing)
NUM_CHANNELS  = 8
NUM_STEPS     = 16          # also the longest pattern of the dataset
NUM_FEATURES  = 3           # trigger, accent, fill
PATTERN_DIM   = NUM_CHANNELS * NUM_STEPS * NUM_FEATURES  # 384
NUM_LENGTHS   = NUM_STEPS   # length classes 1..16 (class = length - 1)

# ── Kit fingerprint (as train_patterns.py and pythonic/pattern_generator.py) ──
KIT_CONTINUOUS_PARAMS = ["OscFreq", "OscDcy", "Mix", "NFilFrq", "DistAmt"]
KIT_LOG_PARAMS = {"OscFreq", "OscDcy", "NFilFrq"}
KIT_PARAM_CLAMP = {
    "OscFreq": (20.0,  20_000.0),
    "OscDcy":  (1.0,   10_000_000.0),
    "Mix":     (0.0,   100.0),
    "NFilFrq": (20.0,  20_000.0),
    "DistAmt": (0.0,   100.0),
}
KIT_CONTINUOUS_DIM = len(KIT_CONTINUOUS_PARAMS) * NUM_CHANNELS  # 40

STEP_RATES = ["1/8", "1/8T", "1/16", "1/16T", "1/32"]
STEP_RATE_DIM = len(STEP_RATES)
GLOBAL_META_DIM = 3 + STEP_RATE_DIM                             # 8
CONDITION_DIM = KIT_CONTINUOUS_DIM + GLOBAL_META_DIM            # 48

TEMPO_MIN, TEMPO_MAX = 60.0, 300.0

# Reconstruction weights per feature: triggers matter most, then accents
FEATURE_WEIGHTS = torch.tensor([3.0, 1.5, 0.5], dtype=torch.float32)  # [trigger, accent, fill]

# %% [markdown]
# ## Cell 4 — Preprocessor

# %%

class MinMaxScaler:
    """Min-max scaling to [0, 1] per feature. Its state dict has the fields
    the app's runtime (pythonic/pattern_generator.py) loads."""

    def __init__(self):
        self.scale_ = self.min_ = self.data_min_ = self.data_max_ = None

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float64)
        self.data_min_ = X.min(axis=0)
        self.data_max_ = X.max(axis=0)
        span = self.data_max_ - self.data_min_
        span[span == 0.0] = 1.0  # a constant feature maps to 0
        self.scale_ = 1.0 / span
        self.min_ = -self.data_min_ * self.scale_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) * self.scale_ + self.min_

    def state_dict(self) -> dict:
        return {"scale_": self.scale_.tolist(), "min_": self.min_.tolist(),
                "data_min_": self.data_min_.tolist(), "data_max_": self.data_max_.tolist()}

    def load_state_dict(self, d: dict):
        self.scale_ = np.array(d["scale_"], dtype=np.float64)
        self.min_ = np.array(d["min_"], dtype=np.float64)
        self.data_min_ = np.array(d["data_min_"], dtype=np.float64)
        self.data_max_ = np.array(d["data_max_"], dtype=np.float64)


def kit_continuous(drum_patches: dict) -> np.ndarray:
    """The 8 drum patches → the 40 raw kit features (log, clamped)."""
    vals = []
    for ch_idx in range(1, NUM_CHANNELS + 1):
        patch = drum_patches.get(str(ch_idx), {})
        for p in KIT_CONTINUOUS_PARAMS:
            v = patch.get(p, 0.0)
            if isinstance(v, (tuple, list)):
                v = v[0]
            try:
                v = float(v)
            except Exception:
                v = 0.0
            lo, hi = KIT_PARAM_CLAMP[p]
            v = max(lo, min(hi, v))
            if p in KIT_LOG_PARAMS:
                v = np.log(v)
            vals.append(v)
    return np.array(vals, dtype=np.float32)


def _number(value, default: float) -> float:
    """A preset value as a float; some presets add a unit ("130.0 bpm")."""
    if value is None:
        return default
    if isinstance(value, str):
        value = value.split()[0] if value.split() else default
    return float(value)


def global_meta(preset_data: dict) -> np.ndarray:
    """Tempo, swing, fill rate and step rate → the 8 timing features."""
    tempo = _number(preset_data.get("Tempo"), 120.0)
    tempo_norm = (np.clip(tempo, TEMPO_MIN, TEMPO_MAX) - TEMPO_MIN) / (TEMPO_MAX - TEMPO_MIN)
    swing_norm = np.clip(_number(preset_data.get("Swing"), 0.0) / 100.0, 0.0, 1.0)
    fill_norm = np.clip(_number(preset_data.get("FillRate"), 4.0) / 8.0, 0.0, 1.0)
    step_rate = str(preset_data.get("StepRate", "1/16"))
    sr_oh = np.zeros(STEP_RATE_DIM, dtype=np.float32)
    sr_oh[STEP_RATES.index(step_rate if step_rate in STEP_RATES else "1/16")] = 1.0
    return np.concatenate([[tempo_norm, swing_norm, fill_norm], sr_oh]).astype(np.float32)


def condition(kit: np.ndarray, meta: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """The 48-dim condition: the scaled kit features, then the timing."""
    return np.concatenate([scaler.transform(kit.reshape(1, -1))[0], meta]).astype(np.float32)


def encode_bank(patterns: dict):
    """A parsed preset's Patterns block → (grid uint8 [12, 8, 16, 3],
    length uint8 [12] (1..16), chain uint8 [11]).

    Steps at or past a pattern's Length are zero: they never play (the
    dataset keeps hidden triggers there). A missing Length is 16, a missing
    Chained is False (pattern L has none).
    """
    grid = np.zeros((NUM_SLOTS, NUM_CHANNELS, NUM_STEPS, NUM_FEATURES), dtype=np.uint8)
    length = np.zeros(NUM_SLOTS, dtype=np.uint8)
    chain = np.zeros(NUM_CHAINS, dtype=np.uint8)
    for s, key in enumerate(SLOT_KEYS):
        pat = patterns[key]
        n = int(pat.get("Length", NUM_STEPS))
        if not 1 <= n <= NUM_STEPS:
            raise ValueError(f"pattern {key}: length {n} is outside 1..{NUM_STEPS}")
        length[s] = n
        if s < NUM_CHAINS:
            chain[s] = bool(pat.get("Chained", False))
        for ch in range(NUM_CHANNELS):
            ch_data = pat.get(str(ch + 1), {})
            if isinstance(ch_data, list):  # a silent channel
                continue
            for f, name in enumerate(("Triggers", "Accents", "Fills")):
                text = ch_data.get(name, "")[:n]
                grid[s, ch, :len(text), f] = np.frombuffer(text.encode("ascii"), np.uint8) == ord("#")
    return grid, length, chain


def decode_bank(grid: np.ndarray, length: np.ndarray, chain: np.ndarray) -> list:
    """The inverse of encode_bank: 12 pattern dicts ("Length", "Chained",
    "1"-"8" with "Triggers" / "Accents" / "Fills" of 16 characters, "-" past
    the length), as PatternManager.apply_pattern_bank takes them."""
    bank = []
    for s in range(NUM_SLOTS):
        n = int(length[s])
        pat = {"Length": n, "Chained": bool(chain[s]) if s < NUM_CHAINS else False}
        for ch in range(NUM_CHANNELS):
            lane = grid[s, ch, :, :].astype(bool) & (np.arange(NUM_STEPS) < n)[:, None]
            pat[str(ch + 1)] = {name: "".join("#" if on else "-" for on in lane[:, f])
                                for f, name in enumerate(("Triggers", "Accents", "Fills"))}
        bank.append(pat)
    return bank


# %% [markdown]
# ## Cell 5 — Dataset

# %%

def pack_grid(grid: np.ndarray) -> np.ndarray:
    """[N, 12, 8, 16, 3] 0/1 → [N, 12, 48] bits (8× smaller cache)."""
    return np.packbits(grid.reshape(grid.shape[0], NUM_SLOTS, PATTERN_DIM), axis=-1)


def unpack_grid(packed: np.ndarray) -> np.ndarray:
    """[N, 12, 48] bits → [N, 12, 8, 16, 3] uint8."""
    flat = np.unpackbits(packed, axis=-1, count=PATTERN_DIM)
    return flat.reshape(packed.shape[0], NUM_SLOTS, NUM_CHANNELS, NUM_STEPS, NUM_FEATURES)


class BankDataset(Dataset):
    """One bank per item: grid float [12, 384], length class long [12]
    (length - 1), chain float [11], cond float [48]."""

    def __init__(self, cache: dict, index=None):
        grid = torch.from_numpy(unpack_grid(cache["grid"].numpy()))
        index = torch.arange(grid.shape[0]) if index is None else torch.as_tensor(index)
        self.grid = grid[index].reshape(len(index), NUM_SLOTS, PATTERN_DIM)
        self.length = cache["length"][index].long() - 1
        self.chain = cache["chain"][index].float()
        self.cond = cache["cond"][index]

    def __len__(self):
        return self.grid.shape[0]

    def __getitem__(self, i):
        return self.grid[i].float(), self.length[i], self.chain[i], self.cond[i]

    def to(self, device):
        """Move every tensor to a device (the whole set fits on a GPU)."""
        self.grid, self.length = self.grid.to(device), self.length.to(device)
        self.chain, self.cond = self.chain.to(device), self.cond.to(device)
        return self

    def batches(self, size: int, shuffle: bool = False, generator=None):
        """(grid, length, chain, cond) batches straight from the tensors."""
        order = torch.randperm(len(self), generator=generator) if shuffle else torch.arange(len(self))
        order = order.to(self.grid.device)
        for start in range(0, len(self), size):
            idx = order[start:start + size]
            yield self.grid[idx].float(), self.length[idx], self.chain[idx], self.cond[idx]


def split_indexes(cache: dict, valid_gens=VALID_GENS):
    """Train and validation indexes: the held-out gen folders validate."""
    held = torch.tensor([cache["gens"].index(g) for g in valid_gens if g in cache["gens"]])
    valid = torch.isin(cache["gen"].long(), held)
    return torch.nonzero(~valid).squeeze(1), torch.nonzero(valid).squeeze(1)


# %% [markdown]
# ## Cell 6 — Cache build / load

# %%

def preset_files(patternarium_dir: str) -> list:
    files = sorted(glob.glob(os.path.join(patternarium_dir, "gen_0*", "*.mtpreset")))
    if not files:
        raise RuntimeError(f"No .mtpreset files found under {patternarium_dir}/gen_0*/")
    return files


def _parse_preset(path: str):
    """Parse one preset (worker process): its raw kit features, timing,
    bank, or the reason it was skipped."""
    from pythonic.preset_manager import PythonicPresetParser
    try:
        data = PythonicPresetParser().parse_file(path)
        patterns = data.get("Patterns") or {}
        missing = [k for k in SLOT_KEYS if k not in patterns]
        if missing or "DrumPatches" not in data:
            return path, None, f"missing {'patterns ' + ''.join(missing) if missing else 'drum patches'}"
        grid, length, chain = encode_bank(patterns)
        return path, (kit_continuous(data["DrumPatches"]), global_meta(data), grid, length, chain), None
    except Exception as exc:
        return path, None, f"{type(exc).__name__}: {exc}"


def build_bank_cache(patternarium_dir: str, cache_path: str, workers: int = None) -> dict:
    """Parse every preset, fit the kit scaler, save the bank cache.

    The cache: grid uint8 [N, 12, 48] (packed bits, see unpack_grid),
    length uint8 [N, 12], chain uint8 [N, 11], cond float32 [N, 48],
    gen int16 [N] (an index into gens), gens (the folder names), names (the
    preset file names), scaler (the kit scaler's state), skipped.
    """
    import sys
    from multiprocessing import get_context
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    files = preset_files(patternarium_dir)
    print(f"Found {len(files)} presets.")
    if workers == 1:
        results = [_parse_preset(f) for f in tqdm(files, desc="Parsing")]
    else:
        # spawn, not fork: torch has started threads by now. The workers
        # import this script, whose build only runs under __main__
        with get_context("spawn").Pool(workers) as pool:
            results = list(tqdm(pool.imap(_parse_preset, files, chunksize=64), total=len(files),
                                desc="Parsing"))

    kept = [(p, r) for p, r, _ in results if r is not None]
    skipped = [(os.path.relpath(p, patternarium_dir), why) for p, r, why in results if r is None]
    print(f"Parsed {len(kept)} banks, skipped {len(skipped)}.")
    for path, why in skipped[:10]:
        print(f"  ✗ {path}: {why}")

    kits = np.stack([r[0] for _, r in kept])
    scaler = MinMaxScaler().fit(kits)
    cond = np.stack([condition(r[0], r[1], scaler) for _, r in kept])
    grid = np.stack([r[2] for _, r in kept])
    folders = [os.path.basename(os.path.dirname(p)) for p, _ in kept]
    gens = sorted(set(folders))
    cache = {
        "grid":    torch.from_numpy(pack_grid(grid)),
        "length":  torch.from_numpy(np.stack([r[3] for _, r in kept])),
        "chain":   torch.from_numpy(np.stack([r[4] for _, r in kept])),
        "cond":    torch.from_numpy(cond),
        "gen":     torch.tensor([gens.index(f) for f in folders], dtype=torch.int16),
        "gens":    gens,
        "names":   [os.path.relpath(p, patternarium_dir) for p, _ in kept],
        "scaler":  scaler.state_dict(),
        "skipped": skipped,
        "format":  {"kind": "pattern_bank", "version": 1, "slots": NUM_SLOTS,
                    "pattern_dim": PATTERN_DIM, "cond_dim": CONDITION_DIM},
    }
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    torch.save(cache, cache_path)
    print(f"Cache saved → {cache_path} ({os.path.getsize(cache_path) / 1e6:.1f} MB)")
    return cache


def load_bank_cache(cache_path: str) -> dict:
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    fmt = cache.get("format", {})
    if fmt.get("kind") != "pattern_bank" or fmt.get("version") != 1:
        raise ValueError(f"{cache_path} is not a version 1 pattern bank cache: {fmt}")
    print(f"Loaded {cache['cond'].shape[0]} banks from {cache_path}")
    return cache


def cache_summary(cache: dict):
    """The dataset statistics the model must reproduce (see the ticket)."""
    length = cache["length"].numpy()
    chain = cache["chain"].numpy().astype(bool)
    grid = unpack_grid(cache["grid"].numpy())
    n = length.shape[0]
    print(f"banks {n}; per gen folder {np.bincount(cache['gen'].numpy()).tolist()}")
    lengths, counts = np.unique(length, return_counts=True)
    print("lengths", {int(l): f"{c / length.size:.2%}" for l, c in zip(lengths, counts)})
    print(f"chain flags set {chain.mean():.1%}; banks with a chain {chain.any(axis=1).mean():.1%}")
    print("chain rate by letter", {f"{SLOT_KEYS[s]}→{SLOT_KEYS[s + 1]}": f"{chain[:, s].mean():.1%}"
                                   for s in range(NUM_CHAINS)})
    runs = {}
    for row in chain:
        s = 0
        while s < NUM_CHAINS:
            if row[s]:
                e = s
                while e < NUM_CHAINS and row[e]:
                    e += 1
                runs[e - s + 1] = runs.get(e - s + 1, 0) + 1
                s = e
            s += 1
    print("chain runs (patterns per chain)", dict(sorted(runs.items())))
    print(f"trigger density {grid[..., 0].mean():.3f}")


# %% [markdown]
# ## Cell 7 — Round-trip check

# %%

def round_trip_check(cache: dict, patternarium_dir: str, samples: int = ROUND_TRIP_SAMPLES,
                     seed: int = 0) -> int:
    """Re-parse sampled presets and compare them with their cached bank:
    every playing step of every lane, the lengths, the chain flags and the
    condition. Returns the number of mismatches (0 expected)."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pythonic.preset_manager import PythonicPresetParser

    parser = PythonicPresetParser()
    scaler = MinMaxScaler()
    scaler.load_state_dict(cache["scaler"])
    rng = random.Random(seed)
    picks = rng.sample(range(len(cache["names"])), min(samples, len(cache["names"])))
    grids = unpack_grid(cache["grid"][picks].numpy())
    bad = 0
    for k, i in enumerate(picks):
        name = cache["names"][i]
        data = parser.parse_file(os.path.join(patternarium_dir, name))
        bank = decode_bank(grids[k], cache["length"][i].numpy(), cache["chain"][i].numpy())
        problems = []
        for s, key in enumerate(SLOT_KEYS):
            src, out = data["Patterns"][key], bank[s]
            n = int(src.get("Length", NUM_STEPS))
            if out["Length"] != n:
                problems.append(f"{key} length {out['Length']} != {n}")
            if out["Chained"] != bool(src.get("Chained", False)):
                problems.append(f"{key} chained {out['Chained']} != {src.get('Chained')}")
            for ch in range(1, NUM_CHANNELS + 1):
                src_ch = src.get(str(ch), {})
                for feat in ("Triggers", "Accents", "Fills"):
                    want = (src_ch.get(feat, "") if isinstance(src_ch, dict) else "")[:n].ljust(n, "-")
                    got = out[str(ch)][feat]
                    if got[:n] != want or set(got[n:]) - {"-"}:
                        problems.append(f"{key} ch{ch} {feat} {got} != {want}")
        cond = condition(kit_continuous(data["DrumPatches"]), global_meta(data), scaler)
        if not np.allclose(cond, cache["cond"][i].numpy(), atol=1e-5):
            problems.append("condition differs")
        if problems:
            bad += 1
            print(f"  ✗ {name}: {problems[:3]}")
    print(f"Round trip: {len(picks) - bad}/{len(picks)} banks match.")
    return bad




# %% [markdown]
# ## Cell 8 — Bank CVAE model
#
# Hierarchical and non-autoregressive. The encoder reads the condition and
# the 12 slots (grid, length, chain-out, position) with a Transformer and
# gives a **bank latent** (the shared style, from the condition token) and a
# **slot latent** per pattern (its variation). The decoder attends across
# the 12 slot queries, so the slots coordinate their chains and style, and
# outputs per slot: grid logits, length logits, and a chain-to-next logit
# (A-K).

# %%

class BankCVAE(nn.Module):
    def __init__(self, d_model: int = D_MODEL, n_heads: int = N_HEADS, enc_layers: int = ENC_LAYERS,
                 dec_layers: int = DEC_LAYERS, bank_latent: int = BANK_LATENT,
                 slot_latent: int = SLOT_LATENT, dropout: float = DROPOUT):
        super().__init__()
        self.bank_latent, self.slot_latent = bank_latent, slot_latent
        self.config = {
            "kind": "pattern_bank", "version": 1, "pattern_dim": PATTERN_DIM, "cond_dim": CONDITION_DIM,
            "num_slots": NUM_SLOTS, "num_lengths": NUM_LENGTHS, "d_model": d_model, "n_heads": n_heads,
            "enc_layers": enc_layers, "dec_layers": dec_layers, "bank_latent": bank_latent,
            "slot_latent": slot_latent, "dropout": dropout,
        }

        def mlp(n_in):
            return nn.Sequential(nn.Linear(n_in, d_model), nn.GELU(), nn.Linear(d_model, d_model))

        def stack(n_layers):
            layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, activation="gelu",
                                               batch_first=True, norm_first=True)
            return nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)

        # Encoder
        self.enc_grid = mlp(PATTERN_DIM)
        self.enc_length = nn.Embedding(NUM_LENGTHS, d_model)
        self.enc_chain = nn.Embedding(3, d_model)  # not chained, chained, none (slot L)
        self.enc_slot = nn.Parameter(torch.randn(NUM_SLOTS, d_model) * 0.02)
        self.enc_cond = mlp(CONDITION_DIM)
        self.encoder = stack(enc_layers)
        self.enc_norm = nn.LayerNorm(d_model)
        self.bank_head = nn.Linear(d_model, 2 * bank_latent)
        self.slot_head = nn.Linear(d_model, 2 * slot_latent)
        # Decoder
        self.dec_cond = mlp(CONDITION_DIM)
        self.dec_bank = nn.Linear(bank_latent, d_model)
        self.dec_z = nn.Linear(slot_latent, d_model)
        self.dec_slot = nn.Parameter(torch.randn(NUM_SLOTS, d_model) * 0.02)
        self.decoder = stack(dec_layers)
        self.dec_norm = nn.LayerNorm(d_model)
        self.grid_head = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                       nn.Linear(2 * d_model, PATTERN_DIM))
        self.length_head = nn.Linear(d_model, NUM_LENGTHS)
        self.chain_head = nn.Linear(d_model, 1)

    def encode(self, grid, length, chain, cond):
        """grid [B, 12, 384] float, length [B, 12] classes, chain [B, 11],
        cond [B, 48] → (bank mu, bank logvar [B, Zb], slot mu, slot logvar
        [B, 12, Zs])."""
        none = torch.full((grid.shape[0], 1), 2, dtype=torch.long, device=grid.device)
        chain_in = torch.cat([chain.long(), none], dim=1)
        slots = self.enc_grid(grid) + self.enc_length(length) + self.enc_chain(chain_in) + self.enc_slot
        tokens = torch.cat([self.enc_cond(cond).unsqueeze(1), slots], dim=1)
        h = self.enc_norm(self.encoder(tokens))
        bank_mu, bank_logvar = self.bank_head(h[:, 0]).chunk(2, dim=-1)
        slot_mu, slot_logvar = self.slot_head(h[:, 1:]).chunk(2, dim=-1)
        return bank_mu, bank_logvar.clamp(-10, 10), slot_mu, slot_logvar.clamp(-10, 10)

    def decode(self, z_bank, z_slot, cond):
        """z_bank [B, Zb], z_slot [B, 12, Zs], cond [B, 48] → grid logits
        [B, 12, 384], length logits [B, 12, 16], chain logits [B, 11]."""
        bank = self.dec_bank(z_bank)
        slots = self.dec_slot + self.dec_z(z_slot) + bank.unsqueeze(1)
        tokens = torch.cat([self.dec_cond(cond).unsqueeze(1), bank.unsqueeze(1), slots], dim=1)
        h = self.dec_norm(self.decoder(tokens))[:, 2:]
        return self.grid_head(h), self.length_head(h), self.chain_head(h[:, :NUM_CHAINS]).squeeze(-1)

    def forward(self, grid, length, chain, cond):
        bank_mu, bank_logvar, slot_mu, slot_logvar = self.encode(grid, length, chain, cond)
        z_bank = bank_mu + torch.randn_like(bank_mu) * torch.exp(0.5 * bank_logvar)
        z_slot = slot_mu + torch.randn_like(slot_mu) * torch.exp(0.5 * slot_logvar)
        return (*self.decode(z_bank, z_slot, cond), bank_mu, bank_logvar, slot_mu, slot_logvar)


def save_checkpoint(path: str, model: BankCVAE, scaler_state: dict, epoch: int, metrics: dict,
                    optimizer=None):
    """A checkpoint the app's runtime reads: model_state, model_config
    (kind "pattern_bank"), scaler; plus the epoch, metrics and optimizer."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "model_state": model.state_dict(), "model_config": dict(model.config), "scaler": scaler_state,
        "epoch": epoch, "metrics": metrics,
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
    }, path)


def load_bank_model(path: str, device="cpu"):
    """(model in eval mode, checkpoint dict) from a bank checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["model_config"]
    if cfg.get("kind") != "pattern_bank":
        raise ValueError(f"{path} is not a pattern bank checkpoint")
    model = BankCVAE(**{k: cfg[k] for k in ("d_model", "n_heads", "enc_layers", "dec_layers",
                                             "bank_latent", "slot_latent")}, dropout=0.0)
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval(), ckpt


# %% [markdown]
# ## Cell 9 — Loss and metrics
#
# A per-bank ELBO: the grid, length and chain negative log-likelihoods are
# summed over the bank (steps past a pattern's length do not count), the KL
# over every latent dim. Length and chain terms are weighed up, β warms up,
# and free bits keep the slot latents in use.

# %%

def step_mask(length):
    """Length classes [B, 12] → [B, 12, 384] float: 1 where the step plays."""
    steps = torch.arange(NUM_STEPS, device=length.device)
    plays = steps[None, None, :] <= length[..., None]  # class = length - 1
    plays = plays[:, :, None, :, None].expand(-1, -1, NUM_CHANNELS, -1, NUM_FEATURES)
    return plays.reshape(*length.shape, PATTERN_DIM).float()


def kl_per_dim(mu, logvar):
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())


def bank_loss(out, grid, length, chain, beta: float, free_bits: float = KL_FREE_BITS):
    """(loss to minimise, parts dict). parts["objective"] (recon + BETA × KL,
    no free bits) picks the best checkpoint independently of the warm-up."""
    grid_logits, length_logits, chain_logits, bank_mu, bank_logvar, slot_mu, slot_logvar = out
    batch = grid.shape[0]
    weights = FEATURE_WEIGHTS.to(grid.device).repeat(NUM_CHANNELS * NUM_STEPS)
    grid_nll = (F.binary_cross_entropy_with_logits(grid_logits.float(), grid, reduction="none")
                * weights * step_mask(length)).sum() / batch
    length_nll = F.cross_entropy(length_logits.float().reshape(-1, NUM_LENGTHS), length.reshape(-1),
                                 reduction="sum") / batch
    chain_nll = F.binary_cross_entropy_with_logits(chain_logits.float(), chain, reduction="sum") / batch
    kl_bank = kl_per_dim(bank_mu.float(), bank_logvar.float()).mean(0)  # [Zb]
    kl_slot = kl_per_dim(slot_mu.float(), slot_logvar.float()).mean(0)  # [12, Zs]
    kl = kl_bank.sum() + kl_slot.sum()
    kl_floor = kl_bank.clamp(min=free_bits).sum() + kl_slot.clamp(min=free_bits).sum()
    recon = grid_nll + LENGTH_WEIGHT * length_nll + CHAIN_WEIGHT * chain_nll
    loss = recon + beta * kl_floor
    parts = {"loss": loss.item(), "grid": grid_nll.item(), "length": length_nll.item(),
             "chain": chain_nll.item(), "kl_bank": kl_bank.sum().item(), "kl_slot": kl_slot.sum().item(),
             "objective": (recon + BETA * kl).item()}
    return loss, parts


class BankMetrics:
    """Accumulated over batches: grid F1 per feature (playing steps only),
    length accuracy, chain accuracy and F1, and the latent means for the
    active units."""

    def __init__(self):
        self.grid = torch.zeros(NUM_FEATURES, 3)  # tp, fp, fn per feature
        self.length_ok = self.chain_ok = self.slots = self.links = 0
        self.chain = torch.zeros(3)  # tp, fp, fn
        self.bank_mu, self.slot_mu = [], []

    @torch.no_grad()
    def add(self, out, grid, length, chain):
        grid_logits, length_logits, chain_logits, bank_mu, _, slot_mu, _ = out
        plays = step_mask(length).bool()
        pred, target = grid_logits > 0, grid > 0.5
        feature = torch.arange(PATTERN_DIM, device=grid.device) % NUM_FEATURES
        for f in range(NUM_FEATURES):
            sel = plays & (feature == f)
            self.grid[f] += torch.stack([(pred & target & sel).sum(), (pred & ~target & sel).sum(),
                                         (~pred & target & sel).sum()]).float().cpu()
        self.length_ok += (length_logits.argmax(-1) == length).sum().item()
        self.slots += length.numel()
        chained, linked = chain_logits > 0, chain > 0.5
        self.chain_ok += (chained == linked).sum().item()
        self.links += chain.numel()
        self.chain += torch.stack([(chained & linked).sum(), (chained & ~linked).sum(),
                                   (~chained & linked).sum()]).float().cpu()
        self.bank_mu.append(bank_mu.float().cpu())
        self.slot_mu.append(slot_mu.float().cpu())

    @staticmethod
    def _f1(tp, fp, fn):
        return float(2 * tp / max(2 * tp + fp + fn, 1))

    def result(self) -> dict:
        bank_mu, slot_mu = torch.cat(self.bank_mu), torch.cat(self.slot_mu)
        slot_mu = slot_mu.reshape(-1, slot_mu.shape[-1])

        def active(mu):  # a latent dim whose mean varies across banks
            return int((mu.var(0) > 0.01).sum()) if mu.shape[0] > 1 else 0
        return {
            "f1_trigger": self._f1(*self.grid[0]), "f1_accent": self._f1(*self.grid[1]),
            "f1_fill": self._f1(*self.grid[2]),
            "length_acc": self.length_ok / max(self.slots, 1), "chain_acc": self.chain_ok / max(self.links, 1),
            "chain_f1": self._f1(*self.chain),
            "au_bank": active(bank_mu), "au_slot": active(slot_mu),
        }


# %% [markdown]
# ## Cell 10 — Trainer

# %%

@torch.no_grad()
def evaluate(model: BankCVAE, data: BankDataset, beta: float, batch_size: int = BATCH_SIZE) -> dict:
    """Validation: the mean loss parts and the metrics."""
    model.eval()
    totals, n, metrics = {}, 0, BankMetrics()
    for grid, length, chain, cond in data.batches(batch_size):
        out = model(grid, length, chain, cond)
        _, parts = bank_loss(out, grid, length, chain, beta)
        for k, v in parts.items():
            totals[k] = totals.get(k, 0.0) + v * grid.shape[0]
        n += grid.shape[0]
        metrics.add(out, grid, length, chain)
    return {**{k: v / n for k, v in totals.items()}, **metrics.result()}


def train_bank_model(cache: dict, train_idx, valid_idx, epochs: int = EPOCHS, device=None,
                     best_path: str = BEST_CKPT, final_path: str = FINAL_CKPT,
                     model_kwargs: dict = None, batch_size: int = BATCH_SIZE,
                     log_every: int = LOG_EVERY) -> tuple:
    """Train on the train banks, keep the checkpoint with the best
    validation objective (once β is warm). Returns (model, history)."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    train = BankDataset(cache, train_idx).to(device)
    valid = BankDataset(cache, valid_idx).to(device)
    model = BankCVAE(**(model_kwargs or {})).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters on {device}; "
          f"{len(train)} train, {len(valid)} validation banks")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    per_epoch = math.ceil(len(train) / batch_size)
    total, warm = epochs * per_epoch, max(1, min(WARMUP_EPOCHS, epochs) * per_epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda i: min(1.0, (i + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, i / total))))
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    kl_warmup = min(KL_WARMUP_EPOCHS, max(1, epochs // 4))  # short runs still reach β

    best, best_epoch, history = float("inf"), 0, []
    for epoch in range(1, epochs + 1):
        started = time.time()
        beta = BETA * min(1.0, epoch / kl_warmup)
        model.train()
        sums, n = {}, 0
        for grid, length, chain, cond in train.batches(batch_size, shuffle=True, generator=generator):
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(grid, length, chain, cond)
            loss, parts = bank_loss(out, grid, length, chain, beta)
            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            scheduler.step()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v * grid.shape[0]
            n += grid.shape[0]
        val = evaluate(model, valid, beta, batch_size)
        record = {"epoch": epoch, "beta": beta, "seconds": time.time() - started,
                  **{f"train_{k}": v / n for k, v in sums.items()}, **{f"val_{k}": v for k, v in val.items()}}
        history.append(record)
        improved = epoch >= kl_warmup and val["objective"] < best
        if improved:
            best, best_epoch = val["objective"], epoch
            save_checkpoint(best_path, model, cache["scaler"], epoch, record)
        if epoch == 1 or epoch % log_every == 0 or epoch == epochs:
            print(f"ep {epoch:4d} β {beta:.2f} | train {record['train_loss']:8.1f} "
                  f"(grid {record['train_grid']:7.1f} len {record['train_length']:5.2f} "
                  f"chain {record['train_chain']:5.2f} kl {record['train_kl_bank']:5.1f}+"
                  f"{record['train_kl_slot']:5.1f}) | val obj {val['objective']:8.1f} "
                  f"F1 t/a/f {val['f1_trigger']:.3f}/{val['f1_accent']:.3f}/{val['f1_fill']:.3f} "
                  f"len {val['length_acc']:.3f} chain {val['chain_acc']:.3f} (F1 {val['chain_f1']:.3f}) "
                  f"AU {val['au_bank']}/{val['au_slot']} | {record['seconds']:.1f}s"
                  + (" *" if improved else ""))
        if epoch >= kl_warmup and epoch - best_epoch > EARLY_STOPPING_PATIENCE:
            print(f"Early stop at epoch {epoch}: best objective {best:.1f} at epoch {best_epoch}")
            break
    save_checkpoint(final_path, model, cache["scaler"], history[-1]["epoch"], history[-1], optimizer)
    print(f"Best objective {best:.1f} at epoch {best_epoch} → {best_path}; last → {final_path}")
    return model, history


# %% [markdown]
# ## Cell 11 — Sampling
#
# Generation draws the bank latent with the bank temperature and the slot
# latents with the slot temperature (how far the 12 patterns stray from
# one another), then decodes: a grid step plays above the threshold; the
# length is the most likely one ("argmax") or drawn from its distribution
# ("sample"); a chain is set above its threshold ("threshold") or drawn
# ("bernoulli").

# %%

@torch.no_grad()
def sample_bank_arrays(model: BankCVAE, cond, bank_temperature: float = 1.0, slot_temperature: float = 1.0,
                       threshold: float = 0.5, chain_threshold: float = 0.5, length_mode: str = "argmax",
                       chain_mode: str = "threshold", generator=None, z_bank=None) -> tuple:
    """cond [N, 48] → (grid bool [N, 12, 8, 16, 3] (zero past each length),
    length [N, 12] (1..16), chain bool [N, 11]) as numpy arrays. z_bank
    [N, Zb] fixes the bank latents (the kit-sensitivity check)."""
    if length_mode not in ("argmax", "sample") or chain_mode not in ("threshold", "bernoulli"):
        raise ValueError(f"length_mode argmax|sample, chain_mode threshold|bernoulli: {length_mode}, {chain_mode}")
    model.eval()
    device = next(model.parameters()).device
    cond = torch.as_tensor(cond, dtype=torch.float32).to(device)
    n = cond.shape[0]
    if z_bank is None:
        z_bank = torch.randn(n, model.bank_latent, generator=generator) * bank_temperature
    z_slot = torch.randn(n, NUM_SLOTS, model.slot_latent, generator=generator) * slot_temperature
    grid_logits, length_logits, chain_logits = model.decode(z_bank.to(device), z_slot.to(device), cond)
    if length_mode == "argmax":
        length = length_logits.argmax(-1) + 1
    else:
        probs = torch.softmax(length_logits.float(), -1).cpu().reshape(-1, NUM_LENGTHS)
        length = (torch.multinomial(probs, 1, generator=generator).reshape(n, NUM_SLOTS) + 1).to(device)
    chain_p = torch.sigmoid(chain_logits.float())
    if chain_mode == "threshold":
        chain = chain_p >= chain_threshold
    else:
        chain = (torch.rand(chain_p.shape, generator=generator).to(device) < chain_p)
    grid = (torch.sigmoid(grid_logits.float()) >= threshold).reshape(n, NUM_SLOTS, NUM_CHANNELS, NUM_STEPS,
                                                                     NUM_FEATURES)
    plays = torch.arange(NUM_STEPS, device=device)[None, None, :] < length[..., None]
    grid = grid & plays[:, :, None, :, None]
    return grid.cpu().numpy(), length.cpu().numpy(), chain.cpu().numpy()


def sample_banks(model: BankCVAE, cond, **options) -> list:
    """cond [N, 48] → N banks, each 12 pattern dicts (decode_bank); the
    options are sample_bank_arrays'."""
    grid, length, chain = sample_bank_arrays(model, cond, **options)
    return [decode_bank(grid[i], length[i], chain[i]) for i in range(len(grid))]


def chain_text(bank: list) -> str:
    """The chains of a bank as runs, e.g. "a-b c-d i-l" ("-" for none)."""
    runs, s = [], 0
    while s < NUM_SLOTS:
        e = s
        while e < NUM_CHAINS and bank[e]["Chained"]:
            e += 1
        if e > s:
            runs.append(f"{SLOT_KEYS[s]}-{SLOT_KEYS[e]}")
        s = e + 1
    return " ".join(runs) or "-"


# %% [markdown]
# ## Cell 12 — Build or load the dataset, then train

# %%
if __name__ == "__main__":
    if os.path.exists(BANK_CACHE_PATH):
        bank_cache = load_bank_cache(BANK_CACHE_PATH)
    else:
        bank_cache = build_bank_cache(PATTERNARIUM_DIR, BANK_CACHE_PATH)
    cache_summary(bank_cache)
    train_idx, valid_idx = split_indexes(bank_cache)
    print(f"Split: {len(train_idx)} train, {len(valid_idx)} validation ({', '.join(VALID_GENS)})")
    if os.path.isdir(PATTERNARIUM_DIR):  # not on Colab, where only the cache is uploaded
        assert round_trip_check(bank_cache, PATTERNARIUM_DIR) == 0, "the cache does not match the presets"

    if TRAIN:
        if TRAIN_SUBSET:
            train_idx = train_idx[:TRAIN_SUBSET]
        model, history = train_bank_model(bank_cache, train_idx, valid_idx)
        best_model, _ = load_bank_model(BEST_CKPT) if os.path.exists(BEST_CKPT) else (model.cpu(), None)
        for bank in sample_banks(best_model.cpu(), bank_cache["cond"][valid_idx[:8]],
                                 generator=torch.Generator().manual_seed(SEED)):
            print(f"lengths {[p['Length'] for p in bank]}  chains {chain_text(bank)}")
