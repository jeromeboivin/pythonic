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
#
# **Outputs (in DRIVE_DIR):**
#   - pattern_bank_cache.pt  — preprocessed tensors (built locally, uploaded)

# %% [markdown]
# ## Cell 1 — Configuration

# %%
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
# PATTERNARIUM_DIR: the dataset's "patterns" folder (contains gen_0*/)
PATTERNARIUM_DIR = os.path.expanduser(os.environ.get("PATTERNARIUM_DIR", "./patternarium/patterns"))
DRIVE_DIR       = "./drum_patterns"
BANK_CACHE_PATH = os.path.join(DRIVE_DIR, "pattern_bank_cache.pt")

# ── Split ─────────────────────────────────────────────────────────────────────
# A whole gen folder is held out: validation runs on presets of a batch the
# model never saw
VALID_GENS = ("gen_029",)

# ── Round-trip check ──────────────────────────────────────────────────────────
ROUND_TRIP_SAMPLES = 50

# %% [markdown]
# ## Cell 2 — Imports

# %%
import glob
import random

import numpy as np
import torch
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
# ## Cell 8 — Build or load the dataset

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
