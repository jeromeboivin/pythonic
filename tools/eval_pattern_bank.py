"""
Compare the banks a pattern model generates with the dataset's.

    venv/bin/python tools/eval_pattern_bank.py [--bank drum_patterns/pattern_bank_best.pt]
        [--legacy <single-pattern checkpoint>] [--json report.json]

Every model draws one bank for each validation condition of the bank cache
(the held-out gen folders of train_pattern_bank.py), so the statistics line
up with the validation banks of those same kits and timings:

- chains: the rate per link A→B … K→L, the banks with a chain, the chain
  runs (patterns per chain);
- lengths: the share of 16, 8, 12 and other step counts;
- density: triggers, accents and fills per playing step, and per channel;
- coherence: trigger Jaccard between two patterns of a bank, between
  neighbours, and between patterns of different banks, and the share of
  channel lanes that neighbours share;
- conditioning: how per-channel trigger density follows the bank of the
  same condition (Pearson r), and how much a bank changes with the kit (the
  same latent noise, another condition) against with the noise (the same
  condition, another noise);
- novelty: each pattern's closest training pattern (Jaccard), and the
  share of near copies (≥ 0.95).

The bank model samples through the kit prior stored in its checkpoint
(--prior normal: from N(0, I)), with triggers thresholded and accents and
fills drawn (--grid-mode). --legacy compares a single-pattern CVAE
checkpoint (train_patterns.py): 12 independent draws, every pattern 16
steps, no chains.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import train_pattern_bank as tpb  # noqa: E402

NEAR_COPY = 0.95


# ---------------------------------------------------------------- statistics

def jaccard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Jaccard of boolean arrays [..., D] (1 when both are empty)."""
    inter = (a & b).sum(-1)
    union = (a | b).sum(-1)
    return np.where(union > 0, inter / np.maximum(union, 1), 1.0)


def chain_runs(chain: np.ndarray) -> dict:
    """Chain runs by number of patterns, from chain flags [N, 11]."""
    runs = {}
    for row in chain:
        s = 0
        while s < tpb.NUM_CHAINS:
            if row[s]:
                e = s
                while e < tpb.NUM_CHAINS and row[e]:
                    e += 1
                runs[e - s + 1] = runs.get(e - s + 1, 0) + 1
                s = e
            s += 1
    return runs


def bank_stats(grid: np.ndarray, length: np.ndarray, chain: np.ndarray, rng: np.random.Generator) -> dict:
    """Statistics of banks: grid bool [N, 12, 8, 16, 3] (zero past each
    length), length [N, 12], chain bool [N, 11]."""
    n = grid.shape[0]
    plays = np.arange(tpb.NUM_STEPS)[None, None, :] < length[..., None]  # [N, 12, 16]
    steps = plays.sum() * tpb.NUM_CHANNELS
    trig = grid[..., 0].reshape(n, tpb.NUM_SLOTS, -1)  # [N, 12, 128]

    runs = chain_runs(chain)
    total_runs = max(sum(runs.values()), 1)
    run_share = {"2": runs.get(2, 0) / total_runs, "3": runs.get(3, 0) / total_runs,
                 "4": runs.get(4, 0) / total_runs,
                 "5-11": sum(v for k, v in runs.items() if 5 <= k <= 11) / total_runs,
                 "12": runs.get(12, 0) / total_runs}
    flat = length.reshape(-1)
    lengths = {"16": float(np.mean(flat == 16)), "8": float(np.mean(flat == 8)),
               "12": float(np.mean(flat == 12)), "other": float(np.mean(~np.isin(flat, (8, 12, 16))))}

    s = rng.integers(0, tpb.NUM_SLOTS, n)
    t = (s + rng.integers(1, tpb.NUM_SLOTS, n)) % tpb.NUM_SLOTS  # another slot of the same bank
    rows = np.arange(n)
    within = jaccard(trig[rows, s], trig[rows, t])
    adjacent = jaccard(trig[:, :-1], trig[:, 1:])
    other = (rows + rng.integers(1, max(n, 2), n)) % n if n > 1 else rows
    across = jaccard(trig[rows, s], trig[other, rng.integers(0, tpb.NUM_SLOTS, n)])
    lanes = grid[..., 0]  # [N, 12, 8, 16]
    same_lane = (lanes[:, :-1] == lanes[:, 1:]).all(-1)  # [N, 11, 8]

    per_channel = grid[..., 0].sum((1, 3)) / np.maximum(plays.sum((1, 2))[:, None], 1)  # [N, 8]
    return {
        "banks": n,
        "chain_rate": float(chain.mean()),
        "chain_rate_by_link": [float(v) for v in chain.mean(0)],
        "banks_with_chain": float(chain.any(1).mean()),
        "chain_runs": run_share,
        "lengths": lengths,
        "density": {name: float(grid[..., f].sum() / max(steps, 1))
                    for f, name in enumerate(("trigger", "accent", "fill"))},
        "density_by_channel": [float(v) for v in per_channel.mean(0)],
        "jaccard_within_bank": float(within.mean()),
        "jaccard_adjacent": float(adjacent.mean()),
        "jaccard_across_banks": float(across.mean()),
        "same_lane_adjacent": float(same_lane.mean()),
        "_per_channel": per_channel,
    }


def condition_fidelity(generated: dict, data: dict) -> float:
    """Mean over channels of the Pearson r between the per-channel trigger
    density of each generated bank and of the data bank of its condition."""
    rs = []
    for ch in range(tpb.NUM_CHANNELS):
        a, b = generated["_per_channel"][:, ch], data["_per_channel"][:, ch]
        if a.std() > 0 and b.std() > 0:
            rs.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(rs)) if rs else 0.0


def novelty(trig: np.ndarray, train_trig: np.ndarray, chunk: int = 256) -> dict:
    """Each pattern's highest Jaccard against the training patterns:
    trig [P, 128] bool, train_trig [Q, 128] bool."""
    b = torch.from_numpy(train_trig.astype(np.float32))
    b_sum = b.sum(1)
    best = []
    for start in range(0, len(trig), chunk):
        a = torch.from_numpy(trig[start:start + chunk].astype(np.float32))
        inter = a @ b.T
        union = a.sum(1, keepdim=True) + b_sum[None, :] - inter
        best.append(torch.where(union > 0, inter / union.clamp(min=1), torch.ones_like(inter)).max(1).values)
    best = torch.cat(best).numpy()
    return {"closest_train_jaccard": float(best.mean()), "near_copies": float(np.mean(best >= NEAR_COPY))}


# ---------------------------------------------------------------- models

class BankSampler:
    """The bank CVAE of train_pattern_bank.py, through its kit prior when
    the checkpoint has one (prior "kit"), or from N(0, I) (prior "normal")."""

    name = "bank"

    def __init__(self, path, prior="kit", bank_temperature=1.0, slot_temperature=1.0, grid_mode="mixed",
                 length_mode="argmax", chain_mode="threshold"):
        self.model, ckpt = tpb.load_bank_model(str(path))
        self.prior = tpb.load_kit_prior(ckpt) if prior == "kit" else None
        if prior == "kit" and self.prior is None:
            print(f"{path} has no kit prior: sampling from N(0, I)")
        self.options = {"bank_temperature": bank_temperature, "slot_temperature": slot_temperature,
                        "grid_mode": grid_mode, "length_mode": length_mode, "chain_mode": chain_mode}

    def latent(self, n, generator):
        return tpb.latent_noise(self.model, n, generator)

    def sample(self, cond, generator, noise=None):
        return tpb.sample_bank_arrays(self.model, cond, prior=self.prior, generator=generator, noise=noise,
                                      **self.options)


class LegacySampler:
    """The single-pattern CVAE (pythonic/pattern_generator.py): 12 draws
    per bank, the way the app builds a bank with it."""

    name = "legacy"

    def __init__(self, path, temperature=0.7):
        from pythonic.pattern_generator import PatternGenerator
        generator = PatternGenerator()
        generator.load_model(str(path))
        self.model = generator._model
        self.temperature = temperature

    def latent(self, n, generator):
        return torch.randn(n, tpb.NUM_SLOTS, self.model.latent_dim, generator=generator)

    @torch.no_grad()
    def sample(self, cond, generator, noise=None):
        cond = torch.as_tensor(cond, dtype=torch.float32)
        n = cond.shape[0]
        z = (self.latent(n, generator) if noise is None else noise) * self.temperature
        c = cond[:, None, :].expand(-1, tpb.NUM_SLOTS, -1).reshape(n * tpb.NUM_SLOTS, -1)
        probs = torch.sigmoid(self.model.decoder(z.reshape(n * tpb.NUM_SLOTS, -1), c))
        grid = (probs >= 0.5).reshape(n, tpb.NUM_SLOTS, tpb.NUM_CHANNELS, tpb.NUM_STEPS,
                                      tpb.NUM_FEATURES).numpy()
        return (grid, np.full((n, tpb.NUM_SLOTS), tpb.NUM_STEPS, dtype=np.int64),
                np.zeros((n, tpb.NUM_CHAINS), dtype=bool))


def kit_sensitivity(sampler, cond, generator) -> dict:
    """Trigger Jaccard between banks decoded from the same latent noise
    with two conditions (low: the kit matters), and from the same
    condition with two noises (low: the noise matters)."""
    n = cond.shape[0]
    other = torch.roll(cond, 1, dims=0)
    noise = sampler.latent(n, generator)

    def triggers(c, z):
        return sampler.sample(c, generator, noise=z)[0][..., 0].reshape(n, tpb.NUM_SLOTS, -1)
    a = triggers(cond, noise)
    return {"same_latent_other_kit": float(jaccard(a, triggers(other, noise)).mean()),
            "same_kit_other_latent": float(jaccard(a, triggers(cond, sampler.latent(n, generator))).mean())}


# ---------------------------------------------------------------- report

def evaluate(cache: dict, samplers: list, seed: int = 0, novelty_banks: int = 300,
             train_banks: int = 3000, valid_gens=tpb.VALID_GENS) -> dict:
    """Statistics of the validation banks and of each sampler's banks for
    the same conditions."""
    train_idx, valid_idx = tpb.split_indexes(cache, valid_gens)
    cond = cache["cond"][valid_idx]
    data_grid = tpb.unpack_grid(cache["grid"][valid_idx].numpy()).astype(bool)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(train_idx), min(train_banks, len(train_idx)), replace=False)
    train_trig = tpb.unpack_grid(cache["grid"][train_idx[pick]].numpy())[..., 0].astype(bool)
    train_trig = train_trig.reshape(-1, tpb.NUM_CHANNELS * tpb.NUM_STEPS)

    report = {}
    data = bank_stats(data_grid, cache["length"][valid_idx].numpy().astype(np.int64),
                      cache["chain"][valid_idx].numpy().astype(bool), np.random.default_rng(seed))
    data["novelty"] = novelty(data_grid[:novelty_banks, ..., 0].reshape(-1, tpb.NUM_CHANNELS * tpb.NUM_STEPS),
                              train_trig)
    report["data"] = data
    for sampler in samplers:
        generator = torch.Generator().manual_seed(seed)
        grid, length, chain = sampler.sample(cond, generator)
        stats = bank_stats(grid.astype(bool), length.astype(np.int64), chain.astype(bool),
                           np.random.default_rng(seed))
        stats["condition_fidelity"] = condition_fidelity(stats, data)
        stats["kit_sensitivity"] = kit_sensitivity(sampler, cond, generator)
        stats["novelty"] = novelty(grid[:novelty_banks, ..., 0].reshape(-1, tpb.NUM_CHANNELS * tpb.NUM_STEPS),
                                   train_trig)
        report[sampler.name] = stats
    return report


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def print_report(report: dict):
    names = list(report)
    rows = [("banks", lambda r: r["banks"]), ("chain rate", lambda r: r["chain_rate"]),
            ("banks with a chain", lambda r: r["banks_with_chain"])]
    rows += [(f"  {tpb.SLOT_KEYS[s]}→{tpb.SLOT_KEYS[s + 1]}", lambda r, s=s: r["chain_rate_by_link"][s])
             for s in range(tpb.NUM_CHAINS)]
    rows += [(f"runs of {k}", lambda r, k=k: r["chain_runs"][k]) for k in ("2", "3", "4", "5-11", "12")]
    rows += [(f"length {k}", lambda r, k=k: r["lengths"][k]) for k in ("16", "8", "12", "other")]
    rows += [(f"density {k}", lambda r, k=k: r["density"][k]) for k in ("trigger", "accent", "fill")]
    rows += [(f"  ch{c + 1} triggers", lambda r, c=c: r["density_by_channel"][c]) for c in range(tpb.NUM_CHANNELS)]
    rows += [("jaccard within bank", lambda r: r["jaccard_within_bank"]),
             ("jaccard adjacent", lambda r: r["jaccard_adjacent"]),
             ("jaccard across banks", lambda r: r["jaccard_across_banks"]),
             ("same lane, neighbours", lambda r: r["same_lane_adjacent"]),
             ("condition fidelity r", lambda r: r.get("condition_fidelity", "")),
             ("same latent, other kit", lambda r: r.get("kit_sensitivity", {}).get("same_latent_other_kit", "")),
             ("same kit, other latent", lambda r: r.get("kit_sensitivity", {}).get("same_kit_other_latent", "")),
             ("closest train jaccard", lambda r: r["novelty"]["closest_train_jaccard"]),
             ("near copies", lambda r: r["novelty"]["near_copies"])]
    width = max(len(label) for label, _ in rows) + 2
    print("".ljust(width) + "".join(n.rjust(10) for n in names))
    for label, get in rows:
        print(label.ljust(width) + "".join(_fmt(get(report[n])).rjust(10) for n in names))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default=str(ROOT / "drum_patterns" / "pattern_bank_cache.pt"))
    ap.add_argument("--bank", default=str(ROOT / "drum_patterns" / "pattern_bank_best.pt"),
                    help="bank checkpoint (skipped when missing)")
    ap.add_argument("--legacy", help="a single-pattern checkpoint to compare with")
    ap.add_argument("--prior", choices=("kit", "normal"), default="kit",
                    help="the checkpoint's kit prior, or N(0, I)")
    ap.add_argument("--grid-mode", choices=tpb.GRID_MODES, default="mixed")
    ap.add_argument("--bank-temperature", type=float, default=1.0)
    ap.add_argument("--slot-temperature", type=float, default=1.0)
    ap.add_argument("--legacy-temperature", type=float, default=0.7, help="the app's default")
    ap.add_argument("--length-mode", choices=("argmax", "sample"), default="argmax")
    ap.add_argument("--chain-mode", choices=("threshold", "bernoulli"), default="threshold")
    ap.add_argument("--valid-gens", nargs="+", default=list(tpb.VALID_GENS),
                    help="the held-out gen folders (train_pattern_bank.VALID_GENS)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", help="also write the report here")
    args = ap.parse_args()

    cache = tpb.load_bank_cache(args.cache)
    samplers = []
    if pathlib.Path(args.bank).is_file():
        samplers.append(BankSampler(args.bank, args.prior, args.bank_temperature, args.slot_temperature,
                                    args.grid_mode, args.length_mode, args.chain_mode))
    else:
        print(f"no bank checkpoint at {args.bank}")
    if args.legacy:
        samplers.append(LegacySampler(args.legacy, args.legacy_temperature))
    report = evaluate(cache, samplers, seed=args.seed, valid_gens=tuple(args.valid_gens))
    print_report(report)
    if args.json:
        clean = {name: {k: v for k, v in stats.items() if not k.startswith("_")} for name, stats in report.items()}
        pathlib.Path(args.json).write_text(json.dumps(clean, indent=2))
        print(f"report → {args.json}")


if __name__ == "__main__":
    main()
