# %% [markdown]
# # Drum Pattern CVAE — Training Script
#
# Trains a CVAE to generate 8-channel × 16-step drum patterns (triggers,
# accents, fills) from the MicrotonicPatternarium dataset (~29k presets,
# 12 patterns each → up to ~348k samples).
#
# The model is **conditioned on a kit fingerprint** — a compact summary of the
# 8 drum patches in the preset — so that generated patterns match the character
# of the sounds.  This model is designed to work alongside the existing patch
# CVAE: generate patches first, compute their fingerprint, then generate
# matching patterns.
#
# **Usage:**
#   python train_patterns.py
#
# **Outputs (in DRIVE_DIR):**
#   - pattern_dataset_cache.pt  — preprocessed tensors
#   - pattern_cvae_best.pt      — best checkpoint
#   - pattern_cvae_final.pt     — final model

# %% [markdown]
# ## Cell 1 — Configuration

# %%
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
PATTERNARIUM_DIR = os.path.expanduser(
    "~/Documents/dev/MicrotonicPatternarium/patterns"
)
DRIVE_DIR  = "./drum_patterns"
CACHE_PATH = os.path.join(DRIVE_DIR, "pattern_dataset_cache.pt")
BEST_CKPT  = os.path.join(DRIVE_DIR, "pattern_cvae_best.pt")
FINAL_CKPT = os.path.join(DRIVE_DIR, "pattern_cvae_final.pt")

os.makedirs(DRIVE_DIR, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
LATENT_DIM       = 64
HIDDEN_DIM       = 1024
BATCH_SIZE       = 512
LR               = 3e-4
BETA             = 0.0      # pure reconstruction — overfit mode
KL_WARMUP_EPOCHS = 0
KL_FREE_BITS     = 0.0
DROPOUT          = 0.0
EPOCHS           = 100000
LOG_EVERY        = 5
TARGET_LOSS      = None
EARLY_STOPPING_PATIENCE  = 10000
EARLY_STOPPING_MIN_DELTA = 1e-6

# %% [markdown]
# ## Cell 2 — Imports & GPU setup

# %%
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP     = DEVICE.type == "cuda"
NUM_WORKERS = 2 if DEVICE.type == "cuda" else 0
torch.backends.cudnn.benchmark = True
print(f"Device: {DEVICE}  |  AMP: {USE_AMP}  |  DataLoader workers: {NUM_WORKERS}")

# %% [markdown]
# ## Cell 3 — Constants & data definitions

# %%

# ── Pattern grid dimensions ───────────────────────────────────────────────────
NUM_CHANNELS  = 8
NUM_STEPS     = 16
NUM_FEATURES  = 3           # trigger, accent, fill
PATTERN_DIM   = NUM_CHANNELS * NUM_STEPS * NUM_FEATURES  # 384

# ── Kit fingerprint: compact summary of 8 drum patches ───────────────────────
# Per-channel features extracted from the raw patch dict:
KIT_CONTINUOUS_PARAMS = [
    "OscFreq",   # pitch (log-domain)
    "OscDcy",    # body / sustain (log-domain)
    "Mix",       # osc vs noise balance
    "NFilFrq",   # noise character (log-domain)
    "DistAmt",   # drive
]
KIT_LOG_PARAMS = {"OscFreq", "OscDcy", "NFilFrq"}
KIT_PARAM_CLAMP = {
    "OscFreq": (20.0,  20_000.0),
    "OscDcy":  (1.0,   10_000_000.0),
    "Mix":     (0.0,   100.0),
    "NFilFrq": (20.0,  20_000.0),
    "DistAmt": (0.0,   100.0),
}
KIT_CONTINUOUS_DIM = len(KIT_CONTINUOUS_PARAMS) * NUM_CHANNELS  # 5 × 8 = 40

# Step rate vocabulary
STEP_RATES = ["1/8", "1/8T", "1/16", "1/16T", "1/32"]
STEP_RATE_DIM = len(STEP_RATES)

# Global metadata: tempo (1) + swing (1) + fill_rate (1) + step_rate one-hot (5) = 8
GLOBAL_META_DIM = 3 + STEP_RATE_DIM

# Total condition dimension
CONDITION_DIM = KIT_CONTINUOUS_DIM + GLOBAL_META_DIM  # 40 + 8 = 48

# ── Reconstruction weights ────────────────────────────────────────────────────
# Triggers are the most important, accents next, fills least.
FEATURE_WEIGHTS = torch.tensor([3.0, 1.5, 0.5], dtype=torch.float32)  # [trigger, accent, fill]


# %% [markdown]
# ## Cell 4 — Preprocessor

# %%

class PatternPreprocessor:
    """Encodes/decodes pattern data and kit conditions for the CVAE."""

    def __init__(self):
        self.kit_scaler = MinMaxScaler()
        self.tempo_min  = 60.0
        self.tempo_max  = 300.0
        self.fitted     = False

    # ── Kit fingerprint extraction ────────────────────────────────────────────

    def _extract_kit_continuous(self, drum_patches: dict) -> np.ndarray:
        """Extract continuous features from 8 drum patches → flat vector."""
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
                if p in KIT_PARAM_CLAMP:
                    lo, hi = KIT_PARAM_CLAMP[p]
                    v = max(lo, min(hi, v))
                if p in KIT_LOG_PARAMS:
                    v = np.log(v)
                vals.append(v)
        return np.array(vals, dtype=np.float32)

    # ── Global metadata ───────────────────────────────────────────────────────

    def _encode_global_meta(self, preset_data: dict) -> np.ndarray:
        """Encode tempo, swing, fill_rate, step_rate → vector."""
        tempo = float(preset_data.get("Tempo", 120))
        tempo_norm = (np.clip(tempo, self.tempo_min, self.tempo_max) - self.tempo_min) / (self.tempo_max - self.tempo_min)

        swing = float(preset_data.get("Swing", 0.0))
        swing_norm = np.clip(swing / 100.0, 0.0, 1.0)

        fill_rate = float(preset_data.get("FillRate", 4.0))
        fill_rate_norm = np.clip(fill_rate / 8.0, 0.0, 1.0)

        step_rate = str(preset_data.get("StepRate", "1/16"))
        sr_oh = np.zeros(STEP_RATE_DIM, dtype=np.float32)
        if step_rate in STEP_RATES:
            sr_oh[STEP_RATES.index(step_rate)] = 1.0
        else:
            sr_oh[STEP_RATES.index("1/16")] = 1.0  # default

        return np.concatenate([[tempo_norm, swing_norm, fill_rate_norm], sr_oh]).astype(np.float32)

    # ── Pattern encoding ──────────────────────────────────────────────────────

    @staticmethod
    def encode_pattern(pattern_data: dict, num_channels: int = NUM_CHANNELS,
                       num_steps: int = NUM_STEPS) -> np.ndarray:
        """Encode a single pattern → binary vector [8 × 16 × 3 = 384]."""
        vec = np.zeros(NUM_CHANNELS * NUM_STEPS * NUM_FEATURES, dtype=np.float32)
        length = pattern_data.get("Length", num_steps)

        for ch_idx in range(NUM_CHANNELS):
            ch_key = str(ch_idx + 1)
            ch_data = pattern_data.get(ch_key, {})

            if isinstance(ch_data, list):
                # Silent channel encoded as list — skip
                continue

            triggers_str = ch_data.get("Triggers", "")
            accents_str  = ch_data.get("Accents", "")
            fills_str    = ch_data.get("Fills", "")

            for step in range(min(length, num_steps)):
                base = (ch_idx * NUM_STEPS + step) * NUM_FEATURES
                if step < len(triggers_str) and triggers_str[step] == "#":
                    vec[base + 0] = 1.0
                if step < len(accents_str) and accents_str[step] == "#":
                    vec[base + 1] = 1.0
                if step < len(fills_str) and fills_str[step] == "#":
                    vec[base + 2] = 1.0

        return vec

    @staticmethod
    def decode_pattern(vec: np.ndarray, threshold: float = 0.5) -> dict:
        """Decode a probability vector back to pattern dict with # / - strings."""
        pattern = {"Length": NUM_STEPS, "Chained": False}
        for ch_idx in range(NUM_CHANNELS):
            triggers = []
            accents  = []
            fills    = []
            for step in range(NUM_STEPS):
                base = (ch_idx * NUM_STEPS + step) * NUM_FEATURES
                triggers.append("#" if vec[base + 0] >= threshold else "-")
                accents.append("#"  if vec[base + 1] >= threshold else "-")
                fills.append("#"    if vec[base + 2] >= threshold else "-")
            pattern[str(ch_idx + 1)] = {
                "Triggers": "".join(triggers),
                "Accents":  "".join(accents),
                "Fills":    "".join(fills),
            }
        return pattern

    # ── Condition encoding ────────────────────────────────────────────────────

    def encode_condition(self, preset_data: dict) -> np.ndarray:
        """Encode kit fingerprint + global metadata → condition vector."""
        kit_cont = self._extract_kit_continuous(preset_data.get("DrumPatches", {}))
        kit_norm = self.kit_scaler.transform(kit_cont.reshape(1, -1))[0]
        meta     = self._encode_global_meta(preset_data)
        return np.concatenate([kit_norm, meta]).astype(np.float32)

    def decode_condition_meta(self, cond: np.ndarray) -> dict:
        """Extract human-readable global metadata from a condition vector."""
        meta_start = KIT_CONTINUOUS_DIM
        tempo_norm   = cond[meta_start]
        swing_norm   = cond[meta_start + 1]
        fill_norm    = cond[meta_start + 2]
        sr_oh        = cond[meta_start + 3 : meta_start + 3 + STEP_RATE_DIM]

        tempo     = tempo_norm * (self.tempo_max - self.tempo_min) + self.tempo_min
        swing     = swing_norm * 100.0
        fill_rate = fill_norm * 8.0
        step_rate = STEP_RATES[int(np.argmax(sr_oh))]

        return {"tempo": tempo, "swing": swing, "fill_rate": fill_rate, "step_rate": step_rate}

    # ── Fit / persistence ─────────────────────────────────────────────────────

    def fit_kit(self, all_kit_continuous: np.ndarray):
        """Fit the kit scaler on all kit continuous vectors."""
        self.kit_scaler.fit(all_kit_continuous)
        self.fitted = True

    def scaler_state_dict(self) -> dict:
        return {
            "scale_":    self.kit_scaler.scale_.tolist(),
            "min_":      self.kit_scaler.min_.tolist(),
            "data_min_": self.kit_scaler.data_min_.tolist(),
            "data_max_": self.kit_scaler.data_max_.tolist(),
        }

    def load_scaler_state(self, d: dict):
        sc             = self.kit_scaler
        sc.scale_      = np.array(d["scale_"])
        sc.min_        = np.array(d["min_"])
        sc.data_min_   = np.array(d["data_min_"])
        sc.data_max_   = np.array(d["data_max_"])
        sc.data_range_ = sc.data_max_ - sc.data_min_
        self.fitted    = True


# %% [markdown]
# ## Cell 5 — Dataset

# %%

class PatternDataset(Dataset):
    """Wraps pre-stacked (pattern, condition) tensors."""

    def __init__(self, x: torch.Tensor, c: torch.Tensor):
        assert x.shape[0] == c.shape[0]
        self.x = x   # [N, 384]  binary pattern grids
        self.c = c   # [N, 48]   condition vectors

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.c[idx]


# %% [markdown]
# ## Cell 6 — Dataset build / cache

# %%

def build_pattern_cache(patternarium_dir: str, cache_path: str,
                        preprocessor: PatternPreprocessor) -> tuple:
    """Parse all .mtpreset files, extract patterns, fit scaler, save cache."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pythonic.preset_manager import PythonicPresetParser

    parser = PythonicPresetParser()

    # Discover files
    gen_folders = sorted(glob.glob(os.path.join(patternarium_dir, "gen_0*")))
    preset_files = []
    for folder in gen_folders:
        preset_files.extend(sorted(glob.glob(os.path.join(folder, "*.mtpreset"))))
    print(f"Found {len(preset_files)} .mtpreset files in {len(gen_folders)} gen folders.")

    if not preset_files:
        raise RuntimeError(f"No .mtpreset files found under {patternarium_dir}/gen_0*/")

    # ── Pass 1: parse all presets, collect kit continuous vectors for scaler fit ──
    print("Pass 1: Parsing presets & collecting kit features…")
    parsed_presets = []
    kit_continuous_list = []
    errors = 0

    for f in tqdm(preset_files, desc="Parsing"):
        try:
            preset_data = parser.parse_file(f)
            if "Patterns" not in preset_data or "DrumPatches" not in preset_data:
                continue
            kit_cont = preprocessor._extract_kit_continuous(preset_data["DrumPatches"])
            parsed_presets.append(preset_data)
            kit_continuous_list.append(kit_cont)
        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"  ✗ {os.path.basename(f)}: {e}")

    print(f"Parsed {len(parsed_presets)} valid presets ({errors} errors).")

    # Fit kit scaler
    all_kit_cont = np.stack(kit_continuous_list)
    preprocessor.fit_kit(all_kit_cont)

    # ── Pass 2: encode all patterns ──────────────────────────────────────────
    print("Pass 2: Encoding patterns…")
    pattern_vecs = []
    condition_vecs = []

    pattern_keys = [chr(ord("a") + i) for i in range(12)]  # a-l

    for preset_data in tqdm(parsed_presets, desc="Encoding"):
        cond = preprocessor.encode_condition(preset_data)
        patterns_block = preset_data["Patterns"]

        for pk in pattern_keys:
            if pk not in patterns_block:
                continue
            pat = patterns_block[pk]
            # Skip patterns with zero triggers (completely empty)
            pvec = PatternPreprocessor.encode_pattern(pat)
            if pvec.sum() == 0:
                continue
            pattern_vecs.append(pvec)
            condition_vecs.append(cond)

    print(f"Total pattern samples: {len(pattern_vecs)}")

    x = torch.tensor(np.stack(pattern_vecs), dtype=torch.float32)
    c = torch.tensor(np.stack(condition_vecs), dtype=torch.float32)

    torch.save({
        "x":      x,
        "c":      c,
        "scaler": preprocessor.scaler_state_dict(),
    }, cache_path)
    print(f"Cache saved → {cache_path}")

    return PatternDataset(x, c), preprocessor


def load_pattern_cache(cache_path: str,
                       preprocessor: PatternPreprocessor) -> tuple:
    ckpt = torch.load(cache_path, map_location="cpu")
    preprocessor.load_scaler_state(ckpt["scaler"])
    ds = PatternDataset(ckpt["x"], ckpt["c"])
    print(f"Cache loaded ← {cache_path}  ({len(ds)} samples)")
    return ds, preprocessor


def get_pattern_dataset(patternarium_dir: str, cache_path: str) -> tuple:
    preprocessor = PatternPreprocessor()
    if os.path.exists(cache_path):
        print(f"Found cache at {cache_path} — skipping parsing.")
        return load_pattern_cache(cache_path, preprocessor)
    return build_pattern_cache(patternarium_dir, cache_path, preprocessor)


# %% [markdown]
# ## Cell 7 — Pattern CVAE model

# %%

class PatternEncoder(nn.Module):
    def __init__(self, pattern_dim, cond_dim, latent_dim, hidden_dim=1024, dropout=0.0):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(pattern_dim + cond_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.res_block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.res_block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.res_act = nn.GELU()
        self.compress = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mu_head     = nn.Linear(hidden_dim // 2, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim // 2, latent_dim)

    def forward(self, x, c):
        h = self.input_proj(torch.cat([x, c], dim=-1))
        h = self.res_act(h + self.res_block1(h))
        h = self.res_act(h + self.res_block2(h))
        h = self.compress(h)
        return self.mu_head(h), self.logvar_head(h)


class PatternDecoder(nn.Module):
    def __init__(self, latent_dim, cond_dim, pattern_dim, hidden_dim=1024, dropout=0.0):
        super().__init__()
        self.pattern_dim = pattern_dim
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.expand = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.res_block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.res_block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.res_act = nn.GELU()
        self.out_head = nn.Linear(hidden_dim, pattern_dim)

    def forward(self, z, c):
        h = self.input_proj(torch.cat([z, c], dim=-1))
        h = self.expand(h)
        h = self.res_act(h + self.res_block1(h))
        h = self.res_act(h + self.res_block2(h))
        return self.out_head(h)  # raw logits — sigmoid applied in loss / inference


class PatternCVAE(nn.Module):
    def __init__(self, pattern_dim, cond_dim,
                 latent_dim=64, hidden_dim=1024, dropout=0.0):
        super().__init__()
        self.encoder    = PatternEncoder(pattern_dim, cond_dim, latent_dim, hidden_dim, dropout)
        self.decoder    = PatternDecoder(latent_dim, cond_dim, pattern_dim, hidden_dim, dropout)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.dropout    = dropout

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x, c):
        mu, logvar = self.encoder(x, c)
        z          = self.reparameterize(mu, logvar)
        logits     = self.decoder(z, c)
        return logits, mu, logvar


# %% [markdown]
# ## Cell 8 — Loss

# %%

def pattern_cvae_loss(logits, x, mu, logvar, beta=0.0, free_bits=0.0):
    """
    Binary cross-entropy for the pattern grid + optional KL.

    logits: [B, 384] raw decoder output
    x:      [B, 384] binary targets
    """
    # Per-feature weights: expand [3] → [384] to match the interleaved layout
    # Layout is (ch0_step0_trig, ch0_step0_acc, ch0_step0_fill, ch0_step1_trig, ...)
    weights = FEATURE_WEIGHTS.to(logits.device)
    weight_vec = weights.repeat(NUM_CHANNELS * NUM_STEPS)  # [384]

    bce = F.binary_cross_entropy_with_logits(
        logits, x, weight=weight_vec, reduction="mean"
    )

    # KL divergence
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim = kl_per_dim.mean(dim=0)
    if free_bits > 0:
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    kl = kl_per_dim.sum()

    return bce + beta * kl, bce, kl


# %% [markdown]
# ## Cell 9 — Trainer

# %%

class PatternTrainer:
    def __init__(
        self,
        train_dataset:  PatternDataset,
        preprocessor:   PatternPreprocessor,
        latent_dim:     int   = 64,
        hidden_dim:     int   = 1024,
        batch_size:     int   = 512,
        lr:             float = 3e-4,
        beta:           float = 0.0,
        kl_warmup_epochs: int = 0,
        kl_free_bits:   float = 0.0,
        dropout:        float = 0.0,
        device:         torch.device = None,
    ):
        self.device           = device or DEVICE
        self.beta             = beta
        self.kl_warmup_epochs = kl_warmup_epochs
        self.kl_free_bits     = kl_free_bits
        self.preprocessor     = preprocessor
        self.use_amp          = USE_AMP
        self.scaler_amp       = GradScaler("cuda", enabled=USE_AMP)

        pin = self.device.type == "cuda"
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            drop_last=True, num_workers=NUM_WORKERS, pin_memory=pin,
            persistent_workers=(NUM_WORKERS > 0),
        )

        self.model = PatternCVAE(
            pattern_dim = PATTERN_DIM,
            cond_dim    = CONDITION_DIM,
            latent_dim  = latent_dim,
            hidden_dim  = hidden_dim,
            dropout     = dropout,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.best_recon = float("inf")
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self._pending_scheduler_state = None

    def train(
        self,
        epochs:          int   = 50,
        log_every:       int   = 5,
        checkpoint_path: str   = BEST_CKPT,
        target_loss:     float = None,
        early_stopping_patience:  int   = None,
        early_stopping_min_delta: float = 0.0,
    ) -> list:
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=200, min_lr=1e-6,
        )

        history = []
        start_epoch = 1

        if not os.path.exists(checkpoint_path):
            self.best_recon = float("inf")
            self.best_epoch = 0
            self.epochs_without_improvement = 0

        if os.path.exists(checkpoint_path):
            print(f"Resuming from checkpoint: {checkpoint_path}")
            start_epoch = self.load(checkpoint_path) + 1
            print(f"  Resuming from epoch {start_epoch}")

        if hasattr(self, '_pending_scheduler_state') and self._pending_scheduler_state is not None:
            self.scheduler.load_state_dict(self._pending_scheduler_state)
            self._pending_scheduler_state = None

        for epoch in range(start_epoch, epochs + 1):

            # KL annealing (simple linear warmup)
            if self.kl_warmup_epochs > 0:
                annealed_beta = self.beta * min(1.0, epoch / self.kl_warmup_epochs)
            else:
                annealed_beta = self.beta

            # ── Training ──────────────────────────────────────────────────────
            self.model.train()
            total_loss = bce_sum = kl_sum = 0.0

            for x, c in self.train_loader:
                x = x.to(self.device, non_blocking=True)
                c = c.to(self.device, non_blocking=True)

                with autocast("cuda", enabled=self.use_amp):
                    logits, mu, logvar = self.model(x, c)
                    loss, bce, kl = pattern_cvae_loss(
                        logits, x, mu, logvar,
                        beta=annealed_beta, free_bits=self.kl_free_bits,
                    )

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler_amp.scale(loss).backward()
                self.scaler_amp.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler_amp.step(self.optimizer)
                self.scaler_amp.update()

                total_loss += loss.item()
                bce_sum    += bce.item()
                kl_sum     += kl.item()

            n = len(self.train_loader)
            train_bce = bce_sum / n
            h = {"epoch": epoch, "beta": annealed_beta,
                 "loss": total_loss / n, "bce": train_bce, "kl": kl_sum / n}
            history.append(h)

            warmup_done = (self.kl_warmup_epochs <= 0 or epoch >= self.kl_warmup_epochs)

            if warmup_done:
                self.scheduler.step(train_bce)

                improved = train_bce < (self.best_recon - early_stopping_min_delta)
                if improved:
                    self.best_recon = train_bce
                    self.best_epoch = epoch
                    self.epochs_without_improvement = 0
                    self.save(checkpoint_path, epoch)
                    print(f"  ✓ Checkpoint saved (epoch {epoch}, BCE {self.best_recon:.6f})")
                else:
                    self.epochs_without_improvement += 1

            if epoch % log_every == 0:
                lr_now = self.optimizer.param_groups[0]['lr']
                warmup_tag = "" if warmup_done else " [WARMUP]"
                print(
                    f"Epoch {epoch:>5}/{epochs} | β {annealed_beta:.3f} | lr {lr_now:.2e} | "
                    f"Loss {h['loss']:.4f} | BCE {train_bce:.4f} | KL {h['kl']:.4f} | "
                    f"Best BCE {self.best_recon:.6f} @ {self.best_epoch} | "
                    f"Wait {self.epochs_without_improvement}{warmup_tag}"
                )

            if target_loss is not None and train_bce <= target_loss:
                print(f"Target BCE {target_loss} reached at epoch {epoch}. Stopping.")
                break

            if (
                early_stopping_patience is not None
                and self.epochs_without_improvement >= early_stopping_patience
            ):
                print(
                    f"Early stopping at epoch {epoch}: no improvement > "
                    f"{early_stopping_min_delta:.1e} for {self.epochs_without_improvement} epochs. "
                    f"Best epoch {self.best_epoch} with BCE {self.best_recon:.6f}."
                )
                break

        return history

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def save(self, path: str, epoch: int = 0):
        torch.save({
            "model_state":     self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_amp_state": self.scaler_amp.state_dict(),
            "epoch":           epoch,
            "best_recon":      self.best_recon,
            "best_epoch":      self.best_epoch,
            "epochs_without_improvement": self.epochs_without_improvement,
            "model_config": {
                "pattern_dim": PATTERN_DIM,
                "cond_dim":    CONDITION_DIM,
                "latent_dim":  self.model.latent_dim,
                "hidden_dim":  self.model.hidden_dim,
                "dropout":     self.model.dropout,
            },
            "scaler": self.preprocessor.scaler_state_dict(),
        }, path)

    def load(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        cfg  = ckpt["model_config"]
        self.model = PatternCVAE(**cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.preprocessor.load_scaler_state(ckpt["scaler"])
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.optimizer.param_groups[0]["lr"]
        )
        self.scaler_amp = GradScaler("cuda", enabled=self.use_amp)
        if "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scaler_amp_state" in ckpt:
            self.scaler_amp.load_state_dict(ckpt["scaler_amp_state"])
        self._pending_scheduler_state = ckpt.get("scheduler_state")
        self.best_recon = ckpt.get("best_recon", float("inf"))
        self.best_epoch = ckpt.get("best_epoch", ckpt.get("epoch", 0))
        self.epochs_without_improvement = ckpt.get("epochs_without_improvement", 0)
        epoch = ckpt.get("epoch", 0)
        print(f"Loaded from {path} (epoch {epoch})")
        return epoch


# %% [markdown]
# ## Cell 10 — Pattern Generator

# %%

class PatternGenerator:
    def __init__(self, trainer: PatternTrainer):
        self.model        = trainer.model
        self.preprocessor = trainer.preprocessor
        self.device       = trainer.device
        self.use_amp      = trainer.use_amp
        self.model.eval()

    def generate(self, condition: np.ndarray, n: int = 1,
                 temperature: float = 1.0, threshold: float = 0.5) -> list:
        """
        Generate n patterns given a condition vector.

        Args:
            condition:    Condition vector from preprocessor.encode_condition()
            n:            Number of patterns to generate
            temperature:  Latent sampling temperature
            threshold:    Sigmoid threshold for binarising output
        Returns:
            List of pattern dicts with Triggers/Accents/Fills strings.
        """
        c = (torch.tensor(condition, dtype=torch.float32)
               .unsqueeze(0).repeat(n, 1).to(self.device))

        with torch.no_grad(), autocast("cuda", enabled=self.use_amp):
            z      = torch.randn(n, self.model.latent_dim, device=self.device) * temperature
            logits = self.model.decoder(z, c)
            probs  = torch.sigmoid(logits).float().cpu().numpy()

        return [
            PatternPreprocessor.decode_pattern(probs[i], threshold=threshold)
            for i in range(n)
        ]

    def generate_for_preset(self, preset_data: dict, n: int = 1,
                            temperature: float = 1.0, threshold: float = 0.5) -> list:
        """
        Generate patterns conditioned on a full preset dict (with DrumPatches).
        Convenience wrapper that computes the condition vector automatically.
        """
        cond = self.preprocessor.encode_condition(preset_data)
        return self.generate(cond, n=n, temperature=temperature, threshold=threshold)


# %% [markdown]
# ## Cell 11 — Round-trip sanity check

# %%

def round_trip_test(preprocessor: PatternPreprocessor, patternarium_dir: str,
                    n_samples: int = 20):
    """Encode a few real patterns, decode them, verify exact binary match."""
    from pythonic.preset_manager import PythonicPresetParser

    parser = PythonicPresetParser()
    gen_folders = sorted(glob.glob(os.path.join(patternarium_dir, "gen_0*")))
    preset_files = []
    for folder in gen_folders[:3]:  # just first 3 gen folders
        preset_files.extend(sorted(glob.glob(os.path.join(folder, "*.mtpreset"))))

    if not preset_files:
        print("No preset files found for round-trip test.")
        return

    rng = np.random.default_rng(42)
    indices = rng.choice(len(preset_files), size=min(n_samples, len(preset_files)), replace=False)

    perfect = 0
    total   = 0
    max_errors = 0

    for idx in indices:
        f = preset_files[idx]
        try:
            preset_data = parser.parse_file(f)
        except Exception:
            continue
        if "Patterns" not in preset_data:
            continue

        for pk in ["a", "b", "c"]:
            if pk not in preset_data["Patterns"]:
                continue
            pat = preset_data["Patterns"][pk]
            vec = PatternPreprocessor.encode_pattern(pat)
            if vec.sum() == 0:
                continue
            rebuilt = PatternPreprocessor.decode_pattern(vec, threshold=0.5)

            # Compare
            errors = 0
            for ch in range(NUM_CHANNELS):
                ch_key = str(ch + 1)
                orig_ch = pat.get(ch_key, {})
                rebu_ch = rebuilt.get(ch_key, {})
                if isinstance(orig_ch, list):
                    continue
                for feat in ["Triggers", "Accents", "Fills"]:
                    orig_str = orig_ch.get(feat, "")[:NUM_STEPS]
                    rebu_str = rebu_ch.get(feat, "")[:NUM_STEPS]
                    orig_padded = orig_str.ljust(NUM_STEPS, "-")
                    for s in range(len(rebu_str)):
                        if s < len(orig_padded) and rebu_str[s] != orig_padded[s]:
                            errors += 1

            total += 1
            if errors == 0:
                perfect += 1
            max_errors = max(max_errors, errors)

    if total > 0:
        print(f"Round-trip: {perfect}/{total} patterns perfect, max errors in one pattern: {max_errors}")
    else:
        print("No patterns tested.")


# %% [markdown]
# ## Cell 12 — Build or load dataset

# %%
if __name__ == "__main__":
    train_ds, preprocessor = get_pattern_dataset(PATTERNARIUM_DIR, CACHE_PATH)
    print(f"Dataset: {len(train_ds)} pattern samples, condition dim: {CONDITION_DIM}")

    # %% [markdown]
    # ## Cell 12b — Round-trip test

    # %%
    round_trip_test(preprocessor, PATTERNARIUM_DIR)

    # %% [markdown]
    # ## Cell 13 — Train

    # %%
    trainer = PatternTrainer(
        train_dataset    = train_ds,
        preprocessor     = preprocessor,
        latent_dim       = LATENT_DIM,
        hidden_dim       = HIDDEN_DIM,
        batch_size       = BATCH_SIZE,
        lr               = LR,
        beta             = BETA,
        kl_warmup_epochs = KL_WARMUP_EPOCHS,
        kl_free_bits     = KL_FREE_BITS,
        dropout          = DROPOUT,
    )

    history = trainer.train(
        epochs          = EPOCHS,
        log_every       = LOG_EVERY,
        checkpoint_path = BEST_CKPT,
        target_loss     = TARGET_LOSS,
        early_stopping_patience  = EARLY_STOPPING_PATIENCE,
        early_stopping_min_delta = EARLY_STOPPING_MIN_DELTA,
    )

    trainer.save(FINAL_CKPT, epoch=history[-1]["epoch"] if history else 0)
    print(f"\nFinal model saved → {FINAL_CKPT}")

    # %% [markdown]
    # ## Cell 14 — Generate patterns

    # %%
    generator = PatternGenerator(trainer)

    # Generate a pattern using the condition from the first training sample
    sample_cond = train_ds.c[0].numpy()
    patterns = generator.generate(sample_cond, n=3, temperature=1.0, threshold=0.5)

    for i, pat in enumerate(patterns):
        print(f"\n── Generated pattern {i+1} ──")
        for ch in range(1, 9):
            ch_data = pat[str(ch)]
            print(f"  Ch {ch}: T={ch_data['Triggers']}  A={ch_data['Accents']}  F={ch_data['Fills']}")

    meta = preprocessor.decode_condition_meta(sample_cond)
    print(f"\nCondition metadata: {meta}")
