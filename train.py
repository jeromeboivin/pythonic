# %% [markdown]
# # Drum CVAE — Google Colab Training Script
#
# **Before running:**
# 1. Runtime → Change runtime type → **T4 GPU**
# 2. Upload your `.zip` of drum patch folders to Drive (or directly to Colab)
# 3. Upload `pythonic/` (your custom parser package) to the Colab working dir,
#    or install it via pip if you have a wheel: `!pip install your_package.whl`
# 4. Run cells top to bottom. The dataset is preprocessed once and saved to a
#    cache file — subsequent runs skip parsing entirely and load in seconds.
#
# **Files written to Drive (DRIVE_DIR):**
#   - `drum_dataset_cache.pt`  ← preprocessed tensors + scaler state
#   - `drum_cvae_best.pt`      ← best checkpoint (saved automatically)
#   - `drum_cvae_final.pt`     ← final model after training

# %% [markdown]
# ## Cell 1 — Install dependencies

# %%
# !pip install -q scikit-learn tqdm
# If your parser isn't already installed, upload the pythonic/ folder to Colab
# and uncomment: !pip install -e /content/pythonic  (or wherever you placed it)

# %% [markdown]
# ## Cell 2 — Mount Google Drive & configure paths

# %%
import os

# ── Mount Drive (comment out if not using Drive) ──────────────────────────────
# from google.colab import drive
# drive.mount("/content/drive")

# ── Paths — edit these ────────────────────────────────────────────────────────
# Root folder that contains your drum-type subfolders (bd/, sd/, oh/, …)
# PATCHES_DIR  = "/content/drum_patches"
PATCHES_DIR  = "./drum_patches"  # current working dir (for non-Drive use)

# All outputs land here. Change to a Drive path to persist across sessions, e.g.
#   "/content/drive/MyDrive/drum_cvae"
# DRIVE_DIR    = "/content/drum_cvae_output"
DRIVE_DIR    = "."  # current working dir (no persistence, but simpler for quick tests)

CACHE_PATH   = os.path.join(DRIVE_DIR, "drum_dataset_cache.pt")
BEST_CKPT    = os.path.join(DRIVE_DIR, "drum_cvae_best.pt")
FINAL_CKPT   = os.path.join(DRIVE_DIR, "drum_cvae_final.pt")

# os.makedirs(DRIVE_DIR, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
LATENT_DIM       = 64
HIDDEN_DIM       = 1024
BATCH_SIZE       = 256
LR               = 3e-4
BETA             = 1e-5   # KL is summed over latent dims while recon is a mean,
                          # so useful values are tiny in this script.
KL_WARMUP_EPOCHS = 1000
KL_FREE_BITS     = 0.01
KL_CYCLICAL      = False
KL_CYCLE_EPOCHS  = 200
DROPOUT          = 0.0
EPOCHS           = 4000
LOG_EVERY        = 5
TARGET_LOSS      = None
EARLY_STOPPING_PATIENCE  = 600
EARLY_STOPPING_MIN_DELTA = 5e-6

# %% [markdown]
# ## Cell 3 — Imports, constants, and all class/function definitions

# %%
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm   # auto picks the notebook progress bar in Colab

# ── GPU setup ─────────────────────────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP     = DEVICE.type == "cuda"        # Automatic Mixed Precision on GPU only
NUM_WORKERS = 2 if DEVICE.type == "cuda" else 0   # Colab gives 2 CPU cores
torch.backends.cudnn.benchmark = True      # auto-tune cuDNN for fixed input sizes
print(f"Device: {DEVICE}  |  AMP: {USE_AMP}  |  DataLoader workers: {NUM_WORKERS}")

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

EXCLUDED_PARAMS = {"Pan", "NStereo", "Output", "Name"}

# "Pan" intentionally omitted — it's in EXCLUDED_PARAMS and always reset to 0.0
# in decode_patch, so training on it would waste a model dimension.
# NOTE: OscAtk removed — never present in .mtdrum files (phantom parameter).
CONTINUOUS_PARAMS = [
    "OscFreq",   # Oscillator frequency (Hz)          — log-domain
    "OscDcy",    # Oscillator envelope decay (ms)      — log-domain
    "ModAmt",    # Pitch modulation amount (semitones)  — linear (signed!)
    "ModRate",   # Pitch modulation rate (ms or Hz)     — log-domain
    "NFilFrq",   # Noise filter frequency (Hz)          — log-domain
    "NFilQ",     # Noise filter Q                       — log-domain
    "NEnvAtk",   # Noise envelope attack (ms)           — log-domain
    "NEnvDcy",   # Noise envelope decay (ms)            — log-domain
    "Mix",       # Osc/Noise mix (0-100)                — linear
    "DistAmt",   # Distortion amount (0-100)            — linear
    "EQFreq",    # EQ frequency (Hz)                    — log-domain
    "EQGain",    # EQ gain (dB, already log)            — linear
    "Level",     # Output level (dB, already log)       — linear
    "OscVel",    # Osc velocity sensitivity (0-200)     — linear
    "NVel",      # Noise velocity sensitivity (0-200)   — linear
    "ModVel",    # Mod velocity sensitivity (0-200)     — linear
]

# Parameters that live in logarithmic perceptual domains (frequencies, times, Q).
# These are log-transformed before MinMaxScaler and exp-transformed on decode.
LOG_PARAMS = {"OscFreq", "OscDcy", "ModRate", "NFilFrq", "NFilQ", "NEnvAtk", "NEnvDcy", "EQFreq"}

# Index lookup for fast numpy vectorised log-transform
_LOG_PARAM_INDICES = [i for i, p in enumerate(CONTINUOUS_PARAMS) if p in LOG_PARAMS]

# Per-parameter reconstruction weights — upweight perceptually critical params.
# Pitch/frequency and mix balance errors are far more audible than velocity sensitivity,
# so we weight them higher in the loss to focus the model on what matters sonically.
_PARAM_WEIGHT_MAP = {
    "OscFreq": 3.0,   # fundamental pitch — most audible
    "OscDcy":  2.0,    # body / sustain
    "ModAmt":  2.5,    # punch / snap
    "ModRate": 2.0,    # modulation speed
    "NFilFrq": 2.0,    # noise character
    "NEnvDcy": 1.5,    # noise tail
    "Mix":     2.5,    # osc vs noise balance
    "DistAmt": 1.5,    # drive / saturation
}
_CONT_PARAM_WEIGHTS = torch.tensor(
    [_PARAM_WEIGHT_MAP.get(p, 1.0) for p in CONTINUOUS_PARAMS], dtype=torch.float32
)

# Clamp raw values to synth-usable ranges before any transform.
# Prevents extreme outliers (e.g. ModRate up to 1.87 billion) from dominating scaler.
PARAM_CLAMP = {
    "OscFreq": (20.0,     20_000.0),
    "OscDcy":  (1.0,      10_000_000.0),
    "ModAmt":  (-96.0,    96.0),
    "ModRate": (0.001,    100_000.0),
    "NFilFrq": (20.0,     20_000.0),
    "NFilQ":   (0.5,      100.0),       # synth clips to 0.5–100 internally
    "NEnvAtk": (0.001,    100_000.0),
    "NEnvDcy": (1.0,      10_000_000.0),
    "Mix":     (0.0,      100.0),
    "DistAmt": (0.0,      100.0),
    "EQFreq":  (20.0,     20_000.0),
    "EQGain":  (-40.0,    40.0),
    "Level":   (-50.0,    20.0),
    "OscVel":  (0.0,      200.0),
    "NVel":    (0.0,      200.0),
    "ModVel":  (0.0,      200.0),
}

CATEGORICAL_PARAMS = {
    "OscWave": ["Sine", "Triangle", "Saw"],
    "ModMode": ["Decay", "Sine", "Noise"],
    "NFilMod": ["LP", "BP", "HP"],
    "NEnvMod": ["Exp", "Linear", "Mod"],
}

DRUM_TYPES = [
    "bass", "bd", "blip", "ch", "clap", "cowbell", "cy", "fuzz", "fx",
    "oh", "perc", "reverse", "sd", "shaker", "synth", "tom", "zap", "other",
]

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def infer_drum_type(name: str) -> str:
    """Infer drum type from the parent folder of the patch file path."""
    folder = os.path.basename(os.path.dirname(name)).lower()
    return folder if folder in DRUM_TYPES else "other"

# ─────────────────────────────────────────────
# Preprocessor
# ─────────────────────────────────────────────

class PatchPreprocessor:
    def __init__(self):
        self.scaler      = MinMaxScaler()
        self.cat_sizes   = {k: len(v) for k, v in CATEGORICAL_PARAMS.items()}
        self.cat_to_idx  = {k: {v: i for i, v in enumerate(vals)} for k, vals in CATEGORICAL_PARAMS.items()}
        self.idx_to_cat  = {k: {i: v for i, v in enumerate(vals)} for k, vals in CATEGORICAL_PARAMS.items()}
        self.type_to_idx = {t: i for i, t in enumerate(DRUM_TYPES)}
        self.idx_to_type = {i: t for i, t in enumerate(DRUM_TYPES)}
        self.fitted      = False

        self.cont_dim  = len(CONTINUOUS_PARAMS)
        self.cat_dim   = sum(self.cat_sizes.values())
        self.param_dim = self.cont_dim + self.cat_dim
        self.type_dim  = len(DRUM_TYPES)

    def _extract_continuous(self, patch: dict) -> np.ndarray:
        """Extract, clamp, and log-transform continuous parameters."""
        vals = []
        for p in CONTINUOUS_PARAMS:
            v = patch.get(p, 0.0)
            if isinstance(v, (tuple, list)):
                v = v[0]
            try:
                v = float(v)
            except Exception:
                v = 0.0
            # Clamp to synth-usable range
            if p in PARAM_CLAMP:
                lo, hi = PARAM_CLAMP[p]
                v = max(lo, min(hi, v))
            # Log-transform for perceptually logarithmic params
            if p in LOG_PARAMS:
                v = np.log(v)
            vals.append(v)
        return np.array(vals, dtype=np.float32)

    def _extract_categorical_onehot(self, patch: dict) -> np.ndarray:
        vecs = []
        for param, vals in CATEGORICAL_PARAMS.items():
            val = patch.get(param, vals[0])
            idx = self.cat_to_idx[param].get(val, 0)
            oh  = np.zeros(len(vals), dtype=np.float32)
            oh[idx] = 1.0
            vecs.append(oh)
        return np.concatenate(vecs)

    def fit(self, patches: list):
        cont_data = np.stack([self._extract_continuous(p) for p in patches])
        self.scaler.fit(cont_data)
        self.fitted = True

    def encode_patch(self, patch: dict) -> np.ndarray:
        cont      = self._extract_continuous(patch)
        cont_norm = self.scaler.transform(cont.reshape(1, -1))[0]
        cat_oh    = self._extract_categorical_onehot(patch)
        return np.concatenate([cont_norm, cat_oh])

    def encode_type(self, drum_type: str) -> np.ndarray:
        idx = self.type_to_idx.get(drum_type, self.type_to_idx["other"])
        oh  = np.zeros(self.type_dim, dtype=np.float32)
        oh[idx] = 1.0
        return oh

    def decode_patch(self, vector: np.ndarray, drum_type: str, name: str = None) -> dict:
        """Convert model output vector back to a patch dict."""
        cont_norm = vector[:self.cont_dim]
        cont      = self.scaler.inverse_transform(cont_norm.reshape(1, -1))[0]

        # Clamp to the scaler's fitted range (allows negative values for signed params)
        cont = np.clip(cont, self.scaler.data_min_, self.scaler.data_max_)

        # Exp-transform to undo log for log-domain params
        for idx in _LOG_PARAM_INDICES:
            cont[idx] = np.exp(cont[idx])

        patch = {param: float(cont[i]) for i, param in enumerate(CONTINUOUS_PARAMS)}

        offset = self.cont_dim
        for param, vals in CATEGORICAL_PARAMS.items():
            size         = len(vals)
            idx          = int(np.argmax(vector[offset: offset + size]))
            patch[param] = vals[idx]
            offset      += size

        patch.update({"Pan": 0.0, "NStereo": "Off", "Output": "A",
                      "Name": name or f"Gen {drum_type.upper()}"})
        return patch

    # ── Scaler persistence helpers ────────────────────────────────────────────

    def scaler_state_dict(self) -> dict:
        return {
            "scale_":    self.scaler.scale_.tolist(),
            "min_":      self.scaler.min_.tolist(),
            "data_min_": self.scaler.data_min_.tolist(),
            "data_max_": self.scaler.data_max_.tolist(),
        }

    def load_scaler_state(self, d: dict):
        sc             = self.scaler
        sc.scale_      = np.array(d["scale_"])
        sc.min_        = np.array(d["min_"])
        sc.data_min_   = np.array(d["data_min_"])
        sc.data_max_   = np.array(d["data_max_"])
        sc.data_range_ = sc.data_max_ - sc.data_min_
        self.fitted    = True

# ─────────────────────────────────────────────
# Dataset — backed by pre-stacked tensors for fast GPU DataLoader transfers
# ─────────────────────────────────────────────

class PatchDataset(Dataset):
    """
    Wraps two pre-stacked float32 tensors (x, c) and a list of drum-type
    strings. Indexing is O(1) and pin_memory works efficiently because the
    tensors are already contiguous in CPU memory.
    """
    def __init__(self, x: torch.Tensor, c: torch.Tensor, types: list):
        assert x.shape[0] == c.shape[0] == len(types)
        self.x     = x
        self.c     = c
        self.types = types

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.c[idx], self.types[idx]

    @staticmethod
    def from_patches(patches: list, preprocessor: PatchPreprocessor) -> "PatchDataset":
        xs, cs, types = [], [], []
        for p in patches:
            name      = p.get("Name", "")
            drum_type = infer_drum_type(name)
            xs.append(preprocessor.encode_patch(p))
            cs.append(preprocessor.encode_type(drum_type))
            types.append(drum_type)
        return PatchDataset(
            torch.tensor(np.stack(xs), dtype=torch.float32),
            torch.tensor(np.stack(cs), dtype=torch.float32),
            types,
        )


def build_empirical_latent_bank(model: nn.Module, dataset: PatchDataset,
                                device: torch.device) -> tuple[dict, dict]:
    """Encode the training set and build per-type latent anchors for sampling."""
    mus_by_type = {drum_type: [] for drum_type in DRUM_TYPES}
    batch_size = 1024

    model.eval()
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            end = min(start + batch_size, len(dataset))
            x_batch = dataset.x[start:end].to(device)
            c_batch = dataset.c[start:end].to(device)
            mu_batch, _ = model.encoder(x_batch, c_batch)
            mu_batch = mu_batch.cpu()
            for row_idx, drum_type in enumerate(dataset.types[start:end]):
                if drum_type in mus_by_type:
                    mus_by_type[drum_type].append(mu_batch[row_idx])

    latent_bank = {}
    latent_jitter = {}
    for drum_type, rows in mus_by_type.items():
        if not rows:
            continue
        mus = torch.stack(rows).to(device)
        median = mus.median(dim=0).values
        mad = (mus - median).abs().median(dim=0).values
        latent_bank[drum_type] = mus
        latent_jitter[drum_type] = torch.clamp(mad * 0.15, min=0.01, max=0.5)

    return latent_bank, latent_jitter

# ─────────────────────────────────────────────
# Dataset cache — build once, reload in seconds
# ─────────────────────────────────────────────

def build_cache(patch_files: list, cache_path: str, preprocessor: PatchPreprocessor) -> tuple:
    """
    Parse all .mtdrum files, encode every patch, fit the scaler, and save
    everything to a single .pt cache file.
    Returns (dataset, preprocessor).
    """
    from pythonic.preset_manager import DrumPatchParser
    patch_parser = DrumPatchParser()

    patches = []
    for f in tqdm(patch_files, desc="Parsing patches"):
        try:
            patch         = patch_parser.parse_file(f)
            patch["Name"] = f       # path → drum-type inference via infer_drum_type
            patches.append(patch)
        except Exception as e:
            print(f"  ✗ {f}: {e}")

    print(f"Parsed {len(patches)} patches.")
    if not patches:
        raise RuntimeError("No valid patches found. Check PATCHES_DIR.")

    preprocessor.fit(patches)

    print("Encoding dataset…")
    ds = PatchDataset.from_patches(patches, preprocessor)

    torch.save({
        "train_x":     ds.x,
        "train_c":     ds.c,
        "train_types": ds.types,
        "scaler":      preprocessor.scaler_state_dict(),
    }, cache_path)
    print(f"Cache saved → {cache_path}")

    return ds, preprocessor


def load_cache(cache_path: str, preprocessor: PatchPreprocessor) -> tuple:
    """
    Restore dataset and the fitted scaler from a cache file.
    Returns (dataset, preprocessor).
    """
    ckpt = torch.load(cache_path, map_location="cpu")
    preprocessor.load_scaler_state(ckpt["scaler"])
    ds = PatchDataset(ckpt["train_x"], ckpt["train_c"], ckpt["train_types"])
    print(f"Cache loaded ← {cache_path}  ({len(ds)} samples)")
    return ds, preprocessor


def get_datasets(patches_dir: str, cache_path: str) -> tuple:
    """
    High-level helper: load from cache if it exists, otherwise build and save it.
    Returns (dataset, preprocessor).
    """
    preprocessor = PatchPreprocessor()

    if os.path.exists(cache_path):
        print(f"Found cache at {cache_path} — skipping parsing.")
        return load_cache(cache_path, preprocessor)

    print(f"No cache found. Scanning {patches_dir} for .mtdrum files…")
    patch_files = []
    for drum_type in DRUM_TYPES:
        folder = os.path.join(patches_dir, drum_type)
        if os.path.isdir(folder):
            patch_files.extend(glob.glob(os.path.join(folder, "*.mtdrum")))
    print(f"Found {len(patch_files)} .mtdrum files.")

    return build_cache(patch_files, cache_path, preprocessor)

# ─────────────────────────────────────────────
# CVAE model
# ─────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, param_dim, type_dim, latent_dim, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(param_dim + type_dim, hidden_dim),
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


class Decoder(nn.Module):
    """
    Split output heads:
      cont_head → sigmoid  (outputs ∈ [0,1], matching MinMax-normalised targets)
      cat_head  → raw logits (fed directly to F.cross_entropy)
    """
    def __init__(self, latent_dim, type_dim, cont_dim, cat_dim, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.cont_dim = cont_dim
        self.cat_dim  = cat_dim
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim + type_dim, hidden_dim // 2),
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
        self.cont_head = nn.Linear(hidden_dim, cont_dim)
        self.cat_head  = nn.Linear(hidden_dim, cat_dim)

    def forward(self, z, c):
        h    = self.input_proj(torch.cat([z, c], dim=-1))
        h    = self.expand(h)
        h    = self.res_act(h + self.res_block1(h))
        h    = self.res_act(h + self.res_block2(h))
        cont = torch.sigmoid(self.cont_head(h))
        cats = self.cat_head(h)
        return torch.cat([cont, cats], dim=-1)


class CVAE(nn.Module):
    def __init__(self, param_dim, type_dim, cont_dim, cat_dim,
                 latent_dim=32, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.encoder    = Encoder(param_dim, type_dim, latent_dim, hidden_dim, dropout)
        self.decoder    = Decoder(latent_dim, type_dim, cont_dim, cat_dim, hidden_dim, dropout)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.dropout    = dropout

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x, c):
        mu, logvar = self.encoder(x, c)
        z          = self.reparameterize(mu, logvar)
        recon      = self.decoder(z, c)
        return recon, mu, logvar

# ─────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────

def cvae_loss(recon, x, mu, logvar, preprocessor: PatchPreprocessor,
              beta: float = 1.0, free_bits: float = 0.0):
    cont_dim = preprocessor.cont_dim
    weights  = _CONT_PARAM_WEIGHTS.to(recon.device)

    # Weighted MSE — perceptually important params contribute more
    diff = (recon[:, :cont_dim] - x[:, :cont_dim]).pow(2)
    mse  = (diff * weights).mean()

    ce_total = 0.0
    offset   = cont_dim
    for vals in CATEGORICAL_PARAMS.values():
        size      = len(vals)
        logits    = recon[:, offset: offset + size]
        targets   = x[:, offset: offset + size].argmax(dim=-1)
        ce_total += F.cross_entropy(logits, targets)
        offset   += size

    # Free-bits KL: only penalise KL above threshold per latent dimension.
    # This prevents posterior collapse by ensuring each dim stays active.
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim = kl_per_dim.mean(dim=0)           # average over batch → [latent_dim]
    if free_bits > 0:
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    kl = kl_per_dim.sum()

    return mse + ce_total + beta * kl, mse, ce_total, kl

# ─────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────

class CVAETrainer:
    def __init__(
        self,
        train_dataset:     PatchDataset,
        preprocessor:      PatchPreprocessor,
        latent_dim:        int   = 32,
        hidden_dim:        int   = 512,
        batch_size:        int   = 256,
        lr:                float = 3e-4,
        beta:              float = 0.02,
        kl_warmup_epochs:  int   = 200,
        kl_free_bits:      float = 0.25,
        kl_cyclical:       bool  = True,
        kl_cycle_epochs:   int   = 200,
        dropout:           float = 0.1,
        device:            torch.device = None,
    ):
        self.device           = device or DEVICE
        self.beta             = beta
        self.kl_warmup_epochs = kl_warmup_epochs
        self.kl_free_bits     = kl_free_bits
        self.kl_cyclical      = kl_cyclical
        self.kl_cycle_epochs  = kl_cycle_epochs
        self.preprocessor     = preprocessor
        self.use_amp          = USE_AMP
        self.scaler_amp       = GradScaler("cuda", enabled=USE_AMP)

        pin = self.device.type == "cuda"
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            drop_last=True, num_workers=NUM_WORKERS, pin_memory=pin,
            persistent_workers=(NUM_WORKERS > 0),
        )

        pp = preprocessor
        self.model = CVAE(
            param_dim  = pp.param_dim,
            type_dim   = pp.type_dim,
            cont_dim   = pp.cont_dim,
            cat_dim    = pp.cat_dim,
            latent_dim = latent_dim,
            hidden_dim = hidden_dim,
            dropout    = dropout,
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

        # Restore scheduler state after it's been created (must happen after self.load)
        if hasattr(self, '_pending_scheduler_state') and self._pending_scheduler_state is not None:
            self.scheduler.load_state_dict(self._pending_scheduler_state)
            self._pending_scheduler_state = None

        for epoch in range(start_epoch, epochs + 1):

            # KL annealing
            if self.kl_cyclical and self.kl_cycle_epochs > 0:
                # Cyclical: linear ramp in first half of each cycle, then hold
                cycle_pos = (epoch % self.kl_cycle_epochs) / self.kl_cycle_epochs
                annealed_beta = self.beta * min(1.0, cycle_pos * 2)
            elif self.kl_warmup_epochs > 0:
                annealed_beta = self.beta * min(1.0, epoch / self.kl_warmup_epochs)
            else:
                annealed_beta = self.beta

            # ── Training ──────────────────────────────────────────────────────
            self.model.train()
            total_loss = mse_sum = ce_sum = kl_sum = 0.0

            for x, c, _ in self.train_loader:
                # non_blocking=True overlaps H→D transfer with GPU compute
                x = x.to(self.device, non_blocking=True)
                c = c.to(self.device, non_blocking=True)

                with autocast("cuda", enabled=self.use_amp):
                    recon, mu, logvar = self.model(x, c)
                    loss, mse, ce, kl = cvae_loss(
                        recon, x, mu, logvar, self.preprocessor,
                        beta=annealed_beta, free_bits=self.kl_free_bits,
                    )

                self.optimizer.zero_grad(set_to_none=True)  # faster than zero_grad()
                self.scaler_amp.scale(loss).backward()
                self.scaler_amp.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler_amp.step(self.optimizer)
                self.scaler_amp.update()

                total_loss += loss.item()
                mse_sum    += mse.item()
                ce_sum     += ce.item()
                kl_sum     += kl.item()

            n = len(self.train_loader)
            train_recon = (mse_sum + ce_sum) / n
            h = {"epoch": epoch, "beta": annealed_beta,
                 "loss": total_loss / n, "mse": mse_sum / n,
                 "ce":   ce_sum / n,    "kl":  kl_sum / n}
            history.append(h)

            # Only track best / step scheduler AFTER warmup completes,
            # because during warmup β≈0 gives artificially low recon.
            warmup_done = (self.kl_warmup_epochs <= 0 or epoch >= self.kl_warmup_epochs)

            if warmup_done:
                self.scheduler.step(train_recon)

                improved = train_recon < (self.best_recon - early_stopping_min_delta)
                if improved:
                    self.best_recon = train_recon
                    self.best_epoch = epoch
                    self.epochs_without_improvement = 0
                    self.save(checkpoint_path, epoch)
                    print(f"  ✓ Checkpoint saved (epoch {epoch}, recon {self.best_recon:.6f})")
                else:
                    self.epochs_without_improvement += 1

            if epoch % log_every == 0:
                lr_now = self.optimizer.param_groups[0]['lr']
                warmup_tag = "" if warmup_done else " [WARMUP]"
                print(
                    f"Epoch {epoch:>4}/{epochs} | β {annealed_beta:.3f} | lr {lr_now:.2e} | "
                    f"Loss {h['loss']:.4f} | Recon {train_recon:.4f} | "
                    f"MSE {h['mse']:.4f} | CE {h['ce']:.4f} | KL {h['kl']:.4f} | "
                    f"Best recon {self.best_recon:.6f} @ {self.best_epoch} | "
                    f"Wait {self.epochs_without_improvement}{warmup_tag}"
                )

            if target_loss is not None and train_recon <= target_loss:
                print(f"Target recon {target_loss} reached at epoch {epoch}. Stopping.")
                break

            if (
                early_stopping_patience is not None
                and self.epochs_without_improvement >= early_stopping_patience
            ):
                print(
                    f"Early stopping at epoch {epoch}: no recon improvement larger than "
                    f"{early_stopping_min_delta:.1e} for {self.epochs_without_improvement} epochs. "
                    f"Best epoch {self.best_epoch} with recon {self.best_recon:.6f}."
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
                "param_dim":  self.preprocessor.param_dim,
                "type_dim":   self.preprocessor.type_dim,
                "cont_dim":   self.preprocessor.cont_dim,
                "cat_dim":    self.preprocessor.cat_dim,
                "latent_dim": self.model.latent_dim,
                "hidden_dim": self.model.hidden_dim,
                "dropout":    self.model.dropout,
            },
            "training_config": {
                "beta": self.beta,
                "kl_warmup_epochs": self.kl_warmup_epochs,
                "kl_free_bits": self.kl_free_bits,
                "kl_cyclical": self.kl_cyclical,
                "kl_cycle_epochs": self.kl_cycle_epochs,
            },
            "scaler": self.preprocessor.scaler_state_dict(),
        }, path)

    def load(self, path: str) -> int:
        """Load checkpoint. Returns the epoch number stored in the checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        cfg  = ckpt["model_config"]
        self.model = CVAE(**cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.preprocessor.load_scaler_state(ckpt["scaler"])
        # Reinitialize optimizer and scheduler, then restore state if available
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.optimizer.param_groups[0]["lr"])
        self.scaler_amp = GradScaler("cuda", enabled=self.use_amp)
        if "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scaler_amp_state" in ckpt:
            self.scaler_amp.load_state_dict(ckpt["scaler_amp_state"])
        # scheduler state is restored after scheduler is created in train()
        self._pending_scheduler_state = ckpt.get("scheduler_state")
        self.best_recon = ckpt.get("best_recon", float("inf"))
        self.best_epoch = ckpt.get("best_epoch", ckpt.get("epoch", 0))
        self.epochs_without_improvement = ckpt.get("epochs_without_improvement", 0)
        epoch = ckpt.get("epoch", 0)
        print(f"Loaded from {path} (epoch {epoch})")
        return epoch

# ─────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────

class PatchGenerator:
    def __init__(self, trainer: CVAETrainer):
        self.model        = trainer.model
        self.preprocessor = trainer.preprocessor
        self.device       = trainer.device
        self.use_amp      = trainer.use_amp
        self.beta         = trainer.beta
        self.model.eval()
        train_dataset = trainer.train_loader.dataset
        self.latent_bank, self.latent_jitter = build_empirical_latent_bank(
            self.model, train_dataset, self.device
        )

    @property
    def sampling_summary(self) -> str:
        if self.beta > 0.0:
            return "prior"
        return "data-guided" if self.latent_bank else "prior"

    def _resolve_sampling_mode(self, sampling_mode: str, drum_type: str) -> str:
        if sampling_mode not in {"auto", "prior", "empirical"}:
            raise ValueError(
                f"Unknown sampling_mode '{sampling_mode}'. Use 'auto', 'prior', or 'empirical'."
            )
        if sampling_mode == "auto":
            if self.beta > 0.0:
                return "prior"
            return "empirical" if drum_type in self.latent_bank else "prior"
        if sampling_mode == "empirical" and drum_type not in self.latent_bank:
            raise RuntimeError(
                f"No empirical latent bank available for drum type '{drum_type}'."
            )
        return sampling_mode

    def _sample_prior_latents(self, n: int, temperature: float):
        scale = max(float(temperature), 0.0)
        return torch.randn(n, self.model.latent_dim, device=self.device) * scale

    def _sample_empirical_latents(self, drum_type: str, n: int, temperature: float):
        bank = self.latent_bank[drum_type]
        count = bank.shape[0]
        primary = bank[torch.randint(count, (n,), device=self.device)]
        if count == 1:
            return primary.clone()

        temp = max(float(temperature), 0.0)
        secondary = bank[torch.randint(count, (n,), device=self.device)]
        blend_cap = min(0.5, temp * 0.35)
        blend = torch.rand(n, 1, device=self.device) * blend_cap
        jitter = self.latent_jitter[drum_type].unsqueeze(0)
        noise = torch.randn(n, self.model.latent_dim, device=self.device) * jitter * min(temp, 1.5)
        return primary + (secondary - primary) * blend + noise

    def generate(self, drum_type: str, n: int = 1,
                 temperature: float = 1.0, name_prefix: str = None,
                 sampling_mode: str = "auto") -> list:
        """
        Generate n patches for the given drum type.

        Args:
            drum_type:    One of the DRUM_TYPES labels, e.g. "bd", "sd", "oh"
            n:            Number of patches to generate
            temperature:  In prior mode this scales Gaussian sampling. In
                          data-guided mode it scales interpolation and jitter.
            name_prefix:  Optional prefix for the patch Name field
            sampling_mode: 'auto', 'prior', or 'empirical'
        """
        c = (torch.tensor(self.preprocessor.encode_type(drum_type), dtype=torch.float32)
               .unsqueeze(0).repeat(n, 1).to(self.device))
        resolved_mode = self._resolve_sampling_mode(sampling_mode, drum_type)

        with torch.no_grad(), autocast("cuda", enabled=self.use_amp):
            if resolved_mode == "empirical":
                z = self._sample_empirical_latents(drum_type, n, temperature)
            else:
                z = self._sample_prior_latents(n, temperature)
            recon = self.model.decoder(z, c).float().cpu().numpy()

        prefix  = name_prefix or f"Gen {drum_type.upper()}"
        patches = [
            self.preprocessor.decode_patch(
                vec, drum_type,
                name=f"{prefix} {i + 1}" if n > 1 else prefix
            )
            for i, vec in enumerate(recon)
        ]
        return patches

    def generate_batch(self, type_counts: dict, temperature: float = 1.0,
                       sampling_mode: str = "auto") -> dict:
        """e.g. type_counts = {"bd": 4, "sd": 4, "ch": 8}"""
        return {
            drum_type: self.generate(
                drum_type,
                n=count,
                temperature=temperature,
                sampling_mode=sampling_mode,
            )
            for drum_type, count in type_counts.items()
        }

# %% [markdown]
# ## Cell 4 — Build or load the dataset cache

# %%
train_ds, preprocessor = get_datasets(PATCHES_DIR, CACHE_PATH)
print(f"Dataset: {len(train_ds)} samples")

# %% [markdown]
# ## Cell 4b — Round-trip sanity check
#
# Verify that encode → decode preserves parameter values within tolerance.
# Run this after building the cache to confirm the log-transform pipeline works.

# %%
def round_trip_test(preprocessor, patches_dir, n_samples=20):
    """Encode a few real patches and decode them back; print max error per param."""
    from pythonic.preset_manager import DrumPatchParser
    import glob as _glob

    patch_files = []
    for dt in DRUM_TYPES:
        folder = os.path.join(patches_dir, dt)
        if os.path.isdir(folder):
            patch_files.extend(_glob.glob(os.path.join(folder, "*.mtdrum")))
    if not patch_files:
        print("No patch files found for round-trip test.")
        return

    parser = DrumPatchParser()
    rng = np.random.default_rng(42)
    indices = rng.choice(len(patch_files), size=min(n_samples, len(patch_files)), replace=False)

    max_err = {p: 0.0 for p in CONTINUOUS_PARAMS}
    for idx in indices:
        f = patch_files[idx]
        orig = parser.parse_file(f)
        orig["Name"] = f
        drum_type = infer_drum_type(f)

        vec     = preprocessor.encode_patch(orig)
        rebuilt = preprocessor.decode_patch(vec, drum_type, name="roundtrip")

        for p in CONTINUOUS_PARAMS:
            orig_v = orig.get(p, 0.0)
            if isinstance(orig_v, (tuple, list)):
                orig_v = orig_v[0]
            orig_v = float(orig_v)
            # apply same clamp as _extract_continuous
            if p in PARAM_CLAMP:
                lo, hi = PARAM_CLAMP[p]
                orig_v = max(lo, min(hi, orig_v))
            recon_v = rebuilt[p]
            err = abs(orig_v - recon_v) / max(abs(orig_v), 1e-6)
            max_err[p] = max(max_err[p], err)

    print("Round-trip max relative error per parameter:")
    all_ok = True
    for p, err in max_err.items():
        status = "✓" if err < 0.01 else "✗"
        if err >= 0.01:
            all_ok = False
        print(f"  {status} {p:12s}: {err:.6f}")
    print(f"\n{'All params OK (< 1% error)' if all_ok else 'WARNING: some params exceed 1% error!'}")

round_trip_test(preprocessor, PATCHES_DIR)

# %% [markdown]
# ## Cell 5 — Train

# %%
trainer = CVAETrainer(
    train_dataset    = train_ds,
    preprocessor     = preprocessor,
    latent_dim       = LATENT_DIM,
    hidden_dim       = HIDDEN_DIM,
    batch_size       = BATCH_SIZE,
    lr               = LR,
    beta             = BETA,
    kl_warmup_epochs = KL_WARMUP_EPOCHS,
    kl_free_bits     = KL_FREE_BITS,
    kl_cyclical      = KL_CYCLICAL,
    kl_cycle_epochs  = KL_CYCLE_EPOCHS,
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
# ## Cell 6 — Generate patches

# %%
generator = PatchGenerator(trainer)
print(f"Sampler: {generator.sampling_summary}")

bd_patch = generator.generate("bd", n=1, temperature=1.0)[0]
print("Generated BD patch:")
for k, v in bd_patch.items():
    print(f"  {k:12s} = {v}")

kit        = generator.generate_batch({"bd": 2, "sd": 2, "oh": 1}, temperature=0.8)
kit_counts = ", ".join(f"{k}: {len(v)}" for k, v in kit.items())
print(f"\nGenerated kit: {{ {kit_counts} }}")
