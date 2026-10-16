"""
Slim inference runtime for the drum CVAE model.

All torch imports are lazy so the base app launches without ML packages.
Only numpy is imported at module level (already a core dependency).
"""

import re
import numpy as np


DEFAULT_DATASET_CACHE = "drum_dataset_cache.pt"
EMPIRICAL_BLEND_SCALE = 0.35
EMPIRICAL_JITTER_SCALE = 0.15
# The data-guided latent bank is saved next to the checkpoint once built
# (<checkpoint>.latents.pt): building it runs the encoder over the whole
# dataset cache, about a minute on a CPU
LATENT_BANK_SUFFIX = ".latents.pt"
LATENT_BANK_VERSION = 1

# ─────────────────────────────────────────────
# Constants (mirrored from drum_patches/train.py)
# ─────────────────────────────────────────────

CONTINUOUS_PARAMS = [
    "OscFreq", "OscAtk", "OscDcy", "ModAmt", "ModRate",
    "NFilFrq", "NFilQ", "NEnvAtk", "NEnvDcy",
    "Mix", "DistAmt", "EQFreq", "EQGain",
    "Level", "OscVel", "NVel", "ModVel",
]

# Parameter list of checkpoints saved before OscAtk was added. Those
# checkpoints carry no "continuous_params" key.
LEGACY_CONTINUOUS_PARAMS = [p for p in CONTINUOUS_PARAMS if p != "OscAtk"]

# Values for parameters a checkpoint does not model.
MISSING_PARAM_DEFAULTS = {"OscAtk": 0.0}

LOG_PARAMS = {"OscFreq", "OscAtk", "OscDcy", "ModRate", "NFilFrq", "NFilQ", "NEnvAtk", "NEnvDcy", "EQFreq"}

PARAM_CLAMP = {
    "OscFreq": (20.0, 20_000.0),
    "OscAtk":  (0.001, 10_000.0),
    "OscDcy":  (1.0, 10_000_000.0),
    "ModAmt":  (-96.0, 96.0),
    "ModRate": (0.001, 100_000.0),
    "NFilFrq": (20.0, 20_000.0),
    "NFilQ":   (0.5, 100.0),
    "NEnvAtk": (0.001, 100_000.0),
    "NEnvDcy": (1.0, 10_000_000.0),
    "Mix":     (0.0, 100.0),
    "DistAmt": (0.0, 100.0),
    "EQFreq":  (20.0, 20_000.0),
    "EQGain":  (-40.0, 40.0),
    "Level":   (-50.0, 20.0),
    "OscVel":  (0.0, 200.0),
    "NVel":    (0.0, 200.0),
    "ModVel":  (0.0, 200.0),
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

# TR-8-inspired 8-lane slot map: slot label → allowed drum types
SLOT_MAP = {
    0: ("BD",           ["bd"]),
    1: ("SD",           ["sd"]),
    2: ("CH",           ["ch", "shaker"]),
    3: ("OH",           ["oh"]),
    4: ("TOM HI/PERC",  ["tom", "perc"]),
    5: ("TOM LO/PERC",  ["tom", "perc", "cowbell"]),
    6: ("CLAP/RIM",     ["clap", "blip", "zap", "cowbell"]),
    7: ("CY/FX",        ["cy", "fx", "fuzz", "reverse", "synth", "bass", "other"]),
}

DEFAULT_HIDDEN_DIM = 1024

# ─────────────────────────────────────────────
# Keyword-based drum type inference (like TR-8 labels)
# ─────────────────────────────────────────────

_KEYWORD_PATTERNS = [
    ('bd',   re.compile(r'\bBD\b|KICK|BASS\s*DR', re.IGNORECASE)),
    ('sd',   re.compile(r'\bSD\b|SNARE|SNR', re.IGNORECASE)),
    ('ch',   re.compile(r'\bCH\b|CLOSED\s*H|CL\s*HAT|SHAKER', re.IGNORECASE)),
    ('oh',   re.compile(r'\bOH\b|OPEN\s*H|OP\s*HAT', re.IGNORECASE)),
    ('tom',  re.compile(r'\bTOM\b', re.IGNORECASE)),
    ('clap', re.compile(r'CLAP|CLP|\bRIM\b', re.IGNORECASE)),
    ('cy',   re.compile(r'\bCY\b|CYMBAL|CRASH|RIDE', re.IGNORECASE)),
    ('perc', re.compile(r'PERC|CONGA|BONGO|COWBELL', re.IGNORECASE)),
    ('fx',   re.compile(r'\bFX\b|\bZAP\b|REVERSE|FUZZ|NOISE|SYNTH|BLIP', re.IGNORECASE)),
]

_DISPLAY_LABELS = {
    'bd': 'BD', 'bass': 'BD',
    'sd': 'SD',
    'ch': 'CH', 'shaker': 'CH',
    'oh': 'OH',
    'tom': 'TOM',
    'clap': 'CLAP', 'blip': 'CLAP',
    'cy': 'CY',
    'perc': 'PERC', 'cowbell': 'PERC',
    'fx': 'FX', 'fuzz': 'FX', 'reverse': 'FX', 'synth': 'FX', 'zap': 'FX',
}


def infer_drum_type(name: str) -> str:
    """Infer a short drum type label from a patch name using keyword matching.

    Returns 'BD', 'SD', 'CH', 'OH', 'TOM', 'CLAP', 'CY', 'PERC', 'FX',
    or '' if no match.
    """
    if not name:
        return ''
    for dtype, pattern in _KEYWORD_PATTERNS:
        if pattern.search(name):
            return _DISPLAY_LABELS.get(dtype, dtype.upper())
    return ''


# ─────────────────────────────────────────────
# Lightweight scaler (no sklearn needed)
# ─────────────────────────────────────────────

class _MinMaxScaler:
    """Minimal MinMaxScaler replacement restored from saved arrays."""

    def __init__(self):
        self.scale_ = None
        self.min_ = None
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None

    def load_state(self, d: dict):
        self.scale_ = np.array(d["scale_"], dtype=np.float64)
        self.min_ = np.array(d["min_"], dtype=np.float64)
        self.data_min_ = np.array(d["data_min_"], dtype=np.float64)
        self.data_max_ = np.array(d["data_max_"], dtype=np.float64)
        self.data_range_ = self.data_max_ - self.data_min_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.min_) / self.scale_


# ─────────────────────────────────────────────
# Preprocessor (decode-only for inference)
# ─────────────────────────────────────────────

class PatchPreprocessor:
    """Lightweight preprocessor for inference: decode model output → patch dict."""

    def __init__(self, continuous_params=None):
        self.continuous_params = list(continuous_params or CONTINUOUS_PARAMS)
        self._log_indices = [i for i, p in enumerate(self.continuous_params) if p in LOG_PARAMS]
        self.scaler = _MinMaxScaler()
        self.cat_sizes = {k: len(v) for k, v in CATEGORICAL_PARAMS.items()}
        self.type_to_idx = {t: i for i, t in enumerate(DRUM_TYPES)}
        self.cont_dim = len(self.continuous_params)
        self.cat_dim = sum(self.cat_sizes.values())
        self.param_dim = self.cont_dim + self.cat_dim
        self.type_dim = len(DRUM_TYPES)

    def load_scaler_state(self, d: dict):
        self.scaler.load_state(d)

    def encode_type(self, drum_type: str) -> np.ndarray:
        idx = self.type_to_idx.get(drum_type, self.type_to_idx["other"])
        oh = np.zeros(self.type_dim, dtype=np.float32)
        oh[idx] = 1.0
        return oh

    def decode_patch(self, vector: np.ndarray, drum_type: str, name: str = None) -> dict:
        """Convert model output vector back to a raw patch dict (OscFreq, OscWave, etc.)."""
        cont_norm = vector[:self.cont_dim]
        cont = self.scaler.inverse_transform(cont_norm.reshape(1, -1))[0]
        cont = np.clip(cont, self.scaler.data_min_, self.scaler.data_max_)

        for idx in self._log_indices:
            cont[idx] = np.exp(cont[idx])

        patch = {param: float(cont[i]) for i, param in enumerate(self.continuous_params)}
        for param, default in MISSING_PARAM_DEFAULTS.items():
            patch.setdefault(param, default)

        offset = self.cont_dim
        for param, vals in CATEGORICAL_PARAMS.items():
            size = len(vals)
            idx = int(np.argmax(vector[offset:offset + size]))
            patch[param] = vals[idx]
            offset += size

        patch.update({
            "Pan": 0.0,
            "NStereo": "Off",
            "Output": "A",
            "Name": name or drum_type.upper(),
        })
        return patch


# ─────────────────────────────────────────────
# CVAE model (decoder-only path kept; encoder
# included for checkpoint compat)
# ─────────────────────────────────────────────

def _build_model(param_dim, type_dim, cont_dim, cat_dim, latent_dim, hidden_dim, dropout=0.1):
    """Build CVAE using lazy torch import. Returns (model, torch_module)."""
    import torch
    import torch.nn as nn

    class Encoder(nn.Module):
        def __init__(self):
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
            self.mu_head = nn.Linear(hidden_dim // 2, latent_dim)
            self.logvar_head = nn.Linear(hidden_dim // 2, latent_dim)

        def forward(self, x, c):
            h = self.input_proj(torch.cat([x, c], dim=-1))
            h = self.res_act(h + self.res_block1(h))
            h = self.res_act(h + self.res_block2(h))
            h = self.compress(h)
            return self.mu_head(h), self.logvar_head(h)

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.cont_dim = cont_dim
            self.cat_dim = cat_dim
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
            self.cat_head = nn.Linear(hidden_dim, cat_dim)

        def forward(self, z, c):
            h = self.input_proj(torch.cat([z, c], dim=-1))
            h = self.expand(h)
            h = self.res_act(h + self.res_block1(h))
            h = self.res_act(h + self.res_block2(h))
            cont = torch.sigmoid(self.cont_head(h))
            cats = self.cat_head(h)
            return torch.cat([cont, cats], dim=-1)

    class CVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()
            self.decoder = Decoder()
            self.latent_dim = latent_dim

        def forward(self, x, c):
            mu, logvar = self.encoder(x, c)
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
            recon = self.decoder(z, c)
            return recon, mu, logvar

    return CVAE()


# ─────────────────────────────────────────────
# Generator service
# ─────────────────────────────────────────────

def is_torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class PatchGenerator:
    """
    High-level service that loads a CVAE checkpoint and generates drum patches.

    All torch usage is contained within this class; nothing is imported at
    module level so the rest of the app is unaffected when torch is absent.
    """

    _EMPIRICAL_BANK_CACHE = {}

    def __init__(self):
        self._model = None
        self._preprocessor = PatchPreprocessor()
        self._device = None
        self._loaded_path = None
        self._training_config = {}
        self._latent_bank = {}
        self._latent_jitter = {}
        self._latent_source = None  # the dataset cache the bank is built from, until it is

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def loaded_path(self) -> str | None:
        return self._loaded_path

    @property
    def has_empirical_bank(self) -> bool:
        self._ensure_latent_bank()
        return bool(self._latent_bank)

    @property
    def sampling_summary(self) -> str:
        return "data-guided" if self._auto_sampling_mode() == "empirical" else "prior"

    def _reset_sampling_support(self):
        self._training_config = {}
        self._latent_bank = {}
        self._latent_jitter = {}
        self._latent_source = None

    def _ensure_latent_bank(self):
        """Build (or read back) the data-guided latent bank on first need."""
        if self._latent_source is not None:
            source, self._latent_source = self._latent_source, None
            self._build_empirical_latent_bank(source)

    def _auto_sampling_mode(self) -> str:
        beta = float(self._training_config.get("beta", 0.0) or 0.0)
        if beta > 0.0:
            return "prior"
        return "empirical" if self.has_empirical_bank else "prior"

    def _resolve_sampling_mode(self, sampling_mode: str, drum_type: str) -> str:
        if sampling_mode not in {"auto", "prior", "empirical"}:
            raise ValueError(
                f"Unknown sampling_mode '{sampling_mode}'. Use 'auto', 'prior', or 'empirical'."
            )
        if sampling_mode == "auto":
            # A model trained with a KL term samples its prior: no bank needed
            if self._auto_sampling_mode() == "empirical" and drum_type in self._latent_bank:
                return "empirical"
            return "prior"
        self._ensure_latent_bank()
        if sampling_mode == "empirical" and drum_type not in self._latent_bank:
            raise RuntimeError(
                f"No empirical latent bank available for drum type '{drum_type}'."
            )
        return sampling_mode

    def _build_empirical_latent_bank(self, cache_path: str):
        import os
        import torch

        self._latent_bank = {}
        self._latent_jitter = {}
        if not os.path.isfile(cache_path):
            return

        cache_key = None
        if self._loaded_path and os.path.isfile(self._loaded_path):
            cache_key = (
                self._loaded_path,
                os.path.getmtime(self._loaded_path),
                cache_path,
                os.path.getmtime(cache_path),
            )
            cached = self._EMPIRICAL_BANK_CACHE.get(cache_key)
            if cached is not None:
                self._latent_bank, self._latent_jitter = cached
                return

        saved_key = self._latent_bank_key(cache_path)
        if self._load_saved_latent_bank(saved_key):
            if cache_key is not None:
                self._EMPIRICAL_BANK_CACHE[cache_key] = (self._latent_bank, self._latent_jitter)
            return

        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        train_x = cache.get("train_x")
        train_c = cache.get("train_c")
        train_types = cache.get("train_types")

        if train_x is None or train_c is None or train_types is None:
            return
        if train_x.ndim != 2 or train_c.ndim != 2:
            return
        if train_x.shape[1] != self._preprocessor.param_dim:
            return
        if train_c.shape[1] != self._preprocessor.type_dim:
            return
        if train_x.shape[0] != train_c.shape[0] or train_x.shape[0] != len(train_types):
            return

        mus_by_type = {drum_type: [] for drum_type in DRUM_TYPES}
        batch_size = 1024

        self._model.eval()
        with torch.no_grad():
            for start in range(0, train_x.shape[0], batch_size):
                end = min(start + batch_size, train_x.shape[0])
                x_batch = train_x[start:end].to(self._device)
                c_batch = train_c[start:end].to(self._device)
                mu_batch, _ = self._model.encoder(x_batch, c_batch)
                mu_batch = mu_batch.cpu()
                type_batch = train_types[start:end]
                for row_idx, drum_type in enumerate(type_batch):
                    if drum_type in mus_by_type:
                        mus_by_type[drum_type].append(mu_batch[row_idx])

        for drum_type, rows in mus_by_type.items():
            if not rows:
                continue
            mus = torch.stack(rows).to(self._device)
            median = mus.median(dim=0).values
            mad = (mus - median).abs().median(dim=0).values
            jitter = torch.clamp(mad * EMPIRICAL_JITTER_SCALE, min=0.01, max=0.5)
            self._latent_bank[drum_type] = mus
            self._latent_jitter[drum_type] = jitter

        if cache_key is not None and self._latent_bank:
            self._EMPIRICAL_BANK_CACHE[cache_key] = (
                self._latent_bank,
                self._latent_jitter,
            )
        if self._latent_bank:
            self._save_latent_bank(saved_key)

    # ── the latent bank on disk ───────────────────────────────────────────────

    def _latent_bank_path(self) -> str | None:
        if not self._loaded_path:
            return None
        import os
        return os.path.splitext(self._loaded_path)[0] + LATENT_BANK_SUFFIX

    def _latent_bank_key(self, cache_path: str) -> dict | None:
        """What a saved latent bank must have been built from: the format,
        the jitter scale, and the size and modification time of the
        checkpoint and of the dataset cache."""
        import os
        try:
            model, cache = os.stat(self._loaded_path), os.stat(cache_path)
        except (OSError, TypeError):
            return None
        return {"version": LATENT_BANK_VERSION, "jitter_scale": EMPIRICAL_JITTER_SCALE,
                "model": [model.st_size, model.st_mtime_ns], "cache": [cache.st_size, cache.st_mtime_ns]}

    def _load_saved_latent_bank(self, key: dict | None) -> bool:
        """Use the saved latent bank when it was built from the same files."""
        import os
        import torch
        path = self._latent_bank_path()
        if key is None or path is None or not os.path.isfile(path):
            return False
        try:
            saved = torch.load(path, map_location=self._device, weights_only=False)
        except Exception:
            return False
        if not isinstance(saved, dict) or saved.get("key") != key or not saved.get("bank"):
            return False
        self._latent_bank, self._latent_jitter = saved["bank"], saved["jitter"]
        return True

    def _save_latent_bank(self, key: dict | None):
        """Save the latent bank next to the checkpoint; a folder that cannot
        be written (an installed app) only means the next load builds it
        again."""
        import os
        import torch
        path = self._latent_bank_path()
        if key is None or path is None:
            return
        temp = f"{path}.{os.getpid()}.tmp"
        try:
            torch.save({"key": key, "bank": self._latent_bank, "jitter": self._latent_jitter}, temp)
            os.replace(temp, path)
        except OSError:
            try:
                os.remove(temp)
            except OSError:
                pass

    def _sample_prior_latents(self, n: int, temperature: float, gen):
        import torch

        scale = max(float(temperature), 0.0)
        return torch.randn(
            n, self._model.latent_dim, device=self._device, generator=gen
        ) * scale

    def _sample_empirical_latents(self, drum_type: str, n: int, temperature: float, gen):
        import torch

        bank = self._latent_bank[drum_type]
        count = bank.shape[0]
        primary_idx = torch.randint(count, (n,), device=self._device, generator=gen)
        primary = bank[primary_idx]

        if count == 1:
            return primary.clone()

        temp = max(float(temperature), 0.0)
        secondary_idx = torch.randint(count, (n,), device=self._device, generator=gen)
        secondary = bank[secondary_idx]
        blend_cap = min(0.5, temp * EMPIRICAL_BLEND_SCALE)
        blend = torch.rand(n, 1, device=self._device, generator=gen) * blend_cap
        jitter = self._latent_jitter[drum_type].unsqueeze(0)
        noise = torch.randn(
            n, self._model.latent_dim, device=self._device, generator=gen
        ) * jitter * min(temp, 1.5)
        return primary + (secondary - primary) * blend + noise

    def load_model(self, path: str):
        """Load a CVAE checkpoint from disk. Raises on failure."""
        import os
        import torch

        self._device = torch.device("cpu")
        ckpt = torch.load(path, map_location=self._device, weights_only=False)

        cfg = ckpt["model_config"]
        hidden_dim = cfg.get("hidden_dim", DEFAULT_HIDDEN_DIM)
        dropout = cfg.get("dropout", 0.1)

        model = _build_model(
            param_dim=cfg["param_dim"],
            type_dim=cfg["type_dim"],
            cont_dim=cfg["cont_dim"],
            cat_dim=cfg["cat_dim"],
            latent_dim=cfg["latent_dim"],
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        continuous_params = ckpt.get("continuous_params", LEGACY_CONTINUOUS_PARAMS)
        if len(continuous_params) != cfg["cont_dim"]:
            raise ValueError(
                f"Checkpoint has {cfg['cont_dim']} continuous outputs but lists "
                f"{len(continuous_params)} parameter names."
            )
        self._preprocessor = PatchPreprocessor(continuous_params)
        self._preprocessor.load_scaler_state(ckpt["scaler"])
        self._reset_sampling_support()
        self._training_config = ckpt.get("training_config", {})
        self._model = model
        self._loaded_path = path
        # The data-guided bank is built when first needed: a model trained
        # with a KL term samples its prior in auto mode and never needs it
        self._latent_source = os.path.join(os.path.dirname(path), DEFAULT_DATASET_CACHE)
        if float(self._training_config.get("beta", 0.0) or 0.0) <= 0.0:
            self._ensure_latent_bank()

    def generate(self, drum_type: str, n: int = 1,
                 temperature: float = 1.0, seed: int | None = None,
                 sampling_mode: str = "auto") -> list[dict]:
        """
        Generate *n* raw patch dicts for *drum_type*.

        Args:
            drum_type:   One of DRUM_TYPES (e.g. "bd", "sd", "oh").
            n:           Number of candidates.
            temperature: Sampling temperature. In data-guided mode this controls
                         interpolation and jitter around real encoded patches.
            seed:        Optional RNG seed for reproducibility.
            sampling_mode: 'auto', 'prior', or 'empirical'.

        Returns:
            List of patch dicts with keys like OscFreq, OscWave, etc.
        """
        if not self.is_loaded:
            raise RuntimeError("No model loaded. Call load_model() first.")
        import torch

        c = (torch.tensor(self._preprocessor.encode_type(drum_type), dtype=torch.float32)
             .unsqueeze(0).repeat(n, 1).to(self._device))

        gen = torch.Generator(device=self._device)
        if seed is not None:
            gen.manual_seed(seed)
        else:
            gen.seed()

        resolved_mode = self._resolve_sampling_mode(sampling_mode, drum_type)

        with torch.no_grad():
            if resolved_mode == "empirical":
                z = self._sample_empirical_latents(drum_type, n, temperature, gen)
            else:
                z = self._sample_prior_latents(n, temperature, gen)
            recon = self._model.decoder(z, c).cpu().numpy()

        return [
            self._preprocessor.decode_patch(
                vec, drum_type,
                name=drum_type.upper()
            )
            for i, vec in enumerate(recon)
        ]

    def generate_for_slot(self, slot_index: int, n: int = 1,
                          temperature: float = 1.0, seed: int | None = None,
                          type_override: str | None = None,
                          sampling_mode: str = "auto") -> list[dict]:
        """
        Generate candidates for a specific TR-8 slot.

        Uses the first allowed drum type for that slot unless *type_override*
        is provided (must be within the slot's allowed types).
        """
        label, allowed = SLOT_MAP[slot_index]
        drum_type = allowed[0]
        if type_override is not None:
            if type_override not in allowed:
                raise ValueError(
                    f"Type '{type_override}' not allowed for slot {slot_index} ({label}). "
                    f"Allowed: {allowed}"
                )
            drum_type = type_override
        return self.generate(
            drum_type,
            n=n,
            temperature=temperature,
            seed=seed,
            sampling_mode=sampling_mode,
        )
