"""
Slim inference runtime for the drum CVAE model.

All torch imports are lazy so the base app launches without ML packages.
Only numpy is imported at module level (already a core dependency).
"""

import re
import numpy as np

# ─────────────────────────────────────────────
# Constants (mirrored from drum_patches/train.py)
# ─────────────────────────────────────────────

CONTINUOUS_PARAMS = [
    "OscFreq", "OscDcy", "ModAmt", "ModRate",
    "NFilFrq", "NFilQ", "NEnvAtk", "NEnvDcy",
    "Mix", "DistAmt", "EQFreq", "EQGain",
    "Level", "OscVel", "NVel", "ModVel",
]

LOG_PARAMS = {"OscFreq", "OscDcy", "ModRate", "NFilFrq", "NFilQ", "NEnvAtk", "NEnvDcy", "EQFreq"}
_LOG_PARAM_INDICES = [i for i, p in enumerate(CONTINUOUS_PARAMS) if p in LOG_PARAMS]

PARAM_CLAMP = {
    "OscFreq": (20.0, 20_000.0),
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

DEFAULT_HIDDEN_DIM = 256

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

    def __init__(self):
        self.scaler = _MinMaxScaler()
        self.cat_sizes = {k: len(v) for k, v in CATEGORICAL_PARAMS.items()}
        self.type_to_idx = {t: i for i, t in enumerate(DRUM_TYPES)}
        self.cont_dim = len(CONTINUOUS_PARAMS)
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

        for idx in _LOG_PARAM_INDICES:
            cont[idx] = np.exp(cont[idx])

        patch = {param: float(cont[i]) for i, param in enumerate(CONTINUOUS_PARAMS)}

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
            self.res_block = nn.Sequential(
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
            h = self.res_act(h + self.res_block(h))
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
            self.res_block = nn.Sequential(
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
            h = self.res_act(h + self.res_block(h))
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


def install_ml_dependencies(on_output=None) -> bool:
    """Run pip to install requirements-ml.txt in the current environment.

    Args:
        on_output: Optional callback(line: str) for streaming progress.

    Returns True on success, False on failure.
    """
    import subprocess
    import sys
    import os

    req_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "requirements-ml.txt")
    if not os.path.isfile(req_file):
        if on_output:
            on_output(f"requirements-ml.txt not found at {req_file}")
        return False

    cmd = [sys.executable, "-m", "pip", "install", "-r", req_file]
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        for line in proc.stdout:
            if on_output:
                on_output(line.rstrip())
        proc.wait()
        return proc.returncode == 0
    except Exception as exc:
        if on_output:
            on_output(f"Install failed: {exc}")
        return False


class PatchGenerator:
    """
    High-level service that loads a CVAE checkpoint and generates drum patches.

    All torch usage is contained within this class; nothing is imported at
    module level so the rest of the app is unaffected when torch is absent.
    """

    def __init__(self):
        self._model = None
        self._preprocessor = PatchPreprocessor()
        self._device = None
        self._loaded_path = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def loaded_path(self) -> str | None:
        return self._loaded_path

    def load_model(self, path: str):
        """Load a CVAE checkpoint from disk. Raises on failure."""
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

        self._preprocessor.load_scaler_state(ckpt["scaler"])
        self._model = model
        self._loaded_path = path

    def generate(self, drum_type: str, n: int = 1,
                 temperature: float = 1.0, seed: int | None = None) -> list[dict]:
        """
        Generate *n* raw patch dicts for *drum_type*.

        Args:
            drum_type:   One of DRUM_TYPES (e.g. "bd", "sd", "oh").
            n:           Number of candidates.
            temperature: Sampling temperature (1.0 = normal).
            seed:        Optional RNG seed for reproducibility.

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

        with torch.no_grad():
            z = torch.randn(n, self._model.latent_dim, device=self._device, generator=gen) * temperature
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
                          type_override: str | None = None) -> list[dict]:
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
        return self.generate(drum_type, n=n, temperature=temperature, seed=seed)
