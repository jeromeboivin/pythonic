"""
Slim inference runtime for the drum pattern CVAE model.

All torch imports are lazy so the base app launches without ML packages.
Only numpy is imported at module level (already a core dependency).
"""

import numpy as np
import os

# ─────────────────────────────────────────────
# Constants (mirrored from train_patterns.py)
# ─────────────────────────────────────────────

NUM_CHANNELS  = 8
NUM_STEPS     = 16
NUM_FEATURES  = 3           # trigger, accent, fill
PATTERN_DIM   = NUM_CHANNELS * NUM_STEPS * NUM_FEATURES  # 384

# Kit fingerprint: 5 perceptual features × 8 channels = 40
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

# Step rate vocabulary
STEP_RATES = ["1/8", "1/8T", "1/16", "1/16T", "1/32"]
STEP_RATE_DIM = len(STEP_RATES)

# Global metadata: tempo(1) + swing(1) + fill_rate(1) + step_rate one-hot(5) = 8
GLOBAL_META_DIM = 3 + STEP_RATE_DIM

# Total condition dimension: 40 + 8 = 48
CONDITION_DIM = KIT_CONTINUOUS_DIM + GLOBAL_META_DIM

PATTERN_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']

# Default bundled checkpoint path (relative to project root)
_BUNDLED_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "drum_patterns", "pattern_cvae_best.pt",
)


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

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.scale_ + self.min_


# ─────────────────────────────────────────────
# Preprocessor (encode condition + decode pattern)
# ─────────────────────────────────────────────

class PatternPreprocessor:
    """Lightweight preprocessor for pattern inference."""

    def __init__(self):
        self.kit_scaler = _MinMaxScaler()
        self.tempo_min = 60.0
        self.tempo_max = 300.0

    def load_scaler_state(self, d: dict):
        self.kit_scaler.load_state(d)

    # ── Kit fingerprint from raw patch dicts ──────────────────────────────────

    def _extract_kit_continuous(self, raw_patches: list[dict]) -> np.ndarray:
        """Extract continuous features from 8 raw patch dicts → flat vector (40,)."""
        vals = []
        for ch_idx in range(NUM_CHANNELS):
            patch = raw_patches[ch_idx] if ch_idx < len(raw_patches) else {}
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

    @staticmethod
    def _encode_global_meta(tempo: float = 120.0, swing: float = 0.0,
                            fill_rate: float = 4.0,
                            step_rate: str = "1/16") -> np.ndarray:
        tempo_norm = (np.clip(tempo, 60.0, 300.0) - 60.0) / (300.0 - 60.0)
        swing_norm = np.clip(swing / 100.0, 0.0, 1.0)
        fill_rate_norm = np.clip(fill_rate / 8.0, 0.0, 1.0)

        sr_oh = np.zeros(STEP_RATE_DIM, dtype=np.float32)
        if step_rate in STEP_RATES:
            sr_oh[STEP_RATES.index(step_rate)] = 1.0
        else:
            sr_oh[STEP_RATES.index("1/16")] = 1.0
        return np.concatenate([[tempo_norm, swing_norm, fill_rate_norm], sr_oh]).astype(np.float32)

    # ── Condition encoding ────────────────────────────────────────────────────

    def encode_condition(self, raw_patches: list[dict],
                         tempo: float = 120.0, swing: float = 0.0,
                         fill_rate: float = 4.0,
                         step_rate: str = "1/16") -> np.ndarray:
        """Build the 48-dim condition vector from 8 raw patch dicts + timing."""
        kit_cont = self._extract_kit_continuous(raw_patches)
        kit_norm = self.kit_scaler.transform(kit_cont.reshape(1, -1))[0]
        meta = self._encode_global_meta(tempo, swing, fill_rate, step_rate)
        return np.concatenate([kit_norm, meta]).astype(np.float32)

    # ── Pattern decode ────────────────────────────────────────────────────────

    @staticmethod
    def decode_pattern(vec: np.ndarray, threshold: float = 0.5) -> dict:
        """Decode a probability vector back to pattern dict with #/- strings."""
        pattern = {"Length": NUM_STEPS, "Chained": False}
        for ch_idx in range(NUM_CHANNELS):
            triggers = []
            accents = []
            fills = []
            for step in range(NUM_STEPS):
                base = (ch_idx * NUM_STEPS + step) * NUM_FEATURES
                triggers.append("#" if vec[base + 0] >= threshold else "-")
                accents.append("#" if vec[base + 1] >= threshold else "-")
                fills.append("#" if vec[base + 2] >= threshold else "-")
            pattern[str(ch_idx + 1)] = {
                "Triggers": "".join(triggers),
                "Accents": "".join(accents),
                "Fills": "".join(fills),
            }
        return pattern


# ─────────────────────────────────────────────
# Pattern CVAE model (lazy torch import)
# ─────────────────────────────────────────────

def _build_pattern_model(pattern_dim, cond_dim, latent_dim, hidden_dim, dropout=0.0):
    """Build the PatternCVAE. Returns the model instance."""
    import torch
    import torch.nn as nn

    class PatternEncoder(nn.Module):
        def __init__(self):
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
            self.mu_head = nn.Linear(hidden_dim // 2, latent_dim)
            self.logvar_head = nn.Linear(hidden_dim // 2, latent_dim)

        def forward(self, x, c):
            h = self.input_proj(torch.cat([x, c], dim=-1))
            h = self.res_act(h + self.res_block1(h))
            h = self.res_act(h + self.res_block2(h))
            h = self.compress(h)
            return self.mu_head(h), self.logvar_head(h)

    class PatternDecoder(nn.Module):
        def __init__(self):
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
            return self.out_head(h)

    class PatternCVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = PatternEncoder()
            self.decoder = PatternDecoder()
            self.latent_dim = latent_dim

        def forward(self, x, c):
            mu, logvar = self.encoder(x, c)
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
            logits = self.decoder(z, c)
            return logits, mu, logvar

    return PatternCVAE()


# ─────────────────────────────────────────────
# Generator service
# ─────────────────────────────────────────────

class PatternGenerator:
    """
    High-level service that loads a pattern CVAE checkpoint and generates
    drum patterns conditioned on a kit fingerprint.

    All torch usage is contained within this class.
    """

    def __init__(self):
        self._model = None
        self._preprocessor = PatternPreprocessor()
        self._device = None
        self._loaded_path = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def loaded_path(self) -> str | None:
        return self._loaded_path

    @staticmethod
    def resolve_model_path(preferences_manager=None) -> str | None:
        """Return the best available pattern model path.

        Priority:
        1. Saved preference ``drum_generator_pattern_model_path`` (if file exists).
        2. Bundled checkpoint ``drum_patterns/pattern_cvae_best.pt``.
        3. ``None`` when neither is available.
        """
        if preferences_manager is not None:
            saved = preferences_manager.get('drum_generator_pattern_model_path', None)
            if saved and os.path.isfile(saved):
                return saved
        if os.path.isfile(_BUNDLED_CHECKPOINT):
            return _BUNDLED_CHECKPOINT
        return None

    def ensure_loaded(self, preferences_manager=None) -> bool:
        """Load the model if not already loaded, using preference + fallback.

        Returns True if the model is (now) loaded, False otherwise.
        """
        if self.is_loaded:
            return True
        path = self.resolve_model_path(preferences_manager)
        if path is None:
            return False
        try:
            self.load_model(path)
            return True
        except Exception as e:
            print(f"PatternGenerator: failed to load {path}: {e}", flush=True)
            return False

    def load_model(self, path: str):
        """Load a pattern CVAE checkpoint from disk. Raises on failure."""
        import torch

        self._device = torch.device("cpu")
        ckpt = torch.load(path, map_location=self._device, weights_only=False)

        cfg = ckpt["model_config"]
        model = _build_pattern_model(
            pattern_dim=cfg["pattern_dim"],
            cond_dim=cfg["cond_dim"],
            latent_dim=cfg["latent_dim"],
            hidden_dim=cfg["hidden_dim"],
            dropout=cfg.get("dropout", 0.0),
        )
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        self._preprocessor.load_scaler_state(ckpt["scaler"])
        self._model = model
        self._loaded_path = path

    def generate(self, raw_patches: list[dict],
                 tempo: float = 120.0, swing: float = 0.0,
                 fill_rate: float = 4.0, step_rate: str = "1/16",
                 n: int = 1, temperature: float = 0.7,
                 threshold: float = 0.5,
                 seed: int | None = None) -> list[dict]:
        """
        Generate *n* pattern dicts conditioned on the current kit.

        Args:
            raw_patches: List of 8 raw patch dicts (OscFreq, OscWave, etc.)
            tempo:       BPM
            swing:       Swing amount (0-100)
            fill_rate:   Fill rate (0-8)
            step_rate:   Step rate string
            n:           Number of patterns to generate
            temperature: Latent sampling temperature
            threshold:   Sigmoid threshold for binarising output
            seed:        Optional RNG seed

        Returns:
            List of pattern dicts with keys "1"-"8", each containing
            Triggers/Accents/Fills strings.
        """
        if not self.is_loaded:
            raise RuntimeError("No model loaded. Call load_model() first.")
        import torch

        cond = self._preprocessor.encode_condition(
            raw_patches, tempo, swing, fill_rate, step_rate
        )
        c = (torch.tensor(cond, dtype=torch.float32)
             .unsqueeze(0).repeat(n, 1).to(self._device))

        gen = torch.Generator(device=self._device)
        if seed is not None:
            gen.manual_seed(seed)
        else:
            gen.seed()

        with torch.no_grad():
            z = torch.randn(n, self._model.latent_dim,
                            device=self._device, generator=gen) * temperature
            logits = self._model.decoder(z, c)
            probs = torch.sigmoid(logits).cpu().numpy()

        return [
            PatternPreprocessor.decode_pattern(probs[i], threshold=threshold)
            for i in range(n)
        ]

    def generate_bank(self, raw_patches: list[dict],
                      tempo: float = 120.0, swing: float = 0.0,
                      fill_rate: float = 4.0, step_rate: str = "1/16",
                      temperature: float = 0.7, threshold: float = 0.5,
                      seed: int | None = None) -> dict[str, dict]:
        """
        Generate a full 12-pattern bank (A-L) for the current kit.

        Returns:
            Dict mapping pattern name ('A'-'L') to pattern data dict.
        """
        patterns = self.generate(
            raw_patches, tempo, swing, fill_rate, step_rate,
            n=12, temperature=temperature, threshold=threshold, seed=seed,
        )
        return {name: pat for name, pat in zip(PATTERN_NAMES, patterns)}
