"""
Slim inference runtime for the drum pattern models.

Two kinds of checkpoint load:

- **bank** (``model_config["kind"] == "pattern_bank"``, train_pattern_bank.py):
  generates a whole bank of 12 patterns A-L at once, with their lengths and
  chain flags, conditioned on the kit and the timing. Its latents are drawn
  through the kit prior stored in the checkpoint; triggers play above 0.5,
  accents and fills are drawn from their probabilities. One pattern can be
  re-rolled in the style of an existing bank.
- **single** (train_patterns.py): one pattern per draw; a bank is 12
  independent draws, every pattern 16 steps, no chains.

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

# Bank model (mirrored from train_pattern_bank.py)
NUM_SLOTS   = len(PATTERN_NAMES)
NUM_CHAINS  = NUM_SLOTS - 1     # A→B … K→L
NUM_LENGTHS = NUM_STEPS         # length classes 1..16
BANK_KIND   = "pattern_bank"

# Default bundled checkpoint path (relative to project root)
_BUNDLED_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "drum_patterns", "pattern_bank_best.pt",
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
# Bank model (lazy torch import; the modules and their parameter names
# mirror BankCVAE and KitPrior of train_pattern_bank.py)
# ─────────────────────────────────────────────

def _build_bank_model(cfg: dict):
    """Build the bank CVAE of a checkpoint's model_config."""
    import torch
    import torch.nn as nn

    d_model, n_heads = cfg["d_model"], cfg["n_heads"]
    pattern_dim, cond_dim = cfg["pattern_dim"], cfg["cond_dim"]
    bank_latent, slot_latent = cfg["bank_latent"], cfg["slot_latent"]

    def mlp(n_in):
        return nn.Sequential(nn.Linear(n_in, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def stack(n_layers):
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, 0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        return nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)

    class BankCVAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.bank_latent, self.slot_latent = bank_latent, slot_latent
            self.enc_grid = mlp(pattern_dim)
            self.enc_length = nn.Embedding(NUM_LENGTHS, d_model)
            self.enc_chain = nn.Embedding(3, d_model)  # not chained, chained, none (slot L)
            self.enc_slot = nn.Parameter(torch.zeros(NUM_SLOTS, d_model))
            self.enc_cond = mlp(cond_dim)
            self.encoder = stack(cfg["enc_layers"])
            self.enc_norm = nn.LayerNorm(d_model)
            self.bank_head = nn.Linear(d_model, 2 * bank_latent)
            self.slot_head = nn.Linear(d_model, 2 * slot_latent)
            self.dec_cond = mlp(cond_dim)
            self.dec_bank = nn.Linear(bank_latent, d_model)
            self.dec_z = nn.Linear(slot_latent, d_model)
            self.dec_slot = nn.Parameter(torch.zeros(NUM_SLOTS, d_model))
            self.decoder = stack(cfg["dec_layers"])
            self.dec_norm = nn.LayerNorm(d_model)
            self.grid_head = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                           nn.Linear(2 * d_model, pattern_dim))
            self.length_head = nn.Linear(d_model, NUM_LENGTHS)
            self.chain_head = nn.Linear(d_model, 1)

        def encode(self, grid, length, chain, cond):
            """grid [B, 12, 384], length classes [B, 12], chain [B, 11],
            cond [B, 48] → the posterior means: bank [B, Zb], slots [B, 12, Zs]."""
            none = torch.full((grid.shape[0], 1), 2, dtype=torch.long)
            chain_in = torch.cat([chain.long(), none], dim=1)
            slots = (self.enc_grid(grid) + self.enc_length(length) + self.enc_chain(chain_in)
                     + self.enc_slot)
            h = self.enc_norm(self.encoder(torch.cat([self.enc_cond(cond).unsqueeze(1), slots], 1)))
            return self.bank_head(h[:, 0]).chunk(2, -1)[0], self.slot_head(h[:, 1:]).chunk(2, -1)[0]

        def decode(self, z_bank, z_slot, cond):
            bank = self.dec_bank(z_bank)
            slots = self.dec_slot + self.dec_z(z_slot) + bank.unsqueeze(1)
            tokens = torch.cat([self.dec_cond(cond).unsqueeze(1), bank.unsqueeze(1), slots], 1)
            h = self.dec_norm(self.decoder(tokens))[:, 2:]
            return self.grid_head(h), self.length_head(h), self.chain_head(h[:, :NUM_CHAINS]).squeeze(-1)

    return BankCVAE()


def _build_kit_prior(latent_dim: int, hidden: int, cond_dim: int = CONDITION_DIM):
    """The kit prior p(z | condition): an MLP mean and a Cholesky factor."""
    import torch
    import torch.nn as nn

    class KitPrior(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(cond_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden),
                                     nn.GELU(), nn.Linear(hidden, latent_dim))
            self.register_buffer("scale_tril", torch.eye(latent_dim))

    return KitPrior()


def encode_bank_patterns(patterns: list) -> tuple:
    """12 pattern dicts (Length, Chained, "1"-"8" with Triggers / Accents /
    Fills) → (grid float32 [12, 384], length classes int64 [12] (length - 1,
    at most 16 steps), chain float32 [11]). Steps past a length are zero."""
    if len(patterns) != NUM_SLOTS:
        raise ValueError(f"a bank has {NUM_SLOTS} patterns, not {len(patterns)}")
    grid = np.zeros((NUM_SLOTS, NUM_CHANNELS, NUM_STEPS, NUM_FEATURES), dtype=np.float32)
    length = np.zeros(NUM_SLOTS, dtype=np.int64)
    chain = np.zeros(NUM_CHAINS, dtype=np.float32)
    for s, pat in enumerate(patterns):
        n = max(1, min(NUM_STEPS, int(pat.get("Length", NUM_STEPS))))
        length[s] = n - 1
        if s < NUM_CHAINS:
            chain[s] = float(bool(pat.get("Chained", False)))
        for ch in range(NUM_CHANNELS):
            ch_data = pat.get(str(ch + 1), {})
            if not isinstance(ch_data, dict):
                continue
            for f, name in enumerate(("Triggers", "Accents", "Fills")):
                for step, char in enumerate(str(ch_data.get(name, ""))[:n]):
                    grid[s, ch, step, f] = char == "#"
    return grid.reshape(NUM_SLOTS, PATTERN_DIM), length, chain


def decode_bank_arrays(grid: np.ndarray, length: np.ndarray, chain: np.ndarray) -> list:
    """grid bool [12, 8, 16, 3], length [12] (1..16), chain [11] → 12 pattern
    dicts ("-" past each length, L never chained)."""
    bank = []
    for s in range(NUM_SLOTS):
        n = int(length[s])
        pat = {"Length": n, "Chained": bool(chain[s]) if s < NUM_CHAINS else False}
        for ch in range(NUM_CHANNELS):
            pat[str(ch + 1)] = {
                name: "".join("#" if grid[s, ch, step, f] and step < n else "-" for step in range(NUM_STEPS))
                for f, name in enumerate(("Triggers", "Accents", "Fills"))}
        bank.append(pat)
    return bank


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
        self._bank = None           # the bank model, when the checkpoint is one
        self._prior = None          # its kit prior (None: latents from N(0, I))
        self._preprocessor = PatternPreprocessor()
        self._device = None
        self._loaded_path = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None or self._bank is not None

    @property
    def is_bank(self) -> bool:
        """True when the loaded checkpoint generates whole banks."""
        return self._bank is not None

    @property
    def sampling_summary(self) -> str | None:
        """How the loaded model samples, for the model line ("bank" models)."""
        if self._bank is None:
            return None
        return "bank" if self._prior is not None else "bank, no kit prior"

    @property
    def loaded_path(self) -> str | None:
        return self._loaded_path

    @staticmethod
    def resolve_model_path(preferences_manager=None) -> str | None:
        """Return the best available pattern model path.

        Priority:
        1. Saved preference ``drum_generator_pattern_model_path`` (if file exists).
        2. Bundled checkpoint ``drum_patterns/pattern_bank_best.pt``.
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
        if cfg.get("kind") == BANK_KIND:
            bank = _build_bank_model(cfg)
            bank.load_state_dict(ckpt["model_state"])
            bank.eval()
            prior = None
            entry = ckpt.get("kit_prior")
            if entry:
                prior = _build_kit_prior(entry["latent_dim"], entry["hidden"], cfg["cond_dim"])
                prior.load_state_dict(entry["state"])
                prior.eval()
            self._preprocessor.load_scaler_state(ckpt["scaler"])
            self._model, self._bank, self._prior = None, bank, prior
            self._loaded_path = path
            return

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
        self._model, self._bank, self._prior = model, None, None
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
        if self._bank is not None:
            # Whole banks: n = 12 is one bank (with its lengths and chains)
            banks = -(-n // NUM_SLOTS)
            patterns = [p for bank in self._sample_banks(cond, banks, temperature, seed) for p in bank]
            return patterns[:n]
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

    def reroll(self, raw_patches: list[dict], bank: list[dict], slot: int,
               tempo: float = 120.0, swing: float = 0.0, fill_rate: float = 4.0,
               step_rate: str = "1/16", temperature: float = 1.0,
               seed: int | None = None) -> dict:
        """A new pattern for one slot (0-11) of an existing bank (12 pattern
        dicts), in the style of the bank: the bank model encodes the bank
        and draws only that slot's latent, given the others (the kit
        prior's Gaussian conditioned on them). The slot keeps its chain
        flag. A single-pattern model draws one pattern, as generate(n=1)."""
        if not self.is_loaded:
            raise RuntimeError("No model loaded. Call load_model() first.")
        if not 0 <= int(slot) < NUM_SLOTS:
            raise ValueError(f"not a slot: {slot!r} (0-{NUM_SLOTS - 1})")
        if self._bank is None:
            return self.generate(raw_patches, tempo, swing, fill_rate, step_rate, n=1,
                                 temperature=temperature, seed=seed)[0]
        import torch

        slot = int(slot)
        model, gen = self._bank, self._generator(seed)
        cond = torch.tensor(self._preprocessor.encode_condition(raw_patches, tempo, swing, fill_rate,
                                                                step_rate), dtype=torch.float32)[None]
        grid, length, chain = (torch.from_numpy(a)[None] for a in encode_bank_patterns(bank))
        zb, zs = model.bank_latent, model.slot_latent
        with torch.no_grad():
            bank_mu, slot_mu = model.encode(grid, length, chain, cond)
            z = torch.cat([bank_mu, slot_mu.flatten(1)], 1).double()[0]
            mine = torch.zeros(z.shape[0], dtype=torch.bool)
            mine[zb + slot * zs: zb + (slot + 1) * zs] = True
            noise = torch.randn(zs, generator=gen, dtype=torch.float64) * float(temperature)
            if self._prior is not None:
                mean = self._prior.net(cond)[0].double()
                tril = self._prior.scale_tril.double()
                cov = tril @ tril.T
                gain = cov[mine][:, ~mine] @ torch.linalg.inv(cov[~mine][:, ~mine])
                cond_mean = mean[mine] + gain @ (z[~mine] - mean[~mine])
                cond_cov = cov[mine][:, mine] - gain @ cov[~mine][:, mine]
                cond_cov = (cond_cov + cond_cov.T) / 2 + 1e-6 * torch.eye(zs, dtype=torch.float64)
                z[mine] = cond_mean + torch.linalg.cholesky(cond_cov) @ noise
            else:
                z[mine] = noise
            z = z.float()[None]
            out = self._decode_banks(z[:, :zb], z[:, zb:].reshape(1, NUM_SLOTS, zs), cond, gen)[0]
        pattern = out[slot]
        pattern["Chained"] = bool(bank[slot].get("Chained", False)) if slot < NUM_CHAINS else False
        return pattern

    # ── bank sampling ─────────────────────────────────────────────────────────

    @staticmethod
    def _generator(seed):
        import torch
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(int(seed))
        else:
            gen.seed()
        return gen

    def _sample_banks(self, cond: np.ndarray, n: int, temperature: float, seed) -> list:
        """n banks for one condition: latents through the kit prior."""
        import torch

        model, gen = self._bank, self._generator(seed)
        zb, zs = model.bank_latent, model.slot_latent
        c = torch.tensor(cond, dtype=torch.float32)[None].repeat(n, 1)
        with torch.no_grad():
            z = torch.randn(n, zb + NUM_SLOTS * zs, generator=gen) * float(temperature)
            if self._prior is not None:
                z = self._prior.net(c) + z @ self._prior.scale_tril.T
            return self._decode_banks(z[:, :zb], z[:, zb:].reshape(n, NUM_SLOTS, zs), c, gen)

    def _decode_banks(self, z_bank, z_slot, cond, gen) -> list:
        """Decode latents to banks: triggers above 0.5, accents and fills
        drawn, the most likely length, chains above 0.5."""
        import torch

        n = cond.shape[0]
        grid_logits, length_logits, chain_logits = self._bank.decode(z_bank, z_slot, cond)
        probs = torch.sigmoid(grid_logits).reshape(n, NUM_SLOTS, NUM_CHANNELS, NUM_STEPS, NUM_FEATURES)
        drawn = torch.rand(probs.shape, generator=gen) < probs
        grid = torch.cat([probs[..., :1] >= 0.5, drawn[..., 1:]], dim=-1).numpy()
        length = (length_logits.argmax(-1) + 1).numpy()
        chain = (torch.sigmoid(chain_logits) >= 0.5).numpy()
        return [decode_bank_arrays(grid[i], length[i], chain[i]) for i in range(n)]
