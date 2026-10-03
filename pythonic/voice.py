"""
Drum voice engine.

A vectorised implementation of the Pythonic drum voice.

Signal flow (per channel):

    oscillator (2x oversampled, band-limited shapes, pitch modulation)
      -> polyphase decimator -> amplitude envelope
    noise (uniform white -> state-variable filter, power-normalised)
      -> noise envelope (exponential / linear / clap "mod")
    mix (osc / noise gains) -> drive -> asymmetric soft shaper
      -> SVF peaking EQ -> level / pan / makeup gains

Everything is processed on a 4-sample block grid in absolute time: envelopes advance per sample but change stage only at block ends,
triggers take effect at the next block boundary, and slow pitch modulation is
updated on an 8 (oversampled) / 128 (oversampled) sample grid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, astuple

import numpy as np
from scipy.signal import lfilter


# Output scaling of the whole engine (a 0 dB patch with a full sine peaks at
# 0.8165 * 0.75 * 0.5 = 0.306 per channel).
MASTER_SCALE = 0.5

# Oscillator amplitude per waveform region
SINE_AMP = 0.8164966
TRI_HF_AMP = 0.8
SAW_HF_AMP = 0.58

# Decimator coefficient (2x -> 1x)
DEC_A = 0.07314892858
# Velocity curve: x * exp(K (x - 1))
VEL_K = 4.259782314
OSC_GAIN = 0.4634255171  # compensates the decimator DC gain (2 / (1 - DEC_A))

LN1000 = math.log(1000.0)

WAVE_SINE, WAVE_TRI, WAVE_SAW = 0, 1, 2
MOD_DECAY, MOD_SINE, MOD_NOISE = 0, 1, 2
FILT_LP, FILT_BP, FILT_HP = 0, 1, 2
NENV_EXP, NENV_LINEAR, NENV_MOD = 0, 1, 2


# --------------------------------------------------------------------------- helpers
def attack_time_s(attack_ms: float) -> float:
    """Attack times are quantised through the knob curve t = p * exp(5.526 p - 3.224)."""
    t = max(0.0, attack_ms) / 1000.0
    if t <= 0.0:
        return 0.0
    p = attack_norm(attack_ms)
    return p * math.exp(p * 5.526204586 - 3.223619223)


def attack_norm(attack_ms: float) -> float:
    """Inverse of the attack knob curve (normalised knob position 0..1)."""
    t = max(0.0, attack_ms) / 1000.0
    if t <= 0.0:
        return 0.0
    if t >= 10.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if mid * math.exp(mid * 5.526204586 - 3.223619223) < t:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def velocity_factor(velocity: float, sensitivity: float) -> float:
    """Linear velocity factor; sensitivity 1.0 == 100 % (range 0..2)."""
    inv = 2.0 - 2.0 * (float(velocity) - 1.0) / 126.0
    return max(0.0, 1.0 - inv * sensitivity * 0.5)


def velocity_gain(velocity: float, sensitivity: float) -> float:
    x = velocity_factor(velocity, sensitivity)
    g = x * math.exp(x * VEL_K - VEL_K)
    return g if abs(g) >= 1e-10 else 0.0


def pow2_dec(x):
    """2**x approximation used by the decaying / random pitch modulation."""
    p = (((x * 4.074710159e-05 + 0.0009640406934) * x + 0.01495699305) * x + 0.1719395369) * x + 1.0
    p = p * p
    return p * p


def pow2_sin(x):
    """2**x approximation used by the sine pitch modulation."""
    p = (((x * 3.834385643e-05 + 0.0008906660951) * x + 0.01501070894) * x + 0.1732061803) * x + 1.0
    p = p * p
    return p * p


def _cos_fine(ph):
    x = np.where(ph <= 0.5, ph, ph - 1.0)
    x2 = x * x
    return (((x2 * 46.29878998 - 82.69654846) * x2 + 64.71379089) * x2 - 19.73277473) * x2 + 0.9999709725


def _cos_coarse(ph):
    x = np.where(ph < 0.5, ph, ph - 1.0)
    x2 = x * x
    return (((x2 * 52.90157318 - 85.04795837) * x2 + 64.88814545) * x2 - 19.72719574) * x2 + 0.9996777177


def svf_coefs(f: float, k: float) -> np.ndarray:
    """Coefficients of the normalised two-integrator SVF (f in cycles/sample)."""
    s = math.sin(f * 2.0 * math.pi) * 0.5
    t = math.sin(f * math.pi)
    a = 1.0 / (s * k + 1.0)
    b = a * t * t
    c = 1.0 - t * t
    sb = math.sqrt(b)
    p = np.empty(8)
    p[0] = c
    p[1] = 1.0 / c
    p[2] = b / c
    p[3] = a * s / c
    p[4] = (a - b) / c
    p[5] = p[3] / sb
    p[6] = 2.0 * sb
    p[7] = (1.0 - a) / sb
    return p


def svf_tf(p: np.ndarray, mode: int):
    """Transfer function (b, a) of an SVF output tap (LP/BP/HP)."""
    p0, p1, p2, p3, p4, p5, p6, p7 = p
    # state x = [s1, s2]; s1' = s1 + p6 s2 ; v = p0 u - s1' - p7 s2 ; s2' = s2 + p6 v
    A = np.array([[1.0, p6], [-p6, 1.0 - p6 * (p6 + p7)]])
    B = np.array([0.0, p6 * p0])
    if mode == FILT_LP:     # y = p1 s1' + p2 v
        C = np.array([p1 - p2, p1 * p6 - p2 * (p6 + p7)])
        D = p2 * p0
    elif mode == FILT_BP:   # y = p5 s2' - p3 v
        q = p5 * p6 - p3
        C = np.array([-q, p5 - q * (p6 + p7)])
        D = q * p0
    else:                   # y = p4 v
        C = np.array([-p4, -p4 * (p6 + p7)])
        D = p4 * p0
    # H(z) = C (zI - A)^-1 B + D  ->  2nd order rational function
    tr = A[0, 0] + A[1, 1]
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    a = np.array([1.0, -tr, det])
    # numerator: C adj(zI - A) B + D * den
    adj_z = np.array([[1.0, 0.0], [0.0, 1.0]])  # coefficient of z in adj(zI-A)
    adj_1 = np.array([[-A[1, 1], A[0, 1]], [A[1, 0], -A[0, 0]]])
    n1 = C @ adj_z @ B
    n0 = C @ adj_1 @ B
    b = np.array([D, n1 + D * a[1], n0 + D * a[2]])
    return b, a


def noise_norm(mode: int, f: float, k: float, sr: float) -> float:
    """Gain that normalises the filtered uniform noise power."""
    if mode == FILT_BP:
        return math.sqrt((1.0 / 6.0) / ((0.5 - f) * f) * k)
    if mode == FILT_HP:
        fv5 = min(22050.0 / sr, 0.5)
        pw = (0.5 / fv5) ** min(k, 1.0)
        return math.sqrt(pw * k / ((((2 * k - 3) * f - k) * (2 * f - 1))))
    return math.sqrt(k / (((4 * k - 6) * f + 3) * f))


def shaper_params(amount: float):
    """(shape d, clamp k, dc offset, pre-drive, makeup) for distortion 0..1."""
    if amount < 1e-4:
        return 0.0, 1.0, 0.0, 1.0, 1.0
    d = amount ** 3 * 200.0
    if d < 1.0:
        k = (math.sqrt((d + 3.0) / d) + 1.0) / 3.0
        c = -1.0 / 3.0
        dc = ((abs(c) - c * c) * d + 1.0) * c
        return d, k, dc, 1.0, 1.0 - d / 30.0
    return d, 1.0, -0.407407403, d, 0.2 / ((d - 0.9589490891) + 0.4761904776) + 0.58


def apply_shaper(x, d, k, dc):
    if d == 0.0:
        return x
    c = x - 0.3333333433
    if d >= 1.0:
        c = np.clip(c, -1.0, 1.0)
        return ((np.abs(c) - c * c) + 1.0) * c + 0.407407403
    c = np.clip(c, -k, k)
    return ((np.abs(c) - c * c) * d + 1.0) * c - dc


def oscillator_wave(wave: int, ph: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Band-limited waveform for phase `ph` (cycles) at increment `w` (cycles/sample @2x)."""
    if wave == WAVE_SINE:
        return SINE_AMP * np.sin(2.0 * np.pi * ph)
    if wave == WAVE_TRI:
        x = 4.0 * ph
        v = np.where(x < 1.0, x, np.where(x < 3.0, 2.0 - x, x - 4.0))
        out = v.copy()
        mid = (w > 1.0 / 60.0) & (w <= 0.1)
        if np.any(mid):
            W = 1.2 - 12.0 * w[mid]
            vm = v[mid]
            norm = 1.0 / (W * 0.2 + 0.8)
            d = 1.0 - np.abs(vm)
            span = 1.0 - W
            z = d / span
            z2 = z * z
            cap = ((z2 * 0.1431494355 - 0.7790188193) * z2 + 0.6363011003) * span + W
            out[mid] = np.where(d < span, np.where(vm >= 0, cap, -cap), vm) * norm
        hi = w > 0.1
        if np.any(hi):
            out[hi] = TRI_HF_AMP * np.sin(2.0 * np.pi * ph[hi])
        return out
    # falling sawtooth
    out = 1.0 - 2.0 * ph
    soft = (w > 0.00125) & (w <= 0.1034082696)
    if np.any(soft):
        ws = w[soft]
        ps = ph[soft]
        D = ((34.40639877 - ws * 239.1292725) * ws + 1.416769266) * ws
        near_lo = ps < D
        near_hi = ps > 1.0 - D
        xe = np.where(near_lo, ps / D, (ps - 1.0) / D)
        edge = ((xe * xe * 0.06132304668 - 0.6244748235) * xe * xe + 1.566493511) * xe
        xl = (ps - 0.5) / (D - 0.5)
        blend = ws > 0.04118342325
        amp = np.where(blend, 1.0 - D * D * D * 26.87999916, 1.0)
        a4 = np.where(blend, amp * 4.0 * D, 0.0)
        lin = ((xl * xl * 0.06132304668 - 0.6244748235) * xl * xl + 1.566493511) * xl * a4 + (amp - a4) * xl
        out[soft] = np.where(near_lo | near_hi, edge * amp, lin)
    hi = w > 0.1034082696
    if np.any(hi):
        out[hi] = SAW_HF_AMP * np.sin(2.0 * np.pi * ph[hi])
    return out


# --------------------------------------------------------------------------- params
@dataclass(frozen=True)
class VoiceParams:
    """Patch parameters in display units (as in .mtpreset files)."""
    wave: int = WAVE_SINE
    osc_freq: float = 632.46            # Hz
    osc_attack_ms: float = 0.0
    osc_decay_ms: float = 316.23
    mod_mode: int = MOD_DECAY
    mod_rate: float = 353.33            # ms (decay) or Hz (sine / noise)
    mod_amount: float = 0.0             # semitones
    noise_filter: int = FILT_LP
    noise_freq: float = 20000.0         # Hz
    noise_q: float = 0.70710683
    noise_stereo: bool = False
    noise_env: int = NENV_EXP
    noise_attack_ms: float = 0.0
    noise_decay_ms: float = 316.23
    osc_mix: float = 0.5                # 1 = oscillator only, 0 = noise only
    distortion: float = 0.0             # 0..1
    eq_freq: float = 632.46
    eq_gain_db: float = 0.0
    level_db: float = 0.0
    pan: float = 0.0                    # -100..100
    osc_vel: float = 0.0                # 1.0 == 100 %
    noise_vel: float = 0.0
    mod_vel: float = 0.0
    pitch_ratio: float = 1.0            # note pitch offset (scales osc, noise filter, EQ)
    mono: bool = False


class _Derived:
    """Engine constants derived from VoiceParams."""

    def __init__(self, p: VoiceParams, sr: float):
        self.sr = sr
        self.max_inc = min(10000.0 / sr, 0.235)
        self.wave = int(p.wave)
        self.inc0 = min(max(p.osc_freq, 1.0) * p.pitch_ratio / (2.0 * sr), self.max_inc)

        # oscillator envelope
        t_atk = attack_time_s(p.osc_attack_ms)
        self.osc_atk_n = t_atk * sr
        self.osc_fade_n = max(min(t_atk, 0.01) * sr, 1.0)
        self.osc_atk_c = 1000.0 ** (1.0 / max(self.osc_atk_n, 1.0))
        t_dcy = min(max(p.osc_decay_ms, 10.0), 10000.0) / 1000.0
        self.osc_dcy_c = 1000.0 ** (-1.0 / max(t_dcy * sr, 1.0))

        # pitch modulation
        self.mod_mode = int(p.mod_mode)
        rng = 96.0 if self.mod_mode == MOD_DECAY else 48.0
        amt = max(-rng, min(rng, float(p.mod_amount)))
        self.mod_a = math.copysign(math.sqrt(abs(amt) / rng), amt)
        self.mod_scale = 8.0 if self.mod_mode == MOD_DECAY else 4.0
        self.mod_dc = 1.0
        self.mod_b0 = 0.0
        self.mod_b4 = 0.0
        self.mod_lpc = 0.0
        self.mod_hold8 = False
        if self.mod_mode == MOD_DECAY:
            T = float(p.mod_rate) / 1000.0
            if math.isfinite(T) and T > 0.0:
                self.mod_dc = 1000.0 ** (-1.0 / max(T * sr * 2.0, 1.0))
        elif self.mod_mode == MOD_SINE:
            self.mod_b0 = min(max(float(p.mod_rate), 0.0) / (2.0 * sr), self.max_inc)
        else:
            b0 = min(max(float(p.mod_rate), 0.0) / (2.0 * sr), self.max_inc)
            r = 2.0 * b0
            if b0 <= 0.0015:
                r *= 4.0
                self.mod_hold8 = True
            self.mod_b4 = math.sqrt(0.5 / r) if r > 1e-13 else 0.0
            self.mod_lpc = ((r * 11.01200485 - 13.33381844) * r + 5.846880913) * r

        # noise
        self.noise_filter = int(p.noise_filter)
        self.stereo = bool(p.noise_stereo) and not p.mono
        f = min(max(p.noise_freq * p.pitch_ratio / sr, 1e-4), 2.0 * self.max_inc)
        k = 1.0 / min(max(float(p.noise_q), 0.1), 10000.0)
        self.nf_b, self.nf_a = svf_tf(svf_coefs(f, k), self.noise_filter)
        self.noise_gain = noise_norm(self.noise_filter, f, k, sr)

        self.noise_env = int(p.noise_env)
        if self.noise_env == NENV_LINEAR:
            n_atk = max(p.noise_attack_ms, 0.0) / 1000.0
            n_dcy = min(max(p.noise_decay_ms, 6.6667), 6666.67) / 1000.0
            atk_p = 0.0
        else:
            atk_p = attack_norm(p.noise_attack_ms)
            n_atk = atk_p * math.exp(atk_p * 5.526204586 - 3.223619223) if atk_p > 1e-13 else 0.0
            n_dcy = min(max(p.noise_decay_ms, 10.0), 10000.0) / 1000.0
        self.n_atk_n = n_atk * sr
        self.n_fade_n = max(min(n_atk, 0.01) * sr, 1.0)
        self.n_atk_c = 1000.0 ** (1.0 / max(self.n_atk_n, 1.0))
        self.n_dcy_c = 1000.0 ** (-1.0 / max(n_dcy * sr, 1.0))
        self.n_lin_up = 1.0 / max(self.n_atk_n, 1.0)
        self.n_lin_dn = 1.0 / max(n_dcy * sr, 1.0)
        self.n_bursts = 1
        self.n_burst_c = 1.0
        if self.noise_env == NENV_MOD and self.n_atk_n > 0.0:
            self.n_bursts = max(1, int(n_atk / (atk_p * 0.095 + 0.005) + 2.12))
            self.n_burst_c = 15.84893227 ** (-1.0 / max(n_atk * sr / self.n_bursts, 1.0))

        # mix, shaper, EQ, output
        u = (1.0 - min(max(p.osc_mix, 0.0), 1.0)) * 2.0 - 1.0
        self.mix_osc = math.exp(u * -2.878231525) * (1.0 - u) if u > 0.0 else 1.0
        self.mix_noise = math.exp(u * 2.878231525) * (1.0 + u) if u < 0.0 else 1.0
        self.sh_d, self.sh_k, self.sh_dc, self.drive, post = shaper_params(min(max(p.distortion, 0.0), 1.0))

        gnorm = (min(max(p.eq_gain_db, -40.0), 40.0) / 40.0 + 1.0) / 2.0
        G = 10.0 ** (2.0 * gnorm - 1.0)
        K = 2.0 if gnorm >= 0.5 else 0.5
        self.eq_amt = G * K - K / G
        ef = min(max(p.eq_freq * p.pitch_ratio / sr, 1e-4), 2.0 * self.max_inc)
        self.eq_b, self.eq_a = svf_tf(svf_coefs(ef, K / G), FILT_BP)
        eq_att = 1.0 / G if gnorm > 0.5 else 1.0

        level = 10.0 ** (p.level_db / 20.0) if p.level_db > -60.0 else 0.0
        pn = (min(max(p.pan, -100.0), 100.0) + 100.0) / 200.0
        g = MASTER_SCALE * level * eq_att * post
        if p.mono:
            self.out_l = self.out_r = 0.75 * g
        else:
            self.out_l = (1.0 - pn * pn) * g
            self.out_r = (2.0 * pn - pn * pn) * g

        self.smooth = 1.0 - math.exp(-2.0 * math.pi * min(40.0 / sr, 0.5))
        self.release_slope = -100.0 / sr


def _sine_mod_granularity(A, b0, inc0, max_inc):
    """1 = per sample, 2 = per 2 samples, 3 = per 8 samples, 4 = per 128 samples (all @2x)."""
    A = abs(A)
    if A <= 1e-4:
        return 0
    d = (pow2_sin(math.sin(b0 * 8.0 + 0.7853981853) * A) - pow2_sin(A * 0.7071067691)) * inc0
    if d >= 1e-7:
        if d >= 8e-5:
            return 2 if d < 0.028 else 1
        return 3
    return 3 if b0 * 128.0 > max_inc else 4


# --------------------------------------------------------------------------- envelopes
class _Env:
    """Amplitude envelope state machine (stages 0 fade-out, 1 attack, 2 decay, 3 release, 4 off)."""

    __slots__ = ('stage', 'level', 'slope', 'target', 'count')

    def __init__(self):
        self.stage = 4
        self.level = 0.0
        self.slope = 0.0
        self.target = 0.0
        self.count = 0


def _ramp(start, step, nblk):
    """Per-sample linear ramp over `nblk` 4-sample blocks: start + step * (1..)."""
    return start + step * np.arange(1, nblk * 4 + 1)


def _expo(start, coef, nblk):
    if start <= 0.0:
        return np.zeros(nblk * 4)
    e = math.log(start) + math.log(coef) * np.arange(1, nblk * 4 + 1)
    return np.exp(np.minimum(e, 50.0))


# --------------------------------------------------------------------------- voice
class DrumVoice:
    """One drum voice (one per channel, monophonic, retriggerable)."""

    def __init__(self, sample_rate: int = 44100, seed: int = 0):
        self.sr = float(sample_rate)
        self._rng = np.random.default_rng(seed)
        self.params = VoiceParams()
        self.d = _Derived(self.params, self.sr)
        self._key = astuple(self.params)
        self.clock = 0                  # samples rendered (block grid is absolute)
        self._carry = np.zeros((0, 2), dtype=np.float32)
        self._pending = []              # ('trig', velocity) / ('choke',)
        self.pitch_drift = 1.0
        # oscillator state
        self.phase = 0.0
        self.lfo = 0.0
        self.penv = 1.0
        self.pmode4 = False
        self.hold_inc = self.d.inc0
        self.A = 0.0
        self.smode = 0
        self.nlp = np.zeros(2)          # random-mod lowpass states
        self.dec_x = 0.0
        self.dec_y = 0.0
        self.trig_vel = 127.0
        self.osc_vel = 0.0
        self.noise_vel = 0.0
        self.oenv = _Env()
        self.nenv = _Env()
        self.nf_zi = [np.zeros(2), np.zeros(2)]
        self.eq_zi = [np.zeros(2), np.zeros(2)]
        # smoothed gains: osc mix, noise mix, drive, out L, out R, eq amount
        self.g = np.array(self._gain_targets())

    # ---------------------------------------------------------------- params
    def set_params(self, params: VoiceParams):
        key = astuple(params)
        if key == self._key:
            return
        old = self.params
        self.params = params
        self._key = key
        self.d = _Derived(params, self.sr)
        if old.noise_filter != params.noise_filter:
            self.nf_zi = [np.zeros(2), np.zeros(2)]
        if not self.is_active:
            self.g = np.array(self._gain_targets())

    def _gain_targets(self):
        d = self.d
        return (d.mix_osc, d.mix_noise, d.drive, d.out_l, d.out_r, d.eq_amt)

    @property
    def is_active(self) -> bool:
        return (self.oenv.stage != 4 or self.nenv.stage != 4 or len(self._pending) > 0
                or len(self._carry) > 0)

    # ---------------------------------------------------------------- events
    def trigger(self, velocity: float = 127.0):
        self._pending.append(('trig', float(velocity)))

    def choke(self):
        self._pending.append(('choke',))

    def _apply_pending(self):
        for ev in self._pending:
            if ev[0] == 'choke':
                for env in (self.oenv, self.nenv):
                    if env.stage != 4:
                        env.slope = max(env.level, 1e-6) * self.d.release_slope
                        env.stage = 3
            else:
                self._do_trigger(ev[1])
        self._pending = []

    def _reset_osc(self):
        d = self.d
        self.phase = 0.125
        self.dec_y = 0.0
        self.lfo = 0.0
        self.penv = 1.0
        self.pmode4 = False
        r = self._rng.uniform(-1.0, 1.0)
        self.nlp[:] = r
        self.smode = (_sine_mod_granularity(self.A, d.mod_b0, d.inc0, d.max_inc)
                      if d.mod_mode == MOD_SINE else 0)
        if d.mod_mode == MOD_SINE and self.smode == 4:
            self._sine_hold_update()
        else:
            self.hold_inc = d.inc0

    def _do_trigger(self, velocity):
        d = self.d
        p = self.params
        self.g = np.array(self._gain_targets())
        self.osc_vel = velocity_gain(velocity, p.osc_vel) * OSC_GAIN
        self.noise_vel = velocity_gain(velocity, p.noise_vel)
        am = self.d.mod_a * velocity_factor(velocity, p.mod_vel)
        self.A = abs(am) * d.mod_scale * am
        oe, ne = self.oenv, self.nenv
        if d.osc_atk_n > 0.0:
            oe.stage = 0
            oe.slope = max(oe.level, 1e-6) * (-1.0 / d.osc_fade_n)
        else:
            oe.level = self.osc_vel
            oe.stage = 2
            self._reset_osc()
        if d.noise_env == NENV_MOD or d.n_atk_n <= 0.0:
            ne.level = self.noise_vel
            ne.stage = 2 if d.n_atk_n <= 0.0 else 1
            ne.count = 0
        else:
            ne.stage = 0
            ne.slope = max(ne.level, 1e-6) * (-1.0 / d.n_fade_n)

    # ---------------------------------------------------------------- envelopes
    def _run_osc_env(self, nb):
        """Returns per-sample envelope (nb*4) and the block index of a phase reset (or -1)."""
        e = self.oenv
        d = self.d
        out = np.zeros(nb * 4)
        reset = -1
        b = 0
        while b < nb:
            n = nb - b
            if e.stage == 4:
                break
            if e.stage == 0 or e.stage == 3:
                v = _ramp(e.level, e.slope, n)
                ends = np.nonzero(v[3::4] <= 1e-13)[0]
                if len(ends) == 0:
                    out[b * 4:] = v
                    e.level = v[-1]
                    break
                m = ends[0]
                out[b * 4:(b + m + 1) * 4] = np.maximum(v[:(m + 1) * 4], 0.0)
                b += m + 1
                if e.stage == 0:
                    e.stage = 1
                    e.level = max(self.osc_vel * 0.001, 1e-6)
                    reset = b
                    # the oscillator restarts at block b; render up to there first
                    if b < nb:
                        out_rest, _ = self._run_osc_env(nb - b)
                        out[b * 4:] = out_rest
                    return out, reset
                e.stage = 4
                e.level = 0.0
            elif e.stage == 1:
                v = _expo(e.level, d.osc_atk_c, n)
                ends = np.nonzero(v[3::4] >= self.osc_vel)[0]
                if len(ends) == 0:
                    out[b * 4:] = v
                    e.level = v[-1]
                    break
                m = ends[0]
                out[b * 4:(b + m + 1) * 4] = np.minimum(v[:(m + 1) * 4], self.osc_vel)
                b += m + 1
                e.stage = 2
                e.level = self.osc_vel
            else:
                v = _expo(e.level, d.osc_dcy_c, n)
                ends = np.nonzero(v[3::4] <= 1e-6)[0]
                if len(ends) == 0:
                    out[b * 4:] = v
                    e.level = v[-1]
                    break
                m = ends[0]
                out[b * 4:(b + m + 1) * 4] = v[:(m + 1) * 4]
                b += m + 1
                e.level = v[(m + 1) * 4 - 1]
                e.slope = max(e.level, 1e-6) * d.release_slope
                e.stage = 3
        return out, reset

    def _run_noise_env(self, nb):
        e = self.nenv
        d = self.d
        vel = self.noise_vel
        out = np.zeros(nb * 4)
        b = 0
        while b < nb:
            n = nb - b
            st = e.stage
            if st == 4:
                break
            if st == 0 or st == 3:
                v = _ramp(e.level, e.slope, n)
                ends = np.nonzero(v[3::4] <= 1e-13)[0]
                if len(ends) == 0:
                    out[b * 4:] = v
                    e.level = v[-1]
                    break
                m = ends[0]
                out[b * 4:(b + m + 1) * 4] = np.maximum(v[:(m + 1) * 4], 0.0)
                b += m + 1
                if st == 0:
                    e.stage = 1
                    e.level = max(vel * 0.001, 1e-6) if d.noise_env == NENV_EXP else 0.0
                else:
                    e.stage = 4
                    e.level = 0.0
            elif st == 1 and d.noise_env == NENV_MOD:
                v = _expo(e.level, d.n_burst_c, n)
                ends = np.nonzero(v[3::4] <= vel * 0.06309573352)[0]
                if len(ends) == 0:
                    out[b * 4:] = v
                    e.level = v[-1]
                    break
                m = ends[0]
                out[b * 4:(b + m + 1) * 4] = v[:(m + 1) * 4]
                b += m + 1
                e.count += 1
                e.level = vel
                if e.count >= d.n_bursts:
                    e.stage = 2
            elif st == 1:
                if d.noise_env == NENV_LINEAR:
                    v = _ramp(e.level, d.n_lin_up * vel, n)
                else:
                    v = _expo(e.level, d.n_atk_c, n)
                ends = np.nonzero(v[3::4] >= vel)[0]
                if len(ends) == 0:
                    out[b * 4:] = v
                    e.level = v[-1]
                    break
                m = ends[0]
                out[b * 4:(b + m + 1) * 4] = np.minimum(v[:(m + 1) * 4], vel)
                b += m + 1
                e.stage = 2
                e.level = vel
            elif d.noise_env == NENV_LINEAR:
                v = _ramp(e.level, -d.n_lin_dn * vel, n)
                ends = np.nonzero(v[3::4] <= 1e-13)[0]
                if len(ends) == 0:
                    out[b * 4:] = v
                    e.level = v[-1]
                    break
                m = ends[0]
                out[b * 4:(b + m + 1) * 4] = np.maximum(v[:(m + 1) * 4], 0.0)
                b += m + 1
                e.stage = 4
                e.level = 0.0
            else:
                v = _expo(e.level, d.n_dcy_c, n)
                ends = np.nonzero(v[3::4] <= 1e-6)[0]
                if len(ends) == 0:
                    out[b * 4:] = v
                    e.level = v[-1]
                    break
                m = ends[0]
                out[b * 4:(b + m + 1) * 4] = v[:(m + 1) * 4]
                b += m + 1
                e.level = v[(m + 1) * 4 - 1]
                e.slope = max(e.level, 1e-6) * d.release_slope
                e.stage = 3
        return out

    # ---------------------------------------------------------------- oscillator
    def _sine_hold_update(self):
        d = self.d
        c = float(_cos_coarse(np.array([self.lfo]))[0])
        self.hold_inc = min(d.inc0 * self.pitch_drift * pow2_sin(self.A * c), d.max_inc)
        self.lfo = (self.lfo + d.mod_b0 * 128.0) % 1.0

    def _increments(self, nb, blk0):
        """Per-oversampled-sample phase increments for nb blocks starting at absolute block blk0."""
        d = self.d
        inc0 = d.inc0 * self.pitch_drift
        n_os = nb * 8
        A = self.A
        if abs(A) <= 1e-4:
            return np.full(n_os, min(inc0, d.max_inc))
        if d.mod_mode == MOD_DECAY:
            j = np.arange(nb)
            env = self.penv * d.mod_dc ** (8.0 * j)
            inc_blk = np.minimum(inc0 * pow2_dec(A * env), d.max_inc)
            blk = blk0 + j
            mode4 = self.pmode4
            held = self.hold_inc
            cut = False
            # 64-sample (16-block) housekeeping: mode-4 latch and env cut-off
            for i in np.nonzero(blk % 16 == 0)[0]:
                e = 0.0 if cut else env[i]
                if e < 0.001:
                    cut = True
                    e = 0.0
                x = A * 0.6931471825 * e
                mode4 = abs(math.exp(x * d.mod_dc) - math.exp(x)) * inc0 < 1e-7
                if mode4:
                    held = min(inc0 * pow2_dec(A * e), d.max_inc)
                    inc_blk[i:] = held
                elif cut:
                    inc_blk[i:] = min(inc0, d.max_inc)
            if self.pmode4:
                first = np.nonzero(blk % 16 == 0)[0]
                stop = first[0] if len(first) else nb
                inc_blk[:stop] = self.hold_inc
            self.pmode4 = mode4
            self.hold_inc = held
            self.penv = 0.0 if cut else self.penv * d.mod_dc ** (8.0 * nb)
            return np.repeat(inc_blk, 8)
        if d.mod_mode == MOD_SINE:
            sm = self.smode
            if sm in (1, 2):
                k = np.arange(n_os)
                if sm == 2:
                    k = (k // 2) * 2
                ph = (self.lfo + k * d.mod_b0) % 1.0
                w = np.minimum(inc0 * pow2_sin(A * _cos_fine(ph)), d.max_inc)
                self.lfo = (self.lfo + n_os * d.mod_b0) % 1.0
                return w
            if sm == 3:
                ph = (self.lfo + np.arange(nb) * 8.0 * d.mod_b0) % 1.0
                w = np.minimum(inc0 * pow2_sin(A * _cos_coarse(ph)), d.max_inc)
                self.lfo = (self.lfo + n_os * d.mod_b0) % 1.0
                return np.repeat(w, 8)
            # per 64 output samples, aligned to the absolute block grid
            w = np.empty(nb)
            for i in range(nb):
                if (blk0 + i) % 16 == 0:
                    self._sine_hold_update()
                w[i] = self.hold_inc
            return np.repeat(w, 8)
        # random (noise) modulation: two cascaded one-pole lowpasses on uniform noise
        step = 8 if d.mod_hold8 else 2
        nv = n_os // step
        u = self._rng.uniform(-1.0, 1.0, nv) * d.mod_b4
        c = d.mod_lpc
        bq, aq = [c], [1.0, -(1.0 - c)]
        s1, zf1 = lfilter(bq, aq, u, zi=[(1.0 - c) * self.nlp[0]])
        s2, zf2 = lfilter(bq, aq, s1, zi=[(1.0 - c) * self.nlp[1]])
        self.nlp[0] = s1[-1]
        self.nlp[1] = s2[-1]
        w = np.minimum(inc0 * pow2_dec(A * np.clip(s2, -2.0, 2.0)), d.max_inc)
        return np.repeat(w, step)

    def _render_osc(self, nb, blk0):
        """Decimated oscillator output for nb blocks (nb*4 samples)."""
        w = self._increments(nb, blk0)
        cs = np.cumsum(w)
        ph = (self.phase + cs - w) % 1.0
        self.phase = (self.phase + cs[-1]) % 1.0
        x = oscillator_wave(self.d.wave, ph, w)
        xe = x[0::2]
        xo = x[1::2]
        xprev = np.concatenate(([self.dec_x], xo[:-1]))
        u = DEC_A * (xe + xprev) + xo + xe
        a2 = DEC_A * DEC_A
        y, _ = lfilter([1.0], [1.0, -a2], u, zi=[a2 * self.dec_y])
        self.dec_x = xo[-1]
        self.dec_y = y[-1]
        return y

    # ---------------------------------------------------------------- main
    def _smoothed(self, nb):
        """Per-sample smoothed gains (6, nb*4); gains glide per 4-sample block."""
        tgt = np.array(self._gain_targets())
        if np.allclose(self.g, tgt, rtol=0.0, atol=1e-7):
            self.g = tgt
            return None
        k = self.d.smooth
        decay = (1.0 - k) ** np.arange(nb)
        g = tgt[:, None] + (self.g - tgt)[:, None] * decay[None, :]
        self.g = tgt + (self.g - tgt) * (1.0 - k) ** nb
        return np.repeat(g, 4, axis=1)

    def _render_blocks(self, nb):
        d = self.d
        blk0 = self.clock // 4
        n = nb * 4
        out = np.zeros((n, 2))
        if self.oenv.stage == 4 and self.nenv.stage == 4:
            self.clock += n
            return out

        osc_alive = self.oenv.stage != 4
        oenv, reset = self._run_osc_env(nb)
        if osc_alive:
            if reset < 0:
                osc = self._render_osc(nb, blk0)
            else:
                osc = np.empty(n)
                if reset > 0:
                    osc[:reset * 4] = self._render_osc(reset, blk0)
                self._reset_osc()
                if reset < nb:
                    osc[reset * 4:] = self._render_osc(nb - reset, blk0 + reset)
        else:
            osc = np.zeros(n)

        noise_on = self.nenv.stage != 4 and (d.mix_noise > 0.0 or self.g[1] > 0.0)
        nenv = self._run_noise_env(nb) if noise_on else None

        g = self._smoothed(nb)
        if g is None:
            go, gn, gd, gl, gr, ge = self.g
        else:
            go, gn, gd, gl, gr, ge = g

        sig_o = osc * oenv * go * gd
        chans = 2 if d.stereo else 1
        sig = np.empty((n, chans))
        for c in range(chans):
            s = sig_o.copy()
            if noise_on:
                wn = self._rng.uniform(-1.0, 1.0, n) * d.noise_gain
                yn, self.nf_zi[c] = lfilter(d.nf_b, d.nf_a, wn, zi=self.nf_zi[c])
                s += yn * nenv * gn * gd
            s = apply_shaper(s, d.sh_d, d.sh_k, d.sh_dc)
            if np.any(np.asarray(ge) != 0.0):
                bp, self.eq_zi[c] = lfilter(d.eq_b, d.eq_a, s, zi=self.eq_zi[c])
                s = s + ge * bp
            sig[:, c] = s
        out[:, 0] = sig[:, 0] * gl
        out[:, 1] = sig[:, -1] * gr
        self.clock += n
        return out

    def idle(self, num_samples: int):
        """Advance time without rendering (silent / muted channel)."""
        self._pending = []
        k = min(len(self._carry), num_samples)
        self._carry = self._carry[k:]
        n = num_samples - k
        if n > 0:
            nb = (n + 3) // 4
            self.clock += nb * 4
            self._carry = np.zeros((nb * 4 - n, 2), dtype=np.float32)

    def process(self, num_samples: int) -> np.ndarray:
        """Render `num_samples` stereo samples (float32, shape (n, 2))."""
        out = np.empty((num_samples, 2), dtype=np.float32)
        pos = 0
        if len(self._carry):
            k = min(len(self._carry), num_samples)
            out[:k] = self._carry[:k]
            self._carry = self._carry[k:]
            pos = k
        if pos < num_samples:
            if self._pending:
                self._apply_pending()
            need = num_samples - pos
            nb = (need + 3) // 4
            blk = self._render_blocks(nb)
            out[pos:] = blk[:need]
            self._carry = blk[need:].astype(np.float32)
        return out


# --------------------------------------------------------------------------- preset helpers
def _num(v, default=0.0) -> float:
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().strip('"')
    if s.lower().startswith('inf'):
        return float('inf')
    num = ''
    for ch in s:
        if ch in '+-.0123456789eE' and not (ch in 'eE' and not num):
            num += ch
        else:
            break
    try:
        return float(num)
    except ValueError:
        return default


def params_from_patch(patch: dict, pitch_ratio: float = 1.0, mono: bool = False) -> VoiceParams:
    """Build VoiceParams from a parsed .mtpreset / .mtdrum patch (display units)."""
    g = patch.get
    s = lambda k, d: str(g(k, d)).strip().strip('"')
    mix = g('Mix', 50.0)
    if isinstance(mix, (tuple, list)):
        osc_pct = _num(mix[0])
    elif isinstance(mix, str) and '/' in mix:
        osc_pct = _num(mix.split('/')[0])
    else:
        osc_pct = _num(mix, 50.0)
    stereo = g('NStereo', False)
    stereo = stereo if isinstance(stereo, bool) else s('NStereo', 'Off').lower() in ('on', 'true', '1')
    return VoiceParams(
        wave={'Sine': 0, 'Triangle': 1, 'Saw': 2}.get(s('OscWave', 'Sine'), 0),
        osc_freq=_num(g('OscFreq', 632.46)),
        osc_attack_ms=_num(g('OscAtk', 0.0)),
        osc_decay_ms=_num(g('OscDcy', 316.23)),
        mod_mode={'Decay': 0, 'Sine': 1, 'Noise': 2}.get(s('ModMode', 'Decay'), 0),
        mod_rate=_num(g('ModRate', 353.33)),
        mod_amount=_num(g('ModAmt', 0.0)),
        noise_filter={'LP': 0, 'BP': 1, 'HP': 2}.get(s('NFilMod', 'LP'), 0),
        noise_freq=_num(g('NFilFrq', 20000.0)),
        noise_q=_num(g('NFilQ', 0.70710683)),
        noise_stereo=bool(stereo),
        noise_env={'Exp': 0, 'Linear': 1, 'Mod': 2}.get(s('NEnvMod', 'Exp'), 0),
        noise_attack_ms=_num(g('NEnvAtk', 0.0)),
        noise_decay_ms=_num(g('NEnvDcy', 316.23)),
        osc_mix=osc_pct / 100.0,
        distortion=_num(g('DistAmt', 0.0)) / 100.0,
        eq_freq=_num(g('EQFreq', 632.46)),
        eq_gain_db=_num(g('EQGain', 0.0)),
        level_db=_num(g('Level', 0.0)),
        pan=_num(g('Pan', 0.0)),
        osc_vel=_num(g('OscVel', 0.0)) / 100.0,
        noise_vel=_num(g('NVel', 0.0)) / 100.0,
        mod_vel=_num(g('ModVel', 0.0)) / 100.0,
        pitch_ratio=pitch_ratio,
        mono=mono,
    )
