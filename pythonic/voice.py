"""
Drum voice engine.

Patch parameters, derived engine constants and the DrumVoice front end; the
per-sample render loop is compiled with numba (voice_kernel.py).

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
from dataclasses import dataclass

import numpy as np

from . import voice_kernel as K


# Output scaling of the whole engine (a 0 dB patch with a full sine peaks at
# 0.8165 * 0.75 * 0.5 = 0.306 per channel).
MASTER_SCALE = 0.5

# Velocity curve: x * exp(K (x - 1))
VEL_K = 4.259782314
OSC_GAIN = 0.4634255171  # compensates the decimator DC gain (2 / (1 - DEC_A))

WAVE_SINE, WAVE_TRI, WAVE_SAW = 0, 1, 2
MOD_DECAY, MOD_SINE, MOD_NOISE = K.MOD_DECAY, K.MOD_SINE, K.MOD_NOISE
FILT_LP, FILT_BP, FILT_HP = 0, 1, 2
NENV_EXP, NENV_LINEAR, NENV_MOD = K.NENV_EXP, K.NENV_LINEAR, K.NENV_MOD


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


# --------------------------------------------------------------------------- voice
def _pack(d: _Derived):
    """Flat parameter arrays for the render kernel."""
    p = np.zeros(K.NUM_P)
    p[K.P_INC0] = d.inc0
    p[K.P_MAX_INC] = d.max_inc
    p[K.P_OSC_ATK_C] = d.osc_atk_c
    p[K.P_OSC_DCY_C] = d.osc_dcy_c
    p[K.P_MOD_DC] = d.mod_dc
    p[K.P_MOD_B0] = d.mod_b0
    p[K.P_MOD_B4] = d.mod_b4
    p[K.P_MOD_LPC] = d.mod_lpc
    p[K.P_NF_B:K.P_NF_B + 3] = d.nf_b
    p[K.P_NF_A:K.P_NF_A + 3] = d.nf_a
    p[K.P_NOISE_GAIN] = d.noise_gain
    p[K.P_N_ATK_C] = d.n_atk_c
    p[K.P_N_DCY_C] = d.n_dcy_c
    p[K.P_N_LIN_UP] = d.n_lin_up
    p[K.P_N_LIN_DN] = d.n_lin_dn
    p[K.P_N_BURST_C] = d.n_burst_c
    p[K.P_GAINS:K.P_GAINS + 6] = (d.mix_osc, d.mix_noise, d.drive, d.out_l, d.out_r, d.eq_amt)
    p[K.P_SH_D] = d.sh_d
    p[K.P_SH_K] = d.sh_k
    p[K.P_SH_DC] = d.sh_dc
    p[K.P_EQ_B:K.P_EQ_B + 3] = d.eq_b
    p[K.P_EQ_A:K.P_EQ_A + 3] = d.eq_a
    p[K.P_SMOOTH] = d.smooth
    p[K.P_RELEASE] = d.release_slope
    ip = np.zeros(K.NUM_IP, dtype=np.int64)
    ip[K.IP_WAVE] = d.wave
    ip[K.IP_MOD_MODE] = d.mod_mode
    ip[K.IP_NOISE_ENV] = d.noise_env
    ip[K.IP_STEREO] = int(d.stereo)
    ip[K.IP_MOD_HOLD8] = int(d.mod_hold8)
    ip[K.IP_N_BURSTS] = d.n_bursts
    return p, ip


class DrumVoice:
    """One drum voice (one per channel, monophonic, retriggerable).

    The voice state lives in two flat arrays that the compiled kernel
    (voice_kernel.py) advances; events are applied here, at the start of the
    next render.
    """

    def __init__(self, sample_rate: int = 44100, seed: int = 0):
        self.sr = float(sample_rate)
        # Separate streams for the noise source and the random pitch modulation, so
        # the output does not depend on how the audio is split into blocks
        self._rng = np.random.default_rng(seed)
        self._mod_rng = np.random.default_rng((seed, 1))
        self.params = VoiceParams()
        self.d = _Derived(self.params, self.sr)
        self._p, self._ip = _pack(self.d)
        self._fs = np.zeros(K.NUM_S)
        self._is = np.zeros(K.NUM_I, dtype=np.int64)
        self._is[K.I_OSTAGE] = 4
        self._is[K.I_NSTAGE] = 4
        self._fs[K.S_HOLD] = self.d.inc0
        self._fs[K.S_PENV] = 1.0
        self._fs[K.S_DRIFT] = 1.0
        self._fs[K.S_G:K.S_G + 6] = self._p[K.P_GAINS:K.P_GAINS + 6]
        self.clock = 0                  # samples rendered (block grid is absolute)
        self._carry = np.zeros((0, 2), dtype=np.float32)
        self._pending = []              # ('trig', velocity) / ('choke',)
        if not DrumVoice._kernel_ready:
            self._warm_up()

    _kernel_ready = False

    def _warm_up(self):
        """Compile (or load from the cache) the kernel now rather than in the audio callback."""
        fs, st = self._fs.copy(), self._is.copy()
        rng = np.random.default_rng(0)
        K.reset_osc(self._p, self._ip, fs, st, rng)
        K.render(self._p, self._ip, fs, st, rng, rng, 1, 0, np.empty((4, 2)))
        DrumVoice._kernel_ready = True

    @property
    def pitch_drift(self) -> float:
        return float(self._fs[K.S_DRIFT])

    @pitch_drift.setter
    def pitch_drift(self, value: float):
        self._fs[K.S_DRIFT] = value

    # ---------------------------------------------------------------- params
    def set_params(self, params: VoiceParams):
        if params == self.params:
            return
        old = self.params
        self.params = params
        self.d = _Derived(params, self.sr)
        self._p, self._ip = _pack(self.d)
        if old.noise_filter != params.noise_filter:
            self._fs[K.S_NF_ZI:K.S_NF_ZI + 4] = 0.0
        if not self.is_active:
            self._fs[K.S_G:K.S_G + 6] = self._p[K.P_GAINS:K.P_GAINS + 6]

    @property
    def is_active(self) -> bool:
        # a silent carry (block remainder) is not activity: idle() consumes it as well
        return (self._is[K.I_OSTAGE] != 4 or self._is[K.I_NSTAGE] != 4 or len(self._pending) > 0
                or bool(self._is[K.I_EQ_TAIL]) or bool(self._carry.any()))

    # ---------------------------------------------------------------- events
    def trigger(self, velocity: float = 127.0):
        self._pending.append(('trig', float(velocity)))

    def choke(self):
        self._pending.append(('choke',))

    def _apply_pending(self):
        fs, st = self._fs, self._is
        for ev in self._pending:
            if ev[0] == 'choke':
                for stage, lvl, slope in ((K.I_OSTAGE, K.S_OLVL, K.S_OSLOPE),
                                          (K.I_NSTAGE, K.S_NLVL, K.S_NSLOPE)):
                    if st[stage] != 4:
                        fs[slope] = max(fs[lvl], 1e-6) * self.d.release_slope
                        st[stage] = 3
            else:
                self._do_trigger(ev[1])
        self._pending = []

    def _do_trigger(self, velocity):
        d = self.d
        p = self.params
        fs, st = self._fs, self._is
        fs[K.S_G:K.S_G + 6] = self._p[K.P_GAINS:K.P_GAINS + 6]
        osc_vel = velocity_gain(velocity, p.osc_vel) * OSC_GAIN
        noise_vel = velocity_gain(velocity, p.noise_vel)
        fs[K.S_OSC_VEL] = osc_vel
        fs[K.S_NOISE_VEL] = noise_vel
        am = d.mod_a * velocity_factor(velocity, p.mod_vel)
        fs[K.S_A] = abs(am) * d.mod_scale * am
        if d.osc_atk_n > 0.0:
            st[K.I_OSTAGE] = 0
            fs[K.S_OSLOPE] = max(fs[K.S_OLVL], 1e-6) * (-1.0 / d.osc_fade_n)
        else:
            fs[K.S_OLVL] = osc_vel
            st[K.I_OSTAGE] = 2
            K.reset_osc(self._p, self._ip, fs, st, self._mod_rng)
        if d.noise_env == NENV_MOD or d.n_atk_n <= 0.0:
            fs[K.S_NLVL] = noise_vel
            st[K.I_NSTAGE] = 2 if d.n_atk_n <= 0.0 else 1
            st[K.I_NCOUNT] = 0
        else:
            st[K.I_NSTAGE] = 0
            fs[K.S_NSLOPE] = max(fs[K.S_NLVL], 1e-6) * (-1.0 / d.n_fade_n)

    # ---------------------------------------------------------------- render
    def _render_blocks(self, nb):
        out = np.empty((nb * 4, 2))
        K.render(self._p, self._ip, self._fs, self._is, self._rng, self._mod_rng,
                 nb, self.clock // 4, out)
        self.clock += nb * 4
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
