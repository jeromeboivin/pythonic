"""
Compiled render loop of the drum voice.

DrumVoice (voice.py) keeps the derived patch parameters and the voice state in
flat arrays; `render` advances that state over a run of 4-sample blocks.  It
runs without the GIL, so the audio thread does not wait on the GUI thread
while it renders.

Envelope runs and oscillator segments restart at the start of every render
call (and at an oscillator reset), and every value is computed from the start
of its run or segment, so the output does not depend on how the audio is split
into calls once the block grid is fixed.
"""

import math

import numpy as np
from numba import njit

# ---------------------------------------------------------------- float parameters
P_INC0 = 0
P_MAX_INC = 1
P_OSC_ATK_C = 2
P_OSC_DCY_C = 3
P_MOD_DC = 4
P_MOD_B0 = 5
P_MOD_B4 = 6
P_MOD_LPC = 7
P_NF_B = 8              # 3 values
P_NF_A = 11             # 3 values
P_NOISE_GAIN = 14
P_N_ATK_C = 15
P_N_DCY_C = 16
P_N_LIN_UP = 17
P_N_LIN_DN = 18
P_N_BURST_C = 19
P_GAINS = 20            # 6 gain targets: osc mix, noise mix, drive, out L, out R, eq amount
P_SH_D = 26
P_SH_K = 27
P_SH_DC = 28
P_EQ_B = 29             # 3 values
P_EQ_A = 32             # 3 values
P_SMOOTH = 35
P_RELEASE = 36
NUM_P = 37

# ---------------------------------------------------------------- int parameters
IP_WAVE = 0
IP_MOD_MODE = 1
IP_NOISE_ENV = 2
IP_STEREO = 3
IP_MOD_HOLD8 = 4
IP_N_BURSTS = 5
NUM_IP = 6

# ---------------------------------------------------------------- float state
S_PHASE = 0
S_LFO = 1
S_PENV = 2
S_HOLD = 3
S_A = 4
S_DEC_X = 5
S_DEC_Y = 6
S_OSC_VEL = 7
S_NOISE_VEL = 8
S_OLVL = 9
S_OSLOPE = 10
S_ORUN = 11             # level at the start of the current osc envelope run
S_NLVL = 12
S_NSLOPE = 13
S_NRUN = 14
S_NLP = 15              # 2 random-mod lowpass states
S_NF_ZI = 17            # 2 channels x 2
S_EQ_ZI = 21            # 2 channels x 2
S_G = 25                # 6 smoothed gains
S_DRIFT = 31
S_ACC = 32              # phase increments summed since the segment start
NUM_S = 33

# ---------------------------------------------------------------- int state
I_PMODE4 = 0
I_SMODE = 1
I_OSTAGE = 2
I_NSTAGE = 3
I_NCOUNT = 4
I_EQ_TAIL = 5
I_ORUNK = 6             # samples into the current osc envelope run
I_NRUNK = 7
I_LATCH = 8             # decay-mod latch inside the current segment
I_SEGCUT = 9
NUM_I = 10

MOD_DECAY, MOD_SINE, MOD_NOISE = 0, 1, 2
NENV_EXP, NENV_LINEAR, NENV_MOD = 0, 1, 2

SINE_AMP = 0.8164966
TRI_HF_AMP = 0.8
SAW_HF_AMP = 0.58
DEC_A = 0.07314892858
TWO_PI = 2.0 * np.pi


# ---------------------------------------------------------------- helpers
@njit(cache=True, nogil=True)
def pow2_dec(x):
    """2**x approximation used by the decaying / random pitch modulation."""
    p = (((x * 4.074710159e-05 + 0.0009640406934) * x + 0.01495699305) * x + 0.1719395369) * x + 1.0
    p = p * p
    return p * p


@njit(cache=True, nogil=True)
def pow2_sin(x):
    """2**x approximation used by the sine pitch modulation."""
    p = (((x * 3.834385643e-05 + 0.0008906660951) * x + 0.01501070894) * x + 0.1732061803) * x + 1.0
    p = p * p
    return p * p


@njit(cache=True, nogil=True)
def _cos_fine(ph):
    x = ph if ph <= 0.5 else ph - 1.0
    x2 = x * x
    return (((x2 * 46.29878998 - 82.69654846) * x2 + 64.71379089) * x2 - 19.73277473) * x2 + 0.9999709725


@njit(cache=True, nogil=True)
def _cos_coarse(ph):
    x = ph if ph < 0.5 else ph - 1.0
    x2 = x * x
    return (((x2 * 52.90157318 - 85.04795837) * x2 + 64.88814545) * x2 - 19.72719574) * x2 + 0.9996777177


@njit(cache=True, nogil=True)
def sine_mod_granularity(A, b0, inc0, max_inc):
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


@njit(cache=True, nogil=True)
def _wave(wave, ph, w):
    """Band-limited waveform at phase `ph` (cycles) and increment `w` (cycles/sample @2x)."""
    if wave == 0:
        return SINE_AMP * math.sin(TWO_PI * ph)
    if wave == 1:
        if w > 0.1:
            return TRI_HF_AMP * math.sin(TWO_PI * ph)
        x = 4.0 * ph
        if x < 1.0:
            v = x
        elif x < 3.0:
            v = 2.0 - x
        else:
            v = x - 4.0
        if w > 1.0 / 60.0:
            W = 1.2 - 12.0 * w
            norm = 1.0 / (W * 0.2 + 0.8)
            d = 1.0 - abs(v)
            span = 1.0 - W
            if d < span:
                z = d / span
                z2 = z * z
                cap = ((z2 * 0.1431494355 - 0.7790188193) * z2 + 0.6363011003) * span + W
                return (cap if v >= 0 else -cap) * norm
            return v * norm
        return v
    # falling sawtooth
    if w > 0.1034082696:
        return SAW_HF_AMP * math.sin(TWO_PI * ph)
    if w <= 0.00125:
        return 1.0 - 2.0 * ph
    D = ((34.40639877 - w * 239.1292725) * w + 1.416769266) * w
    if w > 0.04118342325:
        amp = 1.0 - D * D * D * 26.87999916
        a4 = amp * 4.0 * D
    else:
        amp = 1.0
        a4 = 0.0
    if ph < D or ph > 1.0 - D:
        xe = ph / D if ph < D else (ph - 1.0) / D
        return ((xe * xe * 0.06132304668 - 0.6244748235) * xe * xe + 1.566493511) * xe * amp
    xl = (ph - 0.5) / (D - 0.5)
    return ((xl * xl * 0.06132304668 - 0.6244748235) * xl * xl + 1.566493511) * xl * a4 + (amp - a4) * xl


@njit(cache=True, nogil=True)
def _shape(x, d, k, dc):
    if d == 0.0:
        return x
    c = x - 0.3333333433
    if d >= 1.0:
        c = min(max(c, -1.0), 1.0)
        return ((abs(c) - c * c) + 1.0) * c + 0.407407403
    c = min(max(c, -k), k)
    return ((abs(c) - c * c) * d + 1.0) * c - dc


@njit(cache=True, nogil=True)
def _biquad(p, ib, ia, x, zi, iz):
    """Direct form II transposed second-order section (a0 == 1)."""
    y = zi[iz] + p[ib] * x
    zi[iz] = zi[iz + 1] + p[ib + 1] * x - p[ia + 1] * y
    zi[iz + 1] = p[ib + 2] * x - p[ia + 2] * y
    return y


@njit(cache=True, nogil=True)
def _expo(log_start, log_coef, k):
    """start * coef**k, evaluated as exp(log(start) + log(coef) * k) (0 when start <= 0)."""
    if log_start == -np.inf:
        return 0.0
    return math.exp(min(log_start + log_coef * k, 50.0))


@njit(cache=True, nogil=True)
def _log(x):
    return math.log(x) if x > 0.0 else -np.inf


# ---------------------------------------------------------------- oscillator reset
@njit(cache=True, nogil=True)
def _sine_hold_update(p, fs):
    c = _cos_coarse(fs[S_LFO])
    fs[S_HOLD] = min(p[P_INC0] * fs[S_DRIFT] * pow2_sin(fs[S_A] * c), p[P_MAX_INC])
    fs[S_LFO] = (fs[S_LFO] + p[P_MOD_B0] * 128.0) % 1.0


@njit(cache=True, nogil=True)
def reset_osc(p, ip, fs, ist, mod_rng):
    """Restart the oscillator (phase, decimator, pitch modulation)."""
    fs[S_PHASE] = 0.125
    fs[S_DEC_Y] = 0.0
    fs[S_LFO] = 0.0
    fs[S_PENV] = 1.0
    ist[I_PMODE4] = 0
    r = mod_rng.uniform(-1.0, 1.0)
    fs[S_NLP] = r
    fs[S_NLP + 1] = r
    sine = ip[IP_MOD_MODE] == MOD_SINE
    ist[I_SMODE] = sine_mod_granularity(fs[S_A], p[P_MOD_B0], p[P_INC0], p[P_MAX_INC]) if sine else 0
    if sine and ist[I_SMODE] == 4:
        _sine_hold_update(p, fs)
    else:
        fs[S_HOLD] = p[P_INC0]


# ---------------------------------------------------------------- envelopes
@njit(cache=True, nogil=True)
def _osc_env_block(p, fs, ist, out):
    """One block of the oscillator envelope; True when the oscillator restarts after it."""
    st = ist[I_OSTAGE]
    if st == 4:
        for i in range(4):
            out[i] = 0.0
        return False
    run0 = fs[S_ORUN]
    k = ist[I_ORUNK]
    vel = fs[S_OSC_VEL]
    lr = _log(run0) if st == 1 or st == 2 else 0.0
    lc = math.log(p[P_OSC_ATK_C] if st == 1 else p[P_OSC_DCY_C])
    v = 0.0
    for i in range(4):
        k += 1
        if st == 0 or st == 3:
            v = run0 + fs[S_OSLOPE] * k
            out[i] = max(v, 0.0)
        elif st == 1:
            v = _expo(lr, lc, k)
            out[i] = min(v, vel)
        else:
            v = _expo(lr, lc, k)
            out[i] = v
    fs[S_OLVL] = v
    ist[I_ORUNK] = k
    reset = False
    ended = False
    if st == 0 or st == 3:
        if v <= 1e-13:
            ended = True
            if st == 0:
                ist[I_OSTAGE] = 1
                fs[S_OLVL] = max(vel * 0.001, 1e-6)
                reset = True
            else:
                ist[I_OSTAGE] = 4
                fs[S_OLVL] = 0.0
    elif st == 1:
        if v >= vel:
            ended = True
            ist[I_OSTAGE] = 2
            fs[S_OLVL] = vel
    elif v <= 1e-6:
        ended = True
        fs[S_OSLOPE] = max(v, 1e-6) * p[P_RELEASE]
        ist[I_OSTAGE] = 3
    if ended:
        fs[S_ORUN] = fs[S_OLVL]
        ist[I_ORUNK] = 0
    return reset


@njit(cache=True, nogil=True)
def _noise_env_block(p, ip, fs, ist, out):
    st = ist[I_NSTAGE]
    if st == 4:
        for i in range(4):
            out[i] = 0.0
        return
    mode = ip[IP_NOISE_ENV]
    run0 = fs[S_NRUN]
    k = ist[I_NRUNK]
    vel = fs[S_NOISE_VEL]
    lr = _log(run0)
    if st == 1 and mode == NENV_MOD:
        lc = math.log(p[P_N_BURST_C])
    elif st == 1:
        lc = math.log(p[P_N_ATK_C])
    else:
        lc = math.log(p[P_N_DCY_C])
    v = 0.0
    for i in range(4):
        k += 1
        if st == 0 or st == 3:
            v = run0 + fs[S_NSLOPE] * k
            out[i] = max(v, 0.0)
        elif st == 1 and mode == NENV_MOD:
            v = _expo(lr, lc, k)
            out[i] = v
        elif st == 1:
            if mode == NENV_LINEAR:
                v = run0 + p[P_N_LIN_UP] * vel * k
            else:
                v = _expo(lr, lc, k)
            out[i] = min(v, vel)
        elif mode == NENV_LINEAR:
            v = run0 + -p[P_N_LIN_DN] * vel * k
            out[i] = max(v, 0.0)
        else:
            v = _expo(lr, lc, k)
            out[i] = v
    fs[S_NLVL] = v
    ist[I_NRUNK] = k
    ended = False
    if st == 0 or st == 3:
        if v <= 1e-13:
            ended = True
            if st == 0:
                ist[I_NSTAGE] = 1
                fs[S_NLVL] = max(vel * 0.001, 1e-6) if mode == NENV_EXP else 0.0
            else:
                ist[I_NSTAGE] = 4
                fs[S_NLVL] = 0.0
    elif st == 1 and mode == NENV_MOD:
        if v <= vel * 0.06309573352:
            ended = True
            ist[I_NCOUNT] += 1
            fs[S_NLVL] = vel
            if ist[I_NCOUNT] >= ip[IP_N_BURSTS]:
                ist[I_NSTAGE] = 2
    elif st == 1:
        if v >= vel:
            ended = True
            ist[I_NSTAGE] = 2
            fs[S_NLVL] = vel
    elif mode == NENV_LINEAR:
        if v <= 1e-13:
            ended = True
            ist[I_NSTAGE] = 4
            fs[S_NLVL] = 0.0
    elif v <= 1e-6:
        ended = True
        fs[S_NSLOPE] = max(v, 1e-6) * p[P_RELEASE]
        ist[I_NSTAGE] = 3
    if ended:
        fs[S_NRUN] = fs[S_NLVL]
        ist[I_NRUNK] = 0


# ---------------------------------------------------------------- oscillator
@njit(cache=True, nogil=True)
def _increments(p, ip, fs, ist, mod_rng, j, blk, w):
    """Phase increments (8 oversampled samples) for block `j` of the segment, absolute block `blk`."""
    inc0 = p[P_INC0] * fs[S_DRIFT]
    max_inc = p[P_MAX_INC]
    A = fs[S_A]
    if abs(A) <= 1e-4:
        for i in range(8):
            w[i] = min(inc0, max_inc)
        return
    mode = ip[IP_MOD_MODE]
    b0 = p[P_MOD_B0]
    if mode == MOD_DECAY:
        dc = p[P_MOD_DC]
        env = fs[S_PENV] * dc ** (8.0 * j)
        free = min(inc0 * pow2_dec(A * env), max_inc)
        # 64-sample (16-block) housekeeping: mode-4 latch and env cut-off
        if blk % 16 == 0:
            e = env
            cut = e < 0.001
            if cut:
                e = 0.0
            x = A * 0.6931471825 * e
            m4 = abs(math.exp(x * dc) - math.exp(x)) * inc0 < 1e-7
            ist[I_LATCH] = 1 if m4 else 0
            ist[I_SEGCUT] = 1 if cut else 0
            if m4:
                fs[S_HOLD] = min(inc0 * pow2_dec(A * e), max_inc)
        v = fs[S_HOLD] if ist[I_LATCH] else free
        for i in range(8):
            w[i] = v
        return
    if mode == MOD_SINE:
        sm = ist[I_SMODE]
        if sm == 1 or sm == 2:
            lfo = fs[S_LFO]
            for i in range(8):
                k = j * 8 + i
                if sm == 2:
                    k = (k // 2) * 2
                ph = (lfo + k * b0) % 1.0
                w[i] = min(inc0 * pow2_sin(A * _cos_fine(ph)), max_inc)
            return
        if sm == 3:
            ph = (fs[S_LFO] + j * 8.0 * b0) % 1.0
            v = min(inc0 * pow2_sin(A * _cos_coarse(ph)), max_inc)
        else:
            if blk % 16 == 0:
                _sine_hold_update(p, fs)
            v = fs[S_HOLD]
        for i in range(8):
            w[i] = v
        return
    # random (noise) modulation: two cascaded one-pole lowpasses on uniform noise
    step = 8 if ip[IP_MOD_HOLD8] else 2
    c = p[P_MOD_LPC]
    v = 0.0
    for i in range(8):
        if i % step == 0:
            u = mod_rng.uniform(-1.0, 1.0) * p[P_MOD_B4]
            fs[S_NLP] = (1.0 - c) * fs[S_NLP] + c * u
            fs[S_NLP + 1] = (1.0 - c) * fs[S_NLP + 1] + c * fs[S_NLP]
            m = min(max(fs[S_NLP + 1], -2.0), 2.0)
            v = min(inc0 * pow2_dec(A * m), max_inc)
        w[i] = v


@njit(cache=True, nogil=True)
def _segment_end(p, ip, fs, ist, nseg):
    """Fold the per-segment pitch-modulation state after `nseg` blocks."""
    fs[S_PHASE] = (fs[S_PHASE] + fs[S_ACC]) % 1.0
    fs[S_ACC] = 0.0
    if abs(fs[S_A]) <= 1e-4:
        return
    mode = ip[IP_MOD_MODE]
    if mode == MOD_DECAY:
        ist[I_PMODE4] = ist[I_LATCH]
        if ist[I_SEGCUT]:
            fs[S_PENV] = 0.0
        else:
            fs[S_PENV] = fs[S_PENV] * p[P_MOD_DC] ** (8.0 * nseg)
    elif mode == MOD_SINE and 1 <= ist[I_SMODE] <= 3:
        fs[S_LFO] = (fs[S_LFO] + nseg * 8 * p[P_MOD_B0]) % 1.0


@njit(cache=True, nogil=True)
def _segment_start(fs, ist):
    fs[S_ACC] = 0.0
    ist[I_LATCH] = ist[I_PMODE4]
    ist[I_SEGCUT] = 0


@njit(cache=True, nogil=True)
def _osc_block(p, ip, fs, ist, mod_rng, j, blk, w, out):
    """Render one block of the decimated oscillator (4 samples)."""
    _increments(p, ip, fs, ist, mod_rng, j, blk, w)
    wave = ip[IP_WAVE]
    phase0 = fs[S_PHASE]
    acc = fs[S_ACC]
    a2 = DEC_A * DEC_A
    for k in range(4):
        acc += w[2 * k]
        xe = _wave(wave, (phase0 + acc - w[2 * k]) % 1.0, w[2 * k])
        acc += w[2 * k + 1]
        xo = _wave(wave, (phase0 + acc - w[2 * k + 1]) % 1.0, w[2 * k + 1])
        u = DEC_A * (xe + fs[S_DEC_X]) + xo + xe
        y = u + a2 * fs[S_DEC_Y]
        fs[S_DEC_X] = xo
        fs[S_DEC_Y] = y
        out[k] = y
    fs[S_ACC] = acc


# ---------------------------------------------------------------- render
@njit(cache=True, nogil=True)
def _ring_eq(p, ip, fs, ist, out, n):
    """Let the EQ ring out on silence once both envelopes have ended."""
    gl = fs[S_G + 3]
    gr = fs[S_G + 4]
    ge = fs[S_G + 5]
    chans = 2 if ip[IP_STEREO] else 1
    level = 0.0
    for c in range(chans):
        iz = S_EQ_ZI + 2 * c
        for i in range(n):
            out[i, c] = ge * _biquad(p, P_EQ_B, P_EQ_A, 0.0, fs, iz)
        level = max(level, abs(fs[iz]), abs(fs[iz + 1]))
    for i in range(n):
        if chans == 1:
            out[i, 1] = out[i, 0]
        out[i, 0] *= gl
        out[i, 1] *= gr
    if level < 1e-7 or ge == 0.0:
        for i in range(4):
            fs[S_EQ_ZI + i] = 0.0
        ist[I_EQ_TAIL] = 0


@njit(cache=True, nogil=True)
def render(p, ip, fs, ist, rng, mod_rng, nb, blk0, out):
    """Render `nb` 4-sample blocks starting at absolute block `blk0` into out[:nb*4]."""
    n = nb * 4
    for i in range(n):
        out[i, 0] = 0.0
        out[i, 1] = 0.0
    if ist[I_OSTAGE] == 4 and ist[I_NSTAGE] == 4:
        if ist[I_EQ_TAIL]:
            _ring_eq(p, ip, fs, ist, out, n)
        return

    osc_alive = ist[I_OSTAGE] != 4
    noise_on = ist[I_NSTAGE] != 4 and (p[P_GAINS + 1] > 0.0 or fs[S_G + 1] > 0.0)
    chans = 2 if ip[IP_STEREO] else 1
    fs[S_ORUN] = fs[S_OLVL]
    ist[I_ORUNK] = 0
    fs[S_NRUN] = fs[S_NLVL]
    ist[I_NRUNK] = 0

    # gains glide per block towards their targets
    g0 = np.empty(6)
    smooth = False
    for i in range(6):
        g0[i] = fs[S_G + i]
        if abs(g0[i] - p[P_GAINS + i]) > 1e-7:
            smooth = True
    if not smooth:
        for i in range(6):
            g0[i] = p[P_GAINS + i]
            fs[S_G + i] = g0[i]
    decay = 1.0 - p[P_SMOOTH]
    tge = p[P_GAINS + 5]
    eq_on = g0[5] != 0.0
    if smooth and not eq_on:
        for j in range(nb):
            if tge + (g0[5] - tge) * decay ** j != 0.0:
                eq_on = True
                break

    w = np.empty(8)
    oe = np.empty(4)
    ne = np.zeros(4)
    osc = np.zeros(4)
    g = np.empty(6)
    sig = np.empty(2)
    noise_gain = p[P_NOISE_GAIN]
    sh_d = p[P_SH_D]
    sh_k = p[P_SH_K]
    sh_dc = p[P_SH_DC]
    seg_j = 0
    if osc_alive:
        _segment_start(fs, ist)
    for b in range(nb):
        reset = _osc_env_block(p, fs, ist, oe)
        if osc_alive:
            _osc_block(p, ip, fs, ist, mod_rng, seg_j, blk0 + b, w, osc)
            seg_j += 1
            if reset:
                _segment_end(p, ip, fs, ist, seg_j)
                reset_osc(p, ip, fs, ist, mod_rng)
                _segment_start(fs, ist)
                seg_j = 0
        if noise_on:
            _noise_env_block(p, ip, fs, ist, ne)
        if smooth:
            f = decay ** b
            for i in range(6):
                g[i] = p[P_GAINS + i] + (g0[i] - p[P_GAINS + i]) * f
        else:
            for i in range(6):
                g[i] = g0[i]
        go, gn, gd, gl, gr, ge = g[0], g[1], g[2], g[3], g[4], g[5]
        for k in range(4):
            s0 = osc[k] * oe[k] * go * gd
            for c in range(chans):
                s = s0
                if noise_on:
                    u = rng.uniform(-1.0, 1.0) * noise_gain
                    yn = _biquad(p, P_NF_B, P_NF_A, u, fs, S_NF_ZI + 2 * c)
                    s = s + yn * ne[k] * gn * gd
                s = _shape(s, sh_d, sh_k, sh_dc)
                if eq_on:
                    s = s + ge * _biquad(p, P_EQ_B, P_EQ_A, s, fs, S_EQ_ZI + 2 * c)
                sig[c] = s
            i = b * 4 + k
            out[i, 0] = sig[0] * gl
            out[i, 1] = sig[chans - 1] * gr
    if osc_alive and seg_j > 0:
        _segment_end(p, ip, fs, ist, seg_j)
    if eq_on:
        ist[I_EQ_TAIL] = 1
    if smooth:
        f = decay ** nb
        for i in range(6):
            fs[S_G + i] = p[P_GAINS + i] + (g0[i] - p[P_GAINS + i]) * f
