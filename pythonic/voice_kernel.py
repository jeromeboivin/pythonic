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

The partials section (decaying partials plus noise bands) keeps its own arrays:
`sp` (derived constants), `ss` (state) and `su` (one noise generator per band),
laid out as described at `_section_run`.
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
# metal section (used only while it is on)
P_M_INC = 37            # 6 square-wave phase increments (cycles/sample, 0 = silent)
P_M_B1 = 43             # band 1 band-pass, 3 values
P_M_A1 = 46             # 3 values
P_M_B2 = 49             # band 2 band-pass, 3 values
P_M_A2 = 52             # 3 values
P_M_HB = 55             # high-pass, 3 values
P_M_HA = 58             # 3 values
P_M_DCY = 61            # 2 per-sample band decay coefficients
P_M_LVL = 63            # 2 band levels
P_M_LSTEP = 65          # 2 per-sample steps of the linear band envelopes
P_M_CURVE = 67          # 0 exponential .. 1 linear band envelopes
P_M_DRIVE = 68          # band saturation gain (0 = linear)
P_MOD_HOLD = 69         # blocks the decay pitch modulation holds before it decays
# partials section (used only while it sounds)
P_PS_OUT_L = 70         # output gains of the section (level, pan)
P_PS_OUT_R = 71
P_PS_CUT = 72           # samples after the hit where the cut starts (0 = no cut)
P_PS_FADE = 73          # samples of the cut's fade (-60 dB at its end, then silence)
P_PS_REL = 74           # per-sample step of the release (choke)
P_PS_END = 75           # a band gain under this has ended
P_PS_RISE = 76          # samples the section fades in over from the hit (0 = at once)
# oscillator 2 (used only while its balance is above 0)
P_O2_GAIN = 77          # its gain, of the balance
P_O2_RATIO = 78         # its pitch, as a ratio of the oscillator's
P_O2_Q = 79             # per-sample factor of its decay on top of the oscillator envelope's
P_O2_OSC = 80           # the oscillator's own gain, of the balance
NUM_P = 81

# ---------------------------------------------------------------- int parameters
IP_WAVE = 0
IP_MOD_MODE = 1
IP_NOISE_ENV = 2
IP_STEREO = 3
IP_MOD_HOLD8 = 4
IP_N_BURSTS = 5
IP_METAL = 6            # 1 when the metal section is on
IP_PS = 7               # 1 when the partials section is on (it then replaces osc, noise and metal)
IP_PS_N = 8             # partials
IP_PS_NB = 9            # noise bands
IP_PS_NSEC = 10         # second-order sections a noise band has (4: 24 dB an octave each side, 8: 48)
IP_PS_CLICK = 11        # 1 when the noise bands have click envelopes
IP_O2_WAVE = 12         # oscillator 2's wave
NUM_IP = 13

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
S_M_PH = 33             # 6 metal oscillator phases
S_M_ENV = 39            # 2 band envelopes
S_M_REL = 41            # 2 band release steps (choke)
S_M_VEL = 43            # metal velocity gain
S_M_Z1 = 44             # band 1 filter state, 2 values
S_M_Z2 = 46             # band 2 filter state, 2 values
S_M_ZH = 48             # high-pass state, 2 values
S_M_LIN = 50            # 2 linear band envelopes (S_M_ENV holds the exponential ones)
S_PHOLD = 52            # blocks of pitch-modulation hold left at the segment start
S_PS_VEL = 53           # partials section velocity gain
S_PS_REL = 54           # its release gain (1 until a choke)
S_PS_DRIFT = 55         # pitch drift its rotations were computed for (0 = not yet)
S_O2_PH = 56            # oscillator 2 phase
S_O2_DEC_X = 57         # its decimator states
S_O2_DEC_Y = 58
S_O2_Q = 59             # its decay factor since the oscillator's restart
NUM_S = 60

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
I_MSTAGE = 10           # metal section: 0 silent, 1 decaying, 2 released (choke)
I_PSTAGE = 11           # partials section: 0 silent, 1 sounding, 2 released (choke)
I_PS_T = 12             # samples since its hit
I_PS_M = 13             # partials still sounding (the first ones: they are sorted by length)
NUM_I = 14

MOD_DECAY, MOD_SINE, MOD_NOISE = 0, 1, 2
NENV_EXP, NENV_LINEAR, NENV_MOD = 0, 1, 2

SINE_AMP = 0.8164966
TRI_HF_AMP = 0.8
SAW_HF_AMP = 0.58
DEC_A = 0.07314892858
METAL_DUTY = 0.49       # duty cycle of the metal square waves (measured on a TR-606: 0.487-0.495)
METAL_END = 1e-5        # a band envelope this low has ended
METAL_BIAS = 0.35       # asymmetry of the band saturation: even products, difference tones
METAL_SCALE = 3.0       # band-pass output that reaches the saturation's knee at drive 0
TWO_PI = 2.0 * np.pi
PS_CHUNK = 64           # partials that have ended drop out on this grid of samples after the hit
PS_SECTIONS = 4         # second-order sections of a noise band at 24 dB an octave (two high-pass, two low-pass)
PS_UNIFORM = 1.0 / 9007199254740992.0   # 2**-53
SQRT3 = 1.7320508075688772              # uniform -1..1 to unit variance


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
    fs[S_O2_PH] = 0.125
    fs[S_O2_DEC_X] = 0.0
    fs[S_O2_DEC_Y] = 0.0
    fs[S_O2_Q] = 1.0
    fs[S_LFO] = 0.0
    fs[S_PENV] = 1.0
    fs[S_PHOLD] = p[P_MOD_HOLD]
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
        k = j - fs[S_PHOLD]     # blocks into the decay, after the hold
        env = fs[S_PENV] * dc ** (8.0 * k) if k > 0.0 else fs[S_PENV]
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
        k = nseg - fs[S_PHOLD]
        fs[S_PHOLD] = max(-k, 0.0)
        if ist[I_SEGCUT]:
            fs[S_PENV] = 0.0
        elif k > 0.0:
            fs[S_PENV] = fs[S_PENV] * p[P_MOD_DC] ** (8.0 * k)
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
    gain = p[P_O2_GAIN]
    if gain > 0.0:
        # oscillator 2: the oscillator's increments times its ratio, its own phase and
        # decimator, under its decay factor, balanced against the oscillator
        osc = p[P_O2_OSC]
        wave2 = ip[IP_O2_WAVE]
        ratio = p[P_O2_RATIO]
        max_inc = p[P_MAX_INC]
        ph = fs[S_O2_PH]
        q = fs[S_O2_Q]
        qs = p[P_O2_Q]
        for k in range(4):
            we = min(w[2 * k] * ratio, max_inc)
            xe = _wave(wave2, ph, we)
            ph = (ph + we) % 1.0
            wo = min(w[2 * k + 1] * ratio, max_inc)
            xo = _wave(wave2, ph, wo)
            ph = (ph + wo) % 1.0
            u = DEC_A * (xe + fs[S_O2_DEC_X]) + xo + xe
            y = u + a2 * fs[S_O2_DEC_Y]
            fs[S_O2_DEC_X] = xo
            fs[S_O2_DEC_Y] = y
            out[k] = osc * out[k] + gain * q * y
            q *= qs
        fs[S_O2_PH] = ph
        fs[S_O2_Q] = q


# ---------------------------------------------------------------- metal section
@njit(cache=True, nogil=True)
def _blep(t, dt):
    """PolyBLEP correction of a unit step at phase 0, for phase t and increment dt."""
    if t < dt:
        t = t / dt
        return t + t - t * t - 1.0
    if t > 1.0 - dt:
        t = (t - 1.0) / dt
        return t * t + t + t + 1.0
    return 0.0


@njit(cache=True, nogil=True)
def _saturate(u, g):
    """Asymmetric soft saturation of gain g (no change at g == 0), 0 at 0, slope 1 there for
    small g: the swing of a transistor VCA, whose products fill the lows and mids."""
    if g == 0.0:
        return u
    v = u / METAL_SCALE
    return (math.tanh(g * (v + METAL_BIAS)) - math.tanh(g * METAL_BIAS)) / g * METAL_SCALE


@njit(cache=True, nogil=True)
def _metal_sample(p, fs, ist):
    """One sample of the metal section: six free-running squares into two band-passes,
    each saturated (drive) then under its own decay envelope, summed into a high-pass. A band
    envelope blends
    an exponential decay (-60 dB at the decay time) and a linear one (0 at it) by the curve."""
    st = ist[I_MSTAGE]
    if st == 0:
        return 0.0
    drift = fs[S_DRIFT]
    x = 0.0
    for i in range(6):
        dt = p[P_M_INC + i] * drift
        if dt <= 0.0:
            continue
        ph = fs[S_M_PH + i]
        v = 1.0 if ph < METAL_DUTY else -1.0
        x += v + _blep(ph, dt) - _blep((ph + 1.0 - METAL_DUTY) % 1.0, dt)
        ph += dt
        fs[S_M_PH + i] = ph - 1.0 if ph >= 1.0 else ph
    c = p[P_M_CURVE]
    x1 = fs[S_M_ENV]
    x2 = fs[S_M_ENV + 1]
    l1 = fs[S_M_LIN]
    l2 = fs[S_M_LIN + 1]
    e1 = x1 * (1.0 - c) + l1 * c
    e2 = x2 * (1.0 - c) + l2 * c
    g = p[P_M_DRIVE]
    y = (_saturate(_biquad(p, P_M_B1, P_M_A1, x, fs, S_M_Z1), g) * e1 * p[P_M_LVL]
         + _saturate(_biquad(p, P_M_B2, P_M_A2, x, fs, S_M_Z2), g) * e2 * p[P_M_LVL + 1])
    y = _biquad(p, P_M_HB, P_M_HA, y, fs, S_M_ZH) * fs[S_M_VEL]
    if st == 1:
        x1 *= p[P_M_DCY]
        x2 *= p[P_M_DCY + 1]
        l1 = max(l1 - p[P_M_LSTEP], 0.0)
        l2 = max(l2 - p[P_M_LSTEP + 1], 0.0)
    else:
        x1 = max(x1 - fs[S_M_REL], 0.0)
        x2 = max(x2 - fs[S_M_REL + 1], 0.0)
        l1 = x1
        l2 = x2
    fs[S_M_ENV] = x1
    fs[S_M_ENV + 1] = x2
    fs[S_M_LIN] = l1
    fs[S_M_LIN + 1] = l2
    if e1 * p[P_M_LVL] < METAL_END and e2 * p[P_M_LVL + 1] < METAL_END:
        for i in range(2):
            fs[S_M_ENV + i] = 0.0
            fs[S_M_LIN + i] = 0.0
        ist[I_MSTAGE] = 0
    return y


# ---------------------------------------------------------------- partials section
@njit(cache=True, nogil=True, fastmath=True)
def _partials_loop(c, s, re, im, m, buf, i0, i1):
    """Advance the first m partials (complex one-pole rotations) over buf[i0:i1], adding
    their real parts: a partial at amplitude A, phase ph sounds A r**t cos(w t + ph)."""
    for i in range(i0, i1):
        acc = 0.0
        for k in range(m):
            a = re[k]
            b = im[k]
            re[k] = c[k] * a - s[k] * b
            im[k] = s[k] * a + c[k] * b
            acc += a
        buf[i] += acc


@njit(cache=True, nogil=True, fastmath=True)
def _bands_loop(bc, bz, bg, bd, cg, cd, clicks, su, nb, nsec, buf, i0, i1):
    """Noise bands over buf[i0:i1]: each band its own white noise (xorshift64) through
    nsec second-order sections (bc: b0 b1 b2 a1 a2 each; bz: 2 states each) under its
    gain bg, which decays by bd a sample, plus (with clicks) its click gain cg, decaying by
    cd."""
    s13 = np.uint64(13)
    s7 = np.uint64(7)
    s17 = np.uint64(17)
    s11 = np.uint64(11)
    for i in range(i0, i1):
        acc = 0.0
        for j in range(nb):
            x = su[j]
            x ^= x << s13
            x ^= x >> s7
            x ^= x << s17
            su[j] = x
            y = (float(x >> s11) * PS_UNIFORM * 2.0 - 1.0) * SQRT3
            for q in range(nsec):
                o = 5 * (nsec * j + q)
                z = 2 * (nsec * j + q)
                u = y
                y = bc[o] * u + bz[z]
                bz[z] = bc[o + 1] * u - bc[o + 3] * y + bz[z + 1]
                bz[z + 1] = bc[o + 2] * u - bc[o + 4] * y
            if clicks:
                acc += y * (bg[j] + cg[j])
                cg[j] *= cd[j]
            else:
                acc += y * bg[j]
            bg[j] *= bd[j]
        buf[i] += acc


@njit(cache=True, nogil=True)
def _section_run(p, ip, fs, ist, sp, ss, su, n, out):
    """n samples of the partials section, added to out at its own output gains.

    sp (constants, n partials, nb bands of nsec sections): w (rad/sample), r (decay a
    sample), life (samples a partial lasts), re0, im0 (its start), 5 coefficients for each
    filter section of a band, band decays, band start gains, click decays, click start
    gains.  ss (state): re, im, c, s (the rotations at the current drift), 2 states for each
    filter section of a band, band gains, click gains.  Partials are sorted by life,
    longest first; the
    ones past their life drop out on a PS_CHUNK grid of samples after the hit, so the
    output does not depend on how the audio is split into calls."""
    np_ = ip[IP_PS_N]
    nb = ip[IP_PS_NB]
    nsec = ip[IP_PS_NSEC]
    clicks = ip[IP_PS_CLICK] != 0
    o_r = np_
    o_life = 2 * np_
    o_bc = 5 * np_
    o_bd = o_bc + 5 * nsec * nb
    o_cd = o_bd + 2 * nb
    s_c = 2 * np_
    s_s = 3 * np_
    s_bz = 4 * np_
    s_bg = s_bz + 2 * nsec * nb
    s_cg = s_bg + nb
    drift = fs[S_DRIFT]
    if fs[S_PS_DRIFT] != drift:
        for k in range(np_):
            w = sp[k] * drift
            r = sp[o_r + k]
            ss[s_c + k] = r * math.cos(w)
            ss[s_s + k] = r * math.sin(w)
        fs[S_PS_DRIFT] = drift
    buf = np.zeros(n)
    t0 = ist[I_PS_T]
    i = 0
    m = ist[I_PS_M]
    ended = False
    while i < n:
        t = t0 + i
        if t % PS_CHUNK == 0:
            while m > 0 and sp[o_life + m - 1] <= t:
                m -= 1
            if m == 0:
                ended = True
                for j in range(nb):
                    if ss[s_bg + j] >= p[P_PS_END] or (clicks and ss[s_cg + j] >= p[P_PS_END]):
                        ended = False
                        break
                if ended:
                    break
        i1 = min(n, i + PS_CHUNK - t % PS_CHUNK)
        if m > 0:
            _partials_loop(ss[s_c:s_c + np_], ss[s_s:s_s + np_], ss[0:np_], ss[np_:2 * np_], m, buf, i, i1)
        if nb > 0:
            _bands_loop(sp[o_bc:o_bd], ss[s_bz:s_bg], ss[s_bg:s_bg + nb], sp[o_bd:o_bd + nb],
                        ss[s_cg:s_cg + nb], sp[o_cd:o_cd + nb], clicks, su, nb, nsec, buf, i, i1)
        i = i1
    ist[I_PS_M] = m
    vel = fs[S_PS_VEL]
    gl = p[P_PS_OUT_L] * vel
    gr = p[P_PS_OUT_R] * vel
    cut = p[P_PS_CUT]
    fade = p[P_PS_FADE]
    rise = p[P_PS_RISE]
    rel = fs[S_PS_REL]
    released = ist[I_PSTAGE] == 2
    step = p[P_PS_REL]
    g = 1.0
    for k in range(n):
        g = rel
        if released:
            rel = max(rel - step, 0.0)
        if rise > 0.0 and t0 + k < rise:
            g *= (t0 + k) / rise
        if cut > 0.0:
            u = t0 + k - cut
            if u >= fade:
                g = 0.0
            elif u >= 0.0:
                g *= math.exp(-6.907755279 * u / fade)
        y = buf[k] * g
        out[k, 0] += y * gl
        out[k, 1] += y * gr
    fs[S_PS_REL] = rel
    t1 = t0 + n
    ist[I_PS_T] = t1
    if ended or (released and rel <= 0.0) or (cut > 0.0 and t1 >= cut + fade):
        ist[I_PSTAGE] = 0


# ---------------------------------------------------------------- render
@njit(cache=True, nogil=True)
def _ring_eq(p, ip, fs, ist, out, n):
    """Let the EQ ring out on silence once both envelopes have ended."""
    gl = fs[S_G + 3]
    gr = fs[S_G + 4]
    ge = fs[S_G + 5]
    chans = 2 if ip[IP_STEREO] else 1
    tail = np.zeros((n, 2))
    level = 0.0
    for c in range(chans):
        iz = S_EQ_ZI + 2 * c
        for i in range(n):
            tail[i, c] = ge * _biquad(p, P_EQ_B, P_EQ_A, 0.0, fs, iz)
        level = max(level, abs(fs[iz]), abs(fs[iz + 1]))
    for i in range(n):
        if chans == 1:
            tail[i, 1] = tail[i, 0]
        out[i, 0] += tail[i, 0] * gl
        out[i, 1] += tail[i, 1] * gr
    if level < 1e-7 or ge == 0.0:
        for i in range(4):
            fs[S_EQ_ZI + i] = 0.0
        ist[I_EQ_TAIL] = 0


@njit(cache=True, nogil=True)
def render(p, ip, fs, ist, rng, mod_rng, nb, blk0, out, sp, ss, su):
    """Render `nb` 4-sample blocks starting at absolute block `blk0` into out[:nb*4]."""
    n = nb * 4
    for i in range(n):
        out[i, 0] = 0.0
        out[i, 1] = 0.0
    if ist[I_PSTAGE] != 0:
        _section_run(p, ip, fs, ist, sp, ss, su, n, out)
    if ist[I_OSTAGE] == 4 and ist[I_NSTAGE] == 4 and ist[I_MSTAGE] == 0:
        if ist[I_EQ_TAIL]:
            _ring_eq(p, ip, fs, ist, out, n)
        return

    metal_on = ist[I_MSTAGE] != 0
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
            if metal_on:
                s0 = s0 + _metal_sample(p, fs, ist) * gd
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
            out[i, 0] += sig[0] * gl
            out[i, 1] += sig[chans - 1] * gr
    if osc_alive and seg_j > 0:
        _segment_end(p, ip, fs, ist, seg_j)
    if eq_on:
        ist[I_EQ_TAIL] = 1
    if smooth:
        f = decay ** nb
        for i in range(6):
            fs[S_G + i] = p[P_GAINS + i] + (g0[i] - p[P_GAINS + i]) * f
