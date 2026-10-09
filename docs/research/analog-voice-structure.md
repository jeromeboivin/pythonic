# Voice structure of the analog drum machines

Research for [#25](https://github.com/jeromeboivin/pythonic/issues/25), part of the map [#23](https://github.com/jeromeboivin/pythonic/issues/23). It feeds [#29](https://github.com/jeromeboivin/pythonic/issues/29), the ticket on the sections the universal channel gains ([ADR 0002](../adr/0002-machine-voices-on-one-superset-channel.md)).

**Question.** For every analog voice in scope, what is the signal structure, and which building blocks does today's universal voice lack to play it?

**Scope.**

- TR-909: the analog voices BD, SD, LT/MT/HT, RS and CP. Its hi-hats, crash and ride are PCM from ROM ([909SN] p.6), so they belong to the sample layer.
- TR-808: every voice. This includes the six-oscillator metal source behind CY, OH, CH and CB.
- TR-606: every voice.
- CR-78: every voice, including the ADD VOICE parts (tambourine, guiro and metal beat).

We describe behaviour and structure, not a circuit simulation. The aim is to name the blocks that a factory drum patch needs, so that fitting to recordings can reach the machine (ADR 0002). Each element maps either onto a block of today's voice or onto a missing block `M1`–`M12`, which are defined in [Missing building blocks](#missing-building-blocks-de-duplicated).

## Summary

- **The 808, 606 and CR-78 drums are pinged resonators, not oscillators.** A trigger pulse rings a bridged-T band-pass, or on the CR-78 a transistor or LC ringing filter. The resonator decays on its own, and its state carries over when the voice is retriggered. Today's sine oscillator with an exponential decay approximates one mode. It cannot give a second or third mode, the retrigger behaviour (no "machine gun"), or the 808 BD's attack retune and pitch sigh. → **M1 resonator section** (the 909 RS needs three).
- **The 909 drums are VCO + VCA + envelope voices, which is closest to today's voice.** The gaps are extra partials and envelope details:
  - The 909 SD needs a second VCO with its own decay, and the toms need two more VCOs, all synced at the trigger and sharing one pitch envelope. → **M2 extra tone oscillators**
  - The 909 BD's click is a pulse generator plus filtered noise, with its own short VCA. → **M3 click generator**
  - The 909 SD has two noise VCAs: low-passed noise whose decay the TONE knob sets, and high-passed noise driven by accent. → **M8 second noise path**
  - The 909 VCOs have linear CV, so the pitch envelope moves in hertz, while today's pitch envelope moves in semitones. The tom pitch envelope also has two time constants. → **M9 envelope extras**
  - The 909 VCOs reset at the trigger to a defined point, while today's oscillator always restarts at 1/8 cycle. → **M10 start phase**
- **The metal voices need a source and a filter chain that today's voice does not have:**
  - Up to six free-running square oscillators at inharmonic fixed ratios. → **M4 metal oscillator bank** (808 CY/OH/CH/CB, 606 CY/OH/CH, CR-78 MB/CB)
  - One to three bands, each a band-pass, a VCA with its own envelope and a high-pass. → **M5 metal band section**
  - Swing-type VCAs whose clipping follows the envelope. → **M7 VCA-stage saturation**
- **Smaller gaps:**
  - A filter on the tonal path: the 808 BD's TONE low-pass, the 808 and 909 RS high-pass, and the 808 CB band-pass. → **M6**
  - Two-slope decays: 808 CB, 808 CY, CR-78 CY. → **M9**
  - A diode clipper before the VCA in the 909 RS. → **M7**
  - The diode waveshapers that turn the 909's ramp and triangle VCOs into near-sines. → **M11** (low priority)
- **Accent.** On the 808, 606 and 909, accent raises the trigger amplitude. That already maps onto today's velocity for level and pitch-mod depth. Accent also reaches the nonlinear parts: the 808 BD's attack retune and pitch sigh, the swing VCAs, the 909 RS clipper, and the 909 SD's high-passed noise. So every new section needs its own velocity sensitivity. On the CR-78, accent is a master-VCA step on the whole mix. That is a kit-level feature, not a channel section. → **M12**
- **Hit-to-hit variance** in these machines comes from four things:
  - free-running metal oscillators, so the phase at the trigger is random;
  - resonator state carried across retriggers;
  - free-running noise;
  - component tolerance between units (±20 % capacitors and ±5 % resistors in the 808 voices, [WAS-BD] §11).

  No source documents per-hit pitch jitter or audible rail sag. Today's `vintage` drift restarts identically on every hit, so it adds no variance between hits.
- **Every block can be off with today's render untouched.** Two conditions apply:
  - the code path is skipped when a section is off, not multiplied by zero;
  - new randomness (metal phases, the second noise path) draws from its own random stream, so the existing noise and pitch-mod streams are consumed exactly as today.

## Sources

Trust: **P** = primary (Roland service notes, peer-reviewed papers). **S** = secondary (reputable published circuit analysis), used only where it is marked. Page numbers are the PDF pages of the archive.org scans.

| ID | Source | Trust | Used for |
|----|--------|-------|----------|
| [808SN] | Roland, *TR-808 Service Notes*, 1st ed., 15 June 1981. [archive.org](https://archive.org/details/synthmanual-roland-tr-808-service-notes) | P | Trigger and accent (p.4), bridged-T and swing-type VCA (p.5), voice descriptions (p.5–6), block diagram (p.7), voice check table: amplitude, frequency, decay (p.14), change information (p.15) |
| [909SN] | Roland, *TR-909 Service Notes*, 15 June 1984. [archive.org](https://archive.org/details/synth_Roland_TR-909_service_notes) | P | Accent and trigger (p.5), circuit descriptions and noise generator (p.6), change information (p.7), block diagram (p.4), waveforms with and without accent (p.9), voicing-board circuit diagram (p.11) |
| [606SN] | Roland, *TR-606 Service Notes*, 6 Jan. 1982. [archive.org](https://archive.org/details/RolandTR606ServiceNotesJAN.61982600Dpi) | P | Specifications (p.1), block diagram (p.3), adjustment, parts and engineering changes (p.8) |
| [78SN] | Roland, *CR-78 Service Notes*, 20 June 1979. [archive.org](https://archive.org/details/synthmanual-roland-cr-78-service-notes) | P | Fade and accent (p.13), VG-11A voicing circuit diagram (p.25–26), voice adjustment table (p.30) |
| [WAS-BD] | K. J. Werner, J. S. Abel, J. O. Smith, "A Physically-Informed, Circuit-Bendable, Digital Model of the Roland TR-808 Bass Drum Circuit", *Proc. DAFx-14*, Erlangen, 2014. [PDF](https://dafx14.fau.de/papers/dafx14_kurt_james_werner_a_physically_informed,_ci.pdf) | P | 808 BD sub-circuits, attack retune, pitch sigh, accent, retrigger behaviour, tolerances |
| [WAS-CY] | K. J. Werner, J. S. Abel, J. O. Smith, "The TR-808 Cymbal: a Physically-Informed, Circuit-Bendable, Digital Model", *Proc. ICMC/SMC 2014*, Athens, pp. 1453–1460. [PDF](https://speech.di.uoa.gr/ICMC-SMC-2014/images/VOL_2/1453.pdf) | P | Six-oscillator bank, band-pass filters, attack smoother, envelopes, swing-type VCAs, high-pass filters, tone stage |
| [SOS-BD] | G. Reid, "Synth Secrets: Practical Bass Drum Synthesis", *Sound On Sound*, Feb. 2002. [link](https://www.soundonsound.com/techniques/practical-bass-drum-synthesis) | S | 909 BD: the VCO is a sawtooth shaped to a near-sine; pulse plus noise click |

Not found or not used:

- We found no peer-reviewed analysis of the TR-909, TR-606 or CR-78 voices, so for those machines the service notes are the only primary source.
  - Values marked *(schematic)* were read off the circuit diagrams.
  - Values marked *(computed)* were computed by us from the printed component values. They are estimates, not measurements.
- Werner, Abel and Smith's cowbell paper ("More Cowbell", AES 137th Convention, 2014, paper 9207) is paywalled and was not consulted. The 808 CB rows rely on [808SN] and [WAS-CY].

## Today's voice (the baseline)

These labels are used in the tables. They come from `pythonic/voice.py` (signal-flow docstring, `VoiceParams`), `pythonic/voice_kernel.py`, `pythonic/drum_channel.py`, `pythonic/vintage.py`, `pythonic/sequencer.py` and `pythonic/synthesizer.py`.

| Label | Today's block |
|-------|---------------|
| `OSC` | One oscillator: sine, triangle or falling saw, band-limited and 2× oversampled. On every trigger it resets to **phase 1/8 cycle**, so a sine starts with a step of 0.71 of its peak (`reset_osc`). With an attack time set, the old sound fades and then the oscillator restarts. |
| `PMOD` | Pitch modulation of `OSC`: **Decay**, an exponential envelope in semitones (±96 st) with a time constant in ms; **Sine**, an LFO (±48 st); **Noise**, a random walk. Its depth follows velocity (`mod_vel`). |
| `OENV` | Oscillator amplitude envelope: an attack on a knob curve, then an exponential decay (−60 dB at the decay time, 10 ms–10 s). Its level follows velocity (`osc_vel`). |
| `NOISE` | Uniform white noise into one state-variable filter (LP, BP or HP, with frequency and Q), power-normalised, with an optional stereo pair. |
| `NENV` | Noise envelope: **Exp** (attack and exponential decay), **Linear**, or **Mod**, which gives N bursts that each fall to −24 dB, with N and their spacing derived from the attack time, followed by a decay. Its level follows velocity (`noise_vel`). |
| `MIX` | Crossfade between oscillator and noise. |
| `SHAPER` | Drive and an asymmetric soft shaper after the envelopes and the mix, shared by oscillator and noise. Because it sits after the envelopes, its distortion falls as the level decays, but its curve is fixed. |
| `EQ` | One peaking band (frequency, ±40 dB). |
| `OUT` | Level, pan, mono. A pitch offset scales `OSC`, the noise filter and `EQ`. |
| `VEL` | Accent is velocity 127, versus the step velocity (default 64). Velocity scales oscillator level, noise level and pitch-mod depth, each with its own sensitivity. |
| `LFO` | Channel modulation: two LFOs and a pump, at block rate (one value per audio buffer), onto patch parameters. |
| `VINTAGE` | Channel processor. It adds pitch drift to `OSC` (a block-rate random walk whose state and random generator **reset on every trigger**, so every hit drifts the same way), a thermal noise floor, tanh saturation, a one-pole HF roll-off and a DC blocker. |
| `CHOKE` | One choke group across channels, with a 10 ms fade. |

Not in today's voice:

- no filter on the oscillator;
- one noise path with one envelope;
- one oscillator, with no second oscillator, resonator or metal source;
- no pulse or click source apart from the oscillator's start step;
- no state that carries over when the voice is retriggered.

## Patterns shared by the machines

- **808 and 606: one common trigger, with accent as its amplitude.**
  - The CPU widens the step pulse to about 1 ms. The ACCENT circuit sets its amplitude: about 4 V on a normal step, and 4–14 V on an accented step depending on the ACCENT knob VR3.
  - Each voice ANDs its instrument data with this common trigger, and its output amplitude is proportional to the trigger amplitude.
  - For CB, CY, OH and CH the range is narrowed to 7–14 V ([808SN] p.4).
  - The check table gives normal and accented output amplitudes. Drums go from about 3–3.5 Vpp to 10–12 Vpp (about 10 dB), and metal voices from 3.5 to 7 Vpp ([808SN] p.14). This matches the specified accent range of 0–10 dB ([808SN] p.1).
  - On the 606, output is 2 Vpp at accent minimum and 6 Vpp at maximum ([606SN] p.1), and the accent block (Q9–Q11) drives the common trigger ([606SN] p.3).
- **Bridged-T resonator.** It is an op-amp with a bridged-T network in its feedback path, with f = 1/(2π√(R1R2C1C2)) and Q = √(R2/R1)/(√(C1/C2)+√(C2/C1)). Rung by a pulse, it gives a damped oscillation whose decay lengthens as Q rises ([808SN] p.5, Fig. 11). Every 808 voice uses it, either as the sound source or as a band-pass ([WAS-BD] fn 15).
- **Swing-type VCA.** The envelope feeds the collector of a transistor through a resistor and a diode. Roland uses it for the metallic voices because its output is rich in high harmonics ([808SN] p.5, Fig. 12). [WAS-CY] §8 shows that its biasing clips the output between the envelope voltage and a lower edge that depends on the envelope. This is clipping whose rails follow the envelope.
- **909: VCO, VCA and envelope.**
  - The CPU latches a per-voice ACCENT code that a resistor array converts to an analog level. Together with a 2 ms, 5 V trigger, that level builds the envelopes for pitch, tone colour, contour and loudness ([909SN] p.5).
  - The block diagram multiplies each voice's accent level with its trigger before the envelope generators ([909SN] p.4). The TOTAL ACCENT knob is read by the CPU ([909SN] p.5, port assignment).
- **Noise sources.**
  - 808: a junction-noise generator, set to 130 mV rms at TP-4 ([808SN] p.14). The block diagram feeds white noise (WN) to SD, CP and MA, and pink noise (P.N) to the toms ([808SN] p.7).
  - 909: a quasi-random generator built from two cascaded shift registers (32 stages), clocked fast through XOR gates ([909SN] p.6).
  - 606: a white-noise transistor, 130 mV rms ([606SN] p.3, p.8).
  - CR-78: a transistor junction (Q533) amplified by Q525, trimmed per voice group (VR60–VR63) ([78SN] p.26).

## TR-909

### TR-909 BD

| Element | Circuit detail | Today's block |
|---------|----------------|---------------|
| Oscillator | One VCO (IC13) into a waveshaper (D10, D11) ([909SN] p.4, p.11). The VCO is a resetting ramp, which the shaper turns into a near-sine ([SOS-BD], **S**). The base pitch is fixed by component values. | `OSC` Sine ≈. The shaper's residual harmonics need **M11**. |
| Tuning | **TUNE (VR2) sets the decay of the pitch envelope ENV3** (capacitor C9), not the base pitch ([909SN] p.11). A revision enlarges C9 from 0.22 µF to 0.33 µF to widen the TUNE range, which confirms it ([909SN] p.7). | `PMOD` Decay: the rate plays the part of TUNE ✓. |
| Pitch envelope | ENV3 is charged by trigger × accent and summed into a linear CV generator (IC12) ([909SN] p.4, p.11). Because the VCO's CV is linear, the pitch falls exponentially in **hertz**. | `PMOD` Decay ≈: today's law is exponential in semitones. Hz law: **M9**. |
| Transient | A pulse generator (Q8, Q9) and low-passed noise, mixed by ATTACK (VR3) into VCA Q6 under a short envelope ENV2 (C12 0.033 µF) ([909SN] p.4, p.11). [SOS-BD] (**S**) describes the same pulse-plus-noise click. The scope traces show the click as a large first spike ([909SN] p.9). | Noise part: `NOISE` LP with a short `NENV` ✓. The pulse and its own VCA: **M3**. |
| Noise | Used only in the click. | `NOISE` ✓ (white is close to the shift-register noise). |
| Envelopes, VCAs | ENV1 (C8 0.33 µF, DECAY VR5) drives VCA Q12 on the tone, ENV2 the click VCA, ENV3 the pitch ([909SN] p.4, p.11). | `OENV` ✓, `PMOD` ✓, click envelope **M3**. |
| Filters | A low-pass on the click noise only. No filter on the tone. | `NOISE` LP ✓. |
| Accent | The accent level multiplies the trigger that charges ENV3, ENV1 and ENV2. Accent makes the hit louder, with a wider pitch sweep and a bigger click (the traces with and without accent, [909SN] p.9). | `VEL` on level and `mod_vel` ✓. Click velocity: **M3**. |
| Nonlinearities | The diode waveshaper. | **M11** (`SHAPER` is not equivalent: it sits after the mix). |
| Hit-to-hit variance | The notes do not say whether the BD VCO is reset at the trigger (they say so for the SD and the toms). The noise generator runs free, so the click noise differs from hit to hit. | `OSC` resets ✓ (phase fixed at 1/8, see **M10**). `NOISE` ✓. |

### TR-909 SD

| Element | Circuit detail | Today's block |
|---------|----------------|---------------|
| Oscillators | **Two triangle VCOs**, VCO-1 (IC37) and VCO-2 (IC38). Different timing capacitors set them to different frequencies, VCO-1 the lower ([909SN] p.6). Each goes through a diode shaper (D65/D66, D69/D70) ([909SN] p.11). Their timing capacitors are 0.01 µF and 0.0068 µF *(schematic)*, a ratio of about 1.47 *(computed, assuming equal CV scaling)*. TUNE (VR6) sets the CV of both. | VCO-1: `OSC` ✓. VCO-2: **M2**. Shapers: **M11**. |
| Pitch | ENV-1 scales the CV, so the VCO's charging rate changes over about 20 ms, which bends the pitch ([909SN] p.6). Both VCOs share the CV. The charging rate is proportional to the CV, so the law is linear in Hz. | `PMOD` Decay ≈. The shared pitch envelope drives **M2**. Hz law: **M9**. |
| Transient | No separate click. The trigger resets both VCOs to their starting point, so their first cycles are in phase ([909SN] p.6). | Reset ✓, but the start point is fixed at 1/8 cycle today: **M10**. |
| Noise | One noise source goes through a low-pass (IC40a), then **two VCAs**: <br>(1) the low-passed noise × ENV4 (Q48), whose decay is set by **TONE (VR7)**; <br>(2) the low-passed noise through a 2nd-order high-pass (IC39a) × ENV5 (Q47). <br>Both are summed into IC39b, whose gain is **SNAPPY (VR9)** ([909SN] p.4, p.6, p.11). | Path (1): `NOISE` LP + `NENV` ✓. Path (2): **M8**. |
| Envelopes, VCAs | ENV3 drives VCA Q50 (VCO-1) and ENV2 drives VCA Q51 (VCO-2), so the two partials have two decays. ENV4 and ENV5 drive the noise, ENV1 the pitch ([909SN] p.4). | `OENV` ✓ for VCO-1. **M2** envelope, **M8** envelope. |
| Filters | Noise low-pass, then high-pass on path (2). No filter on the tone. | `NOISE` ✓ and **M8**. |
| Accent | The accent, gated by the trigger, controls ENV3 on VCO-1's VCA. Gated through Q41, it also forms ENV5, which sets how much high-frequency noise the snappy part gets ([909SN] p.6). So accent adds body and **bright noise**, not just level. | `VEL` on oscillator level ✓. Accent on the high-passed noise only: **M8** with its own velocity. |
| Nonlinearities | The diode shapers. | **M11**. |
| Hit-to-hit variance | The VCOs reset at the trigger. The noise varies from hit to hit. | ✓. |

### TR-909 LT, MT, HT

| Element | Circuit detail | Today's block |
|---------|----------------|---------------|
| Oscillators | **Three VCOs per tom**, synced ("SYNC") at the trigger and driven by one CV generator with TUNE (VR10) ([909SN] p.4, p.11): <br>- VCO2: timing capacitor 0.033 µF, the lowest; it goes through a shaper to VCA2. <br>- VCO1: 0.022 µF; it goes through a shaper whose output the diagram marks as square or sine, with a TR16 ON/OFF label. <br>- VCO3: 0.012 µF, the highest. <br>The ratios are about 1 : 1.5 : 2.75 (VCO2 : VCO1 : VCO3) *(computed from the capacitors with equal 47 kΩ resistors; not measured)*. | One: `OSC` ✓. Two more: **M2**. |
| Pitch | ENV4 is built from **two capacitors** (C16 0.1 µF, C17 0.047 µF, through D14 and D15) *(schematic)*, which gives a two-time-constant pitch drop into the CV generator. | `PMOD` Decay ≈. Two-slope and Hz law: **M9**. |
| Transient | VCO1 goes to VCA1 under a short ENV1 (C22 0.22 µF). A shared tom-noise VCA (Q33) is gated by any tom trigger (the OR of D26, D44 and D45) ([909SN] p.4). The scope trace marks noise at the LT attack ([909SN] p.9). Revision 426700 enlarges C54 to emphasise the tom attack ([909SN] p.7). | The noise burst: `NOISE` + short `NENV` ✓. The short VCO1 partial: **M2**. |
| Envelopes, VCAs | ENV1 (VCO1, short), ENV2 (VCO2, DECAY VR11), ENV3 (VCO3) ([909SN] p.11). | `OENV` ✓ for one. **M2**. |
| Filters | None on the tone. | — |
| Accent | The tom accent level multiplies the trigger that feeds the envelopes ([909SN] p.4). | `VEL` ✓, plus velocity on **M2**. |
| Nonlinearities | The waveshapers. | **M11**. |
| Hit-to-hit variance | The VCOs sync at the trigger. The noise varies. | ✓. |

### TR-909 RS

| Element | Circuit detail | Today's block |
|---------|----------------|---------------|
| Resonators | **Three multiple-feedback band-pass filters**, F1 (IC48a), F2 (IC49a) and F3 (IC49b), pinged by trigger × accent ([909SN] p.4). The schematic annotates their responses ([909SN] p.11): <br>- F1: period about 2 ms (about 500 Hz), about 20 ms long, 30 Vpp; <br>- F2: about 4.5 ms (about 220 Hz), about 25 ms, 15 Vpp; <br>- F3: about 1 ms (about 1 kHz), about 5 ms, 30 Vpp. | **M1** × 3. One mode is `OSC` ≈. |
| Transient | The ping itself. | **M1** excitation. |
| Nonlinearities | The sum goes into a **diode clipper (D91, D92) before the VCA**. With 15–30 Vpp swings, the clipper is driven hard ([909SN] p.4, p.11). | **M7** (clip before the VCA). `SHAPER` sits after the envelope, so its drive falls as the sound decays. |
| Envelopes, VCAs | VCA Q65 under an envelope (Q64, C119 0.047 µF) ([909SN] p.4, p.11). | `OENV` ≈. |
| Filters | A 2nd-order high-pass after the VCA (IC50, C121 and C122) ([909SN] p.11). | **M6**. `EQ` ≈ only. |
| Accent | The RS accent level × trigger excites the filters and charges the envelope ([909SN] p.4). Behind the clipper, accent changes the waveform's shape more than its level. | `VEL` + **M7**. |
| Revisions | From serial 415300, R417 goes from 12 kΩ to 3.3 kΩ for a more realistic sound ([909SN] p.7). Both waveforms are shown ([909SN] p.9). | Two factory variants to fit. |
| Hit-to-hit variance | The filters keep ringing across retriggers. | **M1**. |

### TR-909 CP

| Element | Circuit detail | Today's block |
|---------|----------------|---------------|
| Source | Noise into a multiple-feedback band-pass (IC26b) ([909SN] p.4, p.11). | `NOISE` BP ✓. |
| Envelopes, VCAs | **Two VCAs in parallel on the same filtered noise** ([909SN] p.4, p.11): <br>- IC30 (BA662) under a **sawtooth envelope** of repeated decays from comparator IC29, with Q40, and an offset trimmer (TM) on the Q38 pair; <br>- Q37 under a longer envelope (C61) that gives the tail. <br>They are summed at IC28a with LEVEL. | Bursts then decay: `NENV` Mod ≈. A **parallel** tail with its own level and decay: **M8**. |
| Accent | The hand-clap accent level × trigger ([909SN] p.4). | `VEL` ✓. |
| Nonlinearities | None described. | — |
| Hit-to-hit variance | The noise. | ✓. |

## TR-808

Amplitudes, frequencies and decays below are from the voice check table ([808SN] p.14). Its decays are measured to 1/10 of the peak, and the notes call the values typical, not exact.

### TR-808 BD

| Element | Circuit detail | Today's block |
|---------|----------------|---------------|
| Resonator | A multiple-feedback bridged-T with IC12, rung by the trigger pulse. DECAY (VR6) sets the feedback, and so the decay ([808SN] p.5, [WAS-BD] §5–7). Its period is 18 ms (56 Hz) in the check table; [WAS-BD] §5 computes a centre of about 49.5 Hz. Decay is 50 / 300 / 800 ms at short / mid / long. | `OSC` Sine + `OENV` ≈ for one mode. Retrigger behaviour: **M1**. |
| Attack retune | Right after the trigger, with accent present, the filter's time constant is halved: it rings at twice its frequency for half a cycle, then at its own frequency. Q41–Q43 short R165 for 4 ms, after which C39 discharging through R161 re-kicks the filter ([808SN] p.5). [WAS-BD] §8.1 adds that this raises both the Q and the centre frequency, by more than an octave, for about 6 ms, which gives the attack its punch. | `PMOD` Decay with a fast rate ≈. Exact: **M1** option (attack retune). |
| Pitch sigh | Leakage through R161 lowers the effective resistance via Q43 while the internal node swings more than a diode drop below ground, so **the frequency depends on amplitude**. [WAS-BD] fits it with a soft-plus law (§8.2, eq. 8). In their Fig. 11 the frequency glides down a few hertz over the first 300 ms. | `PMOD` Decay ≈ (it does not track the level, so it does not follow the DECAY knob). Exact: **M1** option (level-dependent tuning). |
| Transient | A pulse shaper: a low-shelf and a diode clipper (D53). Its rising edge equals the trigger voltage. Its falling edge is about V_trig·R162/(R162+R163) + 0.71 V, so it depends only partly on the pulse amplitude ([WAS-BD] §4). The edges kick the resonator. | **M3** (shaped pulse with asymmetric edges) as the excitation of **M1**. Today's only click is the `OSC` start step. |
| Noise | None. | — |
| Envelopes, VCAs | None. The resonator's own decay is the envelope ([808SN] p.7, [WAS-BD] §2). | `OENV` decay ≈. |
| Filters | TONE (VR5) is a passive low-pass, followed by the level divider and an output high-pass ([WAS-BD] §9). | `EQ` ≈. Low-pass: **M6**. |
| Accent | 3.5 → 10 Vpp. The attack retune is tied to accent ([808SN] p.5). Because the falling edge is partly fixed, the shape of the attack changes with accent ([WAS-BD] §4). A larger amplitude gives more sigh ([WAS-BD] §8.2). | Level: `VEL` ✓. Shape and sigh: **M1**, **M3**. |
| Nonlinearities | The pulse-shaper diode and the Q43 junction (sigh). [WAS-BD] §12 concludes that the architecture and the interactions between sub-circuits matter more than subtle device nonlinearities. | Covered by the **M1** options. |
| Hit-to-hit variance | Retriggering mixes with the residual ringing, so each note differs slightly and fast repeats avoid the machine-gun effect ([WAS-BD] §11). Between units, the ±20 % capacitors and ±5 % resistors change gain, centre frequency, Q and decay ([WAS-BD] §11). | Today's `OSC` resets on each hit, which gives a machine gun: **M1**. Unit tolerance is a fitting matter. |

### TR-808 SD

| Element | Circuit detail | Today's block |
|---------|----------------|---------------|
| Resonators | Two bridged-T networks, one for the fundamental and one for a harmonic ([808SN] p.6), at **238 Hz and 476 Hz** (4.2 and 2.1 ms). **TONE (VR8)** sets their output ratio ([808SN] p.6, p.7). | One: `OSC` ≈. The second: **M1** (or **M2**). |
| Transient | The trigger rings both. | **M1** excitation. |
| Noise | White noise goes to VCA Q48 (envelope Q47), then a high-pass (Q49) ([808SN] p.7). SNAPPY (VR9) sets the amplitude of the noise envelope ([808SN] p.6). | `NOISE` HP + `NENV` ✓. |
| Envelopes, VCAs | The resonators decay on their own (60 ms). The noise has its own envelope and VCA. | `OENV` ≈, `NENV` ✓. |
| Filters | High-pass on the noise. | `NOISE` ✓. |
| Accent | 3 → 10 Vpp. The trigger amplitude scales the resonators and the snappy envelope. | `VEL` ✓. |
| Nonlinearities | None described. | — |
| Hit-to-hit variance | Resonator state, and the noise. | **M1**, ✓. |

### TR-808 LT, MT, HT and LC, MC, HC

| Element | Circuit detail | Today's block |
|---------|----------------|---------------|
| Resonator | A bridged-T with IC15, IC17 or IC18, with a TUNING knob per voice. SW8 switches C77 (frequency) and R224 (level) between tom and conga ([808SN] p.6, p.7). Ranges: LT 80–100 Hz, MT 120–160, HT 165–220; LC 165–220, MC 250–310, HC 370–455 Hz. Decays: LT 200, MT 130, HT 100, LC 180, MC 100, HC 80 ms ([808SN] p.14). | `OSC` + `OENV` ≈. **M1**. |
| Level-dependent pitch | While the ringing is large, just after the trigger, D80 and D81 conduct and raise the frequency. As the ringing dies, the frequency falls ([808SN] p.6). | `PMOD` Decay ≈. Exact: **M1** option (level-dependent tuning). |
| Noise | Toms only: pink noise into a VCA (Q52, Q55, Q58, with envelopes D55, D57, D59) ([808SN] p.7). The notes describe this pink noise, with a slightly longer decay, as an artificial reverberation ([808SN] p.6). | `NOISE` LP (≈ pink) + `NENV` ✓. |
| Accent | About 3–3.5 → 10–12 Vpp. A larger amplitude gives a bigger pitch drop through D80 and D81. | `VEL` ✓. Pitch coupling: **M1**. |
| Hit-to-hit variance | Resonator state, and the noise. | **M1**. |

### TR-808 RS and CL

| Element | Circuit detail | Today's block |
|---------|----------------|---------------|
| RS resonators | **Two bridged-T networks** (IC20a and IC21) at **1667 Hz and 455 Hz** ([808SN] p.6, p.7, p.14). | **M1** × 2 (or `OSC` + **M2**). |
| RS VCA | Both go into the **swing-type VCA** Q62 (envelope R107, C24), chosen for its rich high harmonics ([808SN] p.6), then a high-pass (Q63) ([808SN] p.7). Decay 10 ms; 3 → 10 Vpp. | **M7** (swing), **M6** (HP). |
| CL resonator | One high-Q bridged-T (IC20) at 2500 Hz, decay 25 ms ([808SN] p.6, p.14). Q74 lets the output buffer IC19 pass signal only while the trigger pulse from the mono-multi Q61 lasts, to keep the high-Q filter's leakage out of the output ([808SN] p.6, p.7). | `OSC` + `OENV` ≈, **M1**. |
| Accent | The trigger amplitude. On RS it also sets how hard the swing VCA clips. | `VEL` + **M7**. |

### TR-808 CP

| Element | Circuit detail | Today's block |
|---------|----------------|---------------|
| Source | White noise into a band-pass (IC21, marked 1000 Hz) ([808SN] p.6, p.7). | `NOISE` BP ✓. |
| Envelopes, VCAs | The filtered noise feeds **two VCAs in parallel** ([808SN] p.6): <br>- IC22, whose gain follows a sawtooth envelope from comparator IC23: within a 30 ms window, C144 charges and discharges repeatedly, about three times. Q72 converts it exponentially, and the notes name it the main component of the clap. <br>- Q70, under a longer envelope that the notes call the clap's reverberation. | Bursts then decay: `NENV` Mod ≈. The parallel tail with its own level: **M8**. |
| Accent | An accent level holder (Q69, D68) keeps the accent through the clap ([808SN] p.7). The accent signal reaches the exponential converter through D68, C143 and R362 ([808SN] p.6). | `VEL` latched at the trigger ✓. |
| Levels | 6 Vpp, decay 100 ms ([808SN] p.14). A revision lowers the gain of both the clap and its reverberation ([808SN] p.15). | — |
| Hit-to-hit variance | The noise. | ✓. |

### TR-808 MA

White noise goes to VCA Q65 under an envelope (Q66, Q67) that the block diagram draws as a rise then a fall, then to a high-pass (Q68). Decay is 25–35 ms; 3 → 5 Vpp ([808SN] p.6, p.7, p.14). → `NOISE` HP + `NENV` (attack and decay) ✓. Nothing is missing.

### TR-808 metal source: six oscillators

| Element | Circuit detail | Today's block |
|---------|----------------|---------------|
| Oscillators | **Six Schmitt-trigger square oscillators** on one HD14584 hex inverter ([WAS-CY] §3). Their nominal frequencies are 205.3, 369.6, 304.4 and 522.7 Hz. Oscillators 5 and 6 are trimmed to **800 Hz and 540 Hz** (TM2, TM1) and also feed the cowbell ([WAS-CY] §3; [808SN] p.14). Duty cycle is 47.98 %, amplitude 5 V ([WAS-CY] §3). The ratios are inharmonic. | **M4**. |
| Running | The oscillators run continuously, and each voice gates them through its own VCA ([808SN] p.6). So the phase relationship at a trigger is arbitrary. | **M4** (free-running). Today's `OSC` resets. |
| Mixing and filters | A passive resistor network sums all six into **two band-pass filters at about 3440 Hz and 7100 Hz** (IC3), which bring out the upper overtones of the squares rather than their fundamentals ([WAS-CY] §4; [808SN] p.6). | **M5**. |
| Variance | Component tolerance on oscillators 5 and 6 gives each unit its own cymbal ([WAS-CY] fn 7). Early units (before serial 000300) used other values ([WAS-CY] fn 9). | A fitting matter. |

### TR-808 CB

| Element | Circuit detail | Today's block |
|---------|----------------|---------------|
| Oscillators | Metal oscillators 5 and 6 (800 Hz and 540 Hz), each through its own VCA (Q14, Q15) ([808SN] p.6, p.7). A revision extends the trimmer range because the cowbell frequency was hard to set ([808SN] p.15). | **M4** (2 oscillators). |
| Filter | A band-pass (IC2) after the VCAs mixes the two outputs ([808SN] p.6, p.7). | **M6** (BP), or a one-band **M5**. |
| Envelope | Envelope D2, C9. R82 and C34 in series, across C9, make the level drop abruptly at first, to stress the attack ([808SN] p.6). Decay 50 ms. | A **two-slope decay**: **M9**. |
| Accent | The trigger range is 7–14 V; 3.5 → 12 Vpp. | `VEL` ✓. |
| VCA type | The notes call Q14 and Q15 gates (VCAs) and do not say whether they are swing-type. | Linear unless recordings say otherwise (**M7** optional). |

### TR-808 CY

| Element | Circuit detail | Today's block |
|---------|----------------|---------------|
| Source | The six-oscillator bank and its two band-pass filters (see above). | **M4**. |
| Bands | The high band is split again, which gives three bands ([808SN] p.6, p.7): <br>- the high band to VCA Q16: the highest range, with a short decay; <br>- the high band to VCA Q17: slightly lower, with the decay under the DECAY knob and **two envelope sources** ([WAS-CY] fn 17); <br>- the low band to VCA Q18. | **M5** (3 bands). |
| Envelopes | An attack smoother (a one-pole, τ ≈ 0.1 ms, minus V_BE) feeds the envelope generators: switched RC stages with a high-shelf stage ([WAS-CY] §6–7). Decay is 350 / 800 / 1200 ms. | `OENV` per band: **M5**. Two sources on band 2: **M9**. |
| VCAs | **Swing-type** VCAs. Their biasing makes the output swing hard between the envelope voltage and a lower edge, which [WAS-CY] reads as intended ([WAS-CY] §8, eq. 13–15). | **M7** (swing). |
| Filters | Each band then goes through a Sallen-Key high-pass of 2nd, 2nd (non-unity gain) and 3rd order. The 3rd-order one resonates at about 10.5 kHz ([WAS-CY] §9). | **M5** (per-band HP). |
| Tone, output | TONE (VR4) sets the band ratio ([808SN] p.6). Its main effect is on the attenuation of band 3, and it also shifts the other bands a little, so the controls are not independent ([WAS-CY] §10). The output buffer is a differentiator, +6 dB/octave ([WAS-CY] §11). | **M5** band levels. `EQ` ≈ for the tilt. |
| Accent | The trigger range is 7–14 V; 3.5 → 7 Vpp. Accent acts through the nonlinear attack smoother, envelope generators and swing VCAs, so its effect is complex ([WAS-CY] §5). | `VEL` + **M7**. |
| Hit-to-hit variance | Free-running oscillators: every edge of any of the six squares kicks the band-pass filters again ([WAS-CY] §4). | **M4**. |

### TR-808 OH, CH

| Element | Circuit detail | Today's block |
|---------|----------------|---------------|
| Source | The high band of the bank (IC3 pin 7) ([808SN] p.6). | **M4** + one **M5** band. |
| OH | VCA Q27 (envelope Q24, Q29, C62, with a decay timer Q21 on DECAY), then a high-pass (Q26) and buffer IC7 ([808SN] p.7). Decay 90 / 450 / 600 ms; 3.5 → 7 Vpp. | **M5** band, **M7** (swing), `OENV`-like decay. |
| CH | The same source into VCA Q30 (envelope Q28, C63), then a high-pass (Q31) ([808SN] p.6, p.7). Decay 50 ms; 3 → 6 Vpp. | Same. |
| Choke | A CH trigger while OH sounds turns on Q23, which ends the OH decay ([808SN] p.6). | `CHOKE` ✓ (two channels in one choke group). |

## TR-606

The 606 has **no tune or decay controls**: there are level knobs for AC, BD, SD, L/H tom, CY and O/C hi-hat ([606SN] p.1). The notes carry only a block diagram, the adjustments and the changes. They give no circuit description and no voice frequencies.

| Voice | Structure ([606SN] p.3 unless noted) | Today's block |
|-------|---------------------------------------|---------------|
| BD | Two resonators ("OSC" IC15A and IC15B) rung by the trigger (Q14, Q15) and summed. Their frequencies are not given. | `OSC` ≈ for one. **M1** × 2 (or `OSC` + **M2**). |
| SD | One resonator (IC10A) with an **ATTACK stage (Q17) driven by the snare envelope** (ENV Q16). White noise goes to VCA Q30 under the same envelope, then a high-pass (Q32). The notes do not explain ATTACK. By analogy with the 808 BD's Q41–Q43, it most likely retunes the resonator during the attack *(inference)*. | `OSC` ≈ / **M1** (+ attack-retune option). `NOISE` HP + `NENV` ✓. |
| LT, HT | One resonator each (IC9A, IC9B), each with an ATTACK stage (Q307, Q308), from a **shared envelope** (Q305, Q306). White noise goes through a low-pass (Q309) into a VCA (Q310) under the same envelope, mixed into both toms. One level knob covers both. | `OSC` ≈ / **M1**. `NOISE` LP + `NENV` ✓. |
| CY | **Six oscillators** (IC16; the parts list gives an HD14584B hex Schmitt trigger, [606SN] p.8) feed two band-pass filters (IC15B, IC15A). These go to two VCAs (Q39, Q38) under **one envelope** (Q35), then to two high-pass filters (Q40, IC12B). Revision R226 330 kΩ → 680 kΩ shifts one oscillator against the others for a better cymbal ([606SN] p.8). | **M4**, **M5** (2 bands sharing one envelope). The VCA type is not stated (**M7** optional). |
| OH, CH | A band-pass output of the same bank goes to **one VCA (Q37)** driven by the OH envelope (Q36) or the CH envelope (D31, C42), then a high-pass (IC12A). An envelope shut-off (Q24, Q25) on the OH envelope is fed by the tempo clock and the CH side; the notes do not describe it. | **M4**, one **M5** band. `CHOKE` ✓ for the CH → OH cut. Check the tempo-clock cut against recordings. |
| Accent | Q9–Q11 with the AC level knob set the common trigger. Output is 2 Vpp at accent minimum and 6 Vpp at maximum ([606SN] p.1). | `VEL` ✓ (and its reach into **M1**, **M5**). |
| Hit-to-hit variance | Resonator state, free-running metal oscillators, and noise (130 mV rms, [606SN] p.8). | **M1**, **M4**. |

## CR-78

The CR-78 predates the 808's bridged-T op-amp voices. Its drums are **single-transistor ringing filters** and passive **LC resonators**. Its metallic voices use **LC tanks** (45 mH inductors) as the load of a transistor VCA. Voice values are from the adjustment table ([78SN] p.30). Structure is from the VG-11A voicing circuit diagram ([78SN] p.25–26). The CR-78 has no per-voice panel controls: the frequency and decay trimmers are internal. The adjustment procedure sets a BALANCE knob to its lowest to check BD, and to its highest to check HH, CY and MA ([78SN] p.30).

| Voice | Structure | Today's block |
|-------|-----------|---------------|
| BD, HB, LB, LC (and the SD tone) | A one-transistor amplifier with a three-capacitor RC network, rung through a diode by the trigger, with a frequency trimmer and a decay trimmer (VR51–VR58) ([78SN] p.25). BD 62.5 Hz / 100 ms; SD 340 Hz / 60 ms; HB 600 Hz / 40 ms; LB 400 Hz / 40 ms; LC 208 Hz / 150 ms ([78SN] p.30). | `OSC` Sine + `OENV` ≈ (one mode). **M1** for the retrigger behaviour. |
| SD noise | Noise goes into a transistor VCA (Q506) whose base gets an exponential envelope (Q514, C514). Its collector load is an RC low-pass (R514, C516) ([78SN] p.25). | `NOISE` LP + `NENV` ✓. |
| RS, CL | Passive **LC resonators** (L2, L1) rung through a diode ([78SN] p.25). RS 1480 Hz / 5 ms; claves 2630 Hz / 18 ms ([78SN] p.30). | `OSC` + `OENV` ≈, **M1**. |
| HH, CY, MA | **Three envelope generators summed into one transistor VCA (Q507) on noise.** Its collector load is an **LC tank**, L3 (45 mH) in parallel with C521 (6.8 nF), a band-pass at about **9.1 kHz** *(computed)* ([78SN] p.25). The envelopes are HH (Q516, C525), CY (Q515, with **two capacitors** C519 and C520) and MA (Q517, C527). Decays are HH 60 ms, CY 350 ms, MA 20 ms. One trimmer (VR60) sets all three levels ([78SN] p.30). | Each as its own channel: `NOISE` BP + `NENV` ✓. CY two-slope: **M9**. The shared VCA, where one voice's hit lifts the others' tails, is not modelled. |
| CB | **Two transistor oscillators**: Q529, a phase-shift type with three 8.2 nF capacitors, at 800 Hz, and Q530 at 555 Hz. The adjustment table measures them free-running at their collectors ([78SN] p.26, p.30). They feed VCA Q511, whose collector load is an **LC tank** (L7, 45 mH), under an envelope (Q528). Decay 60 ms. | **M4** (2 oscillators, near-sine), **M5** (one band) or **M6**. |
| MB (metal beat) | **Three CMOS inverter oscillators** in IC501 (trimmers VR64–VR66) at **6170, 5620 and 4080 Hz** go to VCA Q510, whose load is an **LC tank**, L6 (45 mH) in parallel with C342 (18 nF), about **5.6 kHz** *(computed)*. Envelope Q524; decay 50 ms ([78SN] p.26, p.30). | **M4** (3 squares), **M5** (one band). |
| TB (tambourine) | Noise goes through an envelope (Q522–Q524) into VCA Q509 with an LC tank (L5). Decay 220 ms ([78SN] p.26, p.30). | `NOISE` BP + `NENV` ✓. |
| GU (guiro) | Noise is gated by an astable (Q520, Q521) into VCA Q508 with an LC tank (L4). There are two scrape rates, H at 125 Hz (8.0 ms) and L at 77 Hz (13.0 ms) ([78SN] p.26, p.30). | `NOISE` BP + `NENV` Mod ≈ (bursts derived from the attack time). |
| Accent | **Not per voice.** The summing amp IC117 adds the accent pulses to the volume control voltage, which drives the VCA on the voicing board that sets the rhythm's volume ([78SN] p.13). It is the master VCA (a BA662), and the FADE circuits share it. | Per voice, `VEL` ≈. The real behaviour, a gain step on the whole mix including other voices' tails, is **M12** (kit level). |
| Hit-to-hit variance | The CB and MB oscillators run free. The ringing filters keep their state. The noise varies. | **M4**, **M1**. |

## Missing building blocks (de-duplicated)

"Off = today's render" means that, with the block off, the render is bit-identical to today's and the patch saves as today's `.mtdrum` V3. Two rules make that hold for every block:

- **skip** the code path when the block is off, rather than multiplying by zero;
- give any new randomness **its own random stream**, so today's noise stream and pitch-mod stream are consumed exactly as now.

New sources join the voice before `SHAPER`, so they share drive, `EQ` and `OUT`. They must also follow the 4-sample block grid and the pitch offset, as `OSC` does.

### M1 — Resonator section (pinged band-pass, 1–3 modes)

- **Needed by:**
  - TR-808: BD; SD (2 modes); LT/MT/HT and LC/MC/HC; RS (2); CL.
  - TR-606: BD (2); SD; LT; HT.
  - CR-78: BD; SD tone; HB; LB; LC; RS; CL.
  - TR-909: RS (3).
- **What it adds over `OSC` + `OENV`:**
  - second and third modes;
  - a decay set by Q;
  - **state kept across retriggers**, so there is no machine gun ([WAS-BD] §11);
  - a start from rest, with no 1/8-cycle step.

  Options:

  - **attack retune**: frequency ratio (about ×2) and Q for a few ms, then a re-kick (808 BD, [808SN] p.5, [WAS-BD] §8.1; probably 606 SD and toms);
  - **level-dependent tuning**: frequency rises with the mode's amplitude (808 BD sigh, [WAS-BD] §8.2; 808 toms and congas, [808SN] p.6).
- **Rough DSP:**
  - one 2-pole band-pass per mode, as a TDF-II biquad from the bilinear transform as in [WAS-BD] §10, or an SVF;
  - coefficients updated per 4-sample block when an option moves them;
  - excited by the trigger pulse (**M3**), scaled by velocity;
  - for level-dependent tuning, track the mode amplitude (√(s1²+s2²) of the state) and map it through a soft-plus law, f = f0·(1 + k·softplus(a − a0)), after [WAS-BD] eq. 8;
  - modes summed with per-mode levels; the 808 SD's TONE is the ratio of two modes.
- **Off = today's render:** yes (section disabled, not computed). Fitting note: a single mode without the options is close to today's `OSC` + `OENV`, so a kit can start without M1 and add it where retrigger rolls or the sigh matter.

### M2 — Extra tone oscillators ("partials")

- **Needed by:** TR-909 SD (VCO-2), TR-909 LT/MT/HT (VCO1, VCO3). Also an alternative to M1 for the second mode of the 808 SD, 808 RS and 606 BD.
- **What:** one or two more oscillators. Each has a frequency **ratio** to `OSC` and **shares `OSC`'s pitch modulation and its reset at the trigger** (909 "SYNC", [909SN] p.4, p.6). Each has its own wave (sine, triangle, square), level, **own decay envelope** and velocity sensitivity.
- **Rough DSP:** reuse today's band-limited oscillator at the 2× rate, with increment = `OSC` increment × ratio, and an exponential envelope like `OENV`. Mix before `SHAPER`.
- **Off = today's render:** yes (count 0).

### M3 — Click / pulse generator (and resonator excitation)

- **Needed by:** TR-909 BD (the pulse from Q8 and Q9 under its own short envelope ENV2 with ATTACK, [909SN] p.4, p.11). It is also the excitation of **M1**, for which the 808 BD needs the pulse shaper's **asymmetric edges**: the falling edge partly fixed at about 0.71 V, independent of accent ([WAS-BD] §4).
- **What:** a short pulse with width (0.1–2 ms), level and velocity, shaped by a one-pole low-pass or high-pass, under its own very short envelope, mixed before `SHAPER`. For excitation, it gives rising-edge and falling-edge amplitudes as a·velocity and b·velocity + c.
- **Rough DSP:** generate the pulse at the 2× rate and band-limit it (or use a PolyBLEP-smoothed rectangle), then a one-pole filter. Its cost is negligible.
- **Off = today's render:** yes (level 0, skipped). Today's only click is `OSC`'s start step (see **M10**).

### M4 — Metal oscillator bank

- **Needed by:** TR-808 CY, OH, CH (6 oscillators) and CB (2); TR-606 CY, OH, CH (6); CR-78 MB (3 squares) and CB (2 near-sines).
- **What:** up to six oscillators with **per-oscillator frequency**, level and mute. The wave is square with about 48 % duty ([WAS-CY] §3), or a near-sine (CR-78 CB). They are **free-running**: never reset at the trigger, so the phase relationship differs on every hit. The 808 defaults are 205.3, 304.4, 369.6, 522.7, 540 and 800 Hz.
- **Rough DSP:** PolyBLEP squares at the 2× rate. Aliasing matters, because the filters that follow emphasise 3–10 kHz; [WAS-CY] §12 renders at 4× and leans on the band-pass filters and masking. "Free-running" can be modelled cheaply by drawing random phases at the trigger from the section's own random stream, which is statistically the same.
- **Off = today's render:** yes (bank disabled). Cost: six oscillators plus the M5 filters is several times today's per-voice cost. The real-time budget ticket should note it.

### M5 — Metal band section (band-pass → VCA with own envelope → high-pass)

- **Needed by:**
  - TR-808: CY (3 bands, two of them on the high band-pass, one with two envelopes); OH and CH (1 band); CB (1 band, VCA before the band-pass).
  - TR-606: CY (2 bands, one shared envelope); OH/CH (1 band).
  - CR-78: CB and MB (1 band, an LC-tank band-pass).
- **What:** one to three parallel bands on the M4 output. Each band has:
  - a band-pass of 2nd or 3rd order (808: about 3.44 kHz and 7.1 kHz, [WAS-CY] §4);
  - a VCA with its **own decay envelope**, an optional attack smoother, and an optional swing nonlinearity (**M7**);
  - a high-pass of 2nd or 3rd order with optional resonance (808: about 10.5 kHz on band 3, [WAS-CY] §9);
  - a level, with the TONE knob as the band levels.
- **Rough DSP:** cascaded biquads from a bilinear transform **prewarped** at each centre frequency (the features sit at 3–10 kHz), and exponential envelopes per band.
- **Off = today's render:** yes (no bands).

### M6 — Tone-path filter

- **Needed by:** TR-808 BD (the TONE low-pass, [WAS-BD] §9); TR-808 RS (high-pass Q63); TR-909 RS (2nd-order high-pass after the VCA); TR-808 CB (band-pass after the VCAs); CR-78 LC-tank voices, when they are built without M5.
- **What:** one 2-pole LP, BP or HP (with frequency and Q) on the tonal path (`OSC`, M1, M2, M4) **before** the mix and `SHAPER`. Today's only filter on the tone is the post-shaper peaking `EQ`.
- **Rough DSP:** an SVF like the noise filter's, following the pitch offset.
- **Off = today's render:** yes (bypassed).

### M7 — VCA-stage saturation ("saturation by level")

- **Needed by:**
  - **swing-type VCA**: TR-808 CY, OH, CH, RS (and CB if measured so), probably the 606 metal voices;
  - **clip before the VCA**: TR-909 RS (diode clipper D91 and D92, driven by trigger × accent).
- **What:** two modes.
  - *Swing*: the envelope is the VCA's rail, so the signal clips softly against a ceiling at the envelope level and a floor at a lower edge that depends on the envelope ([WAS-CY] §8, eq. 13–15). Clipping falls as the envelope decays, and it rises with accent.
  - *Pre-VCA clip*: a fixed soft clip of the tone sum **before** the envelope, so clipping depth depends on the trigger level and does not fall with the decay.
- Today's `SHAPER` sits after the envelopes and the mix, with one curve shared by tone and noise. It cannot act per band or before the VCA.
- **Rough DSP:** *swing*, y = e·sc(g·x/e) with an asymmetric soft clip sc and a lower edge as a fitted function of e (the [WAS-CY] fit), computed per sample at the 2× rate. *Pre-VCA*, y = env·sc(g·x).
- **Off = today's render:** yes (mode none; today's `SHAPER` left as is).

### M8 — Second noise path

- **Needed by:**
  - TR-909 SD: low-passed noise under ENV4 (TONE), plus high-passed noise under ENV5, set by accent ([909SN] p.6);
  - TR-808 and TR-909 CP: the same band-passed noise into the burst VCA and a **parallel** reverb-tail VCA ([808SN] p.6; [909SN] p.4).
- **What:** a second noise path with its own filter (LP, BP or HP, or "share filter 1"), its **own envelope** (Exp, Linear or Mod), level and **own velocity sensitivity**, summed with the first path. An optional pink colour would also cover the 808 toms more exactly.
- **Rough DSP:** a second SVF on white noise drawn from its **own random stream**, or on the first path's filtered signal when the filter is shared, with today's `NENV` code reused for the envelope.
- **Off = today's render:** yes, as long as it never draws from today's noise stream.

### M9 — Envelope extras: two-slope decay, Hz pitch law

- **Needed by:**
  - **two-slope decay**: TR-808 CB (an abrupt first drop, [808SN] p.6); TR-808 CY band 2 (two envelope sources, [WAS-CY] fn 17); CR-78 CY (two capacitors); TR-909 tom pitch envelope (C16 and C17);
  - **pitch in Hz**: TR-909 BD, SD and toms (linear-CV VCOs, [909SN] p.6).
- **What:**
  - decay = a·exp(−t/τ_fast) + (1 − a)·exp(−t/τ_slow), available on `OENV`, the `PMOD` Decay envelope and the new sections' envelopes;
  - a pitch-envelope law switch: f = f0 + Δf·env (Hz), against today's f = f0·2^(A·env/12).
- **Rough DSP:** a second exponential state per envelope. For the Hz law, compute the increment as `inc0·(1 + d·env)` instead of `inc0·2^(…)`.
- **Off = today's render:** yes, provided the off case **branches to today's code**. Do not rewrite the existing envelope as the special case a = 0.

### M10 — Oscillator start phase

- **Needed by:** the TR-909 VCOs, reset to a defined starting point ([909SN] p.6). It also helps any voice where `OSC` stands in for a resonator, which starts from rest at a zero crossing.
- **What:** a start-phase parameter (0–1 cycle) for the reset in `reset_osc`. Today it is fixed at 1/8 cycle, so a sine starts with a step of 0.71 of its peak.
- **Rough DSP:** a constant.
- **Off = today's render:** yes (default 0.125).

### M11 — Per-oscillator waveshaper (triangle or ramp → near-sine)

- **Needed by:** TR-909 BD (a ramp through D10 and D11; [SOS-BD] **S**), TR-909 SD and toms (triangles through diode pairs, [909SN] p.11).
- **What:** a shape amount that bends `OSC`'s triangle or saw toward a sine through a diode-pair curve, keeping the shaper's residual odd harmonics. Today's Sine wave is the shaper's ideal output, and today's Triangle is its input. This is **low priority**: fit first with Sine.
- **Rough DSP:** a memoryless curve (for example, a tanh-like diode pair) on the wave sample before decimation.
- **Off = today's render:** yes (amount 0).

### M12 — Kit-level accent bus (outside the channel)

- **Needed by:** CR-78 (master-VCA accent, [78SN] p.13).
- **What:** an accented step raises the gain of the **whole mix**, including the tails of earlier hits, through a shaped control voltage. This is not a channel section. It belongs with the kit or sequencer tickets. Per-voice velocity approximates it.
- **Off = today's render:** yes (not used).

### Cross-cutting requirements (not blocks)

- **Velocity everywhere.** On these machines, accent reaches the pitch-envelope depth, the attack retune, the high-passed noise, the clippers and the swing VCAs. Every new section needs its own velocity sensitivity, latched at the trigger as today (this also covers the 808 CP's accent level holder).
- **Pitch and drift.** The new tonal sections (M1, M2, M4, M6) should follow the channel pitch offset. If `VINTAGE` is on, they should also follow its drift, as `OSC` does.
- **Choke.** The 808's CH stops OH, and the 606 has an OH shut-off. Both map onto today's channel `CHOKE` ✓.

## Voice × missing block

`x` = needed for fidelity; `o` = optional (today's blocks approximate it, or the evidence is an inference).

| Voice | M1 | M2 | M3 | M4 | M5 | M6 | M7 | M8 | M9 | M10 | M11 | M12 |
|-------|----|----|----|----|----|----|----|----|----|-----|-----|-----|
| 909 BD | | | x | | | | | | x | o | o | |
| 909 SD | | x | | | | | | x | x | o | o | |
| 909 LT/MT/HT | | x | | | | | | | x | o | o | |
| 909 RS | x | | | | | x | x | | | | | |
| 909 CP | | | | | | | | x | | | | |
| 808 BD | x | | x | | | o | | | | | | |
| 808 SD | x | o | | | | | | | | | | |
| 808 toms/congas | x | | | | | | | | | | | |
| 808 RS | x | o | | | | x | x | | | | | |
| 808 CL | o | | | | | | | | | | | |
| 808 CP | | | | | | | | x | | | | |
| 808 MA | | | | | | | | | | | | |
| 808 CB | | | | x | o | x | o | | x | | | |
| 808 CY | | | | x | x | | x | | x | | | |
| 808 OH, CH | | | | x | x | | x | | | | | |
| 606 BD | x | o | | | | | | | | | | |
| 606 SD | o | | | | | | | | | | | |
| 606 LT, HT | o | | | | | | | | | | | |
| 606 CY | | | | x | x | | o | | | | | |
| 606 OH, CH | | | | x | x | | o | | | | | |
| CR-78 BD/HB/LB/LC/SD tone | o | | | | | | | | | | | x |
| CR-78 RS, CL | o | | | | | | | | | | | x |
| CR-78 HH/CY/MA | | | | | | | | | o | | | x |
| CR-78 CB | | | | x | o | o | | | | | | x |
| CR-78 MB | | | | x | x | | | | | | | x |
| CR-78 TB, GU | | | | | | | | | | | | x |

## Already covered, or not worth a block

- **Single-mode drums with a fixed decay** (808 CL, the 606 and CR-78 drums) are close to `OSC` Sine + `OENV`. M1 matters for rolls and flams (retrigger) and for the 808 BD and toms (pitch behaviour).
- **Snare and tom noise, maracas, tambourine, and the CR-78 hats and cymbals** fit today's `NOISE` (LP, BP or HP) and `NENV`. The 808 toms' pink noise is close to low-passed white.
- **Clap bursts** fit `NENV` Mod. Only the parallel tail needs M8.
- **909 noise.** The shift-register noise is treated as white.
- **Hit-to-hit variance.** The documented sources (free-running oscillators, resonator state, noise) are covered by M4, M1 and today's `NOISE`. **Rail sag** and **per-hit pitch jitter** are not documented in the service notes or the papers, so we propose no block unless the measured recordings show them. **Unit-to-unit tolerance** ([WAS-BD] §11; [WAS-CY] fn 7) is a matter for the fitted constants of each factory kit.
- **Voices sharing parts.** The CR-78 HH, CY and MA share one VCA, the 606 OH and CH share one VCA, and the 909 tom noise is shared. Separate channels model them. The difference is audible only when voices overlap, and it is not worth a coupling block.

## Open points for the section tickets

- Is M1 needed at all where `OSC` already fits a single mode? Decide on recordings: compare fast rolls (retrigger) and DECAY sweeps (808 BD sigh) against `PMOD` Decay.
- M2 vs M1 for the second mode of the 808 SD, 808 RS and 606 BD. One block with two excitation styles (driven or pinged) would keep the parameter space smaller.
- M5 and M6 overlap: a one-band M5 without the VCA is M6 on the metal path. The section ticket can merge them.
- The ratios we computed for the 909 SD and toms (about 1.47; about 1 : 1.5 : 2.75) and the CR-78 LC-tank centres (about 9.1 kHz and 5.6 kHz) are estimates from component values. Measure them on the recordings.
- 606: the ATTACK stages, and the role of the tempo clock in the OH envelope shut-off, are not described in the notes.
