# PCM voices of the drum machines

Research for [#26](https://github.com/jeromeboivin/pythonic/issues/26), part of the map [#23](https://github.com/jeromeboivin/pythonic/issues/23). It feeds the section-design and sample-library tickets, following [ADR 0002](../adr/0002-machine-voices-on-one-superset-channel.md): the universal channel gains optional sections, one of them a sample layer.

**Question.** How do the PCM voices in scope work, and what must a sample layer emulate so they sound right? The voices are the TR-909's hi-hats, crash and ride, the TR-707, the TR-727, and the TR-8S's own sample engine.

**Scope.** This file covers facts only. It holds no ROM data and no audio, and nothing was downloaded beyond the documents cited. The licensing decision belongs to another ticket.

## Summary

- **The ROMs are linear PCM, with no companding.** The TR-909 cymbals and hats, and the TR-707/727 cymbal slots, use **6-bit** linear codes on a resistor-ladder DAC. The other TR-707/727 voices use **8-bit** linear codes on one time-shared, multiplying 8-bit DAC. In every case Roland flattened the recordings' dynamics before storing them, to win signal-to-noise ratio. **An analog envelope after the DAC (or on its reference) puts the decay back.** Raw ROM playback therefore sounds wrong: the envelope is part of the voice.
- **TR-909**: three 32 KB mask ROMs, one each for the hats, crash and ride. Each sample clock is a gated RC oscillator. The hat clock is about 30 kHz, with a fixed rate. Open and closed hats are two address ranges of the hat ROM: 24,576 and 8,192 samples. The hats' DECAY knobs set an RC envelope through an anti-log VCA. Crash and ride have **TUNE** (a clock-rate change) and no decay knob. Their envelope is derived from the ROM address, so tune changes pitch, length and decay together. Accent is a 6-bit DAC level into each VCA path.
- **TR-707/727**: fixed clocks (25 kHz or 12.5 kHz) divided from one 1.6 MHz master clock, with no tune and no decay controls. Each voice has 4,096 or 8,192 samples; the two cymbal slots have 32,768 samples at 25 kHz. Each voice has a fixed RC envelope whose peak is set by a 6-bit accent/velocity level. The hi-hat (727: whistle) has a second VCA envelope switched between open (long) and closed (short). Per-voice band-pass filters, a ~15.9 kHz low-pass and a fixed pan follow. The 727 is the same circuit with other ROMs and component values.
- **TR-8S**: its 909, 707 and 727 voices are ACB models, not sample tones. Its sample engine imports **WAV or AIFF**: 8, 16, 24 or 32-bit integer, or 32-bit float. WAV goes up to 96 kHz; AIFF can be 44.1, 48 or 96 kHz. Files can be mono or stereo, up to **~180 s per file**, **~600 s in total** (at 44.1 kHz mono), and **400 files**. A sample tone has Start/End, Gain, Tune, Coarse, Rate (varispeed, negative = reverse), Spread, **Bit Reduce 0–12**, Attack, Hold (whole/time/steps), Decay and a filter with its own envelope. The kit's MUTE setting gives choke.
- **Capabilities the sample layer needs** (detail in [section 6](#6-sample-layer-capabilities)):
  - shared with user import: rate/tune by resampling; start/end and regions; an RC-style decay envelope with an accent-scaled peak; bit-depth reduction; choke and monophonic retrigger; alternate samples per slot;
  - needed only to emulate the ROM machines: a zero-order-hold DAC at the native clock; envelope-before-quantisation ordering; position-locked (address-derived) envelopes; per-trigger decay variants; fixed post-filters;
  - needed only for user import: reverse, hold modes, loop (preset loops only), stereo/spread, file decoding.
  - **Companding is not needed for any machine in scope.**

## Sources

Trust: **P** = primary (Roland documents). **R** = a reputable secondary source: an emulation built from the service-note schematics, or a hardware write-up by someone who dumped or rebuilt the circuit. Claims that rest only on R sources are marked *(R only)*. Conclusions drawn here from the sources are marked *(inference)*.

| ID | Source | Trust | Used for |
|----|--------|-------|----------|
| S1 | [TR-909 Service Notes, Roland, 15 Jun 1984 (scan)](https://archive.org/details/synth_Roland_TR-909_service_notes) | P | Block diagram (p. 4), circuit description of the digital voices and hi-hat schematic (p. 5), parts list (ROM part numbers) |
| S2 | [TR-909 Owner's Manual, Roland](https://static.roland.com/assets/media/pdf/TR-909_OM.pdf) | P | Panel controls (Hi-Hat LEVEL, CH DECAY, OH DECAY; Cymbal LEVEL ×2, CRASH TUNE, RIDE TUNE; TOTAL ACCENT) |
| S3 | [TR-707/TR-727 Service Notes, Roland, Jul 1985 (scan)](https://archive.org/details/roland_Roland_TR707) | P | Circuit descriptions (pp. 5–7): MB63H114 counters, ROM layout tables 1–2, multiplying DAC, envelope generators, hi-hat VCA, cymbal "single sound" section |
| S4 | [TR-707 Owner's Manual, Roland](https://static.roland.com/assets/media/pdf/TR-707_OM.pdf) | P | Specifications: 15 voices, level knob per voice, Accent Level knob, two accent strengths; MIDI chart (velocity received) |
| S5 | [TR-8S Reference Manual, v3.00 (eng05)](https://static.roland.com/assets/media/pdf/TR-8S_Reference_eng05_W.pdf) | P | Sample import (p. 41), SAMPLE Edit (p. 42), INST Edit parameters for sample tones (pp. 32–33), KIT: MUTE (p. 29), CRUSHER (p. 36) |
| S6 | [TR-8S Owner's Manual (eng03)](https://static.roland.com/assets/media/pdf/TR-8S_eng03_W.pdf) | P | Import basics (p. 17) |
| S7 | [TR-8S Preset INST Tone List, v3.00](https://static.roland.com/assets/media/pdf/TR-8S_PresetToneList_eng04_W.pdf) | P | Tone type (ACB / FM / SAMPLE) of the 909, 707 and 727 voices |
| S8 | [TR-8S specifications page](https://www.roland.com/global/products/tr-8s/specifications/) | P | Total sample time, preset sample count, 24-bit/96 kHz DAC |
| R1 | [MAME `roland_tr707.cpp`](https://github.com/mamedev/mame/blob/master/src/mame/roland/roland_tr707.cpp) (driver stated to be based on the TR-707 service notes) | R | Component values (envelope RCs, filters, pan), DAC data bits for the cymbals, the clock that steps the cymbal counters, ROM list |
| R2 | [MAME `mb63h114.cpp`](https://github.com/mamedev/mame/blob/master/src/mame/roland/mb63h114.cpp) | R | MB63H114 clock outputs: 1.6 MHz divided to 100/50/25/12.5 kHz; eight 13-bit counters |
| R3 | [MAME `roland_tr909.cpp`](https://github.com/mamedev/mame/blob/master/src/mame/roland/roland_tr909.cpp) | R | TR-909 sample ROM list (sizes, IC positions) |
| R4 | [HKA Design, TR-707 ROM Expansion](https://hkadesign.org.uk/tr707expansion.html) | R | ROM address map, per-voice sample rates and lengths for the 707 and 727, "8-bit linear" / "6-bit linear" encoding (from dumping the ROMs) |
| R5 | [HKA Design, TR-707 Cymbals Board](https://hkadesign.org.uk/tr707cymbals.html) | R | 6-bit R-2R cymbal DAC, compressed cymbal data, RC envelope |
| R6 | [Colin Fraser, 808/909 clone](http://www.colinfraser.com/tr909/my909.htm) | R | TR-909 hat/cymbal ROM data: six bits per byte, ~32 kHz hat rate |
| R7 | [network-909, "HiHat and Cymbals"](http://www.network-909.de/hihatand.htm) | R | A close paraphrase of S1, with remarks on the OH/CH decay interaction |

## Today's voice, for reference

`pythonic/voice.py` (rendered by `voice_kernel.py`) and `pythonic/drum_channel.py`:

- **Sources.** There are two: a band-limited, 2× oversampled oscillator with pitch modulation, and filtered white noise. Each has an attack/decay envelope that falls 60 dB over the decay time (noise also has linear and clap "mod" envelopes). There is no sample source.
- **Processing.** Mix, drive and asymmetric shaper, SVF peaking EQ, then level, pan and makeup. Velocity reaches the oscillator, noise and modulation through `velocity_gain`. `pitch_semitones` scales the oscillator, noise filter and EQ together.
- **Events.** One voice per channel, monophonic and retriggerable. `choke()` fades the voice out over 10 ms (`release_slope = -100/sr`). Events act on a 4-sample block grid in absolute time, which keeps the render independent of block size.
- **Channel extras after the voice.** `VintageProcessor` (drift, noise floor, saturation, high-frequency roll-off), delay, reverb, LFOs and pump.

A sample layer is therefore a new source, with its own envelope, alongside the oscillator and noise. To keep the block-size independence, its triggers should take effect on the same 4-sample grid *(inference)*.

## 1. TR-909: hi-hats, crash and ride

The 909 calls these its "digital voice generators". The hats, ride and crash play back PCM memories made from recordings of real instruments, and the three circuits are essentially the same (S1 p. 5).

| Aspect | Hi-hat (open + closed) | Crash, ride | Source |
|---|---|---|---|
| ROM | One HN61256P (32,768 × 8): PC43 at IC69 | One HN61256P each: PC42 at IC62 (crash), PC44 at IC54 (ride) | S1 parts list and block diagram; R3 |
| ROM size, total | 3 × 32 KB = 96 KB for the three voices | | S1, R3 |
| Bit depth | 6 bits: a 6-bit latch (IC68) drives a 6-resistor array (RA9) | 6 bits: latch IC53/IC63 into array RA10/RA12 | S1 p. 4 (6-line buses), p. 5 schematic |
| Bits used in the byte | The latch takes the ROM's six low outputs (pins 11–13, 15–17) *(schematic reading)*. The two high bits are unused. | Not shown | S1 p. 5; R6 agrees *(R)* |
| Encoding | Linear, unsigned code on a resistor-ladder DAC. No companding. The content itself was compressed (dynamics flattened) before digitising, for a better S/N ratio and effective resolution. | Same | S1 p. 5; R5, R7 *(R)* call the ladder R-2R |
| Clock / sample rate | The trigger starts a two-gate RC oscillator (no crystal) at about 60 kHz. IC73 divides it by two, so the address clock is **about 30 kHz**. The rate is fixed: there is no tune control. | Same oscillator type, with **TUNE** pots in the cymbal oscillators. S1 gives no frequency. | S1 pp. 4–5; R6 uses ~32 kHz for the hat *(R only)* |
| Sample lengths | One ROM, two ranges, selected by diode-OR logic on the top address lines: **open** = 0x0000–0x5FFF (24,576 samples, ~0.82 s at 30 kHz); **closed** = 0x6000–0x7FFF (8,192 samples, ~0.27 s). The counter stops the clock at the end of the range. | 32,768 samples per voice. The length in seconds scales with TUNE. | S1 p. 5 (stop addresses and address table) *(range split: inference from the table and schematic)* |
| Tune | None | **Clock-rate change**: pitch, length and envelope scale together | S1 p. 4 block diagram, S2 |
| Decay | One analog RC envelope (capacitor C135), driving the VCA through an anti-log stage. CLOSED/OPEN picks the charge path: CH DECAY (VR21, 10 k + 100 k) or OH DECAY (VR23, 100 k + 1 M). The OH path is ten times the resistance, so the two knobs interact slightly. | **No decay knob.** Six address lines feed a second DAC (RA11/RA13). Its output is anti-log tapered and drives the VCA, so the envelope is locked to the read position and follows TUNE. | S1 p. 5; R7 notes the interaction *(R)* |
| VCA | Transistor pair Q84/Q85 with op-amp IC67b | Q71 (crash) / Q69 (ride), wired as a voltage-controlled potentiometer | S1 pp. 4–5 |
| Filter | Low-pass after the VCA (Q80, Q81), then the level amp IC65b | Low-pass (Q75/76, Q77/78), then IC65a/IC64a | S1 p. 4. Cutoffs not stated. R7 *(R)* says "two LPFs in series". |
| Accent | The CPU latches a 6-bit accent code per voice group (IC2–IC9). A resistor array converts it to a 0–5 V level. The hat level (IC7) feeds the hat's VCA/anti-log path. The panel's TOTAL ACCENT knob is read by the CPU. | One cymbal accent level (IC9), shared by crash and ride, enters the anti-log stage | S1 pp. 4–5, S2 |
| Voice structure | OH and CH are one circuit: a new trigger resets the counter, picks the range and picks the decay path. They cannot sound together. | Crash and ride are independent circuits | S1 p. 5 |

**Artefacts that define the 909 PCM sound.**

- **6-bit quantisation.** A full-scale sine at 6 bits has about 38 dB S/N *(computed)*. The flattened content keeps the signal near full scale, so the noise stays masked until the analog envelope pulls signal and noise down together (S1, R5).
- **Zero-order-hold steps at about 30 kHz.** Only simple transistor low-pass stages follow, so DAC images near the clock rate and its multiples are attenuated, not removed *(inference from S1 p. 4)*. On the cymbals, lowering TUNE lowers the clock and pulls the images toward the audible band, which adds grit at low tune *(inference)*.
- **Cymbal envelope locked to position.** Six address bits give a staircase of at most 64 steps before the anti-log stage *(inference: S1 does not say whether it is smoothed)*. Tune therefore never changes the "shape" of a cymbal, only its time scale.
- **A truncated open hat.** When the counter reaches its stop address the clock halts. With a long OH DECAY the sound ends at the end of the ROM range rather than decaying fully *(inference)*.
- **An RC clock.** The rate is not crystal-locked, so it varies between units and drifts with temperature *(inference from the circuit; R7 makes the same point)*. Each hit resets the counter and restarts the gated oscillator (S1 p. 5), so every hit begins phase-aligned.

## 2. TR-707

S3 describes both models: all rhythm sounds come from PCM recordings in ROM, and the circuit splits into a "multiplex" section and a "single sound" section (S3 p. 5).

| Aspect | Multiplex voices (BD1/2, SD1/2, LT, MT, HT, HH, RS/CB, HC/TAMB) | Single-sound voices (crash, ride) | Source |
|---|---|---|---|
| ROM | IC34 + IC35, two HN61256 (32,768 × 8) | IC19 (crash) and IC22 (ride), HN61256, 32 KB each | S3 tables 1–2; R1 |
| ROM size, total | 4 × 32 KB = 128 KB per machine | | S3, R1 |
| Bit depth / encoding | **8-bit linear** codes on IC37 (μPC624C, a DAC08-type multiplying DAC). The content was pre-processed to improve S/N and resolution, so it does not have its natural amplitude. No companding. | **6-bit linear**: ROM bits D2–D7 drive a resistor-array DAC (RA3/RA4) *(bit choice R only, from R1)* | S3 p. 6; R1, R4 |
| Addressing | The MB63H114 gate array (IC30) has eight 13-bit counters. A trigger resets one to 0, and it counts to 0x1FFF and stops. The variation pairs share one counter: BD1/BD2, SD1/SD2, RS/CB and HC/TAMB use the even and odd addresses of one 8 KB block, so only one of each pair sounds at a time. | Dedicated counter per voice. It stops when bit 15 sets, after 32,768 samples. | S3 pp. 5–6; R1, R2 |
| Clock / sample rate | 1.6 MHz master clock, divided to 100/50/25/12.5 kHz outputs. One voice is visited every 40 µs (25 kHz). Effective rates: **25 kHz** for BD, SD, HH, RS/CB and HC/TAMB; **12.5 kHz** for LT, MT and HT. Fixed: no tune. | **25 kHz**, fixed | S3 p. 6 (40 µs); R2; R4 rates *(R)* |
| Sample lengths | BD1, BD2, SD1, SD2, RS, CB, HC and TAMB: 4,096 samples each (164 ms). LT, MT, HT: 8,192 (655 ms). HH: 8,192 (328 ms). | 32,768 samples (1.31 s) | S3 table 1 (4 KB / 8 KB); R4 |
| Tune / decay | **No tune or decay controls.** The panel has a level control per voice, an Accent Level control and master volume. | None | S4 specifications |
| Envelope | One RC envelope generator per voice drives the DAC's **+REF**: the multiplying DAC is the VCA, so the envelope acts *before* the analog output. On a trigger the capacitor charges fast through a shared 100 Ω resistor toward the LEVEL/DYNAMICS voltage, then discharges. Per R1's values, the time constant is 0.047 µF × 2.2–4.7 MΩ, i.e. **τ ≈ 0.10–0.22 s**. | Its own RC envelope and single-transistor VCA, applied **after** the DAC. Per R1, τ ≈ 0.4–0.5 s (1 µF with 390–470 kΩ). | S3 pp. 6–7; R1 |
| Hi-hat | The ROM data has no open/closed distinction. A second VCA (Q22, IC42) with its own envelope shapes the decay, switched by the OPEN/CLOSED signal (Q21). Per R1's values: a high-pass at ~723 Hz before that VCA; open τ ≈ 0.19 s, closed τ ≈ 14 ms. | n/a | S3 p. 7; R1 |
| Accent / velocity | Latch IC31 holds the combined LEVEL/DYNAMICS code. DYNAMICS follows MIDI velocity or the accent setting (two programmed strengths plus the Accent Level control). A 6-bit resistor DAC turns it into the envelope's charge target, so accent scales the peak. | Same accent CV charges the cymbal envelope (C50 for the crash) | S3 pp. 6–7; S4; R1 (6-bit DAC) |
| Output stage | The DAC output is demultiplexed (IC41) into a per-voice sample-and-hold. Each voice then has a band-pass filter tuned per voice (the BD/SD/tom ones have diodes, so their response depends on level), its level fader, a ~15.9 kHz RC low-pass and a fixed pan into the mix. The master outputs are band-limited to ~0.35 Hz–12.4 kHz. Every voice also has its own output jack. | A high-pass at ~339 Hz before the VCA, then the same mixer chain | S3 p. 6 (S/H); R1 for filters and pan *(R only)* |

**Artefacts that define the 707/727 sound.**

- **Envelope before quantisation (multiplex voices).** The envelope sets the DAC reference, so the 8-bit step size shrinks with the envelope. Quantisation noise therefore decays with the sound instead of sitting at a fixed floor (S3 p. 6; *inference on the audible effect*). The codes are offset binary, so the envelope also moves the DAC's mid-scale level. The per-voice band-pass filters remove that slow DC shift, but the fast attack edge can leave a small thump *(inference from R1's DAC model)*.
- **Sample-and-hold steps at 25 kHz, and 12.5 kHz data on the toms.** The only filtering is a ~15.9 kHz low-pass per voice and the ~12.4 kHz output corner. Hold images start at 12.5 kHz (at 6.25 kHz for the toms) and meet only gentle first-order roll-offs, so much of them reaches the output. This likely gives the toms their gritty top *(inference from R1/R2 numbers)*.
- **6-bit cymbals** on the same plan as the 909's: flattened content and an RC envelope after the DAC (S3, R5).
- **Non-linear stages.** The single-transistor VCAs on the hats and cymbals are not linear, and the bass, snare and tom band-pass filters change response with level (R1 notes; *R only*).
- **Interacting envelopes.** Voices triggered together charge through one shared 100 Ω resistor, so simultaneous hits reach slightly different peaks (R1; *R only*, low priority).
- **Fixed digital clocks.** All rates divide one 1.6 MHz master clock, so they are fixed and do not drift apart as an RC clock would *(R2; inference)*. A trigger can land up to one sample period (40–80 µs) before the first sample comes out *(inference from R2)*.

## 3. TR-727

S3 states that the 727 differs from the 707 only in its sound data, some audio-stage component values and a couple of pin connections at IC30. So section 2 applies, with these differences:

| Aspect | TR-727 | Source |
|---|---|---|
| Voices | Hi/Low Bongo; Mute Hi/Open Hi Conga; Low Conga; Hi/Low Timbale; Hi/Low Agogo; Cabasa/Maracas; Short/Long Whistle (the hi-hat slot: closed/open envelope = short/long); Quijada and Star Chime (the two 6-bit cymbal slots) | S3; R1 |
| Pairs on one counter | Hi/Low Bongo, Mute/Open Hi Conga, Hi/Low Agogo, Cabasa/Maracas (even/odd addresses, 4,096 samples each at 25 kHz) | S3 table 1; R4 |
| Rates and lengths | Low Conga 8,192 samples at **25 kHz** (the IC30 pin change; the 707's toms run at 12.5 kHz); Hi/Low Timbale 8,192 at 12.5 kHz; Whistle 8,192 at 25 kHz; Quijada and Star Chime 32,768 each at 25 kHz, 6-bit | R4, R1 *(R only for rates)*; S3 tables 1–2 for ROM sizes |
| Analog differences | Different envelope discharge resistors (decay times), band-pass components and pan resistors | R1 (values from S3's schematic) |
| ROM dumps | The MAME driver lists hashes for the TR-707 and TR-909 sample ROMs but marks the TR-727 voice ROMs as not dumped | R1, R3 |

## 4. TR-8S

### 4.1 Its machine voices are models, not sample tones

In the v3.00 preset list, every 909 hat and cymbal tone (909 Closed HH, 909 Open HH, 909 Crash Cymbal, 909 Ride Cymbal) and every 707 and 727 tone has type **ACB**, Roland's circuit-model engine, not **SAMPLE** (S7). Roland documents no internals for these models. On the TR-8S they gain the common **Tune** and **Decay** that the 707/727 never had (S5 p. 32). For example, its 707 toms take a CTRL "Color" that sets how far the pitch moves, a feature the original lacked (S5 p. 33). Tones named with a slash, such as "707 Bass1/2", keep the original's two-variations-per-slot behaviour as "alternate sounds" (S5 p. 19). So the TR-8S offers no evidence about how a sample-based emulation should behave. It only shows which controls Roland chose to expose.

### 4.2 The sample engine and user import

| Topic | Fact | Source |
|---|---|---|
| File formats | WAV, AIFF | S5 p. 41, S6 p. 17 |
| Sampling frequency | WAV up to 96 kHz; AIFF 44.1, 48 or 96 kHz | S5 p. 41 |
| Bit depth | 8, 16, 24 or 32-bit integer, or 32-bit float | S5 p. 41 |
| Channels | Mono or stereo | S5 p. 41 |
| Length limit | About 180 s per file (at 44.1 kHz mono), less depending on free memory | S5 p. 41, S6 p. 17, S8 |
| Total memory | About 600 s at 44.1 kHz mono; up to 400 user samples | S8; S5 p. 41 |
| Import mechanics | Files go in `ROLAND\TR-8S\SAMPLE\` on the SD card. A whole folder can be imported at once (up to 256 folders of up to 256 files). Files can be auditioned before import. Repeated import and delete fragments memory; an Optimize function compacts it. | S5 pp. 41, 44 |
| Internal format | Not documented. The output DAC is 24-bit/96 kHz. | S8 |
| Per-sample settings (SAMPLE Edit, shared by every kit using the sample) | **Start** and **End** (in steps of 10 samples), **Gain** −18 to +18 dB, Category, Name (16 characters) | S5 p. 42 |
| Per-instrument sample-tone parameters (INST Edit, saved per kit) | **Tune** −128…+127 (common to all tones; unit not documented) and **Coarse Tune** ±24 semitones. **Rate** −1.00…+1.00: +1 is original speed, lower values play slower, 0 does not play, negative values play backwards. **Spread** ±50 (skews pitch between left and right). **Bit Reduce** 0–12. **Attack** 0–255. **Hold Mode**: Whole (play to the end), Time (decay starts after Hold Time 0–255) or Step (decay starts after Hold Step, 0.5–128 steps). **Decay** 0–255. **Filter**: LPF or HPF, with Cutoff, Resonance, envelope Attack/Decay/Depth, and Velocity → filter. | S5 pp. 32–33 |
| Level, pan, effects | Level, Gain ±40 dB, Pan, Reverb and Delay sends, LFO targets, and an INST FX slot per instrument. The **CRUSHER** effect has a resample rate (0–255) and a pre-filter, so it is the only sample-rate reduction offered. | S5 pp. 29–36 |
| Loop | Some *preset* samples are flagged "Loop" (they play repeatedly). SAMPLE Edit has no loop-point parameter for user samples. | S5 pp. 32, 42 |
| Choke | KIT: MUTE: the Open HH sound and any sample tone can be set to be silenced when a chosen other instrument sounds (e.g. CH closes OH). [SHIFT]+[START/STOP] silences samples still ringing after stop. | S5 pp. 29, 11, 55 |
| Dynamics | Velocity-sensitive pads; an accent level (velocity) per step | S5 pp. 5, 19 |

## 5. Cross-machine view of the PCM path

| Machine / voice | Bits | Code | Rate | Samples | Tune | Decay | Envelope position | Accent |
|---|---|---|---|---|---|---|---|---|
| 909 hats | 6 | linear | ~30 kHz RC, fixed | OH 24,576 / CH 8,192 | none | RC, two knobs, anti-log VCA | after DAC | 6-bit level into VCA |
| 909 crash, ride | 6 | linear | RC, TUNE | 32,768 | clock rate | address-derived (follows tune) | after DAC | 6-bit level, shared |
| 707/727 multiplex | 8 | linear (offset binary) | 25 / 12.5 kHz, fixed | 4,096 or 8,192 | none | fixed RC per voice | **before** DAC output (multiplying DAC) | 6-bit level sets RC peak; MIDI velocity |
| 707/727 hat/whistle | 8 | linear | 25 kHz | 8,192 | none | extra RC VCA, open/closed | both | as above |
| 707/727 cymbal slots | 6 | linear | 25 kHz | 32,768 | none | fixed RC | after DAC | as above |
| TR-8S user sample | 8–32, float | PCM WAV/AIFF | ≤ 96 kHz | ≤ ~180 s | Tune, Coarse, Rate | Attack/Hold/Decay | digital | velocity, per-step accent |

## 6. Sample-layer capabilities

"ROM only" means the capability is needed only to emulate the ROM machines. "Both" means it also serves user sample import. "Import only" covers what user import needs that no ROM machine needs.

| Capability | Needed by | Use | Why |
|---|---|---|---|
| Decode an imported sample to the engine rate (WAV/AIFF, 8/16/24/32-bit integer or float, ≤ 96 kHz, mono/stereo) | TR-8S import | **Import only** | S5 p. 41 |
| Playback rate / tune by resampling, so pitch and length move together | 909 cymbal TUNE; TR-8S Tune/Coarse/Rate; the channel's existing `pitch_semitones` | **Both** | A clock-rate change is exactly a resampling (S1). For the 909 cymbal the envelope must follow the same rate (next rows). |
| Native-rate zero-order hold: hold each sample for 1/f_clock, with no interpolation, then the analog filter | 909 (~30 kHz, tunable), 707/727 (25/12.5 kHz) | **ROM only** | The images of a stepped DAC output are part of the sound (sections 1–2) |
| Start and end points; named regions within one sample | 909 OH/CH ranges of one ROM; 707/727 even/odd pairs; TR-8S Start/End | **Both** | S1 p. 5, S3 table 1, S5 p. 42 |
| Bit-depth reduction (quantise to N bits) | 6-bit and 8-bit ROMs; TR-8S Bit Reduce 0–12 | **Both** | One quantiser serves both; ROM emulation needs exact 6/8-bit linear codes |
| Sample-rate reduction (decimate and hold) | Recreating ROM character from new content; TR-8S CRUSHER (an effect) | **Both** | For ROM data, the native-rate hold above covers it |
| Companding (μ-law, A-law or a non-linear DAC curve) | Nothing in scope | **Not needed** | Every ROM path is linear (S1, S3, R4) |
| Decay envelope, RC style: fast charge to an accent-scaled peak, then an exponential discharge | 909 hats (knob), 707/727 (fixed τ per voice), TR-8S Decay | **Both** | S1, S3; today's exponential envelopes are close in kind |
| Envelope applied before quantisation (multiplying DAC) or after it (VCA) | 707/727 multiplex vs. cymbals and 909 | **ROM only** | It sets whether quantisation noise falls with the envelope (section 2) |
| Position-locked envelope: gain as a function of read position (64-step staircase, anti-log) | 909 crash/ride | **ROM only** | S1 p. 5; keeps decay locked to TUNE |
| Per-trigger decay variant: two decay constants chosen by the trigger | 909 OH/CH; 707 open/closed hat; 727 long/short whistle | **ROM only** (choke and alternates cover the user case) | S1, S3 |
| Accent and velocity scale the envelope peak, quantised to 6-bit steps for the ROM machines | All | **Both** | 6-bit accent DACs (S1, S3/R1); TR-8S velocity and per-step accent |
| Attack and hold (time or steps) before decay | TR-8S sample tones | **Import only** | S5 p. 33 |
| Fixed post-filters: high-pass before the VCA (~339/723 Hz), per-voice band-pass, ~15.9 kHz and ~12.4 kHz low-pass, 909 LPF stages | 707/727, 909 | **ROM only** (today's EQ and noise SVF may cover part) | R1, S1; 909 cutoffs need measuring |
| Resonant LPF/HPF with its own envelope and velocity amount | TR-8S sample tones | **Import only** | S5 p. 33 |
| Monophonic retrigger: a new hit restarts from the start point | All machines; TR-8S | **Both** | Counter reset on trigger (S1, S3) |
| Choke by another channel (fade or cut) | 909 CH→OH (inside one circuit); TR-8S KIT: MUTE | **Both** | S1, S5 p. 29; today's `choke()` gives a 10 ms fade |
| Alternate sample per slot, chosen per trigger | 707 BD1/2, SD1/2, RS/CB, HC/TAMB; 727 pairs; TR-8S "/" tones | **Both** | S3 table 1, S5 p. 19 |
| Multi-sample by velocity or knob position | Only when recordings replace ROM data (section 7) | **Both** (content choice) | Recordings bake in the envelope and filter |
| Reverse playback (negative rate) | TR-8S Rate < 0 | **Import only** | S5 p. 33 |
| Loop (repeat between points) | TR-8S preset loop tones only | **Import only** | No ROM machine loops: the counters stop at the end (S1, S3) |
| Stereo samples and spread (left/right pitch skew) | TR-8S | **Import only** | ROM voices are mono with a fixed pan, which today's pan covers |
| Gain trim per sample (±18 dB) | TR-8S | **Import only** | S5 p. 42 |
| Analog non-linearities: transistor VCA curve, level-dependent band-pass | 707/727 | **ROM only**, optional | R1 notes; today's shaper may cover them in a fit |
| Shared-charge envelope interaction between simultaneous voices | 707/727 | **ROM only**, low priority | R1 |
| RC clock tolerance (rate set per unit) | 909 | **ROM only**: a fit parameter, not a feature | S1 (RC oscillator) |

## 7. Audio content each machine voice would need

Facts only. Which option is allowed is the licensing ticket's decision.

| Voice group | With ROM data | With recordings of the machine | With resynthesis or new content |
|---|---|---|---|
| 909 OH/CH | One 32 KB ROM image (6 bits per byte) plus emulation of the clock, RC envelope, anti-log VCA and low-pass. Raw ROM playback lacks the envelope (the content is flattened). | Recordings carry the envelope, filters and that unit's clock. The DECAY knobs then need one recording per knob position, or dividing out an estimated envelope. Accent also needs per-level recordings, or a gain scale if it is a pure gain. | New 6-bit content made the same way (record, compress flat, quantise), played through the emulated chain. Or synthesis from the channel's noise and a metal oscillator section (one of ADR 0002's candidate sections), which is approximate. |
| 909 crash, ride | Two 32 KB images plus the address-derived envelope, anti-log VCA and low-pass | A recording at one TUNE setting, resampled for other settings, reproduces the coupled pitch, length and decay. It does not move the fixed analog filter or the noise floor with it. Recordings at several TUNE settings help. | As for the hats |
| 707/727 multiplex voices | 64 KB (the two voice ROMs) plus the multiplying-DAC envelope, sample-and-hold, per-voice band-pass and low-pass | The original has no tune or decay, so a few recordings per voice cover its whole control space (one per accent strength, plus MIDI velocity steps). A channel tune added beyond the original would also shift the baked-in filter. | New 8-bit content at 25/12.5 kHz through the same chain. Synthesis from today's sections is far from these sounds. |
| 707/727 cymbal slots (incl. Quijada, Star Chime) | Two 32 KB images (6 bits used) plus a high-pass, RC envelope and transistor VCA | As for the multiplex voices: there are no controls beyond accent and level | As for the 909 cymbals |
| TR-8S user samples | n/a | n/a | The user's own files; Pythonic ships no content for them |

A practical consequence across all rows: **ROM data alone is not the sound**. Each machine restores its envelope in analog, so a ROM-data voice still needs the envelope and filter stages above, and a factory patch must fit them to recordings, as the map's fidelity rule requires. Recordings include those stages but freeze the knob positions they were made at.

## Open points

- **909 cymbal clock.** S1 gives the hat oscillator (~60 kHz ÷ 2) but no crash or ride frequency or TUNE range. Measure it from recordings: the spectral images, or the length at known knob positions.
- **909 low-pass cutoffs** and the exact address bits feeding the cymbal envelope DAC need the full schematic pages of S1, or measurement.
- **The OH/CH range split** rests on the address table and the diode-OR wiring in S1 p. 5. Measuring the open and closed lengths on recordings (at maximum decay) would confirm it.
- **707/727 component values** come from R1, which flags a few schematic readings as uncertain (crash envelope resistor, hat envelope capacitors).
