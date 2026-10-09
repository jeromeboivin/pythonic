# Machine recordings available as measurement references

Research for [#27](https://github.com/jeromeboivin/pythonic/issues/27), part of the map [#23](https://github.com/jeromeboivin/pythonic/issues/23).

**Question.** Which real-machine recordings do we have locally to fit and measure factory machine kits against, for each machine and voice, and where are the gaps?

**Scope.** The local sample library at `/mnt/nas/musique/Samples`. Every path below is relative to that folder. This file records paths, counts and measurements only. No audio is copied into the repo. Which machines the TR-8S itself carries is [#24](https://github.com/jeromeboivin/pythonic/issues/24)'s question. This survey covers the six machines the map names (TR-909, TR-808, TR-606, TR-707, TR-727, CR-78) and lists the other Roland-family machines it found.

## Summary

- **TR-909 is well covered, twice over.**
  - `Zenhiser - TR-909. The Drum Machine/` (804 files) covers all 11 voices. Its knob grids are labelled in the file names, it has accent and hard/soft variants, it is 24-bit, and it is clean. It is the primary reference.
  - `Drum Machines/Roland TR-909/Drum Broker/` adds a dense bass drum grid (1,067 files: 9 attack × 11 tune × 11 decay) and a snare grid (430 files: 8 tune × 5–8 tone × about 8 snappy steps). Both are raw 24-bit recordings, but they have no accent variants and several mislabelled names.
- **TR-808: only the bass drum has a knob grid.** `Drum Machines/Roland TR-808/Kick/808BD_T*D*_Orig.wav` holds 32 tone × decay positions at 24-bit. The other 808 voices exist only as one hit per voice (`Drum Samples/Roland TR-808/`, the Hollywood Edge and Soundscan kits). Alternatively they come as 808 sounds sampled from *other* instruments: the rest of `Drum Machines/Roland TR-808/` is tagged R8, JD800, XD5 or unidentified sources.
- **TR-606, TR-707, TR-727, CR-78: one hit per voice, from one set each** (`Drum Samples/Roland …`). These are 16-bit, normalized, and have no knob sweeps or accent variants. Voices are missing:
  - TR-707: the ride.
  - TR-727: both high congas and the second whistle.
  - CR-78: the maracas, and probably the cymbal.
- **Not usable as references:**
  - `808 Samples/`, a generic "808" pack with heavy clipping and no provenance;
  - the `EdgeSounds - Drum Mashines VOL-1` "TR-606/808/909/CR-78 DRUMS" folders, which are General MIDI kits with voices those machines never had;
  - the "Distressed" and "WARPED" 909 sets, which are processed;
  - the 808-named files of other machines (Kawai XD-5, Casio RZ-1, MFB-522, Nord Drum 3, JMizzy).
- **Biggest gaps:**
  - knob sweeps for every TR-808 voice except the bass drum;
  - accent and velocity variants for every machine except the TR-909;
  - any second, independent recording of the TR-606, TR-707, TR-727 and CR-78.

## Grades

| Grade | Meaning |
|---|---|
| **A** | Dry, labelled knob positions, clean onset, natural tail. Fit per knob position. |
| **B** | Dry single hits, knob settings unknown. Fit one patch per voice and measure at that point only. |
| **C** | Usable with care: cut tails, mild clipping, unlabelled sweeps, a sampler-era re-sample, or uncertain provenance. Use for cross-checks, not as the fitting target. |
| **X** | Not a reference: processed, wrong provenance, or a General MIDI render. |

## Sets surveyed

Formats and quality figures come from the probe described under [Method](#method). Some terms used in the table:

- **Near-mono stereo:** two channels that differ by only −30 to −75 dB.
- **Dual mono:** two identical channels.
- **Cut tail:** the file ends while the sound is still above −50 dB relative to its peak.
- **Flat top:** at least 3 consecutive samples within 0.2% of full scale, that is, clipped or limited.

| Set (path) | Machine | Files | Format | Dry? | Quality notes | Grade |
|---|---|---|---|---|---|---|
| `Zenhiser - TR-909. The Drum Machine/` | TR-909 | 804 | 44.1 kHz, 24-bit, dual mono | dry | Onset 1–4 ms into the file. Tails decay naturally to −75…−88 dB with 0 cut. No flat tops. Levels not normalized: accent moves the bass drum peak from −8 to −0.5 dBFS. | **A** |
| `Drum Machines/Roland TR-909/Drum Broker/` (BASS DRUM, SNARE DRUM) | TR-909 | 1,067 + 430 | 44.1 kHz, 24-bit, mono | dry, raw | 25–90 ms pre-roll with noise at −50…−66 dB re peak. Peaks −3…−7 dBFS, not normalized. Tails to about −75 dB, 3 cut. No flat tops. Naming errors, see the TR-909 section. | **A** |
| `Drum Machines/Roland TR-909/Drum Broker/` (Hi Tom, MID TOMS, LOW TOMS, HIHAT, CRASH, RIDE, CLAPS, RIMSHOT) | TR-909 | 40 + 48 + 22 + 13 + 12 + 11 + 1 + 1 | 44.1 kHz, 24-bit, mono | dry, raw | Same recording chain as above. Sweeps are numbered but not labelled. `CRASH/crash-31`, `crash-32`, `RIDE/ride-33` and `ride-34` are 10 ms junk files. | **B** |
| `Drum Machines/Roland TR-909/Drum Broker/WARPED CUSTOM SH_T/` | TR-909 | 58 | 44.1 kHz, 24-bit, mono | processed | Warped, reversed and tape effects. | X |
| `Drum Machines/Roland TR-909/{Kicks,Snare,Toms,HiHat,Cymbals}/` (`Distressed_909_*`) | TR-909 | 103 | 44.1 kHz, 16-bit, dual mono | processed | Peak-normalized to 0 dBFS. Labelled "Distressed". `Snare/` mixes snares, claps and rims. No knob labels. | X |
| `Drum Samples/Roland TR-909/` | TR-909 | 14 | 44.1 kHz, 16-bit, near-mono stereo | dry | Normalized to 0 dBFS. `Bassdrum-02` (40-sample flat top) and `Tom H` (21) are clipped. 2 bass drums end at −48…−50 dB re peak. | B/C |
| `Northstar - New Gold VOL-1/Northstar - New Gold VOL-1/TR909+ZAP-4M/` | TR-909 | 56 of 76 | 44.1 kHz, 16-bit, mono | dry | Sampler-library trims. 23 of the 76 files have a cut tail. Mostly normalized to 0…−1.2 dBFS (cymbals lower). The other 20 files are zaps, rap basses and other synth toms. | C |
| `Hollywood Edge - The Big Bad Beatmachine/Hollywood Edge - The Big Bad Beatmachine/TR-909/` | TR-909 | 13 | 44.1 kHz, 16-bit, mono | dry | 6 of 13 tails cut. "909 CONGA" is not a 909 voice and is clipped. | C |
| `USB Soundscan vol.20 - Fresh Disco House I/D/10-TR 909/` (byte-identical copy in `Frash Disco/D/10-TR 909/`) | TR-909 | 13 | 44.1 kHz, 16-bit, mono | dry | Normalized. 5 tails cut. | C |
| `EdgeSounds - Drum Mashines VOL-1/EdgeSounds - Drum Mashines VOL-1/TR-909 DRUMS/` | TR-909 | 53 | 44.1 kHz, 16-bit, stereo | processed | A General MIDI kit: cuica, surdo, whistle, vibraslap and so on. True stereo (side −8 dB). | X |
| `Drum Machines/Roland TR-808/Kick/` (`808BD_T{1..11}D{1..11}_{Orig,Tape,TapeSat,X,X2}`) | TR-808 | 162 | 44.1 kHz, 24-bit, mono | `Orig` dry; `Tape`, `TapeSat`, `X`, `X2` processed | `Orig` peaks are normalized to about −0.15 dBFS. Files start on the onset. Tails decay naturally to about −75 dB, then fade to silence. `TapeSat`: 7 files with 6–12-sample flat tops. `X` and `X2` change the harmonics and the click. | **A** (`Orig`) |
| `Drum Machines/Roland TR-808/Kick/` (other names) | TR-808 (via others) | 54 | 44.1 kHz, 24-bit, mono | mixed | 24 tagged `-R8`, 20 `-Kult`, 3 `-JD800`. Also 7 others: `808BDLong1`, slides, distortion and reversed hits. | C/X |
| `Drum Machines/Roland TR-808/{Snares,Toms,Hats,Hi hat,Open Hat,Clap,Clave,Conga,Cowbell,Cymbal,Rim,Triangle}/` | TR-808 (via others) | 177 | 44.1 kHz, mostly 24-bit mono | mixed | Tags by source: R8 93, Kult 56, JD800 11, SHD 12, XD5 3, DA4 2. Each comes in up to five chain variants (`aOrig`, `C2A`, `C2S`, `T1A`, `T1S`). `Hi hat/` and `Open Hat/` repeat files from `Hats/`. Two Kult toms have 20–22-sample flat tops. | C |
| `Drum Samples/Roland TR-808/` | TR-808 | 18 | 44.1 kHz, 16-bit, near-mono stereo | dry | Normalized. 8 files have short flat tops of 3–8 samples, all 5 bass drums among them. `Bassdrum-03` and `Claves` have cut tails. | B |
| `Hollywood Edge - The Big Bad Beatmachine/Hollywood Edge - The Big Bad Beatmachine/TR-808/` | TR-808 | 18 | 44.1 kHz, 16-bit, mono | dry | Not normalized. `808 BOOM 1` and `808 BOOM 2` are clipped (188- and 97-sample flat tops). 2 tails cut. The only local 808 congas and maracas besides `Drum Samples`. | B/C |
| `USB Soundscan vol.20 - Fresh Disco House I/D/09-TR 808/` (byte-identical copy in `Frash Disco/D/09-TR 808/`) | TR-808 | 18 | 44.1 kHz, 16-bit, mono | dry | Normalized. 9 files with flat tops, 4 of them 10+ samples. 7 tails cut. | C |
| `808 Samples/` | unknown | 171 | 44.1 kHz, 16-bit, mono and stereo (2 files at 22.05 kHz, 8-bit) | unknown | Kick 23, snare 49, hi-hat 40, clap 16, tom 14, cymbal 5, bass 24. 15 of the 23 kicks are clipped, with flat tops up to 221 samples. No provenance. | X |
| `EdgeSounds - …/TR-808 DRUMS/` | TR-808 | 58 | as above | processed | General MIDI kit. | X |
| `Drum Samples/Roland TR-606/` | TR-606 | 7 | 44.1 kHz, 16-bit, near-mono stereo | dry | Normalized. 1.5–13 ms lead. Natural tails, 0 cut. | B |
| `EdgeSounds - …/TR-606 DRUMS/` | TR-606 | 58 | as above | processed | General MIDI kit. | X |
| `Drum Samples/Roland TR-707/` | TR-707 | 14 | 44.1 kHz, 16-bit, near-mono stereo | dry | Normalized. Clean tails. The toms start 11–14 ms into the file. | B |
| `Drum Samples/Roland TR-727/` | TR-727 | 12 | 44.1 kHz, 16-bit, near-mono stereo | dry | Normalized. Clean tails. 4–25 ms lead. | B |
| `Hollywood Edge - The Big Bad Beatmachine/Hollywood Edge - The Big Bad Beatmachine/TR-727/` | TR-727 | 8 (+1 `.part`) | 44.1 kHz, 16-bit, mono | dry | 5 of 8 tails cut. `727 M.HI.wav.part` is an unreadable partial download. `PAD DRUM` and `ZITHER` are not 727 voices. | C |
| `Drum Samples/Roland CompuRhythm-78/` | CR-78 | 20 | 44.1 kHz, 16-bit, near-mono stereo | dry | Normalized. Every file starts exactly on the onset. Cut tails on cowbell, congas and woodblocks (−43…−49 dB at the end). `Woodblock-02` and `-03` are 14 ms long. | B/C |
| `EdgeSounds - …/CR-78 DRUMS/` | CR-78 | 57 | as above | processed | General MIDI kit. | X |
| `Drum Hardware/Hits/Roland TR-8/` | TR-8 (Roland's own models) | 41 | 44.1 kHz, 24-bit, dual mono | dry | Normalized. Clean. Which TR-8 kit each hit comes from is not labelled. 57 beat loops sit in `Drum Hardware/Beats/Roland TR-8/`. | C (model, not machine) |

## Coverage matrix

The best set per voice comes first. "Files" counts that set; a "+" names a secondary set. Knob positions use the file names' own numbering, where 1 is fully left unless noted.

### TR-909

| Voice | Best set: files | Knob coverage | Accent / velocity | Grade |
|---|---|---|---|---|
| BD | Zenhiser: 375 | Tune × Attack × Decay at positions 1, 3, 6, 9, 11 each (125 positions, complete) | Accent knob at 1, 6, 11 (peak −8.0 / −3.3 / −0.5 dBFS) | A |
| BD | + Drum Broker: 1,067 | Attack 1–9 × Tuning 1–11 × Decay 1–11 (1,067 of 1,089 positions; ATTACK 2 and 5 lack D11) | none | A |
| SD | Zenhiser: 210 | Tune × Tone × Snappy at 1, 3, 6, 9, 11 (105 positions; Snappy 1 only at Tone 1) | Hard and Soft at every position (about 4.4 dB apart) | A |
| SD | + Drum Broker: 430 | Tuning 1–8 × Tone 1…5–8 (the count varies per tuning) × Snappy about 8 steps | none | A |
| LT, MT, HT | Zenhiser: 144 | Tune 1, 3, 6, 9, 11 × Decay 1, 6, 11 (44 of 45 positions; one MT position missing) | Hard and Soft, some with 2 takes | A |
| LT, MT, HT | + Drum Broker: 22 + 48 + 40 | Unlabelled. HT is about five runs of up to 8 tune steps (about 152 → 81 Hz), with decay shortening from run to run. LT and MT are irregular. | none | B |
| RS | Zenhiser: 8 | none (no knob on the machine) | Hard ×4, Soft ×4 | A |
| CP | Zenhiser: 9 | none (no knob on the machine) | Hard ×5, Soft ×4 | A |
| CH | Zenhiser: 18 | Decay 1, 3, 6, 9, 11 | Hard and Soft (1–2 takes) | A |
| OH | Zenhiser: 20 | Decay 1, 3, 6, 9, 11 | Hard ×2, Soft ×2 | A |
| Crash | Zenhiser: 10 | Tune 1, 3, 6, 9, 11 (A and B takes) | none | A |
| Ride | Zenhiser: 10 | Tune 1, 3, 6, 9, 11 (A and B takes) | none | A |
| HH (CH and OH unlabelled) | + Drum Broker `HIHAT`: 13 | Unlabelled. Length to −60 dB runs from 167 ms down to 57 ms. | none | B |

The two independent TR-909 bass drum grids agree. The body pitch spans 48–75 Hz in Zenhiser (Tune 1 → 11) and 51–75 Hz in Drum Broker (TUNING 11 → 1). This makes them a useful cross-check of each other.

### TR-808

| Voice | Best set: files | Knob coverage | Accent / velocity | Grade |
|---|---|---|---|---|
| BD | `Roland TR-808/Kick` `_Orig`: 32 | Tone × Decay at 1, 3, 5, 7, 9, 11: 32 of 36 positions. T1D1 is missing entirely; T11 D1, D9 and D11 exist only as processed variants. | none (normalized) | A |
| BD | + `Drum Samples`: 5; Hollywood Edge: 2; Soundscan: 2 | Unlabelled. The 5 `Drum Samples` hits are 5 different decays (113–1,452 ms). | none | B/C |
| SD | `Drum Samples`: 1; Hollywood Edge: 2; Soundscan: 2 | none | none | B |
| LT, MT, HT | `Drum Samples`: 3; Hollywood Edge: 3 | none | none | B |
| LC, MC, HC (congas) | Hollywood Edge: 3 | none | none | B/C |
| RS | `Drum Samples`: 1; Hollywood Edge: 1 | none | none | B |
| CP | `Drum Samples`: 1; Hollywood Edge: 1 | none | none | B |
| CL (claves) | `Drum Samples`: 1 (cut tail); Hollywood Edge: 1 | none | none | B/C |
| MA (maracas) | Hollywood Edge: 1; `Drum Samples` `Cabasa`: 1 | none | none | B/C |
| CB | `Drum Samples`: 1; Hollywood Edge: 1 | none | none | B |
| CY | `Drum Samples` `Crash-01`, `-02`: 2; Hollywood Edge: 1 | none (two lengths) | none | B |
| CH | `Drum Samples`: 1; Hollywood Edge: 1 | none | none | B |
| OH | `Drum Samples`: 1; Hollywood Edge: 1 | none | none | B |

The 808 sounds of other instruments in `Roland TR-808/` (R8 tags, which name Roland R-8 808 samples; JD800; XD5; and the unidentified Kult, SHD and DA4) cover snare, toms, congas, cowbell, cymbal, hats, clap, rim and claves in several variants. They are re-samples or emulations, not the TR-808, so they are grade C at best.

### TR-606

| Voice | Set: files | Knob coverage | Accent / velocity | Grade |
|---|---|---|---|---|
| BD, SD, LT, HT, CH, OH, CY | `Drum Samples/Roland TR-606`: 1 each (7) | none | none | B |

### TR-707

| Voice | Set: files | Knob coverage | Accent / velocity | Grade |
|---|---|---|---|---|
| BD 1, BD 2, SD 1, SD 2 | `Drum Samples/Roland TR-707`: 1 each | none | none | B |
| LT, MT, HT, RS, CB, CP, TAMB, CH, OH, Crash | `Drum Samples/Roland TR-707`: 1 each | none | none | B |
| Ride | none | none | none | **gap** |

### TR-727

| Voice | Set: files | Knob coverage | Accent / velocity | Grade |
|---|---|---|---|---|
| Hi/Lo Bongo, Lo Conga, Hi/Lo Timbale, Hi/Lo Agogo, Cabasa, Maracas, Quijada, Star Chime, one Whistle | `Drum Samples/Roland TR-727`: 1 each (12) | none | none | B |
| Second whistle (short and long) | Hollywood Edge `727 WHIS.1`, `WHIS.2` (`WHIS.2` cut at −14 dB) | none | none | C |
| Mute Hi Conga, Open Hi Conga | none usable. Hollywood Edge `727 M.HI.wav.part` is an unreadable partial file; `727 M.LO` has a cut tail. | none | none | **gap** |

### CR-78

| Voice | Set: files | Knob coverage | Accent / velocity | Grade |
|---|---|---|---|---|
| BD, SD, Cowbell, Tambourine | `Drum Samples/Roland CompuRhythm-78`: 1 each | none | none | B |
| Hi-hat | `Hat Closed-01`, `-02`: 2 | none | none | B |
| Cymbal | `Hat Open-01`, `-02` (about 0.5 s, probably the cymbal) | none | none | B/C |
| Bongos and conga | `Conga H`, `M`, `L`: 3 | none | none | B/C (cut tails) |
| Guiro (long and short) | `Quid-01` to `-04`: 4 | none | none | B |
| Claves and rim shot | `Woodblock-01` to `-04`: 4 (two are 14 ms long) | none | none | C |
| Metal beat | `Hit` (probably) | none | none | C |
| Maracas | none | none | none | **gap** |

### Other Roland-family machines found

These are all one hit per voice, from `Drum Samples/` (44.1 kHz, 16-bit, near-mono stereo, normalized) unless noted:

- TR-505: 16 files at 22.05 kHz;
- TR-626: 30;
- CompuRhythm 8000: 13;
- CompuRhythm 1000: 15;
- Boss DR-55: 13;
- Boss DR-110: 7;
- Boss DR-550: 47;
- R-8: 52, plus 76 in `EdgeSounds …/R-8 DRUMS`;
- TR-8 (Roland's models): 41 in `Drum Hardware/Hits/Roland TR-8`.

None has knob sweeps or accent variants.

## Recommended reference set per machine

### TR-909

Fit to `Zenhiser - TR-909. The Drum Machine/`, all 11 voices. Densify the bass drum and snare with `Drum Machines/Roland TR-909/Drum Broker/BASS DRUM/` and `…/SNARE DRUM/`.

Caveats:

- **Zenhiser:**
  - Knob positions are in 5 steps (1, 3, 6, 9, 11), not continuous.
  - The files are stereo with identical channels; mix them to mono.
  - "Hard" and "Soft" are the only velocity layers below the bass drum's accent knob.
- **Drum Broker bass drum naming:**
  - Both `ATTACK 1 - Slowest` and `ATTACK 9 - slowest` say "slowest". Measured, the click rises steadily from 1 to 9: the 2 kHz high-passed peak goes from −15.8 to −11.3 dB re peak, and the overall peak from −6.9 to −4.1 dBFS. So 1 is the least attack and 9 the most.
  - Files are `BASS DRUM A<attack> D<decay>_<tuning>.wav`. Decay 1 → 11 lengthens the −40 dB time from 102 to 358 ms. Tuning 1 is the highest pitch (75 Hz body) and 11 the lowest (51 Hz).
  - `D91` in `ATTACK 1` is D1.
  - `ATTACK 9 - slowest/TUNING 11 - lowest/` holds files named `A8` that belong to attack 9.
  - `ATTACK 2` and `ATTACK 5` have no D11.
- **Drum Broker snare naming:**
  - Files are `SN  Tuning <t> snappy <n>_<i>.wav`, where `<n>` repeats the TONE folder number, not a snappy setting.
  - The index `<i>` sweeps SNAPPY: `_01` is full noise (HF share 0.87) and `_07`/`_08` is none (below 0.01).
  - A few folders carry extra takes past `_08` that do not follow the sweep.
  - TONE 1 → 7 lengthens the noise tail (−40 dB at 120 → 266 ms); the single TONE 8 folder, in TUNING 7, does not follow. TUNING 1 → 8 raises the body from 118 to 228 Hz.
  - The number of TONE folders differs per tuning (5 to 8), so "TONE 4" is not the same knob angle in every tuning.
- **Drum Broker recordings:** they have 25–90 ms of noisy pre-roll (trim to the onset, as the local fitter already does). They carry no accent.
- **Toms and hats:** the Drum Broker sweeps are numbered but not labelled, so positions must be inferred from the audio.
- **Junk files:** skip `CRASH/crash-31`, `crash-32`, `RIDE/ride-33`, `ride-34` and `Z CHAIN/` (photos).

### TR-808

For the bass drum, fit to `Drum Machines/Roland TR-808/Kick/808BD_T*D*_Orig.wav` (32 positions). T sweeps TONE: the 1 kHz high-passed click rises from −20.6 to −5.1 dB re peak while the body stays at 52 Hz. D sweeps DECAY: the −40 dB time goes from 62 to 1,010 ms.

For the other voices, fit one patch per voice to `Drum Samples/Roland TR-808/`. Cross-check with `Hollywood Edge …/TR-808/`, which is also the only source of the congas.

Caveats:

- The `Orig` files are peak-normalized, so absolute level and accent are lost.
- `Tape`, `TapeSat`, `X` and `X2` are processed versions of the same hits. They suit testing a saturation or vintage section, not fitting the voice.
- Hold the rest of `Roland TR-808/` (R8, JD800, XD5, Kult, SHD, DA4 tags) to grade C. Use it only if its provenance is confirmed.
- The `Drum Samples` hits are normalized and lightly clipped (3–8-sample flat tops on all 5 bass drums). Their knob settings are unknown.

### TR-606

`Drum Samples/Roland TR-606/`, one hit per voice, 7 voices. Their knob settings are unknown and there is no accent. There is no second source.

### TR-707

`Drum Samples/Roland TR-707/`, 14 of the 15 voices, one hit each. As PCM voices these suit the sample layer directly. Nothing local covers the ride or accent.

### TR-727

`Drum Samples/Roland TR-727/`, 12 voices. The Hollywood Edge TR-727 folder adds a second whistle, with cut tails. The high congas are missing.

### CR-78

`Drum Samples/Roland CompuRhythm-78/`, 20 files. Several tails are trimmed short, and every file starts exactly on the onset, so the attack may have been trimmed. The voice names are generic (Quid = guiro, Woodblock = claves or rim shot, Hit = probably the metal beat).

### TR-8 models

`Drum Hardware/Hits/Roland TR-8/` can show what Roland's own models of these machines sound like. It is not a measurement of the originals.

## Gaps

- **TR-808:**
  - no knob sweeps for any voice but the bass drum: snare tone and snappy, tom and conga tuning, cymbal tone and decay, open-hat decay;
  - no accent variants;
  - the snare, toms, hats, cymbal and claves each rest on 1–3 normalized single hits.
- **TR-606:** a single source of 7 single hits; no accent variants.
- **TR-707:** the ride is missing; no accent variants.
- **TR-727:** Mute Hi Conga and Open Hi Conga are missing (the only candidate is a broken `.part` file); only one clean whistle; no accent variants.
- **CR-78:** the maracas are missing, and the cymbal is uncertain; the accent and metal-beat variants are unclear; onsets and tails are trimmed.
- **All machines but the TR-909:** no accent or velocity layers, so accent behaviour cannot be measured from local recordings.
- **TR-909:** close to complete. Remaining limits:
  - Zenhiser's 5-step knob grid;
  - one missing mid-tom position;
  - no accent in Drum Broker;
  - the toms and hats in Drum Broker are unlabelled.

Recording new references, for example from a real machine or a TR-8S at labelled knob positions, would be the way to close the TR-808 sweep and accent gaps. Whether to do that is for the measuring ticket to decide.

## Not machine recordings

These folders carry machine names but are not usable references for those machines:

- **`808 Samples/`:** a generic "808" sample pack. It has no provenance, mixed formats, and heavy clipping.
- **`EdgeSounds - Drum Mashines VOL-1/…/{TR-606,TR-808,TR-909,CR-78} DRUMS/`:** General MIDI kits named after the machines. They include cuica, surdo, whistle, vibraslap, splash and other voices that none of these machines have, and they are in true stereo.
- **808-named sounds of other instruments:**
  - `Drum Machines/Kawai XD-5/XD5_808*` (136 files);
  - `Drum Machines/Casio RZ-1/RZ1_BD808*`;
  - `Drum Machines/MFB-522/`;
  - `Nord Drum 3/808/`;
  - the `808s` folders under `JMizzy The Best Drum Kits/`.
- **Processed 909 sets:** `Drum Machines/Roland TR-909/Drum Broker/WARPED CUSTOM SH_T/` and the `Distressed_909_*` sets in `Drum Machines/Roland TR-909/{Kicks,Snare,Toms,HiHat,Cymbals}/`.
- **Acoustic kits:** `Northstar - Drumscapes Roland/`, despite the name. It has one `PRC_808 Cowbell`.

## Method

### Finding the folders

- **Directory scan:** every directory under `/mnt/nas/musique/Samples` down to depth 4 (8,319 directories) was searched for machine names: 606, 707, 727, 808, 909, CR-78, CompuRhythm, TR-8, Roland, rhythm and drum machine.
- **File listings:** the matching folders, plus `Drum Machines/`, `Drum Samples/` and `808 Samples/`, were listed file by file.
- **Exclusions:** `.asd` analysis files were skipped.
- **Duplicates:** the two Soundscan and "Frash Disco" kit folders were compared with `cmp`; all 35 files are byte-identical.

### First probe

3,783 audio files were read once each, from the share into memory. For each file the probe recorded:

- the header: sample rate, bit depth and channels;
- the peak in dBFS;
- the DC offset;
- the lead time before the onset, the first sample above −40 dB re peak;
- the pre-onset noise level;
- the RMS of the last 10 and 50 ms relative to the peak, to find cut tails;
- the side-to-mid energy ratio, to tell dual mono from processed stereo;
- the time from the onset to −60 dB.

### Second probe

3,458 files from the candidate sets were measured for:

- runs of samples within 0.2% of full scale, to find flat tops;
- the −20 and −40 dB decay times from 2 ms energy frames;
- the strongest partial between 25 and 600 Hz over 20–120 ms after the onset, as the body pitch;
- the share of energy above 2 kHz;
- the spectral centroid.

### Click measurements

For the knob-direction checks, the click was measured as the peak of a 1 kHz (TR-808) or 2 kHz (TR-909) high-passed signal in the first 5 ms. This was done along fixed rows of each grid:

- TR-909: TUNING 5, D5, ATTACK 1–9;
- TR-808: D7 `Orig`, T1–T11.

### Reproduction

The probe scripts were throwaway and are not in the repo. Every figure above can be re-derived with `soundfile` and `numpy` from the paths and definitions given here.
