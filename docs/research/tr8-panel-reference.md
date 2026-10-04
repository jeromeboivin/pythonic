# TR-8 panel design reference

Research for [#3](https://github.com/jeromeboivin/pythonic/issues/3), part of the GUI map [#1](https://github.com/jeromeboivin/pythonic/issues/1).

**Question.** What does the Roland TR-8 (AIRA "Rhythm Performer", 2014) front panel look like, in enough detail to design a TR-8-*inspired* panel for Pythonic? We want the look and feel, not a replica: no Roland name, logo, wordmark or trade dress copied 1:1.

**Scope.** This covers the original TR-8. The TR-8S successor is cited only where its official manual shows a step-entry convention Pythonic also needs (sub-steps). This file contains no Roland images. Every visual claim links to its source.

## Sources

Trust levels: **P** = primary (Roland-published), **R** = reputable review or Roland-affiliated distributor, used only to fill gaps and marked inline.

| ID | Source | Trust | Used for |
|----|--------|-------|----------|
| S1 | [TR-8 Owner's Manual (Roland, 2014, single A3 sheet)](https://static.roland.com/assets/media/pdf/TR-8_OM.pdf) | P | Panel descriptions 1-17 and their diagram, all control names, TR-REC/INST-REC procedures, system settings, dimensions |
| S2 | [7X7-TR8 Drum Machine Expansion Operation Guide (Roland, rev. e03)](http://cdn.roland.com/assets/media/pdf/7X7-TR8_manual_e03_W.pdf) | P | Firmware 1.10-1.20 additions: weak beats, weak accents, flam, tone-family colours, scale/timing options |
| S3 | [TR-8 product page](https://www.roland.com/global/products/tr-8/) and [features page](https://www.roland.com/global/products/tr-8/features/) | P | Marketing spec: full-colour step pads, RGB buttons, display, rolls, 32-step patterns |
| S4 | [TR-8 System Program update history](https://www.roland.com/us/support/by_product/tr-8/updates_drivers/9d8e26ad-8810-479e-84d7-13d3f1362497/) | P | Firmware 1.50-1.61 (STEP LOOP added in 1.60) |
| S5 | [TR-8S Owner's Manual (Roland, eng03)](https://static.roland.com/assets/media/pdf/TR-8S_eng03_W.pdf) | P | Successor's sub-step / flam / weak-beat entry and A-H variations, used for comparison only |
| R1 | [Sound On Sound review, "Roland TR8 Rhythm Performer"](https://www.soundonsound.com/reviews/roland-tr8-rhythm-performer) | R | Physical finish, contextual LED colours |
| R2 | [MusicRadar review, "Roland Aira TR-8"](https://www.musicradar.com/reviews/tech/roland-aira-tr-8-594315) | R | Brushed black panel, green trim, green backlighting |
| R3 | [Roland Australia blog, "The Ultimate Guide to the AIRA TR-8"](https://rolandcorp.com.au/blog/ultimate-guide-aira-tr-8) | R | Step-view colour meanings (distributor blog, not a manual) |

## 1. Overall layout

The unit is 400 x 260 x 65 mm and 1.9 kg (S1, Main Specifications). Seen from above in landscape orientation, the panel has three vertical zones (S1 diagram, items 1-17):

```
+-------+-------------------------------------------------------------+-----------+
| LEFT  |  FX ROW:  ACCENT | REVERB      | DELAY        | EXTERNAL IN | SCATTER   |
| COL   |  (1 knob+STEP) (3 knobs+STEP) (3 knobs+STEP) (2 knobs+STEP)| big knob, |
|       +-------------------------------------------------------------+ 10 LEDs,  |
| VOLUME|  INSTRUMENT STRIPS (INST edit, 11)                          | DEPTH, ON |
| CLEAR |  BD   | SD   | LT MT HT | RS HC | CH OH | CC RC              +-----------+
| MODE  |  2x2  | 2x2  | knobs: TUNE over DECAY per instrument       | 4-char    |
| BTNS  |  knobs| knobs|                                              | 7-seg LED |
|       |  11 vertical LEVEL faders, one per instrument               +-----------+
|       +-------------------------------------------------------------+ TEMPO     |
|       |  INST select buttons: BD SD LT MT HT RS HC CH OH CC RC      | big knob, |
| LAST  +-------------------------------------------------------------+ SHUFFLE,  |
| STEP  |  SCALE button + 4 LEDs, with printed beat-grouping guides   | FINE, TAP |
| SCALE +-------------------------------------------------------------+-----------+
| A  B  |  16 STEP PADS (14), in 4 colour groups of 4, captions below              |
| START/STOP (large)                                                               |
+---------------------------------------------------------------------------------+
```

Notes on the arrangement (S1 diagram):

- The **step pad row spans almost the full width** at the bottom, under both the instrument area and the right-hand column. The large START/STOP button sits at bottom left, in line with the pads.
- The **effects/send row runs across the top** of the instrument area. Each effect block ends in its own small `[STEP]` button, which re-targets the step pads to that effect.
- **Instrument strips are grouped by thin vertical dividers** into BD | SD | LT-MT-HT | RS-HC | CH-OH | CC-RC. BD and SD each get a double-width strip.
- The **right column** holds the performance and time controls: SCATTER at the top, the display in the middle, TEMPO at the bottom.
- The **left column** holds global and mode controls stacked vertically.

## 2. Sections and controls

Names are as printed in S1. The numbers are S1's callout numbers.

| # | Section | Controls | Function (S1) |
|---|---------|----------|---------------|
| 1 | VOLUME | 1 knob | Master output (MIX OUT + PHONES) |
| 2 | CLEAR | 1 round button | Erase an instrument's notes (hold during playback = erase that region) or delete a pattern |
| 3 | Mode buttons | `[TR-REC]`, `[PTN SELECT]`, `[INST PLAY]`, `[INST REC]`, plus a "DRUM SELECT" pair `[KIT]` `[INST]` | Set what the 16 pads do |
| 4 | LAST STEP | 1 button | Hold + pad = pattern length 1-16 |
| 5 | Variation | `[A]` `[B]` | Select variation; both lit = A then B (AB chain) |
| 6 | START/STOP | 1 large button | Transport |
| 7 | ACCENT | `[LEVEL]` knob, `[STEP]` button | Accent amount; STEP lets the pads set the accent steps |
| 8 | REVERB | `[LEVEL]` `[TIME]` `[GATE]` knobs, `[STEP]` button | Per-step reverb send |
| 9 | DELAY | `[LEVEL]` `[TIME]` `[FEEDBACK]` knobs, `[STEP]` button | Per-step delay send |
| 10 | EXTERNAL IN | `[LEVEL]` `[SIDE CHAIN]` knobs, `[STEP]` button | External audio level; pattern-driven ducking; STEP picks the ducking steps |
| 11 | INST edit | Per-instrument knobs + 11 `[LEVEL]` faders (see section 3) | Tone shaping |
| 12 | INST select | 11 buttons labelled BD SD LT MT HT RS HC CH OH CC RC | Choose the instrument for TR-REC or for tone change |
| 13 | SCALE | `[SCALE]` button + 4 LEDs | Step note value: 8th triplet (3 steps/beat), 16th triplet (6), 16th (4), 32nd (8) |
| 14 | Pads | 16 pads: [1]-[11] INST, [12]-[15] ROLL, [16] MUTE | Mode-dependent (see section 6) |
| 15 | SCATTER | Large `[SCATTER]` knob with a ring of 10 numbered positions, `[DEPTH]` and `[ON]` buttons | Type 1-10; with DEPTH lit the knob sets depth instead |
| 16 | Display | 7-segment, 4-character LED (S3) | Tempo, e.g. `128.0`; also shows setting codes (`rSt`, `C10`...) |
| 17 | TEMPO | Large `[TEMPO]` knob, `[SHUFFLE]` and `[FINE]` knobs, `[TAP]` button | Tempo 40-300 BPM (S3), swing, fine tempo, tap |

Shift-style combinations reuse existing controls, so the panel needs no SHIFT key (S1):

- Hold an INST select button and turn TEMPO to set pan (L64-0-R63).
- Hold `[KIT]` + an effect `[STEP]` button, then press INST select buttons, to toggle each instrument's reverb or delay send.
- Hold `[PTN SELECT]` + SCATTER `[ON]` to generate a random pattern.

## 3. Instrument strips (INST edit + INST select)

11 instruments, 11 faders, 26 knobs and 11 select buttons (S1, items 11-12 and diagram):

| Strip group | Instruments | Knobs per instrument | Fader | Select button |
|-------------|-------------|----------------------|-------|---------------|
| Bass drum (double width) | BD | 4, as a 2x2 block: TUNE, ATTACK / COMP, DECAY | 1 LEVEL | BD |
| Snare (double width) | SD | 4, as a 2x2 block: TUNE, SNAPPY / COMP, DECAY | 1 LEVEL | SD |
| Toms | LT, MT, HT | 2 stacked: TUNE over DECAY | 1 each | LT, MT, HT |
| Rim / clap | RS, HC | 2 stacked: TUNE over DECAY | 1 each | RS, HC |
| Hi-hats | CH, OH | 2 stacked: TUNE over DECAY | 1 each | CH, OH |
| Cymbals | CC, RC | 2 stacked: TUNE over DECAY | 1 each | CC, RC |

Each strip reads top to bottom as: name header, knob(s), fader, select button. The headers spell out the full name (Bass Drum, Snare Drum, Low Tom, ... Ride Cymbal). The select buttons below use two-letter abbreviations. Each instrument slot can load a different "tone" (808/909 variant, and after the expansion 707/727/606) through DRUM SELECT `[INST]` (S1, S2). The strip layout stays fixed whatever tone is loaded. S1 notes that "for some tones, there might not be an effect" from a given knob.

## 4. Colour palette

The panel colours come from reviews (R1, R2), because Roland's manual diagram is greyscale. The **hex values below are approximate design targets eyeballed for a TR-8-*inspired* scheme. They were not measured from the hardware.** Use them as a starting point, not as a match.

| Role | Description | Source | Approx. hex |
|------|-------------|--------|-------------|
| Panel face | Black, brushed-aluminium texture | R2 ("black brushed aluminium front panel"); R1 ("black plastic" body) | `#17181A` with a faint horizontal brush to `#222326` |
| Section boxes / dividers | Slightly lighter outlines grouping each section (visible in S1 diagram) | S1 diagram | `#3A3C40` |
| Body trim | Day-glo green frame around the panel | R1, R2 | `#3DFF6E` (neon green) |
| Backlighting of function buttons and fader slots | Uniform green glow | R1, R2 | `#2EE85C` lit / `#0F3A1C` unlit |
| Label print | Light text on the dark panel; section titles in small caps | S1 diagram | `#E8E8E4` labels, `#9A9C9F` secondary |
| Knobs | Black caps with a white pointer line | S1 diagram | `#0E0E0F` cap, `#F2F2F2` pointer |
| Display | Red 7-segment LED | Common hardware convention; colour **not stated in S1/S3** | `#FF3B2F` (unverified) |

### Step pad colours

The 16 pads are RGB ("full-color LEDs", "RGB buttons", S3). Their colours depend on context:

| Context | Colour meaning | Source | Approx. hex |
|---------|----------------|--------|-------------|
| Idle / power-up "home" look | Pads 1-4 red, 5-8 orange, 9-12 yellow, 13-16 white: the classic 4 x 4 grouping that marks beats. The S1 diagram shows the four groups in four distinct tones. | R1 ("808-styled red, orange, yellow and white"); S1 diagram (4 tonal groups) | red `#E8352A`, orange `#F2862E`, yellow `#F4CF3A`, white `#F3EFE3` |
| TR-REC: steps that play | Lit red | R3 ("red illumination marks programmed steps") | `#FF2A1F` |
| TR-REC: empty steps | Dark | R3 | `#1E1F21` |
| TR-REC: weak beat | Same colour, dimly lit | S2 ("The pad lights dimly") | `#7A1A14` |
| Flam step | Purple | S2 ("The pad is lit purple") | `#9B4DFF` |
| Beat or scale markers | Half-lit white or blue keys mark beat divisions in the background | R1 ("half-lit white keys"); R3 ("blue markers" for the first count) | `#5A5A5A` / `#2F6BFF` |
| PTN SELECT | Selected pattern yellow, playing pattern orange; selected pad blinks | R1; S1 (blink behaviour) | `#F4CF3A` / `#F2862E` |
| KIT select | Blue | R1 | `#2F7BFF` |
| INST (tone) select | Colour shows the tone family: **pink = 808, yellow = 909, orange = 707, blue = 727, white = 606** | S2 (verbatim colour list) | pink `#FF5FA8`, yellow `#F4CF3A`, orange `#F2862E`, blue `#2F7BFF`, white `#F3EFE3` |
| Shortened pattern | Green keys show a length other than 16; LAST STEP lights | R1 | `#3DFF6E` |
| Mutes | Violet | R1 | `#8A3DFF` |

Conflicts noted:

- R3 (distributor blog) says flam is pink. S2 (the manual) says purple, so S2 wins.
- R1 claims "there's no display". S1 and S3 both document a 4-character 7-segment display, so R1 is wrong on that point.
- R3 also mentions user-selectable pad colour modes. No Roland manual found for this research confirms them for the original TR-8, so treat them as unverified.

## 5. Typography and labelling

These points come from the S1 panel diagram. The typefaces are not named in any source, so this is observation only:

- **Section titles** (ACCENT, REVERB, DELAY, EXTERNAL IN, VOLUME, DRUM SELECT) are small upper-case sans-serif in light grey, centred above each boxed section.
- **Button legends** (TR-REC, PTN SELECT, INST PLAY, INST REC, KIT, INST, LAST STEP, SCALE, CLEAR, TAP, DEPTH, ON, STEP) are printed *on* the button caps in small upper case. Some are two-line (INST / PLAY, LAST / STEP).
- **Instrument headers** use a mixed-case, letter-spaced small-caps style ("BassDrum", "SnareDrum", "ClosedHihat"), printed on a slightly lighter header bar per strip. This nods to vintage drum-machine panel lettering.
- **Knob captions** (TUNE, ATTACK, COMP, SNAPPY, DECAY) sit just *below* each knob in tiny upper case.
- **Pad captions** sit *below* each pad in two lines: BASS DRUM ... RIDE CYMBAL, then a bracketed "ROLL" span over 8th / 16th / VARI 1 / VARI 2, then MUTE. A small boxed "INST" tag to the left labels the row's instrument function.
- **Model branding** is a stylised outline wordmark top-left. *Do not reproduce it or the maker logo.* A Pythonic panel should use its own name and wordmark.

Taken together, this is a dense but readable hardware idiom: every control has a printed name, abbreviations are reused consistently (BD/SD/LT...), and colour carries state rather than decoration.

## 6. Step-key interaction model (hardware)

### What the 16 pads do in each mode (S1 item 14)

| Mode button lit | Pads [1]-[16] do |
|-----------------|------------------|
| `[TR-REC]` | Toggle whether the selected instrument plays on each step |
| `[PTN SELECT]` | Select pattern 1-16. Press two pads together to select a range that plays in sequence (pattern chain). |
| `[INST PLAY]` | [1]-[11] play instruments live. Hold [12]-[15] (ROLL: 8th, 16th, VARI 1, VARI 2) + an instrument pad to roll; hold INST PLAY to latch a roll. [16] MUTE + instrument pad or select button mutes that instrument. |
| `[INST REC]` | Real-time record from pads [1]-[11]. Hold ACCENT [STEP] + pad for accented hits (S2). |
| DRUM SELECT `[KIT]` | Select kit 1-16 (lit pads = available) |
| DRUM SELECT `[INST]` | Select a tone for the instrument chosen with INST select; pad colour = tone family (S2) |
| an effect `[STEP]` (ACCENT / REVERB / DELAY / EXT IN) | Toggle that effect on each step. Pads now show the effect's step lane instead of an instrument's. |

### Step entry (TR-REC, S1)

1. Press `[TR-REC]`.
2. Choose variation `[A]` or `[B]`. If A and B are both playing, hold one and press the other to pick which one you edit.
3. Optionally set `[SCALE]` (3, 6, 4 or 8 steps per beat).
4. Press an **INST select button** to choose the instrument lane.
5. Press pads to toggle steps. Playback can continue the whole time ("Rec/Play modes have been eliminated", S3).
6. Repeat from step 4 for other instruments.

### Pattern structure

- 16 patterns, each with variations A and B. Patterns combine to "up to 32 steps" through A+B (S1, S3).
- **Pattern length:** hold `[LAST STEP]` + a pad sets the last step, 1-16 (S1). Not available when several patterns are selected.
- **Chaining:** press two pads in PTN SELECT. Both A and B lit = AB alternation (S1).

### Accents and dynamics (S1, S2)

- **Accent** is a separate *global* step lane, not per instrument. Press ACCENT `[STEP]`, then the pads. One ACCENT `[LEVEL]` knob sets the amount.
- **Weak accent** (firmware 1.10+): hold ACCENT `[STEP]` + pad. A step holds either an accent or a weak accent, not both, and their levels move together (S2).
- **Weak beat per instrument** (firmware 1.10+): hold the INST select button + pad. The pad then lights dimly (S2). A system setting (PROGRAMMING mode = PAD) instead cycles Strong, Weak, Off on each pad press (S1).

### Sub-steps and flams

- The original TR-8 has **no sub-step division**. Its closest equivalents are:
  - the **flam** (firmware 1.10+): hold `[TR-REC]` + pad, pad turns purple; flam spacing is hold `[TR-REC]` + TEMPO, range 0-8 (S2)
  - the live **ROLL** pads (S1)
- The TR-8S successor added true sub-steps (S5): `[SUB]` + pad, with `[SUB]` + VALUE choosing 1/2, 1/3 or 1/4 divisions. `[SHIFT]` + `[SUB]` toggles the SUB button between sub-step and flam. `[SHIFT]` + pad enters a weak beat. Variations grew to A-H plus fill-ins. This is the most relevant prior art for Pythonic's substeps.

### Erasing (S1)

- Hold `[CLEAR]` during playback to wipe the selected instrument over the region that plays while it is held.
- INST select + `[CLEAR]` erases that instrument's whole lane.

### Firmware additions after the manual (S4)

- 1.50: USB sample-rate options.
- 1.60: STEP LOOP.
- 1.61: DJ-software sync.

None of these changes the panel layout.

## Implications for an 8-channel Pythonic panel

Each item below is a mapping question raised by this reference. None is decided here.

1. **Strip count and width.** The TR-8 has 11 strips with uneven widths: BD and SD get double width and more knobs. Pythonic has 8 uniform channels, each with about 25 parameters. Should all 8 strips be equal, or should a channel be able to "promote" to a wider strip? Which 2-4 parameters earn an always-visible knob per strip (the TR-8's TUNE/DECAY idea), and where do the other ~20 live: a focus/detail panel, a pop-over, or pages?
2. **Grouping dividers.** The TR-8 groups strips by drum family (toms, hats, cymbals). Pythonic channels are generic patches. Do we group 8 channels as 4+4, as 2+2+2+2, or not at all?
3. **Strip header naming.** The TR-8 prints fixed instrument names. Pythonic channel names come from patches. Should the header show the patch name, a fixed channel number, or both, and with what abbreviation scheme for the select-button row?
4. **Fader semantics.** On the TR-8 the fader is LEVEL. Should Pythonic's fader be level, and where do pan, choke and velocity sensitivity go? The TR-8 hides pan behind INST + TEMPO.
5. **Effects row vs. per-channel FX.** TR-8 reverb and delay are global sends with per-step and per-instrument on/off. Pythonic has *per-channel* reverb, delay and vintage. Does the top "FX row" become per-channel (following the selected channel), or a global send bus? Does the TR-8's per-step FX lane idea carry over?
6. **EXTERNAL IN / SIDE CHAIN vs. pump.** The TR-8's side-chain block ducks external audio from the pattern. Is that the natural visual slot for Pythonic's pump modulator? Where do the LFOs go, given the TR-8 has no equivalent?
7. **Scatter slot.** The TR-8's big right-hand performance knob is SCATTER. Pythonic has no scatter. Should that prominent slot go to program morph, a fill trigger, or something else?
8. **Step pad colours.** Should Pythonic keep the 4 x 4 red/orange/yellow/white beat grouping as its idle look? What colour vocabulary should carry Pythonic-specific step state: probability, per-step velocity (TR-8 has only strong/weak/accent), substeps, fills?
9. **Accent model.** The TR-8 uses one global accent lane plus per-instrument weak beats. Pythonic has per-step velocity per channel. Do we show velocity as brightness (TR-8 dim = weak), add a global accent lane, or both?
10. **Sub-step entry.** The TR-8 has none, and the TR-8S uses a SUB-modifier + pad with a 1/2, 1/3, 1/4 choice. Should Pythonic copy that modifier pattern, or use a per-step popover, given substeps already exist in its sequencer?
11. **Pattern structure mapping.** TR-8 has A/B per pattern, LAST STEP and pad-range chaining. Pythonic has 16+ steps, fills, pattern chains and programs. Do A/B map to Pythonic's fills or to pattern pages? Is "LAST STEP + pad" the right length gesture for patterns longer than 16 steps (paging needed)?
12. **Mode buttons.** The TR-8 multiplexes one pad row across about 7 modes (TR-REC, PTN SELECT, INST PLAY, INST REC, KIT, INST, effect STEP lanes). How much of that modal multiplexing suits a mouse-driven GUI, where a second row or a grid could show the same information without modes?
13. **Kit vs. program.** TR-8 "kits" (16 x 11 tones) map loosely to Pythonic programs/patches, and DRUM SELECT INST to per-channel patch browsing. Is the pad-colour-by-family idea (808 pink, 909 yellow) worth reusing for patch categories?
14. **Display.** The TR-8 has only a 4-character 7-segment readout. Does Pythonic keep a retro segment display for tempo and values (an inspired touch), or use a richer text area?
15. **Trade-dress distance.** Black brushed panel + neon green trim + 808-coloured pad groups is a strongly recognisable combination. How far should Pythonic depart from it so the panel reads as inspired, not cloned? Options include a different accent hue, a different pad-group palette, and its own wordmark and typeface.
