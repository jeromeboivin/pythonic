# TR-8S capability inventory against Pythonic

Research for [#24](https://github.com/jeromeboivin/pythonic/issues/24), part of the map [#23](https://github.com/jeromeboivin/pythonic/issues/23).

**Question.** What does the Roland TR-8S offer, area by area, and how does each capability compare with Pythonic today: present, partial or missing?

**Scope.** This covers the TR-8S at system program 3.00 (September 2023), the latest release. On the TR-8S side every claim cites a Roland-published source. On the Pythonic side every claim cites a file or a closed decision in this repo. Status words:

- **present**: Pythonic does it, perhaps in another form;
- **partial**: Pythonic does part of it, or something close;
- **missing**: Pythonic has nothing like it;
- **n/a**: the TR-8S does not have it either.

## Sources

| ID | Source | Used for |
|----|--------|----------|
| RM | [TR-8S Reference Manual, edition 05](https://static.roland.com/assets/media/pdf/TR-8S_Reference_eng05_W.pdf) (Roland; covers system program 3.00 and later). Its printed page numbers match the PDF page numbers. | Nearly every TR-8S claim, cited as "RM p.N" |
| OM | [TR-8S Owner's Manual, eng03](https://static.roland.com/assets/media/pdf/TR-8S_eng03_W.pdf) (Roland, 2018, written for the 1.0x system) | Pattern and kit structure, last step, the first sample import rules. Cited as "OM p.N" |
| UH | [TR-8S System Program (Ver.3.00) download page with the update history](https://www.roland.com/global/support/by_product/tr-8s/updates_drivers/84163b50-a2c5-4c3e-be2a-6ccee610d6ff/) | The firmware version that added each feature |
| SP | [TR-8S specifications](https://www.roland.com/global/products/tr-8s/specifications/) | Part counts, tone counts, sample memory, effect type lists |
| TL | [TR-8S Ver.3.00 Preset INST Tone List](https://static.roland.com/assets/media/pdf/TR-8S_PresetToneList_eng04_W.pdf) | Tone counts per type and category |
| — | [TR-8S owner's manuals index](https://www.roland.com/global/support/by_product/tr-8s/owners_manuals/) | The current document set: Reference Manual 3.00, Tone List 3.00 |

Pythonic side: `CONTEXT.md`, the module docstrings of `pythonic/app/*.py` (addresses and verbs), `pythonic/app/sound.py` (`SOUND_PARAMS`), `pythonic/sequencer.py`, `pythonic/pattern_manager.py`, `pythonic/drum_channel.py`, `pythonic/synthesizer.py`, `pythonic/voice.py`, `pythonic/lfo.py`, `pythonic/web/static/js/panel.js` and `steps.js`, ADR 0002, and the decisions of the closed GUI map [#1](https://github.com/jeromeboivin/pythonic/issues/1).

## Summary

The area tables below cite a source for every figure here.

- **Tracks:** the TR-8S has **11 instrument tracks** (BD SD LT MT HT RS HC CH OH CC RC), an **accent track** and a **trigger-out track**. It also has an EXT IN input with its own mixer channel. Any tone loads into any track. Pythonic has **8 channels**, also with free sound assignment.
- **Kits:** the TR-8S holds **128 kits**. A kit holds the 11 instruments and also the kit's reverb, delay, master FX, LFO, output routing, choke sources, CTRL assignment and fader colours. A pattern can recall a kit. A Pythonic program holds only the 8 channel sounds (16 programs per preset). Effects live inside each drum patch, and programs are not linked to patterns.
- **Instruments and CTRL:** the TR-8S has about 100 ACB tones (TR-808, 909, 707, 727, 606, CR-78), 72 FM tones and about 340 preset samples, plus user samples. Each tone has Tune, Decay, Level, Gain, Pan, two sends, an LFO destination and depth, and an insert effect. **CTRL is stored in the kit.** It can be set kit-wide (Pan, a send, LFO depth, InstFX) or per instrument as "User", and many of its targets are model-specific: BD Attack, SD Snappy, Tom Color, CR-78 Metallic, 15 sample parameters and 24 FM parameters. Pythonic has one universal voice with 59 parameters per channel. Its CTRL is a panel-wide preference of the web front-end, not part of the patch or the preset.
- **Sample import:** WAV or AIFF, up to 96 kHz, 8 to 32-bit float, mono or stereo. Up to about 180 s per file and 400 files, about 600 s in total. Start and end are trimmed in 10-sample steps, and a sample tone adds coarse tune, reverse and speed, attack and hold, a filter with its own envelope, and bit reduction. Pythonic has no sample playback or audio import at all.
- **Effects:** the TR-8S has one insert per instrument (**16 types** + THRU) and shared send effects per kit: **reverb (7 types)** and **delay (4 types)**. It has one **master FX slot (21 types)**, with COMPRESSOR as one of them; there is no dedicated master compressor. A fixed soft clipper sits on the output, and SCATTER sits before the master FX. Its **side-chain works on EXT IN only**. Pythonic's reverb and delay are inserts inside each channel and are cleared on that channel's next hit. Each channel also has a drive, one EQ band, a vintage processor and a self-triggered pump. Pythonic has no master effects, only a master volume and a soft clip.
- **Sequencer and performance:** the TR-8S stores 128 patterns. Each has variations A–H of up to 16 steps and 2 fill-ins, each holding motion data, plus per-pattern tempo, shuffle, scale, flam spacing, scatter and master probability. It has a last step per track, sub steps, flam, probability, weak beats, alternate sounds, step loop, rolls, INST PLAY/REC and 10 scatter types. Pythonic already has sub-steps (more general), step probability, per-step velocity (which covers weak beats and dynamics), a step roll ("fill"), chains, the queued pattern and randomizers. It lacks a last step per lane, flam, motion recording, fill-in variations, step loop, live recording, per-pattern tempo and swing, scatter, and the accent amount.
- **Pythonic extras the TR-8S lacks:** sound morph, two LFOs plus a pump per channel with 27 targets, patterns of up to 64 steps, rests inside sub-steps, undo, the AI drum generator and PO-32 transfer.

## Terms side by side

The two vocabularies collide in places. Downstream tickets should keep this mapping in view.

| TR-8S term | Nearest Pythonic term (`CONTEXT.md`) | Difference |
|------------|--------------------------------------|------------|
| track | channel (its sound) + lane (its steps) | A TR-8S track also carries motion data and its own last step (RM p.8, p.11) |
| instrument / INST tone | drum patch | A TR-8S tone is one of several models (ACB, FM, sample); a drum patch is one voice model |
| kit | kit = the sounds of one program | A TR-8S kit also holds the effects, routing, choke and CTRL (RM p.24–30) |
| pattern | the preset's 12 patterns, with its tempo and swing | A TR-8S pattern holds 8 variations, 2 fill-ins and its own tempo, shuffle and scale (RM p.17, OM p.8) |
| variation A–H | pattern A–L | Up to 16 steps vs up to 64 steps |
| fill-in | none | **Name clash.** Pythonic's *fill* is a step flag that rolls the hit; a TR-8S fill-in is a whole fill variation (RM p.13) |
| sub step | substeps | — |
| weak beat, dynamics | velocity | — |
| scale | step rate | — |
| shuffle | swing | — |
| motion | none | LFOs, the pump and the morph modulate sounds, but nothing is stored per step |

## 1. Tracks

| Capability | TR-8S detail | Pythonic today | Note |
|------------|--------------|----------------|------|
| Instrument track count | 11 fixed instrument tracks: BD, SD, LT, MT, HT, RS, HC, CH, OH, CC, RC (RM p.8–9; SP: 11 instrument parts) | **partial**: 8 channels | `CONTEXT.md` (Channel); `PythonicSynthesizer.NUM_CHANNELS = 8` in `synthesizer.py`. Decided in [#30](https://github.com/jeromeboivin/pythonic/issues/30) |
| Extra sequencer tracks | An accent track shared by all instruments, and a trigger-out track (RM p.8, p.19; SP: one part reserved for trigger out) | **partial**: accent is a flag in each lane; no trigger-out | `pattern_manager.py` (`PatternStep.accent`). A trigger output is hardware; Pythonic's nearest is MIDI file export (`export.py`) |
| How sounds are assigned | Tracks keep their names and panel slots, but any tone of any category loads into any track: SHIFT + VALUE switches category (RM p.32). The category can be locked per instrument, and the lock is saved in the kit (RM p.32) | **present**: any drum patch loads into any channel. A channel has no fixed drum type; the drum type is guessed from the patch name | `CONTEXT.md` (Drum type); `drum_patch.load` in `presets.py`; the panel's inst mode offers factory patches of the channel's drum type (`steps.js`) |
| What a track holds | Per variation: step data (trigger, sub step or flam, weak beat, alternate sound, dynamics, probability) and motion data. Per track: a last step shared by A–H (RM p.8, p.11–12, p.16, p.19–20) | **partial**: a lane holds per step trig, acc, vel, fill, prob and sub; one length per pattern | `patterns.py` docstring. No motion, no per-lane length, no flam or alternate sound |
| EXT IN as a mixer channel | A stereo or dual-mono external input with gain, pan, reverb and delay sends, an output assignment and the side-chain (RM p.29, p.47, p.56) | **missing** | The only audio input is the PO-32 recording (`po32.py`) |
| Instrument grouping (layering) | Master and slave instruments: hitting the master also plays the slaves, in TR-REC, INST PLAY and INST REC (RM p.24) | **missing** | — |
| MIDI note map | One note per track, one per alternate sound, and one for the trigger track, each set freely (0–127 or off). Separate MIDI channels for patterns and for kit program changes (RM p.46) | **partial**: 8 consecutive notes from a base note (default 36). Program change 0–11 selects patterns A–L | `midi.py` docstring and `midi.base_note`. Matters for the track count |

## 2. Kits

| Capability | TR-8S detail | Pythonic today | Note |
|------------|--------------|----------------|------|
| Number of kits | 128 user kits (OM p.8; SP). Preset kits since 2.00 (UH) | **partial**: 16 programs per preset; the 6 factory kits fill programs 1–6 | `programs.py`; `CONTEXT.md` (Program, Kit, Factory preset) |
| Instruments in a kit | The 11 instruments, each with its tone and parameters (RM p.9, p.32–37) | **present**: a program holds the 8 channel sounds, each a whole drum patch | `programs.py` |
| Kit-level effects | Each kit holds its reverb, delay, master FX, EXT IN and LFO settings (RM p.24–29) | **missing** as kit settings | In Pythonic every effect lives in each channel's drum patch (`drum_channel.py`) |
| Kit level | Level: -INF, -53.0 to +10.0 dB (RM p.24) | **partial**: `global.master` (-60 to +10 dB) is one value per preset, not per program | `core.py` |
| Output routing | Per instrument and EXT IN: MIX, ASSIGN 1–6 (mono) or ASSIGN A–C (stereo) (RM p.29) | **missing**: a per-channel A/B output flag from the `.mtdrum` format is kept, but both buses sum into one stereo output | `synthesizer.py` (`bus_a` + `bus_b`); `sound.py` `mix.output` |
| Choke ("KIT: MUTE") | Per instrument, the one instrument whose hit mutes it. The manual names the OpenHH sound and sample tones (RM p.29) | **partial**: a choke flag per channel; every flagged channel is in one shared group | `sound.py` `mix.choke`; `synthesizer.py` `_apply_choke` |
| CTRL assignment | Stored in the kit: CTRL Sel, plus one User parameter per instrument (RM p.29–30) | **partial**: a panel-wide preference of the web front-end, not saved in the preset | `panel.js` (`pref.ui.ctrl_knob`); [#12](https://github.com/jeromeboivin/pythonic/issues/12) |
| Name and colours | A 16-character name; a fader colour per instrument (12 colours) (RM p.30) | **partial**: the kit name is derived from the words the channel names share; strip colours are fixed by position | `programs.py` (`program.names`); [#16](https://github.com/jeromeboivin/pythonic/issues/16) |
| Pattern recalls a kit | PTN SETTING: KIT Sw and Number, active when UTILITY KitSel = PTN (RM p.17, p.45). Kit program change on the Kit MIDI channel (RM p.46) | **missing**: programs are separate from patterns ("Patterns are not part of a program") | `programs.py` |
| Kit chains | No kit chain in the manuals. The nearest feature is the pattern-to-kit link above | **n/a** | — |
| Kit copy, initialize, export, import | Copy kit or instrument (RM p.18); initialize, exchange, export and import kits to an SD card (RM p.38–40, p.47) | **partial**: preset save and load, drum patch save and load, the preset clipboard; no file per program | `presets.py` |

## 3. Instruments and per-instrument CTRL

| Capability | TR-8S detail | Pythonic today | Note |
|------------|--------------|----------------|------|
| Instrument types | ACB (circuit-modelled) tones of the TR-808, 909, 707, 727, 606 and CR-78: about 100 in the 3.00 list (TL), 81 before 3.00 (SP). The CR-78 tones and "808 Chromatic Bs" came in 3.00 (UH). 72 FM tones, 6 of them editable FM MODEL INST since 2.50 (SP; UH 2.00, 2.50). About 340 preset samples (TL; SP: 300 or more), user samples, and loop tones (RM p.32) | **partial**: one universal synthesized voice. An oscillator (sine, triangle, saw) with decaying, sine or random pitch modulation, plus filtered noise, a drive, a shaper and an EQ band | `voice.py` docstring. Per ADR 0002, dedicated circuit models stay out; machine voices become factory drum patches on optional sections. The sine pitch-modulation mode gives FM-like tones |
| Tune | -128 to +127 per instrument; on the TUNE knob (RM p.32) | **present**: `osc.pitch` (±24 st) and `osc.freq` (20 Hz–20 kHz) | `sound.py`. The strip's tune knob ([#7](https://github.com/jeromeboivin/pythonic/issues/7), [#11](https://github.com/jeromeboivin/pythonic/issues/11)) |
| Decay | 0–255; on the DECAY knob (RM p.32) | **present**: `osc.decay` and `noise.decay` (10 ms–10 s) | `sound.py` |
| Level and gain | Level 0–255 on the fader; Gain -40 to +40 dB after INST FX (RM p.32) | **present**: `mix.level` (-60 to +10 dB) | `sound.py` |
| Pan | L127–CENTER–R127 (RM p.32) | **present**: `mix.pan` (-100 to +100) | `sound.py` |
| Reverb and delay sends | ReverbSend and DelaySend 0–255 per instrument (RM p.32) | **partial**: `fx.reverb_mix` and `fx.delay_mix` are the wet mix of each channel's own insert effects | See Effects |
| LFO | One kit LFO (SIN, TRI, SAW, SQR, S&H; tempo sync or free rate) (RM p.29). Per instrument, a destination (any INST parameter, CTRL parameter or InstFX) and a depth of -128 to +127 (RM p.33) | **present**, and broader: 2 LFOs (6 waveforms, sync, retrigger, polarity, phase) plus a pump per channel, over 27 targets | `lfo.py` (`ModTarget`); `sound.py` |
| CTRL knob | KIT: CTRL Sel = OFF, Pan, ReverbSend, DelaySend, LFO Depth, InstFX or User. User assigns one parameter per instrument (RM p.14, p.29–30). Holding an instrument button and turning the MASTER FX CTRL knob edits that instrument's INST FX (RM p.37) | **partial**: one panel-wide mode (off, pan, reverb mix, delay mix, LFO 1 depth, user); user picks one of 27 sound parameters per strip | `panel.js` (`CTRL_MODES`, `CTRL_USER_CHOICES`). Stored as a front-end preference ([#12](https://github.com/jeromeboivin/pythonic/issues/12)). Decided in [#34](https://github.com/jeromeboivin/pythonic/issues/34) |
| Model-specific parameters | See the CTRL table below (RM p.30, p.33–34) | **partial**: one parameter set for every patch (59 per channel), with no per-model "character" parameter | `sound.py` `SOUND_PARAMS`. New sections ([#29](https://github.com/jeromeboivin/pythonic/issues/29)) would bring such parameters |
| Accent behaviour | Accent steps live on the accent track. ACCENT LEVEL sets how loud accented steps are (RM p.4, p.19) | **partial**: an accented step plays at velocity 127. What velocity changes is up to each patch (`vel.osc`, `vel.noise`, `vel.mod`, 0–200 %) | `pattern_manager.py` (`hit_velocity`); `sound.py`. No accent amount |
| Velocity-sensitive playing | The inst pad is velocity-sensitive; its sensitivity is LIGHT, MEDIUM, HEAVY or FIX (RM p.5, p.21, p.45). Sample tones have Flt Velo (RM p.33) | **partial**: MIDI notes carry their velocity. The panel hits the selected channel at 64, or 127 with Ctrl | `midi.py`; `panel.js` |
| Random instrument | Temporarily swap one or all tones for random tones of the same category; reverts unless saved (RM p.12; UH 2.50) | **present**, in another form: `preset.randomize_all`, and AI drum patch candidates per drum type tried on the live channel | `presets.py`; `ai.py` |

### What CTRL drives, per instrument model

The User CTRL list per instrument (RM p.30) and the LFO destination list (RM p.33) offer these model-specific parameters on top of OFF, Pan, ReverbSend, DelaySend, LFO Depth and InstFX:

| Model | CTRL parameters | Range and meaning |
|-------|-----------------|-------------------|
| ACB bass drum | Attack | 0–255, how hard the kick's attack hits (RM p.33) |
| ACB snare | Snappy | 0–255, volume of the snare wires (RM p.33) |
| ACB tom | Color | -128 to +127. For 808, 909 and 606 toms: ambience (noise) amount. For 808 noise toms: resonance. For 707 toms: amount of pitch change (RM p.33) |
| ACB bass (808 Chromatic Bs) | ACB Coarse | -24 to +24 semitones (RM p.33; UH 3.00) |
| CR-78 hi-hat and cymbal | Metallic | 0–255, level of the metallic sound (RM p.33) |
| Sample tones | Coarse, Rate, Spread, BitReduce, Attack, HoldMode, HoldTime, HoldStep, FltType, FltCutoff, FltReso, FltEnvAtk, FltEnvDecay, FltEnvDepth, FltVelo | See Sample import (RM p.33) |
| FM tones | Morph (all FM tones); FM Coarse (FX/HIT and OTHERS FM tones) | Morph changes the FM setting (RM p.33) |
| FM MODEL INST (Kick, Snare, Tom, Clap, Cymbal, Perc) | FM Depth, FM Ratio, FM Freq, FM Decay, FM Fbk, Feedback, Color, HPF Cutoff, LPF Cutoff, LPF Reso, Pit Env, Pit Attack, Pit Decay, Body, Click, Noise, Claps, Clap Intrvl, Nuance, Note, Hrm Level, Hrm Ratio | Each model uses its own subset (RM p.33–34; UH 2.50) |

The manual describes each CTRL target as one parameter over its own range. It mentions no macro, curve or range setting for CTRL (RM p.14, p.29–30).

## 4. Sample import

| Capability | TR-8S detail | Pythonic today | Note |
|------------|--------------|----------------|------|
| File formats | WAV and AIFF. WAV up to 96 kHz; AIFF at 44.1, 48 or 96 kHz. 8, 16, 24, 32-bit and 32-bit float. Mono or stereo (RM p.41) | **missing** | Pythonic only writes WAV (`export.py`). The PO-32 import decodes synth patches from a modem recording, not audio samples (`po32.py`) |
| Length and count | About 180 s per file (44.1 kHz mono) and at most 400 files, less as memory fills (RM p.41). About 600 s in total at 44.1 kHz mono (SP) | **missing** | Inputs for [#31](https://github.com/jeromeboivin/pythonic/issues/31) |
| Internal sample rate | Not stated in the manuals. The converters are 24-bit/96 kHz (SP) | **n/a** | Pythonic's synth rate is a preference (`pref.audio.synth_rate`, `prefs.py`) |
| Storage | An internal user sample area, filled from the SD card folder `ROLAND\TR-8S\SAMPLE\`, by file or by folder (up to 256 folders of 256 files) (RM p.41). Delete, defragment ("Optimize") and 32 named user categories (RM p.43–45). Batch import since 1.02 (UH) | **missing** | — |
| Preview | The selected file plays while you browse (RM p.41; UH 1.10) | **missing** | — |
| Sample to instrument | The SAMPLE button assigns a user sample to the selected track as its tone (RM p.42) | **missing** | — |
| Per-sample settings | Start and End (in steps of 10 samples), Gain -18 to +18 dB, Category, Name. Shared by every kit that uses the sample (RM p.42) | **missing** | — |
| Per-instrument sample parameters | Tune, Decay, plus Coarse ±24 st; Rate -1.00 to +1.00 (slower, stopped or reversed playback); Spread ±50; Bit Reduce 0–12; Attack; Hold Mode (Whole, Time, Step) with Hold Time or Hold Step (0.5–128 steps); filter LPF or HPF with Cutoff, Reso, envelope Attack, Decay, Depth and Velo (RM p.33) | **missing** | Pythonic's filter and envelopes only shape its own oscillator and noise (`voice.py`) |
| Loop tones | Tones that play over and over (RM p.32) | **missing** | — |
| Stopping and choking samples | A sample may ring past the pattern's stop; SHIFT + START/STOP silences it (RM p.4, p.11). KIT: MUTE chokes sample tones (RM p.29) | **partial**: channel choke exists for synth voices | `synthesizer.py` |

## 5. Effects

Signal flow, TR-8S (RM p.56):

```
per instrument: TONE -> INST FX -> Level -> Gain -> Pan -+-> MIX bus or ASSIGN OUT 1-6 / A-C
                                                         +-> Reverb send, Delay send
EXT IN -> side chain -> Gain -> Pan -> MIX bus (+ sends)
DELAY -> MIX bus, and DELAY -> Reverb send;  REVERB -> MIX bus
MIX bus -> SCATTER -> MASTER FX -> KIT Level -> Soft Clipper -> MIX OUT
```

Signal flow, Pythonic (`voice.py`, `drum_channel.py`, `synthesizer.py`):

```
per channel: osc + noise -> drive -> soft shaper -> peaking EQ -> level / pan
             -> vintage -> delay (insert) -> reverb (insert)        [delay and reverb are cleared on each hit]
sum of channels -> master volume (+ LFO offset) -> soft clip above 0.9 -> stereo out
```

| Capability | TR-8S detail | Pythonic today | Note |
|------------|--------------|----------------|------|
| Per-instrument insert effect (INST FX) | One slot per instrument with 16 types plus THRU: HPF, LPF, LPF/HPF (resonant, -12/-18/-24 dB), H BOOST, L BOOST, L/H BOOST, ISOLATOR, TRANSIENT, COMPRESSOR, DRIVE, COMP+DRV, CRUSHER, SATURATOR, FREQ SHIFT, RING MOD, SPREAD (RM p.34–37). The last four came in 2.00 (UH). One parameter of each type can go on CTRL (RM p.34–37) | **partial**: each channel has a drive and soft shaper, one peaking EQ band (`eq.freq`, `eq.gain`) and the vintage processor | `voice.py`; `sound.py`. No resonant filter on the whole sound (the noise filter only shapes the noise), and no compressor, transient, crusher, ring mod, frequency shift or spread |
| Reverb | One shared send reverb per kit. Types AMBI, ROOM, HALL1, HALL2, PLATE, MOD, HA-DOU (HA-DOU since 2.50). Time, Level, Pre Delay 0–100 ms, Low Cut, High Cut, Density (RM p.24–25; UH) | **partial**: a reverb inside each channel (decay, mix, width), cleared on that channel's next hit | `drum_channel.py` (`trigger` calls `reverb.reset()`); `reverb.py`. No shared bus, types, pre-delay or cut filters |
| Delay | One shared send delay per kit. Types DLY, PAN, TAPE ECHO, PITCH SHFT. TempoSync with 16 values from 1/32 to 1/1 (triplets and dotted), or a free time. Level, Feedback, Reverb Send (since 2.00), High Cut. Damping and tap time (DLY, PAN); three heads, bass, treble, head pans, tape distortion and wow and flutter (TAPE ECHO); coarse and fine pitch (PITCH SHFT, since 2.50) (RM p.25; UH) | **partial**: a delay inside each channel. 14 synced times, feedback up to 0.95, mix, ping-pong. Cleared on that channel's next hit | `delay.py` (`DelayTime`); `drum_channel.py`. Ping-pong is close to the PAN type |
| Sends and returns | ReverbSend and DelaySend per instrument and for EXT IN. The delay feeds the reverb. Returns go to the MIX bus before scatter and master FX (RM p.29, p.32, p.56) | **partial**: the per-channel wet mix only | — |
| Master FX | One slot per kit with 21 types: HPF, LPF, LPF/HPF, H BOOST, L BOOST, L/H BOOST, ISOLATOR, TRANSIENT, TRANSIENT2, COMPRESSOR, DRIVE, OVERDRIVE, DISTORTION, FUZZ, CRUSHER, PHASER, FLANGER, SBF, NOISE, FATTENER, VINYL SIM (RM p.26–29). FATTENER and VINYL SIM came in 2.50 (UH). On/off switch; the MASTER FX CTRL knob drives one chosen parameter; both are motion-recordable (RM p.14, p.16, p.26) | **missing** | `synthesizer.py`: master volume and soft clip only |
| Master compressor | No dedicated compressor. COMPRESSOR is a master FX type: Thre, Ratio 1:1 to 1:INF, Knee, Attack, Release, Gain, Balance (RM p.26–27). It is also an INST FX type (RM p.35). A fixed soft clipper sits before MIX OUT (RM p.56) | **partial**: a soft clip above 0.9 of full scale; no compressor | `synthesizer.py` |
| Side-chain | SIDE CHAIN works on EXT IN only (SP lists the side chain under EXT IN only; RM p.56). The trigger source is any instrument track or the trigger-out track. 8 types (ducking or gate; widths from narrow or half a step to 2 steps); depth 0–255 (RM p.29) | **partial**: each channel's pump is a ducking envelope fired by that channel's own hits. It modulates one target (level by default) and can sync to the tempo | `lfo.py` (`PumpSource`); `drum_channel.py`. No cross-channel trigger and no external input |
| Scatter | A mix-bus effect before the master FX (RM p.56); see Sequencer | **missing** | — |
| Outputs | MIX OUT, 6 assignable outputs (mono 1–6 or stereo pairs A–C, or trigger outs), and USB multichannel audio (RM p.29, p.45, p.47, p.56) | **missing**: one stereo output | `synthesizer.py` |
| Motion on effects | Reverb level, delay level, time and feedback, and master FX on and CTRL can be recorded per step (RM p.9, p.16) | **missing** | See Sequencer |

## 6. Sequencer and performance

| Capability | TR-8S detail | Pythonic today | Note |
|------------|--------------|----------------|------|
| Pattern memory | 128 patterns in 8 banks of 16 (OM p.8) | **partial**: 12 patterns, A–L, per preset | `patterns.py` |
| Variations A–H | 8 variations per pattern, up to 16 steps each, with a last step per variation (RM p.11; OM p.8–9; SP). Lighting several variations plays each once, in order A→H. Recording can target a variation that is not playing (RM p.11, p.15) | **partial**: 12 patterns of 1–64 steps. A chain of neighbouring patterns (A→B→C) plays like lit variations | `CONTEXT.md` (Pattern, Chain). A variation shares its pattern's tempo, kit and settings; a Pythonic pattern shares the preset's |
| Fill-in | 2 fill-in variations per pattern, each with its own last step. A variation A–H or SCATTER can also serve as the fill. Fills trigger by hand (momentary, or latched to the next bar) or automatically every 32, 16, 12, 8, 4 or 2 measures (RM p.11, p.13, p.45) | **missing** | Pythonic's *fill* is a different thing (next rows) |
| Last step per track | Each track can have its own last step, shared by A–H and overriding the variation's last step, which allows polymeter (RM p.11–12; OM p.9) | **missing**: one length per pattern for all lanes | `patterns.py` (`pattern.<P>.length`). The sequencer already keeps ticks counting across loops (`sequencer.py` docstring) |
| Sub steps | 1/2, 1/3 or 1/4 per step (RM p.19) | **present**, and broader: any split into played and silent parts (`o`, `-`) | `patterns.py` (`sub`); `sequencer.py` |
| Flam | Per step (SHIFT + SUB toggles sub step or flam), with Flam Spacing 0–8 per pattern; also playable live (RM p.17, p.19, p.21–22) | **missing** | — |
| Probability | Per step PROB and SUB PROB (sub-step probability), plus Master Probability, -100 to +100 % per pattern, added to every set probability (RM p.17, p.20). All three came in 2.50 (UH) | **partial**: per-step probability 0–100 % | `patterns.py` (`prob`). No sub-step or master probability |
| Accent | An accent track shared by every instrument, with ACCENT LEVEL as the amount (RM p.4, p.8, p.19). Master Probability is also set by TR-REC + the ACCENT knob (RM p.17) | **partial**: an accent flag per lane step that plays at 127 | `pattern_manager.py` |
| Weak beats and per-step dynamics | Strong, weak or off per step (entered with SHIFT + pad, or by cycling the pad). Per-step dynamics are set by holding a pad and turning ACCENT LEVEL (RM p.19, p.45) | **present**: per-step velocity 1–127, default 64 | `patterns.py` (`vel`). A weak beat is a lower velocity |
| Roll | In INST PLAY, rolls at 1/16, 1/32 or 1/64 while a roll pad is held, or latched (RM p.6, p.22) | **partial**: the step flag *fill* repeats a hit at the fill rate (2–8 per step) with falling velocity. It is sequenced, not played live | `sequencer.py`; `CONTEXT.md` (Fill, Fill rate) |
| Alternate sounds (ALT INST) | Tones whose names contain "/" (707Bass1/2) have a second sound, chosen per step or per hit (RM p.19, p.21–22) | **missing** | — |
| Motion recording | Records TUNE, DECAY and CTRL of each instrument, plus reverb level, delay level, time and feedback, and master FX on and CTRL. Per step, per variation. Recorded live, or entered by holding a pad and turning a knob; cleared per variation, track, knob or step (RM p.9, p.16) | **missing** | LFOs, the pump and the morph modulate sounds, but nothing is stored per step |
| Step loop | Holding a pad loops that step for all instruments; it can latch, and it can divide the step into 1/2 or 1/4 rolls (RM p.23). Added in 1.10, rolls in 2.50 (UH) | **missing** | — |
| INST PLAY | Pads 1–11 play the instruments live over the pattern: velocity pad, sub steps, flam, weak beats, alternate sounds, rolls (RM p.21–22) | **partial**: MIDI notes with velocity, and a click on the selected channel's button | `midi.py`; `panel.js`. No pad-row play mode |
| INST REC | Real-time recording of pad hits into the selected variation (RM p.21) | **missing** | — |
| TR-REC step entry | Steps per track on 16 pads, accent through ACCENT STEP; the scale is set per pattern (RM p.19) | **present**: step modes (trig, accent, velocity, fill, prob, substeps), 4 pages of 16 | `CONTEXT.md` (Step mode, Page); [#8](https://github.com/jeromeboivin/pythonic/issues/8) |
| Pattern chain and queue | Pressing two pattern pads chains that range; the next pattern switches at the end of the current one (RM p.11) | **present**: chains and the queued pattern | `CONTEXT.md` (Chain, Queued pattern); `patterns.py` |
| Tempo | Per pattern, 40.0–300.0 BPM; UTILITY TempoSrc picks pattern or system tempo (RM p.17, p.45). 0.1 BPM steps, tap tempo, mark and recall a tempo (RM p.14), and nudge (RM p.15) | **partial**: one integer tempo per preset, 1–300 BPM; MIDI clock sync | `core.py` (`global.tempo`); `midi.py`. No per-pattern tempo, tap, nudge or 0.1 steps |
| Shuffle | Per pattern, -128 to +127; UTILITY Shuffle picks pattern or system (the SHUFFLE knob) (RM p.5, p.17, p.45) | **partial**: one swing per preset, 0–100 %, which only delays the second sixteenth of each eighth pair | `core.py` (`global.swing`); `sequencer.py` |
| Scale (step rate) | Per pattern: 8th triplet, 16th triplet, 16th, 32nd (RM p.17) | **partial**: step rate 1/8, 1/8T, 1/16, 1/16T, 1/32, once per preset | `core.py` (`global.step_rate`) |
| Scatter | Swaps steps, and changes playback direction or gate length,, from the second loop on. 10 types, depth 1–10, set per pattern. Usable as a fill-in. It sits on the mix bus (RM p.11, p.13, p.17, p.56) | **missing** | `pattern.shift_*`, `reverse` and `alter` edit steps, but they are not a live effect (`patterns.py`) |
| Mute and solo | Mute per track and clear all mutes; solo one instrument (RM p.13) | **partial**: `ch<N>.mute` and the MUTE latch. Solo is only a synth method, with no address and no control | `core.py`; `synthesizer.py` (`solo_channel`) |
| Random pattern | Generate a provisional random pattern and keep it with TR-REC (RM p.12) | **present**: `pattern.randomize`, `alter`, `randomize_accents_fills`, and AI patterns | `patterns.py`; `ai.py` |
| Copy and clear | Copy a pattern, variation or track; clear a variation or track; erase a track while held during playback (RM p.12–13, p.18–19) | **present**: pattern cut, copy, paste, exchange and clear; lane copy and paste | `patterns.py` |
| Revert edits | Reload a saved pattern, variation, track, kit or controller (RM p.15, p.18) | **partial**: undo and redo, 50 steps | `undo.py` |

## Gaps that matter most downstream

**Track count ([#30](https://github.com/jeromeboivin/pythonic/issues/30)).**

- The TR-8S plays 11 instrument tracks, plus a shared accent track, a trigger-out track and an EXT IN mixer input (RM p.8, p.29). Pythonic has 8 channels.
- On the TR-8S, any tone goes in any track (RM p.32), as any drum patch goes in any channel in Pythonic. The TR-8S track names are labels, not constraints.
- Its MIDI map gives each track a free note, plus a note per alternate sound (RM p.46). Pythonic uses 8 consecutive notes.
- Everything sized to 8 meets this decision: lanes, programs, the morph endpoints, PO-32 transfer, the panel's 8 strips, and the `.mtpreset` 8-channel format.

**Per-instrument CTRL ([#34](https://github.com/jeromeboivin/pythonic/issues/34)).**

- The TR-8S stores CTRL in the kit: a kit-wide choice, or a "User" parameter per instrument (RM p.29–30).
- Each model brings its own targets: BD Attack, SD Snappy, Tom Color (whose meaning changes by machine), CR-78 Metallic, ACB Coarse, 15 sample parameters and up to 24 FM parameters (RM p.30, p.33–34).
- Each CTRL target is a single parameter over its own range. The manual describes no macro or curve.
- Pythonic's CTRL is a front-end preference, not part of a patch or preset (`panel.js`). Model-specific targets have nothing to point at until the new sections exist ([#29](https://github.com/jeromeboivin/pythonic/issues/29)).

**Effects to add ([#35](https://github.com/jeromeboivin/pythonic/issues/35)).**

- **Reverb and delay become shared send buses.** Pythonic's reverb and delay are per-channel inserts cleared on each hit of their channel (`drum_channel.py`). The TR-8S has one reverb and one delay per kit, fed by sends, with the delay feeding the reverb and tails that survive retriggers (RM p.56).
- **A per-channel insert slot.** The TR-8S INST FX has 16 types; Pythonic's nearest are its drive and one EQ band. The main gaps are the resonant filters, compressor, transient shaper, crusher and saturator (RM p.34–37).
- **A master FX slot.** 21 types, with the compressor as one of them; Pythonic has none (RM p.26–29).
- **Smaller items.** The TR-8S side-chain only ducks EXT IN (SP), so cross-channel ducking would go beyond the TR-8S; Pythonic's pump only ducks its own channel. Scatter is a mix-bus effect (RM p.56). Individual outputs are missing.

**Sequencer and performance features to add ([#36](https://github.com/jeromeboivin/pythonic/issues/36)).**

- **Missing outright:**
  - last step per lane (polymeter) (RM p.11–12);
  - flam with a spacing per pattern (RM p.17, p.19);
  - motion recording, that is per-step values of tune, decay, CTRL and the effect levels (RM p.16);
  - fill-in variations with manual or automatic fills (RM p.13);
  - step loop (RM p.23);
  - live play and real-time recording from the pads, with rolls (RM p.21–22);
  - scatter (RM p.11, p.17);
  - alternate sounds (RM p.19).
- **Per-pattern settings:** tempo, shuffle (including negative), scale, the kit link, and master probability (RM p.17). In Pythonic all of these are per preset or absent.
- **Accent:** the TR-8S has an accent amount and a shared accent track; Pythonic's accent is a fixed 127 per lane (RM p.19).
- **Already covered:** sub-steps (with rests), step probability, per-step velocity (weak beats and dynamics), chains, the queued pattern and randomizers.
- **Vocabulary:** the name *fill* clashes. Pythonic's step roll would need a new name before TR-8S fill-ins can arrive.
