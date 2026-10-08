# Pythonic

A synthetic drum machine: eight synthesized drum channels played by a step sequencer, with sounds and patterns saved together.

## Language

**Preset**:
The whole saved document: the sounds of all eight channels, the patterns, the programs, the morph and the tempo, swing, step rate and fill rate.
_Avoid_: Kit, project, song

**Channel**:
One of the eight drum voices, each with its own sound and its own lane in every pattern.
_Avoid_: Track, instrument, voice

**Drum patch**:
The complete sound of one channel, saved or loaded on its own.
_Avoid_: Drum, sound, instrument preset

**Drum type**:
The kind of drum a drum patch is (bass drum, snare, closed hat, ...), guessed from its name. A channel has no fixed drum type: it is whatever its drum patch is, and some drum patches have none.
_Avoid_: Instrument, role, slot

**Program**:
One of sixteen stored sets of the eight channel sounds inside a preset, switched to change all sounds at once.
_Avoid_: Bank, slot

**Kit**:
The eight channel sounds a program holds, named after the words its drum patch names share ("808"). The factory kits are the six drum machines' sounds, in programs 1 to 6 of every factory preset. The panel's kit mode picks programs on the pads.
_Avoid_: Drum kit, sound set

**Factory preset**:
A read-only preset shipped with Pythonic, one per drum machine (505, 707, 808, 909, DMX, LM2): that machine's sounds and patterns, with all six machines' sounds as programs 1 to 6. Saved only as another file.
_Avoid_: Demo, template, kit

**Factory drum patch**:
One channel sound of a factory kit ("909 SD"), loaded into any channel by name; the panel's inst mode offers those of the selected channel's drum type.
_Avoid_: Tone, sample, factory sound

**Pattern**:
One of the twelve step sequences of a preset, A to L: a lane of steps for each channel, 1 to 64 steps long, played in a loop.
_Avoid_: Sequence, bar, loop

**Selected pattern**:
The pattern being edited. It need not be the one sounding.
_Avoid_: Edited pattern, current pattern

**Playing pattern**:
The pattern sounding during playback.
_Avoid_: Current pattern, active pattern

**Chain**:
A run of neighbouring patterns linked in order, say A→B→C, that play one after another and then start again from the first.
_Avoid_: Song, playlist, sequence

**Queued pattern**:
A pattern picked during playback that takes over when the playing pattern ends, ahead of any chain.
_Avoid_: Next pattern, cued pattern

**Page**:
Sixteen steps of a pattern shown on the pads at once: 1-16, 17-32, 33-48 or 49-64.
_Avoid_: Bar, bank

**Matrix**:
A view of the triggers of all eight channels on one page of the selected pattern.
_Avoid_: Grid, overview

**Sound morph**:
A blend of the sounds of all eight channels between the two morph endpoints, set by the morph position.
_Avoid_: Crossfade, scene, snapshot

**Morph endpoint**:
One of the two stored sets of the eight channel sounds, A and B, that the sound morph blends between; saved in the preset. Unlike a program, it is not switched to but blended toward.
_Avoid_: Snapshot, scene, program

**Morph position**:
Where the sound morph sits between endpoint A (0 %) and endpoint B (100 %).
_Avoid_: Morph amount, crossfade

**Morph learn**:
A mode in which changes to the sounds are captured into one morph endpoint, A or B.
_Avoid_: MIDI learn, capture

**Lane**:
One channel's row of steps in a pattern.
_Avoid_: Track, row, step lane

**Step mode**:
Which step property the pads edit: trigger, accent, velocity, fill, probability or substeps.
_Avoid_: Lane mode, lane, layer

**Mute**:
A channel switched off from playback; its steps stay.
_Avoid_: Solo, bypass

**Step rate**:
The note length of one step: 1/8, 1/8T, 1/16, 1/16T or 1/32.
_Avoid_: Resolution, scale, speed

**Fill rate**:
How many hits a fill step plays within the step, 2 to 8.
_Avoid_: Roll rate, ratchet count

**Swing**:
How late the second sixteenth of every eighth note plays, giving a shuffle feel.
_Avoid_: Shuffle, groove

**MIDI learn**:
Mapping a hardware controller to a control by touching the control and then moving the controller.
_Avoid_: Morph learn, assign, map

**Pickup**:
A mapped hardware controller moves its control only once it has reached or crossed the control's current value; after any other change to the value it lets go again until the next crossing.
_Avoid_: Soft takeover, catch, scaling

**Edit all**:
A mode in which a change to any sound parameter of the selected channel is applied to every unmuted channel as well. It does not reach pattern steps.
_Avoid_: Link, gang, global edit

**Step**:
One position in a channel's lane of a pattern, 1 to 64 per pattern. A step holds its trigger, accent, velocity, fill, probability and substeps.
_Avoid_: Beat, tick, note

**Accent**:
A step flag that plays the hit at full strength, whatever the step's velocity.
_Avoid_: Strong beat

**Velocity**:
How hard an unaccented step hits, from 1 to 127.
_Avoid_: Level, volume, dynamics

**Fill**:
A step flag that repeats the hit at the fill rate until the next step, each repeat softer than the last.
_Avoid_: Roll, flam, ratchet

**Probability**:
The chance, from 0 to 100 %, that a triggered step plays on a given pass.
_Avoid_: Chance, likelihood

**Substeps**:
A step split into equal parts, each part either played or silent.
_Avoid_: Ratchet, sub-division, flam
