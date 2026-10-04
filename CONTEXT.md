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
_Avoid_: Kit, bank, slot

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
