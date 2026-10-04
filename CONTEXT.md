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

**Program**:
One of sixteen stored sets of the eight channel sounds inside a preset, switched to change all sounds at once.
_Avoid_: Kit, bank, slot
