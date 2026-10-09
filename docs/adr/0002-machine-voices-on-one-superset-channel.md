# Machine voices as drum patches on one superset channel

To play TR-909, TR-808, TR-606, CR-78 and PCM-machine voices, the universal channel gains optional sections, such as a second tone oscillator, a click generator, a metal oscillator bank and a sample layer. It does not gain an engine choice per channel. With every section off, the channel is exactly today's voice, so every existing drum patch is a superset patch that leaves the sections off: `.mtdrum` and `.mtpreset` files keep loading and sounding bit-identical, and such a patch still saves as today's `.mtdrum` V3. Each machine voice ("909 SD") is a factory drum patch fitted to recordings of the real machine, not a dedicated circuit model.

## Considered Options

- **An engine per channel** (a "classic" engine, or a dedicated circuit model such as "909 BD" with a few knobs, as on the TR-8S): each voice could sound closer with fewer knobs. But every model would need its own DSP, parameter set, file form, edit-rack page, morph rules and fitting harness, and a channel's sound could no longer be blended or morphed across machines.
- **Both**, with dedicated models added later as further engine choices: kept out to hold one parameter space for morph, Edit all, MIDI learn and the patch files.

## Consequences

- Any section must have an "off" that leaves the render bit-identical to today's voice. The reference-loop tests pin this.
- Fidelity to a machine comes from fitting factory drum patches to recordings, so the sections must be expressive enough to fit, and the fitting harness must learn them.
- Morph, LFO targets, Edit all and the patch files all work over one, larger, parameter space.
