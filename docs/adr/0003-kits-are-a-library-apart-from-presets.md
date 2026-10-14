# Kits are a library apart from presets, picked and saved as on a TR-8S

The web panel picks and saves kits as a TR-8S does. A kit (the eight channel sounds under a name) is a document of its own, kept in a library: read-only factory kits, one per drum machine, and user kits, one JSON file each in the `Kits` folder of the preset folder. A preset plays one kit: it saves the sounds with the kit's id, and a pattern can recall a kit when it starts. KIT and INST turn the display into a selector that the ◀ ▶ buttons, the wheel and the arrow keys step through, and each step plays at once. Edits mark the kit `*` and are kept for the session when you switch kits; WRITE stores them. The sixteen programs inside a preset leave the web face. They still load and save, for the tkinter interface, and WRITE turns them into user kits.

## Considered Options

- **Kits stay the preset's sixteen programs** and only the panel changes. This keeps one document, but a factory kit is then usable only after it is copied into a preset ("restore factory kits"), a kit cannot be shared between presets, and its name is only guessed from its patch names.
- **A browser list in the drawer** next to the display. It shows more at once, but it is a second place to pick from, and the hardware's display-and-VALUE habit is the point.

## Consequences

- `kit.edited` compares the channels with the stored kit, so undoing back to the stored sounds clears the `*`. The stored sounds are kept as a channel reports them, so the comparison is exact.
- Kit 000 is the preset kit: the sounds the preset was loaded with. A preset that names no kit, and every old file, plays it. Old files load and sound as before, and a saved preset gains `kit` (and a pattern `kit`) only when it plays or recalls one.
- When a pattern recalls a kit at its start, the switch happens on the audio thread, outside the undo journal. A browse through kits or drum patches is a single undo step.
