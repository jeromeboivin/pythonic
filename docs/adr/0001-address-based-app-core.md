# App core with an address-based interface, queue in and poll out

Two front-ends (tkinter and an HTML page in a Qt window) drive one UI-free app core, so the core's interface has to work in-process and across a JS bridge. We name every value, sound parameters and pattern step lanes alike, by an address (`ch3.osc.decay`, `pattern.B.ch2.step5.prob`). Values go through generic `get`/`set`/`describe` calls plus a short list of `act` verbs, so MIDI learn, morph, undo and the bridge all share one scheme instead of each keeping its own name table. Threads never share engine state directly: any caller's changes are queued and applied by the audio thread at block start (triggers at their sample offset), and front-ends pull changes, transport state and action results with `poll(since=v)` on their own frame timer, so the core never calls into a UI thread.

## Considered Options

- **One façade method per operation** (`set_osc_decay(ch, v)`, …): familiar, but the interface would be about as wide as the implementation, and MIDI learn, morph and the bridge would each need their own mapping from names to methods.
- **Typed command objects**: explicit, but around 200 command types for what are mostly "set this value".
- **A lock around engine state with pushed change events**: simpler to write, but the audio thread could block behind a long UI operation, and each front-end would have to re-marshal callbacks onto its own event loop.

## Consequences

- Undo is a journal of `(address, old, new)` entries grouped into gestures, with whole-preset snapshots only for bulk actions such as a preset load.
- Long actions (render, model load, stream restart) are asynchronous; their results and errors arrive through `poll`, never as return values.
