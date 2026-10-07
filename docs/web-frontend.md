# Web front-end

The new interface: an HTML/JS panel in a PySide6 QtWebEngine window, on top
of the app core ([core interface](core-interface.md)). No JS build step: ES
modules served as they are. Decisions: map #1, bridge #2, layout #7/#12,
frame budget #10, look #16, testing #17/#18, launch #19.

## Running it

```bash
pip install -e ".[dev]"            # PySide6 is a dependency (not on Windows ARM64)
pythonic                           # web interface (default); same as python run.py
pythonic --ui tk                   # tkinter interface
pythonic --devtools [PORT]         # Chromium DevTools on http://127.0.0.1:9222 (or PORT)
pythonic --quit-after 5            # close after 5 s; non-zero exit if the page did not boot
```

Without QtWebEngine, `--ui web` starts tkinter with a dialog that explains
why and offers to install PySide6 (on Windows ARM64 it only explains).
`pythonic/web/static/index.html` also opens in a plain browser served over
HTTP (`python -m http.server -d pythonic/web/static`): a fake bridge then
stands in for the core (tempo and START/STOP work).

Fonts: `python tools/fetch_fonts.py` downloads the bundled OFL fonts into
`pythonic/web/static/fonts/` (see `SOURCES.txt` there); until they are
committed the panel falls back to system fonts.

## Layout

```
pythonic/web/
  __init__.py      STATIC_DIR
  scheme.py        app:// scheme: register_scheme() before QApplication, install_handler(profile)
  bridge.py        Bridge QObject (slots + frame signal), FrameStats, FRAME_HZ = 60, FRAME_BUDGET_MS = 4
  window.py        PanelWindow(core, owns_core=True): view + page + channel; MIN_SIZE 1280x800;
                   fit_stage_height(old, new) (the edit rack drawer); dispose()
  app.py           run(quit_after, devtools_port, core): the --ui web entry; prepare_headless()
  static/
    index.html     loads css/*, qrc:///qtwebchannel/qwebchannel.js, js/main.js
    css/tokens.css design tokens (palette, fonts, stage size) as custom properties
    css/fonts.css  @font-face of the bundled fonts
    css/panel.css  stage and panel styles
    fonts/         bundled fonts + licences (SOURCES.txt)
    js/bridge.js       transport: connectBridge(), webChannelBridge(remote), createFakeBridge(opts)
    js/core-client.js  createCoreClient(bridge, {schedule})
    js/store.js        createStore(), READOUTS
    js/stage.js        fitStage(w, h, width, height), mountStage(el), setStageHeight(el, h), STAGE_WIDTH/HEIGHT
    js/values.js       toPosition/fromPosition (the core's curves), drag/wheel math, formatValue, parseValue
    js/controls.js     px-knob, px-fader, px-toggle, px-switch, px-list, px-target, px-display; createControlContext, provideContext
    js/modulation.js   modulatedAddresses(readout, value): {address: {offset, source}}; destinationOf,
                       targetAddress, targetName, modulationBand (click to assign, bands)
    js/midi-cues.js    ccsFor, withoutAddress, isBendTarget, ghostPosition
    js/drum-type.js    guessDrumType(name): the channel button label
    js/panel.js        mountPanel(stage, {store, client, meta}) -> {ctx, display, slot(name), drawer, steps, patterns, act, destroy}
    js/steps-logic.js  pure step-row logic: addresses, padView, withStep, editChannels, dragValue, pages, follow, patternStates, chains
    js/steps.js        mountStepRow(...): step-mode buttons, page bars, pads, the matrix; selectedPattern(store)
    js/patterns.js     mountPatterns(...): pattern buttons A-L, chain, pattern menu, lane copy / paste
    js/drawer.js       createDrawer(slot): the edit rack drawer's pages (the rack as base, the matrix, ...)
    js/rack-layout.js  RACK_SECTIONS (the rack's sections and controls as data), FACE_ONLY, stageHeightFor
    js/rack.js         mountRack(...): the edit rack, click to assign, drum patch menu, open / closed
    js/main.js         boot({bridge, stage}), demoBridge(); sets window.pythonic
    test/              runner.html, shim/, *.test.js (pure), *.engine-spec.js (DOM)
```

`window.pythonic` = `{ bridge, client, store, meta, panel, ready }` once
booted (`{ ready: false, error }` if boot failed).

## Bridge (Python <-> JS)

Published on the QWebChannel as `bridge`. JS -> Python slots take and return
**JSON text**; malformed requests answer `{"error": message}`.

| Slot | Request | Reply |
|---|---|---|
| `get` | `["addr", ...]` (or one address) | `{"values": {addr: value}, "errors": {addr: msg}}` |
| `set` | `[{"address", "value", "burst"?, "edit_all"?, "record"?}, ...]` (or one) | `{"errors": {addr: msg}}` |
| `act` | `{"verb", "args"?}` | `{"id": n}`; the result is the poll event with that id |
| `describe` | `"prefix"` (registered addresses under it, `""` = all) or `["addr", ...]` | `{"addresses": {addr: describe()}, "errors": {...}}` |
| `gesture` | `begin` / `end` (plain text) | none: brackets a drag as one undo step |
| `resync` | `null` | none: the next frame carries every readout |
| `fileDialog` | `{"mode": "open"\|"save"\|"folder", "title", "filters": ["Presets (*.mtpreset)"], "folder", "name", "suffix"}` | `{"id": n}`; then the `dialog` signal |
| `resizeWindow` | `{"from": 1000, "to": 700}` (stage design heights) | `{"size": [w, h]}`: the window grows or shrinks by the difference at the panel's scale, its minimum follows (1280x560 closed); a maximized window keeps its size; `{"size": null}` without a window |

Signals (JSON text): `frame` = one `core.poll(since)` result per tick of a
60 Hz main-thread QTimer, sent when it has changes or events or a readout
changed (`version`, `changes`, `events`, `transport`, `modulation`, `audio`,
`midi`); `dialog` = `{"id", "path"}` (`null` when cancelled). Dialogs are
window-modal `QFileDialog`s opened with `open()` (never `exec()`): the frame
timer keeps running. Save dialogs do not confirm overwrites; the core's save
verbs refuse to replace a file and the page asks. Device lists come from
`describe()` labels (`pref.audio.device`, `midi.device`) and the rescan verbs.
`bridge.stats` (`FrameStats`) keeps tick timings against the 4 ms budget.

## JS modules

**bridge.js** - a bridge is `{ kind, call(slot, payload) -> Promise<reply>,
onFrame(fn) -> off, onDialog(fn) -> off }` with parsed values (JSON stays in
this module). `connectBridge()` resolves the QWebChannel bridge, or `null`
outside Qt. `createFakeBridge({describe, values, actions})` has the same
interface plus `calls` (`[slot, payload]`), `values`, `transport`,
`dialogAnswers` (paths the next dialogs return) and `pushFrame(extra)` (emit
the changes and events since the last frame); `actions[verb](args, fake)`
returns a verb's result.

**core-client.js** - `createCoreClient(bridge, {schedule})`:
`get(addresses) -> Promise<{addr: value}>`; `set(addr, value, {burst,
editAll})` (coalesced per animation frame, latest value per address, one
`set` call); `flush() -> Promise<errors>`; `act(verb, args) -> Promise<event>`
(`{status: 'done', result}` or `{status: 'error', error}`);
`describe(prefix | list) -> Promise<{addr: meta}>`; `beginGesture()`,
`endGesture()`; `openFile(opts)`, `saveFile(opts)`, `chooseFolder(opts) ->
Promise<path | null>`; `resizeWindow(from, to)`; `onFrame(fn) -> off`. It
sends `resync` on creation. `act` resolves with the done or error event
(`progress` events of exports are skipped).

**store.js** - `createStore()` (pure, no DOM): `seed({addr: value})`,
`apply(frame)`, `assume(addr, value)` (local echo), `value(addr)`, `has(addr)`,
`watch(addr, fn(value, addr), {now = true}) -> off` (called only when the value
changes), `watched()` (addresses with live watchers), `readout(name)`,
`watchReadout(name, fn, {now = true}) -> off` for `transport`, `modulation`,
`audio`, `midi`; `onEvent(fn) -> off`; `version`.

Boot (`main.js`): connect, client, store (`client.onFrame(store.apply)`),
`mountStage`, `describe('')` (every registered address), `mountPanel`, then
`store.seed(await client.get(store.watched()))`.

**Controls declare what they edit**: `data-address="global.tempo"` on the
element of a control bound to an address, `data-verb="transport.toggle"` on
one that starts a verb. The parity guard reads them.

## Controls (`controls.js`)

Custom elements, each bound to the address in its `data-address` and that
address's `describe()` metadata (range, curve, labels, unit). They bind
themselves when inserted under a root that has a control context
(`provideContext(stage, createControlContext({store, client, meta, root}))`;
the panel does this for the stage, so later slices only insert elements).
Changing `data-address` rebinds; removing it disables the control ("off").

| Element | Attributes | Behaviour |
|---|---|---|
| `<px-knob>` | `label`, `name` (display name), `size` (px), `wheel-step` (engine units, instead of 1 %) | drag up/down, 200 design px = range, Shift x0.1; wheel 1 %; pointer and end dots, modulation band, ghost marker |
| `<px-fader>` | `label`, `name`, `length` (track px = drag range), `orient="h"` (sideways), `reverse` (high end left), `ends` (`osc,noise`) | as the knob, wheel 2 %; a track click jumps; lit by `--cc` |
| `<px-toggle>` | `label`, `variant` (`red`) | a lit button for a bool address |
| `<px-switch>` | `options` (comma-separated display names of the labels), `values` (the labels offered) | segmented buttons (up to 5 options), wheel steps; a value outside `values` shows lit beside them |
| `<px-list>` | `options` | list box and menu (enums, int ranges), wheel steps |
| `<px-target>` | `name` | a source's → destination button (`ch<N>.<source>.target`): the target's short name, wheel steps through the 28 targets, a click dispatches `px-assign` (`{address}`), right-click: assign / clear |
| `<px-display>` | | `setBase(l1, l2)`, `show(l1, l2, ms = 1500)`, `alert(l1, l2, ms = 4000)`, `text()` |

All controls: double-click opens an exact-value field (engine units, ratios
in percent, pans `L30`/`C`/`R30`, `k` and `s` suffixes); right-click opens
reset to default, MIDI learn / cancel (`midi.learn` with the address as
target), remove CC mapping (`midi.cc_map`), assign / remove pitch bend
(`midi.pitchbend_target`). A drag is one gesture (`beginGesture` /
`endGesture`), a wheel turn a burst, typed values, resets and clicks one
undo step each. Each touch dispatches a bubbling `px-touch` event
(`{address, name, value, text}`); the panel display shows it for 1.5 s.
A control bound to an address the store lacks reads it (`ctx.ensure`; the
addresses of one task go in one `get`). MIDI cues come from the store: the CC badge from `midi.cc_map`
(`selected.*` targets on the selected channel), the learn pulse from
`midi.learning`, the LED blink and the pickup ghost from `poll()['midi']
.pickup`. Modulation bands come from `poll()['modulation'].channels`,
coloured by the first source (LFO 1, LFO 2, pump) that is on and aims there:
`modulationBand(meta, value, offset)` gives the band from the set value to
the modulated one (`data-mod` = source, `data-mod-to` = modulated position).
The context also offers `ctx.openMenu(items, x, y)`, `ctx.set(address,
value, {burst})` (local echo + set) and `ctx.ensure(address)`.

## The face (`panel.js`)

Strips, top row, left and right columns and START/STOP as decided in #7 and
#12. The strip CTRL mode is the preference `pref.ui.ctrl_knob` = `{mode:
'off'|'pan'|'reverb_mix'|'delay_mix'|'lfo1_depth'|'user', user: [8 sound
suffixes or null]}`; the CTRL knob's `data-address` follows it. Slots for
the later slices are elements with `data-slot` (`panel.slot(name)`):
`step-entry` and `patterns` (left column), `preset-prev`, `preset-next`,
`preset`, `po32`, `setup` (right column placeholders), `rack-toggle` (the
edit rack button, wired by `rack.js`),
`steps` (bottom row), `rack` (the edit rack drawer, 1568x294 at y = 690).
A slice fills its slot with `px-*` elements and buttons with `data-verb`.
The display's first line is the selected pattern and the page on the pads
(`PATTERN A  17-32`), the second the preset name.

## Step row and patterns (`steps.js`, `patterns.js`)

Decision #8 (vocabulary #15). The page **reads lanes**
(`pattern.<P>.ch<N>.<field>`, lists, for the 8 channels of the selected
pattern, plus `pattern.<P>.length`) and **writes steps**
(`pattern.<P>.ch<N>.step<S>.<field>`), echoing the lanes into the store at
once (`withStep`: a trigger turned off clears accent and fill). The selected
pattern comes from `poll()['transport'].selected_pattern` (else
`pattern.selected`); switching it re-watches the lanes and reads any the
store lacks.

- **Step modes** (left column, `[data-mode]`): trig / accent / fill toggle on
  press and paint along the row (every step between two pointer moves; a
  stroke is one `beginGesture` / `endGesture`); accent and fill reach only
  triggered steps. velo (triggered pads) and prob (any pad) are vertical
  drags, 200 design px = the range, Shift x0.1, one gesture. sub opens the
  substep menu (none, the 14 presets, custom… with an `o`/`-` field).
- **Pads** (`.pads .stp[data-step]`, absolute step numbers) show every
  property whatever the mode: `on` (trigger, lit at `--lvl` = velocity as
  brightness, 1 when accented), `accent` dot, `filled` stripe, probability
  below 100 %, substep dots; velo and prob modes add a bar and the value.
  Steps past the length are `out` (dim, not editable). Beat groups `g1..g4`.
- **last step** (`#last-step`, bound to `pattern.<P>.length`) arms; the next
  pad on any page sets the length (one undo step). **all ch** (`#all-ch`)
  writes the same step on all 8 channels, muted ones included, as one
  gesture. **follow** (`#follow`) keeps the pads on the playing page; a page
  bar (`.pg[data-page]`) turns it off. The playhead (`.ph`) and the pulsing
  page bar show only while the selected pattern is the one playing.
- **Matrix** (`#matrix-toggle`): the 8 channels x 16 steps of the page as a
  page of the edit rack drawer, cells (`.matrix .stp[data-channel][data-step]`)
  drawn and edited like the pads (a paint stroke stays in its row); a
  channel label selects the channel; ⊞ again or `◀ edit rack` hides it.
- **Patterns** (`.pbtn[data-pattern]`, bound to `pattern.<P>.empty`, its
  wrapper to `.chained`, its length badge to `.length`; the grid to
  `pattern.selected`): click = `pattern.select` (queued while playing);
  classes `on` (selected), `playing`, `queued` (blinks), `empty`,
  `chain-in` / `chain-out` (amber links), the length when not 16.
  `#chain-prev` / `#chain-next` toggle the selected pattern's links and show
  the chain on the display; `#lane-copy` / `#lane-paste` copy and paste the
  selected channel's lane. MENU (`#pattern-menu`) and a right-click on a
  pattern open the pattern menu: play next (`pattern.queue`, while playing),
  cut / copy / paste / exchange / clear, shift left / right, reverse,
  randomize, alter, randomize accents + fills, randomize (AI) and randomize
  chN (AI) (`ai.randomize_pattern`, off without `ai.available`), copy / paste
  lane chN, clear all chains. Results show on the display (`nothing to
  paste`); errors alert there.

**Extension points.**

- `panel.drawer` (`drawer.js`): `setBase(element)` (the edit rack is the
  base page), `show(name, element, {onHide})`, `hide(name)`,
  `toggle(name, build)`, `current`, `onChange(fn)`, and
  `setOpener({isOpen, setOpen})` (the edit rack plugs in the drawer's open /
  closed state: window shrink, saved preference). A page opens a closed
  drawer and closing the page restores it (also for the PO-32 and AI pages,
  #13).
- `panel.patterns.addMenuItems((letter, index) => [[label, action], ...])`:
  more pattern menu entries (export to MIDI / audio, #13);
  `panel.patterns.openMenu(index, x, y)`.
- `panel.steps`: `state` (`mode`, `page`, `follow`, `allCh`, `armed`,
  `selected`), `setMode(field)`, `setPage(page)`, `render()`, `matrix`.

## Edit rack (`rack.js`, `rack-layout.js`)

Decision #11 (open / close: #7). The drawer's base page: a header (channel,
patch name, drum type, the click-to-assign bar, **edit all** bound to
`global.edit_all`, **drum patch ▾**: load / save `.mtdrum`, export the
channel's hit as WAV, asking before replacing a file) over one row of
sections: oscillator, noise, envelopes (fader bank), mix (osc · noise
crossfader), velocity (fader bank), FX, then modulation with the LFO 1, LFO 2
and pump rows (dimmed while off). `RACK_SECTIONS` lists every control with
its address suffix, type, label and display name; the rack binds
`ch<N>.<suffix>` of the selected channel (`data-suffix`, `data-address`)
and rebinds when `global.channel` changes. Tune, osc decay and level stay on
the strips (`FACE_ONLY`). Additions to the decided rows, so every sound
address is reachable: LFO **phase**, pump **sync**.

- **Click to assign**: a row's → button (`px-target`) arms the source
  (`panel.rack.arm(source)`): the stage gets `assigning` and `--assign` (the
  source colour), every control the source can aim at gets `assignable`
  (`destinationOf(address, channel)`: the selected channel's controls with a
  mod target, master and the sound morph; the current one `assigned`).
  Clicking one sets `ch<N>.<source>.target` and disarms; any other control
  refuses on the display (`not a destination`, `not on CH1`); **off** clears
  the destination, **cancel**, the → button again, a channel switch or a
  drawer page disarm. Bands show on the rack and every strip (controls.js).
- **Open / closed**: the right column's `#rack-toggle` closes the drawer:
  the stage becomes 1600x700 (`setStageHeight`), the rack slot hides, and
  `client.resizeWindow(1000, 700)` shrinks the window by the rack's height
  at the panel's scale (back on opening). The state is `pref.ui.rack_open`
  (open until set), applied at start-up. `drawer.setOpener` is plugged in: a
  page (the matrix) opens a closed drawer while it shows, without touching
  the preference.
- `panel.rack`: `open`, `setOpen(open, {save})`, `arm(source)`, `disarm()`,
  `assigning`, `element`.

## Tests

All from pytest, offscreen, no Node and no display needed:

```bash
python -m pytest tests/web -q                      # the web layer (~7 s)
node --test pythonic/web/static/test/*.test.js     # optional local shortcut for pure specs
```

`tests/web/conftest.py` sets `QT_QPA_PLATFORM=offscreen` (unless set),
renders Chromium in software (`prepare_headless`) and registers the scheme
before pytest-qt makes the QApplication. CI also sets
`QTWEBENGINE_DISABLE_SANDBOX=1`. To watch a test, run it with
`QT_QPA_PLATFORM=xcb`.

- **JS specs** (`test_js_specs.py`): every `static/test/*.test.js` and
  `*.engine-spec.js` runs inside QtWebEngine via `test/runner.html?spec=...`
  (an import map maps `node:test` / `node:assert/strict` to small shims), one
  pytest test per file. Write specs in `node:test` style: `test(name, fn)`,
  `assert.equal/deepEqual/ok/throws/rejects`. The spec view is hidden, so
  Chromium delays `setTimeout` up to 1 s: wait with a `MessageChannel` tick
  (see `steps.engine-spec.js`), not timers. Shared fixtures that are not
  specs (`pattern-fixtures.js`) sit beside them.
- **Page tests** (the main layer): fixtures `panel` (the real page over a
  `FakeCore`), `open_panel(core, owns_core=True)`, `fake_core`, `core_table`
  (session: `describe()` and values of every address of a real core),
  `real_core` (a started `AppCore` on `FakeAudioBackend`; `real_core.backend
  .stream.pull()` runs one audio callback). Windows are disposed at teardown
  and the test fails on JS console errors.
- **`Page`** (`tests/web/page.py`): `js(expr)` (value via JSON text),
  `run(statements)`, `wait_js(expr, timeout, pump)`, `wait_ready()`,
  `rect(sel)`, `center(sel)`, `click(sel)`, `wheel(sel, steps)`,
  `drag(sel, dy, modifiers)`, `press`, `right_click`, `double_click`,
  `type_text(text)` (real input on the view's focus proxy; `drag` takes `dx`
  for strokes along the pads; Chromium merges
  wheel events sent at once, so turn notch by notch), `pixel(x, y)`, `color_at(sel, fx, fy)`,
  `close_to(color, expected, tolerance)`, `wait_pixels(check)` (coarse
  `grab()` checks, no screenshot diffs; painting lags the DOM, so wait for
  pixels instead of sleeping), `answer_dialog(path | None)`, `tick()`,
  `bound_addresses()`, `console_errors()`; `choose_in_dialog(qtbot, dialog,
  path)`, `same_path(a, b)` (Qt answers paths with `/` on Windows).
- **`FakeCore`** (`tests/web/fake_core.py`): get/set/describe/act/poll from
  the real metadata, sets coerced as the real core does and reported by the
  next poll; `calls` (`('set', addr, value, opts)`, `('act', verb, args)`,
  `('gesture', phase)`), `sets()`, `verbs_called()`, `post_change(addr, v)`
  (a change from the core's side), `transport`, `readouts`, `verbs` (verb ->
  handler), `closed`; `patterns` (`FakePatterns`: step and lane addresses
  with the core's rules, lanes reported on every step set, `set(parsed,
  value)` for core-side edits); `pattern.select` moves the transport's
  selected pattern.
- **Real-core end-to-end tests** (`test_real_core.py`): play and the playhead,
  a tempo edit through a preset file round trip, the frame budget, a tune
  drag and its undo, select and mute, the CTRL preference, a paint stroke
  and its one-step undo, follow over a 32-step pattern, lane copy / paste,
  a rack fader on the selected channel and its undo, click to assign an LFO
  destination, bands moving while an LFO runs, the rack-open preference.
- **Parity guard** (`test_parity_guard.py`): every address the core's
  `describe()` lists is bound by a `data-address` control or listed in
  `tests/web/parity_absent.json`: `{"absent": {"<fnmatch glob>": "<reason>"}}`
  (the guard selects each channel in turn, since the rack binds the selected one's).
  It also fails on a bound address still listed and on globs that match
  nothing. A slice that binds controls deletes their entries; a core change
  that adds addresses lists them there (or binds them).

Each web slice adds page tests for the behaviours it builds and a manual
parity list in `docs/parity/`.
