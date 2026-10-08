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
pythonic --quit-after 5            # close after 5 s; prints the frame timings; non-zero exit if the page did not boot
```

Without QtWebEngine, `--ui web` starts tkinter with a dialog that explains
why and offers to install PySide6 (on Windows ARM64 it only explains).
`pythonic/web/static/index.html` also opens in a plain browser served over
HTTP (`python -m http.server -d pythonic/web/static`): a fake bridge then
stands in for the core (tempo and START/STOP work).

Fonts: `python tools/fetch_fonts.py` downloads the bundled OFL fonts into
`pythonic/web/static/fonts/` (see `SOURCES.txt` there); until they are
there the panel falls back to the system fonts of the stacks in
`tokens.css`. The `app://` handler serves `css/fonts.css` without the
`@font-face` rules of missing files, so the page asks for no file that is
not there (DevTools would log each failed load as a console error).

Real runs for checks: always with a scratch preferences folder (Linux:
`XDG_CONFIG_HOME=/tmp/x pythonic ...`, which `platformdirs` honours), never
the user's own. `--devtools` exposes the page to the Chrome DevTools
protocol (`http://127.0.0.1:9222/json` lists it), enough to script a session
(`Runtime.evaluate`, `Input.dispatchMouseEvent`, `Page.captureScreenshot`).

## Layout

```
pythonic/web/
  __init__.py      STATIC_DIR
  scheme.py        app:// scheme: register_scheme() before QApplication, install_handler(profile)
                   (StaticHandler.missing: paths asked for and not found); available_fonts_css
  bridge.py        Bridge QObject (slots + frame signal), FrameStats, FRAME_HZ = 60, FRAME_BUDGET_MS = 4
  window.py        PanelWindow(core, owns_core=True): view + page + channel; MIN_SIZE 1280x800;
                   fit_stage_height(old, new) (the edit rack drawer); dispose()
  app.py           run(quit_after, devtools_port, core): the --ui web entry; prepare_rendering(gpu)
                   (--disable-gpu offscreen or when pref.web.gpu is off);
                   frame_report(stats): the line a --quit-after run prints
  static/
    index.html     loads css/*, qrc:///qtwebchannel/qwebchannel.js, js/main.js
    css/tokens.css design tokens (palette, fonts, stage size) as custom properties
    css/fonts.css  @font-face of the bundled fonts
    css/panel.css  stage and panel styles
    css/setup.css  the setup sheet
    css/ai.css     the AI drum generator page
    css/po32.css   the PO-32 page
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
    js/sheet.js        createSheets(stage, {onShow}): overlay sheets and the alert sheet
    js/files.js        createFileFlows(...): file verbs with progress, failures and the replace question
    js/presets.js      mountPresets(...): PRESET ◀ ▶ and the preset menu; neighbourFile, canSaveInPlace
    js/exports.js      mountExports(...): the pattern menu's export to MIDI / audio (tail popover)
    js/setup-logic.js  pure setup sheet logic: tabs, note names, CC map rows and edits, target names / menu groups, rates
    js/setup.js        mountSetup({panel, store, client, meta}): the setup sheet (registers the 'setup' page)
    js/ai-logic.js     pure AI page logic: laneView, modelView, bankText, previewArgs, leaveQuestion, parseSeed, ...
    js/ai-page.js      mountAiPage({panel, store, client, stage}): the AI drum generator drawer page
    js/po32-logic.js   pure PO-32 page logic: stage flows, picks and letters, level meter, stage texts
    js/po32.js         mountPo32Page(panel, {store, client}): the 'po32' drawer page (transfer, import)
    js/main.js         boot({bridge, stage}), demoBridge(); sets window.pythonic
    test/              runner.html, shim/, *.test.js (pure), *.engine-spec.js (DOM)
```

`window.pythonic` = `{ bridge, client, store, meta, panel, setup, ready }` once
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
| `trigger` | `{"channel": 1..8, "velocity": 1..127}` (default 127) | `{}`: hits the channel now (`core.trigger`, 0-based there), as a pad or a MIDI note |

Signals (JSON text): `frame` = one `core.poll(since)` result per tick of a
60 Hz main-thread QTimer, sent when it has changes or events or a readout
changed (`version`, `changes`, `events`, `transport`, `modulation`, `audio`,
`midi`); `dialog` = `{"id", "path"}` (`null` when cancelled). Dialogs are
window-modal `QFileDialog`s opened with `open()` (never `exec()`): the frame
timer keeps running. Save dialogs do not confirm overwrites; the core's save
verbs refuse to replace a file and the page asks. Device lists come from
`describe()` labels (`pref.audio.device`, `midi.device`) and the rescan verbs.
`bridge.stats` (`FrameStats`) keeps tick timings (wall clock, so waits for
the GIL count) against the 4 ms budget; a `--quit-after` run prints them
(`UI frames: n ticks, mean, max, k over the 4 ms budget`). Measured on a
real run playing a preset: mean 0.5 ms, max 2.5 ms.

## JS modules

**bridge.js** - a bridge is `{ kind, call(slot, payload) -> Promise<reply>,
onFrame(fn) -> off, onDialog(fn) -> off }` with parsed values (JSON stays in
this module). `connectBridge()` resolves the QWebChannel bridge, or `null`
outside Qt. `createFakeBridge({describe, values, actions})` has the same
interface plus `calls` (`[slot, payload]`), `values`, `transport`,
`dialogAnswers` (paths the next dialogs return), `post(event)` (queue any
event: progress, errors without an action) and `pushFrame(extra)` (emit
the changes and events since the last frame); `actions[verb](args, fake,
id)` returns a verb's result.

**core-client.js** - `createCoreClient(bridge, {schedule})`:
`get(addresses) -> Promise<{addr: value}>`; `set(addr, value, {burst,
editAll})` (coalesced per animation frame, latest value per address, one
`set` call); `flush() -> Promise<errors>`; `act(verb, args, {onProgress}) -> Promise<event>`
(`{status: 'done', result}` or `{status: 'error', error}`; `onProgress(fraction,
event)` gets the verb's `progress` events);
`describe(prefix | list) -> Promise<{addr: meta}>`; `beginGesture()`,
`endGesture()`; `openFile(opts)`, `saveFile(opts)`, `chooseFolder(opts) ->
Promise<path | null>`; `resizeWindow(from, to)`; `trigger(channel,
velocity)`; `onUnclaimedError(fn) -> off` (error events nobody waits for:
those without an action id, and errors of an action still unclaimed two
frames later while no `act` waits for its id); `onFrame(fn) -> off`. It
sends `resync` on creation. `act` resolves with the done or error event.

**store.js** - `createStore()` (pure, no DOM): `seed({addr: value})`,
`apply(frame)`, `assume(addr, value)` (local echo), `value(addr)`, `has(addr)`,
`watch(addr, fn(value, addr), {now = true}) -> off` (called only when the value
changes), `watched()` (addresses with live watchers), `readout(name)`,
`watchReadout(name, fn, {now = true}) -> off` for `transport`, `modulation`,
`audio`, `midi`, `po32`; `onEvent(fn) -> off`; `version`. Local echo:
for `ECHO_HOLD_MS` (250 ms) after `assume(addr)`, a different value a frame
reports for that address is held back (a frame of an earlier set of a drag
would pull the knob back) until the core reports the echoed value; when the
hold ends the latest held value shows. `createStore({now, later})` takes a
clock and a timer for tests.

Boot (`main.js`): connect, client, store (`client.onFrame(store.apply)`),
`mountStage`, `describe('')` (every registered address), `mountPanel`, the
pages (`mountSetup`, `mountAiPage`, `mountPo32Page`), then `store.seed(await client.get(store.watched()))`.

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
| `<px-display>` | | `setBase(l1, l2)`, `show(l1, l2, ms = 1500)`, `alert(l1, l2, ms = 4000)`, `text()`; a line too long to fit scrolls to its end and back (`.line.scroll`) and the message stays until read once |

Knobs and faders: double-click opens an exact-value field (engine units, ratios
in percent, pans `L30`/`C`/`R30`, `k` and `s` suffixes; switches and lists
pick a value with one click). All controls: right-click opens
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
`preset` (presets.js), `po32`, `setup` (open their pages), `rack-toggle` (the
edit rack button, wired by `rack.js`),
`steps` (bottom row), `rack` (the edit rack drawer, 1568x294 at y = 690).
A slice fills its slot with `px-*` elements and buttons with `data-verb`.
The display's first line is the selected pattern and the page on the pads
(`PATTERN A  17-32`), the second the preset name. A click on the selected
channel's button hits it (bridge `trigger`, velocity 64, Ctrl+click 127, as
tkinter); a channel button flashes when a MIDI note (`poll().midi.notes`)
or a click hits its channel. UNDO / REDO show the step's label as the name
of the control bound to it (`undoText`, values.js: `CH2 DECAY`), else the
address in words. A program button's tooltip names its kit
(`program.names`: `program 3: 808`), and a click shows it on the display
(`PROGRAM` / `3 808`).

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
- `panel.rack`: `open`, `setOpen(open, {save})`, `arm(source)`, `disarm()`, `patchItems()`,
  `assigning`, `element`.

## Sheets, pages and errors (`sheet.js`, `panel.js`)

Decisions #13 (where secondary features open) and #22 (the alert sheet).

- **Overlay sheets**: `panel.sheets.show(name, element, {width, dismissable,
  onHide, className}) -> card` puts the element on a sheet over a layer that
  dims the whole stage: frames keep arriving and the panel keeps playing and
  animating, but it takes no input (pointer, wheel, context menu stop at the
  layer); showing a sheet closes open menus and ends click to assign.
  `hide(name?)` (the top one without a name), `current`, `isOpen(name)`.
  Sheets stack (an alert over the setup sheet); `dismissable` sheets close
  with a click outside (the setup sheet, #13), others by their own buttons.
  Menus (`.px-menu`, z 60) open above sheets (z 40+), so lists inside a
  sheet work.
- **Alert sheet**: `panel.sheets.alert({title, text, tone: 'error' | 'ok',
  buttons: [{label, value, primary}]}) -> Promise<value>`: 460 wide, a red
  top edge for errors, green for questions and success, the primary button
  lit on the right, closed only by its buttons; alerts show one at a time
  in order, and an error alert equal to one showing or waiting resolves at
  once with null. `panel.sheets.ask(title, text, {yes, no, tone}) ->
  Promise<boolean>` is a yes / no question (overwrite, discard, ...).
- **Errors**: `panel.act(verb, args)` shows a verb's error on the display.
  File verbs (`panel.files`, files.js) also put it on a red alert (it needs
  reading). Error events nobody waits for (`client.onUnclaimedError`: audio
  callback, stalled stream, AI worker, core jobs, MIDI, PO-32) go through
  `panel.reportError(event)`: the display (`AUDIO ERROR`), plus a red alert
  except for `midi` and `po32` sources (frequent, or shown by their page).
- **File flows** (`panel.files`): `run(verb, args, {label, failTitle}) ->
  result | null` (progress `42 %` on the display under `label`, a failure on
  the display and a red alert titled `failTitle`); `save(verb, args, {label,
  failTitle, done})`: when the core refuses because the file exists
  (`{saved: false, exists: true, path}` or `paths` for `export.drum_wavs`) it
  asks `Replace “name”?` on the alert sheet and saves again with
  `overwrite: true`; `done` (text or `fn(result)`) shows on success.
- **Pages** (the registration API for W6-W8): `panel.registerPage(name,
  open) -> unregister` and `panel.openPage(name, options) -> open(options)`.
  The panel opens pages from: PO-32 button `openPage('po32')`, SETUP button
  `openPage('setup')`, MIDI LED `openPage('setup', {tab: 'midi'})`, a
  control's right-click ▸ CC mappings… `openPage('setup', {tab: 'midi'})`,
  PRESET menu transfer to / import from PO-32 `openPage('po32', {tab:
  'transfer' | 'import'})`, AI drum generator `openPage('ai')`, setup…
  `openPage('setup')`. `open` decides the container: a drawer page
  (`panel.drawer.show(name, element, {onHide})`, PO-32 and AI) or a sheet
  (`panel.sheets.show('setup', element, {dismissable: true})`). Without a
  registration the alert sheet says the page is coming soon. `panel.pages`
  lists the registered names. Pages are mounted in `main.js` after
  `mountPanel` (before the store is seeded, so their watched addresses are
  read at boot).
- **Preset load guards**: `panel.guardPresetLoad(fn) -> off`; a preset load
  from the PRESET menu, ◀ ▶ or reload last waits for every `fn()` (a boolean
  or a promise of one) and is cancelled when one answers false (the AI page
  asks keep or revert first).

## PRESET menu (`presets.js`)

Decisions #12, #13, #14. The right column's ◀ ▶ load the previous / next
file of `preset.files` around `preset.path` (no wrap, as tkinter; with the
current preset outside the folder ▶ loads the first file; disabled at the
ends); while a factory preset is loaded (`preset.factory`) they walk
`factory.presets` instead. PRESET (`#preset-button`) toggles the preset menu
(`#preset-menu`):

- left: the factory presets (`.pm-factory`, marked read-only, the current
  one lit, a click loads it with `preset.load({factory: name})`), the preset
  folder's name (↻ refreshes) over its presets as an in-panel list
  (`.pm-files`, the current one lit, a click loads it by name relative to the
  folder), then the recent files (`.pm-recent`, `pref.recent_files`, a click
  loads the path);
- right: open preset… (native open dialog in the preset folder), save
  “name.json” (over the open JSON preset, no question; a .mtpreset or a
  factory preset falls back to save as), save preset as… (native save dialog, suffix `.json`,
  the replace question), reload last preset (`preset.load_last`), copy /
  cut / paste preset (paste off while `preset.clipboard` is false),
  initialize preset, randomize all, restore factory kits… (asks on the alert
  sheet, saying so when the current program is among 1-6, then
  `program.restore_factory`); the selected channel's drum patch
  entries (`panel.rack.patchItems()`, the rack header's menu, so they work
  with the rack closed), export every drum as WAV… (folder dialog,
  `export.drum_wavs`); preset folder… (folder dialog, sets
  `pref.preset_folder`, `preset.files` follows), refresh list; transfer to
  PO-32…, import from PO-32…, AI drum generator…, setup… (pages, above).

A load or save failure shows on a red alert; the display shows the loaded
preset (`PRESET` / name, `FACTORY PRESET` / name) or `saved`. The menu
follows the folder, recent files, clipboard and factory state while open.
Bound addresses: `preset.name`, `.path`, `.files`, `.clipboard`, `.factory`,
`factory.presets`, `pref.preset_folder`, `pref.recent_files`.

## Pattern exports (`exports.js`)

Decision #13. Two pattern menu entries (`panel.patterns.addMenuItems`):
**export to MIDI…** goes straight to the native save dialog
(`pythonic_pattern_<X>.mid`) and `export.midi`; **export to audio…** opens
the tail popover (`#tail-pop`) on the pattern MENU button: cut / add 2 s /
loop +1 pass (`[data-tail]`, the last choice kept for the session) and
save wav… (`#tail-save`), then the save dialog (`pythonic_pattern_<X>.wav`)
and `export.wav` with the tail; the render's progress shows on the display
(`PATTERN B WAV` / `40 %`), then `saved b.wav`. Both ask before replacing a
file. `panel.exports`: `exportMidi(letter)`, `exportAudio(letter, tail)`,
`openTailPopover(letter)`, `tail`.

## Setup sheet (`setup.js`, `setup-logic.js`)

Decision #22 (container #13, pickers #14). `mountSetup` (main.js, after the
panel) registers the `setup` page: `panel.sheets.show('setup', element,
{dismissable: true})`, tabs on top (`.su-tabs .btn[data-tab]`: audio | midi |
synthesis | ai | display), the card's width follows the tab (`TAB_WIDTHS`). SETUP opens
it on audio; `openPage('setup', {tab})` on a tab (the MIDI LED and CC
mappings… use midi). ✕ or a click outside closes it. `pythonic.setup`:
`open(options)`, `close()`, `tab`, `element`.

- **audio**: lists (`.su-choice[data-address]`, a menu of the options) for
  `pref.audio.device` (options from `audio.output_devices`; `(system
  default)` = null), `.buffer_ms`, `.sample_rate` (the rates `audio.rates`
  reports for the device, with a "takes n of 7 rates" note), `.synth_rate`
  (0 = same as output), `px-toggle` `pref.audio.mono`; `#su-audio-rescan`
  (`audio.rescan`); `#su-restart` (`audio.apply`) lit while
  `pref.audio.pending` lists a field, whose `.su-dot` lights; the running
  stream from the `audio.*` readouts; `pref.audio.input_device`
  (`audio.input_devices`, `audio.default_input`).
- **midi**: `midi.device` list: (off) = `midi.close`, (auto-detect) =
  `midi.open` without a device, a port = `midi.open({device})`; ports from
  `describe('midi.device').labels`, then `#su-midi-rescan` (`midi.rescan`);
  the LED (`midi.connected`, activity); `#su-base-note` (octave menu, wheel,
  `#su-base-down` / `#su-base-up`) and its 8 notes; `px-toggle`
  `midi.clock_sync` and `midi.synced_tempo`. CC mappings (`.su-ccrows`,
  `midi.cc_map`): rows `.su-ccrow[data-cc]` with the CC field (`.su-cc`, type
  0-127 or wheel; a CC another row holds moves here), the control
  (`.su-target`: a menu of sections, then controls: the selected channel's
  sound parameters as `selected.<suffix>`, or globals), live activity
  (`poll().midi.pickup` of that CC: LED and position bar), ✕; `#su-cc-add`
  adds a draft row with a free CC that is written once a control is chosen;
  `#su-cc-clear` asks first. `.su-bend` sets `midi.pitchbend_target`. A learn
  in progress (`midi.learning`) shows with cancel; `#su-learn` closes the
  sheet and points to the panel's right-click ▸ MIDI learn.
- **synthesis**: `px-knob` `pref.smoothing_ms`.
- **ai**: `pref.ai.pattern_model` / `.patch_model` (`.su-browse[data-kind]`:
  native open dialog for `*.pt`; `.su-clear`: back to the bundled one, as
  `ai.models` shows), `px-knob` `pref.ai.pattern_temperature` /
  `.patch_temperature`; a note when `ai.available` is false.
- **display**: `px-toggle` `pref.web.gpu` (GPU rendering of the panel); it
  applies at the next start, which the panel display says on a click.

Right-click menus leave out MIDI learn and pitch bend for settings
(`midi.*`, `audio.*`, `pref.*`: the core refuses them as targets) and CC
mappings… for controls on the setup sheet.

## AI drum generator page (`ai-page.js`, `ai-logic.js`)

Decisions #13, #21 (settings shared with the setup sheet's ai tab: #22). A
drawer page (`panel.openPage('ai')`, from the PRESET menu) whose three
columns line up with the face's, so each lane sits under its strip:

- **left**: patch and pattern model lines (`ai.models`: file ✓ (sampling),
  loading…, error, no model; `load…` opens the native dialog and runs
  `ai.load_model` with the path, which the core saves as `pref.ai.<kind>_model`
  once loaded), the patch temperature knob (`pref.ai.patch_temperature`),
  candidates ‹ n › (1..32, default 8, wheel too), seed (blank = random, ⟳
  reseeds), **generate all 8** (`ai.generate` with candidates and seed; in
  generate-new mode it then runs `ai.generate_patterns` with the seed). The
  saved or bundled models still `unloaded` load when the page opens.
- **lanes** (`.ai-lane[data-channel]`): drum type (`px-list` on
  `ai.ch<N>.type`, any of 18), gen (`ai.generate` with channel and type), ‹
  `i/n` › (`ai.try` with `step` ±1), the candidate name, try / trying ✓
  (`ai.try` / `ai.untry`), and a note (generating…, error: …). A lane that
  generates, steps or tries is hit at 127 while the transport is stopped
  (tkinter's preview). The strip tab of a channel trying a candidate gets
  `aitry` (its name, now the channel's, in italics with a dashed outline).
  Generating a lane or stepping it drops the AI pattern bank
  (`ai.clear_patterns`, as tkinter).
- **right**: keep current / generate new patterns (page state, as the
  tkinter dialog), the pattern temperature knob
  (`pref.ai.pattern_temperature`), the bank note (`ai.bank`) with ↻ (a new
  bank), ▶ loop / ▶ bank A→L (`ai.pattern_try`, a second click stops; `bank`
  only in generate-new mode), keep tried (`ai.keep`, one undo step; off
  while `ai.tried` is empty), replace patterns (`ai.replace_patterns`, only
  with a bank ready in generate-new mode), revert all (`ai.revert`).
- **Without the ML extras** (`ai.available` false) the lanes give way to
  the `ai.install_command` with copy (clipboard) and install now (asks, then
  `ai.install`; `ai.installing` greys it; success says so and loads the
  models: the worker needs no restart).
- **Header**: `◀ ch n edit`, the state (`ai.state`: generating…, loading a
  model…, installing…, ML extras missing), the channels trying, ✕.
- **Leaving with tried sounds** asks keep or revert on the alert sheet: ✕
  and `◀ ch n edit` ask first (cancel stays); another drawer page or closing
  the rack ask once the page is gone (keep / revert); a preset load asks
  first (a guard). Keep also stops a pattern preview; leaving without tried
  sounds just stops it. Verb errors show on the display; generation, model
  and install failures also on a red alert.

`panel.ai` = `{element, open, leave, settle(cancellable), patternMode,
candidates, destroy}`.

## PO-32 page (`po32.js`, `po32-logic.js`)

Decisions #13 (one edit rack page, transfer and import tabs, the panel live)
and #20 (inner layout). `mountPo32Page(panel, {store, client})` registers the
`po32` page: `openPage('po32', {tab: 'transfer' | 'import'})` shows it as a
drawer page (`drawer.show('po32', ...)`: a closed drawer opens and closing
restores it); without a tab it opens on the last one, and the PO-32 button
toggles it (lit while it shows). `◀ ch N edit` and ✕ close it. Each tab is a
numbered stage flow (`.po-stage[data-stage][data-state]`): the current stage
`now` (lit), done ones `done` (ticked), later ones `later` (dimmed, usable).

- **Transfer**: 1 choose: sounds to 1–8 / 9–16 (`[data-send-bank]`), the
  pattern chain (`#po32-chain`, `po32.chain_options`) with the note of the
  PO-32 slots sent empty (`#po32-slots`, from `po32.prepare`'s `slots`), the 8
  channels (`.po-check[data-channel]`, copied from the face mutes each time
  the page opens, independent of MUTE after that); every change runs
  `po32.prepare` (the length on `#po32-status`). 2 prepare the PO-32
  (instructions). 3 send: progress bar (`#po32-progress`, `poll().po32
  .progress`, also on the display), `#po32-send` (transfer / stop:
  `po32.send` / `po32.cancel`), `#po32-save-wav` (native save dialog,
  `po32.save_wav`, the replace question). A failed send shows on a red alert.
- **Import**: 1 listen: input device (`#po32-input`, `audio.input_devices`;
  the page's choice, defaulting to `pref.audio.input_device`, then the
  default input; ↻ `audio.rescan`), monitor (`po32.listen`) and the level
  meter (`#po32-meter`, `poll().po32.level`, dB, held peak), record / stop
  (`po32.record` / `po32.stop`, which decodes), import wav… (native open
  dialog, `po32.decode`), the decode status (`#po32-source`), keep
  recordings (`pref.po32.save_recordings`). 2 bank: bank 0 / 1
  (`[data-bank]`, enabled when decoded) and the bank's 8 sounds. 3 pick
  patterns: 16 buttons (`.po-pat[data-pattern]`, 1-based, `n→L` when
  picked; a click toggles the pick, at most 12, letters in order; a
  right-click opens the letter menu, a held letter swaps), first 12, clear,
  the count, ▶ preview (`po32.preview`; the core stops the transport and
  ends the preview on a bank, focus or pick change), the focused pattern's
  8×16 grid (`#po32-grid`, the preview step outlined). 4 import:
  `po32.import` (channels 1–8 and all 12 patterns, one undo step); the
  display confirms and the page stays open with every stage ticked.
- Picks echo at once (`po32-logic.js` `pick` follows the core's rules), then
  the core's values replace them. Verb failures that need reading (send,
  input, decode, preview, import) go to a red alert; others to the display.
  Closing the page cancels a send, ends a preview and closes the input, as
  the tkinter dialogs do.

## Tests

All from pytest, offscreen, no Node and no display needed:

```bash
python -m pytest tests/web -q                      # the web layer (~2 min)
node --test pythonic/web/static/test/*.test.js     # optional local shortcut for pure specs
```

`tests/web/conftest.py` sets `QT_QPA_PLATFORM=offscreen` (unless set),
renders Chromium in software (`prepare_rendering`) and registers the scheme
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
  `settle()` (waits until the page has seen the window's size and the
  window matches the stage's height: the edit rack resizes it through the
  bridge after the page changed; `rect` calls it, so input lands on fresh
  boxes), `rect(sel)`, `center(sel)`, `click(sel)`, `wheel(sel, steps)`,
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
  selected pattern; `trigger` is recorded (`('trigger', channel0, velocity)`);
  a verb handler returning `FakeCore.DEFERRED` leaves its action open
  (`core.running` is its id) for `progress(id, fraction)` and `finish(id,
  result | error=)`.
- **Real-core end-to-end tests** (`test_real_core.py`): play and the playhead,
  a tempo edit through a preset file round trip, the frame budget, a tune
  drag and its undo, select and mute, the CTRL preference, a paint stroke
  and its one-step undo, follow over a 32-step pattern, lane copy / paste,
  a rack fader on the selected channel and its undo, click to assign an LFO
  destination, bands moving while an LFO runs, the rack-open preference, a
  preset saved from the PRESET menu and loaded back from the folder list
  (after choosing the folder), export to MIDI from the pattern menu (and the
  replace question on the second export). `test_setup_real_core.py`: a base
  note step, an added CC mapping and a buffer change with restart audio.
- **AI page**: `test_ai_page.py` (fake core: layout under the strips, every
  state, the leave question on every way out) and `test_ai_real_core.py`
  (the real core with `tests/fake_ai_worker.py`: generate, try on the face,
  a knob edit, keep tried, one undo step; leave and revert).
- **PO-32 page**: `test_po32_page.py` (fake core: both tabs, every stage,
  opening, closing and the tab memory) and `test_po32_real_core.py` (a card
  transfer made with the codec pushed into the fake input while recording,
  decoded, picked and imported: the lanes change, one undo step; the meter;
  a WAV decode and bank switch; a transfer sent through the pulled stream and
  its WAV saved).
- **Parity guard** (`test_parity_guard.py`): every address the core's
  `describe()` lists (with each registered page opened in turn) is bound by a `data-address` control or listed in
  `tests/web/parity_absent.json`: `{"absent": {"<fnmatch glob>": "<reason>"}}`
  (the guard selects each channel in turn, since the rack binds the selected one's).
  The guard also opens each setup sheet tab. It fails on a bound address
  still listed and on globs that match nothing. A slice that binds controls deletes their entries; a core change
  that adds addresses lists them there (or binds them).

Each web slice adds page tests for the behaviours it builds and a manual
parity list in `docs/parity/`.
