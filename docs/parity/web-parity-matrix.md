# Web parity matrix: the tkinter GUI in the web panel

Every control, menu and dialog of the tkinter GUI (inventory on branch
`research/tkinter-gui-inventory`, `docs/research/tkinter-gui-inventory.md`,
part 1) and where the web panel has it, or the slice that builds it, or why
it is left out. Slices: W2 face, W3 step row and patterns, W4 edit rack, W5
menus and sheets, W6 setup sheet, W7 PO-32 page, W8 AI page. "Core" means
the behaviour lives in the app core and both GUIs share it.

Status: **done** (in the web panel), **W6 / W7 / W8** (planned there),
**absent** (a decided absence, with the reason).

## 1.0 Main window

| tkinter | Web | Status |
|---|---|---|
| Window title, 960×600, min 800×480, resizable | `PanelWindow`: 1600×1000 stage scaled letterboxed, min 1280×800 (1280×560 with the rack closed) (#7) | done |
| Scroll container, mouse-wheel page scroll | none: the whole panel always fits (scaled stage) | absent (#7) |
| Section order | face, step row, edit rack drawer (#7, #12) | done |

## 1.1 Toolbar

| tkinter | Web | Status |
|---|---|---|
| `program:` combobox 1-16 | left column programs 1-16 grid (`program.select`, `program.current`, `program.occupied`) | done W2 |
| Morph learn A / B | right column learn a / learn b (`morph.learn`); right-click: capture (`morph.capture`) | done W2 |
| `sound morph` slider | right column sound morph knob (`morph.position`), dims while A = B (`morph.differs`) | done W2 |
| Undo ↶ / Redo ↷ | left column undo / redo (`undo`, `redo`, `undo.can_*`) | done W2 |
| `master` knob | top row master (`global.master`) | done W2 |
| MIDI activity LED, click opens MIDI settings | right column MIDI LED (`midi.connected`, `poll().midi.activity`); click opens setup on the midi tab (`panel.openPage('setup', {tab: 'midi'})`) | done W2 / W5 / W6 |

## 1.2 Preset and channel strip

| tkinter | Web | Status |
|---|---|---|
| `PYTHONIC` label | wordmark, left column | done W2 |
| `PO-32` button | right column po-32 → `panel.openPage('po32')`: the PO-32 page on its last tab (again: closes it) | done W5 / W7 |
| Preset ◀ / ▶ | right column ◀ ▶: previous / next file of `preset.files`, no wrap (`preset.load`) | done W5 |
| Preset combobox (folder list, current preset) | PRESET menu: the folder's presets as an in-panel list (current lit), a click loads; the display's second line shows `preset.name` | done W5 |
| Preset ▼ | PRESET button: the preset menu (1.6.1) | done W5 |
| Patch name display | white tab on each strip, rack header (`chN.name`) | done W2 / W4 |
| Channel buttons 1-8: select; click on the selected one hits it (64, Ctrl 127); LED blue / red / green flash | strip channel buttons: select (`global.channel`), MUTE latch toggles mutes, a click on the selected one hits it (bridge `trigger`, 64, Ctrl+click 127); lit selected, red muted, a flash when a MIDI note or a click hits the channel | done W2 / W5 |
| Mute buttons ×8 | MUTE latch + channel buttons (`chN.mute`) (#12) | done W2 |
| Drum type labels ×8 | the channel buttons show the guessed drum type (#12) | done W2 |

## 1.3 Drum patch area (selected channel)

| tkinter | Web | Status |
|---|---|---|
| 1.3.1 Mixing: osc / noise mix, eq freq, edit all, distort, eq gain, level, pan, choke, output | edit rack mix section (crossfader, eq, distort, pan, out A / B, choke); level on the strip fader; edit all in the rack header (`global.edit_all`) | done W4 |
| 1.3.2 Oscillator: waveform, freq, pitch, pitch mod, amount, rate, attack, decay | rack oscillator section; pitch (tune) and decay on the strips; attack in the envelope bank | done W2 / W4 |
| 1.3.3 Noise: filter mode, freq, q, stereo, envelope, attack, decay | rack noise section and envelope bank | done W4 |
| 1.3.4 FX: vintage, reverb time / mix / width, delay time, feedback, mix, P.P | rack FX section | done W4 |
| 1.3.5 Velocity: osc, noise, mod | rack velocity fader bank | done W4 |
| 1.3.6 LFO 1 / LFO 2: on, wave, rate, depth, sync, re, uni, destination | rack modulation rows (plus phase); destination by click to assign (#11) | done W4 |
| 1.3.7 Pump: on, amount, attack, release, curve, destination | rack pump row (plus sync); click to assign | done W4 |
| 1.3.8 Modulation indicators | modulation bands on rack and strip controls, every channel (#11) | done W2 / W4 |

## 1.4 Pattern section

| tkinter | Web | Status |
|---|---|---|
| ⊞ matrix toggle, matrix editor | ⊞ matrix: the drawer page with 8 channels × 16 steps, every step property | done W3 |
| Pattern buttons A-L: select / queue, colours | left column pattern grid (`pattern.select`, playing / queued / empty / chained states) | done W3 |
| Pattern right-click → menu | right-click a pattern: its pattern menu | done W3 |
| ◀◀ / ▶▶ chain | left column ◀◀ ▶▶ (`pattern.chain_prev` / `chain_next`) | done W3 |
| `Menu` | MENU: the pattern menu (1.6.2) | done W3 |
| `Copy` / `Paste` (lane) | copy / paste (`pattern.copy_lane` / `paste_lane`; velocities included, substeps not, as in tkinter) | done W3 |
| `Prob` mode | step mode prob (#8) | done W3 |
| Channel label `chN` | the selected strip and the display | done W3 |
| Lane editor (trig, acc, fill, sub, len) | pads with step modes trig / accent / velo / fill / prob / sub, last step, pages and follow (#8) | done W3 |

## 1.5 Global bar

| tkinter | Web | Status |
|---|---|---|
| ■ stop / ▶ play | START/STOP, one toggle (#12) | done W2 |
| `BPM` entry | tempo knob, wheel over the red tempo display, double-click the knob to type a value | done W2 |
| Step rate buttons | top row step rate switch | done W2 |
| `swing` slider | right column swing knob | done W2 |
| `fill rate` buttons | top row fill rate list | done W2 |
| Hint text "Keys 1-8" | none (no keyboard input) | absent (map #1: keyboard shortcuts out of scope) |

## 1.6 Popup menus

### 1.6.1 Preset menu (PRESET button, W5)

| tkinter entry | Web entry | Status |
|---|---|---|
| Open Preset... | open preset… (native dialog, `preset.load`) | done W5 |
| Save Preset As... | save preset as… (native dialog; an existing file asks on the alert sheet, then `overwrite=true`); also save “name.json” over the open JSON preset | done W5 |
| Load / Save Drum Patch | load drum patch into CHn… / save drum patch of CHn… (also in the rack header's drum patch ▾) | done W4 / W5 |
| Export Drum to WAV... | export CHn hit as WAV… (`export.drum_wav`) | done W4 / W5 |
| Export All Drums to WAV... | export every drum as WAV… (folder dialog, `export.drum_wavs`, asks before replacing) | done W5 |
| Cut / Copy / Paste Preset | cut / copy / paste preset (paste off while `preset.clipboard` is empty) | done W5 |
| Initialize Preset, Randomize All | initialize preset, randomize all | done W5 |
| Select Preset Folder... | preset folder… (folder dialog, `pref.preset_folder`) | done W5 |
| Refresh Preset List | refresh list, and ↻ beside the folder list (`preset.refresh`) | done W5 |
| Transfer to PO-32... / Import from PO-32... | transfer to PO-32… / import from PO-32… → `openPage('po32', {tab})` | done W5 / W7 |
| AI Drum Generator... | AI drum generator… → `openPage('ai')` | done W5; the page done W8 |
| Audio / MIDI / Synthesis / AI Settings... | setup… → `openPage('setup')` (one sheet with tabs, #22) | done W5 / W6 |
| (recent files, never shown in tkinter) | recent files in the preset menu (`pref.recent_files`) | done W5 (new) |
| (last preset, loaded at start-up) | reload last preset (`preset.load_last`) | done W5 (new) |

### 1.6.2 Pattern menu (MENU or right-click a pattern)

| tkinter entry | Web entry | Status |
|---|---|---|
| Cut / Copy / Paste / Exchange Pattern | cut / copy / paste / exchange (and clear) | done W3 |
| Shift Left / Right, Reverse | shift left / right, reverse | done W3 |
| Randomize, Alter Pattern, Randomize Accents/Fills | randomize, alter, randomize accents + fills | done W3 |
| Randomize Pattern (AI) / Randomize Channel (AI) | randomize (AI) / randomize chN (AI) (`ai.randomize_pattern`, off without `ai.available`) | done W3 |
| Export Pattern to MIDI File... | export to MIDI… (straight to the save dialog, `export.midi`) | done W5 |
| Export Pattern to Audio File... | export to audio… (tail popover, then the save dialog, `export.wav`, progress on the display) | done W5 |
| (new) play next, copy / paste lane, clear all chains | play next (queue), copy / paste lane chN, clear all chains | done W3 |

### 1.6.3 MIDI-learn context menu (right-click a control)

| tkinter entry | Web entry | Status |
|---|---|---|
| Mapped to CCn (disabled) + Remove CC Mapping | CC badge on the control + remove CC n mapping | done W2 |
| Pitch Bend → This Parameter + Remove Pitch Bend Mapping, Assign Pitch Bend | assign / remove pitch bend | done W2 |
| MIDI Learn (CC) / Cancel MIDI Learn | MIDI learn (CC) / cancel MIDI learn | done W2 |
| MIDI Settings... (opens CC mappings) | CC mappings… → `openPage('setup', {tab: 'midi'})`; left out on the sheet's own controls | done W5 / W6 |
| (new) | reset to default | done W2 |

### 1.6.4 Substeps menu

| tkinter | Web | Status |
|---|---|---|
| No substeps, 14 presets, Custom... | sub mode: none, the 14 presets, custom… field (`o` / `-`) | done W3 |

## 1.7 Dialogs

| tkinter dialog | Web | Status |
|---|---|---|
| 1.7.1 Export Audio Options (tail radios + OK) | tail popover on the pattern MENU button: cut / add 2 s / loop +1 pass, save wav… (#13) | done W5 |
| 1.7.2 Audio Settings | setup sheet, audio tab (#22): output device + rescan, buffer, sample rate (the device's rates), synth rate, mono, input device, the running stream; values live, stream settings wait for restart audio (a dot each). Apply Now / OK / Cancel: none (live values, #13) | done W6 |
| 1.7.3 Synthesis Settings | setup sheet, synthesis tab: smoothing knob, applies at once | done W6 |
| 1.7.4 AI Settings | setup sheet, ai tab: pattern and drum patch models (browse… native dialog, clear) and temperatures, the same `pref.ai.*` as the AI page; a note when the ML extras are missing | done W6 |
| 1.7.5 MIDI Settings | setup sheet, midi tab: device (off / auto-detect / ports) + rescan, connection LED, base note with its 8 notes, follow MIDI clock with the synced tempo, the help line | done W6 |
| 1.7.6 MIDI CC Mappings | setup sheet, midi tab, beside the input: unlimited rows (scroll after 10), any CC 0-127, + add, ✕, clear all (asks), live activity, pitch bend target; learning stays on the panel (#22) | done W6 |
| 1.7.7 PO-32 transfer | PO-32 rack page, transfer tab (#20): choose (sounds to 1–8 / 9–16, chain with its empty PO-32 slots, channels from the face mutes) › prepare › send (progress, stop, save WAV) | done W7 |
| 1.7.8 PO-32 import | PO-32 rack page, import tab (#20): listen (input, rescan, monitor and meter, record / stop, import WAV, keep recordings) › bank › pick patterns (12, letters, swap by the right-click letter menu, first 12, clear, preview, grid) › import (one undo step, the page stays open); tkinter's "open folder" of the recordings is the folder shown beside keep recordings | done W7 |
| 1.7.9 AI Drum Generator | AI rack page (#21): models with load…, patch temperature, candidates, seed with reseed, generate all 8 (left); one lane per strip with type (18), gen, ‹ i/n ›, name, try (tried sounds play on the face, the strip tab in italics); keep current / generate new patterns, pattern temperature, bank, ▶ loop, ▶ bank A→L, keep tried (= apply selected; one undo step), replace patterns, revert all (right); the install command with copy and install now | done W8 |
| 1.7.9 per-slot Preview (one-shot at 127), Apply checkboxes, Close (clear) | trying a candidate plays it on the face (and hits it while stopped); try / trying ✓ per lane replaces the Apply checkboxes; leaving asks keep or revert instead of dropping the candidates | done W8 (changed by #21) |
| Custom substeps prompt | the sub menu's custom… field | done W3 |
| 1.7.10 File dialogs: open / save preset, drum patch, preset folder, MIDI / WAV export | native `QFileDialog` from the bridge `fileDialog` slot, opened with `open()` (#14); the core checks overwrites and the page asks | done W4 / W5 |
| 1.7.10 File dialogs: AI checkpoint, PO-32 WAV open / save | the same native dialogs (AI checkpoints in the setup sheet: done W6; on the AI page: done W8; PO-32: done W7) | done W6 / W7 / W8 |
| 1.7.10 Message boxes (errors, questions) | the alert sheet (red errors, green questions, closed only by its buttons) plus the display (#13, #22); errors nobody waits for go to the display, and to the alert sheet unless they come from MIDI or the PO-32 module | done W5 |

## 1.8 Widget conventions

| tkinter | Web | Status |
|---|---|---|
| Knob / slider vertical drag, Shift fine, wheel 1 % / 2 %, log scaling | `px-knob`, `px-fader` (#9) | done W2 |
| Knob Alt+click circular drag | none | absent (#9: vertical drag only) |
| Ctrl+click or double-click: reset | right-click: reset to default; double-click types an exact value (#9) | done W2 (changed by #9) |
| Floating value hint | the value printed under every control and the display (#9) | done W2 |
| Toggle, selectors, waveform icons, circular buttons | `px-toggle`, `px-switch`, `px-list`, START/STOP | done W2 / W4 |
| Lane gestures: click / drag toggles along the lane | trig / accent / fill paint strokes | done W3 |
| Ctrl+click in trig: trigger and accent together | accent step mode | absent (#8: step modes replace modifier clicks) |
| Shift+click: all 8 channels (muted included) | all ch latch | done W3 |
| Click in `len`: pattern length | last step + a pad | done W3 |
| Probability drag | prob step mode | done W3 |

## 1.9 Keyboard

| tkinter | Web | Status |
|---|---|---|
| `1`-`8` hit a channel | a click on the selected channel's button hits it; MIDI notes | absent as keys (map #1: keyboard shortcuts out of scope) |
| `s` save preset as, `l` open preset | PRESET menu | absent as keys (map #1) |
| `Ctrl+Z` / `Ctrl+Y` | undo / redo buttons | absent as keys (map #1) |

## 1.10 MIDI input

Notes, program change 0-11, start / stop / continue, clock sync, mapped CCs
with pickup, pitch bend and MIDI learn are handled by the core for both
GUIs (slice 3). Web cues: the MIDI LED, CC badges, the learn pulse, pickup
ghost markers (W2), and the channel button flash on notes (W5). Device and
settings: setup sheet midi tab (done W6).

## 1.11 Preferences

Every preference is an address (`pref.*`, `midi.*`): the preset folder and
recent files in the PRESET menu (W5); audio, MIDI, smoothing and AI
settings in the setup sheet (W6); AI models and temperatures also on the AI
page (W8); PO-32 recordings on the PO-32 page (W7). The web-only `pref.ui.*`
keep the strip CTRL mode (W2) and the rack's open state (W4).

## 1.12 Entries that failed in tkinter

| tkinter | Web | Status |
|---|---|---|
| Cut / Copy / Paste Preset, Initialize Preset, Randomize All | core verbs, PRESET menu | done W5 |
| MIDI Program Change | core (selects patterns A-L) | done (core) |
| Export all drums / current drum to WAV (unreachable) | PRESET menu and rack drum patch ▾ | done W4 / W5 |
| LFO / pump target Morph | a destination of click to assign (core applies it) | done W4 |
