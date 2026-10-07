# App core interface

The reference for front-ends (tkinter today, the web GUI next) of the UI-free
app core in `pythonic/app/` (decision: [ADR 0001](adr/0001-address-based-app-core.md);
terms: [CONTEXT.md](../CONTEXT.md)). A front-end never touches the synth, the
pattern manager or the preferences file: it reads and writes **addresses**,
starts **verbs**, and draws what **poll** reports.

```python
from pythonic.app import AppCore

core = AppCore()             # preferences=PreferencesManager(), audio and MIDI backends
core.act('preset.load_last') # optional: the last preset, before the stream starts
core.start()                 # freezes the GC heap, opens the stream and the saved MIDI input
...                          # per frame: state = core.poll(version)
core.close()                 # stops MIDI, the action thread and the stream
```

## The five calls

| Call | Returns | Thread rules |
|---|---|---|
| `get(addr)` | the value, in engine units | any thread; reads the live value |
| `set(addr, value, *, edit_all=None, burst=False, record=True)` | nothing | any thread; queued and applied by the audio thread at the next block start (settings, modes and the CC map apply at once on the caller's thread). Raises `KeyError` (unknown address) or `ValueError` (read-only, or not a valid value) at once |
| `describe(addr)` | `{address, kind, minimum, maximum, default, unit, curve, labels, readonly}` | any thread |
| `act(verb, **args)` | an action id, at once | the verb runs on the core's action thread; its result arrives through `poll` |
| `poll(since=0)` | everything newer than version `since` (below) | any thread; front-ends call it on their own frame timer. The core never calls into a UI thread |

Also: `trigger(channel, velocity=127, at=None)` queues a pad hit with its
arrival time (`channel` is **0-based** here, 0..7; `at` defaults to now) and
the hit lands at the matching sample offset; `begin_gesture()` /
`end_gesture()` (undo, below); `wait(action_id, timeout)` blocks until an
action has finished (tests and shutdown only).

### Values and `describe`

- `kind`: `float`, `int`, `bool`, `enum` (the value is one of `labels`; `set`
  also takes the index), `str`, `list`, `map` (`midi.cc_map`) or `json`
  (`pref.ui.*`, any JSON value).
- Numbers are clamped to `minimum`..`maximum` by `set`. Units: `Hz`, `ms`,
  `dB`, `st` (semitones), `BPM`, `%` (already a percentage), `ratio` (0..1 or
  0..2, shown as a percentage), `x`, `frames`.
- `curve` is `linear` or `log`. The core owns knob and controller scaling:
  `registry[addr].normalize(value)` / `denormalize(position)` map a value to
  and from a 0..1 knob position along the curve (a log range starting at 0
  starts its curve at 1e-5 of the maximum).
- `labels`: the names of an enum, or the values a menu offers for a device,
  rate or buffer setting (devices are refreshed by the rescan verbs).
- Channels in addresses are **1..8**, patterns are letters **A..L**, steps
  **1..64**, programs **1..16**.

## Addresses

`ch<N>` is a channel 1..8, `<P>` a pattern A..L. Values are engine units.

| Address | Kind | What |
|---|---|---|
| `ch<N>.osc.wave|freq|pitch|attack|decay|mod_mode|mod_amount|mod_rate` | sound | oscillator |
| `ch<N>.noise.filter|freq|q|stereo|env|attack|decay` | sound | noise |
| `ch<N>.mix.osc_noise|distortion|level|pan|output|choke` | sound | mixing |
| `ch<N>.eq.freq|gain` | sound | EQ |
| `ch<N>.fx.vintage|reverb_decay|reverb_mix|reverb_width|delay_time|delay_feedback|delay_mix|delay_pingpong` | sound | effects |
| `ch<N>.vel.osc|noise|mod` | sound | velocity sensitivity |
| `ch<N>.lfo1.*`, `ch<N>.lfo2.*` (`on|wave|rate|sync|depth|target|retrig|unipolar|phase`) | sound | LFOs |
| `ch<N>.pump.on|amount|attack|release|curve|sync|target` | sound | pump |
| `ch<N>.mute` | bool | mute (saved in the preset, not undone) |
| `ch<N>.name` | str, read-only | drum patch name |
| `global.tempo` (1..300 BPM), `global.swing` (0..1), `global.step_rate` (enum), `global.fill_rate` (2..8), `global.master` (-60..10 dB) | | preset globals |
| `global.channel` (1..8) | int | the selected channel (not undone) |
| `global.edit_all` | bool | Edit all mode |
| `pattern.<P>.ch<N>.step<S>.trig|acc|vel|fill|prob|sub` | step | one step; turning `trig` off clears `acc` and `fill`; past the length it reads its default and ignores sets |
| `pattern.<P>.ch<N>.trig|acc|vel|fill|prob|sub` | list | a whole lane (one value per step of the length) |
| `pattern.<P>.length` (1..64), `pattern.<P>.chained` (to the next; read-only on L), `pattern.<P>.empty` (read-only) | | pattern |
| `pattern.selected` | enum A..L | the selected pattern (plain selection) |
| `program.current` (1..16), `program.occupied` (16 bools) | read-only | program bank |
| `morph.position` (0..1) | float | morph position |
| `morph.learning` (`off`, `a`, `b`), `morph.differs` | read-only | morph learn, endpoints differ |
| `undo.can_undo`, `undo.can_redo` | read-only bool | journal state |
| `preset.name`, `preset.path` (None before any load or save) | read-only | the current preset |
| `preset.files` | read-only list | `.mtpreset` / `.json` file names in `pref.preset_folder`, sorted |
| `preset.clipboard` | read-only bool | the preset clipboard is full |
| `midi.device`, `midi.connected`, `midi.synced_tempo`, `midi.learning` | read-only | MIDI input state; `describe('midi.device')['labels']` lists the ports of the last scan |
| `midi.base_note` (0..120), `midi.clock_sync`, `midi.cc_map` ({CC: target}), `midi.pitchbend_target` | settings | saved at once; a target is an address or `selected.<sound suffix>` (the selected channel) |
| `audio.running|device|device_is_default|sample_rate|synth_rate|block_size|buffer_ms|mono` | read-only | the running stream |
| `audio.output_devices`, `audio.input_devices`, `audio.default_input` | read-only | device names |
| `pref.*` | settings | below |

Step and lane addresses are resolved on first use: `registry.names()` lists
the registered addresses only (not `pattern.<P>.ch<N>...` nor `pref.ui.*`).

### Preferences (`pref.*`)

A `set` saves the preferences file at once (the keys tkinter has always used,
so both GUIs and older versions share the file) and is not undoable.

| Address | Applies | Saved key |
|---|---|---|
| `pref.audio.device` (output name, None = system default) | on `audio.apply` | `audio_output_device` |
| `pref.audio.buffer_ms` (labels: 2 .. 100 ms, default 23.8) | on `audio.apply` | `audio_buffer_ms` |
| `pref.audio.sample_rate` (labels: 96000 .. 8000 Hz) | on `audio.apply` | `audio_sample_rate` |
| `pref.audio.synth_rate` (0 = same as output; labels 0, 22050, 11025, 8000) | on `audio.apply` | `synth_sample_rate` (stored as the effective rate) |
| `pref.audio.pending` (read-only) | | the four above whose saved value waits for `audio.apply` (the "restart audio" dots) |
| `pref.audio.mono` | at once (`audio.mono` follows) | `audio_mono` |
| `pref.audio.input_device` (PO-32 recording input) | when the input opens | `audio_input_device` |
| `pref.smoothing_ms` (5..100 ms) | at once, every channel | `param_smoothing_ms` |
| `pref.ai.pattern_model`, `pref.ai.patch_model` (paths, None = bundled) | next model load | `drum_generator_pattern_model_path`, `drum_generator_model_path` |
| `pref.ai.pattern_temperature`, `pref.ai.patch_temperature` (0.1..3) | next generation | `drum_generator_pattern_temperature`, `drum_generator_patch_temperature` |
| `pref.preset_folder` (must exist) | at once (`preset.files` follows) | `preset_folder` |
| `pref.recent_files` (read-only) | | `recent_files` |
| `pref.ui.<name>` (`[a-z0-9_]+`, any JSON value, None until set) | front-end state (strip CTRL mode, rack open, ...) | `ui_<name>` |

MIDI settings are `midi.*` (above); the MIDI device and MIDI on/off are saved
by `midi.open` / `midi.close`. `last_preset` is written by `preset.load`.
`po32_debug_save_recordings` stays with the PO-32 dialog; `window_width`,
`window_height`, `master_volume_db` and `max_recent_files` are not used.

## Verbs

Arguments are keywords. `pattern` is a letter or 0..11 (default: the selected
pattern), `channel` 1..8.

| Verb | Arguments | Result |
|---|---|---|
| `transport.play` | pattern | starts it from step 1 |
| `transport.stop`, `transport.toggle`, `transport.continue` | | |
| `pattern.select` | pattern | selects it (queued while playing) |
| `pattern.queue` | pattern or None | |
| `pattern.cut|copy|paste|exchange|clear|shift_left|shift_right|reverse|randomize|alter|randomize_accents_fills` | pattern | pattern ops, one undo step each |
| `pattern.copy_lane`, `pattern.paste_lane` | pattern, channel | lane clipboard |
| `pattern.chain_prev`, `pattern.chain_next`, `pattern.chain_clear` | pattern | chains |
| `undo`, `redo` | | `{'done', 'label'}` |
| `program.select` | program 1..16 | `{'program', 'recalled'}` |
| `morph.learn` | endpoint `'a'`, `'b'` or None (stop) | `{'learning'}` |
| `morph.capture` | endpoint | |
| `preset.load` | path | `{'path', 'name', 'format'}` (`mtpreset` or `json`) |
| `preset.load_last` | | as `preset.load`, plus `loaded`; `{'loaded': False}` without a last preset |
| `preset.save` | path, overwrite=False | `{'saved', 'exists', 'path'}` |
| `drum_patch.load` | path, channel (default selected) | `{'channel', 'name', 'path'}` |
| `drum_patch.save` | path, channel, overwrite=False | `{'saved', 'exists', 'path', 'channel'}` |
| `preset.copy`, `preset.cut`, `preset.paste`, `preset.initialize`, `preset.randomize_all` | | preset clipboard, init, randomize (one undo step each) |
| `preset.refresh` | | `{'files'}`; rescans the preset folder |
| `audio.start`, `audio.stop` | | stream status |
| `audio.apply` | optional device, sample_rate, synth_rate, buffer_ms, mono (saved first) | restarts the stream with the saved `pref.audio.*`; stream status |
| `audio.rescan` | | `{'output_devices', 'input_devices'}`; refreshes the device labels |
| `audio.rates` | device (None = default) | `{'device', 'rates'}`: the menu rates the device accepts |
| `midi.open` | device (None = first port), fallback | `{'device'}`; saves the choice |
| `midi.close`, `midi.rescan` | | `{'device': None}`, `{'devices'}` |
| `midi.learn` | target | finishes when a CC arrives (`{'cc', 'target'}`) or with status `cancelled` |
| `midi.learn_cancel` | | `{'cancelled'}` |

**Files.** Paths are absolute, or names relative to `pref.preset_folder`. A
preset load replaces the whole preset (sounds, globals, programs, morph,
patterns; mutes when the file has them) and makes it the last preset and the
first recent file. Saving refuses to replace an existing file unless
`overwrite=True`: the result is `{'saved': False, 'exists': True, 'path'}`, and
the front-end asks, then sends the verb again with `overwrite=True`. The core
adds `.json` / `.mtdrum` to a path without an extension before that check
(native dialogs add the default suffix after their own check). File dialogs
live in the front-ends and hand the chosen path to these verbs.

## Poll

```python
state = core.poll(version)
version = state['version']
```

| Key | Content |
|---|---|
| `version` | the newest version; pass it back next frame |
| `changes` | `{address: value}` changed since `since` (a queued set appears once the audio thread has applied it; a bulk change reports every value it may have changed) |
| `events` | action events newer than `since`: `{'id', 'verb', 'status': 'done', 'result', 'version'}` or `{'id', 'verb', 'status': 'error', 'error': message, 'version'}`; `midi.learn` may end `cancelled`. Errors not tied to an action (audio callback, stalled stream) have `id` None and a `source` |
| `transport` | `playing`, `position` (0-based step of the playing pattern), `playing_pattern`, `selected_pattern`, `queued_pattern` (0..11 or None), `chain` (indexes of the chain being played) |
| `modulation` | `channel` (0-based selected channel), `offsets` (`{mod target: offset}`) for the knobs' modulation arcs |
| `audio` | `running`, `device`, `default_device`, `sample_rate`, `synth_rate`, `block_size`, `latency_ms`, `mono`, `callbacks`, `underruns`, `dropped` |
| `midi` | `activity` (message counter), `notes` (per-channel note counters), `pickup` (`{address: {cc, physical, linked, count}}` for ghost markers) |

A front-end keeps no copy of engine state it cannot rebuild from `get` and
`changes`: after any verb, the values it changed arrive in `changes`.

## Undo

- Every `set` of an undoable address is one step. Bracket a drag or a paint
  stroke with `begin_gesture()` / `end_gesture()` to make it one step; pass
  `burst=True` for a wheel turn (the changes of one address are one step until
  400 ms pass without one); `record=False` keeps a set out of the journal.
- Verbs that rewrite many values (pattern ops, program select, morph learn
  and capture, preset load / paste / initialize / randomize all, drum patch
  load) are one snapshot step each. Depth 50.
- Not undone: transport, selection (`global.channel`, `pattern.selected`),
  mutes, Edit all, settings (`pref.*`, `midi.*`).
- Code that still writes the engine directly (the AI generator and PO-32
  dialogs, until their slices) wraps its writes in
  `with core.bulk_change(label, parts=...):` to make them one step reported by
  poll.

## Edit all

`set('chN.<sound>', v)` also sets the same parameter of every **unmuted**
channel when `global.edit_all` is on (or with `edit_all=True`); restoring
code passes `edit_all=False`. Pattern steps are never fanned out.
