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
action has finished (tests and shutdown only); `call_soon(fn)` runs fn on the
action thread (modules finishing deferred actions).

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
| `program.current` (1..16), `program.occupied` (16 bools), `program.names` (16 strings) | read-only | program bank; a name is the words the program's channel names start with (`808`), `""` when empty or nothing in common, reported with any channel name |
| `morph.position` (0..1) | float | morph position |
| `morph.learning` (`off`, `a`, `b`), `morph.differs` | read-only | morph learn, endpoints differ |
| `undo.can_undo`, `undo.can_redo` | read-only bool | journal state |
| `preset.name`, `preset.path` (None before any load or save) | read-only | the current preset |
| `preset.files` | read-only list | `.mtpreset` / `.json` file names in `pref.preset_folder`, sorted |
| `preset.clipboard` | read-only bool | the preset clipboard is full |
| `preset.factory` | read-only bool | the current preset is a factory preset (read-only file) |
| `factory.presets` | read-only list | the factory presets' file names, in machine order (`505 Beats.json` .. `LM2 Beats.json`) |
| `factory.patches` | read-only list | the factory drum patches: the 48 channel sounds of the six factory kits, in machine order (`505 BD` .. `LM2 OH`); each name starts with its machine |
| `midi.device`, `midi.connected`, `midi.synced_tempo`, `midi.learning` | read-only | MIDI input state; `describe('midi.device')['labels']` lists the ports of the last scan |
| `midi.base_note` (0..120), `midi.clock_sync`, `midi.cc_map` ({CC: target}), `midi.pitchbend_target` | settings | saved at once; a target is an address or `selected.<sound suffix>` (the selected channel) |
| `audio.running|device|device_is_default|sample_rate|synth_rate|block_size|buffer_ms|mono` | read-only | the running stream |
| `audio.output_devices`, `audio.input_devices`, `audio.default_input` | read-only | device names |
| `pref.*` | settings | below |
| `ai.*` | | the AI generators, below |
| `po32.*` | | the PO-32 transfer and import, below |

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
| `pref.po32.save_recordings` | next PO-32 recording, saved as a WAV file in `po32.recordings_folder` | `po32_debug_save_recordings` |
| `pref.smoothing_ms` (5..100 ms) | at once, every channel | `param_smoothing_ms` |
| `pref.web.gpu` (bool; default off on Windows, on elsewhere) | next start of the web interface (`web_gpu(manager)` reads it before Qt starts) | `web_gpu` (absent = the platform default) |
| `pref.ai.pattern_model`, `pref.ai.patch_model` (paths, None = bundled) | next model load | `drum_generator_pattern_model_path`, `drum_generator_model_path` |
| `pref.ai.pattern_temperature`, `pref.ai.patch_temperature` (0.1..3) | next generation | `drum_generator_pattern_temperature`, `drum_generator_patch_temperature` |
| `pref.preset_folder` (must exist) | at once (`preset.files` follows) | `preset_folder` |
| `pref.recent_files` (read-only) | | `recent_files` |
| `pref.ui.<name>` (`[a-z0-9_]+`, any JSON value, None until set) | front-end state (strip CTRL mode, rack open, ...) | `ui_<name>` |

MIDI settings are `midi.*` (above); the MIDI device and MIDI on/off are saved
by `midi.open` / `midi.close`. `last_preset` is written by `preset.load`.
`window_width`, `window_height`, `master_volume_db` and `max_recent_files` are not used.

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
| `program.restore_factory` | | `{'programs': [1..6]}`: the factory kits back into programs 1-6 (the current one plays at once when among them); one undo step |
| `morph.learn` | endpoint `'a'`, `'b'` or None (stop) | `{'learning'}` |
| `morph.capture` | endpoint | |
| `preset.load` | path, or factory (a `factory.presets` name, with or without `.json`) | `{'path', 'name', 'format'}` (`mtpreset` or `json`) |
| `preset.load_last` | | as `preset.load`, plus `loaded`; `{'loaded': False}` without a last preset |
| `preset.save` | path, overwrite=False | `{'saved', 'exists', 'path'}` |
| `drum_patch.load` | path, or factory (a `factory.patches` name: the whole channel sound), channel (default selected) | `{'channel', 'name', 'path'}` (path None for a factory drum patch); one undo step |
| `drum_patch.save` | path, channel, overwrite=False | `{'saved', 'exists', 'path', 'channel'}` |
| `export.midi` | path, pattern, overwrite=False | `{'saved', 'exists', 'path', 'pattern'}`: the pattern's MIDI file |
| `export.wav` | path, pattern, tail=`'cut'` (`'cut'`, `'append'` +2 s, `'loop'` +1 pass), overwrite=False | `{'saved', 'exists', 'path', 'pattern', 'tail', 'frames', 'sample_rate', 'channels'}`; progress events while it renders |
| `export.drum_wav` | path, channel (default selected), overwrite=False | `{'saved', 'exists', 'path', 'channel', 'frames', 'sample_rate', 'channels'}`: 2 s of one hit at 127; progress events |
| `export.drum_wavs` | folder, overwrite=False | `{'saved', 'exists', 'folder', 'paths'}`: `01_<name>.wav` .. `08_<name>.wav`; refused with `paths` = the files that exist; progress events |
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
| `ai.*` | | the AI generators, below |
| `po32.*` | | the PO-32 transfer and import, below |

**Files.** Paths are absolute, or names relative to `pref.preset_folder`. A
preset load replaces the whole preset (sounds, globals, programs, morph,
patterns; mutes when the file has them) and makes it the last preset and the
first recent file. Saving refuses to replace an existing file unless
`overwrite=True`: the result is `{'saved': False, 'exists': True, 'path'}`, and
the front-end asks, then sends the verb again with `overwrite=True`. The core
adds `.json` / `.mtdrum` / `.mid` / `.wav` to a path without an extension
before that check (native dialogs add the default suffix after their own
check). File dialogs live in the front-ends and hand the chosen path to these
verbs. The factory presets ship inside the package (`pythonic/factory/presets`,
built locally, outside the repository); `preset.save` refuses a path in that
folder.

**Export.** The MIDI file has one note per triggered step on channel 10
(GM drum notes 36, 38, 42, 46, 45, 41, 39, 37 for channels 1..8) at the step's
velocity (127 when accented), swung like the sequencer, ending where the
pattern loops. WAV files are 16-bit at the synth rate, mono when the output
is mono; a pattern plays alone (no chain), once plus its tail. The state is
taken when the verb starts; renders then run one at a time on the core's
export thread with an offline synth of their own, so the live stream and the
action thread carry on, and poll reports `progress` events until the done or
error event.

## Poll

```python
state = core.poll(version)
version = state['version']
```

| Key | Content |
|---|---|
| `version` | the newest version; pass it back next frame |
| `changes` | `{address: value}` changed since `since` (a queued set appears once the audio thread has applied it; a bulk change reports every value it may have changed) |
| `events` | action events newer than `since`: `{'id', 'verb', 'status': 'done', 'result', 'version'}` or `{'id', 'verb', 'status': 'error', 'error': message, 'version'}`; `midi.learn` and `po32.send` may end `cancelled`; WAV exports and `po32.send` first post `{'id', 'verb', 'status': 'progress', 'progress': 0..1, 'version'}` events (not an end: `wait` skips them). Errors not tied to an action (audio callback, stalled stream) have `id` None and a `source` |
| `transport` | `playing`, `position` (0-based step of the playing pattern), `playing_pattern`, `selected_pattern`, `queued_pattern` (0..11 or None), `chain` (indexes of the chain being played) |
| `modulation` | `channel` (0-based selected channel), `offsets` (`{mod target: offset}` of the selected channel, in the target address's units), `channels` (the offsets of every channel, 8 dicts) for the knobs' modulation arcs (a channel keeps its last block's offsets while it is silent) |
| `audio` | `running`, `device`, `default_device`, `sample_rate`, `synth_rate`, `block_size`, `latency_ms`, `mono`, `callbacks`, `underruns`, `dropped` |
| `midi` | `activity` (message counter), `notes` (per-channel note counters), `pickup` (`{address: {cc, physical, linked, count}}` for ghost markers) |
| `po32` | `level` (input peak 0..1 of the last block), `recorded_seconds`, `progress` (send 0..1), `preview_step` (0..15, -1 without a preview) |

A front-end keeps no copy of engine state it cannot rebuild from `get` and
`changes`: after any verb, the values it changed arrive in `changes`.

## AI generators (`ai.*`)

The drum patch and pattern models run in a **worker subprocess**
(`pythonic/app/ai_worker.py`, JSON lines over its stdin and stdout), so the
GUI process never imports torch. It starts on first use, a request after a
failure starts a new one, and `close()` stops it. `AppCore(ai_worker=argv)`
replaces it (tests use `tests/fake_ai_worker.py`); `ai_worker=None` turns the
AI off. Verbs that need the worker return at once on the action thread and
finish when its reply arrives (their event comes later through poll).

**Trying candidates (the preview overlay).** Each channel has a *lane*: a
drum type, its candidates and the current one. Trying a candidate puts it on
the live channel, so the running pattern plays it; the channel's sound before
the first try is kept and nothing is journaled. `ai.keep` commits the tried
sounds as **one undo step**; `ai.revert` puts the old sounds back.

- Generating a lane tries its candidate 1 at once.
- **Knob edits on a trying channel go into the tried sound:** a `set` of a
  `ch<N>.*` sound address of a trying channel changes the live channel and is
  not an undo step of its own; `ai.keep` commits it with the candidate,
  `ai.revert` / `ai.untry` drop it. Pattern sets during a swapped AI pattern
  preview work the same way (dropped when the preview ends).
- Anything else that replaces the sounds (preset load / paste / initialize /
  randomize all, drum patch load, program select, the undo or redo of a step
  touching a trying channel) reverts the trial first; anything that replaces
  patterns (pattern ops, AI randomize, preset load, their undo) ends the
  pattern preview first. Front-ends ask "keep or revert" before leaving the
  AI page (or loading a preset); when they do not, the core reverts.
- Limits: the morph position (endpoints differing) and LFO / pump morph
  modulation set every channel's sound, tried lanes included; Edit all from
  another channel onto a trying channel journals the tried values.

| Address | Kind | What |
|---|---|---|
| `ai.available` | bool | the ML extras (torch) are installed, found without importing them |
| `ai.install_command` | str | the command that installs them (`pip install` of the `[ml]` extra's packages with this Python) |
| `ai.installing` | bool | `ai.install` is running |
| `ai.state` | enum | `unavailable`, `idle`, `installing`, `loading`, `generating` |
| `ai.models` | json | `{'patch': m, 'pattern': m}`, `m = {path, bundled, status, error, sampling}`; `path` is `pref.ai.<kind>_model` if the file exists, else the bundled checkpoint, else None; `status` `missing`, `unloaded`, `loading`, `loaded`, `error` |
| `ai.ch<N>.type` | enum, settable | the lane's drum type, one of 18; until set, the channel's drum type, else the slot default (BD, SD, CH, OH, TOM, TOM, CLAP, CY) |
| `ai.ch<N>.candidates`, `ai.ch<N>.candidate` | int | n and i of `‹ i/n ›` (i is 1-based, 0 without candidates) |
| `ai.ch<N>.name`, `ai.ch<N>.trying`, `ai.ch<N>.generating`, `ai.ch<N>.error` | | the current candidate's name, the lane is trying it, generating…, the last error |
| `ai.tried` | list | channels trying a candidate (for keep tried and the leave prompt) |
| `ai.bank` | enum | AI patterns: `none`, `generating`, `ready` |
| `ai.preview` | enum | pattern preview: `off`, `loop`, `bank` |

| Verb | Arguments | Result |
|---|---|---|
| `ai.load_model` | kind (`patch` / `pattern`), path=None (the resolved one) | `{'kind', 'path', 'sampling'}`; an explicit path is saved to `pref.ai.<kind>_model` once loaded |
| `ai.generate` | channel=None (all 8), type=None, temperature=None (`pref.ai.patch_temperature`), candidates=8 (1..32), seed=None | `{'channels', 'failed': [{'channel', 'error'}]}` once every lane has its candidates; an error when all failed |
| `ai.try` | channel, candidate=None (the current), step=0 (±1 for the arrows, wraps) | `{'channel', 'candidate', 'name'}` |
| `ai.untry` | channel | `{'channel', 'reverted'}`: back to the old sound, the candidates stay |
| `ai.keep` | channels=None (every trying lane; listed lanes not trying are tried first) | `{'kept'}`; one undo step |
| `ai.revert` | | `{'reverted', 'preview'}`: old sounds back, pattern preview stopped |
| `ai.clear` | | revert, then forget candidates, lane types and the bank |
| `ai.generate_patterns` | temperature=None (`pref.ai.pattern_temperature`), seed=None | `{'patterns': 12}`: a bank A-L for the drum patches on the face (no swing) |
| `ai.clear_patterns` | | drops the bank |
| `ai.pattern_try` | mode `'loop'` / `'bank'` / None (stop), bank=True | loop: the AI version of the playing (else selected) pattern; bank: all 12 chained from A; without a bank, the preset's patterns. Starts the transport; stopping it (any way) ends the preview and puts the preset's patterns back |
| `ai.replace_patterns` | channels=None (as `ai.keep`) | `{'kept', 'patterns'}`: kept sounds plus all 12 bank patterns, one undo step |
| `ai.randomize_pattern` | pattern=None, channel=None (the whole pattern) | `{'pattern', 'channel'}`: one AI pattern (tempo, swing, fill and step rate, `pref.ai.pattern_temperature`) over the pattern or one lane, one undo step |
| `ai.install` | | `{'installed', 'output'}` (the last lines); runs in the background, the worker uses the new packages without a restart |

## PO-32 transfer and import (`po32.*`)

**Audio.** Every stream is the core's, opened on its audio backend (a fake
backend in tests). Sending and previewing play a buffer rendered off the
audio thread at the output rate through the **live output stream**: the
audio engine's `Player` is installed at block start, and after the synth has
rendered a block the callback copies the next slice of the buffer into it.
A transfer **replaces** the block while it plays (pad hits or a running
pattern never reach the PO-32); a preview is **mixed** in. Both need the
stream to run (`po32.send` errors otherwise: save the WAV instead), and both
end when the stream stops. Listening and recording open a mono 44.1 kHz
**input stream**; its callback writes the block's peak level and copies the
samples into a buffer allocated when the recording starts (120 s; a full
buffer stops the recording and decodes it). The decode runs on the action
thread.

**Transfer.** The PO-32 receives the 8 sounds (to its sounds 1-8 or 9-16)
with a default morph patch each, and as many **empty** pattern slots as the
chosen chain has patterns, from PO-32 pattern 1 (`slots` in the result), and
the default state block. Channels not chosen are sent as a silent patch.

| Verb | Arguments | Result |
|---|---|---|
| `po32.prepare` | bank=0 (0: PO-32 sounds 1-8, 1: 9-16), chain=None (a `po32.chain_options` label, a letter or a list of letters / indexes; default the first option), channels=None (1..8 to send; default the unmuted channels) | `{'seconds', 'bank', 'chain', 'slots', 'channels'}`: renders the signal of the current sounds |
| `po32.send` | as `po32.prepare` | renders again and plays it; `progress` events, then `{'sent': True, 'seconds'}`, or status `cancelled` (`po32.cancel`), or an error (the stream stopped) |
| `po32.cancel` | | `{'cancelled'}` |
| `po32.save_wav` | path, overwrite=False, and the `po32.prepare` arguments | `{'saved', 'exists', 'path', 'seconds'}`: mono 16-bit 44.1 kHz |
| `po32.listen` | on=True, device=None (an input device name; default `pref.audio.input_device`, else the system default) | `{'listening', 'device'}`; `on=False` closes the input (and drops a recording) |
| `po32.record` | device=None | `{'recording': True, 'device'}`; drops the previous decode |
| `po32.stop` | | ends the recording, closes the input, decodes: as `po32.decode` (`{'decoded': False}` when nothing was recorded) |
| `po32.decode` | path (a WAV file) | `{'source', 'drums', 'patterns', 'card', 'banks'}`; picks the first 12 non-empty patterns (letters A.. in order), focuses the first, selects the first bank with sounds |
| `po32.select_bank` | bank 0 / 1 (only a decoded one) | `{'bank'}` |
| `po32.focus` | pattern (1..16, the decoded patterns) | `{'focus'}` |
| `po32.pick` | pattern, picked=None (toggle), letter=None | `{'pattern', 'picked', 'letter'}`: at most 12 picks; a new pick takes the first free letter; a letter another pick holds swaps the two; focuses the pattern |
| `po32.pick_first`, `po32.pick_clear` | | `{'picks'}`: the first 12 (non-empty first), none |
| `po32.preview` | on=True | `{'previewing', 'pattern'}`: stops the panel transport and loops the focused pattern with the bank's sounds at the panel tempo (velocity 100). Starting the transport, a bank or focus change, a pick toggled, or `on=False` ends it; the panel stays stopped |
| `po32.import` | | `{'drums', 'patterns': [{'pattern', 'letter'}]}`: the bank's sounds on channels 1-8 (names kept), all 12 patterns (picked ones on their letters: steps 1-16, no accent or fill, probability 100, velocity 64, a pattern shorter than 16 steps grows to 16; the others emptied, lengths kept), morph A = the sounds, B = the PO-32's morph patches, morph at A. One undo step |

| Address | Kind | What |
|---|---|---|
| `po32.chain_options` | list | the chain groups of the patterns (`'A - B'`, `'C'`, ...) |
| `po32.transfer` | enum | `none`, `ready`, `sending`, `sent`, `stopped`, `error` |
| `po32.transfer_seconds` | float | length of the last rendered signal |
| `po32.progress` | float | send progress 0..1 (also in `poll()['po32']`) |
| `po32.listening`, `po32.recording`, `po32.input` | | the input is open, recording, its device |
| `po32.level`, `po32.recorded_seconds` | float | input peak 0..1, recording length (also in `poll()['po32']`) |
| `po32.decode` | enum | `none`, `decoding`, `decoded`, `error` (kept until the next record or decode) |
| `po32.decoded` | json | None or `{'source', 'drums', 'patterns', 'card', 'banks'}` |
| `po32.banks` | list | two bools: bank 0 / 1 has decoded sounds |
| `po32.bank` | int | the selected bank |
| `po32.sounds` | list | 8 summaries of the bank's decoded sounds (None where none) |
| `po32.patterns` | list | `{'number', 'empty', 'summary'}` per decoded pattern |
| `po32.picks` | list | `{'pattern', 'letter'}` in pattern order |
| `po32.focus` | int | the focused pattern (0: none) |
| `po32.grid` | list | 8 lists of 16 bools: the focused pattern's triggers in the bank (a PO-32 step holds drums 1-8 only, so bank 1 grids are empty) |
| `po32.previewing`, `po32.preview_step` | | a preview plays, its step (also in `poll()['po32']`) |
| `po32.imported` | bool | an import was done since the decode |
| `po32.error` | str | the last PO-32 error (also an error event) |
| `po32.recordings_folder` | str | where `pref.po32.save_recordings` saves recordings |

## Undo

- Every `set` of an undoable address is one step. Bracket a drag or a paint
  stroke with `begin_gesture()` / `end_gesture()` to make it one step; pass
  `burst=True` for a wheel turn (the changes of one address are one step until
  400 ms pass without one); `record=False` keeps a set out of the journal.
- Verbs that rewrite many values (pattern ops, program select, morph learn
  and capture, preset load / paste / initialize / randomize all, drum patch
  load, PO-32 import) are one snapshot step each. Depth 50.
- Not undone: transport, selection (`global.channel`, `pattern.selected`),
  mutes, Edit all, settings (`pref.*`, `midi.*`).
- Core modules that build a new state and install it (preset loads, the
  PO-32 import) wrap the install in
  `with core.bulk_change(label, parts=...):` to make it one step reported by
  poll. Front-ends never write the engine.
- Tried AI candidates and AI pattern previews stay outside the journal until
  `ai.keep` / `ai.replace_patterns` (one step each), above.

## Edit all

`set('chN.<sound>', v)` also sets the same parameter of every **unmuted**
channel when `global.edit_all` is on (or with `edit_all=True`); restoring
code passes `edit_all=False`. Pattern steps are never fanned out.
