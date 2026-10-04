# Inventory of the tkinter GUI: parity checklist and app logic

Research for issue #4 (map: #1). Fact-finding only: nothing here is a design decision.

- **Snapshot:** commit `e77f389` on `main`. Every `file:line` pointer refers to that commit.
- **Files read:** `gui/main_window.py` (5,293 lines), `gui/widgets.py` (1,758), `gui/drum_generator_dialog.py` (932), `gui/po32_import_dialog.py` (1,385), `gui/po32_transfer.py` (848), `gui/po32_import.py` (634), `gui/__init__.py`. Engine modules were read only where the GUI calls them.
- **Entry point:** `run.py:13` imports `gui.main_window.main`; `main()` (`main_window.py:5286`) builds `PythonicGUI` and calls `run()`.

## Headline facts

| What | Count |
|---|---|
| Interactive controls on the main window | 123 (each button in a group counted once) |
| ...of which per-channel drum patch controls | 56 (plus the `edit all` toggle) |
| Popup menus | 4: preset menu (18 commands), pattern menu (14), MIDI-learn context menu, substeps menu (16 entries) |
| Dialog windows | 10: export-audio options, Audio, Synthesis, AI, MIDI, CC Mappings, PO-32 transfer, PO-32 import, AI Drum Generator, custom-substeps prompt |
| Keyboard bindings on the main window | 5 (`1`-`8`, `s`, `l`, `Ctrl+Z`, `Ctrl+Y`) |
| MIDI CC-learnable parameters | 35 |
| Preference keys the GUI reads or writes | 21 |
| App-logic clusters in the GUI layer | 17 (Part 2) |

---

# Part 1: parity checklist

Conventions used in the tables:

- **Engine target** names the call or attribute that the handler drives. `ch` is `synth.get_selected_channel()` unless the row says otherwise. `pm` is the `PatternManager`.
- **Scale** is the transform from the widget value to the engine value.
- **Undo** says how the control enters undo history: `drag` = pre-drag snapshot pushed on mouse release (`command_end`); `push` = snapshot pushed before the change; `-` = no undo entry.
- **CC** is the parameter name in the MIDI CC registry (section 1.10). `-` = not learnable.
- **Edit all** = when the `edit all` toggle is on, the change is also applied to every unmuted channel.

## 1.0 Main window

| Item | Behaviour | Pointer |
|---|---|---|
| Window | Title "Pythonic Drum Synthesizer", default 960×600, min 800×480, resizable | `main_window.py:83-88` |
| Scroll container | Whole UI sits in a canvas with a vertical scrollbar; content width follows window width | `main_window.py:234-259` |
| Mouse-wheel scroll | `bind_all('<MouseWheel>')` scrolls the canvas (Windows/macOS event; Linux `Button-4/5` not bound for the page) | `main_window.py:261-264` |
| Section order (top to bottom) | toolbar, preset/channel strip, drum patch area, pattern section, global bar | `main_window.py:266-279` |

## 1.1 Toolbar

| Control | Widget / range | Engine target | Undo | CC | Pointer |
|---|---|---|---|---|---|
| `program:` | read-only combobox, values `1`-`16` | stores current state in old slot (`synth.store_program`), then `synth.recall_program(new)`; an empty slot is filled with the current state. Patterns are not part of a program | - | - | `:297-306`, `:1463-1491` |
| Morph learn `A` | button, toggles learn A (green while learning) | `morph_manager.start_learn_a / stop_learn`, `apply_effective_position` | - | - | `:316-324`, `:1508-1526` |
| `sound morph` slider | `tk.Scale` 0-100, horizontal | `morph_manager._position = v/100`, then `apply_effective_position()` unless learning. Disabled when endpoints are equal and no learn is active | - | `sound_morph` | `:326-333`, `:1493-1506`, `:1548-1583` |
| Morph learn `B` | button, toggles learn B | `start_learn_b / stop_learn` | - | - | `:335-343`, `:1528-1546` |
| Undo `↶` / Redo `↷` | buttons, disabled when the stack is empty | snapshot restore (Part 2, H) | n/a | - | `:349-364`, `:1643-1670` |
| `master` | knob -60..+10 dB, default 0 | `synth.set_master_volume(v)` | - | `master_volume` | `:369-377`, `:1802-1804` |
| MIDI activity LED | 12 px canvas dot; green for 100 ms on any incoming MIDI message; click opens MIDI Settings | - | - | - | `:379-388`, `:3600-3611` |

## 1.2 Preset and channel strip

| Control | Widget / range | Engine target | Undo | Pointer |
|---|---|---|---|---|
| `PYTHONIC` label | static logo | - | - | `:410-412` |
| `PO-32` button | opens the PO-32 transfer dialog (1.7.7) | - | - | `:414-423` |
| Preset `◀` / `▶` | step the preset combobox and load the file | `_load_preset_file` | push | `:429-443`, `:1672-1686` |
| Preset combobox | read-only list of `*.mtpreset` and `*.json` in the preset folder, sorted; shows the last loaded preset | `_load_preset_file(..., show_message=False)` | push | `:435-437`, `:4803-4838` |
| Preset `▼` | opens the preset menu (1.6.1) | - | - | `:445-449`, `:1688-1721` |
| Patch name display | name of the selected channel, fallback "Channel N" | reads `ch.name` | - | `:458-465`, `:2744-2746` |
| Channel buttons `1`-`8` | `ChannelButton` with LED (blue selected, red muted, green flash on trigger) | click: `synth.select_channel(i)` and switch editors. Click on the already-selected channel: `synth.trigger_drum(i, 64)`; with Ctrl: velocity 127 | - | `:471-491`, `:1756-1795`, `widgets.py:727-796` |
| Mute buttons `m` ×8 | `ToggleButton` | `synth.mute_channel(i, on)`; channel LED turns red | - | `:493-497`, `:1797-1800` |
| Drum type labels ×8 | orange text under each channel | `infer_drum_type(channel.name)` for all 8 channels | - | `:499-505`, `:2748-2751` |

## 1.3 Drum patch area (selected channel)

All rows act on the selected channel. `_update_ui_from_channel()` (`:2732-2838`) re-reads every control in this area from the engine under an `updating_ui` guard so that handlers ignore the programmatic `set_value` calls.

### 1.3.1 Mixing (`:672-773`, handlers `:1806-1978`)

| Control | Widget / range / default | Engine target | Scale | Undo | CC | Edit all |
|---|---|---|---|---|---|---|
| osc / noise mix | `tk.Scale` from 100 (left, "osc") to 0 (right, "noise"), default 50, no value text | `ch.set_osc_noise_mix` | v/100 (1 = all osc) | - | `osc_noise_mix` | no |
| eq freq | knob, log 20-20000 Hz, default 632 | `ch.set_eq_frequency` | 1:1 | drag | `eq_freq` | no |
| edit all | toggle | GUI flag `edit_all_mode` | - | - | - | - |
| distort | knob 0-100, default 0 | `ch.set_distortion` | v/100 | drag | `distortion` | no |
| eq gain | knob -40..+40 dB, default 0 | `ch.set_eq_gain` | 1:1 | drag | `eq_gain` | no |
| level | knob -60..+10 dB, default 0 | `ch.level_db` | 1:1 | drag | `level` | no |
| pan | knob -100..+100, default 0 (shown L/C/R) | `ch.pan` | 1:1 | drag | `pan` | no |
| choke | toggle | `ch.choke_enabled` | bool | push | - | no |
| output | 2-way selector `A` / `B` | `ch.output_pair` | `'A'`/`'B'` | push | - | no |

### 1.3.2 Oscillator (`:775-871`, handlers `:1980-2033`)

| Control | Widget / range / default | Engine target | Scale | Undo | CC | Edit all |
|---|---|---|---|---|---|---|
| waveform | 3 icons: sine, triangle, saw | `ch.set_osc_waveform(WaveformType)` | index 0-2 | push | - | no |
| osc freq | knob, log 20-20000 Hz, default 440 | `ch.set_osc_frequency` | 1:1 | drag | `osc_freq` | no |
| pitch | knob -24..+24 st, default 0 | `ch.set_pitch_semitones` | 1:1 | drag | `pitch` | yes |
| pitch mod | 3-way `Decay` / `Sine` / `Rand` | `ch.set_pitch_mod_mode(PitchModMode)` (DECAYING/SINE/RANDOM) | index | push | - | no |
| amount | knob -120..+120, default 0 | `ch.set_pitch_mod_amount` | 1:1 | drag | `pitch_amount` | no |
| rate | knob, log 1-2000, default 100 | `ch.set_pitch_mod_rate` | 1:1 | drag | `pitch_rate` | no |
| attack | knob, log 0-10000 ms (log floor 0.1), default 0 | `ch.set_osc_attack` | 1:1 | drag | `osc_attack` | no |
| decay | knob, log 10-10000 ms, default 316 | `ch.set_osc_decay` | 1:1 | drag | `osc_decay` | no |

### 1.3.3 Noise (`:873-959`, handlers `:2035-2078`)

| Control | Widget / range / default | Engine target | Scale | Undo | CC |
|---|---|---|---|---|---|
| filter mode | 3-way `LP` / `BP` / `HP` | `ch.set_noise_filter_mode(NoiseFilterMode)` | index | push | - |
| filter freq | knob, log 20-20000 Hz, default 20000 | `ch.set_noise_filter_freq` | 1:1 | drag | `noise_freq` |
| filter q | knob, log 0.5-20, default 0.707 | `ch.set_noise_filter_q` | 1:1 | drag | `noise_q` |
| stereo | toggle | `ch.set_noise_stereo` | bool | push | - |
| envelope | 3-way `Exp` / `Lin` / `Mod` | `ch.set_noise_envelope_mode(NoiseEnvelopeMode)` | index | push | - |
| attack | vertical slider, log 0-10000 ms, default 0 | `ch.set_noise_attack` | 1:1 | drag | `noise_attack` |
| decay | vertical slider, log 10-10000 ms, default 316 | `ch.set_noise_decay` | 1:1 | drag | `noise_decay` |

### 1.3.4 FX (`:1232-1300`, handlers `:1830-1948`)

| Control | Widget / range / default | Engine target | Scale | Undo | CC | Edit all |
|---|---|---|---|---|---|---|
| vintage | knob 0-100, default 0 | `ch.vintage_amount` | v/100 | drag | `vintage` | no |
| rvb time | knob 0-100, default 0 | `ch.reverb_decay` | v/100 | drag | `reverb_decay` | yes |
| rvb mix | knob 0-100, default 0 | `ch.reverb_mix` | v/100 | drag | `reverb_mix` | yes |
| rvb wide | knob 0-200, default 100 | `ch.reverb_width` | v/100 | drag | `reverb_width` | yes |
| delay time | 5-way `1/4` `1/8` `1/16` `1/8T` `1/4.` | `ch.delay_time = DelayTime(n)` with n = 2, 3, 4, 8, 11. Any other stored `DelayTime` displays as `1/8` | table | push | - | yes |
| dly fdbk | knob 0-95, default 30 | `ch.delay_feedback` | v/100 | drag | `delay_feedback` | yes |
| dly mix | knob 0-100, default 0 | `ch.delay_mix` | v/100 | drag | `delay_mix` | yes |
| P.P | toggle | `ch.delay_ping_pong` | bool | push | - | yes |

### 1.3.5 Velocity (`:961-989`, handlers `:2080-2096`)

| Control | Widget / range / default | Engine target | Scale | Undo | CC |
|---|---|---|---|---|---|
| osc | vertical slider 0-200 %, default 0 | `ch.osc_vel_sensitivity` | v/100 | drag | `osc_vel` |
| noise | vertical slider 0-200 %, default 0 | `ch.noise_vel_sensitivity` | v/100 | drag | `noise_vel` |
| mod | vertical slider 0-200 %, default 0 | `ch.mod_vel_sensitivity` | v/100 | drag | `mod_vel` |

### 1.3.6 LFO 1 and LFO 2 (`:1011-1094`, handlers `:1151-1199`)

Two identical racks bound to `ch.lfo1` and `ch.lfo2`. No control in these racks enters undo history.

| Control | Widget / range / default | Engine target | Scale | CC |
|---|---|---|---|---|
| on | toggle | `lfo.enabled` | bool | - |
| wave | combobox `Sin` `Tri` `Saw▲` `Saw▼` `Sq` `S&H` | `lfo.waveform = LFOWaveform(i)` (0-5) | index | - |
| rate | knob, log 0.01-50 Hz, default 1.0 | `lfo.rate_hz` | 1:1 | `lfo1_rate` / `lfo2_rate` |
| depth | knob 0-100, default 0 | `lfo.depth` | 1:1 (not divided) | `lfo1_depth` / `lfo2_depth` |
| sync | combobox `Free` `1/1` `1/2` `1/4` `1/8` `1/16` `1/4.` `1/8.` `1/4T` `1/8T` `2bar` `4bar` | `lfo.sync = SyncDivision(i)` (0-11) | index | - |
| re | toggle, default on | `lfo.retrigger` = RETRIGGER / FREE | bool | - |
| uni | toggle, default off | `lfo.polarity` = UNIPOLAR / BIPOLAR | bool | - |
| destination | combobox: `Off` + 27 targets in 7 groups (Oscillator 6, Noise 4, Mix 4, EQ 2, FX 6, Velocity 3, Global 2: Master Vol, Morph); labels from `MOD_TARGET_LABELS` | `lfo.target = ModTarget` | lookup | - |

### 1.3.7 Pump (`:1096-1147`, handlers `:1203-1230`)

Bound to `ch.pump`. No undo entries.

| Control | Widget / range / default | Engine target | Scale | CC |
|---|---|---|---|---|
| on | toggle | `pump.enabled` | bool | - |
| amount | knob 0-100, default 0 | `pump.amount` | v/100 | `pump_amount` |
| attack | knob, log 0.1-100 ms, default 1 | `pump.attack_ms` | 1:1 | `pump_attack` |
| release | knob, log 1-1000 ms, default 100 | `pump.release_ms` | 1:1 | `pump_release` |
| curve | knob 0-100, default 50 | `pump.curve` | v/100 | `pump_curve` |
| destination | combobox, same 28 options as the LFOs | `pump.target` | lookup | - |

### 1.3.8 Modulation indicators

A 50 ms timer reads `ch._last_mod_offsets` of the selected channel and draws an orange secondary pointer on 21 knobs/sliders whose target is modulated (`:5129-5151`, `:5187-5206`, `widgets.py:245-253`, `:541-548`). The targets Osc/Noise Mix, the three velocity targets, Master Vol and Morph have no indicator widget.

## 1.4 Pattern section (`:509-657`)

| Control | Behaviour | Engine target | Undo | Pointer |
|---|---|---|---|---|
| `⊞` matrix toggle | swaps the single-channel lane editor for the 8×16 matrix (triggers only); button turns blue while the matrix shows | - | - | `:529-535`, `:2609-2623` |
| Pattern buttons `A`-`L` (3 rows of 4) | click: select. Stopped: switch now and reset position. Playing: queue the pattern (applied at the end of the playing pattern); clicking the playing pattern cancels the queue | `pm.select_pattern`, `pm.queued_pattern_index` | - | `:537-557`, `:2100-2126` |
| Pattern button right-click | selects that pattern, then opens the pattern menu | as above | - | `:554`, `:2274-2280` |
| Pattern button colours | playing: flashes green every 250 ms; queued: flashes blue; selected: blue; empty pattern: dim text; chained pattern: blue text | reads `pm` | - | `:5219-5270` |
| `◀◀` chain | toggles chain from the previous pattern into the selected one (not on `A`) | `pm.toggle_chain_from_prev(idx)` | - | `:570-574`, `:2593-2599` |
| `▶▶` chain | toggles chain from the selected pattern to the next (not on `L`) | `pm.toggle_chain_to_next(idx)` | - | `:576-580`, `:2601-2607` |
| `Menu` | opens the pattern menu for the selected pattern | - | - | `:594-598` |
| `Copy` | copies the selected channel's triggers, accents, fills and probabilities from the selected pattern into `root.clipboard_data` (substeps are not copied) | reads pattern channel | - | `:600-604`, `:2636-2649` |
| `Paste` | writes the clipboard into the selected channel of the selected pattern; warns "Nothing in clipboard" when empty | `PatternChannel.set_triggers/accents/fills/probabilities` | - | `:606-610`, `:2652-2668` |
| `Prob` | toggles probability-edit mode on all 8 lane editors (button turns green) | GUI flag | - | `:612-619`, `:2147-2159` |
| Channel label `chN` | names the channel shown in the lane editor | - | - | `:633-636`, `:1790-1792` |
| Lane editor | one `PatternEditor` per channel, only the selected one packed; 16 steps; lanes `trig` `acc` `fill` `sub` `len` + step numbers; green playhead | see 1.8.3 | push per edit | `:638-650`, `widgets.py:994-1585` |
| Matrix editor | 8 channels × 16 steps of triggers; click or drag along a row toggles cells | `PatternChannel.set_trigger` | push per cell | `:652-657`, `:2670-2686`, `widgets.py:1587-1759` |

## 1.5 Global bar (`:1302-1422`)

| Control | Widget / range | Engine target | Undo | Pointer |
|---|---|---|---|---|
| `■` stop | circular button | `pm.stop_playback()`, playhead to 0 | - | `:1322-1325`, `:2213-2225` |
| `▶` play | circular button (green ring while playing) | `pm.start_playback(selected)` | - | `:1327-1330`, `:2202-2211` |
| `BPM` | entry, integer 1-300, applied on Return or focus-out; invalid input reverts | `pm.set_bpm`, `synth.set_bpm` | - | `:1336-1348`, `:1449-1459` |
| Step rate | 5 buttons `1/8` `1/8T` `1/16` `1/16T` `1/32` | `pm.set_step_rate` | - | `:1350-1363`, `:1424-1432` |
| `swing` | `tk.Scale` 0-100 % | `pm.swing = v/100` (attribute write) | - | `:1365-1389`, `:1444-1447` |
| `fill rate` | 7 buttons `2x`-`8x` | `pm.set_fill_rate(n)` | - | `:1391-1409`, `:1434-1442` |
| Hint text | "Keys 1-8: Trigger channels" | - | - | `:1411-1419` |

## 1.6 Popup menus

### 1.6.1 Preset menu (`▼` button, `:1688-1721`)

| Entry | Action | Pointer |
|---|---|---|
| Open Preset... | file dialog (`*.mtpreset`, `*.json`) in the preset folder, then load (Part 2, N) | `:2870-2884` |
| Save Preset As... | file dialog, writes JSON (Part 2, N) | `:2840-2868` |
| Load Drum Patch (.mtdrum)... | loads one patch into the selected channel | `:2886-2910` |
| Save Drum Patch (.mtdrum)... | saves the selected channel, default name = channel name | `:2912-2933` |
| Cut Preset / Copy Preset / Paste Preset | in-memory preset clipboard | `:1723-1737` |
| Initialize Preset | resets all channels and all patterns | `:1739-1745` |
| Randomize All | randomizes all channels and the selected pattern | `:1747-1754` |
| Select Preset Folder... | directory dialog, then refresh the list | `:3105-3115` |
| Refresh Preset List | rescans the preset folder | `:4803-4829` |
| Transfer to PO-32... | 1.7.7 | `:3613-3621` |
| Import from PO-32... | 1.7.8 | `:3623-3646` |
| AI Drum Generator... | 1.7.9 | `:3648-3695` |
| Audio Settings... / MIDI Settings... / Synthesis Settings... / AI Settings... | 1.7.2 to 1.7.5 | `:3721`, `:4404`, `:4175`, `:4277` |

### 1.6.2 Pattern menu (`Menu` button or right-click on a pattern button, `:2227-2321`)

Every entry targets the selected pattern. Errors show "Pattern operation failed". Only the two AI entries push undo.

| Entry | Engine call |
|---|---|
| Cut Pattern / Copy Pattern / Paste Pattern | `pm.cut_pattern` / `copy_pattern` / `paste_pattern` |
| Exchange Pattern | `pm.exchange_pattern` |
| Shift Left / Shift Right | `pm.shift_pattern_left` / `shift_pattern_right` |
| Reverse | `pm.reverse_pattern` |
| Randomize | `pm.randomize_pattern` |
| Alter Pattern | `pm.alter_pattern` |
| Randomize Accents/Fills | `pm.randomize_accents_fills` |
| Randomize Pattern (AI) | `PatternGenerator.generate(...)` then `pm.apply_single_pattern` (`:2345-2369`) |
| Randomize Channel (AI) | same, then `pm.apply_single_channel(idx, selected_channel, ...)` (`:2371-2395`) |
| Export Pattern to MIDI File... | Part 2, O (`:2397-2480`) |
| Export Pattern to Audio File... | Part 2, O (`:2482-2591`) |

### 1.6.3 MIDI-learn context menu (right-click on any of the 35 CC-registered widgets, `:3376-3418`)

Entries depend on state: "Mapped to CCn (name)" (disabled) + "Remove CC Mapping"; "Pitch Bend → This Parameter" (disabled) + "Remove Pitch Bend Mapping"; "MIDI Learn (CC)" or "Cancel MIDI Learn"; "Assign Pitch Bend" (hidden if already assigned here); "MIDI Settings..." (opens the CC Mappings dialog, not the MIDI Settings dialog).

### 1.6.4 Substeps menu (right-click on any step, or click in the `sub` lane, `widgets.py:1194-1232`)

"No substeps", 14 presets (`oo` `o-` `-o` `ooo` `oo-` `o-o` `o--` `-oo` `--o` `oooo` `o-o-` `-o-o` `ooo-` `o---`), "Custom..." (prompt accepting `o`/`O`/`-`, `widgets.py:1242-1260`). The current value is ticked. Writes `step.substeps` (`main_window.py:2142-2145`).

## 1.7 Dialogs

All dialog windows are `Toplevel`s, transient to the main window, with `grab_set()`, centred on the parent.

### 1.7.1 Export Audio Options (`:2491-2510`)
Radio buttons: `None (truncate)`, `Append (add silence)` (+2 s), `Loop (repeat pattern)` (+1 pass); `OK`. Then a save dialog (`pythonic_pattern_<X>.wav`). Closing the window keeps the default `none`.

### 1.7.2 Audio Settings (`:3721-4173`, 450×810)

| Control | Values | Preference |
|---|---|---|
| Audio Output Device | `(System Default)` + output devices by name; "Currently using: ..." label | `audio_output_device` |
| Audio Input Device (for PO-32 recording) | `(System Default)` + input devices; default-input label | `audio_input_device` |
| Audio Buffer Size | 2, 5, 10, 15, 23.8 (default), 30, 50, 75, 100 ms | `audio_buffer_ms` |
| Output Sample Rate | 96000, 48000, 44100 (default), 32000, 22050, 11025, 8000 Hz; filtered to the rates `sd.check_output_settings` accepts for the chosen device; status line "Device supports n of 7..." | `audio_sample_rate` |
| Internal Synth Rate | Same as output (default), 22050, 11025, 8000 Hz | `synth_sample_rate` (stored as the effective rate) |
| Mono output | checkbox | `audio_mono` |
| Buttons | `Refresh` (re-lists devices), `Apply Now` (save + apply + restart stream), `OK` (save only, for next launch), `Cancel` | |

### 1.7.3 Synthesis Settings (`:4175-4275`, 400×200)
`Parameter Smoothing Time` slider 5-100 ms → `param_smoothing_ms` and `channel.set_smoothing_time(ms)` on all channels. Buttons `Apply`, `OK`, `Cancel`.

### 1.7.4 AI Settings (`:4277-4402`, 500×260)
`Pattern Model Checkpoint` entry + `Browse...` (`*.pt`) + `Clear`; note whether the bundled checkpoint exists; `Pattern Temperature` spinbox 0.1-3.0 step 0.1. `OK` writes `drum_generator_pattern_model_path`, `drum_generator_pattern_temperature` and drops the cached generator; `Cancel`.

### 1.7.5 MIDI Settings (`:4404-4619`, 400×420)
`MIDI Input Device` (`(Auto-detect)` + ports) with connection status; `Base Note for Drum Mapping` (C-1 to note 120, "Channels 1-8 will respond to notes n - n+7"); `Sync BPM to MIDI Clock` checkbox with current synced BPM; static help text (Note On, Program Change 0-11, Start, Stop, Continue, Clock). Buttons `Refresh Devices`, `CC Mappings...` (closes this dialog, opens 1.7.6), `Apply`, `OK`. Apply writes `midi_base_note`, `midi_input_device`, `midi_clock_sync`, sets `midi_enabled = True`, reconnects when the device changed.

### 1.7.6 MIDI CC Mappings (`:4621-4801`, 500×450)
8 rows of (CC combobox, parameter combobox, `Clear`). CC choices: `(None)` + CC 1, 2, 4, 7, 10, 11, 12, 13, 16, 17, 18, 19, 71, 74 + any already-mapped CC. Parameters: `(None)` + the 35 registry names, sorted. Existing mappings fill the rows. Buttons `Clear All`, `Apply`, `OK` (Apply clears all mappings, re-adds the rows, writes `midi_cc_mappings`). Mappings beyond 8 are not shown and are dropped on Apply.

### 1.7.7 PO-32 transfer (`po32_transfer.py:317-848`, 420×550)
Shows the preset name; `Transfer sounds to:` `1 - 8` / `9 - 16`; `Pattern (chain) to:` options built from the chain groups of `pm` (e.g. `A - B`, `C`); 8 channel checkboxes (initially unchecked for muted channels); receive-mode instruction text; progress bar + status; `▶ Transfer` / `■ Stop`; `💾 Save WAV`; `Close`. Every settings change regenerates the FSK signal. Pattern payloads are always the codec's default (empty) patterns and the state block is the default one (`:203-226`, `:262-289`); only the 8 sounds carry live data, with a default right-hand (morph) patch.

### 1.7.8 PO-32 import (`po32_import_dialog.py:31-1385`, 720×680, scrollable)
- **Source:** input device combobox (preferred = `audio_input_device`), `🔊 Monitor` toggle with VU meter and dB readout, `💾 Save recorded audio to file (debug)` checkbox (`po32_debug_save_recordings`) + `📂 Open Folder`, `Import WAV File...`, `● Record` / `■ Stop`, source status line.
- **Drum Bank:** `Bank 0 (Drums 1-8)` / `Bank 1 (Drums 9-16)` radios, enabled per decoded content.
- **Patterns:** 16 numbered buttons (toggle selection, max 12, shows `n→X`), `Select First 12`, `Clear All`, `n/12 patterns selected`, destination-letter menus (`A`-`L`, conflicts swap), 8×16 trigger grid for the focused pattern, summary text, `▶ Preview` / `■ Stop`.
- **Drum Patches:** text summary of the 8 patches of the bank.
- **Actions:** `Import Drums + n Patterns`, `Cancel`.
- Import result is described in Part 2, Q.

### 1.7.9 AI Drum Generator (`drum_generator_dialog.py:42-932`, 920×680)
- **Top:** title, `Install ML Support` (asks, then runs `pip install -r requirements-ml.txt`).
- **Models:** `Patch Model:` status + `Load...` (`*.pt`), `Pattern Model:` status + `Load...`.
- **Controls:** `Patch Temp` 0.1-3.0 (pref `drum_generator_patch_temperature`), `Candidates` 1-32 (default 8), `Seed` entry + `Reseed`, `Generate All 8`.
- **Patterns:** `Keep Current` / `Generate New AI Patterns`; `Pattern Temp` 0.1-3.0 (generate mode only); bank status.
- **8 slot lanes** (`SLOT_MAP`: BD, SD, CH, OH, TOM HI/PERC, TOM LO/PERC, CLAP/RIM, CY/FX): type combobox (allowed drum types), `Generate`, `<` `i / n` `>`, candidate name, `Preview`, `Apply` checkbox.
- **Bottom:** `Loop Preview` / `Stop Loop`, `Preview Bank` / `Stop Bank`, `Close`, `Apply Selected`, `Replace All Patterns From AI` (enabled only with a cached bank).

### 1.7.10 Other system dialogs
File dialogs: open/save preset, load/save drum patch, select preset folder, MIDI export, WAV export, AI checkpoint, PO-32 WAV open/save. Message boxes for errors and for "No Pattern Model" (`:2332-2338`). Errors also go to stdout with `print`.

## 1.8 Widget interaction conventions (`widgets.py`)

### 1.8.1 Knob (`RotaryKnob`, `:63-394`) and vertical slider (`VerticalSlider`, `:397-665`)
- Vertical drag: knob 200 px = full range, slider track height = full range; Shift = ×0.1 fine.
- Knob only: Alt+click toggles circular drag mode (tested with state mask `0x20000`).
- Ctrl+click or double-click: reset to the widget default.
- Mouse wheel (`MouseWheel` and Linux `Button-4/5`): knob ±1 % of normalized range per notch, slider ±2 %.
- Log scaling when `logarithmic=True`, with a floor of 0.1 for attack/decay labels, 1.0 for freq/rate labels when `min_val <= 0`.
- Floating hint window (singleton, `:18-60`) shows label and formatted value while dragging; the unit is guessed from the label (Hz, dB, ms, L/C/R for pan; sliders default to `%`).
- `command` fires on every `set_value`; `command_end('start'|'end')` fires on press and release.
- Wheel, reset and programmatic `set_value` do not call `command_end`.

### 1.8.2 Toggle, selectors, buttons
`ToggleButton` (`:933-992`) flips on click and shows an LED. `ModeSelector` (`:873-930`) picks by x position. `WaveformSelector` (`:799-870`) has 3 icons. `CircularButton` (`:668-724`) has an active ring. `ChannelButton` (`:727-796`) passes the click event so the main window can read Ctrl.

### 1.8.3 Lane editor gestures (`PatternEditor`, `:994-1585`)
- Click in `trig`/`acc`/`fill`: toggle; drag continues toggling along the lane. Turning a trigger off clears its accent and fill; accent and fill only toggle on steps with a trigger (`:1293-1315`).
- Ctrl+click in `trig`: toggles trigger and accent together (`:1141-1162`).
- Shift+click in `trig`/`acc`/`fill`: applies the new value to the same step on all 8 channels; the editor passes an empty muted set, so muted channels are included (`:1164-1186`, `main_window.py:2161-2191`).
- Click in `sub` or right-click anywhere: substeps menu.
- Click in `len`: pattern length = step + 1, applied to the selected pattern and shown on all 8 editors (`:1129-1134`, `main_window.py:2193-2200`). Length changes do not enter undo history.
- Probability mode: vertical drag on any step changes its probability, 1 px = 1 %, range 0-100 (`:1263-1280`); triggered steps show the value and a red-to-green colour.

## 1.9 Keyboard

| Key | Action | Pointer |
|---|---|---|
| `1`-`8` | `synth.trigger_drum(n-1, 127)` + channel LED flash 100 ms | `:2712-2714`, `:2724-2730` |
| `s` | Save Preset As dialog | `:2717-2718` |
| `l` | Open Preset dialog | `:2721-2722` |
| `Ctrl+Z` / `Ctrl+Y` | undo / redo | `:201-202` |

`<Key>` is bound on the root window (`:200`, again at `:1422`). Root bindings are part of every main-window widget's bind tags, so these keys also fire while typing in the BPM entry.

## 1.10 MIDI input behaviour

| Input | Behaviour | Thread path | Pointer |
|---|---|---|---|
| Note On (base..base+7, velocity > 0) | `synth.trigger_drum(ch, velocity)`; channel LED flash | trigger runs on the MIDI thread; flash via `root.after(0)` | `:3176-3188` |
| Program Change 0-11 | select pattern A-L (`_select_pattern_by_index`) | `root.after(0)` | `:3190-3206` |
| Start | stop, then start the selected pattern from step 0 | `root.after(0)` | `:3208-3221` |
| Stop | same as `■` | `root.after(0)` | `:3223-3225` |
| Continue | sets `pm.is_playing = True` without resetting position (no-op if playing) | `root.after(0)` | `:3227-3241` |
| Clock | when sync is on, BPM from `MidiManager`, rounded and clamped 1-300, written to `pm`, `synth` and the BPM entry | `root.after(0)` | `:3249-3265` |
| Control Change (mapped) | normalize 0-127 to 0-1; log widgets use the widget's own log curve over the widget range, other widgets use the registry min/max; result goes through `widget.set_value` (or `Scale.set`), which drives the normal handler | `root.after(0)` | `:3267-3270`, `:3333-3357` |
| Pitch bend (assigned parameter) | temporary offset of ±50 % of the registry range around the value at bend start, linear, clamped; value restored when the wheel is within ±0.02 of centre | `root.after(0)` | `:3272-3331` |
| Any message | activity LED | `root.after(0)` | `:3243-3247` |
| MIDI learn | right-click → "MIDI Learn (CC)": widget background flashes orange every 300 ms; the next CC is mapped, saved to preferences, flashing stops | learned CC handed over via `root.after(0)` | `:3420-3476` |

Auto-connect at startup: the preferred device first, else the first available port, when `midi_enabled` (`:3119-3174`). Pitch-bend default target is `pitch`; default CC mappings in preferences are CC1 → `osc_freq`, CC2 → `noise_freq`.

**CC registry** (`:3539-3598`): `level`, `pan`, `distortion`, `eq_freq`, `eq_gain`, `vintage`, `reverb_decay`, `reverb_mix`, `reverb_width`, `delay_feedback`, `delay_mix`, `osc_freq`, `pitch`, `pitch_amount`, `pitch_rate`, `osc_attack`, `osc_decay`, `noise_freq`, `noise_q`, `noise_attack`, `noise_decay`, `osc_vel`, `noise_vel`, `mod_vel`, `osc_noise_mix`, `master_volume`, `sound_morph`, `lfo1_rate`, `lfo1_depth`, `lfo2_rate`, `lfo2_depth`, `pump_amount`, `pump_attack`, `pump_release`, `pump_curve`.

The registry min/max differ from the widget range for: `eq_freq` (100-10000 vs 20-20000), `eq_gain` (±12 vs ±40), `reverb_width` (0-100 vs 0-200), `delay_feedback` (0-100 vs 0-95), `osc_freq` (20-2000 vs 20-20000), `pitch_amount` (0-96 vs ±120), `pitch_rate` (0-500 vs 1-2000), `osc_attack` (0-1000 vs 0-10000), `osc_decay` (1-5000 vs 10-10000), `noise_freq` (100-15000 vs 20-20000), `noise_attack` (0-1000 vs 0-10000), `noise_decay` (1-5000 vs 10-10000), `osc_vel`/`noise_vel`/`mod_vel` (0-100 vs 0-200). CC on log widgets ignores the registry range; pitch bend always uses it.

## 1.11 Preferences used by the GUI

Stored by `PreferencesManager` (JSON in the platform config dir). Every `set()` writes the file at once (`preferences_manager.py:166-169`).

| Key | Read | Written |
|---|---|---|
| `audio_output_device`, `audio_buffer_ms`, `audio_sample_rate`, `synth_sample_rate`, `audio_mono` | startup (`:94-107`, `:174`), `_start_audio`, Audio dialog | Audio dialog OK / Apply Now |
| `audio_input_device` | Audio dialog, PO-32 import dialog | Audio dialog |
| `param_smoothing_ms` | startup (`:110-112`), synth-rate change, Synthesis dialog | Synthesis dialog |
| `midi_enabled`, `midi_base_note`, `midi_input_device`, `midi_clock_sync` | `_init_midi` | MIDI dialog Apply/OK |
| `midi_cc_mappings` | `_init_midi` | MIDI learn, remove mapping, CC dialog Apply |
| `midi_pitchbend_target` | `_init_midi` | Assign/Remove Pitch Bend |
| `drum_generator_model_path` | generator dialog | generator dialog on successful load |
| `drum_generator_pattern_model_path` | `PatternGenerator.resolve_model_path`, AI dialog, generator dialog | AI dialog OK, generator dialog load |
| `drum_generator_patch_temperature` | generator dialog | generator dialog close |
| `drum_generator_pattern_temperature` | AI randomize, AI dialog, generator dialog | AI dialog OK, generator dialog close |
| `preset_folder` | preset list, file dialogs | Select Preset Folder |
| `last_preset` | startup auto-load, preset list selection | every successful preset load |
| `recent_files` | never read by the GUI | save and load (`add_recent_file`) |
| `po32_debug_save_recordings` | PO-32 import dialog | PO-32 import checkbox |

Defined in `DEFAULT_PREFERENCES` but not used by the GUI: `window_width`, `window_height`, `master_volume_db`, `max_recent_files` (the last one is used inside the manager).

## 1.12 Entries that fail or cannot be reached today

Parity work needs to know these; they are recorded as observed, not as decisions.

| Item | Fact | Pointer |
|---|---|---|
| Cut / Copy / Paste Preset | call `PresetManager.export_preset_to_dict` / `import_preset_from_dict`, which do not exist in `pythonic/preset_manager.py` | `:1731`, `:1736` |
| Initialize Preset, Cut Preset | call `DrumChannel.reset_to_defaults()`, which does not exist | `:1742` |
| Randomize All | calls `DrumChannel.randomize()`, which does not exist | `:1751` |
| MIDI Program Change | `_select_pattern_by_index` calls `set_selected` on plain `tk.Button` pattern buttons, which have no such method | `:3201-3203` |
| Export all drums / current drum to WAV | `_export_all_wavs`, `_export_current_drum` exist but nothing calls them | `:2935-2964` |
| `_on_swing_change`, `_update_pattern_position_display`, `_build_header`, `_push_undo_state_now` | defined, never called | `:2625`, `:5208`, `:390`, `:1638` |
| `gui/po32_import.py` | older single-purpose import dialog; no module imports it (`main_window` uses `po32_import_dialog.py`) | whole file |
| LFO/pump target `Morph` | the synth applies a MORPH offset only `if hasattr(self, '_morph_manager')` (`synthesizer.py:313`); nothing in the repo sets `synth._morph_manager` | - |

---

# Part 2: app-logic inventory

Thread names used below:

- **Tk**: the Tk main loop thread (all widget callbacks and `after` timers).
- **Audio**: the PortAudio callback thread of the main `sd.OutputStream`.
- **MIDI**: the `MidiManager._midi_loop` polling thread (`midi_manager.py:381-394`, 1 ms sleep).
- **Worker**: a `threading.Thread` started by a dialog.
- **Dialog audio**: PortAudio callback threads of streams opened by dialogs.
- **Synth pool**: the engine's own channel-processing pool (`parallel_channel_processing=True`), shut down in `run()`.

## A. Composition, startup and shutdown

| Behaviour | Pointer | State owned | Thread | Engine APIs |
|---|---|---|---|---|
| Builds every service: preferences, synth at the internal rate (capped to the output rate), mono, smoothing on all channels, `PresetManager`, `PatternManager(8, 16)`, `MidiManager` + `_init_midi`, `MorphManager` | `main_window.py:81-127` | `preferences_manager`, `synth`, `preset_manager`, `pattern_manager`, `midi_manager`, `morph_manager`, `sample_rate`, `synth_sample_rate`, `_resample_ratio` | Tk | `PythonicSynthesizer(sr, parallel_channel_processing=True)`, `set_mono`, `channel.set_smoothing_time`, `PresetManager`, `PatternManager`, `MidiManager`, `MorphManager` |
| Order after the UI is built: CC registry, undo registration (first snapshot), key bindings, audio start, `synth.set_bpm(pm.bpm)`, UI refresh, flash timer, UI tick, auto-load of `last_preset` | `:190-222` | - | Tk | `synth.set_bpm` |
| Shutdown after `mainloop` exits: stop stream and UI timer, `midi_manager.cleanup()`, `synth.cleanup()` | `:5272-5283` | - | Tk | `MidiManager.cleanup`, `synth.cleanup` |

## B. Audio stream and callback

| Behaviour | Pointer | State owned | Thread | Engine APIs |
|---|---|---|---|---|
| Open stream: resolve preferred device by name, validate the rate with `sd.check_output_settings` (fallback to the device default rate), then try (device, rate) → (device, 44100) → (default, 44100); stereo float32, block size from `audio_buffer_ms`; prints a summary | `:4982-5060` | `audio_stream`, `_audio_block_size`, `sample_rate` (may change on fallback, preference not updated) | Tk | - |
| Callback: count underruns; adaptive drop when the last 10 callbacks average above 90 % of the buffer time (outputs the last good block, discards sequencer events); lazily creates the `StepSequencer`; restarts it when `pm.playback_generation` changes; advances it by the synth-rate frame count; renders with `process_audio_events`; linear upsample when synth rate < output rate; keeps a copy as "last good audio"; records timings | `:4852-4943` | `sequencer`, `_seq_generation`, `callback_times`, `trigger_times`, `process_times`, `underrun_count`, `callback_count`, `dropped_callback_count`, `_last_good_audio`, `current_play_position` (under `position_lock`) | Audio | `StepSequencer(pm, synth.sr)`, `start(None, synth_clock=synth.sample_clock)`, `advance`, `stop`, `synth.process_audio_events` |
| Linear upsampler with cached index arrays | `:5098-5111` | `_resample_x_out`, `_resample_x_in`, `_resample_buffer`, `_resample_in/out_frames` | Audio | - |
| Change output rate (no synth rebuild): recompute ratio, block size, buffer time, last-good buffer, reset resampler cache | `:5062-5074` | as above | Tk | - |
| Change synth rate: snapshot preset data, build a new `PythonicSynthesizer`, restore mono, preset and smoothing, re-point `preset_manager.synth` and `morph_manager.synth`, drop the sequencer | `:5076-5096` | `synth` (replaced), `sequencer`, `_seq_generation` | Tk | `get_preset_data`, `load_preset_data`, `set_mono`, `set_smoothing_time` |
| Stop stream and cancel the UI tick | `:5113-5123` | `audio_stream`, `ui_update_timer` | Tk | - |
| Device listing (outputs, inputs) | `:3697-3719` | - | Tk | - |
| Unused fields: `audio_buffer`, `buffer_lock` | `:131-132` | - | - | - |

## C. Performance reporting

| Behaviour | Pointer | State owned | Thread | Engine APIs |
|---|---|---|---|---|
| Every 5 s (or after the first underrun once 50 callbacks ran) prints callback count, underruns, drops, buffer time, active channels, sample-drop flag, rates, mono, average/max callback time, utilization, average render and sequencing time, warnings | `:4937-4943`, `:4945-4980` | `last_perf_report`, timing deques | Audio (prints from the callback) | reads `channel.is_active` |

## D. Transport and sequencer hand-off

| Behaviour | Pointer | State owned | Thread | Engine APIs |
|---|---|---|---|---|
| Play: start the selected pattern | `:2202-2211` | `_last_playing_pattern_idx` | Tk | `pm.start_playback` (bumps `playback_generation`) |
| Stop: stop, reset editors' playhead | `:2213-2225` | same | Tk | `pm.stop_playback` |
| MIDI Start / Stop / Continue | `:3208-3241` | same | MIDI → Tk | `pm.stop_playback`, `start_playback`, direct `pm.is_playing = True` |
| Sample-accurate sequencing runs inside the audio callback (B). It writes `pm.play_position`, and on pattern end calls `pm.advance_to_next_pattern()` (chains, queue) | `sequencer.py:76-88` via `:4895` | `pm` playback fields | Audio | `StepSequencer` |
| UI tick every 50 ms: copies the play position into all lane editors; when `pm.playing_pattern_index` changed (chain or queue), selects that pattern and refreshes buttons and editors; updates modulation indicators | `:5125-5206` | `ui_update_timer`, `_last_playing_pattern_idx`, `_mod_target_widget_map`, `_mod_active_targets` | Tk | `pm.select_pattern`, reads `ch._last_mod_offsets` |
| Flash timer every 250 ms toggles `button_flash_state` and repaints pattern buttons | `:5219-5270` | `button_flash_state` | Tk | reads `pm`, `Pattern.is_empty`, `chained_to_next/from_prev` |
| Drum generator dialog stops the transport while open and restarts the saved pattern on close | `:3648-3695` | locals | Tk | `pm.start_playback`, `stop_playback` |

## E. Pattern selection, queueing and chains

| Behaviour | Pointer | State owned | Thread | Engine APIs |
|---|---|---|---|---|
| Select: when playing, queue a different pattern or cancel the queue; when stopped, reset `play_position` and `current_step` | `:2100-2126` | - | Tk | `pm.select_pattern`, `queued_pattern_index`, `play_position`, `current_step` |
| MIDI program change → select | `:3190-3206` | - | MIDI → Tk | `pm.selected_pattern_index` |
| Chain toggles | `:2593-2607` | - | Tk | `pm.toggle_chain_from_prev`, `toggle_chain_to_next` |
| Follow playback across chains | `:5162-5174` | `_last_playing_pattern_idx` | Tk | `pm.select_pattern` |

## F. Pattern editing

| Behaviour | Pointer | State owned | Thread | Engine APIs |
|---|---|---|---|---|
| Lane edit dispatch by lane (`trig`, `acc`, `fill`, `prob`, `sub`) into the selected pattern | `:2128-2145` | - | Tk | `PatternChannel.set_trigger/accent/fill/probability`, `steps[i].substeps` |
| Edit on all channels (Shift+click) and per-editor refresh | `:2161-2191` | - | Tk | same |
| Pattern length | `:2193-2200` | `editor.pattern_length` ×8 | Tk | `Pattern.set_length` |
| Matrix edit and matrix refresh | `:2670-2686` | `matrix_view_active` | Tk | `set_trigger`, `get_triggers` |
| Lane editor refresh from the model | `:2688-2705` | editor data copies | Tk | `PatternChannel.steps` |
| Channel clipboard (copy/paste of one channel's lanes) | `:2636-2668` | `root.clipboard_data` (attribute set on the Tk root) | Tk | `get_*`/`set_*` lane lists |
| Pattern menu operations | `:2282-2321` | - | Tk | `pm.cut/copy/paste/exchange/shift/reverse/randomize/alter/randomize_accents_fills` (clipboard lives in `pm.clipboard_pattern`) |
| Probability mode flag | `:2147-2159` | `probability_mode_active` | Tk | - |
| Editor-local rules (accent/fill need a trigger, Ctrl+click toggles trigger+accent, drag painting) live in the widget | `widgets.py:1113-1315` | editor lists | Tk | - |

## G. AI pattern randomize

| Behaviour | Pointer | State owned | Thread | Engine APIs |
|---|---|---|---|---|
| Lazy `PatternGenerator`, `ensure_loaded(preferences)`, warning when no model; cache dropped by AI Settings OK | `:2325-2339`, `:4386-4388` | `_pattern_gen` | Tk | `PatternGenerator.ensure_loaded` |
| Build 8 raw patches from live channels; generate 1 pattern with tempo, swing, fill rate, step rate, temperature preference; apply to a pattern or one channel; push undo first | `:2341-2395` | - | Tk (model inference runs on Tk) | `channel_to_raw_patch`, `PatternGenerator.generate`, `pm.apply_single_pattern`, `apply_single_channel` |

## H. Undo / redo

| Behaviour | Pointer | State owned | Thread | Engine APIs |
|---|---|---|---|---|
| Snapshot = deep copies of `synth.get_preset_data()`, `pm.to_dict()`, `morph_manager.to_dict()` | `:1585-1590` | - | Tk | those three |
| Restore: load the three, set morph slider, refresh channel UI, lane editors, matrix, undo buttons, morph UI; accepts old 2-tuples | `:1592-1610` | - | Tk | `synth.load_preset_data`, `pm.from_dict`, `morph_manager.from_dict` |
| Push (clears redo, max 50) | `:1612-1620` | `_undo_stack`, `_redo_stack`, `_max_undo` | Tk | - |
| Drag-coalesced push: snapshot on press, commit on release; wired on 24 knobs/sliders | `:1622-1636`, `:3519-3537` | `_pre_drag_snapshot` | Tk | - |
| Undo / redo | `:1652-1670` | stacks | Tk | - |
| Callers that push before a change: delay time, P.P, choke, output, waveform, pitch mod mode, noise filter mode, stereo, noise envelope mode, lane/matrix edits, AI randomize, preset file load | listed handlers, `:2968` | - | Tk | - |
| Drum generator apply pushes **after** the change has been applied | `:3661-3666` | - | Tk | - |
| Not recorded: mix slider, master, morph, LFO and pump controls, mutes, BPM, step rate, fill rate, swing, pattern length, channel clipboard paste, pattern menu edits (non-AI), chain toggles, program switch, drum patch load, PO-32 import, wheel/reset gestures on knobs | - | - | - | - |

## I. Program bank (16 slots)

| Behaviour | Pointer | State owned | Thread | Engine APIs |
|---|---|---|---|---|
| Switch slot: store old, recall new, or copy current into an empty slot and write `synth._current_program` | `:1463-1491` | `program_var` | Tk | `store_program`, `recall_program`, `get_current_program`, `_current_program` |
| Save/load with JSON presets (`programs` key); a JSON preset without it resets `synth._programs` and `_current_program` | `:2863`, `:3086-3095` | - | Tk | `get_programs_data`, `load_programs_data`, `_programs`, `NUM_PROGRAMS` |

## J. Sound morph

| Behaviour | Pointer | State owned | Thread | Engine APIs |
|---|---|---|---|---|
| Slider → position (written to `_position`), apply unless learning, refresh channel UI | `:1493-1506` | - | Tk | `MorphManager._position`, `is_learning`, `apply_effective_position` |
| Learn A / B state machine (A and B exclusive) | `:1508-1546` | - | Tk | `get_learn_mode`, `start_learn_a/b`, `stop_learn` |
| Button colours and slider enable rule | `:1548-1583` | - | Tk | `has_different_endpoints` |
| Endpoint init on `.mtpreset` load and on JSON without `morph` (slider centred to 50) | `:3029-3036`, `:3080-3084` | - | Tk | `_init_endpoints` |
| PO-32 import sets endpoint A from decoded left patches and endpoint B from right patches (or equal to A), then the main window puts the slider at 0 | `po32_import_dialog.py:1279-1325`, `main_window.py:3625-3631` | - | Tk | `capture_endpoint_a`, `capture_endpoint_b` |
| The import dialog finds the morph manager through `root.morph_manager`, an attribute the main window sets on the Tk root (`:3641`), with a fallback that scans the parent's attributes | `po32_import_dialog.py:1281-1292` | - | Tk | - |

## K. Edit-all fan-out

| Behaviour | Pointer | State owned | Thread | Engine APIs |
|---|---|---|---|---|
| When `edit_all_mode` is on, 8 handlers loop over unmuted channels: reverb time, reverb mix, reverb width, delay time, delay feedback, delay mix, ping-pong, pitch. All other handlers act on the selected channel only. The widget display is not changed for the other channels | `:1843-1948`, `:1987-1996` | `edit_all_mode` | Tk | `channel.muted`, the setters above |

## L. Parameter binding and UI refresh

| Behaviour | Pointer | State owned | Thread | Engine APIs |
|---|---|---|---|---|
| ~50 handlers translate widget values to engine units (see 1.3) and guard on `updating_ui` | `:1151-1230`, `:1806-2096` | `updating_ui` | Tk (also reached from MIDI CC / pitch bend through `after`) | DrumChannel setters and attributes |
| Engine → widgets refresh of the whole patch area, patch name and 8 drum-type labels | `:2732-2838` | - | Tk | `ch.*`, `ch.oscillator.*`, `ch.osc_envelope.*`, `ch.noise_gen.*`, `lfo*`, `pump`, `infer_drum_type` |
| Channel selection keeps two copies of the selection (`self.selected_channel` and `synth.select_channel`) and swaps the packed lane editor | `:1756-1795` | `selected_channel`, `current_pattern_editor_index` | Tk | `synth.select_channel` |
| Global controls (BPM, step rate, swing, fill rate, master) write `pm`/`synth` directly | `:1424-1459`, `:1802-1804` | button highlight state | Tk | `pm.set_bpm/set_step_rate/set_fill_rate`, `pm.swing`, `synth.set_bpm`, `set_master_volume` |
| Mutes | `:1797-1800` | - | Tk | `synth.mute_channel` |
| Manual preview triggers (keys 1-8, re-click of the selected channel) | `:1763-1774`, `:2724-2730` | - | Tk | `synth.trigger_drum` |

## M. MIDI routing, learn and CC mapping

| Behaviour | Pointer | State owned | Thread | Engine APIs |
|---|---|---|---|---|
| Init: read preferences, configure base note, clock sync, CC map (string keys → int), pitch-bend target, register 9 callbacks, auto-connect | `:3119-3174` | - | Tk | `MidiManager.set_*`, `connect`, `auto_connect` |
| Callbacks from the MIDI thread; all but the drum trigger re-post to Tk with `root.after(0, ...)` | `:3176-3275` | `_midi_activity_time` | MIDI → Tk | `synth.trigger_drum` (on MIDI) |
| CC apply via widget conversion | `:3333-3357` | - | Tk | `get_parameter_for_cc` |
| Pitch-bend temporary modulation with restore | `:3277-3331` | `_pitchbend_original_value`, `_pitchbend_active`, `_pitchbend_center_threshold` | Tk | `get_pitchbend_target` |
| CC registry: name → (widget, min, max, setter); the setter slot is always `None` | `:3359-3374`, `:3539-3598` | `_cc_parameter_registry` | Tk | - |
| Learn / cancel / remove / assign pitch bend / persist | `:3376-3513` | `_midi_learn_target`, `_midi_learn_original_bg`, `_midi_learn_flash_id` | Tk (learn result arrives from MIDI via `after`) | `start_midi_learn`, `stop_midi_learn`, `add/remove_cc_mapping`, `get_cc_for_parameter`, `set_pitchbend_target`, `get_cc_name` |
| MIDI Settings and CC Mappings apply logic (reconnect, persist) | `:4541-4585`, `:4749-4771` | - | Tk | `disconnect`, `connect`, `auto_connect`, `clear_cc_mappings`, `add_cc_mapping`, `get_synced_bpm`, `get_available_ports` |
| Activity LED | `:3600-3611` | - | Tk | - |

## N. Preset, drum patch and program I/O

| Behaviour | Pointer | State owned | Thread | Engine APIs |
|---|---|---|---|---|
| Save JSON preset: `synth.get_preset_data()` (`version`, `master_volume_db`, `channels`) + `patterns` (`pm.to_dict()`) + `tempo`, `step_rate`, `swing`, `fill_rate`, `morph`, `programs`; records a recent file and refreshes the list. Mutes are not written. The `.json` extension is the default; the file is JSON whatever the name | `:2840-2868` | - | Tk | `get_preset_data`, `pm.to_dict`, `morph_manager.to_dict`, `get_programs_data`, `PreferencesManager.add_recent_file` |
| Load `.mtpreset`: parse, `channel.set_parameters` for 8 drums, patterns, tempo, step rate, swing, fill rate, master volume, mutes, morph endpoints re-initialised from the loaded state, morph position; updates the matching widgets | `:2970-3043` | - | Tk | `preset_manager.load_mtpreset`, `pm.load_from_preset_data`, `set_bpm`, `set_step_rate`, `set_swing`, `set_fill_rate`, `synth.set_master_volume`, `mute_channel`, `_init_endpoints` |
| Load JSON: `synth.load_preset_data`, `pm.from_dict`, tempo, step rate, swing, fill rate (`int`), morph, programs; updates BPM entry, step-rate buttons, morph slider, program combobox. Does not update the master knob, swing slider, fill-rate buttons or mutes | `:3044-3100` | - | Tk | `load_preset_data`, `pm.from_dict`, `pm.set_*`, `morph_manager.from_dict`, `load_programs_data` |
| Both load paths push undo first, then set `last_preset`, add a recent file, refresh the list; errors shown unless loaded silently (combobox, startup) | `:2966-3103` | - | Tk | `PreferencesManager.set/add_recent_file` |
| Preset folder list and selection | `:4803-4838`, `:3105-3115` | combobox values | Tk | `get_preset_folder`, `set_preset_folder` |
| Auto-load of `last_preset` at startup | `:4840-4848` | - | Tk | - |
| Drum patch load into / save from the selected channel | `:2886-2933` | - | Tk | `preset_manager.load_drum_patch`, `save_drum_patch` |
| Preset clipboard (non-working, see 1.12) | `:1723-1737` | `_preset_clipboard` | Tk | missing methods |

## O. Render and export

| Behaviour | Pointer | State owned | Thread | Engine APIs |
|---|---|---|---|---|
| MIDI file: one track, tempo meta from BPM, channel 10, notes 36, 38, 42, 46, 45, 41, 39, 37 for channels 1-8, velocity 127 on accent else 64, each step placed one quarter note apart (`ticks_per_beat` per step), note-off 10 ticks later. The time-sorted list is a local (`track = sorted(...)`) and is not written back to `mid.tracks` | `:2397-2480` | - | Tk | `pm.get_pattern`, `PatternChannel.get_step` (uses `mido`) |
| WAV pattern export: tail option dialog; renders at the synth rate through a new `StepSequencer` driving the **live** synth (`process_audio_events` in 1024-frame chunks) while temporarily setting `pm.playing_pattern_index`, `is_playing = True`, `play_position = 0`; length from `STEP_TICKS`, BPM and rate; tail none / +2 s / +1 pass; 16-bit, mono when the synth is mono; restores the two `pm` fields | `:2482-2591` | - | Tk (the audio callback keeps running on the same synth) | `StepSequencer`, `STEP_TICKS`, `synth.sample_clock`, `process_audio_events`, `synth.mono` |
| Drum WAV exports (unreachable) | `:2935-2964` | - | Tk | `export_all_drums_to_wav`, `export_drum_to_wav` |
| PO-32 FSK transfer audio | Q | | | |

## P. Preference application

| Behaviour | Pointer | State owned | Thread | Engine APIs |
|---|---|---|---|---|
| Audio "Apply Now": save all audio keys; apply output rate; apply synth rate (rebuild synth) **before** stopping the stream; apply mono; recompute block size; stop and restart the stream; update the device label | `:4083-4147` | see B | Tk | `_apply_sample_rate`, `_apply_synth_rate`, `synth.set_mono`, `_stop_audio`, `_start_audio` |
| Audio "OK": save only | `:4059-4081` | - | Tk | - |
| Supported-rate probe per device | `:3887-3962` | - | Tk | `sd.check_output_settings` |
| Smoothing apply | `:4239-4254` | - | Tk | `channel.set_smoothing_time` |
| AI settings apply + generator cache invalidation | `:4380-4389` | - | Tk | - |

## Q. Dialog-owned logic

| Behaviour | Pointer | State owned | Thread | Engine APIs |
|---|---|---|---|---|
| **PO-32 transfer:** convert each live channel to normalized codec values (mode-dependent rate/amount curves, linear-envelope ×1.5 correction, Q log mapping, mix inversion, EQ gain ±40 dB mapping), serialize 21 uint16 per patch, build TLV packet (8 left patches with muted channels silenced, 8 default right patches, default patterns, default state, trailer), bit-reverse, FSK-modulate | `po32_transfer.py:53-310`, `:633-652` | `audio_samples`, `bank`, `pattern_chain`, `mute_mask` | Tk (regenerated on every setting change) | `channel.get_parameters`, `po32_codec.ModemEncoder`, `generate_fsk_signal`, `bit_reverse_bytes` |
| Transfer playback: normalize to 0.85 peak, blocking writes of 1 s chunks to a separate mono 44.1 kHz `OutputStream`, progress polled every 100 ms, stop event, completion posted with `dialog.after(0)`; close joins the thread for up to 1 s | `po32_transfer.py:661-801`, `:834-848` | `is_transferring`, `transfer_thread`, `transfer_progress`, `_stop_event`, `_closed` | Worker → Tk | - |
| Save transfer WAV | `po32_transfer.py:803-832` | - | Tk | `po32_codec.save_wav` |
| **PO-32 import:** decode WAV or recorded samples in a worker, result posted with `dialog.after(0)` | `po32_import_dialog.py:709-730`, `:786-830`, `:832-887` | `decoded_preset` | Worker → Tk | `decode_wav_file`, `decode_audio_samples` |
| Input monitor and recording: `sd.InputStream` (mono, decoder rate) whose callback writes `_vu_level` and appends buffers; 50 ms VU timer with peak hold; optional debug WAV in `~/Documents/Pythonic Debug Recordings`; open folder with the OS file manager | `po32_import_dialog.py:481-703`, `:744-784` | `recording`, `recorded_samples`, `record_stream`, `_vu_*`, `_debug_save` | Dialog audio + Tk | - |
| Pattern selection model: max 12, auto-assign letters, swap on conflict, prefer non-empty | `po32_import_dialog.py:893-1125` | `selected_bank`, `selected_pattern_idx`, `selected_patterns`, `pattern_destinations` | Tk | `get_pattern_triggers_for_bank`, `get_pattern_summary`, `get_patch_summary` |
| Preview: builds a **separate** `PythonicSynthesizer` at the decoder rate, applies the bank's patches, opens its own `OutputStream` (block 1050) whose callback steps 16 sixteenths at `pm.bpm`, velocity 100 | `po32_import_dialog.py:1138-1251` | `_preview_stream`, `preview_playing`, `preview_stop_flag`, callback dict | Dialog audio | `PythonicSynthesizer`, `channel.set_parameters`, `trigger_drum`, `process_audio` |
| Import: `set_parameters` on channels 1-8 from the bank's left patches; morph endpoints (J); clears all 12 patterns; writes triggers of each selected pattern into its destination letter with accent/fill off, probability 100, no substeps; callback refreshes the main UI; no undo entry | `po32_import_dialog.py:1257-1375` | - | Tk | `channel.set_parameters`, `pm.get_pattern`, `Pattern.clear`, `steps[s].*` |
| **AI drum generator:** load patch/pattern models (on success the path is saved), generate candidates per slot or for all 8 with temperature, count and seed; candidate navigation; pattern bank generation (12 patterns, swing passed as 0.0) when in generate mode | `drum_generator_dialog.py:390-748` | `generator`, `pattern_gen`, `slot_state[8]`, `_pattern_mode`, `_cached_pattern_bank` | Tk (inference on Tk) | `PatchGenerator.load_model/generate`, `PatternGenerator.load_model/generate_bank/resolve_model_path`, `is_torch_available` |
| Install ML dependencies in a worker, result via `dialog.after(0)` | `drum_generator_dialog.py:532-575` | - | Worker → Tk | `install_ml_dependencies` |
| Tentative kit on the **live** synth: one-shot preview applies the candidate to the live channel and triggers it; loop preview applies all candidates, optionally swaps in the bank pattern, and starts the main transport; bank preview chains all 12 patterns and starts from A | `drum_generator_dialog.py:679-837` | `preview_playing`, `_saved_patterns` | Tk (sound via Audio) | `convert_drum_patch_data`, `apply_drum_patch_to_channel`, `trigger_drum`, `pm.apply_single_pattern`, `apply_pattern_bank`, `Pattern.copy`, `chained_to_next/from_prev` |
| Restore on stop/close: channels not applied get their saved parameters back; patterns restored unless replaced; temperatures saved | `drum_generator_dialog.py:699-714`, `:843-852`, `:920-932` | `_saved_channel_states`, `_applied_slots`, `_patterns_applied` | Tk | `channel.get_parameters/set_parameters`, `pm.patterns` |
| Apply selected / replace all patterns, then main-window callback | `drum_generator_dialog.py:858-914` | same | Tk | as above |

## R. Unreferenced legacy module

`gui/po32_import.py` (634 lines) is an earlier import dialog with fixed 12 s `sd.rec` recording in a worker thread, per-channel selection and a callback per imported channel. No module imports it.

---

# Observations for the core boundary

Facts only. No design decisions are made here.

## Coupling hot spots

1. `PythonicGUI` (`main_window.py:61-5283`) owns every service and almost all app state; app logic and widget updates sit in the same methods. Example: `_load_preset_file` (`:2966-3103`) mutates synth, patterns, morph, programs and preferences, and updates eight different widget groups along the way.
2. The two preset loaders refresh different sets of widgets (N): the `.mtpreset` path updates swing, fill rate, master and mutes; the JSON path does not, but restores programs and morph data.
3. Dialogs receive raw engine objects (`synth`, `pattern_manager`, `preferences_manager`) and mutate them directly: the drum generator writes channels and `pm.patterns`; the PO-32 import dialog writes channels, patterns and morph endpoints.
4. Widgets act as the value model for MIDI: CC and pitch bend set widget values, and the widget callback writes the engine (`:3352-3357`, `:3327-3331`). The log curves live in `widgets.py` (`_normalized_to_value`), and the registry ranges in `main_window.py` disagree with the widget ranges for 15 parameters (1.10).
5. UI-to-engine unit conversion is spread across ~50 handlers (`/100` for most, raw for LFO depth, an index table for delay time, an inverted mix slider).
6. Reach-ins to private members: `synth._current_program` (`:1489`), `synth._programs` (`:3093`), `morph_manager._position` (`:1502`), `morph_manager._init_endpoints()` (`:3029`, `:3082`), `channel._last_mod_offsets` (`:5190`), `widget._normalized_to_value` (`:3347`), `editor._draw()` (`:2200`). Ad-hoc attributes on the Tk root carry app data: `root.clipboard_data` (`:2642`), `root.morph_manager` (`:3641`).
7. Undo snapshots cover synth preset data, `pm.to_dict()` and morph. Programs, mutes and swing are outside the snapshot. `pm.from_dict` on restore also rewrites `bpm`, `fill_rate`, `step_rate` and `playing_pattern_index` without updating the BPM entry or calling `synth.set_bpm`.
8. Selection is held twice: `self.selected_channel` and `synth.select_channel()`. Pattern length is held by each of the 8 editors (`editor.pattern_length`) and by the pattern; `_update_pattern_editors` does not refresh the editors' length.
9. AI inference, model loading, MIDI/WAV export rendering and PO-32 FSK encoding all run on the Tk thread.

## Shared mutable state

| State | Written by | Read by | Guard |
|---|---|---|---|
| `pattern_manager` playback fields (`is_playing`, `play_position`, `current_step`, `playing_pattern_index`, `queued_pattern_index`, `playback_generation`) | Tk (select, queue, play/stop, MIDI continue, WAV export, undo restore), Audio (sequencer) | Audio, Tk | none |
| `pattern_manager.patterns` list and step data | Tk (edits, menus, AI, undo restore replaces the list, drum generator, PO-32 import) | Audio (sequencer) | none |
| `pm.bpm`, `swing`, `fill_rate`, `step_rate` | Tk (incl. MIDI clock via `after`) | Audio | none |
| Synth channel parameters | Tk (handlers, morph, loads, dialogs), MIDI thread (`trigger_drum`) | Audio, Synth pool | none in the GUI |
| `channel._last_mod_offsets` | Audio (render) | Tk (50 ms tick) | none |
| `current_play_position` | Audio | Tk | `position_lock` (the only lock the GUI uses) |
| `self.synth` reference | Tk (`_apply_synth_rate` replaces it, `:5083`) | Audio callback | none; the replacement happens before the stream is stopped (`:4115` vs `:4130-4131`) |
| `self.sequencer`, `_seq_generation` | Tk (set to `None`), Audio (create, restart) | Audio | none |
| `_audio_block_size`, `_buffer_time_ms`, `_last_good_audio`, `_resample_ratio` | Tk (rate changes) | Audio | none |
| Performance deques and counters | Audio | Audio | none |
| `MorphManager` position and endpoints | Tk | Tk (and the synth's morph branch if it were wired) | none |
| Preferences file | Tk (dialogs, learn callbacks posted to Tk) | Tk | none; every `set()` rewrites the file |

## Thread hand-offs

- **MIDI → Tk:** `root.after(0, ...)` is called from the MIDI thread for pattern select, transport, BPM, CC, pitch bend, activity and learn (`:3182`, `:3194`, `:3210`, `:3225`, `:3229`, `:3247`, `:3252`, `:3270`, `:3275`, `:3446`). The drum trigger is the exception: `synth.trigger_drum` runs on the MIDI thread (`:3179`).
- **Audio → Tk:** the play position goes through `position_lock` (`:4899-4900`, `:5159-5160`); pattern changes from chains and queues are detected by polling `pm.playing_pattern_index` every 50 ms (`:5163-5174`).
- **Tk → Audio:** plain attribute writes on `pm` and `synth`; a sequencer restart is signalled by `pm.playback_generation` (`:4890-4893`) or by setting `self.sequencer = None` (`:5095`).
- **Dialog workers → Tk:** `dialog.after(0, ...)` from worker threads (`po32_transfer.py:737-743`, `po32_import_dialog.py:728`, `:828`, `drum_generator_dialog.py:573`).
- **Concurrent audio streams:** the main `OutputStream` keeps running while dialogs open their own: PO-32 transfer output (`po32_transfer.py:714`), PO-32 import preview output with a second synth (`po32_import_dialog.py:1223`), and input streams for monitoring and recording (`:581`, `:770`).
- **Live synth used outside the audio callback:** WAV pattern export renders the live synth on the Tk thread while the stream is running (`main_window.py:2554-2569`); drum generator previews write and trigger live channels from Tk (`drum_generator_dialog.py:754-767`).

## Timers on the Tk thread

| Timer | Period | Pointer |
|---|---|---|
| UI tick (playhead, chain follow, mod indicators) | 50 ms | `:5185` |
| Pattern button flash | 250 ms | `:216`, `:5256`, `:5270` |
| Channel LED flash reset, MIDI LED reset | 100 ms one-shot | `:1773`, `:2730`, `:3188`, `:3606` |
| MIDI-learn widget flash | 300 ms | `:3465` |
| PO-32 import VU meter | 50 ms | `po32_import_dialog.py:620` |
| PO-32 transfer progress | 100 ms | `po32_transfer.py:753` |

Every call to `_update_pattern_button_states` schedules a new `_toggle_button_flash` (`:5256`), and `_toggle_button_flash` reschedules itself when stopped (`:5270`) or calls `_update_pattern_button_states` when playing (`:5262`). Each call made outside the timer chain (pattern select, play, stop, chain toggle, MIDI start/continue) therefore starts one more 250 ms chain, and the chains do not end.

## Other cross-cutting facts

- `canvas.bind_all('<MouseWheel>')` in the PO-32 import dialog (`po32_import_dialog.py:132`) replaces the main window's application-wide wheel binding (`main_window.py:264`).
- The PO-32 import dialog has no `WM_DELETE_WINDOW` handler; closing the window with the title bar skips `_on_cancel` (`:1377-1385`), which is where streams are stopped.
- The PO-32 transfer dialog is not awaited (`main_window.py:3616-3621`); the import and generator dialogs are (`:3643`, `:3685`).
- `gui/__init__.py` re-exports six widget classes; `PatternEditor`, `MatrixEditor` and `CircularButton` are imported directly from `gui.widgets`.
- One test imports a GUI module: `tests/test_drum_generator.py:243` imports `DrumGeneratorDialog`.
