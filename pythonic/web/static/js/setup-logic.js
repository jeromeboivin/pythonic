// Pure logic of the setup sheet (setup.js; map decisions #13, #14, #22): its
// tabs, note names, the CC map as rows and its edits, the names and menu
// groups of the controls a CC can drive, the "restart audio" fields and the
// readouts of the audio stream. No DOM, no bridge.

import { FACE_ONLY, rackControls } from './rack-layout.js';

export const SETUP_TABS = ['audio', 'midi', 'synthesis', 'ai'];

/** Sheet width per tab: audio and midi have two columns, the others one. */
export const TAB_WIDTHS = { audio: 820, midi: 900, synthesis: 520, ai: 600 };

/** The tab an open request asks for (SETUP opens on audio). */
export const tabOf = (tab) => (SETUP_TABS.includes(tab) ? tab : 'audio');

/** Stream settings that wait for "restart audio" (pref.audio.pending lists them). */
export const STREAM_FIELDS = ['pref.audio.device', 'pref.audio.buffer_ms', 'pref.audio.sample_rate',
  'pref.audio.synth_rate'];

/** The stream fields waiting for a restart, as a Set (any stored value). */
export const pendingFields = (pending) => new Set(Array.isArray(pending)
  ? pending.filter((a) => STREAM_FIELDS.includes(a)) : []);

// ---------------------------------------------------------------- notes

const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

/** A MIDI note's name, as the core prints it: 36 -> C2, 0 -> C-1. */
export function noteName(note) {
  const n = Math.round(Number(note));
  if (!Number.isFinite(n) || n < 0 || n > 127) return '—';
  return `${NOTE_NAMES[n % 12]}${Math.floor(n / 12) - 1}`;
}

/** The notes channels 1-8 respond to from a base note: "C2–G2 (36–43)". */
export function baseNoteRange(base) {
  const n = Math.round(Number(base));
  if (!Number.isFinite(n)) return '';
  return `${noteName(n)}–${noteName(n + 7)} (${n}–${n + 7})`;
}

/** The base note menu: the C of every octave the core accepts (0..120). */
export const BASE_NOTE_CHOICES = Array.from({ length: 11 }, (_, i) => i * 12);

// ---------------------------------------------------------------- the CC map

/** The CC map ({cc: target}, keys may be strings) as rows sorted by CC. */
export function ccRows(map) {
  return Object.entries(map || {})
    .map(([cc, target]) => ({ cc: Number(cc), target }))
    .filter((r) => Number.isInteger(r.cc))
    .sort((a, b) => a.cc - b.cc);
}

/** A CC number typed or stepped into a row: an integer 0..127, else null. */
export function parseCc(text) {
  const raw = String(text ?? '').trim().replace(/^cc\s*/i, '');
  if (!/^\d{1,3}$/.test(raw)) return null;
  const n = Number(raw);
  return n <= 127 ? n : null;
}

/** A copy of the map with number keys. */
function copy(map) {
  return Object.fromEntries(ccRows(map).map((r) => [r.cc, r.target]));
}

/**
 * The map with a row written: `cc` drives `target`, the row's old CC
 * (`from`, null for a new row) is dropped. A CC another row held moves to
 * this row: `replaced` names that row's old target (null if none).
 */
export function withRow(map, from, cc, target) {
  const next = copy(map);
  if (from !== null && from !== undefined && from !== cc) delete next[from];
  const old = next[cc];
  const replaced = old !== undefined && from !== cc ? old : null;
  next[cc] = target;
  return { map: next, replaced };
}

/** The map without one CC. */
export function withoutCc(map, cc) {
  const next = copy(map);
  delete next[cc];
  return next;
}

/** A free CC for a new row: the first unused of the usual controller
 * numbers (as tkinter's list), then any unused one; null when all 128 are used. */
export function freeCc(map, taken = []) {
  const used = new Set([...ccRows(map).map((r) => r.cc), ...taken]);
  const usual = [1, 2, 4, 7, 10, 11, 12, 13, 16, 17, 18, 19, 71, 74];
  for (const cc of usual) if (!used.has(cc)) return cc;
  for (let cc = 0; cc <= 127; cc += 1) if (!used.has(cc)) return cc;
  return null;
}

// ---------------------------------------------------------------- targets

const SELECTED = 'selected.';
const SECTION_TITLES = { osc: 'oscillator', noise: 'noise', env: 'envelopes', mix: 'mix', vel: 'velocity',
  fx: 'fx', mod: 'modulation' };
const FACE_NAMES = { 'osc.pitch': 'tune', 'osc.decay': 'osc decay', 'mix.level': 'level' };

/** Global controls a CC can drive: [address, name]. */
export const GLOBAL_TARGETS = [
  ['global.tempo', 'tempo'], ['global.swing', 'swing'], ['global.master', 'master'],
  ['morph.position', 'sound morph'], ['global.fill_rate', 'fill rate'], ['global.step_rate', 'step rate'],
];

function soundNames() {
  const names = { ...FACE_NAMES };
  for (const c of rackControls()) if (!names[c.s]) names[c.s] = c.name || c.label || c.s;
  return names;
}
const SOUND_NAMES = soundNames();

/** The sound suffixes of the selected channel a CC can drive, in rack order
 * (the → destination buttons left out), grouped by section. */
export function targetGroups() {
  const sections = [
    ['face strip', FACE_ONLY],
    ...['osc', 'noise', 'env', 'mix', 'vel', 'fx'].map((key) => [SECTION_TITLES[key],
      rackControls().filter((c) => sectionOf(c.s) === key && c.type !== 'target').map((c) => c.s)]),
    ['lfo 1', rackControls().filter((c) => c.s.startsWith('lfo1.') && c.type !== 'target').map((c) => c.s)],
    ['lfo 2', rackControls().filter((c) => c.s.startsWith('lfo2.') && c.type !== 'target').map((c) => c.s)],
    ['pump', rackControls().filter((c) => c.s.startsWith('pump.') && c.type !== 'target').map((c) => c.s)],
  ];
  return [
    ...sections.map(([title, suffixes]) => [title, suffixes.map((s) => [SELECTED + s, SOUND_NAMES[s] || s])]),
    ['global', GLOBAL_TARGETS],
  ];
}

const SECTION_OF = {
  'osc.wave': 'osc', 'osc.mod_mode': 'osc', 'osc.freq': 'osc', 'osc.mod_amount': 'osc', 'osc.mod_rate': 'osc',
  'noise.filter': 'noise', 'noise.stereo': 'noise', 'noise.freq': 'noise', 'noise.q': 'noise',
  'noise.env': 'env', 'osc.attack': 'env', 'noise.attack': 'env', 'noise.decay': 'env',
};
function sectionOf(suffix) {
  if (SECTION_OF[suffix]) return SECTION_OF[suffix];
  const head = suffix.split('.')[0];
  if (head === 'mix' || head === 'eq') return 'mix';
  return head;
}

/** A CC target's name: "osc freq" (the selected channel's), "CH3 osc freq", "tempo". */
export function targetLabel(target) {
  if (!target) return '(none)';
  if (target.startsWith(SELECTED)) {
    const s = target.slice(SELECTED.length);
    return SOUND_NAMES[s] || s;
  }
  const m = /^ch([1-8])\.(.+)$/.exec(target);
  if (m) return `CH${m[1]} ${SOUND_NAMES[m[2]] || m[2]}`;
  const g = GLOBAL_TARGETS.find(([a]) => a === target);
  return g ? g[1] : target;
}

/** "selected ch" for a target that follows the selected channel. */
export const followsChannel = (target) => typeof target === 'string' && target.startsWith(SELECTED);

// ---------------------------------------------------------------- CC activity

/** What poll()['midi'].pickup says of a CC: {physical, count} (summed over the
 * controls it drives), or null when it has not moved one yet. */
export function ccActivity(pickup, cc) {
  let found = null;
  for (const state of Object.values(pickup || {})) {
    if (!state || Number(state.cc) !== cc) continue;
    found = found || { physical: null, count: 0 };
    found.count += Number(state.count) || 0;
    if (state.physical !== null && state.physical !== undefined) found.physical = state.physical;
  }
  return found;
}

// ---------------------------------------------------------------- audio

/** A sample rate in a menu: "44100 Hz"; synth rate 0 is "same as output". */
export function rateText(rate, { synth = false } = {}) {
  if (synth && Number(rate) === 0) return 'same as output';
  return `${Number(rate)} Hz`;
}

/** A buffer size in a menu: "23.8 ms". */
export const bufferText = (ms) => `${+Number(ms).toFixed(1)} ms`;

/** The smallest buffer the frame-budget measurements found safe (issue #10). */
export const SAFE_BUFFER_FRAMES = 512;

/** The frames a buffer in ms makes at a rate (as the core rounds it). */
export const bufferFrames = (ms, rate) => Math.max(64, Math.round((Number(ms) / 1000) * Number(rate)));

/** The warning under the buffer when it is below the safe size, else ''. */
export function bufferWarning(ms, rate) {
  if (!(Number(ms) > 0) || !(Number(rate) > 0)) return '';
  const frames = bufferFrames(ms, rate);
  if (frames >= SAFE_BUFFER_FRAMES) return '';
  return `${frames} frames at ${Number(rate)} Hz is under the safe ${SAFE_BUFFER_FRAMES}: expect dropouts`;
}

/** The note under the sample rate: how many of the menu's rates the device takes. */
export function ratesNote(rates, all) {
  const n = Array.isArray(rates) ? rates.length : 0;
  const total = Array.isArray(all) ? all.length : 0;
  if (!total || !n || n >= total) return `this device takes all ${total} rates`;
  return `this device takes ${n} of ${total} rates`;
}

/** The sample rates the menu offers: the device's, plus the current one. */
export function rateChoices(all, rates, current) {
  const ok = new Set((Array.isArray(rates) && rates.length ? rates : all).map(Number));
  return all.filter((r) => ok.has(Number(r)) || Number(r) === Number(current));
}

/** A device menu: [value, text] with the system default (null) first. */
export function deviceChoices(devices, defaultText = '(system default)') {
  return [[null, defaultText], ...(Array.isArray(devices) ? devices : []).map((d) => [d, d])];
}

/** The device a choice list shows (a saved device that is gone says so). */
export function deviceText(device, devices, defaultText = '(system default)') {
  if (!device) return defaultText;
  return Array.isArray(devices) && !devices.includes(device) ? `${device} (not found)` : device;
}

/** The running stream in one line, from the audio.* addresses. */
export function streamText(v) {
  if (!v || !v['audio.running']) return 'audio stopped';
  const device = v['audio.device_is_default'] || !v['audio.device'] ? 'system default' : v['audio.device'];
  const synth = Number(v['audio.synth_rate']) !== Number(v['audio.sample_rate'])
    ? ` · synth ${v['audio.synth_rate']} Hz` : '';
  const buffer = v['audio.buffer_ms'] !== undefined && v['audio.buffer_ms'] !== null
    ? ` · ${bufferText(v['audio.buffer_ms'])}` : '';
  return `running: ${device} · ${v['audio.sample_rate']} Hz${synth}${buffer} (${v['audio.block_size']} frames)`
    + ` · ${v['audio.mono'] ? 'mono' : 'stereo'}`;
}

/** The AI model line: the chosen file's name, or the bundled checkpoint. */
export function modelText(path, model) {
  if (path) return String(path).split(/[\\/]/).pop();
  if (model && model.bundled) return 'bundled';
  if (model && model.path) return String(model.path).split(/[\\/]/).pop();
  return 'none (no bundled checkpoint)';
}
