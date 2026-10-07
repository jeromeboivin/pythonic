// Pure logic behind the AI drum generator page (ai-page.js): what a lane, a
// model line and the pattern bank show, the leave question, the candidates
// count, the seed and the preview arguments. No DOM.

export const CHANNELS = [1, 2, 3, 4, 5, 6, 7, 8];
export const LANE_FIELDS = ['type', 'candidates', 'candidate', 'name', 'trying', 'generating', 'error'];
export const MIN_CANDIDATES = 1;
export const MAX_CANDIDATES = 32;
export const DEFAULT_CANDIDATES = 8;
export const MAX_SEED = 2 ** 31 - 1;

/** The address of a lane field: ai.ch<N>.<field>. */
export const laneAddress = (channel, field) => `ai.ch${channel}.${field}`;

/** Every address the page reads. */
export function aiAddresses() {
  return [
    'ai.available', 'ai.install_command', 'ai.installing', 'ai.state', 'ai.models', 'ai.tried',
    'ai.bank', 'ai.preview', 'pref.ai.patch_model', 'pref.ai.pattern_model',
    'pref.ai.patch_temperature', 'pref.ai.pattern_temperature',
    ...CHANNELS.flatMap((n) => LANE_FIELDS.map((f) => laneAddress(n, f))),
  ];
}

/**
 * What a lane shows, from its fields ({candidates, candidate, name, trying,
 * generating, error}): state 'generating' | 'ready' | 'error' | 'empty', the
 * ‹ i/n › counter, the candidate name, the try button and a note.
 */
export function laneView(lane = {}) {
  const n = Number(lane.candidates) || 0;
  const i = Number(lane.candidate) || 0;
  const trying = !!lane.trying;
  const ready = n > 0;
  let state = ready ? 'ready' : 'empty';
  if (lane.generating) state = 'generating';
  else if (lane.error && !ready) state = 'error';
  let note = 'no candidates';
  if (state === 'generating') note = 'generating…';
  else if (state === 'error') note = `error: ${lane.error}`;
  else if (ready) note = trying ? 'the face plays this' : 'the face plays its sound';
  return {
    state,
    counter: ready ? `${i}/${n}` : '–/–',
    name: ready ? String(lane.name || '') : '',
    canStep: ready && !lane.generating,
    canTry: ready && !lane.generating,
    tryLabel: trying ? 'trying ✓' : 'try',
    trying,
    note,
  };
}

/** The last part of a path ('' for none). */
export function fileName(path) {
  if (!path) return '';
  return String(path).split(/[\\/]/).filter(Boolean).pop() || '';
}

/** A model line: {text, tone: 'ok' | 'busy' | 'error' | 'dim'} from ai.models[kind]. */
export function modelView(model, available = true) {
  if (!available) return { text: 'needs the ML extras', tone: 'error' };
  if (!model) return { text: '…', tone: 'dim' };
  const file = fileName(model.path);
  switch (model.status) {
    case 'loaded': return { text: `${file} ✓${model.sampling ? ` (${model.sampling})` : ''}`, tone: 'ok' };
    case 'loading': return { text: `loading ${file}…`, tone: 'busy' };
    case 'error': return { text: `error: ${model.error || file}`, tone: 'error' };
    case 'missing': return { text: 'no model: load one', tone: 'error' };
    default: return { text: `${file} (not loaded)`, tone: 'dim' };
  }
}

/** What the header says for ai.state ('' when idle). */
export function stateText(state) {
  return {
    generating: 'generating…', loading: 'loading a model…', installing: 'installing the ML extras…',
    unavailable: 'ML extras missing',
  }[state] || '';
}

/** The bank note of the right column: pattern mode 'keep' | 'generate', ai.bank. */
export function bankText(mode, bank) {
  if (mode !== 'generate') return 'the preset’s patterns stay';
  if (bank === 'ready') return 'bank of 12 ready';
  if (bank === 'generating') return 'generating patterns…';
  return '(generate all 8 first)';
}

/** Whether replace patterns can run: generate mode with a bank ready (tkinter). */
export const canReplacePatterns = (mode, bank) => mode === 'generate' && bank === 'ready';

/** The ai.pattern_try arguments of a ▶ loop / ▶ bank click: a second click stops. */
export function previewArgs(current, mode, patternMode) {
  if (current === mode) return { mode: null };
  return { mode, bank: patternMode === 'generate' };
}

/** The candidates count clamped to 1..32 (whole numbers). */
export function clampCandidates(n) {
  const v = Math.round(Number(n));
  if (!Number.isFinite(v)) return DEFAULT_CANDIDATES;
  return Math.min(MAX_CANDIDATES, Math.max(MIN_CANDIDATES, v));
}

/** The seed typed in the field: a whole number 0..2^31-1, or null (random). */
export function parseSeed(text) {
  const s = String(text ?? '').trim();
  if (!/^\d+$/.test(s)) return null;
  const v = Number(s);
  return v <= MAX_SEED ? v : null;
}

/** A new random seed (reseed), as tkinter: 0..2^31-1. */
export const randomSeed = (rand = Math.random) => Math.floor(rand() * (MAX_SEED + 1));

/**
 * The keep-or-revert question when leaving with tried sounds: tried is
 * ai.tried (channels 1..8), names maps a channel to its tried candidate name.
 */
export function leaveQuestion(tried, names = {}) {
  const list = (tried || []).map((ch) => (names[ch] ? `CH${ch} ${names[ch]}` : `CH${ch}`));
  const which = list.length === 1 ? `${list[0]} is` : `${list.join(', ')} are`;
  return {
    title: 'Keep the tried sounds?',
    text: `${which} trying AI candidates. Keep them (one undo step), or revert to the old sounds.`,
  };
}

/** The display's lines after ai.keep / ai.replace_patterns / ai.revert. */
export function keptText(result) {
  const kept = (result && result.kept) || [];
  if (!kept.length) return 'nothing tried';
  return `kept ch ${kept.join(', ')}`;
}
