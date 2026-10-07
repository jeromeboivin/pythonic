// Pure logic of the PO-32 page (map decisions #13, #20): which stage of each
// tab's flow is current, the pattern picks (the core's rules, for the local
// echo of a click), the level meter and the texts of the stages. No DOM.

export const LETTERS = 'ABCDEFGHIJKL';
export const MAX_PICKS = 12;
export const PATTERN_BUTTONS = 16;

/** A stage's look: 'now' (lit), 'done' (ticked) or 'later' (dimmed, still usable). */
const flow = (count, current) => Array.from({ length: count }, (_, i) => {
  const n = i + 1;
  if (n < current) return 'done';
  return n === current ? 'now' : 'later';
});

/** Transfer: choose › prepare the PO-32 › send, from `po32.transfer`. */
export function transferStages(transfer) {
  if (transfer === 'sent') return flow(3, 4);
  if (transfer === 'sending' || transfer === 'stopped' || transfer === 'error') return flow(3, 3);
  return flow(3, 1);
}

/** Import: listen › bank › pick patterns › import. */
export function importStages({ decoded = false, picks = [], imported = false } = {}) {
  if (!decoded) return flow(4, 1);
  if (imported) return flow(4, 5);
  return flow(4, picks.length ? 4 : 3);
}

/** The picks as {pattern: letter}. */
export const pickMap = (picks) => Object.fromEntries((picks || []).map((p) => [p.pattern, p.letter]));

const pickList = (map) => Object.keys(map).map(Number).sort((a, b) => a - b)
  .map((pattern) => ({ pattern, letter: map[pattern] }));

/**
 * The picks after `po32.pick(pattern, picked, letter)`, as the core does it:
 * a toggle when `picked` is null, at most 12 picks, a new pick takes the
 * first free letter, a letter another pick holds swaps the two. Returns
 * {picks, refused} (refused: a 13th pick).
 */
export function pick(picks, pattern, { picked = null, letter = null } = {}) {
  const map = pickMap(picks);
  if (letter !== null) picked = true;
  if (picked === null) picked = !(pattern in map);
  if (picked && !(pattern in map)) {
    if (Object.keys(map).length >= MAX_PICKS) return { picks: pickList(map), refused: true };
    const used = new Set(Object.values(map));
    map[pattern] = [...LETTERS].find((l) => !used.has(l));
  } else if (!picked && pattern in map) {
    delete map[pattern];
  }
  if (letter !== null) {
    const old = map[pattern];
    const holder = Object.keys(map).map(Number).find((p) => p !== pattern && map[p] === letter);
    if (holder !== undefined) map[holder] = old;
    map[pattern] = letter;
  }
  return { picks: pickList(map), refused: false };
}

/** The letters a picked pattern can move to: [{letter, holder (pattern or null), current}]. */
export function letterChoices(picks, pattern) {
  const map = pickMap(picks);
  return [...LETTERS].map((letter) => {
    const holder = Object.keys(map).map(Number).find((p) => map[p] === letter);
    return { letter, holder: holder === undefined || holder === pattern ? null : holder,
      current: map[pattern] === letter };
  });
}

/** A pattern button: {text, picked, letter, focused, empty, usable}. */
export function patternButton(number, { patterns = [], picks = [], focus = 0 } = {}) {
  const map = pickMap(picks);
  const decoded = patterns[number - 1];
  const letter = map[number] || null;
  return {
    text: letter ? `${number}→${letter}` : String(number),
    picked: !!letter,
    letter,
    focused: number === focus,
    empty: decoded ? !!decoded.empty : true,
    usable: !!decoded,
  };
}

// ---------------------------------------------------------------- level meter

/** The input level (0..1 peak) in dB, as tkinter prints it. */
export function levelText(level) {
  if (!(level > 1e-6)) return '-∞ dB';
  const db = 20 * Math.log10(level);
  return `${db >= 0 ? '+' : ''}${db.toFixed(1)} dB`;
}

/** low / good / hot / clip, tkinter's thresholds. */
export function levelTone(level) {
  if (level > 0.95) return 'clip';
  if (level > 0.5) return 'hot';
  if (level > 0.05) return 'good';
  return 'low';
}

/**
 * Peak hold of the meter, one call per frame: holds a new peak for `hold`
 * frames (750 ms at 60 frames a second), then decays it as tkinter's meter
 * does (8 % every 50 ms).
 */
export function holdPeak(state, level, { hold = 45, decay = 0.97 } = {}) {
  const s = state || { peak: 0, wait: 0 };
  if (level >= s.peak) return { peak: level, wait: hold };
  if (s.wait > 0) return { peak: s.peak, wait: s.wait - 1 };
  const peak = s.peak * decay;
  return { peak: peak < 1e-4 ? 0 : peak, wait: 0 };
}

// ---------------------------------------------------------------- texts

const range = (slots) => {
  const list = (slots || []).filter((n) => Number.isFinite(n));
  if (!list.length) return '';
  return list.length === 1 ? `pattern ${list[0]}` : `patterns ${list[0]}–${list[list.length - 1]}`;
};

/** The note under the chain choice: which PO-32 pattern slots are sent empty. */
export function slotsNote(slots) {
  const which = range(slots);
  return which ? `PO-32 ${which} ${slots.length === 1 ? 'is' : 'are'} sent empty: the transfer carries the sounds only`
    : 'the chain’s PO-32 pattern slots are sent empty: the transfer carries the sounds only';
}

/** The transfer status line. */
export function transferText({ transfer = 'none', seconds = 0, progress = 0, error = null } = {}) {
  switch (transfer) {
    case 'sending': return `sending… ${Math.round(Math.max(0, Math.min(1, progress)) * 100)} %`;
    case 'sent': return 'sent: the PO-32 has the sounds';
    case 'stopped': return 'stopped';
    case 'error': return `failed: ${error || 'the transfer stopped'}`;
    case 'ready': return `ready: ${Number(seconds || 0).toFixed(1)} s of signal`;
    default: return 'preparing the signal…';
  }
}

/** The listen stage's status line. */
export function sourceText({ decode = 'none', decoded = null, recording = false, seconds = 0, listening = false,
  error = null } = {}) {
  if (recording) return `recording ${Number(seconds || 0).toFixed(1)} s: play the PO-32 transfer now`;
  if (decode === 'decoding') return 'decoding…';
  if (decode === 'error') return error || 'the signal could not be decoded';
  if (decoded) {
    const kind = decoded.card ? 'PO-32 card' : 'Pythonic transfer';
    return `decoded ${kind}: ${decoded.drums} sounds, ${decoded.patterns} patterns · ${decoded.source}`;
  }
  return listening ? 'monitoring the input: record, then play the PO-32 transfer' : 'nothing decoded yet';
}

/** The import stage's summary. */
export function importText({ decoded = null, bank = 0, picks = [], imported = false } = {}) {
  if (!decoded) return 'Decode a PO-32 transfer first.';
  const letters = (picks || []).map((p) => p.letter).sort();
  const lands = letters.length ? `${letters.length} land on ${letters.join(' ')}` : 'none picked';
  const done = imported ? ' Imported.' : '';
  return `Bank ${bank} replaces the sounds of channels 1–8 and all 12 patterns: ${lands}, `
    + `the rest are emptied. One undo step.${done}`;
}

/** The import button's text. */
export const importButtonText = (picks) => {
  const n = (picks || []).length;
  return `import 8 sounds + ${n} pattern${n === 1 ? '' : 's'}`;
};

/** The channels sent by default: the unmuted ones (the face mutes, 1..8). */
export const unmutedChannels = (mutes) => mutes.map((m, i) => (m ? null : i + 1)).filter((n) => n !== null);
