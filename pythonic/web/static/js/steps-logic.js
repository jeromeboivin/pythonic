// Pure logic of the step row (map decision #8), no DOM: step and lane
// addresses, what a pad shows of a step, step edits and their local echo,
// the velocity / probability drag math, pages and follow, the states of the
// pattern buttons and chains.
//
// Steps and pattern indexes: `index` is 0-based (0..63), `step` 1-based as
// in the addresses (1..64); patterns are letters A..L or indexes 0..11.

import { dragPosition, fromPosition, toPosition } from './values.js';

export const PATTERNS = 'ABCDEFGHIJKL';
export const PAGE_SIZE = 16;
export const MAX_STEPS = 64;
export const PAGES = MAX_STEPS / PAGE_SIZE;
export const CHANNELS = [1, 2, 3, 4, 5, 6, 7, 8];

/** Step fields (address suffixes) and their defaults. */
export const FIELDS = ['trig', 'acc', 'vel', 'fill', 'prob', 'sub'];
const DEFAULTS = { trig: false, acc: false, vel: 64, fill: false, prob: 100, sub: '' };

/** Ranges of the dragged fields (as describe() reports them). */
export const FIELD_META = {
  vel: { kind: 'int', minimum: 1, maximum: 127, default: 64, unit: '', curve: 'linear' },
  prob: { kind: 'int', minimum: 0, maximum: 100, default: 100, unit: '%', curve: 'linear' },
};

/** Step modes: [field, button label, display name]. */
export const MODES = [
  ['trig', 'trig', 'trigger'], ['acc', 'accent', 'accent'], ['vel', 'velo', 'velocity'],
  ['fill', 'fill', 'fill'], ['prob', 'prob', 'probability'], ['sub', 'sub', 'substeps'],
];

/** The substep presets of the popover (tkinter's list): o plays, - rests. */
export const SUBSTEP_PRESETS = ['oo', 'o-', '-o', 'ooo', 'oo-', 'o-o', 'o--', '-oo', '--o',
  'oooo', 'o-o-', '-o-o', 'ooo-', 'o---'];

export const stepAddress = (pattern, channel, step, field) => `pattern.${pattern}.ch${channel}.step${step}.${field}`;
export const laneAddress = (pattern, channel, field) => `pattern.${pattern}.ch${channel}.${field}`;
export const lengthAddress = (pattern) => `pattern.${pattern}.length`;

/** Beat group 1..4 of a step (1-based) by its place on its page. */
export const groupOf = (step) => (((step - 1) % PAGE_SIZE) >> 2) + 1;
/** Pages a pattern of `length` steps covers. */
export const pageCount = (length) => Math.max(1, Math.ceil(length / PAGE_SIZE));
export const pageRange = (page) => `${page * PAGE_SIZE + 1}-${page * PAGE_SIZE + PAGE_SIZE}`;

/** One step from its lanes ({field: list}); past the length it reads its defaults. */
export function stepAt(lanes, index, length) {
  const out = index >= length;
  const step = { out };
  for (const f of FIELDS) {
    const lane = lanes && lanes[f];
    step[f] = !out && Array.isArray(lane) && lane[index] !== undefined && lane[index] !== null ? lane[index] : DEFAULTS[f];
  }
  return { trig: step.trig, acc: step.acc, vel: step.vel, fill: step.fill, prob: step.prob, sub: step.sub, out };
}

/**
 * What a pad shows of a step: the trigger lit at `level` (velocity as
 * brightness, 1 when accented), the accent dot, the fill stripe, the
 * probability below 100 %, the substep dots; in the velocity and probability
 * modes a bar (0..1) and the value.
 */
export function padView(step, mode) {
  if (step.out) {
    return { on: false, acc: false, fill: false, level: 0, prob: null, sub: [], bar: null, text: null, out: true };
  }
  const on = !!step.trig;
  const view = {
    on,
    acc: on && !!step.acc,
    fill: on && !!step.fill,
    level: on ? (step.acc ? 1 : 0.3 + 0.7 * (step.vel - 1) / 126) : 0,
    prob: step.prob < 100 ? `${step.prob}%` : null,
    sub: [...(step.sub || '')].map((c) => c === 'o'),
    bar: null,
    text: null,
    out: false,
  };
  if (mode === 'vel' && on) {
    view.bar = step.acc ? 1 : step.vel / 127;
    view.text = step.acc ? 'ACC' : String(step.vel);
  } else if (mode === 'prob') {
    view.bar = step.prob / 100;
    view.text = `${step.prob}%`;
  }
  return view;
}

/**
 * The lanes a step edit changes, with the new step in them (the local echo):
 * {field: new list}; turning a trigger off also clears its accent and fill.
 * Steps past the lane (the length) change nothing.
 */
export function withStep(lanes, index, field, value) {
  const lane = lanes && lanes[field];
  if (!Array.isArray(lane) || index >= lane.length) return {};
  const put = (f, v) => {
    const list = [...(out[f] || lanes[f] || [])];
    if (index < list.length) list[index] = v;
    out[f] = list;
  };
  const out = {};
  put(field, value);
  if (field === 'trig' && !value) {
    if (Array.isArray(lanes.acc)) put('acc', false);
    if (Array.isArray(lanes.fill)) put('fill', false);
  }
  return out;
}

/**
 * The channels an edit of `field` at `index` reaches: the channel, or all 8
 * with all ch (muted ones included); accent and fill only where the step is
 * triggered. byChannel = {channel: lanes}.
 */
export function editChannels(field, channel, index, allChannels, byChannel) {
  const channels = allChannels ? CHANNELS : [channel];
  if (field !== 'acc' && field !== 'fill') return [...channels];
  return channels.filter((ch) => {
    const trig = byChannel[ch] && byChannel[ch].trig;
    return Array.isArray(trig) && !!trig[index];
  });
}

/** The value of a velocity or probability drag of `pixels` design px up (#9: 200 px = range). */
export function dragValue(field, start, pixels, fine = false) {
  const meta = FIELD_META[field];
  return fromPosition(meta, dragPosition(toPosition(meta, start), pixels, { fine }));
}

/** A typed substep pattern: o (plays) and - (rests) only, lower case. */
export const cleanSubsteps = (text) => [...String(text || '').toLowerCase()].filter((c) => c === 'o' || c === '-').join('');

/** The 0-based step the playhead is on in the selected pattern, or -1. */
export function playheadIndex(transport, selected) {
  if (!transport || !transport.playing || transport.playing_pattern !== selected) return -1;
  return transport.position || 0;
}

/** The page follow shows (the playhead's), or null when nothing plays there. */
export function followPage(transport, selected) {
  const i = playheadIndex(transport, selected);
  return i < 0 ? null : Math.floor(i / PAGE_SIZE);
}

/** States of the 12 pattern buttons from the transport and the empty / chained flags. */
export function patternStates(transport, empty = [], chained = []) {
  const t = transport || {};
  return [...PATTERNS].map((letter, i) => ({
    letter,
    index: i,
    selected: t.selected_pattern === i,
    playing: !!t.playing && t.playing_pattern === i,
    queued: !!t.playing && t.queued_pattern === i,
    empty: !!empty[i],
    chainIn: i > 0 && !!chained[i - 1],
    chainOut: !!chained[i],
  }));
}

/** Indexes of the chain a pattern is in (just itself when unchained). */
export function chainOf(index, chained) {
  let first = index;
  while (first > 0 && chained[first - 1]) first -= 1;
  let last = index;
  while (last < PATTERNS.length - 1 && chained[last]) last += 1;
  return Array.from({ length: last - first + 1 }, (_, k) => first + k);
}

export const chainText = (indexes) => (indexes.length > 1 ? indexes.map((i) => PATTERNS[i]).join('-') : 'off');
