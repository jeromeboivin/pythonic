// Values of addresses as controls see them, from describe() metadata
// (pure, no DOM): the 0..1 position along the curve (the core's
// Address.normalize / denormalize), drag and wheel steps, the readout text in
// engine units and the parsing of a typed value.
//
// meta = { kind, minimum, maximum, default, unit, curve, labels }

const clamp01 = (n) => Math.min(1, Math.max(0, n));

/** True for kinds a knob or fader can show. */
export const isNumeric = (meta) => !!meta && (meta.kind === 'float' || meta.kind === 'int');

// A log range starting at 0 starts its curve at 1e-5 of the maximum
const logFloor = (meta) => (meta.minimum > 0 ? meta.minimum : meta.maximum * 1e-5);

/** The value clamped to the range (ints rounded, enums by name or index). */
export function coerce(meta, value) {
  if (!meta) return value;
  switch (meta.kind) {
    case 'bool': return !!value;
    case 'enum': {
      const labels = meta.labels || [];
      if (typeof value === 'number' && Number.isInteger(value)) {
        return labels[Math.min(labels.length - 1, Math.max(0, value))];
      }
      return labels.includes(value) ? value : labels[0];
    }
    case 'int':
    case 'float': {
      let v = Number(value);
      if (!Number.isFinite(v)) v = meta.default ?? meta.minimum ?? 0;
      if (meta.minimum != null) v = Math.max(meta.minimum, v);
      if (meta.maximum != null) v = Math.min(meta.maximum, v);
      return meta.kind === 'int' ? Math.round(v) : v;
    }
    default: return value;
  }
}

/** The 0..1 position of a value along the address's curve. */
export function toPosition(meta, value) {
  if (!meta) return 0;
  if (meta.kind === 'bool') return value ? 1 : 0;
  if (meta.kind === 'enum') {
    const labels = meta.labels || [];
    return labels.length < 2 ? 0 : Math.max(0, labels.indexOf(value)) / (labels.length - 1);
  }
  const lo = meta.minimum;
  const hi = meta.maximum;
  const v = Number(value);
  if (!Number.isFinite(v) || hi === lo) return 0;
  if (meta.curve === 'log') {
    const floor = logFloor(meta);
    return clamp01(Math.log(Math.max(v, floor) / floor) / Math.log(hi / floor));
  }
  return clamp01((v - lo) / (hi - lo));
}

/** The value at a 0..1 position (the inverse of toPosition). */
export function fromPosition(meta, position) {
  const n = clamp01(Number(position) || 0);
  if (meta.kind === 'bool') return n >= 0.5;
  if (meta.kind === 'enum') {
    const labels = meta.labels || [];
    return labels[Math.round(n * (labels.length - 1))];
  }
  const lo = meta.minimum;
  const hi = meta.maximum;
  if (n >= 1) return coerce(meta, hi);
  if (meta.curve === 'log') {
    if (n <= 0) return coerce(meta, lo);
    const floor = logFloor(meta);
    return coerce(meta, floor * (hi / floor) ** n);
  }
  return coerce(meta, lo + n * (hi - lo));
}

/** Design pixels of vertical drag that cover a knob's full range. */
export const DRAG_RANGE_PX = 200;
/** Shift: fine adjust. */
export const FINE = 0.1;
/** Wheel notch: 1 % of the range on knobs, 2 % on faders. */
export const WHEEL_KNOB = 0.01;
export const WHEEL_FADER = 0.02;

/**
 * The position after a vertical drag of `pixels` design pixels (up is
 * positive) from `start`: `length` pixels cover the range, x0.1 when fine.
 */
export function dragPosition(start, pixels, { length = DRAG_RANGE_PX, fine = false } = {}) {
  return clamp01(start + (pixels / length) * (fine ? FINE : 1));
}

/**
 * The value after `notches` wheel notches (up is positive): enums move one
 * option per notch; numbers move `fraction` of the range along the curve, or
 * `step` engine units when given; an int moves at least one unit.
 */
export function wheelValue(meta, value, notches, { fraction = WHEEL_KNOB, step = null } = {}) {
  if (!notches) return value;
  if (meta.kind === 'enum') {
    const labels = meta.labels || [];
    const i = Math.max(0, labels.indexOf(value));
    return labels[Math.min(labels.length - 1, Math.max(0, i + notches))];
  }
  if (meta.kind === 'bool') return notches > 0;
  if (!isNumeric(meta)) return value;
  let next = step != null
    ? coerce(meta, Number(value) + notches * step)
    : fromPosition(meta, toPosition(meta, value) + notches * fraction);
  if (meta.kind === 'int' && next === value) next = coerce(meta, value + Math.sign(notches));
  return next;
}

/** The value where a fader track was clicked: fraction 0 at the bottom, 1 at the top. */
export const trackValue = (meta, fraction) => fromPosition(meta, fraction);

// ---------------------------------------------------------------- readout

const signed = (v, digits) => `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v).toFixed(digits)}`;

function plain(v) {
  const a = Math.abs(v);
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  return v.toFixed(2);
}

/** The readout of a value in engine units (Hz/kHz, ms/s, dB, st, L/C/R, %). */
export function formatValue(meta, value) {
  if (value === undefined || value === null) return '—';
  if (!meta) return String(value);
  switch (meta.kind) {
    case 'bool': return value ? 'on' : 'off';
    case 'enum':
    case 'str': return String(value);
    case 'int':
    case 'float': break;
    default: return String(value);
  }
  const v = Number(value);
  // The core's infinity reaches the page as the largest float (a pitch rate of inf)
  if (Math.abs(v) >= 1e300) return v > 0 ? '∞' : '−∞';
  switch (meta.unit) {
    case 'Hz':
      if (v >= 10000) return `${(v / 1000).toFixed(1)} kHz`;
      if (v >= 1000) return `${(v / 1000).toFixed(2)} kHz`;
      return `${v < 10 ? v.toFixed(2) : v.toFixed(0)} Hz`;
    case 'ms':
      if (v >= 1000) return `${(v / 1000).toFixed(2)} s`;
      return `${v < 10 ? v.toFixed(1) : v.toFixed(0)} ms`;
    case 'dB': return `${signed(v, 1)} dB`;
    case 'st': return `${signed(v, 1)} st`;
    case 'pan': return Math.abs(v) < 0.5 ? 'C' : `${v < 0 ? 'L' : 'R'}${Math.abs(v).toFixed(0)}`;
    case 'ratio': return `${(v * 100).toFixed(0)} %`;
    case '%': return `${v.toFixed(0)} %`;
    case 'BPM': return `${meta.kind === 'int' ? v.toFixed(0) : v.toFixed(1)} BPM`;
    case 'x': return `${v.toFixed(0)}x`;
    default: return meta.kind === 'int' ? v.toFixed(0) : plain(v);
  }
}

/** The text an exact-value field starts with (the readout's number). */
export function editText(meta, value) {
  if (meta.kind === 'enum' || meta.kind === 'str') return String(value ?? '');
  const v = Number(value);
  if (meta.unit === 'ratio') return String(+(v * 100).toFixed(2));
  if (meta.unit === 'pan') return Math.abs(v) < 0.5 ? 'C' : `${v < 0 ? 'L' : 'R'}${+Math.abs(v).toFixed(1)}`;
  return String(+v.toFixed(meta.kind === 'int' ? 0 : 3));
}

/**
 * The value of a typed text in engine units, or null when it is not one:
 * units may follow the number ("1.2 kHz", "1.5 s", "-6 dB"), ratios are typed
 * in percent, pans as L30 / C / R30, enums by name or position (1-based).
 */
export function parseValue(meta, text) {
  const raw = String(text ?? '').trim().replace(/[−–]/g, '-');
  if (!raw) return null;
  if (meta.kind === 'enum') {
    const labels = meta.labels || [];
    const hit = labels.find((l) => l.toLowerCase() === raw.toLowerCase());
    if (hit !== undefined) return hit;
    const i = Number(raw);
    return Number.isInteger(i) && i >= 1 && i <= labels.length ? labels[i - 1] : null;
  }
  if (meta.kind === 'bool') {
    if (/^(on|1|true|yes)$/i.test(raw)) return true;
    if (/^(off|0|false|no)$/i.test(raw)) return false;
    return null;
  }
  if (!isNumeric(meta)) return null;
  if (meta.unit === 'pan') {
    const m = /^([LRC])\s*(\d*\.?\d*)$/i.exec(raw);
    if (m) {
      const side = m[1].toUpperCase();
      if (side === 'C') return 0;
      const n = Number(m[2] || 0);
      return coerce(meta, side === 'L' ? -n : n);
    }
  }
  const m = /^([-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s*([a-z%]*)$/i.exec(raw);
  if (!m) return null;
  let v = Number(m[1]);
  const suffix = m[2].toLowerCase();
  if (suffix === 'k' || suffix === 'khz') v *= 1000;
  else if (suffix === 's' && meta.unit === 'ms') v *= 1000;
  else if (meta.unit === 'ratio') v /= 100;
  return coerce(meta, v);
}

/** The display text of an undo / redo step's label: the name of the control
 * bound to it (nameOf(address) -> name | null), else the label in words. */
export function undoText(label, nameOf = () => null) {
  const text = String(label || '');
  return (nameOf(text) || text.replace(/\./g, ' ')).toUpperCase();
}
