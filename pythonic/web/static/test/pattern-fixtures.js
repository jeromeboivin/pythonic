// Shared by the DOM specs (not a spec itself): pattern values as the core reports them.

/** Pattern addresses as the core reports them: lengths, flags and the lanes of every pattern. */
export function patternValues(length = 16) {
  const out = { 'pattern.selected': 'A' };
  const defaults = { trig: false, acc: false, vel: 64, fill: false, prob: 100, sub: '' };
  for (const p of 'ABCDEFGHIJKL') {
    out[`pattern.${p}.length`] = length;
    out[`pattern.${p}.empty`] = true;
    out[`pattern.${p}.chained`] = false;
    for (let ch = 1; ch <= 8; ch += 1) {
      for (const [f, d] of Object.entries(defaults)) out[`pattern.${p}.ch${ch}.${f}`] = Array(length).fill(d);
    }
  }
  return out;
}

/** The step addresses of some patterns (a fake bridge answers sets of known addresses only). */
export function stepValues(letters = 'AB', length = 16) {
  const out = {};
  const defaults = { trig: false, acc: false, vel: 64, fill: false, prob: 100, sub: '' };
  for (const p of letters) {
    for (let ch = 1; ch <= 8; ch += 1) {
      for (let s = 1; s <= length; s += 1) {
        for (const [f, d] of Object.entries(defaults)) out[`pattern.${p}.ch${ch}.step${s}.${f}`] = d;
      }
    }
  }
  return out;
}
