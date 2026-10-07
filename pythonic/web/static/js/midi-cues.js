// MIDI cues of a control (pure): the CCs mapped to its address (a CC map
// target is an address or `selected.<suffix>` for the selected channel), and
// where the pickup ghost marker sits.

const SELECTED = 'selected.';

/** The channel-relative suffix of a channel address (ch3.osc.decay -> osc.decay), or null. */
export function channelSuffix(address) {
  const m = /^ch([1-8])\.(.+)$/.exec(address || '');
  return m ? { channel: Number(m[1]), suffix: m[2] } : null;
}

/** CC numbers mapped to an address, sorted; `ccMap` is {cc: target} (keys may be strings). */
export function ccsFor(address, ccMap, selectedChannel) {
  if (!address || !ccMap) return [];
  const parts = channelSuffix(address);
  const out = [];
  for (const [cc, target] of Object.entries(ccMap)) {
    if (target === address) out.push(Number(cc));
    else if (parts && parts.channel === selectedChannel && target === SELECTED + parts.suffix) {
      out.push(Number(cc));
    }
  }
  return out.sort((a, b) => a - b);
}

/** The CC map without the mappings of an address. */
export function withoutAddress(address, ccMap, selectedChannel) {
  const drop = new Set(ccsFor(address, ccMap, selectedChannel).map(String));
  return Object.fromEntries(Object.entries(ccMap || {}).filter(([cc]) => !drop.has(String(cc))));
}

/** True when the pitch-bend target names this address now. */
export function isBendTarget(address, target, selectedChannel) {
  if (!target || !address) return false;
  if (target === address) return true;
  const parts = channelSuffix(address);
  return !!parts && parts.channel === selectedChannel && target === SELECTED + parts.suffix;
}

// A linked controller this close to the value is on it (CC steps, int rounding)
const ON_IT = 0.02;

/**
 * The 0..1 position of the pickup ghost marker, or null when none shows: a
 * controller has moved the control (`pickup` = poll()['midi'].pickup[address])
 * and it is not following, because it has not crossed the value yet or the
 * value changed since (`position` = the control's position now; the core
 * notices a local change only at the next CC).
 */
export function ghostPosition(pickup, position) {
  if (!pickup || pickup.physical === null || pickup.physical === undefined) return null;
  if (pickup.linked && Math.abs(pickup.physical - position) <= ON_IT) return null;
  return pickup.physical;
}
