// Modulation (pure): which address each LFO / pump target moves, the offsets
// poll()['modulation'] reports, by address, with the colour source (the first
// of LFO 1, LFO 2, pump that is on and aims at it), the band a modulated
// control draws, and the destinations click to assign may pick (map #11).

import { isNumeric, toPosition } from './values.js';

/** Mod target (engine name) -> address suffix; global targets are full addresses. */
export const MOD_TARGETS = {
  osc_frequency: 'osc.freq',
  osc_attack: 'osc.attack',
  osc_decay: 'osc.decay',
  pitch_mod_amount: 'osc.mod_amount',
  pitch_mod_rate: 'osc.mod_rate',
  pitch_semitones: 'osc.pitch',
  noise_filter_freq: 'noise.freq',
  noise_filter_q: 'noise.q',
  noise_attack: 'noise.attack',
  noise_decay: 'noise.decay',
  osc_noise_mix: 'mix.osc_noise',
  level_db: 'mix.level',
  pan: 'mix.pan',
  distortion: 'mix.distortion',
  eq_frequency: 'eq.freq',
  eq_gain_db: 'eq.gain',
  vintage_amount: 'fx.vintage',
  reverb_decay: 'fx.reverb_decay',
  reverb_mix: 'fx.reverb_mix',
  reverb_width: 'fx.reverb_width',
  delay_feedback: 'fx.delay_feedback',
  delay_mix: 'fx.delay_mix',
  osc_vel_sensitivity: 'vel.osc',
  noise_vel_sensitivity: 'vel.noise',
  mod_vel_sensitivity: 'vel.mod',
};

/** Global mod targets -> their addresses (any channel may aim at them). */
export const GLOBAL_MOD_TARGETS = { master_volume: 'global.master', morph: 'morph.position' };

/** The modulation sources of a channel, in colour priority order. */
export const MOD_SOURCES = ['lfo1', 'lfo2', 'pump'];

/** Display names of the sources. */
export const SOURCE_NAMES = { lfo1: 'LFO 1', lfo2: 'LFO 2', pump: 'PUMP' };

/** The engine's targets in enum order (describe() labels of chN.lfo1.target). */
export const TARGET_LABELS = [
  'none', 'osc_frequency', 'osc_attack', 'osc_decay', 'pitch_mod_amount', 'pitch_mod_rate',
  'pitch_semitones', 'noise_filter_freq', 'noise_filter_q', 'noise_attack', 'noise_decay',
  'osc_noise_mix', 'level_db', 'pan', 'distortion', 'eq_frequency', 'eq_gain_db', 'vintage_amount',
  'reverb_decay', 'reverb_mix', 'reverb_width', 'delay_feedback', 'delay_mix', 'osc_vel_sensitivity',
  'noise_vel_sensitivity', 'mod_vel_sensitivity', 'master_volume', 'morph',
];

/** Short names of the targets (the → buttons and the display). */
const TARGET_NAMES = {
  none: 'off', osc_frequency: 'osc freq', osc_attack: 'osc attack', osc_decay: 'osc decay',
  pitch_mod_amount: 'pitch amt', pitch_mod_rate: 'pitch rate', pitch_semitones: 'tune',
  noise_filter_freq: 'noise freq', noise_filter_q: 'noise q', noise_attack: 'noise atk',
  noise_decay: 'noise decay', osc_noise_mix: 'osc/noise', level_db: 'level', pan: 'pan',
  distortion: 'distort', eq_frequency: 'eq freq', eq_gain_db: 'eq gain', vintage_amount: 'vintage',
  reverb_decay: 'rvb time', reverb_mix: 'rvb mix', reverb_width: 'rvb wide', delay_feedback: 'dly fdbk',
  delay_mix: 'dly mix', osc_vel_sensitivity: 'osc vel', noise_vel_sensitivity: 'noise vel',
  mod_vel_sensitivity: 'mod vel', master_volume: 'master', morph: 'morph',
};

/** The short name of a target (the label itself when unknown). */
export const targetName = (target) => TARGET_NAMES[target] || String(target);

const SUFFIX_TARGETS = Object.fromEntries(Object.entries(MOD_TARGETS).map(([t, s]) => [s, t]));
const ADDRESS_TARGETS = Object.fromEntries(Object.entries(GLOBAL_MOD_TARGETS).map(([t, a]) => [a, t]));

/** The address a target of channel n (1..8) moves, or null for none. */
export function targetAddress(target, channel) {
  if (target in MOD_TARGETS) return `ch${channel}.${MOD_TARGETS[target]}`;
  return GLOBAL_MOD_TARGETS[target] || null;
}

/**
 * Click to assign: the target a control's address gives a source of the
 * selected channel, `{target}`, or `{refused}` with the reason the display
 * shows. A source modulates its own channel, master or the sound morph.
 */
export function destinationOf(address, selectedChannel) {
  if (address && address in ADDRESS_TARGETS) return { target: ADDRESS_TARGETS[address] };
  const m = /^ch([1-8])\.(.+)$/.exec(address || '');
  if (!m || !(m[2] in SUFFIX_TARGETS)) return { refused: 'not a destination' };
  if (Number(m[1]) !== selectedChannel) return { refused: `not on CH${selectedChannel}` };
  return { target: SUFFIX_TARGETS[m[2]] };
}

/**
 * The modulation band of a control: [from, to] positions (0..1 along the
 * address's curve) from the set value to the modulated one, or null when
 * nothing moves. The modulated end stops at the ends of the range.
 */
export function modulationBand(meta, value, offset) {
  if (!isNumeric(meta) || !offset) return null;
  const v = Number(value);
  if (!Number.isFinite(v)) return null;
  return [toPosition(meta, v), toPosition(meta, v + offset)];
}

/** The addresses that tell which source of channel n (1..8) aims where. */
export function sourceAddresses(channel) {
  return MOD_SOURCES.flatMap((s) => [`ch${channel}.${s}.on`, `ch${channel}.${s}.target`]);
}

function sourceOf(channel, target, value) {
  for (const s of MOD_SOURCES) {
    if (value(`ch${channel}.${s}.on`) && value(`ch${channel}.${s}.target`) === target) return s;
  }
  return null;
}

/**
 * {address: {offset, source}} of everything modulated now. `readout` is
 * poll()['modulation'] ({channel, offsets, channels?}); `value(address)` reads
 * the store. Offsets are in the address's units; global targets sum over
 * channels and take the first channel's source.
 */
export function modulatedAddresses(readout, value) {
  const out = {};
  if (!readout) return out;
  const channels = readout.channels
    || Object.assign(Array(8).fill(null), { [readout.channel ?? 0]: readout.offsets || {} });
  channels.forEach((offsets, index) => {
    if (!offsets) return;
    const channel = index + 1;
    for (const [target, offset] of Object.entries(offsets)) {
      if (!offset) continue;
      const source = sourceOf(channel, target, value) || 'lfo1';
      if (target in GLOBAL_MOD_TARGETS) {
        const address = GLOBAL_MOD_TARGETS[target];
        if (out[address]) out[address].offset += offset;
        else out[address] = { offset, source };
      } else if (target in MOD_TARGETS) {
        out[`ch${channel}.${MOD_TARGETS[target]}`] = { offset, source };
      }
    }
  });
  return out;
}
