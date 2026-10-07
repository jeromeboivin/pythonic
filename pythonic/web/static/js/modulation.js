// Modulation indicators (pure): which address each LFO / pump target moves,
// and the offsets poll()['modulation'] reports, by address, with the colour
// source (the first of LFO 1, LFO 2, pump that is on and aims at it).

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
