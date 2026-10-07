// Pure logic behind the face: drum type labels, modulation by address and
// the MIDI cues of a control.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { guessDrumType } from '../js/drum-type.js';
import { ccsFor, ghostPosition, isBendTarget, withoutAddress } from '../js/midi-cues.js';
import { modulatedAddresses, sourceAddresses } from '../js/modulation.js';

test('drum types from the names real presets use', () => {
  const names = {
    '808 BD': 'BD', '808 MT': 'MT', '808 RS': 'RS', '808 CP': 'CP', '808 SD': 'SD', '808 CB': 'CB',
    '808 CH': 'CH', '808 OH': 'OH', '909 Tom Low': 'LT', '909 Tom Mid': 'MT', '909 Tom High': 'HT',
    '909 Rimshot': 'RS', '909 Clap': 'CP', 'DMX Shaker': 'SH', 'DMX Tambourine': 'TB', 'DMX Tom': 'TOM',
    '505 Conga High': 'CG', 'LM2 Cowbell': 'CB', 'LM2 Cabasa': 'SH', '707 BD 2': 'BD', 'Kick': 'BD',
    'Deep Kick': 'BD', 'Snare': 'SD', 'Closed Hat': 'CH', 'Open Hat': 'OH', 'Hi-Hat': 'HH', 'HH 1': 'HH',
    'Ride': 'RC', 'Crash': 'CC', 'Cymbal': 'CY', '808BD': 'BD', 'Zap': 'FX',
  };
  for (const [name, label] of Object.entries(names)) assert.equal(guessDrumType(name), label, name);
});

test('names without a drum type give an empty label', () => {
  for (const name of ['', null, 'Metal Ping', 'Chord', 'Bass Pluck', 'Ohm', 'Shimmer', 'Child']) {
    assert.equal(guessDrumType(name), '', String(name));
  }
});

test('modulation offsets land on channel addresses with their source', () => {
  const values = { 'ch2.lfo1.on': true, 'ch2.lfo1.target': 'osc_decay', 'ch2.pump.on': true,
    'ch2.pump.target': 'level_db', 'ch5.lfo2.on': true, 'ch5.lfo2.target': 'pitch_semitones' };
  const channels = Array(8).fill(null).map(() => ({}));
  channels[1] = { osc_decay: 40, level_db: -6 };
  channels[4] = { pitch_semitones: 2 };
  const out = modulatedAddresses({ channel: 0, offsets: {}, channels }, (a) => values[a]);
  assert.deepEqual(out, {
    'ch2.osc.decay': { offset: 40, source: 'lfo1' },
    'ch2.mix.level': { offset: -6, source: 'pump' },
    'ch5.osc.pitch': { offset: 2, source: 'lfo2' },
  });
});

test('global targets sum over channels', () => {
  const values = { 'ch1.lfo2.on': true, 'ch1.lfo2.target': 'master_volume',
    'ch3.lfo1.on': true, 'ch3.lfo1.target': 'master_volume' };
  const channels = Array(8).fill({});
  channels[0] = { master_volume: -2 };
  channels[2] = { master_volume: -1 };
  const out = modulatedAddresses({ channel: 0, offsets: {}, channels }, (a) => values[a]);
  assert.deepEqual(out, { 'global.master': { offset: -3, source: 'lfo2' } });
});

test('a readout without channels shows the selected channel', () => {
  const out = modulatedAddresses({ channel: 2, offsets: { pan: 10, delay_mix: 0 } }, () => undefined);
  assert.deepEqual(out, { 'ch3.mix.pan': { offset: 10, source: 'lfo1' } });
  assert.deepEqual(sourceAddresses(4), ['ch4.lfo1.on', 'ch4.lfo1.target', 'ch4.lfo2.on',
    'ch4.lfo2.target', 'ch4.pump.on', 'ch4.pump.target']);
});

test('CC badges come from the CC map, selected targets on the selected channel', () => {
  const map = { 21: 'ch3.osc.decay', 1: 'selected.osc.decay', 7: 'global.master', 30: 'ch3.osc.decay' };
  assert.deepEqual(ccsFor('ch3.osc.decay', map, 1), [21, 30]);
  assert.deepEqual(ccsFor('ch3.osc.decay', map, 3), [1, 21, 30]);
  assert.deepEqual(ccsFor('global.master', map, 3), [7]);
  assert.deepEqual(ccsFor('ch2.osc.pitch', map, 2), []);
  assert.deepEqual(withoutAddress('ch3.osc.decay', map, 3), { 7: 'global.master' });
  assert.ok(isBendTarget('ch4.osc.pitch', 'selected.osc.pitch', 4));
  assert.ok(!isBendTarget('ch4.osc.pitch', 'selected.osc.pitch', 2));
  assert.ok(isBendTarget('morph.position', 'morph.position', 1));
});

test('the ghost marker shows until the controller picks the value up', () => {
  assert.equal(ghostPosition(undefined, 0.5), null);
  assert.equal(ghostPosition({ physical: 0.2, linked: false }, 0.5), 0.2);
  assert.equal(ghostPosition({ physical: 0.5, linked: true }, 0.5), null);
  assert.equal(ghostPosition({ physical: 0.2, linked: true }, 0.6), 0.2); // edited since
  assert.equal(ghostPosition({ physical: null, linked: false }, 0.5), null);
});
