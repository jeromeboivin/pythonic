import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  baseNoteRange, BASE_NOTE_CHOICES, ccActivity, ccRows, deviceChoices, deviceText, freeCc, modelText,
  noteName, parseCc, pendingFields, rateChoices, ratesNote, rateText, bufferText, streamText, tabOf,
  targetGroups, targetLabel, withoutCc, withRow,
} from '../js/setup-logic.js';

test('setup opens on audio unless a tab is asked for', () => {
  assert.equal(tabOf(undefined), 'audio');
  assert.equal(tabOf('midi'), 'midi');
  assert.equal(tabOf('cc'), 'audio');
});

test('note names and the eight notes of a base note', () => {
  assert.equal(noteName(36), 'C2');
  assert.equal(noteName(0), 'C-1');
  assert.equal(noteName(61), 'C#4');
  assert.equal(baseNoteRange(36), 'C2–G2 (36–43)');
  assert.deepEqual(BASE_NOTE_CHOICES.slice(0, 3), [0, 12, 24]);
  assert.equal(BASE_NOTE_CHOICES.at(-1), 120);
});

test('the CC map as rows, string keys sorted by number', () => {
  assert.deepEqual(ccRows({ 21: 'global.tempo', 2: 'selected.noise.freq', 10: 'ch3.mix.pan' }),
    [{ cc: 2, target: 'selected.noise.freq' }, { cc: 10, target: 'ch3.mix.pan' }, { cc: 21, target: 'global.tempo' }]);
  assert.deepEqual(ccRows(null), []);
});

test('typed CC numbers: 0..127 only', () => {
  assert.equal(parseCc('74'), 74);
  assert.equal(parseCc('CC 7'), 7);
  assert.equal(parseCc('0'), 0);
  assert.equal(parseCc('128'), null);
  assert.equal(parseCc('-1'), null);
  assert.equal(parseCc('x'), null);
});

test('editing a row: a new CC moves it, a CC another row held is taken over', () => {
  const map = { 1: 'selected.osc.freq', 2: 'selected.noise.freq' };
  assert.deepEqual(withRow(map, 1, 21, 'selected.osc.freq'),
    { map: { 2: 'selected.noise.freq', 21: 'selected.osc.freq' }, replaced: null });
  assert.deepEqual(withRow(map, 1, 1, 'global.tempo'),
    { map: { 1: 'global.tempo', 2: 'selected.noise.freq' }, replaced: null });
  assert.deepEqual(withRow(map, 1, 2, 'selected.osc.freq'),
    { map: { 2: 'selected.osc.freq' }, replaced: 'selected.noise.freq' });
  assert.deepEqual(withRow(map, null, 7, 'global.master').map,
    { 1: 'selected.osc.freq', 2: 'selected.noise.freq', 7: 'global.master' });
  assert.deepEqual(withoutCc({ '1': 'a', '2': 'b' }, 1), { 2: 'b' });
});

test('a new row takes the first free usual CC, then any free one', () => {
  assert.equal(freeCc({ 1: 'a', 2: 'b' }), 4);
  assert.equal(freeCc({ 1: 'a' }, [2, 4]), 7);
  const usual = [1, 2, 4, 7, 10, 11, 12, 13, 16, 17, 18, 19, 71, 74];
  assert.equal(freeCc(Object.fromEntries(usual.map((c) => [c, 'x']))), 0);
  const all = Object.fromEntries(Array.from({ length: 128 }, (_, i) => [i, 'x']));
  assert.equal(freeCc(all), null);
});

test('target names and the target menu groups', () => {
  assert.equal(targetLabel('selected.osc.freq'), 'osc freq');
  assert.equal(targetLabel('selected.osc.pitch'), 'tune');
  assert.equal(targetLabel('ch3.fx.reverb_mix'), 'CH3 reverb mix');
  assert.equal(targetLabel('global.tempo'), 'tempo');
  assert.equal(targetLabel('morph.position'), 'sound morph');
  assert.equal(targetLabel(null), '(none)');
  const groups = targetGroups();
  const titles = groups.map(([t]) => t);
  assert.deepEqual(titles, ['face strip', 'oscillator', 'noise', 'envelopes', 'mix', 'velocity', 'fx',
    'lfo 1', 'lfo 2', 'pump', 'global']);
  const targets = groups.flatMap(([, xs]) => xs.map(([t]) => t));
  assert.equal(new Set(targets).size, targets.length);
  // every sound parameter but the three → destination buttons, plus six globals
  assert.equal(targets.filter((t) => t.startsWith('selected.')).length, 59 - 3);
  assert.ok(targets.includes('selected.lfo1.phase') && targets.includes('global.master'));
  assert.ok(!targets.includes('selected.lfo1.target'));
});

test('CC activity from the pickup readout', () => {
  const pickup = { 'ch1.osc.freq': { cc: 1, physical: 0.5, linked: true, count: 3 },
    'global.tempo': { cc: 21, physical: 0.2, linked: false, count: 1 } };
  assert.deepEqual(ccActivity(pickup, 1), { physical: 0.5, count: 3 });
  assert.equal(ccActivity(pickup, 2), null);
  assert.equal(ccActivity(undefined, 1), null);
});

test('restart dots: the stream fields pref.audio.pending lists', () => {
  const p = pendingFields(['pref.audio.device', 'pref.audio.sample_rate', 'pref.audio.mono']);
  assert.ok(p.has('pref.audio.device') && p.has('pref.audio.sample_rate'));
  assert.equal(p.size, 2);
  assert.equal(pendingFields(null).size, 0);
});

test('rates, buffers and devices in the menus', () => {
  const all = [96000, 48000, 44100, 32000, 22050, 11025, 8000];
  assert.equal(rateText(0, { synth: true }), 'same as output');
  assert.equal(rateText(22050, { synth: true }), '22050 Hz');
  assert.equal(bufferText(23.8), '23.8 ms');
  assert.equal(bufferText(2), '2 ms');
  assert.equal(ratesNote([48000, 44100], all), 'this device takes 2 of 7 rates');
  assert.equal(ratesNote(all, all), 'this device takes all 7 rates');
  assert.deepEqual(rateChoices(all, [48000, 44100], 96000), [96000, 48000, 44100]);
  assert.deepEqual(rateChoices(all, null, 44100), all);
  assert.deepEqual(deviceChoices(['A', 'B']), [[null, '(system default)'], ['A', 'A'], ['B', 'B']]);
  assert.equal(deviceText(null, ['A']), '(system default)');
  assert.equal(deviceText('Gone', ['A']), 'Gone (not found)');
  assert.equal(deviceText('A', ['A']), 'A');
});

test('the stream line and the model line', () => {
  assert.equal(streamText({ 'audio.running': false }), 'audio stopped');
  assert.equal(streamText({ 'audio.running': true, 'audio.device': 'Fake Out', 'audio.device_is_default': false,
    'audio.sample_rate': 44100, 'audio.synth_rate': 22050, 'audio.buffer_ms': 23.8, 'audio.block_size': 1050,
    'audio.mono': true }), 'running: Fake Out · 44100 Hz · synth 22050 Hz · 23.8 ms (1050 frames) · mono');
  assert.equal(modelText('/m/pattern.pt', null), 'pattern.pt');
  assert.equal(modelText(null, { bundled: true, path: '/x/b.pt' }), 'bundled');
  assert.equal(modelText(null, { bundled: false, path: null }), 'none (no bundled checkpoint)');
});
