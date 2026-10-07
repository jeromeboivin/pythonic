import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createStore } from '../js/store.js';

test('seeded values are read and watched, now and on change', () => {
  const store = createStore();
  store.seed({ 'global.tempo': 120 });
  const seen = [];
  store.watch('global.tempo', (v, a) => seen.push([a, v]));
  store.apply({ version: 3, changes: { 'global.tempo': 128, 'global.swing': 0.2 } });
  assert.equal(store.value('global.tempo'), 128);
  assert.equal(store.value('global.swing'), 0.2);
  assert.equal(store.version, 3);
  assert.deepEqual(seen, [['global.tempo', 120], ['global.tempo', 128]]);
});

test('an unchanged value does not notify', () => {
  const store = createStore();
  store.seed({ 'ch1.mute': false });
  let calls = 0;
  store.watch('ch1.mute', () => { calls += 1; }, { now: false });
  store.apply({ changes: { 'ch1.mute': false } });
  store.apply({ changes: { 'ch1.mute': true } });
  assert.equal(calls, 1);
});

test('unwatch stops notifications and watched() lists live watches', () => {
  const store = createStore();
  const off = store.watch('a', () => {});
  store.watch('b', () => {});
  assert.deepEqual(store.watched().sort(), ['a', 'b']);
  off();
  assert.deepEqual(store.watched(), ['b']);
});

test('readouts notify when they change', () => {
  const store = createStore();
  const seen = [];
  store.watchReadout('transport', (t) => seen.push(t.position));
  const transport = { playing: true, position: 0 };
  store.apply({ transport });
  store.apply({ transport: { ...transport } });
  store.apply({ transport: { playing: true, position: 1 } });
  assert.deepEqual(seen, [0, 1]);
  assert.equal(store.readout('transport').position, 1);
});

test('events reach listeners and assume() echoes locally', () => {
  const store = createStore();
  const events = [];
  store.onEvent((e) => events.push(e.id));
  store.apply({ events: [{ id: 4, status: 'done' }, { id: null, status: 'error' }] });
  assert.deepEqual(events, [4, null]);
  store.assume('global.tempo', 99);
  assert.equal(store.value('global.tempo'), 99);
});
