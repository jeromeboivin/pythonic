// Needs the DOM: runs inside QtWebEngine only (not a `node --test` pattern).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createFakeBridge } from '../js/bridge.js';
import { createCoreClient } from '../js/core-client.js';
import { mountPanel } from '../js/panel.js';
import { createStore } from '../js/store.js';

function setup() {
  const bridge = createFakeBridge({ values: { 'global.tempo': 120 },
    actions: { 'transport.toggle': (_a, f) => { f.transport.playing = !f.transport.playing; } } });
  const queued = [];
  const client = createCoreClient(bridge, { schedule: (fn) => queued.push(fn) });
  const store = createStore();
  client.onFrame((f) => store.apply(f));
  const stage = document.createElement('div');
  document.body.append(stage);
  mountPanel(stage, { store, client, meta: { 'global.tempo': { minimum: 1, maximum: 300 } } });
  store.seed({ 'global.tempo': 120 });
  return { bridge, client, store, stage, runFrame: () => queued.splice(0).forEach((fn) => fn()) };
}

test('the tempo display follows the store', () => {
  const { stage, bridge } = setup();
  assert.equal(stage.querySelector('#tempo-value').textContent, '120');
  bridge.values['global.tempo'] = 133;
  bridge.pushFrame({ changes: { 'global.tempo': 133 } });
  assert.equal(stage.querySelector('#tempo-value').textContent, '133');
});

test('tempo + sets the next value, clamped to the range', async () => {
  const { stage, bridge, client, runFrame } = setup();
  stage.querySelector('#tempo-up').click();
  stage.querySelector('#tempo-up').click();
  runFrame();
  await client.flush();
  assert.deepEqual(bridge.calls.filter(([s]) => s === 'set').map(([, c]) => c), [[{ address: 'global.tempo', value: 122 }]]);
  assert.equal(stage.querySelector('#tempo-value').textContent, '122');
});

test('transport lights START/STOP and outlines the playhead pad', () => {
  const { stage, bridge } = setup();
  bridge.transport.playing = true;
  bridge.transport.position = 21;
  bridge.pushFrame();
  assert.ok(stage.querySelector('#start-stop').classList.contains('on'));
  const lit = [...stage.querySelectorAll('.pad.ph')].map((p) => p.dataset.step);
  assert.deepEqual(lit, ['6']);
  assert.ok(stage.querySelector('#display-line1').textContent.includes('STEP 22'));
});

test('START/STOP starts the toggle verb', async () => {
  const { stage, bridge } = setup();
  stage.querySelector('#start-stop').click();
  await new Promise((r) => setTimeout(r, 0));
  assert.deepEqual(bridge.calls.filter(([s]) => s === 'act').map(([, p]) => p.verb), ['transport.toggle']);
});
