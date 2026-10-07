import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createFakeBridge } from '../js/bridge.js';
import { createCoreClient } from '../js/core-client.js';

const TEMPO = { address: 'global.tempo', kind: 'int', minimum: 1, maximum: 300 };

function setup() {
  const bridge = createFakeBridge({ describe: { 'global.tempo': TEMPO }, values: { 'global.tempo': 120 },
    actions: { 'preset.load': ({ path }) => ({ path }), 'boom': () => { throw new Error('nope'); } } });
  const queued = [];
  const client = createCoreClient(bridge, { schedule: (fn) => queued.push(fn) });
  const runFrame = () => queued.splice(0).forEach((fn) => fn());
  return { bridge, client, runFrame };
}

test('sets of one animation frame go out as one call, latest value wins', async () => {
  const { bridge, client, runFrame } = setup();
  client.set('global.tempo', 121);
  client.set('global.tempo', 125, { burst: true });
  assert.deepEqual(bridge.calls, [['resync', null]]);  // sent on creation
  runFrame();
  await client.flush();
  assert.deepEqual(bridge.calls, [['resync', null], ['set', [{ address: 'global.tempo', value: 125, burst: true }]]]);
  assert.equal(bridge.values['global.tempo'], 125);
});

test('act resolves with the poll event of its id', async () => {
  const { bridge, client } = setup();
  const pending = client.act('preset.load', { path: 'a.json' });
  await new Promise((r) => setTimeout(r, 0));
  bridge.pushFrame();
  const event = await pending;
  assert.equal(event.status, 'done');
  assert.deepEqual(event.result, { path: 'a.json' });
});

test('a failing verb resolves with an error event', async () => {
  const { bridge, client } = setup();
  const pending = client.act('boom');
  await new Promise((r) => setTimeout(r, 0));
  bridge.pushFrame();
  const event = await pending;
  assert.equal(event.status, 'error');
  assert.equal(event.error, 'nope');
});

test('get and describe return plain objects', async () => {
  const { client } = setup();
  assert.deepEqual(await client.get(['global.tempo']), { 'global.tempo': 120 });
  assert.deepEqual(await client.describe('global.'), { 'global.tempo': TEMPO });
});

test('a file dialog resolves with the chosen path, or null', async () => {
  const { bridge, client } = setup();
  bridge.dialogAnswers.push('/tmp/x.mtpreset');
  assert.equal(await client.openFile({ title: 'Open' }), '/tmp/x.mtpreset');
  assert.equal(await client.saveFile({}), null);
  assert.equal(bridge.calls.at(-1)[1].mode, 'save');
});

test('frames reach listeners', () => {
  const { bridge, client } = setup();
  const seen = [];
  client.onFrame((f) => seen.push(f.version));
  bridge.pushFrame();
  bridge.pushFrame();
  assert.deepEqual(seen, [1, 2]);
});

test('progress events reach onProgress and the act resolves with the done event', async () => {
  const { bridge, client } = setup();
  const seen = [];
  const pending = client.act('preset.load', { path: 'x' }, { onProgress: (p) => seen.push(p) });
  await new Promise((r) => setTimeout(r, 0));
  const id = bridge.calls.filter(([s]) => s === 'act').length;
  bridge.pushFrame({ events: [{ id, verb: 'preset.load', status: 'progress', progress: 0.25 }] });
  bridge.post({ id, verb: 'preset.load', status: 'progress', progress: 0.5 });
  bridge.post({ id, verb: 'preset.load', status: 'done', result: { path: 'x' } });
  bridge.pushFrame();
  const event = await pending;
  assert.equal(event.status, 'done');
  assert.deepEqual(seen, [0.25, 0.5]);
});

test('errors nobody waits for reach onUnclaimedError, awaited ones do not', async () => {
  const { bridge, client } = setup();
  const seen = [];
  client.onUnclaimedError((e) => seen.push(e.error));
  bridge.post({ id: null, verb: null, status: 'error', source: 'audio', error: 'stream stalled' });
  bridge.pushFrame();
  assert.deepEqual(seen, ['stream stalled']);
  const pending = client.act('boom');
  await new Promise((r) => setTimeout(r, 0));
  for (let i = 0; i < 4; i += 1) bridge.pushFrame();
  assert.equal((await pending).status, 'error');
  // an error of an action no caller waits for, reported a few frames later
  bridge.post({ id: 99, verb: 'po32.send', status: 'error', error: 'orphan' });
  bridge.pushFrame();
  assert.deepEqual(seen, ['stream stalled']);
  bridge.pushFrame();
  bridge.pushFrame();
  assert.deepEqual(seen, ['stream stalled', 'orphan']);
});

test('trigger goes to the bridge slot', async () => {
  const { bridge, client } = setup();
  await client.trigger(3, 64);
  assert.deepEqual(bridge.calls.at(-1), ['trigger', { channel: 3, velocity: 64 }]);
});
