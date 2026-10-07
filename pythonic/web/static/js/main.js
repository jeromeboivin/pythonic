// Boot: connect the bridge (a demo fake outside the app), build the client
// and the store, fit the stage, mount the panel, then read the values the
// panel watches. window.pythonic exposes { bridge, client, store, meta, ready }.

import { mountAiPage } from './ai-page.js';
import { connectBridge, createFakeBridge } from './bridge.js';
import { createCoreClient } from './core-client.js';
import { mountPanel } from './panel.js';
import { mountSetup } from './setup.js';
import { mountStage } from './stage.js';
import { createStore } from './store.js';

/** A fake core for opening index.html in a plain browser (no Qt). */
export function demoBridge(scope = globalThis) {
  const fake = createFakeBridge({
    describe: { 'global.tempo': { address: 'global.tempo', kind: 'int', minimum: 1, maximum: 300, default: 120,
      unit: 'BPM', curve: 'linear', labels: [], readonly: false } },
    values: { 'global.tempo': 120, 'global.channel': 1 },
    actions: {
      'transport.toggle': (_args, f) => { f.transport.playing = !f.transport.playing; f.transport.position = 0; },
    },
  });
  let last = Date.now();
  scope.setInterval(() => {
    const t = fake.transport;
    const stepMs = 60000 / fake.values['global.tempo'] / 4;
    if (t.playing && Date.now() - last >= stepMs) { last = Date.now(); t.position = (t.position + 1) % 16; }
    fake.pushFrame();
  }, 16);
  return fake;
}

export async function boot({ bridge = null, stage = document.getElementById('stage') } = {}) {
  bridge = bridge || (await connectBridge()) || demoBridge();
  const client = createCoreClient(bridge);
  const store = createStore();
  client.onFrame((frame) => store.apply(frame));
  mountStage(stage);
  const meta = await client.describe(''); // every registered address
  const panel = mountPanel(stage, { store, client, meta });
  const setup = mountSetup({ panel, store, client, meta });
  panel.ai = mountAiPage({ panel, store, client, stage }); // the AI drum generator page
  store.seed(await client.get(store.watched()));
  return { bridge, client, store, meta, panel, setup };
}

if (document.getElementById('stage') && !globalThis.__pythonicNoBoot) {
  globalThis.pythonic = { ready: false };
  boot().then((app) => { globalThis.pythonic = { ...app, ready: true }; }, (err) => {
    globalThis.pythonic = { ready: false, error: String(err && err.stack || err) };
    console.error('boot failed', err);
  });
}
