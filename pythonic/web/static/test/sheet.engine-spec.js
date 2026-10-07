// Needs the DOM: runs inside QtWebEngine only. The overlay sheet primitive,
// the alert sheet, and the panel's pages, error handler and channel hit.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createFakeBridge } from '../js/bridge.js';
import { createCoreClient } from '../js/core-client.js';
import { mountPanel } from '../js/panel.js';
import { createSheets } from '../js/sheet.js';
import { createStore } from '../js/store.js';
import { patternValues } from './pattern-fixtures.js';

const tick = () => new Promise((resolve) => {
  const channel = new MessageChannel();
  channel.port1.onmessage = () => resolve();
  channel.port2.postMessage(null);
});

function stageOf() {
  const stage = document.createElement('div');
  stage.style.cssText = 'position:absolute;width:1600px;height:1000px';
  document.body.append(stage);
  return stage;
}

const button = (root, label) => [...root.querySelectorAll('.alert-buttons .btn')].find((b) => b.textContent === label);

test('a sheet covers the stage until hidden; a dismissable one closes from outside', () => {
  const stage = stageOf();
  const sheets = createSheets(stage);
  const hidden = [];
  const card = sheets.show('setup', document.createElement('p'), { width: 700, onHide: () => hidden.push('setup') });
  const layer = stage.querySelector('.sheet-layer');
  assert.equal(layer.dataset.sheet, 'setup');
  assert.equal(card.style.width, '700px');
  assert.equal(sheets.current, 'setup');
  layer.click(); // not dismissable
  assert.ok(sheets.isOpen('setup'));
  assert.ok(sheets.hide('setup'));
  assert.deepEqual(hidden, ['setup']);
  assert.equal(stage.querySelector('.sheet-layer'), null);

  sheets.show('setup', document.createElement('p'), { dismissable: true });
  stage.querySelector('.sheet .sheet, .sheet')?.click(); // inside: stays
  assert.ok(sheets.isOpen('setup'));
  stage.querySelector('.sheet-layer').click();
  assert.ok(!sheets.isOpen('setup'));
});

test('the alert sheet resolves with its button, closes only by its buttons and queues', async () => {
  const stage = stageOf();
  const shown = [];
  const sheets = createSheets(stage, { onShow: (name) => shown.push(name) });
  const first = sheets.alert({ title: 'Could not load the preset', text: 'not a preset file' });
  const second = sheets.ask('Replace “a.json”?', 'It exists.', { yes: 'replace' });
  const box = stage.querySelector('.alert-sheet');
  assert.ok(box.classList.contains('tone-error'));
  assert.equal(box.querySelector('.alert-title').textContent, 'Could not load the preset');
  assert.equal(box.closest('.sheet').style.width, '460px');
  stage.querySelector('.sheet-layer').click(); // outside: nothing
  assert.equal(stage.querySelectorAll('.alert-sheet').length, 1);
  button(stage, 'OK').click();
  assert.equal(await first, true);
  const ask = stage.querySelector('.alert-sheet');
  assert.ok(ask.classList.contains('tone-ok'));
  assert.deepEqual([...ask.querySelectorAll('.btn')].map((b) => [b.textContent, b.classList.contains('primary')]),
    [['cancel', false], ['replace', true]]);
  button(stage, 'cancel').click();
  assert.equal(await second, false);
  assert.equal(stage.querySelector('.sheet-layer'), null);
  assert.deepEqual(shown, ['alert', 'alert']);
});

test('an error alert already showing is not shown twice; an alert stacks over a sheet', async () => {
  const stage = stageOf();
  const sheets = createSheets(stage);
  sheets.show('setup', document.createElement('p'));
  const a = sheets.alert({ title: 'Audio problem', text: 'stalled' });
  const b = sheets.alert({ title: 'Audio problem', text: 'stalled' });
  assert.equal(await b, null);
  const layers = [...stage.querySelectorAll('.sheet-layer')];
  assert.deepEqual(layers.map((l) => l.dataset.sheet), ['setup', 'alert']);
  assert.ok(Number(layers[1].style.zIndex) > Number(layers[0].style.zIndex));
  button(stage, 'OK').click();
  assert.equal(await a, true);
  assert.equal(sheets.current, 'setup');
});

function setup() {
  const all = { 'global.channel': 2, 'preset.name': '808 Beats', 'undo.can_undo': false, 'undo.can_redo': false,
    'program.current': 1, 'program.occupied': Array(16).fill(false), 'morph.learning': 'off', 'pref.ui.ctrl_knob': null,
    ...patternValues() };
  for (let n = 1; n <= 8; n += 1) { all[`ch${n}.mute`] = false; all[`ch${n}.name`] = `Drum ${n}`; }
  const bridge = createFakeBridge({ values: all });
  const client = createCoreClient(bridge, { schedule: (fn) => fn() });
  const store = createStore();
  client.onFrame((f) => store.apply(f));
  const stage = stageOf();
  const panel = mountPanel(stage, { store, client, meta: {} });
  store.seed(all);
  return { bridge, stage, panel, store };
}

test('PO-32 and SETUP open their registered pages, else say coming soon', async () => {
  const { stage, panel } = setup();
  stage.querySelector('#po32-button').click();
  assert.equal(stage.querySelector('.alert-title').textContent, 'PO-32 transfer and import');
  assert.match(stage.querySelector('.alert-text').textContent, /coming soon/i);
  button(stage, 'OK').click();
  const opened = [];
  const off = panel.registerPage('setup', (options) => opened.push(['setup', options]));
  panel.registerPage('po32', (options) => opened.push(['po32', options]));
  stage.querySelector('#setup-button').click();
  stage.querySelector('#po32-button').click();
  stage.querySelector('#midi-row').click();
  assert.deepEqual(opened, [['setup', {}], ['po32', {}], ['setup', { tab: 'midi' }]]);
  assert.deepEqual(panel.pages.sort(), ['po32', 'setup']);
  off();
  assert.deepEqual(panel.pages, ['po32']);
  assert.equal(stage.querySelector('.sheet-layer'), null);
});

test('errors nobody waits for show on the display, and on the alert sheet when they need reading', async () => {
  const { bridge, stage, panel } = setup();
  bridge.pushFrame({ events: [{ id: null, verb: null, status: 'error', source: 'midi', error: 'bad message' }] });
  assert.equal(panel.display.text()[0], 'MIDI ERROR');
  assert.equal(stage.querySelector('.alert-sheet'), null);
  bridge.pushFrame({ events: [{ id: null, verb: null, status: 'error', source: 'audio', error: 'Audio stream stalled' }] });
  assert.equal(panel.display.text()[0], 'AUDIO ERROR');
  const box = stage.querySelector('.alert-sheet');
  assert.ok(box.classList.contains('tone-error'));
  assert.equal(box.querySelector('.alert-title').textContent, 'Audio problem');
  assert.equal(box.querySelector('.alert-text').textContent, 'Audio stream stalled');
});

test('a sheet blocks the panel: menus close and nothing behind takes the wheel', () => {
  const { stage, panel } = setup();
  panel.ctx.openMenu([['x', () => {}]], 10, 10);
  assert.ok(stage.querySelector('.px-menu'));
  panel.sheets.show('setup', document.createElement('p'));
  assert.equal(stage.querySelector('.px-menu'), null);
  const wheel = new WheelEvent('wheel', { deltaY: -120, bubbles: true, cancelable: true });
  stage.querySelector('.sheet-layer').dispatchEvent(wheel);
  assert.ok(wheel.defaultPrevented);
});

test('clicking the selected channel button hits it (Ctrl: full velocity), another selects', async () => {
  const { bridge, stage } = setup();
  stage.querySelector('.strip[data-channel="2"] .chb').click();
  stage.querySelector('.strip[data-channel="2"] .chb').dispatchEvent(new MouseEvent('click', { bubbles: true, ctrlKey: true }));
  stage.querySelector('.strip[data-channel="3"] .chb').click();
  await tick();
  assert.deepEqual(bridge.calls.filter(([s]) => s === 'trigger').map(([, p]) => p),
    [{ channel: 2, velocity: 64 }, { channel: 2, velocity: 127 }]);
  assert.deepEqual(bridge.calls.filter(([s]) => s === 'set').flatMap(([, c]) => c).map((c) => [c.address, c.value]),
    [['global.channel', 3]]);
});

test('a channel button flashes when a MIDI note hits its channel', () => {
  const { bridge, stage } = setup();
  const midi = (notes) => ({ activity: 1, notes, pickup: {} });
  bridge.pushFrame({ midi: midi([0, 0, 0, 0, 0, 0, 0, 0]) });
  bridge.pushFrame({ midi: midi([0, 0, 1, 0, 0, 0, 0, 0]) });
  assert.ok(stage.querySelector('.strip[data-channel="3"] .chb').classList.contains('hit'));
  assert.ok(!stage.querySelector('.strip[data-channel="1"] .chb').classList.contains('hit'));
});
