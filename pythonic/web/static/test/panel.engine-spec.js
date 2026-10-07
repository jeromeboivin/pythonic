// Needs the DOM: runs inside QtWebEngine only (not a `node --test` pattern).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createFakeBridge } from '../js/bridge.js';
import { createCoreClient } from '../js/core-client.js';
import { ctrlAddress, ctrlSetup, mountPanel } from '../js/panel.js';
import { createStore } from '../js/store.js';

const num = (address, minimum, maximum, unit, extra = {}) => ({ address, kind: 'float', minimum, maximum,
  default: 0, unit, curve: 'linear', labels: [], readonly: false, ...extra });

function setup(values = {}) {
  const describe = {
    'global.tempo': num('global.tempo', 1, 300, 'BPM', { kind: 'int', default: 120 }),
    'global.channel': num('global.channel', 1, 8, '', { kind: 'int', default: 1 }),
  };
  const all = { 'global.tempo': 120, 'global.channel': 1, 'preset.name': '808 Beats', 'undo.can_undo': false,
    'undo.can_redo': false, 'program.current': 1, 'program.occupied': Array(16).fill(false), 'morph.learning': 'off' };
  for (let n = 1; n <= 8; n += 1) {
    for (const [s, lo, hi, unit] of [['osc.pitch', -24, 24, 'st'], ['osc.decay', 10, 10000, 'ms'],
      ['mix.level', -60, 10, 'dB'], ['mix.pan', -100, 100, 'pan'], ['fx.reverb_mix', 0, 1, 'ratio']]) {
      describe[`ch${n}.${s}`] = num(`ch${n}.${s}`, lo, hi, unit);
      all[`ch${n}.${s}`] = 0;
    }
    all[`ch${n}.mute`] = false;
    all[`ch${n}.name`] = ['808 BD', '808 MT', 'Metal Ping'][n - 1] || `Drum ${n}`;
  }
  all['pref.ui.ctrl_knob'] = null;
  const bridge = createFakeBridge({ describe, values: { ...all, ...values },
    actions: { 'transport.toggle': (_a, f) => { f.transport.playing = !f.transport.playing; } } });
  const queued = [];
  const client = createCoreClient(bridge, { schedule: (fn) => queued.push(fn) });
  const store = createStore();
  client.onFrame((f) => store.apply(f));
  const stage = document.createElement('div');
  stage.style.cssText = 'position:absolute;width:1600px;height:1000px';
  document.body.append(stage);
  const panel = mountPanel(stage, { store, client, meta: describe });
  store.seed({ ...all, ...values });
  const flush = async () => { queued.splice(0).forEach((fn) => fn()); await client.flush(); };
  const sets = () => bridge.calls.filter(([s]) => s === 'set').flatMap(([, c]) => c).map((c) => [c.address, c.value]);
  const verbs = () => bridge.calls.filter(([s]) => s === 'act').map(([, p]) => p);
  return { bridge, store, stage, panel, flush, sets, verbs };
}

const tick = () => new Promise((r) => setTimeout(r, 0));

test('the tempo display follows the store', () => {
  const { stage, bridge } = setup();
  assert.equal(stage.querySelector('#tempo-value').textContent, '120');
  bridge.values['global.tempo'] = 133;
  bridge.pushFrame({ changes: { 'global.tempo': 133 } });
  assert.equal(stage.querySelector('#tempo-value').textContent, '133');
});

test('the wheel over the tempo display moves it by 1 BPM as a burst', async () => {
  const { stage, flush, bridge } = setup();
  const seg = stage.querySelector('#tempo-value');
  seg.dispatchEvent(new WheelEvent('wheel', { deltaY: -120, bubbles: true, cancelable: true }));
  seg.dispatchEvent(new WheelEvent('wheel', { deltaY: -120, bubbles: true, cancelable: true }));
  await flush();
  assert.deepEqual(bridge.calls.filter(([s]) => s === 'set').map(([, c]) => c),
    [[{ address: 'global.tempo', value: 122, burst: true }]]);
  assert.equal(seg.textContent, '122');
});

test('transport lights START/STOP, outlines the playhead pad and shows the step', () => {
  const { stage, bridge, panel } = setup();
  bridge.transport.playing = true;
  bridge.transport.position = 21;
  bridge.pushFrame();
  assert.ok(stage.querySelector('#start-stop').classList.contains('on'));
  assert.deepEqual([...stage.querySelectorAll('.pad.ph')].map((p) => p.dataset.step), ['6']);
  assert.deepEqual(panel.display.text(), ['PATTERN A  STEP 22', '808 BEATS']);
});

test('START/STOP starts the toggle verb', async () => {
  const { stage, verbs } = setup();
  stage.querySelector('#start-stop').click();
  await tick();
  assert.deepEqual(verbs().map((v) => v.verb), ['transport.toggle']);
});

test('strips show patch names and drum types, else the channel number', () => {
  const { stage } = setup();
  const tabs = [...stage.querySelectorAll('.strip .tab')].map((t) => t.textContent);
  const buttons = [...stage.querySelectorAll('.strip .chb')].map((b) => b.textContent);
  assert.deepEqual(tabs.slice(0, 4), ['808 BD', '808 MT', 'Metal Ping', 'Drum 4']);
  assert.deepEqual(buttons.slice(0, 4), ['BD', 'MT', '3', '4']);
});

test('channel buttons select; with the MUTE latch they toggle mutes', async () => {
  const { stage, flush, sets } = setup();
  const button = (n) => stage.querySelector(`.strip[data-channel="${n}"] .chb`);
  button(3).click();
  assert.ok(stage.querySelector('.strip[data-channel="3"]').classList.contains('sel'));
  assert.ok(button(3).classList.contains('on') && !button(1).classList.contains('on'));
  stage.querySelector('#mute-latch').click();
  button(5).click();
  assert.ok(stage.querySelector('.strip[data-channel="5"]').classList.contains('muted'));
  assert.ok(button(5).classList.contains('mut'));
  await flush();
  assert.deepEqual(sets(), [['global.channel', 3], ['ch5.mute', true]]);
});

test('the CTRL knobs follow the panel-wide mode', async () => {
  const { stage, store, flush, sets } = setup();
  const knob = (n) => stage.querySelector(`.strip[data-channel="${n}"] .ctrl`);
  assert.equal(knob(2).dataset.address, 'ch2.mix.pan');
  store.assume('pref.ui.ctrl_knob', { mode: 'reverb_mix', user: [] });
  assert.equal(knob(2).dataset.address, 'ch2.fx.reverb_mix');
  assert.equal(knob(2).querySelector('.val').textContent, '0 %');
  store.assume('pref.ui.ctrl_knob', { mode: 'off' });
  assert.equal(knob(2).dataset.address, undefined);
  assert.equal(knob(2).querySelector('.val').textContent, 'off');
  stage.querySelector('#ctrl-mode').click();
  const items = [...stage.querySelectorAll('.px-menu .it')];
  assert.deepEqual(items.map((i) => i.textContent), ['off', 'pan', 'reverb mix', 'delay mix', 'lfo 1 depth', 'user']);
  items[5].click();
  assert.equal(knob(1).dataset.address, undefined);
  assert.equal(knob(1).querySelector('.lbl').textContent, 'pick ▾');
  await flush();
  assert.deepEqual(sets(), [['pref.ui.ctrl_knob', { mode: 'user', user: Array(8).fill(null) }]]);
});

test('CTRL setup and addresses', () => {
  assert.deepEqual(ctrlSetup(null), { mode: 'pan', user: Array(8).fill(null) });
  const s = ctrlSetup({ mode: 'user', user: ['osc.freq', null, 'fx.delay_mix'] });
  assert.equal(ctrlAddress(s, 1), 'ch1.osc.freq');
  assert.equal(ctrlAddress(s, 2), null);
  assert.equal(ctrlAddress(s, 3), 'ch3.fx.delay_mix');
  assert.equal(ctrlAddress(ctrlSetup({ mode: 'lfo1_depth' }), 8), 'ch8.lfo1.depth');
  assert.equal(ctrlAddress(ctrlSetup({ mode: 'bogus' }), 4), 'ch4.mix.pan');
});

test('programs light the current one and select through the verb', async () => {
  const occupied = Array(16).fill(false);
  occupied[0] = true;
  occupied[4] = true;
  const { stage, verbs } = setup({ 'program.current': 5, 'program.occupied': occupied });
  const buttons = [...stage.querySelectorAll('#programs .btn')];
  assert.deepEqual(buttons.map((b, i) => (b.classList.contains('on') ? i + 1 : 0)).filter(Boolean), [5]);
  assert.ok(buttons[1].classList.contains('empty') && !buttons[0].classList.contains('empty'));
  buttons[11].click();
  await tick();
  assert.deepEqual(verbs(), [{ verb: 'program.select', args: { program: 12 } }]);
});

test('undo and redo follow the journal and run their verbs', async () => {
  const { stage, bridge, verbs } = setup();
  assert.ok(stage.querySelector('#undo').disabled && stage.querySelector('#redo').disabled);
  bridge.pushFrame({ changes: { 'undo.can_undo': true } });
  assert.ok(!stage.querySelector('#undo').disabled);
  stage.querySelector('#undo').click();
  await tick();
  assert.deepEqual(verbs().map((v) => v.verb), ['undo']);
});

test('learn A starts morph learn, a second press stops it', async () => {
  const { stage, store, verbs } = setup();
  stage.querySelector('#learn-a').click();
  store.assume('morph.learning', 'a');
  assert.ok(stage.querySelector('#learn-a').classList.contains('on'));
  stage.querySelector('#learn-a').click();
  await tick();
  assert.deepEqual(verbs(), [{ verb: 'morph.learn', args: { endpoint: 'a' } },
    { verb: 'morph.learn', args: { endpoint: null } }]);
});

test('errors without an action show on the display', () => {
  const { panel, bridge } = setup();
  bridge.pushFrame({ events: [{ id: null, verb: null, status: 'error', source: 'audio',
    error: 'Audio stream stalled: no callback for 2 s; the stream was stopped' }] });
  assert.equal(panel.display.text()[0], 'AUDIO ERROR');
  assert.ok(panel.display.classList.contains('alert'));
});
