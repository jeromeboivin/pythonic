// Needs the DOM: runs inside QtWebEngine only. The control components over a
// fake bridge: binding, drag, wheel, track jump, typed values, menus, cues.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createFakeBridge } from '../js/bridge.js';
import { createControlContext, provideContext } from '../js/controls.js';
import { createCoreClient } from '../js/core-client.js';
import { createStore } from '../js/store.js';

const DECAY = { address: 'ch1.osc.decay', kind: 'float', minimum: 10, maximum: 10000, default: 316.23,
  unit: 'ms', curve: 'log', labels: [], readonly: false };
const LEVEL = { address: 'ch1.mix.level', kind: 'float', minimum: -60, maximum: 10, default: 0,
  unit: 'dB', curve: 'linear', labels: [], readonly: false };
const RATE = { address: 'global.step_rate', kind: 'enum', minimum: null, maximum: null, default: '1/16',
  unit: '', curve: 'linear', labels: ['1/8', '1/8T', '1/16', '1/16T', '1/32'], readonly: false };
const MUTE = { address: 'ch1.mute', kind: 'bool', minimum: null, maximum: null, default: false,
  unit: '', curve: 'linear', labels: [], readonly: false };
const FILL = { address: 'global.fill_rate', kind: 'int', minimum: 2, maximum: 8, default: 4,
  unit: 'x', curve: 'linear', labels: [], readonly: false };
const PAN = { address: 'ch1.mix.pan', kind: 'float', minimum: -100, maximum: 100, default: 0,
  unit: 'pan', curve: 'linear', labels: [], readonly: false };
const META = { [DECAY.address]: DECAY, [LEVEL.address]: LEVEL, [RATE.address]: RATE,
  [MUTE.address]: MUTE, [FILL.address]: FILL, [PAN.address]: PAN };

function setup(html, values = {}) {
  const all = { 'ch1.osc.decay': 100, 'ch1.mix.level': 0, 'global.step_rate': '1/16', 'ch1.mute': false,
    'global.fill_rate': 4, 'ch1.mix.pan': 0, 'midi.cc_map': {}, 'midi.learning': null,
    'midi.pitchbend_target': null, 'global.channel': 1, ...values };
  const bridge = createFakeBridge({ describe: META, values: all });
  const queued = [];
  const client = createCoreClient(bridge, { schedule: (fn) => queued.push(fn) });
  const store = createStore();
  client.onFrame((f) => store.apply(f));
  store.seed(all);
  const root = document.createElement('div');
  root.style.cssText = 'position:absolute;left:0;top:0;width:800px;height:600px';
  document.body.append(root);
  const ctx = createControlContext({ store, client, meta: META, root });
  provideContext(root, ctx);
  root.innerHTML = html;
  const touches = [];
  root.addEventListener('px-touch', (e) => touches.push(e.detail));
  const flush = async () => { queued.splice(0).forEach((fn) => fn()); await client.flush(); };
  const sets = () => bridge.calls.filter(([s]) => s === 'set').flatMap(([, c]) => c);
  const gestures = () => bridge.calls.filter(([s]) => s === 'gesture').map(([, p]) => p);
  return { bridge, client, store, root, ctx, touches, flush, sets, gestures };
}

function pointer(target, type, { x = 0, y = 0, button = 0, shiftKey = false } = {}) {
  target.dispatchEvent(new PointerEvent(type, { bubbles: true, cancelable: true, clientX: x, clientY: y,
    button, buttons: type === 'pointerup' ? 0 : 1, shiftKey, pointerId: 1 }));
}
const wheel = (target, deltaY) => target.dispatchEvent(new WheelEvent('wheel', { bubbles: true, cancelable: true, deltaY }));

test('a knob binds to its address and prints the value', () => {
  const { root, store } = setup('<px-knob data-address="ch1.osc.decay" label="decay"></px-knob>');
  const knob = root.querySelector('px-knob');
  assert.equal(knob.querySelector('.val').textContent, '100 ms');
  assert.equal(knob.querySelector('.lbl').textContent, 'decay');
  store.assume('ch1.osc.decay', 1500);
  assert.equal(knob.querySelector('.val').textContent, '1.50 s');
  assert.equal(Number(knob.dataset.position).toFixed(3), (Math.log(150) / Math.log(1000)).toFixed(3));
});

test('a vertical drag is one gesture: 200 px is the full range, Shift is fine', async () => {
  const { root, flush, sets, gestures, store } = setup('<px-knob data-address="ch1.mix.pan"></px-knob>');
  const knob = root.querySelector('px-knob');
  pointer(knob, 'pointerdown', { y: 300 });
  pointer(knob, 'pointermove', { y: 250 }); // up 50 px = +25 % = +50
  assert.equal(store.value('ch1.mix.pan'), 50);
  pointer(knob, 'pointermove', { y: 250, shiftKey: true }); // re-base for fine
  pointer(knob, 'pointermove', { y: 150, shiftKey: true }); // 100 px fine = +5 %
  assert.equal(Math.round(store.value('ch1.mix.pan')), 60);
  pointer(knob, 'pointerup', { y: 150 });
  await flush();
  assert.deepEqual(gestures(), ['begin', 'end']);
  assert.equal(Math.round(sets().at(-1).value), 60);
});

test('a click without a move is no gesture', async () => {
  const { root, flush, gestures, touches } = setup('<px-knob data-address="ch1.mix.pan" name="ch1 pan"></px-knob>');
  const knob = root.querySelector('px-knob');
  pointer(knob, 'pointerdown', { y: 100 });
  pointer(knob, 'pointerup', { y: 100 });
  await flush();
  assert.deepEqual(gestures(), []);
  assert.deepEqual(touches.map((t) => [t.name, t.text]), [['CH1 PAN', 'C']]);
});

test('the wheel moves 1 % per notch as a burst', async () => {
  const { root, flush, sets } = setup('<px-knob data-address="ch1.mix.pan"></px-knob>');
  wheel(root.querySelector('px-knob'), -120);
  await flush();
  assert.deepEqual(sets(), [{ address: 'ch1.mix.pan', value: 2, burst: true }]);
});

test('a fader moves 2 % per notch; a track click jumps there', async () => {
  const { root, flush, sets, gestures, store } = setup(
    '<px-fader data-address="ch1.mix.level" length="100"></px-fader>');
  const fader = root.querySelector('px-fader');
  wheel(fader, 120);
  assert.equal(store.value('ch1.mix.level').toFixed(6), '-1.400000');
  const track = fader.querySelector('.track').getBoundingClientRect();
  pointer(fader.querySelector('.track'), 'pointerdown', { y: track.top + track.height * 0.5 });
  assert.equal(store.value('ch1.mix.level'), -25);
  pointer(fader, 'pointermove', { y: track.top + track.height * 0.5 - 10 }); // 10 % of 100 px
  pointer(fader, 'pointerup', {});
  await flush();
  assert.equal(Math.round(store.value('ch1.mix.level')), -18);
  assert.deepEqual(gestures(), ['begin', 'end']);
  assert.equal(sets()[0].burst, true);
});

test('double-click types an exact value; Esc cancels', async () => {
  const { root, flush, sets } = setup('<px-knob data-address="ch1.osc.decay"></px-knob>');
  const knob = root.querySelector('px-knob');
  knob.dispatchEvent(new MouseEvent('dblclick', { bubbles: true }));
  let input = root.querySelector('.px-entry input');
  assert.equal(input.value, '100');
  input.value = '1.2 s';
  input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
  assert.equal(root.querySelector('.px-entry'), null);
  knob.dispatchEvent(new MouseEvent('dblclick', { bubbles: true }));
  input = root.querySelector('.px-entry input');
  input.value = '50';
  input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
  await flush();
  assert.deepEqual(sets(), [{ address: 'ch1.osc.decay', value: 1200 }]);
});

test('the right-click menu resets, learns, removes a CC and assigns pitch bend', async () => {
  const { root, flush, sets, bridge } = setup('<px-knob data-address="ch1.osc.decay"></px-knob>',
    { 'midi.cc_map': { 21: 'ch1.osc.decay', 7: 'global.master' } });
  const knob = root.querySelector('px-knob');
  const menu = () => [...root.querySelectorAll('.px-menu .it')];
  const open = () => knob.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 10, clientY: 10 }));
  open();
  assert.deepEqual(menu().map((i) => i.textContent),
    ['Reset to default (316 ms)', 'MIDI learn (CC)', 'Remove CC 21 mapping', 'Assign pitch bend']);
  menu()[0].click();
  open();
  menu()[2].click(); // remove CC 21
  open();
  assert.deepEqual(menu().map((i) => i.textContent).slice(1), ['MIDI learn (CC)', 'Assign pitch bend']);
  menu()[2].click();
  open();
  assert.equal(menu()[2].textContent, 'Remove pitch bend');
  menu()[1].click();
  await flush();
  await new Promise((r) => setTimeout(r, 0));
  assert.deepEqual(sets(), [
    { address: 'ch1.osc.decay', value: 316.23 },
    { address: 'midi.cc_map', value: { 7: 'global.master' } },
    { address: 'midi.pitchbend_target', value: 'ch1.osc.decay' },
  ]);
  const acts = bridge.calls.filter(([s]) => s === 'act').map(([, p]) => p);
  assert.deepEqual(acts, [{ verb: 'midi.learn', args: { target: 'ch1.osc.decay' } }]);
});

test('CC badge, learn pulse, blinking LED and the pickup ghost', () => {
  const { root, bridge, store } = setup('<px-knob data-address="ch1.osc.decay"></px-knob>');
  const knob = root.querySelector('px-knob');
  assert.equal(knob.querySelector('.cc').textContent, '');
  bridge.values['midi.cc_map'] = { 21: 'selected.osc.decay' };
  bridge.pushFrame({ changes: { 'midi.cc_map': { 21: 'selected.osc.decay' }, 'midi.learning': 'ch1.osc.decay' } });
  assert.equal(knob.querySelector('.cc').textContent, 'CC 21');
  assert.ok(knob.classList.contains('mapped') && knob.classList.contains('learning'));
  const pickup = (count, physical, linked) => bridge.pushFrame({ midi: { activity: count, notes: [],
    pickup: { 'ch1.osc.decay': { cc: 21, physical, linked, count } } } });
  pickup(1, 0.9, false);
  assert.equal(knob.dataset.ghost, '0.9000');
  assert.ok(!knob.classList.contains('cc-blink'));
  pickup(2, store.value('ch1.osc.decay') && Number(knob.dataset.position), true);
  assert.equal(knob.dataset.ghost, undefined);
  assert.ok(knob.classList.contains('cc-blink'));
});

test('modulation draws an arc in the source colour', () => {
  const { root, bridge } = setup('<px-knob data-address="ch1.osc.decay"></px-knob><px-fader data-address="ch1.mix.level"></px-fader>',
    { 'ch1.lfo2.on': true, 'ch1.lfo2.target': 'osc_decay', 'ch1.pump.on': true, 'ch1.pump.target': 'level_db' });
  const channels = Array(8).fill(null).map(() => ({}));
  channels[0] = { osc_decay: 400, level_db: -12 };
  bridge.pushFrame({ modulation: { channel: 0, offsets: channels[0], channels } });
  const knob = root.querySelector('px-knob');
  assert.equal(knob.dataset.mod, 'lfo2');
  assert.ok(knob.querySelector('.mod').getAttribute('d').startsWith('M'));
  assert.equal(root.querySelector('px-fader').dataset.mod, 'pump');
  bridge.pushFrame({ modulation: { channel: 0, offsets: {}, channels: Array(8).fill({}) } });
  assert.equal(knob.querySelector('.mod').getAttribute('d'), '');
  assert.equal(knob.dataset.mod, undefined);
});

test('changing data-address rebinds; removing it disables the control', () => {
  const { root, store } = setup('<px-knob data-address="ch1.mix.pan"></px-knob>', { 'ch1.mix.pan': -30 });
  const knob = root.querySelector('px-knob');
  assert.equal(knob.querySelector('.val').textContent, 'L30');
  knob.dataset.address = 'ch1.osc.decay';
  assert.equal(knob.querySelector('.val').textContent, '100 ms');
  knob.removeAttribute('data-address');
  assert.equal(knob.querySelector('.val').textContent, 'off');
  assert.ok(knob.classList.contains('off'));
  store.assume('ch1.mix.pan', 10);
  assert.equal(knob.querySelector('.val').textContent, 'off');
});

test('a toggle flips its bool; a switch lights its option; a list offers its range', async () => {
  const { root, flush, sets } = setup(`<px-toggle data-address="ch1.mute" label="mute"></px-toggle>
    <px-switch data-address="global.step_rate"></px-switch><px-list data-address="global.fill_rate"></px-list>`);
  root.querySelector('px-toggle button').click();
  assert.ok(root.querySelector('px-toggle button').classList.contains('on'));
  const sw = root.querySelector('px-switch');
  assert.deepEqual([...sw.querySelectorAll('.btn')].map((b) => b.textContent), ['1/8', '1/8T', '1/16', '1/16T', '1/32']);
  assert.equal(sw.querySelector('.btn.on').textContent, '1/16');
  sw.querySelectorAll('.btn')[4].click();
  wheel(sw, 120);
  const list = root.querySelector('px-list');
  assert.equal(list.querySelector('.lbox').textContent, '4x');
  list.querySelector('.lbox').click();
  const items = [...root.querySelectorAll('.px-menu .it')];
  assert.deepEqual(items.map((i) => i.textContent), ['2x', '3x', '4x', '5x', '6x', '7x', '8x']);
  items[4].click();
  await flush();
  assert.deepEqual(sets().map((c) => [c.address, c.value]),
    [['ch1.mute', true], ['global.step_rate', '1/16T'], ['global.fill_rate', 6]]);
});

test('the display shows a touch for a while, then its base text', async () => {
  const { root } = setup('<px-display></px-display>');
  const display = root.querySelector('px-display');
  display.setBase('PATTERN A', 'KIT');
  assert.deepEqual(display.text(), ['PATTERN A', 'KIT']);
  display.show('CH1 DECAY', '100 ms', 20);
  assert.deepEqual(display.text(), ['CH1 DECAY', '100 ms']);
  display.setBase('PATTERN B', 'KIT');
  assert.deepEqual(display.text(), ['CH1 DECAY', '100 ms']);
  await new Promise((r) => setTimeout(r, 40));
  assert.deepEqual(display.text(), ['PATTERN B', 'KIT']);
  display.alert('AUDIO', 'stream stalled', 20);
  assert.ok(display.classList.contains('alert'));
});

test('a control bound to an address the store lacks reads it', async () => {
  const { root, bridge, store } = setup('<px-knob></px-knob>');
  bridge.values['ch1.fx.reverb_mix'] = 0.25;
  const knob = root.querySelector('px-knob');
  knob.dataset.address = 'ch1.osc.decay';
  assert.ok(!bridge.calls.some(([s]) => s === 'get'));
  store.seed({});
  knob.dataset.address = 'ch1.mix.pan';
  delete bridge.values['ch1.mix.pan'];
  knob.dataset.address = 'ch1.fx.reverb_mix';
  await new Promise((r) => setTimeout(r, 0));
  assert.deepEqual(bridge.calls.filter(([s]) => s === 'get').map(([, p]) => p), [['ch1.fx.reverb_mix']]);
  assert.equal(store.value('ch1.fx.reverb_mix'), 0.25);
});

// ---------------------------------------------------------------- edit rack additions (W4)

const MIXM = { address: 'ch1.mix.osc_noise', kind: 'float', minimum: 0, maximum: 1, default: 0.5, unit: 'ratio',
  curve: 'linear', labels: [], readonly: false };
const DELAY = { address: 'ch1.fx.delay_time', kind: 'enum', minimum: null, maximum: null, default: 'eighth', unit: '',
  curve: 'linear', readonly: false, labels: ['whole', 'half', 'quarter', 'eighth', 'sixteenth', 'thirtysecond',
    'half_t', 'quarter_t', 'eighth_t', 'sixteenth_t', 'half_d', 'quarter_d', 'eighth_d', 'sixteenth_d'] };
const TARGET = { address: 'ch1.lfo1.target', kind: 'enum', minimum: null, maximum: null, default: 'none', unit: '',
  curve: 'linear', readonly: false, labels: ['none', 'osc_frequency', 'osc_attack', 'osc_decay'] };
Object.assign(META, { [MIXM.address]: MIXM, [DELAY.address]: DELAY, [TARGET.address]: TARGET });
const DELAY_NAMES = '1/1,1/2,1/4,1/8,1/16,1/32,1/2T,1/4T,1/8T,1/16T,1/2.,1/4.,1/8.,1/16.';

test('a reversed horizontal fader puts the high end on the left and drags sideways', async () => {
  const { root, store, flush, gestures } = setup(
    '<px-fader data-address="ch1.mix.osc_noise" orient="h" reverse ends="osc,noise" length="100"></px-fader>',
    { 'ch1.mix.osc_noise': 0.75 });
  const fader = root.querySelector('px-fader');
  assert.ok(fader.classList.contains('h'));
  assert.deepEqual([...fader.querySelectorAll('.ends span')].map((s) => s.textContent), ['osc', 'noise']);
  assert.equal(fader.querySelector('.cap').style.left, '25%');
  assert.equal(fader.querySelector('.val').textContent, '75 %');
  const cap = fader.querySelector('.cap').getBoundingClientRect();
  pointer(fader.querySelector('.cap'), 'pointerdown', { x: cap.left, y: cap.top });
  pointer(fader, 'pointermove', { x: cap.left + 20, y: cap.top }); // 20 % of 100 px, towards noise
  pointer(fader, 'pointerup', {});
  await flush();
  assert.equal(store.value('ch1.mix.osc_noise').toFixed(2), '0.55');
  assert.deepEqual(gestures(), ['begin', 'end']);
  const track = fader.querySelector('.track').getBoundingClientRect();
  pointer(fader.querySelector('.track'), 'pointerdown', { x: track.left + track.width * 0.9, y: track.top });
  pointer(fader, 'pointerup', {});
  assert.equal(store.value('ch1.mix.osc_noise').toFixed(2), '0.10');
});

test('a switch offers a subset of the labels and shows a value outside it', async () => {
  const { root, store, flush, sets } = setup(`<px-switch data-address="ch1.fx.delay_time"
    options="${DELAY_NAMES}" values="quarter,eighth,sixteenth,eighth_t,quarter_d"></px-switch>`,
  { 'ch1.fx.delay_time': 'eighth' });
  const sw = root.querySelector('px-switch');
  const buttons = () => [...sw.querySelectorAll('.btn')];
  assert.deepEqual(buttons().map((b) => b.textContent), ['1/4', '1/8', '1/16', '1/8T', '1/4.']);
  assert.equal(sw.querySelector('.btn.on').textContent, '1/8');
  assert.equal(sw.querySelector('.other').style.display, 'none');
  wheel(sw, -120); // up: the next offered one
  assert.equal(store.value('ch1.fx.delay_time'), 'sixteenth');
  await flush();
  store.assume('ch1.fx.delay_time', 'thirtysecond'); // from a preset
  assert.equal(sw.querySelector('.btn.on'), null);
  assert.equal(sw.querySelector('.other').textContent, '1/32');
  assert.notEqual(sw.querySelector('.other').style.display, 'none');
  buttons()[3].click();
  await flush();
  assert.deepEqual(sets().map((c) => c.value), ['sixteenth', 'eighth_t']);
});

test('a destination button names its target, steps with the wheel and asks to assign', async () => {
  const { root, store, flush, sets } = setup(
    '<px-target data-address="ch1.lfo1.target" name="lfo 1 destination"></px-target>',
    { 'ch1.lfo1.target': 'osc_decay' });
  const dest = root.querySelector('px-target');
  assert.equal(dest.querySelector('button').textContent, '→ osc decay');
  const asked = [];
  root.addEventListener('px-assign', (e) => asked.push(e.detail.address));
  dest.querySelector('button').click();
  assert.deepEqual(asked, ['ch1.lfo1.target']);
  wheel(dest, 120); // down: the previous target in engine order
  assert.equal(store.value('ch1.lfo1.target'), 'osc_attack');
  await flush();
  dest.dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
  const items = [...root.querySelectorAll('.px-menu .it')];
  assert.deepEqual(items.map((i) => i.textContent), ['Assign: click a control', 'Clear destination']);
  items[1].click();
  assert.equal(dest.querySelector('button').textContent, '→ off');
  assert.ok(dest.classList.contains('none'));
  await flush();
  assert.deepEqual(sets().map((c) => c.value), ['osc_attack', 'none']);
});

test('controls bound in one go read their values in one get', async () => {
  const { root, bridge, store } = setup('<px-knob></px-knob><px-knob></px-knob>');
  const [a, b] = root.querySelectorAll('px-knob');
  bridge.values['ch1.mix.osc_noise'] = 0.3;
  bridge.values['ch1.lfo1.target'] = 'osc_decay';
  a.dataset.address = 'ch1.mix.osc_noise';
  b.dataset.address = 'ch1.lfo1.target';
  await new Promise((r) => setTimeout(r, 0));
  assert.deepEqual(bridge.calls.filter(([s]) => s === 'get').map(([, p]) => p),
    [['ch1.mix.osc_noise', 'ch1.lfo1.target']]);
  assert.equal(store.value('ch1.mix.osc_noise'), 0.3);
});

test('a modulation band ends at the modulated position', () => {
  const { root, bridge } = setup('<px-knob data-address="ch1.osc.decay"></px-knob>'
    + '<px-fader data-address="ch1.mix.osc_noise" orient="h" reverse length="100"></px-fader>',
  { 'ch1.lfo1.on': true, 'ch1.lfo1.target': 'osc_decay', 'ch1.lfo2.on': true, 'ch1.lfo2.target': 'osc_noise_mix',
    'ch1.mix.osc_noise': 0.5 });
  const channels = Array(8).fill(null).map(() => ({}));
  channels[0] = { osc_decay: 900, osc_noise_mix: 0.25 };
  bridge.pushFrame({ modulation: { channel: 0, offsets: channels[0], channels } });
  const knob = root.querySelector('px-knob');
  assert.equal(knob.dataset.modTo, (Math.log(100) / Math.log(1000)).toFixed(4)); // 100 + 900 ms
  const fader = root.querySelector('px-fader');
  assert.equal(fader.dataset.mod, 'lfo2');
  assert.equal(fader.dataset.modTo, '0.7500');
  assert.equal(fader.querySelector('.mod').style.left, '25%');
});
