// Needs the DOM: the step row, step entry, matrix and pattern controls of the
// panel over a fake bridge (runs inside QtWebEngine only).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createFakeBridge } from '../js/bridge.js';
import { createCoreClient } from '../js/core-client.js';
import { createDrawer } from '../js/drawer.js';
import { mountPanel } from '../js/panel.js';
import { createStore } from '../js/store.js';
import { patternValues, stepValues } from './pattern-fixtures.js';

async function setup(values = {}, actions = {}) {
  const all = { 'global.channel': 1, 'preset.name': 'Kit', 'undo.can_undo': false, 'undo.can_redo': false,
    'program.current': 1, 'program.occupied': Array(16).fill(false), 'morph.learning': 'off',
    'pref.ui.ctrl_knob': null, 'ai.available': false, ...patternValues(), ...stepValues() };
  for (let n = 1; n <= 8; n += 1) { all[`ch${n}.mute`] = false; all[`ch${n}.name`] = `Drum ${n}`; }
  all['ch1.name'] = '808 BD';
  Object.assign(all, values);
  const bridge = createFakeBridge({ values: all, actions });
  const queued = [];
  const client = createCoreClient(bridge, { schedule: (fn) => queued.push(fn) });
  const store = createStore();
  client.onFrame((f) => store.apply(f));
  const stage = document.createElement('div');
  stage.style.cssText = 'position:absolute;width:1600px;height:1000px';
  document.body.append(stage);
  const panel = mountPanel(stage, { store, client, meta: {} });
  store.seed(all);
  const flush = async () => { queued.splice(0).forEach((fn) => fn()); await client.flush(); await tick(); };
  const calls = () => bridge.calls.filter(([s]) => s === 'set' || s === 'gesture' || s === 'act')
    .flatMap(([s, p]) => (s === 'set' ? p.map((c) => ['set', c.address, c.value]) : s === 'act' ? [['act', p.verb, p.args]] : [[s, p]]));
  const sets = () => calls().filter(([k]) => k === 'set').map(([, a, v]) => [a, v]);
  const verbs = () => calls().filter(([k]) => k === 'act').map(([, v, a]) => [v, a]);
  const $ = (sel) => stage.querySelector(sel);
  const $$ = (sel) => [...stage.querySelectorAll(sel)];
  const pad = (step) => $(`.pads .stp[data-step="${step}"]`);
  const settle = async () => { await tick(); bridge.pushFrame(); await tick(); };
  await tick(); // the step row renders the seeded lanes
  return { bridge, store, stage, panel, flush, calls, sets, verbs, $, $$, pad, settle };
}

// A task boundary that hidden pages do not throttle (setTimeout waits up to 1 s there)
const tick = () => new Promise((r) => { const c = new MessageChannel(); c.port1.onmessage = () => r(); c.port2.postMessage(0); });
const lane = (store, field, ch = 1, p = 'A') => store.value(`pattern.${p}.ch${ch}.${field}`);
const point = (type, target, { x = 10, y = 500, shift = false } = {}) => target.dispatchEvent(new PointerEvent(type,
  { bubbles: true, cancelable: true, button: 0, pointerId: 1, clientX: x, clientY: y, shiftKey: shift }));
const press = (target, opts) => { point('pointerdown', target, opts); point('pointerup', target, opts); };
const items = (stage) => [...stage.querySelectorAll('.px-menu .it')];
const choose = (stage, text) => items(stage).find((i) => i.textContent === text).click();

function withSteps(length = 16) {
  const v = patternValues(length);
  const set = (f, i, x, ch = 1) => { v[`pattern.A.ch${ch}.${f}`][i] = x; };
  set('trig', 0, true); set('acc', 0, true);
  set('trig', 4, true); set('vel', 4, 32);
  set('trig', 8, true); set('fill', 8, true);
  set('prob', 12, 50);
  set('trig', 2, true); set('sub', 2, 'o-o');
  set('trig', 6, true, 2);
  return v;
}

// ---------------------------------------------------------------- what pads show

test('pads show every property of the selected channel, whatever the mode', async () => {
  const { pad } = await setup(withSteps());
  assert.ok(pad(1).classList.contains('on') && pad(1).classList.contains('accent'));
  assert.equal(pad(1).style.getPropertyValue('--lvl'), '1.000');
  assert.ok(pad(5).classList.contains('on') && !pad(5).classList.contains('accent'));
  assert.ok(Number(pad(5).style.getPropertyValue('--lvl')) < 0.55); // velocity 32: dimmer
  assert.ok(pad(9).classList.contains('filled'));
  assert.equal(pad(13).querySelector('.prob').textContent, '50%');
  assert.ok(!pad(13).classList.contains('on'));
  assert.deepEqual([...pad(3).querySelectorAll('.sub i')].map((i) => i.className), ['', 'x', '']);
  assert.ok(!pad(7).classList.contains('on'));
  assert.deepEqual([1, 5, 9, 13].map((s) => pad(s).className.match(/\bg\d\b/)[0]), ['g1', 'g2', 'g3', 'g4']);
});

test('selecting another channel shows its lane', async () => {
  const { pad, store } = await setup(withSteps());
  store.assume('global.channel', 2);
  await tick();
  assert.ok(pad(7).classList.contains('on') && !pad(1).classList.contains('on'));
});

test('the velocity and probability modes add a bar and the value', async () => {
  const { pad, $ } = await setup(withSteps());
  $('[data-mode="vel"]').click();
  assert.equal(pad(5).querySelector('.pv').textContent, '32');
  assert.equal(pad(1).querySelector('.pv').textContent, 'ACC');
  assert.equal(pad(2).querySelector('.pv').textContent, ''); // no trigger
  $('[data-mode="prob"]').click();
  assert.equal(pad(13).querySelector('.pv').textContent, '50%');
  assert.equal(pad(13).querySelector('.bar i').style.height, '50%');
  assert.ok($('[data-mode="prob"]').classList.contains('on'));
});

// ---------------------------------------------------------------- edits

test('a trig press toggles the step as one gesture and echoes the lane', async () => {
  const { pad, calls, store, flush, panel } = await setup();
  press(pad(3));
  assert.ok(pad(3).classList.contains('on'));
  assert.equal(lane(store, 'trig')[2], true);
  await flush();
  assert.deepEqual(calls(), [['gesture', 'begin'], ['set', 'pattern.A.ch1.step3.trig', true], ['gesture', 'end']]);
  assert.deepEqual(panel.display.text(), ['CH1 STEP 3', 'trigger on']);
});

test('turning a trigger off clears its accent and fill in the echo', async () => {
  const { pad, store } = await setup(withSteps());
  press(pad(1));
  assert.equal(lane(store, 'trig')[0], false);
  assert.equal(lane(store, 'acc')[0], false);
  assert.ok(!pad(1).classList.contains('accent'));
});

test('accent and fill toggle only triggered steps', async () => {
  const { pad, $, sets, flush } = await setup(withSteps());
  $('[data-mode="acc"]').click();
  press(pad(2)); // no trigger
  press(pad(5));
  $('[data-mode="fill"]').click();
  press(pad(9));
  await flush();
  assert.deepEqual(sets(), [['pattern.A.ch1.step5.acc', true], ['pattern.A.ch1.step9.fill', false]]);
  assert.ok(pad(5).classList.contains('accent') && !pad(9).classList.contains('filled'));
});

test('a velocity drag on a triggered pad: 200 px = the range, one gesture', async () => {
  const { pad, $, calls, flush, store } = await setup(withSteps());
  $('[data-mode="vel"]').click();
  point('pointerdown', pad(5), { y: 500 });
  point('pointermove', pad(5), { y: 450 }); // 50 px up = +31.5
  point('pointermove', pad(5), { y: 400 });
  point('pointerup', pad(5), { y: 400 });
  await flush();
  assert.equal(lane(store, 'vel')[4], 95);
  assert.equal(pad(5).querySelector('.pv').textContent, '95');
  assert.deepEqual(calls(), [['gesture', 'begin'], ['set', 'pattern.A.ch1.step5.vel', 95], ['gesture', 'end']]);
});

test('no velocity drag on an empty pad; probability drags on any pad', async () => {
  const { pad, $, sets, flush } = await setup(withSteps());
  $('[data-mode="vel"]').click();
  point('pointerdown', pad(2), { y: 500 });
  point('pointermove', pad(2), { y: 400 });
  point('pointerup', pad(2), { y: 400 });
  $('[data-mode="prob"]').click();
  point('pointerdown', pad(2), { y: 500 });
  point('pointermove', pad(2), { y: 600, shift: true }); // re-bases for fine
  point('pointermove', pad(2), { y: 700, shift: true }); // 100 px fine = -5 %
  point('pointerup', pad(2), { y: 700 });
  await flush();
  assert.deepEqual(sets(), [['pattern.A.ch1.step2.prob', 95]]);
});

test('the substep menu: none, the 14 presets and a custom entry', async () => {
  const { pad, $, stage, sets, flush } = await setup(withSteps());
  $('[data-mode="sub"]').click();
  press(pad(3));
  const texts = items(stage).map((i) => i.textContent);
  assert.equal(texts[0], 'substeps · step 3');
  assert.equal(texts.length, 17);
  assert.deepEqual(texts.slice(1, 4), ['none', 'oo', 'o-']);
  assert.ok(items(stage).find((i) => i.textContent === 'o-o').classList.contains('cur'));
  choose(stage, 'oooo');
  press(pad(4));
  choose(stage, 'custom…');
  const input = stage.querySelector('.px-entry input');
  input.value = 'O-x-';
  input.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
  await flush();
  assert.deepEqual(sets(), [['pattern.A.ch1.step3.sub', 'oooo'], ['pattern.A.ch1.step4.sub', 'o--']]);
  assert.equal(pad(3).querySelectorAll('.sub i').length, 4);
});

test('all ch: an edit hits all 8 channels, muted ones included, as one gesture', async () => {
  const { pad, $, calls, flush } = await setup(withSteps());
  $('#all-ch').click();
  press(pad(6));
  await flush();
  const c = calls();
  assert.deepEqual(c[0], ['gesture', 'begin']);
  assert.deepEqual(c.at(-1), ['gesture', 'end']);
  assert.deepEqual(c.slice(1, -1).map(([, a]) => a), [1, 2, 3, 4, 5, 6, 7, 8].map((n) => `pattern.A.ch${n}.step6.trig`));
});

test('all ch accent reaches only the channels triggered there', async () => {
  const v = withSteps();
  v['pattern.A.ch4.trig'][4] = true;
  const { pad, $, sets, flush } = await setup(v);
  $('#all-ch').click();
  $('[data-mode="acc"]').click();
  press(pad(5));
  await flush();
  assert.deepEqual(sets(), [['pattern.A.ch1.step5.acc', true], ['pattern.A.ch4.step5.acc', true]]);
});

test('last step arms; the next pad on any page ends the pattern', async () => {
  const { pad, $, $$, sets, flush, store } = await setup(withSteps(16));
  $('#last-step').click();
  assert.ok($('#last-step').classList.contains('on'));
  $$('.pg')[1].click(); // a page past the length shows, dimmed
  assert.ok($$('.pg')[1].classList.contains('out'));
  assert.ok(pad(20).classList.contains('out'));
  press(pad(20));
  await flush();
  assert.deepEqual(sets(), [['pattern.A.length', 20]]);
  assert.ok(!$('#last-step').classList.contains('on'));
  assert.equal(store.value('pattern.A.length'), 20);
  assert.ok(!pad(20).classList.contains('out') && pad(21).classList.contains('out'));
  assert.equal($('.pnums .end').textContent, '20]');
  assert.equal($('#last-step').dataset.address, 'pattern.A.length');
});

test('steps past the length do not edit', async () => {
  const { pad, $$, sets, flush } = await setup();
  $$('.pg')[2].click();
  press(pad(35));
  await flush();
  assert.deepEqual(sets(), []);
});

// ---------------------------------------------------------------- pages and follow

test('page bars switch the pads and turn follow off', async () => {
  const { pad, $, $$, panel } = await setup(withSteps(64));
  assert.ok($('#follow').classList.contains('on'));
  $$('.pg')[3].click();
  assert.equal($('.pads .stp').dataset.step, '49');
  assert.equal($('.pnums').firstChild.textContent, '49');
  assert.ok($$('.pg')[3].classList.contains('on') && !$('#follow').classList.contains('on'));
  assert.ok(pad(64).dataset.step === '64');
  assert.deepEqual(panel.display.text(), ['PAGE', 'steps 49-64']);
});

test('follow keeps the pads on the playing page; the playhead only on its pattern', async () => {
  const { bridge, $, $$, pad, panel } = await setup(withSteps(64));
  Object.assign(bridge.transport, { playing: true, position: 37 });
  bridge.pushFrame();
  assert.equal($('.pads .stp').dataset.step, '33');
  assert.ok(pad(38).classList.contains('ph'));
  assert.ok($$('.pg')[2].classList.contains('play') && $$('.pg')[2].classList.contains('on'));
  assert.deepEqual(panel.display.base, ['PATTERN A  33-48', 'KIT']);
  Object.assign(bridge.transport, { playing_pattern: 1 }); // B plays, A is edited
  bridge.pushFrame();
  assert.equal($$('.pads .ph').length, 0);
});

// ---------------------------------------------------------------- matrix

test('⊞ matrix shows all channels of the page in the drawer and edits them', async () => {
  const { $, $$, panel, sets, flush, store } = await setup(withSteps());
  $('#matrix-toggle').click();
  assert.equal(panel.drawer.current, 'matrix');
  assert.ok($('[data-slot="rack"] .matrix') && $('#matrix-toggle').classList.contains('on'));
  const cell = (ch, step) => $(`.matrix .stp[data-channel="${ch}"][data-step="${step}"]`);
  assert.ok(cell(1, 1).classList.contains('on') && cell(2, 7).classList.contains('on'));
  assert.ok(!cell(2, 1).classList.contains('on'));
  press(cell(3, 2));
  assert.ok(cell(3, 2).classList.contains('on'));
  $$('.mlbl')[4].click();
  await flush();
  assert.deepEqual(sets(), [['pattern.A.ch3.step2.trig', true], ['global.channel', 5]]);
  assert.equal(store.value('global.channel'), 5);
  assert.ok($$('.mlbl')[4].classList.contains('sel'));
  $('#matrix-toggle').click();
  assert.equal(panel.drawer.current, null);
  assert.equal($('[data-slot="rack"] .matrix'), null);
});

test('the drawer: a page opens it when closed and closing restores it', async () => {
  const slot = document.createElement('div');
  const base = document.createElement('p');
  const drawer = createDrawer(slot);
  let open = false;
  drawer.setOpener({ isOpen: () => open, setOpen: (o) => { open = o; } });
  drawer.setBase(base);
  assert.equal(slot.firstChild, base);
  const page = document.createElement('div');
  assert.equal(drawer.toggle('matrix', () => page), true);
  assert.ok(open && slot.firstChild === page && drawer.current === 'matrix');
  const other = document.createElement('div');
  drawer.show('po32', other);
  assert.ok(open && slot.firstChild === other);
  drawer.hide();
  assert.ok(!open && slot.firstChild === base && drawer.current === null);
  open = true;
  drawer.toggle('matrix', () => page);
  assert.equal(drawer.toggle('matrix', () => page), false);
  assert.ok(open);
});

// ---------------------------------------------------------------- patterns

test('pattern buttons: selected, playing, queued, empty, chained and long', async () => {
  const v = withSteps();
  v['pattern.A.empty'] = false;
  v['pattern.C.chained'] = true;
  v['pattern.E.length'] = 48;
  const { bridge, $$ } = await setup(v);
  Object.assign(bridge.transport, { playing: true, playing_pattern: 0, selected_pattern: 1, queued_pattern: 3 });
  bridge.pushFrame();
  const b = $$('.pbtn');
  const has = (cls) => b.filter((x) => x.classList.contains(cls)).map((x) => x.dataset.pattern).join('');
  assert.equal(has('on'), 'B');
  assert.equal(has('playing'), 'A');
  assert.equal(has('queued'), 'D');
  assert.equal(b.filter((x) => !x.classList.contains('empty')).map((x) => x.dataset.pattern).join(''), 'A');
  assert.equal(has('chain-out'), 'C');
  assert.equal(has('chain-in'), 'D');
  assert.equal(b[4].querySelector('.len').textContent, '48');
  assert.equal(b[0].querySelector('.len').textContent, '');
});

test('a pattern button selects through the verb; the pads follow the selection', async () => {
  const v = withSteps();
  v['pattern.B.ch1.trig'][10] = true;
  const { $$, verbs, sets, pad, bridge, flush } = await setup(v, { 'pattern.select': (a, f) => { f.transport.selected_pattern = 'ABCDEFGHIJKL'.indexOf(a.pattern); } });
  $$('.pbtn')[1].click();
  await tick();
  bridge.pushFrame();
  assert.deepEqual(verbs(), [['pattern.select', { pattern: 'B' }]]);
  assert.ok($$('.pbtn')[1].classList.contains('on'));
  assert.ok(pad(11).classList.contains('on') && !pad(1).classList.contains('on'));
  press(pad(2));
  await flush();
  assert.deepEqual(sets(), [['pattern.B.ch1.step2.trig', true]]);
});

test('the pattern menu runs ops on the selected pattern and shows results', async () => {
  const { $, stage, verbs, panel, settle } = await setup({}, { 'pattern.paste': () => ({ pasted: false }) });
  $('#pattern-menu').click();
  const texts = items(stage).map((i) => i.textContent);
  for (const t of ['PATTERN A', 'cut', 'copy', 'paste', 'exchange', 'clear', 'shift left', 'shift right', 'reverse',
    'randomize', 'alter', 'randomize accents + fills', 'randomize (AI)', 'randomize ch1 (AI)', 'copy lane ch1',
    'paste lane ch1', 'clear all chains']) assert.ok(texts.includes(t), t);
  assert.ok(!texts.includes('play next (queue)'));
  assert.ok(items(stage).find((i) => i.textContent === 'randomize (AI)').classList.contains('dis'));
  choose(stage, 'reverse');
  await settle();
  assert.deepEqual(panel.display.text(), ['PATTERN A', 'reverse']);
  panel.patterns.openMenu(0, 10, 10);
  choose(stage, 'paste');
  await settle();
  assert.deepEqual(verbs(), [['pattern.reverse', { pattern: 'A' }], ['pattern.paste', { pattern: 'A' }]]);
  assert.deepEqual(panel.display.text(), ['PATTERN A', 'nothing to paste']);
});

test('right-click opens the menu of that pattern; queue while playing; AI entries', async () => {
  const { $$, stage, verbs, bridge, store } = await setup({ 'ai.available': true });
  Object.assign(bridge.transport, { playing: true });
  bridge.pushFrame();
  store.assume('global.channel', 3);
  $$('.pbtn')[2].dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true, clientX: 5, clientY: 5 }));
  assert.equal(items(stage)[0].textContent, 'PATTERN C');
  choose(stage, 'play next (queue)');
  $$('.pbtn')[2].dispatchEvent(new MouseEvent('contextmenu', { bubbles: true, cancelable: true }));
  choose(stage, 'randomize ch3 (AI)');
  await tick();
  assert.deepEqual(verbs(), [['pattern.queue', { pattern: 'C' }], ['ai.randomize_pattern', { pattern: 'C', channel: 3 }]]);
});

test('chain buttons link the selected pattern and show the chain', async () => {
  const { $, verbs, panel, bridge } = await setup({}, { 'pattern.chain_next': (_a, f) => {
    f.values['pattern.A.chained'] = true;
    bridge.pushFrame({ changes: { 'pattern.A.chained': true } });
    return { chained: true };
  } });
  assert.ok($('#chain-prev').disabled);
  $('#chain-next').click();
  await tick();
  bridge.pushFrame();
  await tick();
  assert.deepEqual(verbs(), [['pattern.chain_next', { pattern: 'A' }]]);
  assert.deepEqual(panel.display.text(), ['CHAIN', 'A-B']);
  assert.ok($('.pbtn[data-pattern="B"]').classList.contains('chain-in'));
});

test('copy and paste work on the selected channel lane; extra menu entries', async () => {
  const { $, stage, verbs, panel, store, settle } = await setup({}, { 'pattern.paste_lane': () => ({ pasted: true }) });
  store.assume('global.channel', 4);
  $('#lane-copy').click();
  await settle();
  assert.deepEqual(panel.display.text(), ['LANE CH4', 'copied']);
  $('#lane-paste').click();
  await settle();
  assert.deepEqual(verbs(), [['pattern.copy_lane', { pattern: 'A', channel: 4 }],
    ['pattern.paste_lane', { pattern: 'A', channel: 4 }]]);
  assert.deepEqual(panel.display.text(), ['LANE CH4', 'pasted']);
  let picked = null;
  panel.patterns.addMenuItems((letter) => [[`export ${letter}`, () => { picked = letter; }]]);
  $('#pattern-menu').click();
  choose(stage, 'export A');
  assert.equal(picked, 'A');
});
