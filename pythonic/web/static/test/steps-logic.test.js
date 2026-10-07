// Pure logic of the step row, the step modes and the pattern buttons:
// addresses, what a pad shows, edits and their local echo, drag math, pages
// and follow, pattern button states and chains.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  chainOf, chainText, cleanSubsteps, dragValue, editChannels, followPage, groupOf, laneAddress,
  padView, pageCount, pageRange, patternStates, playheadIndex, stepAddress, stepAt, SUBSTEP_PRESETS,
  withStep,
} from '../js/steps-logic.js';

const lanes = (n = 16, over = {}) => ({
  trig: Array(n).fill(false), acc: Array(n).fill(false), vel: Array(n).fill(64),
  fill: Array(n).fill(false), prob: Array(n).fill(100), sub: Array(n).fill(''), ...over,
});

test('step and lane addresses', () => {
  assert.equal(stepAddress('C', 4, 17, 'vel'), 'pattern.C.ch4.step17.vel');
  assert.equal(laneAddress('L', 8, 'trig'), 'pattern.L.ch8.trig');
});

test('beat groups of four by the position on the page', () => {
  assert.deepEqual([1, 4, 5, 8, 9, 12, 13, 16].map(groupOf), [1, 1, 2, 2, 3, 3, 4, 4]);
  assert.deepEqual([17, 21, 29, 32, 64].map(groupOf), [1, 2, 4, 4, 4]);
});

test('pages of 16 steps', () => {
  assert.deepEqual([1, 16, 17, 32, 33, 64].map(pageCount), [1, 1, 2, 2, 3, 4]);
  assert.deepEqual([0, 1, 2, 3].map(pageRange), ['1-16', '17-32', '33-48', '49-64']);
});

test('a step reads its lanes; past the length it is out (defaults)', () => {
  const l = lanes(16);
  l.trig[2] = true; l.acc[2] = true; l.vel[2] = 90; l.prob[2] = 50; l.sub[2] = 'o-o';
  assert.deepEqual(stepAt(l, 2, 16), { trig: true, acc: true, vel: 90, fill: false, prob: 50, sub: 'o-o', out: false });
  assert.deepEqual(stepAt(l, 20, 16), { trig: false, acc: false, vel: 64, fill: false, prob: 100, sub: '', out: true });
  assert.deepEqual(stepAt({}, 0, 16), { trig: false, acc: false, vel: 64, fill: false, prob: 100, sub: '', out: false });
});

test('a pad shows every property, whatever the mode', () => {
  const on = { trig: true, acc: false, vel: 127, fill: true, prob: 25, sub: 'o-', out: false };
  const v = padView(on, 'trig');
  assert.equal(v.on, true);
  assert.equal(v.acc, false);
  assert.equal(v.fill, true);
  assert.equal(v.level, 1);
  assert.equal(v.prob, '25%');
  assert.deepEqual(v.sub, [true, false]);
  assert.equal(v.bar, null);
  assert.equal(v.text, null);
  // Velocity as brightness: 1 -> dim, 127 -> full; an accent always full
  assert.ok(padView({ ...on, vel: 1 }, 'trig').level < 0.4);
  assert.equal(padView({ ...on, vel: 1, acc: true }, 'trig').level, 1);
  // Accent and fill only show on a triggered step; full probability shows nothing
  const off = padView({ ...on, trig: false, acc: true, prob: 100 }, 'trig');
  assert.deepEqual([off.on, off.acc, off.fill, off.level, off.prob], [false, false, false, 0, null]);
});

test('the velocity and probability modes add a bar and the value', () => {
  const s = { trig: true, acc: false, vel: 32, fill: false, prob: 75, sub: '', out: false };
  assert.deepEqual([padView(s, 'vel').bar, padView(s, 'vel').text], [32 / 127, '32']);
  assert.deepEqual([padView({ ...s, acc: true }, 'vel').bar, padView({ ...s, acc: true }, 'vel').text], [1, 'ACC']);
  assert.deepEqual([padView({ ...s, trig: false }, 'vel').bar, padView({ ...s, trig: false }, 'vel').text], [null, null]);
  assert.deepEqual([padView({ ...s, trig: false }, 'prob').bar, padView(s, 'prob').text], [0.75, '75%']);
  assert.deepEqual(padView({ ...s, out: true }, 'prob'), { on: false, acc: false, fill: false, level: 0, prob: null,
    sub: [], bar: null, text: null, out: true });
});

test('a step edit echoes the lanes; a trigger turned off clears accent and fill', () => {
  const l = lanes(4, { trig: [true, true, false, false], acc: [false, true, false, false],
    fill: [false, true, false, false] });
  assert.deepEqual(withStep(l, 0, 'acc', true), { acc: [true, true, false, false] });
  assert.deepEqual(withStep(l, 1, 'trig', false), {
    trig: [true, false, false, false], acc: [false, false, false, false], fill: [false, false, false, false] });
  assert.deepEqual(withStep(l, 2, 'vel', 100), { vel: [64, 64, 100, 64] });
  assert.deepEqual(withStep(l, 9, 'trig', true), {}); // past the length
  assert.deepEqual(l.trig, [true, true, false, false]); // the input stays
});

test('which channels an edit reaches: all ch hits all 8, accent and fill only triggered steps', () => {
  const byChannel = {};
  for (let ch = 1; ch <= 8; ch += 1) byChannel[ch] = lanes(16, { trig: Array(16).fill(ch % 2 === 1) });
  assert.deepEqual(editChannels('trig', 3, 0, false, byChannel), [3]);
  assert.deepEqual(editChannels('trig', 3, 0, true, byChannel), [1, 2, 3, 4, 5, 6, 7, 8]);
  assert.deepEqual(editChannels('acc', 3, 0, true, byChannel), [1, 3, 5, 7]);
  assert.deepEqual(editChannels('fill', 2, 0, false, byChannel), []);
  assert.deepEqual(editChannels('vel', 2, 0, true, byChannel), [1, 2, 3, 4, 5, 6, 7, 8]);
  assert.deepEqual(editChannels('prob', 2, 0, false, byChannel), [2]);
});

test('velocity and probability drags: 200 px = the full range, Shift x0.1', () => {
  assert.equal(dragValue('vel', 64, 0), 64);
  assert.equal(dragValue('vel', 64, 100), 127);
  assert.equal(dragValue('vel', 64, -300), 1);
  assert.equal(dragValue('vel', 64, 50), 64 + Math.round(126 / 4));
  assert.equal(dragValue('prob', 100, -100), 50);
  assert.equal(dragValue('prob', 50, 100, true), 55);
  assert.equal(dragValue('prob', 0, -10), 0);
});

test('substeps: the 14 presets and the custom entry text', () => {
  assert.equal(SUBSTEP_PRESETS.length, 14);
  assert.deepEqual(SUBSTEP_PRESETS.slice(0, 3), ['oo', 'o-', '-o']);
  assert.equal(cleanSubsteps(' O-x o '), 'o-o');
  assert.equal(cleanSubsteps(''), '');
});

test('the playhead shows only on the pattern it plays', () => {
  const t = { playing: true, position: 21, playing_pattern: 2, selected_pattern: 2 };
  assert.equal(playheadIndex(t, 2), 21);
  assert.equal(playheadIndex(t, 1), -1);
  assert.equal(playheadIndex({ ...t, playing: false }, 2), -1);
  assert.equal(playheadIndex(undefined, 0), -1);
});

test('follow moves to the playing page', () => {
  const t = { playing: true, position: 37, playing_pattern: 0, selected_pattern: 0 };
  assert.equal(followPage(t, 0), 2);
  assert.equal(followPage({ ...t, position: 3 }, 0), 0);
  assert.equal(followPage(t, 1), null);
  assert.equal(followPage({ ...t, playing: false }, 0), null);
});

test('pattern buttons: selected, playing, queued, empty and chains', () => {
  const empty = Array(12).fill(true);
  empty[0] = false; empty[1] = false;
  const chained = Array(12).fill(false);
  chained[0] = true; // A -> B
  chained[2] = true; chained[3] = true; // C -> D -> E
  const t = { playing: true, playing_pattern: 0, selected_pattern: 1, queued_pattern: 4 };
  const s = patternStates(t, empty, chained);
  assert.deepEqual(s.map((x) => x.letter).join(''), 'ABCDEFGHIJKL');
  assert.deepEqual(s.filter((x) => x.selected).map((x) => x.letter), ['B']);
  assert.deepEqual(s.filter((x) => x.playing).map((x) => x.letter), ['A']);
  assert.deepEqual(s.filter((x) => x.queued).map((x) => x.letter), ['E']);
  assert.deepEqual(s.filter((x) => !x.empty).map((x) => x.letter), ['A', 'B']);
  assert.deepEqual(s.slice(0, 6).map((x) => [x.chainIn, x.chainOut]),
    [[false, true], [true, false], [false, true], [true, true], [true, false], [false, false]]);
  const stopped = patternStates({ ...t, playing: false }, empty, chained);
  assert.deepEqual(stopped.filter((x) => x.playing || x.queued), []);
});

test('the chain of a pattern and its text', () => {
  const chained = Array(12).fill(false);
  chained[2] = true; chained[3] = true;
  assert.deepEqual(chainOf(3, chained), [2, 3, 4]);
  assert.deepEqual(chainOf(0, chained), [0]);
  assert.equal(chainText([2, 3, 4]), 'C-D-E');
  assert.equal(chainText([0]), 'off');
});
