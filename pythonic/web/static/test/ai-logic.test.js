// Pure logic behind the AI drum generator page: lanes, model lines, the
// pattern bank note, preview arguments, candidates, seed, the leave question.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  aiAddresses, bankText, canReplacePatterns, clampCandidates, fileName, keptText, laneAddress,
  laneView, leaveQuestion, modelView, parseSeed, previewArgs, randomSeed, stateText,
} from '../js/ai-logic.js';

test('lane addresses and every address the page reads', () => {
  assert.equal(laneAddress(3, 'trying'), 'ai.ch3.trying');
  const all = aiAddresses();
  assert.equal(all.length, 12 + 8 * 7);
  assert.ok(all.includes('ai.ch8.error') && all.includes('pref.ai.pattern_temperature'));
});

test('a lane without candidates, generating, failed, ready and trying', () => {
  assert.deepEqual(laneView({ candidates: 0, candidate: 0 }), {
    state: 'empty', counter: '–/–', name: '', canStep: false, canTry: false, tryLabel: 'try',
    trying: false, note: 'no candidates',
  });
  const busy = laneView({ candidates: 0, generating: true });
  assert.equal(busy.state, 'generating');
  assert.equal(busy.note, 'generating…');
  const failed = laneView({ candidates: 0, error: 'no model' });
  assert.equal(failed.state, 'error');
  assert.equal(failed.note, 'error: no model');
  const ready = laneView({ candidates: 8, candidate: 3, name: 'BD 3', trying: false });
  assert.equal(ready.state, 'ready');
  assert.equal(ready.counter, '3/8');
  assert.equal(ready.name, 'BD 3');
  assert.ok(ready.canStep && ready.canTry);
  assert.equal(ready.tryLabel, 'try');
  const trying = laneView({ candidates: 8, candidate: 1, name: 'BD 1', trying: true });
  assert.equal(trying.tryLabel, 'trying ✓');
  assert.equal(trying.note, 'the face plays this');
  // an old error stays hidden while the lane has candidates; regenerating blocks the arrows
  assert.equal(laneView({ candidates: 4, candidate: 1, error: 'x' }).state, 'ready');
  assert.equal(laneView({ candidates: 4, candidate: 1, generating: true }).canStep, false);
});

test('model lines', () => {
  assert.equal(fileName('/a/b/drum_cvae_best.pt'), 'drum_cvae_best.pt');
  assert.equal(fileName('C:\\m\\p.pt'), 'p.pt');
  assert.equal(fileName(null), '');
  assert.deepEqual(modelView({ path: '/m/p.pt', status: 'loaded', sampling: 'prior' }), { text: 'p.pt ✓ (prior)', tone: 'ok' });
  assert.deepEqual(modelView({ path: '/m/p.pt', status: 'loading' }), { text: 'loading p.pt…', tone: 'busy' });
  assert.deepEqual(modelView({ path: '/m/p.pt', status: 'error', error: 'broken' }), { text: 'error: broken', tone: 'error' });
  assert.deepEqual(modelView({ path: null, status: 'missing' }), { text: 'no model: load one', tone: 'error' });
  assert.deepEqual(modelView({ path: '/m/p.pt', status: 'unloaded' }), { text: 'p.pt (not loaded)', tone: 'dim' });
  assert.equal(modelView({ status: 'loaded' }, false).text, 'needs the ML extras');
});

test('the header state, the bank note and replace patterns', () => {
  assert.equal(stateText('idle'), '');
  assert.equal(stateText('generating'), 'generating…');
  assert.equal(stateText('unavailable'), 'ML extras missing');
  assert.equal(bankText('keep', 'ready'), 'the preset’s patterns stay');
  assert.equal(bankText('generate', 'none'), '(generate all 8 first)');
  assert.equal(bankText('generate', 'generating'), 'generating patterns…');
  assert.equal(bankText('generate', 'ready'), 'bank of 12 ready');
  assert.ok(canReplacePatterns('generate', 'ready'));
  assert.ok(!canReplacePatterns('keep', 'ready'));
  assert.ok(!canReplacePatterns('generate', 'none'));
});

test('a preview click starts its mode, or stops it when it runs', () => {
  assert.deepEqual(previewArgs('off', 'loop', 'keep'), { mode: 'loop', bank: false });
  assert.deepEqual(previewArgs('loop', 'bank', 'generate'), { mode: 'bank', bank: true });
  assert.deepEqual(previewArgs('bank', 'bank', 'generate'), { mode: null });
});

test('candidates and seed', () => {
  assert.equal(clampCandidates(0), 1);
  assert.equal(clampCandidates(40), 32);
  assert.equal(clampCandidates(7.6), 8);
  assert.equal(clampCandidates('x'), 8);
  assert.equal(parseSeed(''), null);
  assert.equal(parseSeed(' 42 '), 42);
  assert.equal(parseSeed('-3'), null);
  assert.equal(parseSeed('1.5'), null);
  assert.equal(parseSeed(String(2 ** 31)), null);
  assert.equal(randomSeed(() => 0), 0);
  assert.equal(randomSeed(() => 0.9999999999), 2 ** 31 - 1);
});

test('the leave question names the tried lanes', () => {
  assert.deepEqual(leaveQuestion([2], { 2: 'SD 4' }), {
    title: 'Keep the tried sounds?',
    text: 'CH2 SD 4 is trying AI candidates. Keep them (one undo step), or revert to the old sounds.',
  });
  assert.match(leaveQuestion([1, 3], { 1: 'BD 1' }).text, /^CH1 BD 1, CH3 are trying/);
  assert.equal(keptText({ kept: [1, 3] }), 'kept ch 1, 3');
  assert.equal(keptText({ kept: [] }), 'nothing tried');
});
