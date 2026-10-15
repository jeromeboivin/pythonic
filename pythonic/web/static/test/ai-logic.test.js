// Pure logic behind the AI generator page: lanes, the patterns column, the
// header, the model problems, the leave question and what keep says.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  aiAddresses, fileName, hasTrial, keptText, laneAddress, laneView, leaveQuestion, modelProblem,
  patternView, stateText, trialText,
} from '../js/ai-logic.js';

test('lane addresses and every address the page reads', () => {
  assert.equal(laneAddress(3, 'trying'), 'ai.ch3.trying');
  const all = aiAddresses();
  assert.equal(all.length, 10 + 8 * 7);
  assert.ok(all.includes('ai.ch8.error') && all.includes('pref.ai.pattern_temperature'));
});

test('a lane without a sound, generating, failed, with a new sound and back on the old one', () => {
  assert.deepEqual(laneView({ candidates: 0, candidate: 0 }), {
    state: 'empty', name: '', canSwitch: false, trying: false, note: '',
  });
  const busy = laneView({ candidates: 0, generating: true });
  assert.equal(busy.state, 'generating');
  assert.equal(busy.note, 'generating…');
  const failed = laneView({ candidates: 0, error: 'no model' });
  assert.equal(failed.state, 'error');
  assert.equal(failed.note, 'error: no model');
  const trying = laneView({ candidates: 1, candidate: 1, name: 'BD 1', trying: true });
  assert.deepEqual(trying, { state: 'ready', name: 'BD 1', canSwitch: true, trying: true, note: 'the new sound plays' });
  assert.equal(laneView({ candidates: 1, name: 'BD 1', trying: false }).note, 'the old sound plays');
  // an old error stays hidden while the lane has a sound; regenerating blocks old | new
  assert.equal(laneView({ candidates: 1, error: 'x' }).state, 'ready');
  assert.equal(laneView({ candidates: 1, generating: true }).canSwitch, false);
});

test('the patterns column: none, generating, new patterns playing, back on the old ones', () => {
  assert.deepEqual(patternView('none', 'off'), {
    canSwitch: false, trying: false, generating: false, note: 'make 12 patterns (A–L) for the sounds on the face',
  });
  assert.equal(patternView('generating', 'off').generating, true);
  assert.equal(patternView('generating', 'off').note, 'generating patterns…');
  const trying = patternView('ready', 'try');
  assert.ok(trying.canSwitch && trying.trying);
  assert.match(trying.note, /^the new patterns play/);
  assert.equal(patternView('ready', 'off').note, 'the old patterns play');
});

test('what is new on the face', () => {
  assert.ok(!hasTrial([], 'off'));
  assert.ok(hasTrial([2], 'off'));
  assert.ok(hasTrial([], 'try'));
  assert.ok(!hasTrial(null, 'loop'));
  assert.equal(trialText([], 'off'), '');
  assert.equal(trialText([1, 3], 'off'), 'new sounds on ch 1, 3');
  assert.equal(trialText([2], 'try'), 'new sounds on ch 2 + new patterns');
  assert.equal(trialText([], 'try'), 'new patterns');
});

test('the header state and the model problems', () => {
  assert.equal(stateText('idle'), '');
  assert.equal(stateText('generating'), 'generating…');
  assert.equal(stateText('unavailable'), 'ML extras missing');
  assert.equal(fileName('/a/b/drum_cvae_best.pt'), 'drum_cvae_best.pt');
  assert.equal(fileName('C:\\m\\p.pt'), 'p.pt');
  assert.equal(fileName(null), '');
  const ok = { path: '/m/p.pt', status: 'loaded' };
  assert.equal(modelProblem({ patch: ok, pattern: { ...ok, status: 'unloaded' } }), '');
  assert.equal(modelProblem(null), '');
  assert.equal(modelProblem({ patch: { path: null, status: 'missing' } }), 'no sound model: set one in models…');
  assert.equal(modelProblem({ patch: ok, pattern: { path: '/m/q.pt', status: 'error', error: 'broken' } }),
    'pattern model: broken');
  assert.equal(modelProblem({ pattern: { path: '/m/q.pt', status: 'error' } }), 'pattern model: q.pt');
});

test('the leave question names the new sounds and patterns', () => {
  assert.deepEqual(leaveQuestion([2], { 2: 'SD 4' }), {
    title: 'Keep what is new?',
    text: 'CH2 SD 4 plays on the face. Keep (one undo step), or revert to what was there.',
  });
  assert.match(leaveQuestion([1, 3], { 1: 'BD 1' }).text, /^CH1 BD 1 and CH3 play on the face/);
  assert.match(leaveQuestion([1], {}, true).text, /^CH1 and the 12 new patterns play/);
  assert.match(leaveQuestion([], {}, true).text, /^The 12 new patterns play on the face/);
});

test('what keep says', () => {
  assert.equal(keptText({ kept: [3] }), 'kept ch 3');
  assert.equal(keptText({ kept: [1, 3] }), 'kept 2 sounds');
  assert.equal(keptText({ kept: [1, 2], patterns: 12 }), 'kept 2 sounds + 12 patterns');
  assert.equal(keptText({ kept: [], patterns: 12 }), 'kept 12 patterns');
  assert.equal(keptText({ kept: [] }), 'nothing new');
});
