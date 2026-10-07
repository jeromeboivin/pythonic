// Pure logic of the PO-32 page: the stage flow of both tabs, the pattern
// picks (letters assigned in order, a conflict swaps, at most 12), the pattern
// buttons, the level meter and the stage texts.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  holdPeak, importButtonText, importStages, importText, letterChoices, levelText, levelTone, LETTERS, MAX_PICKS,
  patternButton, pick, slotsNote, sourceText, transferStages, transferText, unmutedChannels,
} from '../js/po32-logic.js';

test('transfer: choose is current until a send starts, then send; a sent transfer ticks all three', () => {
  assert.deepEqual(transferStages('none'), ['now', 'later', 'later']);
  assert.deepEqual(transferStages('ready'), ['now', 'later', 'later']);
  assert.deepEqual(transferStages('sending'), ['done', 'done', 'now']);
  assert.deepEqual(transferStages('stopped'), ['done', 'done', 'now']);
  assert.deepEqual(transferStages('error'), ['done', 'done', 'now']);
  assert.deepEqual(transferStages('sent'), ['done', 'done', 'done']);
});

test('import: listen until decoded, then pick, then import; an import ticks all four', () => {
  assert.deepEqual(importStages({}), ['now', 'later', 'later', 'later']);
  assert.deepEqual(importStages({ decoded: true, picks: [] }), ['done', 'done', 'now', 'later']);
  const picks = [{ pattern: 1, letter: 'A' }];
  assert.deepEqual(importStages({ decoded: true, picks }), ['done', 'done', 'done', 'now']);
  assert.deepEqual(importStages({ decoded: true, picks, imported: true }), ['done', 'done', 'done', 'done']);
});

test('a new pick takes the first free letter; a click again unpicks', () => {
  let picks = [{ pattern: 1, letter: 'A' }, { pattern: 4, letter: 'C' }];
  ({ picks } = pick(picks, 2));
  assert.deepEqual(picks, [{ pattern: 1, letter: 'A' }, { pattern: 2, letter: 'B' }, { pattern: 4, letter: 'C' }]);
  ({ picks } = pick(picks, 1));
  assert.deepEqual(picks, [{ pattern: 2, letter: 'B' }, { pattern: 4, letter: 'C' }]);
  ({ picks } = pick(picks, 9));
  assert.deepEqual(picks.find((p) => p.pattern === 9), { pattern: 9, letter: 'A' });
  assert.deepEqual(pick(picks, 9, { picked: true }).picks, picks); // already picked
  assert.deepEqual(pick(picks, 3, { picked: false }).picks, picks); // not picked
});

test('a letter another pick holds swaps the two; a letter picks an unpicked pattern', () => {
  const picks = [{ pattern: 1, letter: 'A' }, { pattern: 2, letter: 'B' }, { pattern: 3, letter: 'C' }];
  assert.deepEqual(pick(picks, 3, { letter: 'A' }).picks,
    [{ pattern: 1, letter: 'C' }, { pattern: 2, letter: 'B' }, { pattern: 3, letter: 'A' }]);
  assert.deepEqual(pick(picks, 2, { letter: 'L' }).picks,
    [{ pattern: 1, letter: 'A' }, { pattern: 2, letter: 'L' }, { pattern: 3, letter: 'C' }]);
  assert.deepEqual(pick(picks, 5, { letter: 'B' }).picks,
    [{ pattern: 1, letter: 'A' }, { pattern: 2, letter: 'D' }, { pattern: 3, letter: 'C' }, { pattern: 5, letter: 'B' }]);
});

test('at most 12 patterns are picked', () => {
  let picks = [];
  for (let n = 1; n <= MAX_PICKS; n += 1) ({ picks } = pick(picks, n));
  assert.deepEqual(picks.map((p) => p.letter).join(''), LETTERS);
  const thirteenth = pick(picks, 13);
  assert.ok(thirteenth.refused);
  assert.deepEqual(thirteenth.picks, picks);
  assert.ok(!pick(picks, 12).refused); // unpicking still works
});

test('the letter menu names the pick each letter would swap with', () => {
  const picks = [{ pattern: 1, letter: 'A' }, { pattern: 5, letter: 'B' }];
  const choices = letterChoices(picks, 1);
  assert.equal(choices.length, 12);
  assert.deepEqual(choices[0], { letter: 'A', holder: null, current: true });
  assert.deepEqual(choices[1], { letter: 'B', holder: 5, current: false });
  assert.deepEqual(choices[2], { letter: 'C', holder: null, current: false });
});

test('pattern buttons: 1-based numbers, the letter after the arrow, empty and undecoded ones', () => {
  const state = { patterns: [{ number: 1, empty: false }, { number: 2, empty: true }],
    picks: [{ pattern: 1, letter: 'B' }], focus: 1 };
  assert.deepEqual(patternButton(1, state), { text: '1→B', picked: true, letter: 'B', focused: true, empty: false, usable: true });
  assert.deepEqual(patternButton(2, state), { text: '2', picked: false, letter: null, focused: false, empty: true, usable: true });
  assert.equal(patternButton(3, state).usable, false);
});

test('the level meter: dB text, tone, and a held peak that decays', () => {
  assert.equal(levelText(0), '-∞ dB');
  assert.equal(levelText(1), '+0.0 dB');
  assert.equal(levelText(0.5), '-6.0 dB');
  assert.deepEqual([0.01, 0.2, 0.6, 0.99].map(levelTone), ['low', 'good', 'hot', 'clip']);
  let s = holdPeak(null, 0.8, { hold: 2 });
  assert.deepEqual(s, { peak: 0.8, wait: 2 });
  s = holdPeak(s, 0.1, { hold: 2 });
  s = holdPeak(s, 0.1, { hold: 2 });
  assert.equal(s.peak, 0.8);
  s = holdPeak(s, 0.1, { hold: 2, decay: 0.5 });
  assert.equal(s.peak, 0.4);
});

test('stage texts', () => {
  assert.equal(slotsNote([1, 2]), 'PO-32 patterns 1–2 are sent empty: the transfer carries the sounds only');
  assert.equal(slotsNote([1]), 'PO-32 pattern 1 is sent empty: the transfer carries the sounds only');
  assert.equal(transferText({ transfer: 'ready', seconds: 6.43 }), 'ready: 6.4 s of signal');
  assert.equal(transferText({ transfer: 'sending', progress: 0.424 }), 'sending… 42 %');
  assert.equal(transferText({ transfer: 'error', error: 'the stream stopped' }), 'failed: the stream stopped');
  assert.equal(sourceText({ recording: true, seconds: 3.21 }), 'recording 3.2 s: play the PO-32 transfer now');
  assert.equal(sourceText({ decode: 'decoded', decoded: { card: true, drums: 16, patterns: 3, source: 'card.wav' } }),
    'decoded PO-32 card: 16 sounds, 3 patterns · card.wav');
  assert.equal(sourceText({}), 'nothing decoded yet');
  assert.match(importText({ decoded: {}, bank: 1, picks: [{ pattern: 2, letter: 'C' }, { pattern: 1, letter: 'A' }] }),
    /^Bank 1 replaces the sounds of channels 1–8 and all 12 patterns: 2 land on A C, the rest are emptied/);
  assert.equal(importButtonText([{}]), 'import 8 sounds + 1 pattern');
  assert.equal(importButtonText([]), 'import 8 sounds + 0 patterns');
  assert.deepEqual(unmutedChannels([false, true, false, false, false, false, true, false]), [1, 3, 4, 5, 6, 8]);
});
