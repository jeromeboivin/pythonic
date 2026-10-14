// Pure logic of the KIT and INST selectors (kits.js): the kits stepped
// through, the display lines, VALUE steps and SHIFT jumps, where INST starts.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { groupJump, instLines, instStart, kitEntries, kitLines, kitNumber, stepIndex, writeName } from '../js/kits.js';

const LIST = [{ id: 'factory:505', name: '505', factory: true }, { id: 'factory:909', name: '909', factory: true },
  { id: 'user:Dub', name: 'Dub', factory: false }];
const LIBRARY = [['factory:505 BD', '505 BD', 'BD'], ['factory:909 BD', '909 BD', 'BD'], ['user:Kick 1.mtdrum', 'Kick 1', 'BD'],
  ['factory:909 SD', '909 SD', 'SD'], ['factory:808 CH', '808 CH', 'CH'], ['user:a/Hat.mtdrum', 'Hat', 'CH']];

test('kit 000 is the preset kit, then the library in its order', () => {
  const entries = kitEntries(LIST, '808 Beats');
  assert.deepEqual(entries.map((k) => k.id), ['', 'factory:505', 'factory:909', 'user:Dub']);
  assert.equal(entries[0].name, '808 Beats');
  assert.ok(entries[0].preset);
  assert.deepEqual(kitEntries(null, '').map((k) => k.id), ['']);
});

test('the kit selector shows the number, the edited mark, the group and the name', () => {
  const entries = kitEntries(LIST, 'Beats');
  assert.equal(kitNumber(4), '004');
  assert.deepEqual(kitLines(entries, 2, true), ['KIT 002* FACTORY', '909']);
  assert.deepEqual(kitLines(entries, 3, false), ['KIT 003  USER', 'DUB']);
  assert.deepEqual(kitLines(entries, 0, false), ['KIT 000  PRESET', 'BEATS']);
  assert.deepEqual(kitLines(entries, 9, false), ['KIT ---', 'no kit']);
});

test('a VALUE step stays inside the list', () => {
  assert.equal(stepIndex(4, 0, -1), 0);
  assert.equal(stepIndex(4, 2, 1), 3);
  assert.equal(stepIndex(4, 3, 1), 3);
});

test('SHIFT jumps to the next group, or back to the start of this one, then the previous', () => {
  const group = (i) => LIBRARY[i][2];
  assert.equal(groupJump(LIBRARY.length, 1, 1, group), 3); // BD -> SD
  assert.equal(groupJump(LIBRARY.length, 3, 1, group), 4); // SD -> CH
  assert.equal(groupJump(LIBRARY.length, 5, 1, group), 5); // the last group: stays
  assert.equal(groupJump(LIBRARY.length, 5, -1, group), 4); // the start of CH
  assert.equal(groupJump(LIBRARY.length, 4, -1, group), 3); // the start of SD
  assert.equal(groupJump(LIBRARY.length, 3, -1, group), 0); // the start of BD
  assert.equal(groupJump(LIBRARY.length, 0, -1, group), 0);
  assert.equal(groupJump(0, 0, 1, group), 0);
});

test('INST starts at the drum patch last loaded, else the same name, else the category', () => {
  assert.equal(instStart(LIBRARY, 'Kick 1', 'user:Kick 1.mtdrum'), 2);
  assert.equal(instStart(LIBRARY, '909 SD', 'user:Kick 1.mtdrum'), 3); // the channel changed since
  assert.equal(instStart(LIBRARY, '808 CH'), 4);
  assert.equal(instStart(LIBRARY, 'My Snare'), 3);
  assert.equal(instStart(LIBRARY, 'Something'), 0);
  assert.equal(instStart([], 'Kick'), -1);
});

test('the INST selector shows the channel, the category and the place in it', () => {
  assert.deepEqual(instLines(LIBRARY, 1, 3), ['INST 3 BD 2/3', '909 BD']);
  assert.deepEqual(instLines(LIBRARY, 5, 8), ['INST 8 CH 2/2', 'HAT']);
  assert.deepEqual(instLines([], 0, 1), ['INST 1', 'no drum patches']);
});

test('write as offers the user kit its own name, a copy for the others', () => {
  const entries = kitEntries(LIST, 'Beats');
  assert.equal(writeName(entries[3]), 'Dub');
  assert.equal(writeName(entries[2]), '909 copy');
  assert.equal(writeName(entries[0]), 'Beats copy');
});
