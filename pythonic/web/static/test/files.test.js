import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fileName, folderOf, percent, replaceQuestion, samePath } from '../js/files.js';
import { canSaveInPlace, neighbourFile } from '../js/presets.js';

test('file names and folders of paths, either separator', () => {
  assert.equal(fileName('/a/b/808.mtpreset'), '808.mtpreset');
  assert.equal(fileName('C:\\kits\\909.json'), '909.json');
  assert.equal(fileName(null), '');
  assert.equal(folderOf('/a/b/808.mtpreset'), '/a/b');
  assert.equal(folderOf('C:\\kits\\909.json'), 'C:\\kits');
  assert.equal(folderOf('808.json'), '');
  assert.ok(samePath('/a/b/', '/a/b'));
  assert.ok(samePath('C:\\kits', 'C:/kits'));
  assert.ok(!samePath('/a/b', '/a/c'));
  assert.ok(!samePath(null, null));
});

test('progress as a percentage, clamped', () => {
  assert.equal(percent(0.424), '42 %');
  assert.equal(percent(2), '100 %');
  assert.equal(percent(undefined), '0 %');
});

test('the replace question names the file, or the files of a folder', () => {
  const [title, text] = replaceQuestion({ saved: false, exists: true, path: '/k/kick.mtdrum' });
  assert.equal(title, 'Replace “kick.mtdrum”?');
  assert.match(text, /already exists in \/k\./);
  const [t2, x2] = replaceQuestion({ saved: false, exists: true, folder: '/w', paths: ['/w/01_BD.wav', '/w/02_SD.wav'] });
  assert.equal(t2, 'Replace 2 files?');
  assert.match(x2, /01_BD\.wav, 02_SD\.wav already exist in \/w/);
  assert.equal(replaceQuestion({ paths: ['/w/01_BD.wav'] })[0], 'Replace 1 file?');
});

test('◀ ▶ step through the preset folder without wrapping', () => {
  const files = ['505.mtpreset', '808.mtpreset', 'mine.json'];
  assert.equal(neighbourFile(files, '/p/808.mtpreset', '/p', 1), 'mine.json');
  assert.equal(neighbourFile(files, '/p/808.mtpreset', '/p/', -1), '505.mtpreset');
  assert.equal(neighbourFile(files, '/p/mine.json', '/p', 1), null);
  assert.equal(neighbourFile(files, '/p/505.mtpreset', '/p', -1), null);
  // not in the folder: ▶ starts at the first file, ◀ does nothing
  assert.equal(neighbourFile(files, '/elsewhere/808.mtpreset', '/p', 1), '505.mtpreset');
  assert.equal(neighbourFile(files, null, '/p', -1), null);
  assert.equal(neighbourFile([], null, '/p', 1), null);
});

test('only a JSON preset is saved over its own file', () => {
  assert.ok(canSaveInPlace('/p/mine.json'));
  assert.ok(canSaveInPlace('/p/MINE.JSON'));
  assert.ok(!canSaveInPlace('/p/808.mtpreset'));
  assert.ok(!canSaveInPlace(null));
});
