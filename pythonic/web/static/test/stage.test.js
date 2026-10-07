import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fitStage } from '../js/stage.js';

test('the minimum window shows the panel at 0.8', () => {
  assert.deepEqual(fitStage(1280, 800), { scale: 0.8, x: 0, y: 0 });
});

test('a wider window letterboxes left and right', () => {
  assert.deepEqual(fitStage(1920, 1000), { scale: 1, x: 160, y: 0 });
});

test('a taller window letterboxes top and bottom', () => {
  assert.deepEqual(fitStage(1600, 1200), { scale: 1, x: 0, y: 100 });
});

test('an empty view gives scale 0', () => {
  assert.equal(fitStage(0, 0).scale, 0);
});
