import { test } from 'node:test';
import assert from 'node:assert/strict';
import { dragToValue } from './model.js';
test('200 px up covers the full range', () => assert.equal(dragToValue(-24, -200, -24, 24), 24));
test('clamps', () => assert.equal(dragToValue(0, 10000, -24, 24), -24));
test('fine is 10x slower', () => assert.ok(Math.abs(dragToValue(0, -200, -24, 24, true) - 4.8) < 1e-9));
