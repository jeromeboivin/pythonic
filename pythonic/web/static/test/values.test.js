import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  coerce, dragPosition, editText, formatValue, fromPosition, parseValue, toPosition, wheelValue,
} from '../js/values.js';

// Metadata as the core's describe() gives it
const DECAY = { kind: 'float', minimum: 10, maximum: 10000, default: 316.23, unit: 'ms', curve: 'log', labels: [] };
const ATTACK = { kind: 'float', minimum: 0, maximum: 10000, default: 0, unit: 'ms', curve: 'log', labels: [] };
const PITCH = { kind: 'float', minimum: -24, maximum: 24, default: 0, unit: 'st', curve: 'linear', labels: [] };
const LEVEL = { kind: 'float', minimum: -60, maximum: 10, default: 0, unit: 'dB', curve: 'linear', labels: [] };
const FREQ = { kind: 'float', minimum: 20, maximum: 20000, default: 440, unit: 'Hz', curve: 'log', labels: [] };
const PAN = { kind: 'float', minimum: -100, maximum: 100, default: 0, unit: 'pan', curve: 'linear', labels: [] };
const SWING = { kind: 'float', minimum: 0, maximum: 1, default: 0, unit: 'ratio', curve: 'linear', labels: [] };
const TEMPO = { kind: 'int', minimum: 1, maximum: 300, default: 120, unit: 'BPM', curve: 'linear', labels: [] };
const FILL = { kind: 'int', minimum: 2, maximum: 8, default: 4, unit: 'x', curve: 'linear', labels: [] };
const RATE = { kind: 'enum', minimum: null, maximum: null, default: '1/16', unit: '', curve: 'linear',
  labels: ['1/8', '1/8T', '1/16', '1/16T', '1/32'] };
const MUTE = { kind: 'bool', minimum: null, maximum: null, default: false, unit: '', curve: 'linear', labels: [] };

const near = (a, b, eps = 1e-6) => assert.ok(Math.abs(a - b) <= eps, `${a} is not ${b}`);

test('linear positions map the range to 0..1', () => {
  assert.equal(toPosition(PITCH, -24), 0);
  assert.equal(toPosition(PITCH, 0), 0.5);
  assert.equal(toPosition(PITCH, 24), 1);
  assert.equal(fromPosition(PITCH, 0.75), 12);
  near(toPosition(LEVEL, 0), 60 / 70);
});

test('log positions follow the curve as the core does', () => {
  near(toPosition(DECAY, 10), 0);
  near(toPosition(DECAY, 10000), 1);
  near(toPosition(DECAY, Math.sqrt(10 * 10000)), 0.5);
  near(fromPosition(DECAY, 0.5), Math.sqrt(10 * 10000), 1e-9);
  near(toPosition(FREQ, 632.455532), 0.5, 1e-6);
});

test('a log range from 0 starts its curve at 1e-5 of the maximum', () => {
  assert.equal(toPosition(ATTACK, 0), 0);
  near(toPosition(ATTACK, 0.1), 0);
  near(toPosition(ATTACK, 1), 0.2);
  assert.equal(fromPosition(ATTACK, 0), 0);
  near(fromPosition(ATTACK, 0.2), 1, 1e-9);
});

test('positions round trip and clamp', () => {
  for (const meta of [DECAY, PITCH, FREQ, LEVEL, SWING]) {
    for (const p of [0, 0.1, 0.33, 0.5, 0.9, 1]) near(toPosition(meta, fromPosition(meta, p)), p, 1e-9);
  }
  assert.equal(fromPosition(PITCH, 2), 24);
  assert.equal(fromPosition(PITCH, -1), -24);
  assert.equal(toPosition(PITCH, 99), 1);
});

test('ints round, enums step by option, bools split at the middle', () => {
  assert.equal(fromPosition(TEMPO, 0.5), 151);
  assert.equal(toPosition(RATE, '1/16'), 0.5);
  assert.equal(fromPosition(RATE, 0.74), '1/16T');
  assert.equal(fromPosition(MUTE, 0.6), true);
  assert.equal(coerce(RATE, 4), '1/32');
  assert.equal(coerce(TEMPO, 400.4), 300);
  assert.equal(coerce(MUTE, 0), false);
});

test('200 design pixels of drag cover the range, Shift is x0.1', () => {
  assert.equal(dragPosition(0, 200), 1);
  assert.equal(dragPosition(0.5, -100), 0);
  near(dragPosition(0.5, 20), 0.6);
  near(dragPosition(0.5, 20, { fine: true }), 0.51);
  near(dragPosition(0.2, 64, { length: 128 }), 0.7);
  assert.equal(dragPosition(0.9, 500), 1);
});

test('the wheel moves 1 % per notch along the curve, or a fixed step', () => {
  near(wheelValue(PITCH, 0, 1), 0.48, 1e-9);
  near(wheelValue(PITCH, 0, -2), -0.96, 1e-9);
  near(toPosition(DECAY, wheelValue(DECAY, 100, 1)), toPosition(DECAY, 100) + 0.01, 1e-9);
  near(wheelValue(LEVEL, 0, 1, { fraction: 0.02 }), 1.4, 1e-9);
  assert.equal(wheelValue(TEMPO, 120, 1, { step: 1 }), 121);
  assert.equal(wheelValue(TEMPO, 300, 1, { step: 1 }), 300);
});

test('an int moves at least one unit per notch', () => {
  assert.equal(wheelValue(FILL, 4, 1), 5);
  assert.equal(wheelValue(FILL, 4, -1), 3);
  assert.equal(wheelValue(FILL, 8, 1), 8);
});

test('the wheel steps enums one option per notch', () => {
  assert.equal(wheelValue(RATE, '1/16', 1), '1/16T');
  assert.equal(wheelValue(RATE, '1/16', -5), '1/8');
  assert.equal(wheelValue(RATE, '1/32', 1), '1/32');
});

test('readouts are in engine units', () => {
  assert.equal(formatValue(DECAY, 316.23), '316 ms');
  assert.equal(formatValue(DECAY, 1500), '1.50 s');
  assert.equal(formatValue(ATTACK, 2.5), '2.5 ms');
  assert.equal(formatValue(FREQ, 440), '440 Hz');
  assert.equal(formatValue(FREQ, 1234), '1.23 kHz');
  assert.equal(formatValue(FREQ, 12000), '12.0 kHz');
  assert.equal(formatValue(PITCH, 3.25), '+3.3 st');
  assert.equal(formatValue(PITCH, -12), '−12.0 st');
  assert.equal(formatValue(PITCH, 0), '0.0 st');
  assert.equal(formatValue(LEVEL, -6), '−6.0 dB');
  assert.equal(formatValue(PAN, 0.2), 'C');
  assert.equal(formatValue(PAN, -30), 'L30');
  assert.equal(formatValue(PAN, 45), 'R45');
  assert.equal(formatValue(SWING, 0.25), '25 %');
  assert.equal(formatValue(TEMPO, 128), '128 BPM');
  assert.equal(formatValue(FILL, 4), '4x');
  assert.equal(formatValue(RATE, '1/8T'), '1/8T');
  assert.equal(formatValue(MUTE, true), 'on');
  assert.equal(formatValue(DECAY, undefined), '—');
});

test('typed values take units, percents for ratios and L/C/R for pans', () => {
  assert.equal(parseValue(FREQ, '1.5k'), 1500);
  assert.equal(parseValue(FREQ, '2 kHz'), 2000);
  assert.equal(parseValue(DECAY, '1.2 s'), 1200);
  assert.equal(parseValue(DECAY, '250ms'), 250);
  assert.equal(parseValue(LEVEL, '-6 dB'), -6);
  assert.equal(parseValue(LEVEL, '−6'), -6);
  assert.equal(parseValue(SWING, '25'), 0.25);
  assert.equal(parseValue(SWING, '25 %'), 0.25);
  assert.equal(parseValue(PAN, 'L30'), -30);
  assert.equal(parseValue(PAN, 'c'), 0);
  assert.equal(parseValue(PAN, '-20'), -20);
  assert.equal(parseValue(TEMPO, '128.6'), 129);
  assert.equal(parseValue(PITCH, '99'), 24);
  assert.equal(parseValue(RATE, '1/8t'), '1/8T');
  assert.equal(parseValue(RATE, '5'), '1/32');
  assert.equal(parseValue(PITCH, 'abc'), null);
  assert.equal(parseValue(PITCH, ''), null);
});

test('the exact-value field starts with the readout number', () => {
  assert.equal(editText(SWING, 0.25), '25');
  assert.equal(editText(DECAY, 316.23), '316.23');
  assert.equal(editText(PAN, -30), 'L30');
  assert.equal(editText(TEMPO, 120), '120');
  assert.equal(editText(RATE, '1/16'), '1/16');
});
