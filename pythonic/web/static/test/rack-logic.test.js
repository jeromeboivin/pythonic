// Pure logic behind the edit rack: which controls a modulation source can aim
// at (click to assign), the target names, the modulation band of a control,
// the rack's layout and the stage heights of the drawer.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  destinationOf, MOD_SOURCES, modulationBand, SOURCE_NAMES, TARGET_LABELS, targetAddress, targetName,
} from '../js/modulation.js';
import { FACE_ONLY, RACK_SECTIONS, rackSuffixes, stageHeightFor } from '../js/rack-layout.js';
import { fitStage } from '../js/stage.js';

const near = (a, b, eps = 1e-6) => assert.ok(Math.abs(a - b) <= eps, `${a} != ${b}`);

test('controls of the selected channel map to their mod targets', () => {
  assert.deepEqual(destinationOf('ch3.osc.freq', 3), { target: 'osc_frequency' });
  assert.deepEqual(destinationOf('ch3.osc.pitch', 3), { target: 'pitch_semitones' });
  assert.deepEqual(destinationOf('ch3.mix.level', 3), { target: 'level_db' });
  assert.deepEqual(destinationOf('ch3.mix.osc_noise', 3), { target: 'osc_noise_mix' });
  assert.deepEqual(destinationOf('ch3.vel.mod', 3), { target: 'mod_vel_sensitivity' });
  assert.deepEqual(destinationOf('ch3.eq.gain', 3), { target: 'eq_gain_db' });
});

test('master and the sound morph are targets from any channel', () => {
  assert.deepEqual(destinationOf('global.master', 5), { target: 'master_volume' });
  assert.deepEqual(destinationOf('morph.position', 1), { target: 'morph' });
});

test('other controls refuse with a reason for the display', () => {
  assert.deepEqual(destinationOf('ch3.osc.wave', 3), { refused: 'not a destination' });
  assert.deepEqual(destinationOf('ch3.lfo1.depth', 3), { refused: 'not a destination' });
  assert.deepEqual(destinationOf('global.tempo', 3), { refused: 'not a destination' });
  assert.deepEqual(destinationOf('ch2.osc.freq', 3), { refused: 'not on CH3' });
  assert.deepEqual(destinationOf(null, 3), { refused: 'not a destination' });
});

test('every engine target has a name and an address, none is off', () => {
  assert.equal(TARGET_LABELS.length, 28);
  assert.equal(TARGET_LABELS[0], 'none');
  assert.equal(targetName('none'), 'off');
  assert.equal(targetName('osc_frequency'), 'osc freq');
  assert.equal(targetName('master_volume'), 'master');
  assert.equal(targetName('unknown_thing'), 'unknown_thing');
  for (const t of TARGET_LABELS.slice(1)) {
    const address = targetAddress(t, 4);
    assert.ok(address, t);
    assert.deepEqual(destinationOf(address, 4), { target: t }, t);
  }
  assert.equal(targetAddress('none', 4), null);
  assert.equal(targetAddress('morph', 4), 'morph.position');
});

test('the sources and their display names', () => {
  assert.deepEqual(MOD_SOURCES, ['lfo1', 'lfo2', 'pump']);
  assert.deepEqual(SOURCE_NAMES, { lfo1: 'LFO 1', lfo2: 'LFO 2', pump: 'PUMP' });
});

const LIN = { kind: 'float', minimum: -24, maximum: 24, unit: 'st', curve: 'linear' };
const LOG = { kind: 'float', minimum: 10, maximum: 10000, unit: 'ms', curve: 'log' };

test('the modulation band runs from the set value to the modulated one', () => {
  const [from, to] = modulationBand(LIN, 0, 12);
  near(from, 0.5);
  near(to, 0.75);
  const [a, b] = modulationBand(LIN, 0, -6);
  near(a, 0.5);
  near(b, 0.375);
});

test('the band follows the curve and stops at the ends of the range', () => {
  const [from, to] = modulationBand(LOG, 100, 900);
  near(from, 1 / 3);
  near(to, 2 / 3);
  assert.deepEqual(modulationBand(LIN, 20, 30), [44 / 48, 1]);
  assert.deepEqual(modulationBand(LIN, -20, -30), [4 / 48, 0]);
  assert.equal(modulationBand({ kind: 'enum', labels: ['a'] }, 'a', 1), null);
  assert.equal(modulationBand(LIN, 0, 0), null);
});

test('the rack holds every sound parameter once, except those on the face', () => {
  const suffixes = rackSuffixes();
  assert.equal(new Set(suffixes).size, suffixes.length, 'no duplicates');
  for (const s of FACE_ONLY) assert.ok(!suffixes.includes(s), s);
  assert.deepEqual(FACE_ONLY, ['osc.pitch', 'osc.decay', 'mix.level']);
  for (const s of ['osc.attack', 'noise.attack', 'noise.decay', 'mix.pan', 'mix.output', 'fx.delay_time',
    'lfo1.target', 'lfo2.phase', 'pump.sync', 'pump.target', 'vel.mod']) {
    assert.ok(suffixes.includes(s), s);
  }
  assert.deepEqual(RACK_SECTIONS.map((s) => s.title),
    ['oscillator', 'noise', 'envelopes', 'mix', 'velocity', 'fx', 'modulation']);
});

test('closing the drawer leaves the face; the scale stays when the window follows', () => {
  assert.equal(stageHeightFor(true), 1000);
  assert.equal(stageHeightFor(false), 700);
  assert.deepEqual(fitStage(1280, 560, 1600, 700), { scale: 0.8, x: 0, y: 0 });
  assert.deepEqual(fitStage(1280, 800, 1600, 700), { scale: 0.8, x: 0, y: 120 });
});
