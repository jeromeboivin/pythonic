// The edit rack's layout (map decision #11), as data (pure, no DOM): every
// section in one row, left to right, each a list of rows of controls bound
// to a sound address suffix of the selected channel (ch<N>.<suffix>). Tune,
// osc decay and level live on the face strips only.
//
// A control: { s: suffix, type, label, name?, options?, values?, size?, length? }
//   type: knob | fader (vertical, the track length is the drag range) |
//         hfader (horizontal crossfader) | switch (segmented, up to 5) |
//         list | toggle | target (the → destination button of a source)
//   label: under the control; name: on the display (default: the label)
//   options: display names of the enum labels (or of `values`, when given)
//   values: the enum labels a switch offers (the decided subset)

import { STAGE_HEIGHT } from './stage.js';

/** The face alone: the stage height with the edit rack drawer closed. */
export const FACE_HEIGHT = 700;

/** The stage height for the drawer state. */
export const stageHeightFor = (open) => (open ? STAGE_HEIGHT : FACE_HEIGHT);

/** Sound parameters the rack leaves to the face strips. */
export const FACE_ONLY = ['osc.pitch', 'osc.decay', 'mix.level'];

const knob = (s, label, name = null, extra = {}) => ({ s, type: 'knob', label, name: name || label, ...extra });
const fader = (s, label, name) => ({ s, type: 'fader', label, name, length: 112 });
const toggle = (s, label, name) => ({ s, type: 'toggle', label, name });
const sw = (s, label, name, options, extra = {}) => ({ s, type: 'switch', label, name, options, ...extra });
const list = (s, label, name, options) => ({ s, type: 'list', label, name, options });

const LFO_WAVES = 'sin,tri,saw▲,saw▼,sq,s&h';
const SYNCS = 'free,1/1,1/2,1/4,1/8,1/16,1/4.,1/8.,1/4T,1/8T,2 bar,4 bar';
const DELAY_TIMES = '1/1,1/2,1/4,1/8,1/16,1/32,1/2T,1/4T,1/8T,1/16T,1/2.,1/4.,1/8.,1/16.';

function lfoRow(src, title) {
  const p = `${src}.`;
  return {
    source: src,
    title,
    controls: [
      toggle(`${p}on`, 'on', `${title} on`),
      list(`${p}wave`, 'wave', `${title} wave`, LFO_WAVES),
      knob(`${p}rate`, 'rate', `${title} rate`, { size: 26 }),
      knob(`${p}depth`, 'depth', `${title} depth`, { size: 26 }),
      list(`${p}sync`, 'sync', `${title} sync`, SYNCS),
      toggle(`${p}retrig`, 're', `${title} retrigger`),
      toggle(`${p}unipolar`, 'uni', `${title} unipolar`),
      knob(`${p}phase`, 'phase', `${title} phase`, { size: 26 }),
      { s: `${p}target`, type: 'target', label: '', name: `${title} destination` },
    ],
  };
}

export const RACK_SECTIONS = [
  { key: 'osc', title: 'oscillator', rows: [
    [sw('osc.wave', 'wave', 'osc wave', 'sin,tri,saw')],
    [sw('osc.mod_mode', 'pitch mod', 'pitch mod', 'dec,sine,rand')],
    [knob('osc.freq', 'freq', 'osc freq'), knob('osc.mod_amount', 'amount', 'pitch amount'),
      knob('osc.mod_rate', 'rate', 'pitch rate')],
  ] },
  { key: 'noise', title: 'noise', rows: [
    [sw('noise.filter', 'filter', 'noise filter', 'LP,BP,HP'), toggle('noise.stereo', 'stereo', 'noise stereo')],
    [knob('noise.freq', 'freq', 'noise freq'), knob('noise.q', 'Q', 'noise q')],
  ] },
  { key: 'env', title: 'envelopes', rows: [
    [sw('noise.env', 'noise env', 'noise envelope', 'exp,lin,mod')],
    [fader('osc.attack', 'osc atk', 'osc attack'), fader('noise.attack', 'nse atk', 'noise attack'),
      fader('noise.decay', 'nse dec', 'noise decay')],
  ] },
  { key: 'mix', title: 'mix', rows: [
    [{ s: 'mix.osc_noise', type: 'hfader', label: '', name: 'osc / noise mix', length: 112,
      ends: 'osc,noise' }],
    [knob('eq.freq', 'eq freq'), knob('eq.gain', 'eq gain'), knob('mix.distortion', 'distort', 'distortion')],
    [knob('mix.pan', 'pan'), sw('mix.output', 'out', 'output', 'A,B'), toggle('mix.choke', 'choke', 'choke')],
  ] },
  { key: 'vel', title: 'velocity', rows: [
    [fader('vel.osc', 'osc', 'osc velocity'), fader('vel.noise', 'noise', 'noise velocity'),
      fader('vel.mod', 'mod', 'mod velocity')],
  ] },
  { key: 'fx', title: 'fx', rows: [
    [knob('fx.vintage', 'vintage'), knob('fx.reverb_decay', 'rvb time', 'reverb time'),
      knob('fx.reverb_mix', 'rvb mix', 'reverb mix'), knob('fx.reverb_width', 'rvb wide', 'reverb width')],
    [sw('fx.delay_time', 'delay', 'delay time', DELAY_TIMES,
      { values: 'quarter,eighth,sixteenth,eighth_t,quarter_d' }), toggle('fx.delay_pingpong', 'P.P', 'ping-pong')],
    [knob('fx.delay_feedback', 'dly fdbk', 'delay feedback'), knob('fx.delay_mix', 'dly mix', 'delay mix')],
  ] },
  { key: 'mod', title: 'modulation', sources: [
    lfoRow('lfo1', 'LFO 1'),
    lfoRow('lfo2', 'LFO 2'),
    { source: 'pump', title: 'PUMP', controls: [
      toggle('pump.on', 'on', 'pump on'),
      knob('pump.amount', 'amount', 'pump amount', { size: 26 }),
      knob('pump.attack', 'attack', 'pump attack', { size: 26 }),
      knob('pump.release', 'release', 'pump release', { size: 26 }),
      knob('pump.curve', 'curve', 'pump curve', { size: 26 }),
      list('pump.sync', 'sync', 'pump sync', SYNCS),
      { s: 'pump.target', type: 'target', label: '', name: 'pump destination' },
    ] },
  ] },
];

/** Every control of the rack, in layout order. */
export function rackControls() {
  return RACK_SECTIONS.flatMap((section) => (section.rows
    ? section.rows.flat()
    : section.sources.flatMap((row) => row.controls)));
}

/** The sound address suffixes the rack binds. */
export const rackSuffixes = () => rackControls().map((c) => c.s);
