// The setup sheet (map decisions #13, #14, #22): an overlay sheet with tabs
// on top (audio | midi | synthesis | ai | display) that sizes to its tab. SETUP opens it
// on audio; the MIDI LED and a control's right-click ▸ CC mappings… open it
// on midi (panel.openPage('setup', {tab})). Values apply live, like any
// address; only the stream settings (output device, buffer, sample rate,
// synth rate) wait for "restart audio" (audio.apply): each waiting field shows
// a dot (pref.audio.pending) and the button lights.
//
//   audio      output device (rescan: audio.rescan), buffer, sample rate (the
//              rates the device takes: audio.rates), synth rate, mono; the
//              running stream; the input device of the PO-32 import
//   midi       input device (midi.open / close / rescan), connection LED, base
//              note and its 8 notes, follow MIDI clock and the synced tempo;
//              CC mappings beside it: one row per mapping (any CC 0-127 → a
//              control, ✕), + add, clear all, live CC activity, the pitch bend
//              target. Learning stays on the panel (right-click ▸ MIDI learn),
//              since the sheet blocks the panel
//   synthesis  parameter smoothing
//   ai         pattern and patch models (native open dialog, clear) and their
//              temperatures: the same pref.ai.* addresses the AI page shows
//   display    GPU rendering of this panel (pref.web.gpu): saved at once,
//              applied at the next start (Chromium reads it before Qt starts)
//
// mountSetup({panel, store, client, meta}) registers the page; returns
// {open(options), close(), tab, element, destroy()}.

import {
  BASE_NOTE_CHOICES, baseNoteRange, bufferText, bufferWarning, ccActivity, ccRows, deviceChoices, deviceText, followsChannel,
  freeCc, modelText, noteName, parseCc, pendingFields, rateChoices, ratesNote, rateText, SETUP_TABS, streamText,
  TAB_WIDTHS, tabOf, targetGroups, targetLabel, withoutCc, withRow,
} from './setup-logic.js';

const STREAM_ADDRESSES = ['audio.running', 'audio.device', 'audio.device_is_default', 'audio.sample_rate',
  'audio.synth_rate', 'audio.block_size', 'audio.buffer_ms', 'audio.mono'];
const MODEL_FILTERS = ['Model checkpoints (*.pt)', 'All files (*)'];

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function button(text, cls = '') {
  const b = el('button', `btn ${cls}`.trim(), text);
  b.type = 'button';
  return b;
}

const same = (a, b) => a === b || JSON.stringify(a) === JSON.stringify(b);

export function mountSetup({ panel, store, client, meta = {} }) {
  const { ctx, display, sheets } = panel;
  let root = null; // the sheet's element while open
  let card = null;
  let tab = 'audio';
  let tabOffs = []; // watchers of the tab showing
  let drafts = []; // CC rows added but without a target yet: {id, cc}
  let draftIds = 0;
  let midiPorts = (meta['midi.device'] && meta['midi.device'].labels) || [];
  let deviceRates = null; // audio.rates of the chosen output device

  const value = (address) => store.value(address);
  /** Watch an address for the tab showing (read it if the store lacks it). */
  const watch = (address, fn) => {
    tabOffs.push(store.watch(address, fn));
    ctx.ensure(address);
  };
  const watchAll = (addresses, fn) => {
    const call = () => fn();
    for (const a of addresses) watch(a, call);
    fn();
  };
  /** Run a verb; its error shows on the display and under the sheet's title. */
  const act = (verb, args = {}) => client.act(verb, args).then((event) => {
    if (event.status === 'error') {
      display.alert(verb.toUpperCase(), event.error);
      showError(`${verb}: ${event.error}`);
    }
    return event;
  });
  const showError = (text) => {
    const line = root && root.querySelector('.su-error');
    if (line) { line.textContent = text || ''; line.hidden = !text; }
  };

  // ------------------------------------------------------------ parts
  const section = (title, ...children) => {
    const s = el('section', 'su-sec');
    s.append(el('h4', '', title), ...children);
    return s;
  };
  const field = (label, ...children) => {
    const row = el('div', 'su-field');
    row.append(el('span', 'su-lbl', label), ...children);
    return row;
  };
  const note = (text = '', cls = '') => el('div', `su-note ${cls}`.trim(), text);

  /**
   * A list box bound to an address: its text follows the value, a click opens
   * a menu of options() ([value, text]) and pick(value) applies the choice.
   */
  function choice(address, { options, text, pick, width = 150, cls = '' }) {
    const box = el('div', `lbox su-choice ${cls}`.trim());
    box.dataset.address = address;
    box.style.setProperty('--lw', `${width}px`);
    const render = () => { box.textContent = text(value(address)); };
    watch(address, render);
    render();
    box.addEventListener('click', (e) => {
      e.stopPropagation();
      const r = box.getBoundingClientRect();
      const current = value(address);
      ctx.openMenu(options().map(([v, t]) => [t, () => pick(v), { current: same(v, current) }]),
        r.left, r.bottom + 2, { className: 'list su-menu' });
    });
    box.render = render;
    return box;
  }

  /** The dot of a stream field waiting for "restart audio". */
  const dot = (address) => {
    const d = el('i', 'su-dot');
    d.dataset.pending = address;
    d.title = 'waits for restart audio';
    return d;
  };

  /** The control a CC (or pitch bend) drives: a menu of sections, then their controls. */
  function openTargets(x, y, current, onPick, { none = null } = {}) {
    const groups = targetGroups();
    const top = [];
    if (none) top.push([none, () => onPick(null), { current: !current }], null);
    for (const [title, targets] of groups) {
      const has = targets.some(([t]) => t === current);
      top.push([`${title} ▸`, () => ctx.openMenu(
        [[title, null], null, ...targets.map(([t, name]) => [name, () => onPick(t), { current: t === current }])],
        x, y, { className: 'list su-menu' }), { current: has }]);
    }
    ctx.openMenu(top, x, y, { className: 'list su-menu' });
  }

  // ------------------------------------------------------------ audio tab
  function audioTab() {
    const body = el('div', 'su-cols two');
    const all = (meta['pref.audio.sample_rate'] && meta['pref.audio.sample_rate'].labels) || [];
    const buffers = (meta['pref.audio.buffer_ms'] && meta['pref.audio.buffer_ms'].labels) || [];
    const synthRates = (meta['pref.audio.synth_rate'] && meta['pref.audio.synth_rate'].labels) || [];
    const outputs = () => value('audio.output_devices') || (meta['pref.audio.device'] || {}).labels || [];
    const inputs = () => value('audio.input_devices') || (meta['pref.audio.input_device'] || {}).labels || [];

    const ratesLine = note('', 'su-rates');
    const askRates = (device) => client.act('audio.rates', { device: device || null }).then((event) => {
      deviceRates = event.status === 'done' && event.result && Array.isArray(event.result.rates)
        ? event.result.rates : null;
      ratesLine.textContent = ratesNote(deviceRates || all, all);
    });

    const device = choice('pref.audio.device', {
      options: () => deviceChoices(outputs()),
      text: (v) => deviceText(v, outputs()),
      pick: (v) => { ctx.set('pref.audio.device', v); askRates(v); },
      width: 190,
    });
    const devices = el('span', 'su-wrap');
    devices.dataset.address = 'audio.output_devices';
    devices.append(device);
    watch('audio.output_devices', () => device.render());
    const rescan = button('rescan', 'su-small');
    rescan.id = 'su-audio-rescan';
    rescan.dataset.verb = 'audio.rescan';
    rescan.addEventListener('click', () => act('audio.rescan').then((event) => {
      if (event.status !== 'done') return;
      display.show('AUDIO DEVICES', 'rescanned');
      const r = event.result || {};
      if (Array.isArray(r.output_devices)) store.assume('audio.output_devices', r.output_devices);
      if (Array.isArray(r.input_devices)) store.assume('audio.input_devices', r.input_devices);
      askRates(value('pref.audio.device'));
    }));

    const buffer = choice('pref.audio.buffer_ms', {
      options: () => buffers.map((ms) => [ms, bufferText(ms)]), text: (v) => (v === undefined ? '—' : bufferText(v)),
      pick: (v) => ctx.set('pref.audio.buffer_ms', v), width: 90 });
    // Small buffers stay selectable; below the safe frame count a line warns.
    const bufferLine = note('', 'su-buffer-warning');
    const bufferRow = field('', bufferLine);
    watchAll(['pref.audio.buffer_ms', 'pref.audio.sample_rate'], () => {
      bufferLine.textContent = bufferWarning(value('pref.audio.buffer_ms'), value('pref.audio.sample_rate'));
      bufferRow.hidden = !bufferLine.textContent;
    });
    const rate = choice('pref.audio.sample_rate', {
      options: () => rateChoices(all, deviceRates, value('pref.audio.sample_rate')).map((r) => [r, rateText(r)]),
      text: (v) => (v === undefined ? '—' : rateText(v)), pick: (v) => ctx.set('pref.audio.sample_rate', v),
      width: 90 });
    const synth = choice('pref.audio.synth_rate', {
      options: () => synthRates.map((r) => [r, rateText(r, { synth: true })]),
      text: (v) => (v === undefined ? '—' : rateText(v, { synth: true })),
      pick: (v) => ctx.set('pref.audio.synth_rate', v), width: 120 });
    const mono = el('px-toggle');
    mono.dataset.address = 'pref.audio.mono';
    mono.setAttribute('label', 'mono output');
    mono.setAttribute('name', 'mono output');

    const restart = button('restart audio', 'su-restart');
    restart.id = 'su-restart';
    restart.dataset.verb = 'audio.apply';
    restart.dataset.address = 'pref.audio.pending';
    const restartNote = note('', 'su-restart-note');
    restart.addEventListener('click', () => {
      display.show('AUDIO', 'restarting');
      act('audio.apply').then((event) => {
        if (event.status === 'done') display.show('AUDIO', 'restarted');
      });
    });

    const stream = el('div', 'su-stream');
    for (const a of STREAM_ADDRESSES) {
      const span = el('span');
      span.dataset.address = a;
      stream.append(span);
    }
    const streamLine = el('span', 'su-stream-text');
    stream.append(streamLine);
    watchAll(STREAM_ADDRESSES, () => {
      streamLine.textContent = streamText(Object.fromEntries(STREAM_ADDRESSES.map((a) => [a, value(a)])));
      stream.classList.toggle('stopped', !value('audio.running'));
    });

    const output = section('audio output',
      field('device', devices, rescan, dot('pref.audio.device')),
      field('buffer', buffer, dot('pref.audio.buffer_ms')),
      bufferRow,
      field('sample rate', rate, dot('pref.audio.sample_rate')),
      field('', ratesLine),
      field('synth rate', synth, dot('pref.audio.synth_rate')),
      field('', mono),
      el('div', 'su-row su-restart-row'),
      stream);
    output.querySelector('.su-restart-row').append(restart, restartNote);

    const inputDevice = choice('pref.audio.input_device', {
      options: () => deviceChoices(inputs(), `(system default: ${value('audio.default_input') || 'none'})`),
      text: (v) => deviceText(v, inputs(), `(system default: ${value('audio.default_input') || 'none'})`),
      pick: (v) => ctx.set('pref.audio.input_device', v), width: 230 });
    const inputWrap = el('span', 'su-wrap');
    inputWrap.dataset.address = 'audio.input_devices';
    const defaultWrap = el('span', 'su-wrap');
    defaultWrap.dataset.address = 'audio.default_input';
    defaultWrap.append(inputDevice);
    inputWrap.append(defaultWrap);
    watch('audio.input_devices', () => inputDevice.render());
    watch('audio.default_input', () => inputDevice.render());
    const input = section('audio input (PO-32)', field('device', inputWrap),
      note('the PO-32 import records from this input; it applies when the input opens'));

    body.append(output, input);
    watch('pref.audio.pending', (pending) => {
      const waiting = pendingFields(pending);
      restart.classList.toggle('on', waiting.size > 0);
      for (const d of body.querySelectorAll('.su-dot')) d.classList.toggle('on', waiting.has(d.dataset.pending));
      restartNote.textContent = waiting.size
        ? 'device, buffer and rates wait for a restart' : 'mono and the input apply at once';
    });
    askRates(value('pref.audio.device'));
    return body;
  }

  // ------------------------------------------------------------ midi tab
  function midiTab() {
    const body = el('div', 'su-cols two midi');

    // MIDI input
    const deviceBox = choice('midi.device', {
      options: () => [['__off', '(off)'], ['__auto', '(auto-detect)'], ...midiPorts.map((p) => [p, p])],
      text: (v) => (v ? v : '(off)'),
      pick: (v) => {
        if (v === '__off') act('midi.close').then(() => display.show('MIDI', 'off'));
        else act('midi.open', { device: v === '__auto' ? null : v }).then((event) => {
          if (event.status === 'done') display.show('MIDI', String((event.result || {}).device || ''));
        });
      },
      width: 230,
    });
    const rescan = button('rescan', 'su-small');
    rescan.id = 'su-midi-rescan';
    rescan.dataset.verb = 'midi.rescan';
    rescan.addEventListener('click', () => act('midi.rescan').then((event) => {
      if (event.status !== 'done') return;
      midiPorts = (event.result && event.result.devices) || [];
      display.show('MIDI PORTS', `${midiPorts.length} found`);
    }));
    const led = el('span', 'mled su-led');
    led.dataset.address = 'midi.connected';
    const ledText = el('span', 'su-note', '');
    watch('midi.connected', (v) => {
      led.classList.toggle('connected', !!v);
      ledText.textContent = v ? 'connected' : 'not connected';
    });
    tabOffs.push(store.watchReadout('midi', (midi) => {
      const count = midi ? midi.activity : null;
      if (led.dataset.activity !== undefined && String(count) !== led.dataset.activity) {
        led.classList.add('on');
        clearTimeout(led.timer);
        led.timer = setTimeout(() => led.classList.remove('on'), 80);
      }
      led.dataset.activity = String(count);
    }));

    const base = choice('midi.base_note', {
      options: () => BASE_NOTE_CHOICES.map((n) => [n, `${noteName(n)} (${n})`]),
      text: (v) => (v === undefined ? '—' : `${noteName(v)} (${v})`),
      pick: (v) => setBase(v), width: 96, cls: 'su-base' });
    base.id = 'su-base-note';
    const setBase = (n) => {
      const m = meta['midi.base_note'] || { minimum: 0, maximum: 120 };
      const next = Math.min(m.maximum, Math.max(m.minimum, Math.round(n)));
      ctx.set('midi.base_note', next);
      display.show('BASE NOTE', `${noteName(next)} (${next})`);
    };
    base.addEventListener('wheel', (e) => {
      e.preventDefault();
      if (e.deltaY) setBase((value('midi.base_note') ?? 36) - Math.sign(e.deltaY));
    }, { passive: false });
    const down = button('−', 'su-small su-step');
    const up = button('+', 'su-small su-step');
    down.id = 'su-base-down';
    up.id = 'su-base-up';
    down.addEventListener('click', () => setBase((value('midi.base_note') ?? 36) - 1));
    up.addEventListener('click', () => setBase((value('midi.base_note') ?? 36) + 1));
    const range = note('', 'su-range');
    watch('midi.base_note', (v) => { range.textContent = `channels 1–8 play on notes ${baseNoteRange(v)}`; });

    const clock = el('px-toggle');
    clock.dataset.address = 'midi.clock_sync';
    clock.setAttribute('label', 'follow MIDI clock');
    clock.setAttribute('name', 'follow MIDI clock');
    const synced = el('span', 'su-note su-synced');
    synced.dataset.address = 'midi.synced_tempo';
    watchAll(['midi.synced_tempo', 'midi.clock_sync'], () => {
      const bpm = value('midi.synced_tempo');
      synced.textContent = !value('midi.clock_sync') ? '' : bpm ? `synced tempo ${bpm} BPM` : 'no clock yet';
    });

    const input = section('MIDI input',
      field('device', deviceBox, rescan),
      field('', led, ledText),
      field('base note', base, down, up),
      field('', range),
      field('clock', clock, synced),
      note('Also responds to program change 0–11 (patterns A–L), start, stop, continue and clock.'));

    // CC mappings
    const learning = el('div', 'su-learning');
    learning.dataset.address = 'midi.learning';
    const learningText = el('span');
    const cancel = button('cancel', 'su-small');
    cancel.addEventListener('click', () => client.act('midi.learn_cancel'));
    learning.append(learningText, cancel);
    watch('midi.learning', (target) => {
      learning.hidden = !target;
      learningText.textContent = target ? `learning ${targetLabel(target)}: move a controller` : '';
    });

    const rows = el('div', 'su-ccrows');
    rows.dataset.address = 'midi.cc_map';
    const empty = note('no mappings', 'su-empty');
    const renderRows = () => {
      const map = value('midi.cc_map') || {};
      const list = [...ccRows(map).map((r) => ({ ...r, draft: null })),
        ...drafts.map((d) => ({ cc: d.cc, target: null, draft: d }))];
      rows.replaceChildren(...list.map((r) => ccRow(r, map)), ...(list.length ? [] : [empty]));
      renderActivity(store.readout('midi'));
    };
    const commitMap = (next) => ctx.set('midi.cc_map', next);
    function ccRow(r, map) {
      const row = el('div', `su-ccrow${r.draft ? ' draft' : ''}`);
      row.dataset.cc = String(r.cc);
      const input = el('input', 'su-cc');
      input.value = String(r.cc);
      input.setAttribute('inputmode', 'numeric');
      input.title = 'CC 0–127: type, or turn the wheel';
      const apply = (cc) => {
        if (cc === null || cc === r.cc) { input.value = String(r.cc); return; }
        if (r.draft) {
          r.draft.cc = cc;
          renderRows();
          return;
        }
        const { map: next, replaced } = withRow(map, r.cc, cc, r.target);
        commitMap(next);
        display.show(`CC ${cc}`, replaced ? `replaces ${targetLabel(replaced)}` : targetLabel(r.target));
      };
      input.addEventListener('keydown', (e) => {
        e.stopPropagation();
        if (e.key === 'Enter') input.blur();
        else if (e.key === 'Escape') { input.value = String(r.cc); input.blur(); }
      });
      input.addEventListener('change', () => apply(parseCc(input.value)));
      input.addEventListener('wheel', (e) => {
        e.preventDefault();
        if (!e.deltaY) return;
        apply(Math.min(127, Math.max(0, r.cc - Math.sign(e.deltaY))));
      }, { passive: false });
      const target = el('div', 'lbox su-target', r.target ? targetLabel(r.target) : 'choose a control');
      if (followsChannel(r.target)) target.append(el('small', '', ' sel ch'));
      target.title = r.target || '';
      target.addEventListener('click', (e) => {
        e.stopPropagation();
        const box = target.getBoundingClientRect();
        openTargets(box.left, box.bottom + 2, r.target, (t) => {
          if (!t) return;
          if (r.draft) drafts = drafts.filter((d) => d !== r.draft);
          const { map: next, replaced } = withRow(value('midi.cc_map'), r.draft ? null : r.cc, r.cc, t);
          commitMap(next);
          if (r.draft) renderRows();
          display.show(`CC ${r.cc}`, replaced && replaced !== t ? `replaces ${targetLabel(replaced)}` : targetLabel(t));
        });
      });
      const activity = el('span', 'su-act');
      activity.append(el('i', 'su-act-led'), el('b', 'su-act-bar'));
      const remove = button('✕', 'su-small su-remove');
      remove.title = 'remove this mapping';
      remove.addEventListener('click', () => {
        if (r.draft) { drafts = drafts.filter((d) => d !== r.draft); renderRows(); return; }
        commitMap(withoutCc(value('midi.cc_map'), r.cc));
        display.show(`CC ${r.cc}`, 'removed');
      });
      row.append(el('span', 'su-cclbl', 'CC'), input, el('span', 'su-arrow', '→'), target, activity, remove);
      return row;
    }
    const lastCounts = new Map();
    function renderActivity(midi) {
      const pickup = midi && midi.pickup;
      for (const row of rows.querySelectorAll('.su-ccrow')) {
        const cc = Number(row.dataset.cc);
        const a = ccActivity(pickup, cc);
        const bar = row.querySelector('.su-act-bar');
        bar.style.width = a && a.physical !== null ? `${Math.round(a.physical * 100)}%` : '0';
        const count = a ? a.count : 0;
        if (lastCounts.has(cc) && lastCounts.get(cc) !== count) {
          row.classList.add('cc-hit');
          clearTimeout(row.timer);
          row.timer = setTimeout(() => row.classList.remove('cc-hit'), 120);
        }
        lastCounts.set(cc, count);
      }
    }
    watch('midi.cc_map', renderRows);
    tabOffs.push(store.watchReadout('midi', renderActivity, { now: false }));

    const add = button('+ add', 'su-small');
    add.id = 'su-cc-add';
    add.addEventListener('click', () => {
      const cc = freeCc(value('midi.cc_map'), drafts.map((d) => d.cc));
      if (cc === null) { display.show('CC MAPPINGS', 'all 128 CCs are mapped'); return; }
      drafts.push({ id: (draftIds += 1), cc });
      renderRows();
      rows.scrollTop = rows.scrollHeight;
    });
    const clear = button('clear all', 'su-small');
    clear.id = 'su-cc-clear';
    clear.addEventListener('click', () => {
      const n = ccRows(value('midi.cc_map')).length;
      drafts = [];
      if (!n) { renderRows(); return; }
      sheets.ask('Clear every CC mapping?', `${n} mapping${n === 1 ? '' : 's'} will be removed.`, { yes: 'clear all' })
        .then((yes) => {
          if (!yes) return;
          commitMap({});
          display.show('CC MAPPINGS', 'cleared');
        });
    });

    const bend = el('div', 'lbox su-choice su-bend');
    bend.dataset.address = 'midi.pitchbend_target';
    bend.style.setProperty('--lw', '170px');
    watch('midi.pitchbend_target', (t) => {
      bend.textContent = targetLabel(t);
      if (followsChannel(t)) bend.append(el('small', '', ' sel ch'));
    });
    bend.addEventListener('click', (e) => {
      e.stopPropagation();
      const r = bend.getBoundingClientRect();
      openTargets(r.left, r.bottom + 2, value('midi.pitchbend_target'), (t) => {
        ctx.set('midi.pitchbend_target', t);
        display.show('PITCH BEND', targetLabel(t));
      }, { none: '(none)' });
    });

    const learnHere = button('learn on the panel', 'su-small');
    learnHere.id = 'su-learn';
    learnHere.addEventListener('click', () => {
      close();
      display.show('MIDI LEARN', 'right-click a control', 4000);
    });

    const ccs = section('CC mappings', learning, rows,
      el('div', 'su-row su-cc-tools'),
      field('pitch bend', bend),
      note('Sound parameters follow the selected channel (sel ch). To learn a mapping, right-click a control on the panel ▸ MIDI learn.'));
    ccs.querySelector('.su-cc-tools').append(add, clear, learnHere);

    body.append(input, ccs);
    return body;
  }

  // ------------------------------------------------------------ synthesis tab
  function synthesisTab() {
    const body = el('div', 'su-cols one');
    const knob = el('px-knob');
    knob.dataset.address = 'pref.smoothing_ms';
    knob.setAttribute('label', 'smoothing');
    knob.setAttribute('name', 'parameter smoothing');
    knob.setAttribute('size', '56');
    const row = el('div', 'su-row');
    row.append(knob, note('Parameter smoothing: how long a knob change glides, on all 8 channels. '
      + 'Shorter is snappier, longer avoids zipper noise. Applies at once.'));
    body.append(section('synthesis', row));
    return body;
  }

  // ------------------------------------------------------------ ai tab
  function aiTab() {
    const body = el('div', 'su-cols one');
    const extras = note('', 'su-ml');
    watchAll(['ai.available', 'ai.install_command'], () => {
      const available = value('ai.available');
      extras.hidden = available !== false;
      extras.textContent = available === false
        ? `The ML extras are not installed, so the AI generators are off. Install them from the AI drum generator page (${value('ai.install_command') || 'pip install'}).`
        : '';
    });
    const model = (kind, title, temperatureLabel) => {
      const address = `pref.ai.${kind}_model`;
      const name = el('span', 'su-path');
      name.dataset.address = address;
      const render = () => {
        const models = value('ai.models') || {};
        name.textContent = modelText(value(address), models[kind]);
        name.title = value(address) || (models[kind] && models[kind].path) || '';
      };
      watchAll([address, 'ai.models'], render);
      const browse = button('browse…', 'su-small');
      browse.dataset.kind = kind;
      browse.classList.add('su-browse');
      browse.addEventListener('click', () => {
        const current = value(address);
        client.openFile({ title: `${title} checkpoint`, filters: MODEL_FILTERS,
          folder: current ? String(current).replace(/[\\/][^\\/]*$/, '') : undefined }).then((path) => {
          if (!path) return;
          ctx.set(address, path);
          display.show(title.toUpperCase(), modelText(path));
        });
      });
      const clear = button('clear', 'su-small');
      clear.dataset.kind = kind;
      clear.classList.add('su-clear');
      clear.addEventListener('click', () => {
        ctx.set(address, null);
        display.show(title.toUpperCase(), 'bundled');
      });
      const temp = el('px-knob');
      temp.dataset.address = `pref.ai.${kind}_temperature`;
      temp.setAttribute('label', 'temperature');
      temp.setAttribute('name', temperatureLabel);
      temp.setAttribute('size', '44');
      const row = el('div', 'su-row');
      row.append(temp, note(kind === 'pattern'
        ? 'Used by the pattern menu ▸ randomize (AI) and the AI generator\'s patterns. Higher is wilder.'
        : 'Used by the AI drum generator for new drum patches. Higher is wilder.'));
      return section(title, field('model', name, browse, clear), row);
    };
    body.append(extras,
      model('pattern', 'pattern model', 'pattern temperature'),
      model('patch', 'drum patch model', 'patch temperature'),
      note('The AI drum generator page shows the same models and temperatures.'));
    return body;
  }

  // ------------------------------------------------------------ display tab
  function displayTab() {
    const body = el('div', 'su-cols one');
    const gpu = el('px-toggle');
    gpu.dataset.address = 'pref.web.gpu';
    gpu.setAttribute('label', 'use the GPU');
    gpu.setAttribute('name', 'GPU rendering');
    const row = el('div', 'su-row');
    row.append(gpu, note('Draws this panel with the graphics card. Off, it draws in software: '
      + 'a little more CPU, but it avoids crashes of some graphics drivers (off by default on Windows). '
      + 'Applies at the next start of Pythonic.'));
    body.append(section('display', row));
    // The display says when it applies, instead of the toggle's plain on / off
    gpu.addEventListener('px-touch', (e) => {
      e.stopPropagation();
      display.show('GPU RENDERING', `${e.detail.value ? 'on' : 'off'} at the next start`);
    });
    return body;
  }

  const BUILDERS = { audio: audioTab, midi: midiTab, synthesis: synthesisTab, ai: aiTab, display: displayTab };

  // ------------------------------------------------------------ the sheet
  function clearTab() {
    tabOffs.forEach((off) => off());
    tabOffs = [];
  }

  function showTab(name) {
    tab = tabOf(name);
    clearTab();
    if (tab !== 'midi') drafts = [];
    const bodyEl = root.querySelector('.su-body');
    bodyEl.replaceChildren(); // dots and lists of the old tab go first
    for (const b of root.querySelectorAll('.su-tabs .btn')) b.classList.toggle('on', b.dataset.tab === tab);
    root.dataset.tab = tab;
    showError('');
    bodyEl.append(BUILDERS[tab]());
    if (card) card.style.width = `${TAB_WIDTHS[tab]}px`;
  }

  function build() {
    const node = el('div', 'setup');
    const head = el('div', 'su-head');
    const tabs = el('div', 'su-tabs');
    for (const name of SETUP_TABS) {
      const b = button(name);
      b.dataset.tab = name;
      b.addEventListener('click', () => showTab(name));
      tabs.append(b);
    }
    const closeButton = button('✕', 'su-close');
    closeButton.title = 'close';
    closeButton.addEventListener('click', () => close());
    head.append(el('span', 'su-title', 'setup'), tabs, el('span', 'su-sp'), closeButton);
    const error = el('div', 'su-error');
    error.hidden = true;
    node.append(head, error, el('div', 'su-body'));
    return node;
  }

  function open(options = {}) {
    if (!root) {
      root = build();
      card = sheets.show('setup', root, {
        dismissable: true, className: 'setup-sheet',
        onHide: () => {
          clearTab();
          drafts = [];
          root = null;
          card = null;
        },
      });
    }
    showTab(options.tab);
    return root;
  }

  function close() {
    if (root) sheets.hide('setup');
  }

  const unregister = panel.registerPage('setup', open);

  return {
    open,
    close,
    get tab() { return root ? tab : null; },
    get element() { return root; },
    destroy() { close(); unregister(); },
  };
}
