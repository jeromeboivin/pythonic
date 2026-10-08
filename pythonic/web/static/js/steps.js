// The step row and step entry (map decision #8): the step-mode buttons in the
// left column (trig, accent, velo, fill, prob, sub, last step, all ch, follow,
// ⊞ matrix), the page bars, step numbers and 16 pads of the bottom row, and the
// matrix (all 8 channels of the page) shown in the edit rack drawer.
//
// Pads always show every property of the selected channel's steps (trigger
// in its beat-group colour, velocity as brightness, accent dot, fill stripe,
// probability below 100 %, substep dots); the step mode decides what a press
// edits: trig / accent / fill toggle on click and paint along the row (a
// stroke is one undo step), velo and prob are vertical drags (200 design px =
// the range, Shift x0.1), sub opens the substep menu. "last step" arms: the
// next pad pressed, on any page, ends the pattern. "all ch" makes edits hit
// the same step on all 8 channels (muted ones included). Follow keeps the
// pads on the playing page; picking a page turns it off. The matrix cells
// edit the same way, row by row.
//
// The page reads lanes (`pattern.<P>.ch<N>.<field>`, lists) and writes steps
// (`pattern.<P>.ch<N>.step<S>.<field>`), echoing the lanes at once.
//
// Kit mode (TR-8 KIT, the `kit` button beside the PROGRAM heading): the pads
// become the 16 programs, named by their kits (`program.names`), lit when
// they hold sounds, the current one blinking; a press switches program
// (`program.select`) while the pattern plays on. A step mode, last step or a
// page bar goes back to the steps.
//
// Inst mode (TR-8 INST, the `inst` button beside it): the pads become the
// factory drum patches (`factory.patches`) of the selected channel's drum
// family (toms, hats, cymbals and shakers stand in for each other; all of
// them, paged by the page bars, when the channel has no drum type), the one
// playing blinking; a press loads it into the channel (`drum_patch.load`
// with `factory`) and, while stopped, plays it. A channel button picks the
// channel whose sound the pads choose.

import { guessDrumType, sameKind } from './drum-type.js';
import {
  CHANNELS, cleanSubsteps, dragValue, editChannels, FIELDS, followPage, groupOf, laneAddress,
  lengthAddress, MODES, padView, PAGE_SIZE, PAGES, pageCount, pageRange, PATTERNS, playheadIndex,
  stepAddress, stepAt, SUBSTEP_PRESETS, withStep,
} from './steps-logic.js';

const MODE_NAMES = Object.fromEntries(MODES.map(([f, , name]) => [f, name]));

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

/** The selected pattern's index: the transport's, else pattern.selected. */
export function selectedPattern(store) {
  const t = store.readout('transport');
  if (t && Number.isInteger(t.selected_pattern)) return t.selected_pattern;
  const i = PATTERNS.indexOf(store.value('pattern.selected'));
  return i < 0 ? 0 : i;
}

/** A pad or matrix cell: the lit layer, accent dot, fill stripe, probability, substeps, bar, value. */
function stepCell(cls, step, channel = null) {
  const cell = el('div', `${cls} stp g${groupOf(step)}`);
  cell.dataset.step = String(step);
  if (channel) cell.dataset.channel = String(channel);
  const bar = el('span', 'bar');
  bar.append(el('i'));
  cell.append(el('i', 'glow'), el('i', 'acc'), el('i', 'fill'), el('span', 'prob'), el('span', 'sub'), bar,
    el('span', 'pv'));
  return cell;
}

function paintCell(cell, step, view) {
  if (cell.dataset.step !== String(step)) {
    cell.dataset.step = String(step);
    cell.className = cell.className.replace(/\bg\d\b/, `g${groupOf(step)}`);
  }
  cell.classList.toggle('on', view.on);
  cell.classList.toggle('out', view.out);
  cell.classList.toggle('accent', view.acc);
  cell.classList.toggle('filled', view.fill);
  cell.classList.toggle('barred', view.bar !== null);
  cell.style.setProperty('--lvl', view.level.toFixed(3));
  const [, , , prob, sub, bar, pv] = cell.children;
  prob.textContent = view.prob || '';
  const dots = view.sub.map((on) => (on ? 'o' : '-')).join('');
  if (sub.dataset.dots !== dots) {
    sub.dataset.dots = dots;
    sub.replaceChildren(...view.sub.map((on) => el('i', on ? '' : 'x')));
  }
  bar.firstChild.style.height = view.bar === null ? '0' : `${(view.bar * 100).toFixed(1)}%`;
  pv.textContent = view.text || '';
}

/**
 * Fill the step-entry and steps slots and build the matrix. `drawer` shows
 * the matrix; `onView({letter, page})` tells the panel what the pads show.
 */
export function mountStepRow({ store, client, ctx, display, slot, drawer, onView = () => {} }) {
  const state = { mode: 'trig', page: 0, follow: true, allCh: false, armed: false, selected: 0, select: null,
    instPage: 0 };
  const offs = [];
  let patternOffs = [];
  let stroke = null;

  const letter = () => PATTERNS[state.selected];
  const length = () => {
    const n = store.value(lengthAddress(letter()));
    return Number.isInteger(n) ? n : PAGE_SIZE;
  };
  const lanesOf = (channel) => Object.fromEntries(FIELDS.map((f) => [f, store.value(laneAddress(letter(), channel, f))]));
  const channel = () => store.value('global.channel') || 1;

  // ------------------------------------------------------------ step entry (left column)
  const entry = slot('step-entry');
  entry.classList.remove('slot-mark');
  const modes = el('div', 'modes');
  const modeButtons = MODES.map(([field, label]) => {
    const b = el('button', 'btn sq mode', label);
    b.type = 'button';
    b.dataset.mode = field;
    b.addEventListener('click', () => setMode(field));
    return b;
  });
  const button = (id, label, cls = '') => {
    const b = el('button', `btn sq ${cls}`.trim(), label);
    b.type = 'button';
    b.id = id;
    return b;
  };
  const lastStep = button('last-step', 'last step', 'red');
  const allCh = button('all-ch', 'all ch');
  const follow = button('follow', 'follow');
  const matrixToggle = button('matrix-toggle', '⊞ matrix');
  modes.append(...modeButtons, lastStep, allCh, follow, matrixToggle);
  entry.replaceChildren(modes);

  lastStep.addEventListener('click', () => {
    state.select = null;
    state.armed = !state.armed;
    display.show('LAST STEP', state.armed ? 'press the last pad' : 'off');
    render();
  });
  allCh.addEventListener('click', () => {
    state.allCh = !state.allCh;
    display.show('ALL CH', state.allCh ? 'edits hit all 8 channels' : 'off');
    render();
  });
  follow.addEventListener('click', () => {
    state.follow = !state.follow;
    display.show('FOLLOW', state.follow ? 'on' : 'off');
    showPlayhead();
    render();
  });
  matrixToggle.addEventListener('click', () => {
    const shown = drawer.toggle('matrix', () => matrix, { onHide: () => render() });
    display.show('MATRIX', shown ? `pattern ${letter()}` : 'off');
    render();
  });

  function setMode(field) {
    state.select = null;
    state.mode = field;
    display.show('STEP MODE', MODE_NAMES[field]);
    render();
  }

  // ------------------------------------------------------------ the step row (bottom)
  const row = slot('steps');
  const bars = el('div', 'pgbars');
  const pageBars = Array.from({ length: PAGES }, (_, p) => {
    const bar = el('div', 'pg');
    bar.dataset.page = String(p);
    bar.append(el('span', '', `steps ${pageRange(p).replace('-', '–')}`), el('i'));
    bar.addEventListener('click', () => {
      if (state.select === 'inst' && p < pageCount(instChoices().length)) {
        state.instPage = p;
        render();
        return;
      }
      state.select = null;
      state.page = p;
      state.follow = false;
      display.show('PAGE', `steps ${pageRange(p)}`);
      render();
    });
    return bar;
  });
  bars.append(...pageBars);
  const numbers = el('div', 'pnums');
  const numberCells = Array.from({ length: PAGE_SIZE }, () => el('div'));
  numbers.append(...numberCells);
  const padRow = el('div', 'pads');
  const pads = Array.from({ length: PAGE_SIZE }, (_, i) => stepCell('pad', i + 1));
  padRow.append(...pads);

  // ------------------------------------------------------------ kit mode
  const kitRow = el('div', 'kits');
  const kitPads = Array.from({ length: PAGE_SIZE }, (_, i) => {
    const pad = el('div', 'kit');
    pad.dataset.program = String(i + 1);
    pad.append(el('b', '', String(i + 1)), el('span', 'kn'));
    pad.addEventListener('click', () => {
      if (state.select === 'inst') pickInst(i); else pickKit(i + 1);
    });
    return pad;
  });
  kitRow.append(...kitPads);
  row.replaceChildren(bars, numbers, padRow, kitRow);
  const kitButton = slot('kit-mode');
  if (kitButton) {
    kitButton.disabled = false;
    kitButton.classList.remove('slot');
    kitButton.id = 'kit-mode';
    kitButton.addEventListener('click', () => setSelect(state.select === 'kit' ? null : 'kit'));
  }
  const instButton = slot('inst-mode');
  if (instButton) {
    instButton.disabled = false;
    instButton.classList.remove('slot');
    instButton.id = 'inst-mode';
    instButton.addEventListener('click', () => setSelect(state.select === 'inst' ? null : 'inst'));
  }
  const channelName = () => String(store.value(`ch${channel()}.name`) || '');
  /** The factory drum patches the inst pads offer the selected channel. */
  const instChoices = () => sameKind(store.value('factory.patches'), channelName());
  const kitName = (program) => String((store.value('program.names') || [])[program - 1] || '');

  /** Show kits or drum patches on the pads instead of the steps: 'kit', 'inst' or null. */
  function setSelect(select) {
    state.select = select;
    state.armed = false;
    state.instPage = 0;
    if (select === 'kit') display.show('KIT', 'pads pick programs');
    else if (select === 'inst') showInst();
    else display.show('STEP MODE', MODE_NAMES[state.mode]);
    render();
  }

  function showInst() {
    const type = guessDrumType(channelName());
    const n = instChoices().length;
    display.show(`INST CH${channel()}`, `${type ? `${type}: ` : ''}${n} factory sound${n === 1 ? '' : 's'}`);
  }

  function pickInst(i) {
    const name = instChoices()[state.instPage * PAGE_SIZE + i];
    if (!name) return;
    const ch = channel();
    display.show(`INST CH${ch}`, name.toUpperCase());
    client.act('drum_patch.load', { factory: name, channel: ch }).then((event) => {
      if (event.status === 'error') { display.alert('DRUM_PATCH.LOAD', event.error); return; }
      const t = store.readout('transport');
      if (!(t && t.playing)) client.trigger(ch, 100);
    });
  }

  function renderInsts() {
    const choices = instChoices();
    const current = channelName();
    const base = state.instPage * PAGE_SIZE;
    kitPads.forEach((pad, i) => {
      const name = choices[base + i] || '';
      const [machine, ...rest] = name.split(' ');
      delete pad.dataset.program;
      pad.dataset.patch = name;
      pad.classList.toggle('on', !!name);
      pad.classList.toggle('cur', !!name && name === current);
      pad.firstChild.textContent = machine || '';
      pad.lastChild.textContent = rest.join(' ');
    });
  }

  function pickKit(program) {
    const name = kitName(program);
    const occupied = (store.value('program.occupied') || [])[program - 1];
    display.show('KIT', name ? `${program} ${name.toUpperCase()}` : `${program} ${occupied ? '' : 'new: a copy'}`.trim());
    client.act('program.select', { program }).then((event) => {
      if (event.status === 'error') display.alert('PROGRAM.SELECT', event.error);
    });
  }

  function renderKits() {
    const current = store.value('program.current');
    const occupied = store.value('program.occupied') || [];
    kitPads.forEach((pad, i) => {
      pad.dataset.program = String(i + 1);
      delete pad.dataset.patch;
      pad.firstChild.textContent = String(i + 1);
      pad.classList.toggle('on', !!occupied[i]);
      pad.classList.toggle('cur', i + 1 === current);
      pad.lastChild.textContent = kitName(i + 1);
    });
  }
  for (const address of ['program.current', 'program.occupied', 'program.names']) {
    offs.push(store.watch(address, () => { if (state.select === 'kit') renderKits(); }, { now: false }));
  }
  for (const address of ['factory.patches', ...CHANNELS.map((ch) => `ch${ch}.name`)]) {
    offs.push(store.watch(address, () => { if (state.select === 'inst') schedule(); }, { now: false }));
  }

  // ------------------------------------------------------------ the matrix (drawer page)
  const matrix = el('div', 'matrix');
  const head = el('div', 'mhead');
  const title = el('span', 'mtitle');
  const close = el('button', 'btn sq', '◀ edit rack');
  close.type = 'button';
  close.addEventListener('click', () => { drawer.hide('matrix'); render(); });
  head.append(el('span', 'hl', 'matrix'), title, close);
  const grid = el('div', 'mgrid');
  const labels = [];
  const cells = CHANNELS.map((ch) => {
    const label = el('div', 'mlbl');
    label.dataset.channel = String(ch);
    label.append(el('b', '', String(ch)), el('span', 'ty'), el('span', 'nm'));
    label.addEventListener('click', () => {
      ctx.set('global.channel', ch);
      display.show(`CH${ch}`, String(store.value(`ch${ch}.name`) || '').toUpperCase());
    });
    labels.push(label);
    const rowCells = Array.from({ length: PAGE_SIZE }, (_, i) => stepCell('cell', i + 1, ch));
    grid.append(label, ...rowCells);
    return rowCells;
  });
  matrix.append(head, grid);
  CHANNELS.forEach((ch, i) => offs.push(store.watch(`ch${ch}.name`, (name) => {
    labels[i].querySelector('.ty').textContent = guessDrumType(name) || '';
    labels[i].querySelector('.nm').textContent = name || '';
  })));

  // ------------------------------------------------------------ rendering
  let queued = false;
  const schedule = () => {
    if (queued) return;
    queued = true;
    queueMicrotask(() => { queued = false; render(); });
  };

  function render() {
    const n = length();
    const pages = pageCount(n);
    const base = state.page * PAGE_SIZE;
    lastStep.classList.toggle('on', state.armed);
    lastStep.dataset.address = lengthAddress(letter());
    allCh.classList.toggle('on', state.allCh);
    follow.classList.toggle('on', state.follow);
    const matrixShown = drawer.current === 'matrix';
    matrixToggle.classList.toggle('on', matrixShown);
    row.dataset.mode = state.mode;
    if (state.select) row.dataset.select = state.select; else delete row.dataset.select;
    if (kitButton) kitButton.classList.toggle('on', state.select === 'kit');
    if (instButton) instButton.classList.toggle('on', state.select === 'inst');
    if (state.select === 'kit') renderKits();
    if (state.select === 'inst') {
      if (state.instPage >= pageCount(instChoices().length)) state.instPage = 0;
      renderInsts();
    }
    modeButtons.forEach((b) => b.classList.toggle('on', !state.select && b.dataset.mode === state.mode));
    row.classList.toggle('armed', state.armed);
    const inst = state.select === 'inst';
    const barPages = inst ? pageCount(instChoices().length) : pages;
    pageBars.forEach((bar, p) => {
      bar.classList.toggle('on', p === (inst ? state.instPage : state.page));
      bar.classList.toggle('out', p >= barPages);
    });
    numberCells.forEach((cell, i) => {
      const step = base + i + 1;
      cell.textContent = step === n ? `${step}]` : String(step);
      cell.classList.toggle('end', step === n);
      cell.classList.toggle('out', step > n);
    });
    const lanes = lanesOf(channel());
    pads.forEach((pad, i) => paintCell(pad, base + i + 1, padView(stepAt(lanes, base + i, n), state.mode)));
    if (matrixShown) {
      title.textContent = `pattern ${letter()} · steps ${pageRange(state.page)}`;
      grid.dataset.mode = state.mode;
      CHANNELS.forEach((ch, r) => {
        const chLanes = lanesOf(ch);
        labels[r].classList.toggle('sel', ch === channel());
        cells[r].forEach((cell, i) => paintCell(cell, base + i + 1, padView(stepAt(chLanes, base + i, n), state.mode)));
      });
    }
    showPlayhead(false);
    onView({ letter: letter(), page: state.page });
  }

  /** Playhead outline, playing page bar and follow (from the transport readout). */
  function showPlayhead(followNow = true) {
    const t = store.readout('transport');
    const index = playheadIndex(t, state.selected);
    if (followNow && state.follow) {
      const page = followPage(t, state.selected);
      if (page !== null && page !== state.page && !stroke) {
        state.page = page;
        render();
        return;
      }
    }
    const playing = index < 0 ? -1 : Math.floor(index / PAGE_SIZE);
    pageBars.forEach((bar, p) => bar.classList.toggle('play', p === playing));
    const step = String(index + 1);
    pads.forEach((pad) => pad.classList.toggle('ph', pad.dataset.step === step));
    if (drawer.current === 'matrix') {
      for (const rowCells of cells) for (const cell of rowCells) cell.classList.toggle('ph', cell.dataset.step === step);
    }
  }

  // ------------------------------------------------------------ the selected pattern's lanes
  function watchPattern() {
    patternOffs.forEach((off) => off());
    patternOffs = [];
    const names = [lengthAddress(letter())];
    for (const ch of CHANNELS) for (const f of FIELDS) names.push(laneAddress(letter(), ch, f));
    for (const name of names) patternOffs.push(store.watch(name, schedule, { now: false }));
    const missing = names.filter((name) => !store.has(name));
    if (missing.length) {
      client.get(missing).then((values) => {
        const fresh = Object.fromEntries(Object.entries(values || {}).filter(([a]) => !store.has(a)));
        store.seed(fresh);
      });
    }
  }

  const followSelection = () => {
    const selected = selectedPattern(store);
    if (selected === state.selected && patternOffs.length) return false;
    state.selected = selected;
    state.armed = false;
    watchPattern();
    const n = store.has(lengthAddress(letter())) ? length() : PAGE_SIZE;
    if (state.page >= pageCount(n)) state.page = 0;
    return true;
  };
  followSelection();
  offs.push(store.watchReadout('transport', () => {
    if (followSelection()) render(); else showPlayhead();
  }));
  offs.push(store.watch('pattern.selected', () => { if (followSelection()) render(); }, { now: false }));
  offs.push(store.watch('global.channel', schedule, { now: false }));

  // ------------------------------------------------------------ edits
  const who = (ch) => (state.allCh ? 'ALL CH' : `CH${ch}`);

  /** Write one field of a step on the channels the edit reaches; returns how many changed. */
  function edit(ch, index, field, value, { gesture = null } = {}) {
    const n = length();
    if (index >= n) return 0;
    const byChannel = Object.fromEntries(CHANNELS.map((c) => [c, lanesOf(c)]));
    const targets = editChannels(field, ch, index, state.allCh, byChannel)
      .filter((c) => stepAt(byChannel[c], index, n)[field] !== value);
    if (!targets.length) return 0;
    if (gesture && !gesture.open) { gesture.open = true; client.beginGesture(); }
    for (const c of targets) {
      for (const [f, lane] of Object.entries(withStep(byChannel[c], index, field, value))) {
        store.assume(laneAddress(letter(), c, f), lane);
      }
      client.set(stepAddress(letter(), c, index + 1, field), value);
    }
    render(); // now, not at the end of the task: the stroke goes on under the pointer
    return targets.length;
  }

  /** A single edit: one undo step even when all ch writes 8 channels. */
  function editOnce(ch, index, field, value) {
    const gesture = state.allCh ? { open: false } : null;
    const changed = edit(ch, index, field, value, { gesture });
    if (gesture && gesture.open) client.endGesture();
    return changed;
  }

  function setLength(step) {
    state.armed = false;
    ctx.set(lengthAddress(letter()), step);
    display.show(`PATTERN ${letter()}`, `last step ${step}`);
    render();
  }

  function openSubsteps(cell, ch, index) {
    const current = stepAt(lanesOf(ch), index, length()).sub;
    const apply = (value) => {
      editOnce(ch, index, 'sub', value);
      display.show(`${who(ch)} STEP ${index + 1}`, `substeps ${value || 'none'}`);
    };
    const r = cell.getBoundingClientRect();
    const items = [[`substeps · step ${index + 1}`, null], ['none', () => apply(''), { current: current === '' }]];
    for (const s of SUBSTEP_PRESETS) items.push([s, () => apply(s), { current: current === s }]);
    items.push(['custom…', () => openCustom(cell, current, apply)]);
    ctx.openMenu(items, r.left, r.top, { className: 'subs' });
  }

  /** The custom substep field over a pad: o and - only; Enter applies, Esc cancels. */
  function openCustom(cell, current, apply) {
    const root = ctx.root;
    const box = el('div', 'px-entry');
    const input = el('input');
    input.value = current;
    input.placeholder = 'o-o-';
    box.append(input);
    root.append(box);
    const r = cell.getBoundingClientRect();
    const s = ctx.scale();
    const rootBox = root.getBoundingClientRect();
    box.style.left = `${(r.left - rootBox.left) / s}px`;
    box.style.top = `${(r.top - rootBox.top) / s - 34}px`;
    let done = false;
    const finish = (ok) => {
      if (done) return;
      done = true;
      if (ok) apply(cleanSubsteps(input.value));
      box.remove();
    };
    input.addEventListener('keydown', (e) => {
      e.stopPropagation();
      if (e.key === 'Enter') finish(true);
      else if (e.key === 'Escape') finish(false);
    });
    input.addEventListener('blur', () => finish(false));
    input.addEventListener('pointerdown', (e) => e.stopPropagation());
    input.focus();
    input.select();
  }

  function cellAt(container, x, y) {
    const hit = document.elementFromPoint(x, y);
    const cell = hit && hit.closest && hit.closest('.stp');
    return cell && container.contains(cell) ? cell : null;
  }

  function onPointerDown(e, container) {
    if (e.button !== 0) return;
    const cell = e.target.closest && e.target.closest('.stp');
    if (!cell || !container.contains(cell)) return;
    e.preventDefault();
    const index = Number(cell.dataset.step) - 1;
    const ch = cell.dataset.channel ? Number(cell.dataset.channel) : channel();
    if (state.armed) { setLength(index + 1); return; }
    const n = length();
    if (index >= n) return;
    const step = stepAt(lanesOf(ch), index, n);
    const field = state.mode;
    if (field === 'sub') { openSubsteps(cell, ch, index); return; }
    if (field === 'vel' || field === 'prob') {
      if (field === 'vel' && !step.trig) {
        display.show(`${who(ch)} STEP ${index + 1}`, 'velocity: no trigger');
        return;
      }
      stroke = { kind: 'drag', field, ch, index, start: step[field], value: step[field], y: e.clientY,
        fine: e.shiftKey, open: false, container };
      display.show(`${who(ch)} STEP ${index + 1}`, `${MODE_NAMES[field]} ${step[field]}${field === 'prob' ? ' %' : ''}`);
    } else {
      if (field !== 'trig' && !step.trig && !state.allCh) return;
      stroke = { kind: 'paint', field, ch, value: !step[field], seen: new Set(), open: false, container,
        row: cell.dataset.channel ? ch : null, last: index };
      paintAt(index);
    }
    try { container.setPointerCapture(e.pointerId); } catch (_) { /* synthetic events */ }
  }

  function paintAt(index) {
    const s = stroke;
    if (s.seen.has(index)) return;
    s.seen.add(index);
    if (edit(s.ch, index, s.field, s.value, { gesture: s })) {
      display.show(`${who(s.ch)} STEP ${index + 1}`, `${MODE_NAMES[s.field]} ${s.value ? 'on' : 'off'}`);
    }
  }

  function onPointerMove(e) {
    const s = stroke;
    if (!s) return;
    if (s.kind === 'paint') {
      const cell = cellAt(s.container, e.clientX, e.clientY);
      if (!cell || (s.row !== null && Number(cell.dataset.channel) !== s.row)) return;
      // Every step between the last one and this one (moves skip cells, events coalesce)
      const index = Number(cell.dataset.step) - 1;
      const dir = Math.sign(index - s.last);
      for (let k = s.last + dir; dir && k !== index + dir; k += dir) paintAt(k);
      s.last = index;
      return;
    }
    if (e.shiftKey !== s.fine) { // re-base so switching to fine does not jump
      s.start = s.value;
      s.y = e.clientY;
      s.fine = e.shiftKey;
      return;
    }
    const value = dragValue(s.field, s.start, (s.y - e.clientY) / ctx.scale(), s.fine);
    if (value === s.value) return;
    s.value = value;
    edit(s.ch, s.index, s.field, value, { gesture: s });
    display.show(`${who(s.ch)} STEP ${s.index + 1}`, `${MODE_NAMES[s.field]} ${value}${s.field === 'prob' ? ' %' : ''}`);
  }

  function endStroke() {
    const s = stroke;
    if (!s) return;
    stroke = null;
    if (s.open) client.endGesture();
    showPlayhead();
  }

  for (const container of [padRow, grid]) {
    container.addEventListener('pointerdown', (e) => onPointerDown(e, container));
    container.addEventListener('pointermove', onPointerMove);
    container.addEventListener('pointerup', endStroke);
    container.addEventListener('pointercancel', endStroke);
    container.addEventListener('lostpointercapture', endStroke);
    container.addEventListener('contextmenu', (e) => e.preventDefault());
  }
  offs.push(drawer.onChange(() => render()));

  render();

  return {
    state,
    matrix,
    render,
    setMode,
    setSelect,
    /** Show a page (0..3); turns follow off as a page pick does. */
    setPage(page) { state.page = page; state.follow = false; render(); },
    destroy() {
      offs.forEach((off) => off());
      patternOffs.forEach((off) => off());
      if (drawer.current === 'matrix') drawer.hide('matrix');
    },
  };
}
