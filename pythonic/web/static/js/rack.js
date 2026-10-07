// The selected channel's edit rack (map decisions #7, #11, #12, #16): the base
// page of the drawer under the face. One row of sections (rack-layout.js),
// every control bound to ch<N>.<suffix> of the selected channel and rebound
// when global.channel changes; the header shows the channel, its patch name
// and drum type, edit all and the drum patch menu.
//
// Modulation: the LFO 1 / LFO 2 / pump rows end with a → button (px-target).
// Pressing it arms click to assign: the header shows a bar in the source's
// colour (off, cancel), every control a source of this channel can modulate
// is outlined, the selected strip's tune / decay / level, master and morph
// included; clicking one sets the destination, any other control refuses on
// the display. The bands of modulated controls come from controls.js.
//
// Open / closed: the right column's edit rack button closes the drawer: the
// stage becomes the face (1600x700) and the window shrinks by the rack's
// height (bridge resizeWindow); the state is the preference pref.ui.rack_open
// (open until set). Pages of the drawer (the matrix, later PO-32 and AI)
// open a closed drawer for as long as they show (drawer.setOpener).

import { CONTROL_TAGS } from './controls.js';
import { guessDrumType } from './drum-type.js';
import { destinationOf, MOD_SOURCES, SOURCE_NAMES, targetName } from './modulation.js';
import { RACK_SECTIONS, stageHeightFor } from './rack-layout.js';
import { setStageHeight } from './stage.js';

export const RACK_PREF = 'pref.ui.rack_open';

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

const TAGS = {
  knob: 'px-knob', fader: 'px-fader', hfader: 'px-fader', switch: 'px-switch', list: 'px-list',
  toggle: 'px-toggle', target: 'px-target',
};

/** A control of the layout, bound later (data-suffix) to the selected channel. */
function controlElement(spec) {
  const node = document.createElement(TAGS[spec.type]);
  node.dataset.suffix = spec.s;
  node.dataset.title = spec.name || spec.label;
  if (spec.label) node.setAttribute('label', spec.label);
  node.setAttribute('size', String(spec.size || 30));
  if (spec.options) node.setAttribute('options', spec.options);
  if (spec.values) node.setAttribute('values', spec.values);
  if (spec.length) node.setAttribute('length', String(spec.length));
  if (spec.type === 'hfader') {
    node.setAttribute('orient', 'h');
    node.setAttribute('reverse', '');
    if (spec.ends) node.setAttribute('ends', spec.ends);
  }
  return node;
}

function sectionElement(section) {
  const node = el('section', `sec sec-${section.key}`);
  node.dataset.section = section.key;
  node.append(el('h4', '', section.title));
  if (section.rows) {
    for (const row of section.rows) {
      const r = el('div', 'r');
      r.append(...row.map(controlElement));
      node.append(r);
    }
    return node;
  }
  for (const source of section.sources) {
    const row = el('div', 'modrow');
    row.dataset.source = source.source;
    row.style.setProperty('--src', `var(--${source.source})`);
    row.append(el('span', 'nm', source.title), ...source.controls.map((spec) => {
      const node = controlElement(spec);
      if (spec.type === 'target') node.style.setProperty('--src', `var(--${source.source})`);
      return node;
    }));
    node.append(row);
  }
  return node;
}

/**
 * Build the rack into the drawer and wire the right column's edit rack
 * button. Returns { element, open, setOpen(open, {save}), arm(source),
 * disarm(), assigning, destroy() }.
 */
export function mountRack({ store, client, ctx, display, act, stage, slot, drawer }) {
  const offs = [];
  const watch = (address, fn, options) => { const off = store.watch(address, fn, options); offs.push(off); return off; };
  const selected = () => store.value('global.channel') || 1;

  // ------------------------------------------------------------ the rack
  const rack = el('div', 'rack');
  const head = el('div', 'rhead');
  const chLabel = el('span', 'rch', 'CH 1');
  const nameLabel = el('span', 'rname');
  const typeLabel = el('span', 'rtype');
  const bar = el('span', 'abar');
  const barText = el('span', 'abar-text');
  const barOff = el('button', 'btn sq', 'off');
  const barCancel = el('button', 'btn sq', 'cancel');
  barOff.type = 'button';
  barCancel.type = 'button';
  barOff.id = 'assign-off';
  barCancel.id = 'assign-cancel';
  bar.append(barText, barOff, barCancel);
  const note = el('span', 'ea-note', 'edits hit every unmuted channel');
  const editAll = document.createElement('px-toggle');
  editAll.id = 'edit-all';
  editAll.dataset.address = 'global.edit_all';
  editAll.setAttribute('label', 'edit all');
  editAll.setAttribute('name', 'edit all');
  const patchMenu = el('button', 'btn sq', 'drum patch ▾');
  patchMenu.type = 'button';
  patchMenu.id = 'patch-menu';
  head.append(chLabel, nameLabel, typeLabel, bar, el('span', 'sp'), note, editAll, patchMenu);
  const sections = el('div', 'secs');
  sections.append(...RACK_SECTIONS.map(sectionElement));
  rack.append(head, sections);
  const bound = [...rack.querySelectorAll('[data-suffix]')];
  const rows = [...rack.querySelectorAll('.modrow')];

  let armed = null; // the source whose destination click to assign sets
  let swallowClick = false;

  // ------------------------------------------------------------ the selected channel
  let channel = null;
  let channelOffs = [];
  const bindChannel = (n) => {
    if (n === channel) return;
    channel = n;
    disarm();
    for (const node of bound) {
      node.setAttribute('name', `ch${n} ${node.dataset.title}`);
      node.dataset.address = `ch${n}.${node.dataset.suffix}`;
    }
    chLabel.textContent = `CH ${n}`;
    channelOffs.forEach((off) => off());
    channelOffs = [
      store.watch(`ch${n}.name`, (name) => {
        nameLabel.textContent = name || '';
        typeLabel.textContent = guessDrumType(name) || '';
      }),
      ...rows.map((row) => store.watch(`ch${n}.${row.dataset.source}.on`, (on) => {
        row.classList.toggle('off', !on);
      })),
      // The outline of the current destination follows the target
      ...MOD_SOURCES.map((src) => store.watch(`ch${n}.${src}.target`, () => { if (armed) mark(); },
        { now: false })),
    ];
    for (const a of [`ch${n}.name`, ...rows.map((row) => `ch${n}.${row.dataset.source}.on`)]) ctx.ensure(a);
  };
  bindChannel(selected());
  watch('global.channel', (n) => bindChannel(n || 1));
  watch('global.edit_all', (on) => rack.classList.toggle('edit-all', !!on));
  drawer.setBase(rack);

  // ------------------------------------------------------------ click to assign

  const controls = () => [...stage.querySelectorAll(CONTROL_TAGS.join(','))]
    .filter((node) => node.tagName !== 'PX-TARGET');

  function mark() {
    const n = selected();
    const current = armed ? store.value(`ch${n}.${armed}.target`) : null;
    for (const node of controls()) {
      const hit = armed ? destinationOf(node.dataset.address, n).target : null;
      node.classList.toggle('assignable', !!hit);
      node.classList.toggle('assigned', !!hit && hit === current);
    }
  }

  function arm(source) {
    armed = source;
    stage.classList.add('assigning');
    stage.style.setProperty('--assign', `var(--${source})`);
    bar.style.setProperty('--src', `var(--${source})`);
    barText.textContent = `${SOURCE_NAMES[source]} → click a control, here or on the face`;
    for (const node of rack.querySelectorAll('px-target')) {
      node.classList.toggle('armed', node.dataset.suffix === `${source}.target`);
    }
    mark();
    display.show(`CH${selected()} ${SOURCE_NAMES[source]}`, 'click a destination');
  }

  function disarm() {
    if (!armed) return;
    armed = null;
    stage.classList.remove('assigning');
    for (const node of stage.querySelectorAll('.assignable, .assigned, px-target.armed')) {
      node.classList.remove('assignable', 'assigned', 'armed');
    }
  }

  function assign(target) {
    const n = selected();
    const source = armed;
    ctx.set(`ch${n}.${source}.target`, target);
    const on = store.value(`ch${n}.${source}.on`);
    display.show(`CH${n} ${SOURCE_NAMES[source]}`,
      `→ ${targetName(target)}${on || target === 'none' ? '' : ' (turn it on)'}`);
    disarm();
  }

  const onAssign = (e) => {
    const source = String(e.detail.address || '').split('.')[1];
    if (!MOD_SOURCES.includes(source)) return;
    if (armed === source) disarm(); else arm(source);
  };
  const onPointer = (e) => {
    swallowClick = false;
    if (!armed || e.button !== 0) return;
    if (e.target.closest('.abar, px-target, .px-menu, .px-entry')) return;
    const control = e.target.closest(CONTROL_TAGS.join(','));
    if (!control) return;
    e.preventDefault();
    e.stopPropagation();
    swallowClick = true;
    const result = destinationOf(control.dataset.address, selected());
    if (result.target) assign(result.target);
    else display.show(control.name || 'THIS CONTROL', result.refused);
  };
  const onClick = (e) => {
    if (!swallowClick) return;
    swallowClick = false;
    e.preventDefault();
    e.stopPropagation();
  };
  stage.addEventListener('px-assign', onAssign);
  stage.addEventListener('pointerdown', onPointer, true);
  stage.addEventListener('click', onClick, true);
  barOff.addEventListener('click', () => { if (armed) assign('none'); });
  barCancel.addEventListener('click', () => {
    disarm();
    display.show('DESTINATION', 'cancelled');
  });
  // The outlines follow the CTRL knobs (strip ctrl mode) and the current target
  watch('pref.ui.ctrl_knob', () => { if (armed) mark(); }, { now: false });
  offs.push(drawer.onChange(() => disarm()));

  // ------------------------------------------------------------ drum patch menu
  const finishSave = (verb, args, label) => act(verb, args).then((event) => {
    const r = event.result || {};
    if (event.status !== 'done') return;
    if (r.saved) { display.show(label, 'saved'); return; }
    if (!r.exists) return;
    const box = patchMenu.getBoundingClientRect();
    ctx.openMenu([
      [`${String(r.path || '').split(/[\\/]/).pop()} exists`, null],
      ['Replace it', () => finishSave(verb, { ...args, overwrite: true }, label)],
      ['Cancel', () => {}],
    ], box.left, box.bottom + 2);
  });
  const patchName = (n) => String(store.value(`ch${n}.name`) || `channel ${n}`);
  const patchItems = () => {
    const n = selected();
    return [
      [`Load drum patch into CH${n}…`, async () => {
        const path = await client.openFile({ title: `Load drum patch into channel ${n}`,
          filters: ['Drum patches (*.mtdrum)', 'All files (*)'] });
        if (!path) return;
        const event = await act('drum_patch.load', { path, channel: n });
        if (event.status === 'done') display.show(`CH${n} DRUM PATCH`, String(event.result.name || '').toUpperCase());
      }],
      [`Save drum patch of CH${n}…`, async () => {
        const path = await client.saveFile({ title: `Save drum patch of channel ${n}`,
          filters: ['Drum patches (*.mtdrum)'], name: `${patchName(n)}.mtdrum`, suffix: 'mtdrum' });
        if (path) finishSave('drum_patch.save', { path, channel: n }, `CH${n} DRUM PATCH`);
      }],
      [`Export CH${n} hit as WAV…`, async () => {
        const path = await client.saveFile({ title: `Export channel ${n} as WAV`,
          filters: ['WAV audio (*.wav)'], name: `${patchName(n)}.wav`, suffix: 'wav' });
        if (path) finishSave('export.drum_wav', { path, channel: n }, `CH${n} WAV`);
      }],
    ];
  };
  patchMenu.addEventListener('click', (e) => {
    e.stopPropagation();
    const box = patchMenu.getBoundingClientRect();
    ctx.openMenu(patchItems(), box.left, box.bottom + 2);
  });

  // ------------------------------------------------------------ open / closed
  const toggle = slot('rack-toggle');
  toggle.disabled = false;
  toggle.classList.remove('slot');
  toggle.id = 'rack-toggle';
  toggle.textContent = 'edit rack';
  let open = true;

  /** Open or close the drawer; `save` keeps the state as the preference. */
  function setOpen(next, { save = false } = {}) {
    next = !!next;
    if (save) ctx.set(RACK_PREF, next);
    if (next === open) return;
    const from = stageHeightFor(open);
    open = next;
    if (!open) disarm();
    stage.classList.toggle('rack-closed', !open);
    toggle.classList.toggle('on', open);
    setStageHeight(stage, stageHeightFor(open));
    client.resizeWindow(from, stageHeightFor(open)).catch((err) => console.warn('resizeWindow', err));
  }
  toggle.classList.add('on');
  drawer.setOpener({ isOpen: () => open, setOpen: (next) => setOpen(next) });
  watch(RACK_PREF, (value) => { if (value === true || value === false) setOpen(value); });
  toggle.addEventListener('click', () => {
    const next = !open;
    if (!next && drawer.current) drawer.hide(); // a page that opened the drawer closes it as it hides
    setOpen(next, { save: true });
    display.show('EDIT RACK', next ? 'open' : 'closed');
  });

  return {
    element: rack,
    get open() { return open; },
    setOpen,
    arm,
    disarm,
    get assigning() { return armed; },
    destroy() {
      offs.forEach((off) => off());
      channelOffs.forEach((off) => off());
      stage.removeEventListener('px-assign', onAssign);
      stage.removeEventListener('pointerdown', onPointer, true);
      stage.removeEventListener('click', onClick, true);
    },
  };
}
