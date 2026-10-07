// Panel controls as custom elements, every one driven by the address in its
// data-address attribute and its describe() metadata:
//
//   <px-knob data-address="ch3.osc.decay" label="decay" name="CH3 DECAY" size="34">
//   <px-fader data-address="ch3.mix.level" length="128">
//   <px-fader data-address="ch3.mix.osc_noise" orient="h" reverse ends="osc,noise">
//   <px-toggle data-address="ch3.mute" label="mute">          (a lit button)
//   <px-switch data-address="global.step_rate">               (segmented, up to 5 options)
//   <px-list data-address="global.fill_rate">                 (a list for more options)
//   <px-target data-address="ch3.lfo1.target">                (a source's → destination button)
//   <px-display>                                              (the green two-line display)
//
// Controls bind themselves when they are inside a root that carries a control
// context (provideContext(root, ctx)); changing data-address rebinds them, and
// removing it disables the control (shown "off"). Values come from the store
// and go to the core through the client: drags are gestures (one undo step),
// wheel turns are bursts. Every touch dispatches a bubbling `px-touch` event
// ({address, name, value, text}) for the panel display. MIDI cues (CC badge,
// blinking LED, pickup ghost marker, learn pulse) and modulation arcs are
// drawn from the context, which reads them from the store and poll.

import { ccsFor, ghostPosition, isBendTarget, withoutAddress } from './midi-cues.js';
import { modulatedAddresses, modulationBand, sourceAddresses, targetName } from './modulation.js';
import {
  coerce, DRAG_RANGE_PX, dragPosition, editText, formatValue, fromPosition, isNumeric, parseValue,
  toPosition, WHEEL_FADER, WHEEL_KNOB, wheelValue,
} from './values.js';

const SVG = 'http://www.w3.org/2000/svg';

// ---------------------------------------------------------------- geometry

const angle = (n) => -135 + 270 * n;
function point(r, deg) {
  const t = (deg * Math.PI) / 180;
  return [50 + r * Math.sin(t), 50 - r * Math.cos(t)];
}
/** SVG path of an arc on the knob ring between two positions. */
export function ringArc(n0, n1, r = 44) {
  let [a0, a1] = [angle(n0), angle(n1)];
  if (a1 < a0) [a0, a1] = [a1, a0];
  if (a1 - a0 < 1) a1 = a0 + 1;
  const [x0, y0] = point(r, a0);
  const [x1, y1] = point(r, a1);
  return `M${x0.toFixed(2)} ${y0.toFixed(2)} A${r} ${r} 0 ${a1 - a0 > 180 ? 1 : 0} 1 ${x1.toFixed(2)} ${y1.toFixed(2)}`;
}

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function svg(tag, attrs) {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
}

// ---------------------------------------------------------------- context

/**
 * The context controls bind through: the store, the client and the metadata,
 * plus what is shared between controls (MIDI map and pickup, modulation,
 * the context menu and the exact-value field). `root` is the element the
 * menus open in (the stage).
 */
export function createControlContext({ store, client, meta = {}, root }) {
  const byAddress = new Map(); // address -> Set(control)
  const offs = [];
  let modulated = {};
  let lastCounts = {};
  const fetching = new Set();
  const queued = [];

  const ctx = {
    store, client, meta, root,
    /** Panel scale (the stage's transform), to turn screen pixels into design pixels. */
    scale() {
      const s = Number(root && root.dataset.scale);
      return s > 0 ? s : 1;
    },
    register(control, address) {
      if (!byAddress.has(address)) byAddress.set(address, new Set());
      byAddress.get(address).add(control);
    },
    unregister(control, address) {
      const set = byAddress.get(address);
      if (set) set.delete(control);
    },
    controls(address) { return [...(byAddress.get(address) || [])]; },
    /** Echo a value at once and send it to the core. */
    set(address, value, { burst = false } = {}) {
      store.assume(address, value);
      client.set(address, value, { burst });
    },
    /** Read an address the store has not seen yet (a control bound after boot);
     * the addresses asked for in one task go in one get. */
    ensure(address) {
      if (store.has(address) || fetching.has(address)) return;
      fetching.add(address);
      queued.push(address);
      if (queued.length > 1) return;
      Promise.resolve().then(() => {
        const batch = queued.splice(0);
        const done = () => batch.forEach((a) => fetching.delete(a));
        client.get(batch).then((values) => {
          done();
          const fresh = Object.fromEntries(Object.entries(values || {}).filter(([a]) => !store.has(a)));
          store.seed(fresh);
        }, done);
      });
    },
    selectedChannel: () => store.value('global.channel') || 1,
    ccs: (address) => ccsFor(address, store.value('midi.cc_map'), ctx.selectedChannel()),
    learning: () => store.value('midi.learning'),
    modulation: (address) => modulated[address] || null,
    pickup(address) {
      const midi = store.readout('midi');
      return (midi && midi.pickup && midi.pickup[address]) || null;
    },
    menu: null,
    destroy() { offs.forEach((off) => off()); ctx.closeMenus(); },
  };

  const refreshCues = () => { for (const set of byAddress.values()) for (const c of set) c.renderCues(); };
  // MIDI map, learn target and pitch bend change the cues of many controls
  for (const a of ['midi.cc_map', 'midi.learning', 'midi.pitchbend_target', 'global.channel']) {
    offs.push(store.watch(a, refreshCues, { now: false }));
  }
  // The sources' on / target values colour the modulation arcs
  for (let ch = 1; ch <= 8; ch += 1) {
    for (const a of sourceAddresses(ch)) offs.push(store.watch(a, () => {}, { now: false }));
  }

  offs.push(store.watchReadout('modulation', (readout) => {
    const next = modulatedAddresses(readout, store.value);
    const touched = new Set([...Object.keys(modulated), ...Object.keys(next)]);
    modulated = next;
    for (const address of touched) for (const c of ctx.controls(address)) c.renderModulation();
  }));

  offs.push(store.watchReadout('midi', (midi) => {
    const pickup = (midi && midi.pickup) || {};
    for (const [address, state] of Object.entries(pickup)) {
      const blink = lastCounts[address] !== undefined && state.count !== lastCounts[address];
      for (const c of ctx.controls(address)) {
        c.renderGhost();
        if (blink) c.blinkCc();
      }
    }
    lastCounts = Object.fromEntries(Object.entries(pickup).map(([a, s]) => [a, s.count]));
  }));

  // ------------------------------------------------------------ menus
  const toStage = (clientX, clientY) => {
    const r = root.getBoundingClientRect();
    const s = ctx.scale();
    return [(clientX - r.left) / s, (clientY - r.top) / s];
  };

  ctx.closeMenus = () => {
    if (!root) return;
    root.querySelectorAll(':scope > .px-menu, :scope > .px-entry').forEach((m) => m.remove());
  };

  /** Open a menu of [label, action] items (null = a separator) at a pointer position. */
  ctx.openMenu = (items, clientX, clientY, { className = '' } = {}) => {
    ctx.closeMenus();
    const menu = el('div', `px-menu ${className}`.trim());
    for (const item of items) {
      if (!item) { menu.append(el('hr')); continue; }
      const [label, action, opts = {}] = item;
      const row = el('div', `it${opts.current ? ' cur' : ''}${action ? '' : ' dis'}`, label);
      if (action) {
        row.addEventListener('pointerdown', (e) => e.stopPropagation());
        row.addEventListener('click', (e) => { e.stopPropagation(); ctx.closeMenus(); action(); });
      }
      menu.append(row);
    }
    root.append(menu);
    const [x, y] = toStage(clientX, clientY);
    const w = menu.offsetWidth;
    const h = menu.offsetHeight;
    menu.style.left = `${Math.max(4, Math.min(x, root.offsetWidth - w - 4))}px`;
    menu.style.top = `${Math.max(4, y + h > root.offsetHeight - 4 ? y - h : y)}px`;
    ctx.menu = menu;
    return menu;
  };

  /** The right-click menu of a control: reset, MIDI learn, CC mappings, pitch bend. */
  ctx.controlMenu = (control, clientX, clientY) => {
    const { address, meta: m } = control;
    if (!address || !m) return null;
    const items = [];
    if (!m.readonly && m.default !== null && m.default !== undefined) {
      items.push([`Reset to default (${formatValue(m, m.default)})`, () => control.commit(m.default)]);
    }
    if (!m.readonly) {
      items.push(null);
      items.push(ctx.learning() === address
        ? ['Cancel MIDI learn', () => client.act('midi.learn_cancel')]
        : ['MIDI learn (CC)', () => ctx.learn(control)]);
      const map = store.value('midi.cc_map');
      for (const cc of ctx.ccs(address)) {
        items.push([`Remove CC ${cc} mapping`, () => {
          const next = withoutAddress(address, map, ctx.selectedChannel());
          ctx.set('midi.cc_map', next);
        }]);
      }
      const bend = store.value('midi.pitchbend_target');
      items.push(isBendTarget(address, bend, ctx.selectedChannel())
        ? ['Remove pitch bend', () => ctx.set('midi.pitchbend_target', null)]
        : ['Assign pitch bend', () => ctx.set('midi.pitchbend_target', address)]);
      if (ctx.openCcMappings) items.push(null, ['CC mappings…', () => ctx.openCcMappings()]);
    }
    return ctx.openMenu(items, clientX, clientY);
  };

  /** Start MIDI learn for a control: the next CC maps to its address. */
  ctx.learn = (control) => {
    control.touch('MIDI LEARN', 'move a controller');
    return client.act('midi.learn', { target: control.address }).then((event) => {
      if (event.status === 'done' && event.result) {
        control.touch(control.name, `CC ${event.result.cc} learned`);
      }
      return event;
    });
  };

  /** The exact-value field over a control: Enter applies, Esc cancels. */
  ctx.openEntry = (control) => {
    const { meta: m } = control;
    if (!m || m.readonly || !(isNumeric(m) || m.kind === 'enum')) return null;
    ctx.closeMenus();
    const box = el('div', 'px-entry');
    const input = el('input');
    input.value = editText(m, control.value);
    box.append(input);
    root.append(box);
    const r = control.getBoundingClientRect();
    const [x, y] = toStage(r.left + r.width / 2, r.top + r.height / 2);
    box.style.left = `${x - 45}px`;
    box.style.top = `${y - 14}px`;
    let done = false;
    const finish = (apply) => {
      if (done) return;
      done = true;
      if (apply) {
        const value = parseValue(m, input.value);
        if (value !== null) control.commit(value);
      }
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
    return box;
  };

  if (root) {
    const close = (e) => {
      if (!e.target.closest || !e.target.closest('.px-menu, .px-entry')) ctx.closeMenus();
    };
    root.addEventListener('pointerdown', close, true);
    offs.push(() => root.removeEventListener('pointerdown', close, true));
  }
  return ctx;
}

/** Make a context available to every control inside root (now and later). */
export function provideContext(root, ctx) {
  root.dataset.pxRoot = '';
  root.pxContext = ctx;
  for (const node of root.querySelectorAll(CONTROL_TAGS.join(','))) {
    if (typeof node.attach === 'function') node.attach(ctx);
  }
}

function findContext(node) {
  const root = node.closest('[data-px-root]');
  return root ? root.pxContext : null;
}

// ---------------------------------------------------------------- base

class PxControl extends HTMLElement {
  static get observedAttributes() { return ['data-address', 'label']; }

  constructor() {
    super();
    this.ctx = null;
    this.address = null;
    this.meta = null;
    this._offs = [];
    this._built = false;
  }

  connectedCallback() {
    if (!this._built) { this._built = true; this.build(); this.listen(); }
    const ctx = findContext(this);
    if (ctx) this.attach(ctx);
  }

  disconnectedCallback() { this.detach(); }

  attributeChangedCallback(name) {
    if (!this._built) return;
    if (name === 'label' && this.lbl) this.lbl.textContent = this.getAttribute('label') || '';
    if (name === 'data-address' && this.ctx) this.attach(this.ctx, true);
  }

  /** Bind to the context and the address in data-address. */
  attach(ctx, force = false) {
    const address = this.dataset.address || null;
    if (!force && this.ctx === ctx && this.address === address && this._offs.length) return;
    this.detach();
    this.ctx = ctx;
    this.address = address;
    this.meta = address ? ctx.meta[address] || null : null;
    this.classList.toggle('off', !address);
    if (!address) { this.render(undefined); return; }
    ctx.register(this, address);
    this._offs.push(ctx.store.watch(address, (v) => this.render(v)));
    if (!ctx.store.has(address)) { this.render(undefined); ctx.ensure(address); }
    this.renderCues();
    this.renderModulation();
  }

  detach() {
    this._offs.forEach((off) => off());
    this._offs = [];
    if (this.ctx && this.address) this.ctx.unregister(this, this.address);
  }

  get value() { return this.ctx && this.address ? this.ctx.store.value(this.address) : undefined; }

  get enabled() { return !!(this.ctx && this.address && this.meta && !this.meta.readonly); }

  /** The name the display shows while the control is touched. */
  get name() {
    return (this.getAttribute('name') || this.getAttribute('label') || this.address || '').toUpperCase();
  }

  get text() { return formatValue(this.meta, this.value); }

  /** Tell the panel the control was touched (the display shows it). */
  touch(name = this.name, text = this.text) {
    this.dispatchEvent(new CustomEvent('px-touch', { bubbles: true,
      detail: { address: this.address, name, value: this.value, text } }));
  }

  /** Set a new value (one undo step, or part of the open gesture / burst). */
  commit(value, { burst = false } = {}) {
    if (!this.enabled) return;
    const next = coerce(this.meta, value);
    if (next !== this.value) this.ctx.set(this.address, next, { burst });
    this.touch();
  }

  // Hooks of the subclasses
  build() {}
  listen() {
    this.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      if (this.ctx && this.address) this.ctx.controlMenu(this, e.clientX, e.clientY);
    });
  }
  render() {}
  renderModulation() {}
  renderGhost() {}

  renderCues() {
    if (!this.ctx) return;
    const ccs = this.address ? this.ctx.ccs(this.address) : [];
    if (this.badge) this.badge.textContent = ccs.length ? `CC ${ccs.join(' ')}` : '';
    this.classList.toggle('mapped', ccs.length > 0);
    this.classList.toggle('learning', !!this.address && this.ctx.learning() === this.address);
    this.renderGhost();
  }

  blinkCc() {
    this.classList.add('cc-blink');
    clearTimeout(this._blink);
    this._blink = setTimeout(() => this.classList.remove('cc-blink'), 120);
  }

  /** Common parts under a continuous control: label, value, CC badge. */
  buildFoot() {
    this.lbl = el('span', 'lbl', this.getAttribute('label') || '');
    this.val = el('span', 'val', '');
    this.badge = el('span', 'cc', '');
    this.append(this.lbl, this.val, this.badge);
  }
}

// ---------------------------------------------------------------- continuous

/** Shared drag and wheel handling of knobs and faders. */
class PxContinuous extends PxControl {
  get dragLength() { return DRAG_RANGE_PX; }
  get wheelFraction() { return WHEEL_KNOB; }
  get position() { return toPosition(this.meta, this.value); }

  listen() {
    super.listen();
    this.addEventListener('pointerdown', (e) => this.onPointerDown(e));
    this.addEventListener('pointermove', (e) => this.onPointerMove(e));
    this.addEventListener('pointerup', (e) => this.endDrag(e));
    this.addEventListener('pointercancel', (e) => this.endDrag(e));
    this.addEventListener('lostpointercapture', (e) => this.endDrag(e));
    this.addEventListener('dblclick', (e) => {
      e.preventDefault();
      if (this.enabled) this.ctx.openEntry(this);
    });
    this.addEventListener('wheel', (e) => this.onWheel(e), { passive: false });
  }

  startPosition() { return this.position; }

  /** Screen pixels of a drag towards higher values (up). */
  dragPixels(d, e) { return d.y - e.clientY; }

  onPointerDown(e) {
    if (e.button !== 0 || !this.enabled || !isNumeric(this.meta)) return;
    e.preventDefault();
    try { this.setPointerCapture(e.pointerId); } catch (_) { /* synthetic events */ }
    this.drag = { start: this.position, x: e.clientX, y: e.clientY, fine: e.shiftKey, gesture: false };
    const jump = this.startPosition(e);
    if (jump !== this.drag.start) {
      this.beginGesture();
      this.drag.start = jump;
      this.commit(fromPosition(this.meta, jump));
    } else {
      this.touch();
    }
  }

  beginGesture() {
    if (this.drag && !this.drag.gesture) { this.drag.gesture = true; this.ctx.client.beginGesture(); }
  }

  onPointerMove(e) {
    const d = this.drag;
    if (!d) return;
    if (e.shiftKey !== d.fine) { // re-base so switching to fine does not jump
      d.start = this.position;
      d.x = e.clientX;
      d.y = e.clientY;
      d.fine = e.shiftKey;
      return;
    }
    const pixels = this.dragPixels(d, e) / this.ctx.scale();
    const value = fromPosition(this.meta, dragPosition(d.start, pixels, { length: this.dragLength, fine: d.fine }));
    if (value !== this.value) {
      this.beginGesture();
      this.commit(value);
    }
  }

  endDrag() {
    const d = this.drag;
    if (!d) return;
    this.drag = null;
    if (d.gesture) this.ctx.client.endGesture();
  }

  onWheel(e) {
    if (!this.enabled) return;
    e.preventDefault();
    if (!e.deltaY) return;
    // One notch per event: pixel deltas per notch differ by platform (53 to 120)
    const notches = -Math.sign(e.deltaY);
    const step = this.hasAttribute('wheel-step') ? Number(this.getAttribute('wheel-step')) : null;
    this.commit(wheelValue(this.meta, this.value, notches, { fraction: this.wheelFraction, step }), { burst: true });
  }
}

/** A knob: pointer and end dots on a cap, modulation band and ghost marker on the ring. */
class PxKnob extends PxContinuous {
  build() {
    const size = Number(this.getAttribute('size') || 34);
    this.style.setProperty('--size', `${size}px`);
    const face = svg('svg', { viewBox: '0 0 100 100', class: 'dial' });
    face.append(
      svg('circle', { class: 'end', cx: 16.8, cy: 83.2, r: 3 }),
      svg('circle', { class: 'end', cx: 83.2, cy: 83.2, r: 3 }),
    );
    this.modArc = svg('path', { class: 'mod', d: '' });
    this.cap = svg('circle', { class: 'cap', cx: 50, cy: 50, r: 31 });
    this.pointer = svg('line', { class: 'ptr', x1: 50, y1: 50, x2: 50, y2: 24 });
    this.ghost = svg('circle', { class: 'ghost', cx: 50, cy: 6, r: 5 });
    face.append(this.modArc, this.cap, this.pointer, this.ghost);
    const wrap = el('div', 'kwrap');
    wrap.append(face, el('i', 'cc-led'));
    this.face = face;
    this.append(wrap);
    this.buildFoot();
  }

  render(value) {
    const n = this.meta && value !== undefined ? toPosition(this.meta, value) : 0;
    const [x, y] = point(27, angle(n));
    this.pointer.setAttribute('x2', x.toFixed(2));
    this.pointer.setAttribute('y2', y.toFixed(2));
    this.dataset.position = n.toFixed(4);
    this.val.textContent = this.address ? formatValue(this.meta, value) : 'off';
    this.renderModulation();
    this.renderGhost();
  }

  renderModulation() {
    const mod = this.ctx && this.address && this.meta ? this.ctx.modulation(this.address) : null;
    const band = mod ? modulationBand(this.meta, this.value, mod.offset) : null;
    if (!band) {
      this.modArc.setAttribute('d', '');
      delete this.dataset.mod;
      delete this.dataset.modTo;
      return;
    }
    this.modArc.setAttribute('d', ringArc(band[0], band[1]));
    this.modArc.style.stroke = `var(--${mod.source})`;
    this.dataset.mod = mod.source;
    this.dataset.modTo = band[1].toFixed(4);
  }

  renderGhost() {
    const g = this.ctx && this.address && this.meta
      ? ghostPosition(this.ctx.pickup(this.address), this.position) : null;
    this.ghost.style.display = g === null ? 'none' : '';
    if (g !== null) {
      const [x, y] = point(44, angle(g));
      this.ghost.setAttribute('cx', x.toFixed(2));
      this.ghost.setAttribute('cy', y.toFixed(2));
      this.dataset.ghost = g.toFixed(4);
    } else {
      delete this.dataset.ghost;
    }
  }
}

/**
 * A fader: the track length is the drag range; clicking the track jumps.
 * Vertical by default; orient="h" lays it on its side (drag right), with
 * `reverse` the high end on the left, and `ends="left,right"` names the ends.
 */
class PxFader extends PxContinuous {
  get dragLength() { return Number(this.getAttribute('length') || 128); }
  get wheelFraction() { return WHEEL_FADER; }
  get horizontal() { return this.getAttribute('orient') === 'h'; }
  get reversed() { return this.hasAttribute('reverse'); }

  /** The place along the track (0 bottom / left, 1 top / right) of a position. */
  place(n) { return this.horizontal && this.reversed ? 1 - n : n; }

  build() {
    this.style.setProperty('--length', `${this.dragLength}px`);
    this.classList.toggle('h', this.horizontal);
    this.track = el('div', 'track');
    // The drag range, whatever the styles
    this.track.style[this.horizontal ? 'width' : 'height'] = `${this.dragLength}px`;
    this.track.style.position = 'relative';
    this.capEl = el('i', 'cap');
    this.modLine = el('b', 'mod');
    this.ghostLine = el('b', 'ghost');
    this.track.append(this.capEl, this.modLine, this.ghostLine);
    const wrap = el('div', 'fwrap');
    wrap.append(this.track, el('i', 'cc-led'));
    this.append(wrap);
    if (this.horizontal && this.getAttribute('ends')) {
      const ends = el('div', 'ends');
      ends.style.width = `${this.dragLength}px`;
      ends.append(...this.getAttribute('ends').split(',').map((t) => el('span', '', t.trim())));
      this.append(ends);
    }
    this.buildFoot();
  }

  dragPixels(d, e) {
    if (!this.horizontal) return d.y - e.clientY;
    return (e.clientX - d.x) * (this.reversed ? -1 : 1);
  }

  startPosition(e) {
    if (e.target === this.capEl || !this.track.contains(e.target)) return this.position;
    const r = this.track.getBoundingClientRect();
    const along = this.horizontal ? (e.clientX - r.left) / r.width : 1 - (e.clientY - r.top) / r.height;
    return this.place(Math.min(1, Math.max(0, along)));
  }

  /** Put a marker (cap, modulation or ghost line) at a position. */
  put(node, n) {
    const p = this.place(n) * 100;
    if (this.horizontal) node.style.left = `${p}%`; else node.style.top = `${100 - p}%`;
  }

  render(value) {
    const n = this.meta && value !== undefined ? toPosition(this.meta, value) : 0;
    this.put(this.capEl, n);
    this.dataset.position = n.toFixed(4);
    this.val.textContent = this.address ? formatValue(this.meta, value) : 'off';
    this.renderModulation();
    this.renderGhost();
  }

  renderModulation() {
    const mod = this.ctx && this.address && this.meta ? this.ctx.modulation(this.address) : null;
    const band = mod ? modulationBand(this.meta, this.value, mod.offset) : null;
    if (!band) {
      this.modLine.style.display = 'none';
      delete this.dataset.mod;
      delete this.dataset.modTo;
      return;
    }
    this.modLine.style.display = 'block';
    this.put(this.modLine, band[1]);
    this.modLine.style.background = `var(--${mod.source})`;
    this.modLine.style.color = `var(--${mod.source})`;
    this.dataset.mod = mod.source;
    this.dataset.modTo = band[1].toFixed(4);
  }

  renderGhost() {
    const g = this.ctx && this.address && this.meta
      ? ghostPosition(this.ctx.pickup(this.address), this.position) : null;
    this.ghostLine.style.display = g === null ? 'none' : 'block';
    if (g !== null) {
      this.put(this.ghostLine, g);
      this.dataset.ghost = g.toFixed(4);
    } else {
      delete this.dataset.ghost;
    }
  }
}

// ---------------------------------------------------------------- stepped

/** Wheel and right-click shared by the stepped controls. */
class PxStepped extends PxControl {
  listen() {
    super.listen();
    this.addEventListener('wheel', (e) => {
      if (!this.enabled) return;
      e.preventDefault();
      if (!e.deltaY) return;
      this.commit(this.stepValue(-Math.sign(e.deltaY)), { burst: true });
    }, { passive: false });
  }

  /** The value `notches` options away (a `values` subset steps within itself). */
  stepValue(notches) {
    if (!this.hasAttribute('values')) return wheelValue(this.meta, this.value, notches);
    const values = this.options().map(([v]) => v);
    const i = values.indexOf(this.value);
    if (i < 0) return values[notches > 0 ? 0 : values.length - 1];
    return values[Math.min(values.length - 1, Math.max(0, i + notches))];
  }

  /** Every value of the address with its display name: the enum labels
   * (named by `options`), an int range or on / off. */
  allOptions() {
    const m = this.meta;
    if (!m) return [];
    const names = (this.getAttribute('options') || '').split(',').map((s) => s.trim()).filter(Boolean);
    let values = [];
    if (m.kind === 'enum') values = m.labels || [];
    else if (m.kind === 'int' && m.minimum !== null && m.maximum !== null) {
      for (let v = m.minimum; v <= m.maximum; v += 1) values.push(v);
    } else if (m.kind === 'bool') values = [false, true];
    return values.map((v, i) => [v, names[i] || formatValue(m, v)]);
  }

  /** The options offered: [value, text], all or the `values` subset (enum labels). */
  options() {
    const all = this.allOptions();
    if (!this.hasAttribute('values')) return all;
    const keep = this.getAttribute('values').split(',').map((s) => s.trim());
    return keep.map((v) => all.find(([value]) => String(value) === v)).filter(Boolean);
  }

  /** The display name of a value, offered or not. */
  optionText(value) {
    const hit = this.allOptions().find(([v]) => v === value);
    return hit ? hit[1] : formatValue(this.meta, value);
  }
}

/** A lit button for a bool address (the LED is the lit text). */
class PxToggle extends PxControl {
  build() {
    this.button = el('button', `btn ${this.getAttribute('variant') || ''}`.trim(), this.getAttribute('label') || '');
    this.button.type = 'button';
    this.badge = el('span', 'cc', '');
    this.append(this.button, el('i', 'cc-led'), this.badge);
  }

  listen() {
    super.listen();
    this.button.addEventListener('click', () => { if (this.enabled) this.commit(!this.value); });
  }

  attributeChangedCallback(name) {
    if (name === 'label' && this.button) this.button.textContent = this.getAttribute('label') || '';
    super.attributeChangedCallback(name === 'label' ? 'x' : name);
  }

  render(value) {
    this.button.classList.toggle('on', !!value);
    this.button.disabled = !this.address;
  }
}

/**
 * Segmented buttons, one per option (up to 5). `values` offers a subset of
 * the enum labels; a value outside it (from a preset) shows lit beside them.
 */
class PxSwitch extends PxStepped {
  build() {
    this.opts = el('div', 'opts');
    this.append(this.opts);
    this.buildFoot();
    this.val.remove(); // the lit option is the value
  }

  attach(ctx, force) {
    super.attach(ctx, force);
    const buttons = this.options().map(([v, text]) => {
      const b = el('button', 'btn', text);
      b.type = 'button';
      b.dataset.value = String(v);
      b.addEventListener('click', () => this.commit(v));
      return b;
    });
    this.other = el('span', 'other', '');
    this.opts.replaceChildren(...buttons, this.other);
    this.render(this.value);
  }

  render(value) {
    if (!this.opts) return;
    let lit = false;
    for (const b of this.opts.children) {
      if (b === this.other) continue;
      b.classList.toggle('on', b.dataset.value === String(value));
      lit = lit || b.dataset.value === String(value);
    }
    if (this.other) {
      const shown = !lit && value !== undefined && value !== null && !!this.address;
      this.other.style.display = shown ? '' : 'none';
      this.other.textContent = shown ? this.optionText(value) : '';
    }
  }
}

/** A list box: the value, and a menu of the options on click. */
class PxList extends PxStepped {
  build() {
    this.box = el('div', 'lbox', '');
    this.append(this.box);
    this.buildFoot();
    this.val.remove();
  }

  listen() {
    super.listen();
    this.box.addEventListener('click', (e) => {
      if (!this.enabled) return;
      e.stopPropagation();
      const r = this.box.getBoundingClientRect();
      this.ctx.openMenu(this.options().map(([v, text]) => [text, () => this.commit(v), { current: v === this.value }]),
        r.left, r.bottom + 2, { className: 'list' });
    });
  }

  render(value) {
    this.box.textContent = this.address ? this.optionText(value) : 'off';
  }
}

/**
 * The → destination button of a modulation source (bound to a
 * ch<N>.<source>.target address): shows the target's short name; a click
 * dispatches a bubbling `px-assign` event ({address}) for click to assign
 * (the edit rack arms it); the wheel steps through the targets in engine
 * order; the right-click menu assigns or clears.
 */
class PxTarget extends PxStepped {
  build() {
    this.button = el('button', 'btn dest', '');
    this.button.type = 'button';
    this.append(this.button);
    this.buildFoot();
    this.val.remove();
  }

  listen() {
    this.addEventListener('wheel', (e) => {
      if (!this.enabled) return;
      e.preventDefault();
      if (!e.deltaY) return;
      this.commit(wheelValue(this.meta, this.value, -Math.sign(e.deltaY)), { burst: true });
    }, { passive: false });
    this.button.addEventListener('click', () => this.assign());
    this.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      if (!this.enabled) return;
      this.ctx.openMenu([
        ['Assign: click a control', () => this.assign()],
        ['Clear destination', () => this.commit('none'), { current: this.value === 'none' }],
      ], e.clientX, e.clientY);
    });
  }

  assign() {
    if (!this.enabled) return;
    this.dispatchEvent(new CustomEvent('px-assign', { bubbles: true, detail: { address: this.address } }));
  }

  get text() { return targetName(this.value); }

  render(value) {
    this.button.textContent = this.address ? `→ ${targetName(value)}` : '→';
    this.button.disabled = !this.address;
    this.classList.toggle('none', !value || value === 'none');
  }
}

// ---------------------------------------------------------------- display

/**
 * The green two-line dot display. `setBase(line1, line2)` is what it shows
 * at rest; `show(line1, line2, ms)` replaces it for a while (a touched
 * control, 1.5 s); `alert(line1, line2, ms)` shows a message (errors, 4 s).
 */
class PxDisplay extends HTMLElement {
  connectedCallback() {
    if (this.lines) return;
    this.lines = [el('div', 'line'), el('div', 'line')];
    this.append(...this.lines);
    this.base = ['', ''];
    this.until = 0;
  }

  setBase(line1, line2) {
    this.connectedCallback();
    this.base = [line1, line2];
    if (!this.timer) this.paint(this.base);
  }

  show(line1, line2, ms = 1500, cls = '') {
    this.connectedCallback();
    this.paint([line1, line2], cls);
    clearTimeout(this.timer);
    this.timer = setTimeout(() => { this.timer = null; this.paint(this.base); }, ms);
  }

  alert(line1, line2, ms = 4000) { this.show(line1, line2, ms, 'alert'); }

  paint([a, b], cls = '') {
    this.lines[0].textContent = a;
    this.lines[1].textContent = b;
    this.title = cls ? `${a} ${b}` : '';
    this.classList.toggle('alert', cls === 'alert');
  }

  text() { return this.lines.map((l) => l.textContent); }
}

// ---------------------------------------------------------------- registry

const ELEMENTS = {
  'px-knob': PxKnob, 'px-fader': PxFader, 'px-toggle': PxToggle,
  'px-switch': PxSwitch, 'px-list': PxList, 'px-target': PxTarget, 'px-display': PxDisplay,
};

/** Tag names of the address-bound controls. */
export const CONTROL_TAGS = ['px-knob', 'px-fader', 'px-toggle', 'px-switch', 'px-list', 'px-target'];

/** Define the custom elements (once per page). */
export function defineControls(registry = globalThis.customElements) {
  for (const [tag, cls] of Object.entries(ELEMENTS)) if (!registry.get(tag)) registry.define(tag, cls);
}

defineControls();
