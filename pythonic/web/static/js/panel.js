// The hardware face (map decisions #7, #12, #16): left column (undo / redo,
// programs), the top row (master, step rate, fill rate, strip ctrl) over the
// eight channel strips, the right column (display, MUTE latch, morph learn,
// tempo, MIDI LED, sound morph) and START/STOP. Everything is address-driven:
// controls are px-* elements (controls.js) bound by data-address, buttons
// that start verbs carry data-verb.
//
// Extension slots for later slices (elements with data-slot):
//   step-entry  left column: step-mode buttons, last step, all ch, follow, matrix (steps.js)
//   patterns    left column: patterns A-L, chain, menu, copy, paste (patterns.js)
//   preset-prev, preset-next, preset   the PRESET buttons and menu (presets.js)
//   po32, setup           open the PO-32 page and the setup sheet (panel.openPage)
//   rack-toggle           the edit rack button (rack.js)
//   steps       bottom row: page bars, step numbers and the 16 pads (steps.js)
//   rack        the edit rack drawer under the face: panel.drawer (drawer.js) shows
//               one page at a time (the ⊞ matrix, the edit rack as its base page,
//               rack.js, which also owns the rack-toggle button and the open state)
// A slice fills a slot (replaceChildren) with px-* controls; they bind to the
// panel's control context on insertion (panel.ctx).
//
// Pages: panel.registerPage(name, open) registers what opens a secondary
// feature (open(options): show a drawer page or a sheet); panel.openPage(name,
// options) opens it from the PO-32 and SETUP buttons, the PRESET menu, the
// MIDI LED and a control's right-click ▸ CC mappings. Names: 'po32' ({tab:
// 'transfer' | 'import'}), 'ai', 'setup' ({tab: 'audio' | 'midi' | 'synthesis'
// | 'ai'}). An unregistered page says so on the alert sheet.
// Sheets: panel.sheets (sheet.js) shows overlay sheets and the alert sheet;
// error events nobody waits for go to the display and, when they need
// reading, to the alert sheet (reportError).

import { guessDrumType } from './drum-type.js';
import { createControlContext, provideContext } from './controls.js';
import { createDrawer } from './drawer.js';
import { mountExports } from './exports.js';
import { createFileFlows } from './files.js';
import { mountPatterns } from './patterns.js';
import { mountRack } from './rack.js';
import { mountPresets } from './presets.js';
import { createSheets } from './sheet.js';
import { pageRange } from './steps-logic.js';
import { mountStepRow, selectedPattern } from './steps.js';
import { formatValue, undoText } from './values.js';

const CHANNELS = [1, 2, 3, 4, 5, 6, 7, 8];
export const CTRL_PREF = 'pref.ui.ctrl_knob';

/** Strip CTRL modes: [mode, label in the selector, address suffix]. */
export const CTRL_MODES = [
  ['off', 'off', null],
  ['pan', 'pan', 'mix.pan'],
  ['reverb_mix', 'reverb mix', 'fx.reverb_mix'],
  ['delay_mix', 'delay mix', 'fx.delay_mix'],
  ['lfo1_depth', 'lfo 1 depth', 'lfo1.depth'],
  ['user', 'user', null],
];

/** Short names of the sound parameters a user CTRL knob can pick, by section. */
export const CTRL_USER_CHOICES = [
  ['Oscillator', [['osc.freq', 'osc freq'], ['osc.mod_amount', 'pitch amt'], ['osc.mod_rate', 'pitch rate'],
    ['osc.attack', 'osc attack']]],
  ['Noise', [['noise.freq', 'noise freq'], ['noise.q', 'noise q'], ['noise.attack', 'noise atk'],
    ['noise.decay', 'noise decay']]],
  ['Mix', [['mix.osc_noise', 'osc/noise'], ['mix.pan', 'pan'], ['mix.distortion', 'distort'],
    ['eq.freq', 'eq freq'], ['eq.gain', 'eq gain']]],
  ['FX', [['fx.vintage', 'vintage'], ['fx.reverb_decay', 'rvb time'], ['fx.reverb_mix', 'rvb mix'],
    ['fx.reverb_width', 'rvb wide'], ['fx.delay_feedback', 'dly fdbk'], ['fx.delay_mix', 'dly mix']]],
  ['Velocity', [['vel.osc', 'osc vel'], ['vel.noise', 'noise vel'], ['vel.mod', 'mod vel']]],
  ['Modulation', [['lfo1.rate', 'lfo 1 rate'], ['lfo1.depth', 'lfo 1 depth'], ['lfo2.rate', 'lfo 2 rate'],
    ['lfo2.depth', 'lfo 2 depth'], ['pump.amount', 'pump amt']]],
];
const USER_NAMES = Object.fromEntries(CTRL_USER_CHOICES.flatMap(([, xs]) => xs));

/** The CTRL setup of the preference (any stored value; defaults: pan, no user picks). */
export function ctrlSetup(value) {
  const v = value && typeof value === 'object' ? value : {};
  const mode = CTRL_MODES.some(([m]) => m === v.mode) ? v.mode : 'pan';
  const user = CHANNELS.map((_, i) => (Array.isArray(v.user) && typeof v.user[i] === 'string' ? v.user[i] : null));
  return { mode, user };
}

/** The address a strip's CTRL knob shows, or null (off, or no user pick). */
export function ctrlAddress(setup, channel) {
  if (setup.mode === 'user') {
    const suffix = setup.user[channel - 1];
    return suffix ? `ch${channel}.${suffix}` : null;
  }
  const mode = CTRL_MODES.find(([m]) => m === setup.mode);
  return mode && mode[2] ? `ch${channel}.${mode[2]}` : null;
}

/** The addresses the face shows (main.js reads every registered address's metadata). */
/** Pages a later slice registers (panel.registerPage), with their titles. */
export const PAGE_TITLES = { po32: 'PO-32 transfer and import', ai: 'AI drum generator', setup: 'Setup' };

/** Error sources whose errors need reading (the alert sheet); others only show on the display. */
const DISPLAY_ONLY_SOURCES = new Set(['midi', 'po32']);

/** The alert sheet's title for an error nobody waited for. */
export function errorTitle(event) {
  const source = event.source || null;
  if (source === 'audio') return 'Audio problem';
  if (source === 'ai') return 'The AI generator stopped';
  if (source === 'core') return 'Something went wrong';
  return event.verb ? `${event.verb} failed` : 'Something went wrong';
}

export const PANEL_ADDRESSES = [
  'global.tempo', 'global.swing', 'global.step_rate', 'global.fill_rate', 'global.master',
  'global.channel', 'global.edit_all', 'morph.position', 'morph.learning', 'morph.differs',
  'program.current', 'program.occupied', 'undo.can_undo', 'undo.can_redo', 'preset.name',
  'midi.connected', 'midi.cc_map', 'midi.learning', 'midi.pitchbend_target',
  ...CHANNELS.flatMap((n) => [`ch${n}.name`, `ch${n}.mute`, `ch${n}.osc.pitch`, `ch${n}.osc.decay`,
    `ch${n}.mix.level`]),
];

const html = (strings, ...values) => strings.reduce((out, s, i) => out + s + (i < values.length ? values[i] : ''), '');

function stripHtml(n) {
  return html`
  <div class="strip" data-channel="${n}" style="--cc: var(--ch${n})">
    <div class="tab" data-address="ch${n}.name"></div>
    <px-knob data-address="ch${n}.osc.pitch" label="tune" name="ch${n} tune" size="40"></px-knob>
    <px-knob data-address="ch${n}.osc.decay" label="decay" name="ch${n} decay" size="40"></px-knob>
    <px-knob class="ctrl" label="ctrl" size="40"></px-knob>
    <div class="grow"><px-fader data-address="ch${n}.mix.level" name="ch${n} level" length="118"></px-fader></div>
    <div class="chsel" data-address="ch${n}.mute">
      <button class="btn chb" type="button" data-address="global.channel" data-value="${n}">${n}</button>
    </div>
  </div>`;
}

function faceHtml() {
  const programs = Array.from({ length: 16 }, (_, i) =>
    `<button class="btn" type="button" data-verb="program.select" data-program="${i + 1}">${i + 1}</button>`).join('');
  return html`
  <div class="panel-face">
    <div class="col left">
      <div class="wordmark">PYTHON<b>IC</b></div>
      <div class="row undo-row">
        <button class="btn rnd" type="button" id="undo" data-verb="undo" data-address="undo.can_undo">undo</button>
        <button class="btn rnd" type="button" id="redo" data-verb="redo" data-address="undo.can_redo">redo</button>
      </div>
      <div class="slot-step-entry slot-mark" data-slot="step-entry"></div>
      <div class="hl">pattern</div>
      <div class="slot-patterns slot-mark" data-slot="patterns"></div>
      <div class="hl">program</div>
      <div class="programs" id="programs" data-address="program.occupied">
        <div style="display:contents" data-address="program.current">${programs}</div>
      </div>
    </div>
    <div class="col centre">
      <div class="toprow">
        <div class="sec"><span class="hl">master</span>
          <px-knob data-address="global.master" label="level" name="master" size="40"></px-knob></div>
        <div class="sec" style="flex:1.6"><span class="hl">step rate</span>
          <px-switch data-address="global.step_rate" name="step rate"></px-switch></div>
        <div class="sec"><span class="hl">fill rate</span>
          <px-list data-address="global.fill_rate" name="fill rate"></px-list></div>
        <div class="sec" style="flex:1.2"><span class="hl">strip ctrl</span>
          <div class="lbox" id="ctrl-mode" style="--lw:110px"></div>
          <span class="hint" id="ctrl-hint"></span></div>
      </div>
      <div class="strips">${CHANNELS.map(stripHtml).join('')}</div>
    </div>
    <div class="col right">
      <px-display id="display"></px-display>
      <div class="rgrid">
        <div style="display:contents" data-address="preset.files">
        <div style="display:contents" data-address="preset.clipboard">
        <div style="display:contents" data-address="pref.preset_folder">
        <div style="display:contents" data-address="pref.recent_files">
        <button class="btn sq slot" type="button" data-slot="preset-prev" data-address="preset.path" disabled>◀</button>
        <button class="btn sq slot" type="button" data-slot="preset-next" data-address="preset.path" disabled>▶</button>
        <button class="btn sq slot w2" type="button" data-slot="preset" data-address="preset.name" disabled>preset</button>
        </div></div></div></div>
        <button class="btn sq slot w2" type="button" data-slot="rack-toggle" disabled>edit rack</button>
        <button class="btn sq" type="button" data-slot="po32" id="po32-button">po-32</button>
        <button class="btn sq" type="button" data-slot="setup" id="setup-button">setup</button>
        <button class="btn sq red w2" type="button" id="mute-latch">mute</button>
        <button class="btn sq" type="button" id="learn-a" data-verb="morph.learn" data-endpoint="a" data-address="morph.learning">learn a</button>
        <button class="btn sq" type="button" id="learn-b" data-verb="morph.learn" data-endpoint="b" data-address="morph.learning">learn b</button>
      </div>
      <div class="row tempo-row">
        <div class="seg7" id="tempo-value" data-address="global.tempo">---</div>
        <px-knob data-address="global.tempo" label="tempo" name="tempo" size="40" wheel-step="1"></px-knob>
        <px-knob data-address="global.swing" label="swing" name="swing" size="32"></px-knob>
      </div>
      <div class="row midi-row" id="midi-row" title="MIDI setup">midi <span class="mled" id="midi-led" data-address="midi.connected"></span></div>
      <div class="col morph-box" id="morph" data-address="morph.differs">
        <px-knob data-address="morph.position" label="sound morph" name="sound morph" size="116"></px-knob>
      </div>
    </div>
    <div class="transport">
      <button class="ss" id="start-stop" type="button" data-verb="transport.toggle" title="start / stop"></button>
      <span class="wtab">start / stop</span>
    </div>
    <div class="slot-steps" data-slot="steps"></div>
  </div>
  <div class="slot-rack" data-slot="rack"></div>`;
}

/** Build the face in the stage; meta maps addresses to describe() metadata. */
export function mountPanel(stage, { store, client, meta = {} }) {
  const ctx = createControlContext({ store, client, meta, root: stage });
  stage.innerHTML = faceHtml();
  provideContext(stage, ctx);
  const $ = (sel) => stage.querySelector(sel);
  const $$ = (sel) => [...stage.querySelectorAll(sel)];
  const display = $('#display');
  const offs = [];
  const watch = (address, fn) => offs.push(store.watch(address, fn));
  let muteLatch = false;

  // ------------------------------------------------------------ display
  // Pattern and page on the first line (the pads' view), the preset on the second
  let view = { letter: 'A', page: 0 };
  const baseDisplay = () => {
    display.setBase(`PATTERN ${view.letter}  ${pageRange(view.page).padStart(5)}`,
      String(store.value('preset.name') || 'PYTHONIC').toUpperCase());
  };
  const onTouch = (e) => display.show(e.detail.name, e.detail.text);
  stage.addEventListener('px-touch', onTouch);
  watch('preset.name', baseDisplay);

  /** Run a verb; its error (if any) shows on the display. */
  const act = (verb, args = {}) => client.act(verb, args).then((event) => {
    if (event.status === 'error') display.alert(verb.toUpperCase(), event.error);
    return event;
  });
  // ------------------------------------------------------------ sheets and pages
  let rack = null;
  const sheets = createSheets(stage, { onShow: () => { ctx.closeMenus(); if (rack) rack.disarm(); } });
  const files = createFileFlows({ client, display, sheets });
  const pages = new Map();
  /** Register what opens a page: open(options); returns a function that unregisters it. */
  const registerPage = (name, open) => {
    pages.set(name, open);
    return () => { if (pages.get(name) === open) pages.delete(name); };
  };
  /** Open a page (PO-32, AI, setup); an unregistered one says so on the alert sheet. */
  const openPage = (name, options = {}) => {
    const open = pages.get(name);
    if (open) return open(options);
    sheets.alert({ title: PAGE_TITLES[name] || name, text: 'Coming soon: this part of the panel is not built yet.',
      tone: 'ok' });
    return null;
  };
  ctx.openCcMappings = () => openPage('setup', { tab: 'midi' });
  // Guards a preset load waits for (the AI page settles its tried sounds):
  // fn() -> boolean | Promise<boolean>, false cancels the load
  const loadGuards = new Set();
  const guardPresetLoad = (fn) => { loadGuards.add(fn); return () => loadGuards.delete(fn); };
  const beforePresetLoad = async () => {
    for (const fn of [...loadGuards]) if (!(await fn())) return false;
    return true;
  };

  /** Errors nobody waits for: audio callback, stalled stream, MIDI, AI worker, ... */
  const reportError = (event) => {
    const source = String(event.source || 'core');
    display.alert(`${(event.source || event.verb || 'core').toUpperCase()} ERROR`, event.error);
    if (!DISPLAY_ONLY_SOURCES.has(source)) sheets.alert({ title: errorTitle(event), text: event.error, tone: 'error' });
  };
  offs.push(client.onUnclaimedError(reportError));

  // ------------------------------------------------------------ strips
  const ctrlKnobs = CHANNELS.map((n) => $(`.strip[data-channel="${n}"] .ctrl`));
  const strips = CHANNELS.map((n) => $(`.strip[data-channel="${n}"]`));
  const chButtons = CHANNELS.map((n) => $(`.strip[data-channel="${n}"] .chb`));
  const showStrips = () => {
    const selected = store.value('global.channel') || 1;
    const editAll = !!store.value('global.edit_all');
    stage.classList.toggle('edit-all', editAll);
    CHANNELS.forEach((n, i) => {
      const muted = !!store.value(`ch${n}.mute`);
      strips[i].classList.toggle('sel', n === selected);
      strips[i].classList.toggle('muted', muted);
      strips[i].classList.toggle('linked', editAll && !muted && n !== selected);
      chButtons[i].classList.toggle('on', n === selected);
      chButtons[i].classList.toggle('mut', muted);
    });
  };
  watch('global.channel', showStrips);
  watch('global.edit_all', showStrips);
  CHANNELS.forEach((n, i) => {
    watch(`ch${n}.mute`, showStrips);
    watch(`ch${n}.name`, (name) => {
      strips[i].querySelector('.tab').textContent = name || '';
      chButtons[i].textContent = guessDrumType(name) || String(n);
    });
    chButtons[i].addEventListener('click', (e) => {
      if (!muteLatch && store.value('global.channel') === n) {
        // The selected channel again: hit it (tkinter: 64, Ctrl+click 127)
        const velocity = e.ctrlKey || e.metaKey ? 127 : 64;
        client.trigger(n, velocity).catch((err) => console.warn('trigger', err));
        flash(i);
        display.show(`CH${n}`, `hit ${velocity}`);
        return;
      }
      if (muteLatch) {
        const muted = !store.value(`ch${n}.mute`);
        ctx.set(`ch${n}.mute`, muted);
        display.show(`CH${n} MUTE`, muted ? 'on' : 'off');
      } else {
        ctx.set('global.channel', n);
        display.show(`CH${n}`, String(store.value(`ch${n}.name`) || '').toUpperCase());
      }
    });
  });

  // A channel button flashes when a MIDI note (or a click) hits it, as in tkinter
  const flashTimers = [];
  function flash(i) {
    chButtons[i].classList.add('hit');
    clearTimeout(flashTimers[i]);
    flashTimers[i] = setTimeout(() => chButtons[i].classList.remove('hit'), 100);
  }
  let lastNotes = null;
  offs.push(store.watchReadout('midi', (midi) => {
    const notes = (midi && midi.notes) || [];
    if (lastNotes) notes.forEach((count, i) => { if (i < 8 && count !== lastNotes[i]) flash(i); });
    lastNotes = notes;
  }));

  // ------------------------------------------------------------ strip CTRL
  const ctrlList = $('#ctrl-mode');
  const setup = () => ctrlSetup(store.value(CTRL_PREF));
  const saveSetup = (next) => ctx.set(CTRL_PREF, next);
  const showCtrl = () => {
    const s = setup();
    const mode = CTRL_MODES.find(([m]) => m === s.mode);
    ctrlList.textContent = mode[1];
    ctrlList.dataset.mode = s.mode;
    $('#ctrl-hint').textContent = s.mode === 'user' ? 'per channel: click a ctrl label' : 'same for all 8 channels';
    ctrlKnobs.forEach((knob, i) => {
      const address = ctrlAddress(s, i + 1);
      if (address) {
        if (knob.dataset.address !== address) knob.dataset.address = address;
      } else if (knob.hasAttribute('data-address')) knob.removeAttribute('data-address');
      const label = s.mode === 'user' ? (USER_NAMES[s.user[i]] || 'pick ▾') : 'ctrl';
      knob.setAttribute('label', label);
      knob.setAttribute('name', `ch${i + 1} ${s.mode === 'user' ? label : mode[1]}`);
      knob.classList.toggle('user', s.mode === 'user');
    });
  };
  watch(CTRL_PREF, showCtrl);
  if (!store.has(CTRL_PREF)) showCtrl();
  ctrlList.addEventListener('click', (e) => {
    e.stopPropagation();
    const r = ctrlList.getBoundingClientRect();
    const s = setup();
    ctx.openMenu(CTRL_MODES.map(([m, label]) => [label, () => {
      saveSetup({ ...s, mode: m });
      display.show('STRIP CTRL', label);
    }, { current: m === s.mode }]), r.left, r.bottom + 2, { className: 'list' });
  });
  ctrlKnobs.forEach((knob, i) => {
    knob.addEventListener('pointerdown', (e) => {
      if (setup().mode !== 'user' || !e.target.closest('.lbl')) return;
      e.stopPropagation();
      e.preventDefault();
      const s = setup();
      const items = [];
      for (const [section, choices] of CTRL_USER_CHOICES) {
        items.push([section, null]);
        for (const [suffix, label] of choices) {
          items.push([`  ${label}`, () => {
            const user = [...s.user];
            user[i] = suffix;
            saveSetup({ ...s, user });
          }, { current: s.user[i] === suffix }]);
        }
      }
      ctx.openMenu(items, e.clientX, e.clientY);
    }, true);
  });

  // ------------------------------------------------------------ left column
  // The name a control bound to an address shows on the display (null: none)
  const controlName = (address) => {
    const control = [...stage.querySelectorAll('[data-address]')]
      .find((e) => e.dataset.address === address && (e.getAttribute('name') || e.getAttribute('label')));
    return control ? control.getAttribute('name') || control.getAttribute('label') : null;
  };
  const undo = $('#undo');
  const redo = $('#redo');
  watch('undo.can_undo', (v) => { undo.disabled = !v; });
  watch('undo.can_redo', (v) => { redo.disabled = !v; });
  for (const button of [undo, redo]) {
    button.addEventListener('click', () => act(button.dataset.verb).then((event) => {
      const r = event.result || {};
      display.show(button.dataset.verb.toUpperCase(), r.done ? undoText(r.label, controlName) : 'nothing to ' + button.dataset.verb);
    }));
  }
  const programButtons = $$('#programs .btn');
  const showPrograms = () => {
    const current = store.value('program.current');
    const occupied = store.value('program.occupied') || [];
    programButtons.forEach((b, i) => {
      b.classList.toggle('on', i + 1 === current);
      b.classList.toggle('empty', !occupied[i]);
    });
  };
  watch('program.current', showPrograms);
  watch('program.occupied', showPrograms);
  for (const b of programButtons) {
    b.addEventListener('click', () => {
      const program = Number(b.dataset.program);
      display.show('PROGRAM', String(program));
      act('program.select', { program });
    });
  }

  // ------------------------------------------------------------ right column
  const latch = $('#mute-latch');
  latch.addEventListener('click', () => {
    muteLatch = !muteLatch;
    latch.classList.toggle('on', muteLatch);
    stage.classList.toggle('mute-mode', muteLatch);
    display.show('MUTE', muteLatch ? 'buttons mute' : 'buttons select');
  });

  const learnButtons = [$('#learn-a'), $('#learn-b')];
  watch('morph.learning', (learning) => {
    for (const b of learnButtons) b.classList.toggle('on', learning === b.dataset.endpoint);
  });
  for (const b of learnButtons) {
    const endpoint = b.dataset.endpoint;
    b.addEventListener('click', () => {
      const stop = store.value('morph.learning') === endpoint;
      act('morph.learn', { endpoint: stop ? null : endpoint });
      display.show('MORPH LEARN', stop ? 'off' : endpoint.toUpperCase());
    });
    b.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      ctx.openMenu([[`Capture the current sounds as ${endpoint.toUpperCase()}`, () => {
        act('morph.capture', { endpoint });
        display.show('MORPH CAPTURE', endpoint.toUpperCase());
      }]], e.clientX, e.clientY);
    });
  }
  watch('morph.differs', (v) => { $('#morph').classList.toggle('same', !v); });

  const tempoValue = $('#tempo-value');
  watch('global.tempo', (bpm) => { tempoValue.textContent = String(bpm); });
  tempoValue.addEventListener('wheel', (e) => {
    e.preventDefault();
    if (!e.deltaY) return;
    const m = meta['global.tempo'] || { minimum: 1, maximum: 300 };
    const next = Math.min(m.maximum, Math.max(m.minimum, (store.value('global.tempo') ?? 120) + (e.deltaY < 0 ? 1 : -1)));
    ctx.set('global.tempo', next, { burst: true });
    display.show('TEMPO', formatValue(m, next));
  }, { passive: false });

  const led = $('#midi-led');
  let lastActivity = null;
  let ledTimer = null;
  watch('midi.connected', (v) => led.classList.toggle('connected', !!v));
  $('#midi-row').addEventListener('click', () => openPage('setup', { tab: 'midi' }));
  $('#po32-button').addEventListener('click', () => openPage('po32'));
  $('#setup-button').addEventListener('click', () => openPage('setup'));
  offs.push(store.watchReadout('midi', (midi) => {
    const activity = midi ? midi.activity : null;
    if (lastActivity !== null && activity !== lastActivity) {
      led.classList.add('on');
      clearTimeout(ledTimer);
      ledTimer = setTimeout(() => led.classList.remove('on'), 80);
    }
    lastActivity = activity;
  }));

  // ------------------------------------------------------------ transport
  const startStop = $('#start-stop');
  offs.push(store.watchReadout('transport', (t) => {
    startStop.classList.toggle('on', !!t.playing);
  }));
  startStop.addEventListener('click', () => { act(startStop.dataset.verb); });

  // ------------------------------------------------------------ steps and patterns
  const slot = (name) => stage.querySelector(`[data-slot="${name}"]`);
  const drawer = createDrawer(slot('rack'));
  const steps = mountStepRow({ store, client, ctx, display, slot, drawer,
    onView: (next) => { view = next; baseDisplay(); } });
  const patterns = mountPatterns({ store, ctx, display, act, slot, selected: () => selectedPattern(store) });
  rack = mountRack({ store, client, ctx, display, stage, slot, drawer, files });
  const presets = mountPresets({ store, client, ctx, display, act, stage, slot, files, openPage,
    patchItems: rack.patchItems, beforeLoad: beforePresetLoad });
  const exportsMenu = mountExports({ client, ctx, display, stage, patterns, files });
  baseDisplay();

  return {
    ctx,
    display,
    /** The element of an extension slot (see the module comment). */
    slot,
    /** The edit rack drawer's pages (drawer.js): the matrix, later the rack pages. */
    drawer,
    /** The step row and step entry (steps.js): state, setMode, setPage, render. */
    steps,
    /** The pattern buttons and menu (patterns.js): addMenuItems, openMenu. */
    patterns,
    /** The edit rack (rack.js): open, setOpen, arm, disarm, assigning. */
    rack,
    /** Run a verb; its error shows on the display. */
    act,
    /** Overlay sheets and the alert sheet (sheet.js): show, hide, alert, ask. */
    sheets,
    /** File verbs with dialogs, progress and the overwrite question (files.js): run, save. */
    files,
    /** The PRESET buttons and menu (presets.js). */
    presets,
    /** The pattern menu's exports (exports.js): exportMidi, exportAudio, openTailPopover. */
    exports: exportsMenu,
    registerPage,
    openPage,
    /** Ask before a preset load: fn() -> boolean | Promise (false cancels); returns off. */
    guardPresetLoad,
    /** Registered page names. */
    get pages() { return [...pages.keys()]; },
    /** Show an error nobody waited for (the global handler). */
    reportError,
    destroy() {
      sheets.destroy();
      exportsMenu.destroy();
      presets.destroy();
      steps.destroy();
      patterns.destroy();
      rack.destroy();
      offs.forEach((off) => off());
      ctx.destroy();
      stage.removeEventListener('px-touch', onTouch);
      stage.replaceChildren();
    },
  };
}
