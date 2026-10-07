// The PO-32 page (map decisions #13, #20): one edit rack drawer page with a
// transfer tab and an import tab, each a numbered left-to-right stage flow.
// The current stage is lit, later ones are dimmed but work, done ones show a
// tick; stages only show progress, nothing is locked. The face stays live.
//
//   transfer: 1 choose (sounds to 1-8 / 9-16, the pattern chain whose PO-32
//             slots are sent empty, the channels sent, copied from the face
//             mutes each time the page opens) › 2 prepare the PO-32 ›
//             3 send (progress, transfer / stop, save WAV)
//   import:   1 listen (input device, monitor and level meter, record / stop,
//             import a WAV, keep recordings) › 2 bank › 3 pick patterns (up to
//             12, letters in order, a conflict swaps; preview; the focused
//             pattern's grid) › 4 import (channels 1-8 and all 12 patterns,
//             one undo step; the page stays open)
//
// The core does the work (po32.* verbs and addresses, docs/core-interface.md):
// the page reads its state, runs its verbs and shows their errors (display,
// and the alert sheet when they need reading). Registered as the 'po32' page
// (panel.openPage('po32', {tab: 'transfer' | 'import'})); it remembers the
// last tab. Closing the page stops a send, a preview and the input, as the
// tkinter dialogs do.
//
//   const po32 = mountPo32Page(panel, { store, client });
//   po32.open({ tab }), po32.hide(), po32.tab, po32.element, po32.destroy()

import { fileName, percent } from './files.js';
import {
  holdPeak, importButtonText, importStages, importText, letterChoices, levelText, levelTone, pick,
  patternButton, PATTERN_BUTTONS, slotsNote, sourceText, transferStages, transferText, unmutedChannels,
} from './po32-logic.js';

const CHANNELS = [1, 2, 3, 4, 5, 6, 7, 8];
const TABS = ['transfer', 'import'];

/** What the page reads; watched from the start, so the store has them at boot. */
export const PO32_ADDRESSES = [
  'po32.chain_options', 'po32.transfer', 'po32.transfer_seconds', 'po32.listening', 'po32.recording',
  'po32.input', 'po32.decode', 'po32.decoded', 'po32.banks', 'po32.bank', 'po32.sounds', 'po32.patterns',
  'po32.picks', 'po32.focus', 'po32.grid', 'po32.previewing', 'po32.imported', 'po32.error',
  'po32.recordings_folder', 'audio.input_devices', 'audio.default_input', 'pref.audio.input_device',
];
// Read again on every opening (the core does not report them changing)
const VOLATILE = ['po32.chain_options', 'audio.input_devices', 'audio.default_input', 'pref.audio.input_device'];

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function button(cls, text, attrs = {}) {
  const b = el('button', `btn sq ${cls}`.trim(), text);
  b.type = 'button';
  for (const [k, v] of Object.entries(attrs)) b.setAttribute(k, v);
  return b;
}

/** A numbered stage: {section, body, set(state)}. */
function stage(n, title, flex) {
  const section = el('section', 'po-stage');
  section.dataset.stage = String(n);
  section.style.flex = `${flex} 1 0`;
  const head = el('h4');
  const num = el('span', 'po-num', String(n));
  head.append(num, el('span', 'po-title', title));
  const body = el('div', 'po-body');
  section.append(head, body);
  return {
    section,
    body,
    set(state) {
      section.dataset.state = state;
      num.textContent = state === 'done' ? '✓' : String(n);
    },
  };
}

const arrow = () => el('div', 'po-arrow', '›');

export function mountPo32Page(panel, { store, client }) {
  const { ctx, display, drawer, sheets, files } = panel;
  const offs = [];
  const value = (address) => store.value(address);
  let tab = 'transfer';
  let shown = false;

  // ------------------------------------------------------------ errors
  /** Run a verb; an error shows on the display and, when it needs reading, on an alert. */
  async function run(verb, args, { label = 'PO-32', failTitle = null } = {}) {
    const event = await client.act(verb, args);
    if (event.status === 'error') {
      display.alert(label, event.error);
      if (failTitle) sheets.alert({ title: failTitle, text: event.error, tone: 'error' });
      return null;
    }
    return event.status === 'done' ? (event.result ?? {}) : null;
  }

  // ------------------------------------------------------------ the page frame
  const root = el('div', 'po32');
  root.id = 'po32-page';
  const head = el('div', 'po-head');
  const back = button('', '◀ ch 1 edit', { id: 'po32-back' });
  const tabBar = el('div', 'po-tabs');
  const tabButtons = TABS.map((name) => {
    const b = button('', name, { 'data-tab': name });
    b.addEventListener('click', () => showTab(name));
    return b;
  });
  tabBar.append(...tabButtons);
  const close = button('', '✕', { id: 'po32-close', title: 'close the PO-32 page' });
  head.append(back, el('span', 'hl po-name', 'PO-32'), tabBar, el('span', 'sp'), close);
  back.addEventListener('click', () => hide());
  close.addEventListener('click', () => hide());
  const flows = { transfer: el('div', 'po-flow'), import: el('div', 'po-flow') };
  flows.transfer.dataset.flow = 'transfer';
  flows.import.dataset.flow = 'import';
  root.append(head, flows.transfer, flows.import);

  // ============================================================ transfer
  const choose = stage(1, 'choose', 1.55);
  const prepare = stage(2, 'prepare the PO-32', 1.05);
  const send = stage(3, 'send', 1.25);
  flows.transfer.append(choose.section, arrow(), prepare.section, arrow(), send.section);

  // 1 choose: sounds to, the chain, the channels
  const settings = { bank: 0, chain: null, channels: new Set(CHANNELS), slots: null };
  const soundsTo = el('div', 'po-seg');
  const bankButtons = [['1–8', 0], ['9–16', 1]].map(([text, bank]) => {
    const b = button('', text, { 'data-send-bank': String(bank) });
    b.addEventListener('click', () => { settings.bank = bank; renderTransfer(); prepareSignal(); });
    return b;
  });
  soundsTo.append(...bankButtons);
  const chainBox = el('div', 'lbox po-chain');
  chainBox.id = 'po32-chain';
  chainBox.dataset.address = 'po32.chain_options';
  const slotsLine = el('div', 'po-note po-slots');
  slotsLine.id = 'po32-slots';
  const chRows = el('div', 'po-channels');
  const chChecks = CHANNELS.map((n) => {
    const row = el('label', 'po-check');
    row.dataset.channel = String(n);
    row.style.setProperty('--cc', `var(--ch${n})`);
    const box = el('i');
    const num = el('b', '', String(n));
    const name = el('span', 'nm', '');
    row.append(box, num, name);
    row.addEventListener('click', (e) => {
      e.preventDefault();
      if (settings.channels.has(n)) settings.channels.delete(n); else settings.channels.add(n);
      renderTransfer();
      prepareSignal();
    });
    return { row, name };
  });
  chRows.append(...chChecks.map((c) => c.row));
  const field = (label, ...nodes) => {
    const row = el('div', 'po-field');
    row.append(el('span', 'po-lbl', label), ...nodes);
    return row;
  };
  choose.body.append(field('sounds to', soundsTo), field('pattern chain', chainBox), slotsLine, chRows);
  chainBox.addEventListener('click', (e) => {
    e.stopPropagation();
    const r = chainBox.getBoundingClientRect();
    const options = value('po32.chain_options') || [];
    ctx.openMenu(options.map((label) => [label, () => {
      settings.chain = label;
      renderTransfer();
      prepareSignal();
    }, { current: label === chainLabel() }]), r.left, r.bottom + 2, { className: 'list' });
  });
  const chainLabel = () => {
    const options = value('po32.chain_options') || [];
    return options.includes(settings.chain) ? settings.chain : (options[0] ?? null);
  };
  const transferArgs = () => ({ bank: settings.bank, chain: chainLabel(),
    channels: CHANNELS.filter((n) => settings.channels.has(n)) });

  // 2 prepare the PO-32
  const steps = el('div', 'po-howto');
  steps.innerHTML = 'Turn its volume up.<br>Hold <b>write</b> and press <b>sound</b>.<br>It waits for the signal.';
  prepare.body.append(steps);

  // 3 send
  const bar = el('div', 'po-bar');
  bar.id = 'po32-progress';
  bar.dataset.address = 'po32.progress';
  const fill = el('i');
  bar.append(fill);
  const status = el('div', 'po-status');
  status.dataset.address = 'po32.transfer';
  const statusText = el('span', '', '');
  statusText.id = 'po32-status';
  statusText.dataset.address = 'po32.transfer_seconds';
  status.append(statusText);
  const sendButton = button('on po-send', '▶ transfer', { id: 'po32-send', 'data-verb': 'po32.send' });
  const saveButton = button('', 'save wav…', { id: 'po32-save-wav', 'data-verb': 'po32.save_wav' });
  const sendRow = el('div', 'po-row');
  sendRow.append(sendButton, saveButton);
  const savedLine = el('div', 'po-note po-saved', '');
  savedLine.id = 'po32-saved';
  send.body.append(el('div', 'po-note', 'On the PO-32: hold write and press sound to receive, then send.'),
    bar, status, sendRow, savedLine);

  let preparing = 0;
  async function prepareSignal() {
    if (value('po32.transfer') === 'sending') return;
    const mine = ++preparing;
    const event = await client.act('po32.prepare', transferArgs());
    if (mine !== preparing) return; // a newer choice
    if (event.status === 'done') {
      settings.slots = (event.result && event.result.slots) || [];
      renderTransfer();
    } else if (event.status === 'error') {
      display.alert('PO-32', event.error);
    }
  }

  async function sendOrStop() {
    if (value('po32.transfer') === 'sending') {
      await run('po32.cancel', {});
      return;
    }
    store.assume('po32.transfer', 'sending');
    renderTransfer();
    const event = await client.act('po32.send', transferArgs(), {
      onProgress: (fraction) => display.show('PO-32 SEND', percent(fraction), 3000),
    });
    if (event.status === 'done') display.show('PO-32 SEND', 'sent');
    else if (event.status === 'cancelled') display.show('PO-32 SEND', 'stopped');
    else if (event.status === 'error') {
      display.alert('PO-32 SEND', event.error);
      sheets.alert({ title: 'Could not send to the PO-32', text: event.error, tone: 'error' });
    }
  }
  sendButton.addEventListener('click', () => { sendOrStop(); });

  async function saveWav() {
    const preset = String(value('preset.name') || 'Untitled');
    const name = `${preset} (PO-32 transfer).wav`.replace(/[<>:"/\\|?*]/g, '');
    const path = await client.saveFile({ title: 'Save the PO-32 transfer', filters: ['WAV audio (*.wav)'],
      name, suffix: 'wav' });
    if (!path) return null;
    const result = await files.save('po32.save_wav', { path, ...transferArgs() }, { label: 'PO-32 WAV',
      failTitle: 'Could not save the PO-32 transfer', done: (r) => `saved ${fileName(r.path)}` });
    if (result && result.saved) savedLine.textContent = `saved ${fileName(result.path)}`;
    return result;
  }
  saveButton.addEventListener('click', () => { saveWav(); });

  function renderTransfer() {
    const transfer = value('po32.transfer') || 'none';
    const states = transferStages(transfer);
    [choose, prepare, send].forEach((s, i) => s.set(states[i]));
    bankButtons.forEach((b, i) => b.classList.toggle('on', i === settings.bank));
    chainBox.textContent = chainLabel() || '—';
    slotsLine.textContent = slotsNote(settings.slots);
    chChecks.forEach(({ row, name }, i) => {
      const n = i + 1;
      row.classList.toggle('on', settings.channels.has(n));
      name.textContent = String(value(`ch${n}.name`) || `Drum ${n}`);
    });
    const sending = transfer === 'sending';
    sendButton.textContent = sending ? '■ stop' : '▶ transfer';
    sendButton.classList.toggle('on', !sending);
    sendButton.classList.toggle('stop', sending);
    saveButton.disabled = sending;
    statusText.textContent = transferText({ transfer, seconds: value('po32.transfer_seconds'),
      progress: (store.readout('po32') || {}).progress, error: value('po32.error') });
    status.dataset.state = transfer;
    renderProgress();
  }

  function renderProgress() {
    const readout = store.readout('po32') || {};
    const transfer = value('po32.transfer');
    const progress = transfer === 'sent' ? 1 : (transfer === 'sending' ? Number(readout.progress) || 0 : 0);
    fill.style.width = `${Math.round(progress * 1000) / 10}%`;
    if (transfer === 'sending') statusText.textContent = transferText({ transfer, progress });
  }

  // ============================================================ import
  const listen = stage(1, 'listen', 1.2);
  const bankStage = stage(2, 'bank', 0.95);
  const pickStage = stage(3, 'pick patterns', 1.75);
  const importStage = stage(4, 'import', 0.9);
  flows.import.append(listen.section, arrow(), bankStage.section, arrow(), pickStage.section, arrow(),
    importStage.section);

  // 1 listen
  let device = null; // the input chosen here (null: the saved one, else the default)
  const deviceBox = el('div', 'lbox po-device');
  deviceBox.id = 'po32-input';
  deviceBox.dataset.address = 'po32.input';
  const rescan = button('', '↻', { id: 'po32-rescan', 'data-verb': 'audio.rescan', title: 'rescan the input devices' });
  const inputRow = el('div', 'po-row');
  inputRow.append(deviceBox, rescan);
  const monitor = button('', 'monitor', { id: 'po32-monitor', 'data-verb': 'po32.listen',
    'data-address': 'po32.listening' });
  const meter = el('div', 'po-meter');
  meter.id = 'po32-meter';
  meter.dataset.address = 'po32.level';
  const meterFill = el('i', 'lvl');
  const meterPeak = el('b', 'pk');
  meter.append(meterFill, meterPeak);
  const meterDb = el('span', 'po-db', '-∞ dB');
  const meterRow = el('div', 'po-row');
  meterRow.append(monitor, meter, meterDb);
  const record = button('red', '● record', { id: 'po32-record', 'data-verb': 'po32.record',
    'data-address': 'po32.recording' });
  const openWav = button('', 'import wav…', { id: 'po32-open-wav', 'data-verb': 'po32.decode' });
  const recordRow = el('div', 'po-row');
  recordRow.append(record, openWav);
  const source = el('div', 'po-note po-source');
  source.dataset.address = 'po32.decode';
  const sourceTextEl = el('span', '', '');
  sourceTextEl.id = 'po32-source';
  sourceTextEl.dataset.address = 'po32.recorded_seconds';
  source.append(sourceTextEl);
  const keep = document.createElement('px-toggle');
  keep.setAttribute('label', 'keep recordings');
  keep.dataset.address = 'pref.po32.save_recordings';
  keep.id = 'po32-keep';
  const keepNote = el('span', 'po-note po-folder', '');
  keepNote.dataset.address = 'po32.recordings_folder';
  const keepRow = el('div', 'po-row po-keep');
  keepRow.append(keep, keepNote);
  const decodedWrap = el('div', 'po-decoded');
  decodedWrap.dataset.address = 'po32.decoded';
  decodedWrap.append(source);
  listen.body.append(field('input', inputRow), meterRow, recordRow, decodedWrap, keepRow);

  const inputDevices = () => value('audio.input_devices') || [];
  const chosenDevice = () => {
    const names = inputDevices();
    for (const name of [device, value('pref.audio.input_device'), value('audio.default_input')]) {
      if (name && names.includes(name)) return name;
    }
    return names[0] || null;
  };
  deviceBox.addEventListener('click', (e) => {
    e.stopPropagation();
    const r = deviceBox.getBoundingClientRect();
    const names = inputDevices();
    const current = chosenDevice();
    const items = names.length ? names.map((name) => [name, () => chooseDevice(name), { current: name === current }])
      : [['(no input devices)', null]];
    ctx.openMenu(items, r.left, r.bottom + 2, { className: 'list' });
  });
  function chooseDevice(name) {
    device = name;
    renderImport();
    if (value('po32.listening') && !value('po32.recording')) {
      run('po32.listen', { on: true, device: name }, { failTitle: 'Could not open the input' });
    }
  }
  rescan.addEventListener('click', async () => {
    const result = await run('audio.rescan', {});
    if (result) {
      store.seed(await client.get(['audio.input_devices', 'audio.default_input']));
      display.show('INPUT DEVICES', `${(result.input_devices || []).length} found`);
      renderImport();
    }
  });
  monitor.addEventListener('click', () => {
    const on = !(value('po32.listening') && !value('po32.recording'));
    run('po32.listen', { on, device: chosenDevice() }, { failTitle: 'Could not open the input' });
  });
  record.addEventListener('click', async () => {
    if (value('po32.recording')) {
      store.assume('po32.decode', 'decoding');
      renderImport();
      const result = await run('po32.stop', {}, { label: 'PO-32 DECODE', failTitle: 'Could not decode the PO-32 signal' });
      if (result && result.decoded === false) display.show('PO-32 IMPORT', 'nothing recorded');
      else if (result) display.show('PO-32 IMPORT', `decoded ${result.drums} + ${result.patterns}`);
    } else {
      const result = await run('po32.record', { device: chosenDevice() }, { failTitle: 'Could not record the input' });
      if (result) display.show('PO-32 IMPORT', 'recording…');
    }
  });
  openWav.addEventListener('click', async () => {
    const path = await client.openFile({ title: 'Import a PO-32 recording', filters: ['WAV audio (*.wav)'] });
    if (!path) return;
    const result = await files.run('po32.decode', { path }, { label: 'PO-32 DECODE',
      failTitle: 'Could not decode the PO-32 signal' });
    if (result) display.show('PO-32 IMPORT', `decoded ${result.drums} + ${result.patterns}`);
  });

  // 2 bank
  const banksWrap = el('div', 'po-seg po-banks');
  banksWrap.dataset.address = 'po32.banks';
  const bankInner = el('div', 'po-seg-in');
  bankInner.dataset.address = 'po32.bank';
  const decodedBanks = [['bank 0 (1–8)', 0], ['bank 1 (9–16)', 1]].map(([text, bank]) => {
    const b = button('', text, { 'data-bank': String(bank), 'data-verb': 'po32.select_bank' });
    b.addEventListener('click', () => {
      store.assume('po32.bank', bank);
      renderImport();
      run('po32.select_bank', { bank });
    });
    return b;
  });
  bankInner.append(...decodedBanks);
  banksWrap.append(bankInner);
  const soundList = el('ol', 'po-sounds');
  soundList.dataset.address = 'po32.sounds';
  const soundItems = CHANNELS.map(() => soundList.appendChild(el('li', '', '—')));
  bankStage.body.append(banksWrap, soundList);

  // 3 pick patterns
  const patGrid = el('div', 'po-pats');
  patGrid.dataset.address = 'po32.patterns';
  const patButtons = Array.from({ length: PATTERN_BUTTONS }, (_, i) => {
    const number = i + 1;
    const b = button('po-pat', String(number), { 'data-pattern': String(number), 'data-verb': 'po32.pick' });
    b.addEventListener('click', () => togglePick(number));
    b.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      e.stopPropagation();
      patternMenu(number, e.clientX, e.clientY);
    });
    return b;
  });
  patGrid.append(...patButtons);
  const first = button('', 'first 12', { id: 'po32-first', 'data-verb': 'po32.pick_first' });
  const clear = button('', 'clear', { id: 'po32-clear', 'data-verb': 'po32.pick_clear' });
  const count = el('span', 'po-note po-count', '0 / 12 picked');
  count.id = 'po32-count';
  count.dataset.address = 'po32.picks';
  const preview = button('', '▶ preview', { id: 'po32-preview', 'data-verb': 'po32.preview',
    'data-address': 'po32.previewing' });
  const pickRow = el('div', 'po-row');
  pickRow.append(first, clear, count, el('span', 'sp'), preview);
  const focusLine = el('div', 'po-note po-focus', '');
  focusLine.id = 'po32-focus';
  focusLine.dataset.address = 'po32.focus';
  const gridWrap = el('div', 'po-gridwrap');
  gridWrap.dataset.address = 'po32.preview_step';
  const grid = el('div', 'po-grid');
  grid.id = 'po32-grid';
  grid.dataset.address = 'po32.grid';
  const gridLabels = [];
  const gridCells = CHANNELS.map((n) => {
    const label = el('span', 'po-glbl', `D${n}`);
    gridLabels.push(label);
    grid.append(label);
    return Array.from({ length: 16 }, (_, s) => {
      const cell = el('i', `g${Math.floor(s / 4) + 1}`);
      cell.style.setProperty('--cc', `var(--ch${n})`);
      cell.dataset.step = String(s + 1);
      grid.append(cell);
      return cell;
    });
  });
  gridWrap.append(grid);
  pickStage.body.append(patGrid, pickRow, focusLine, gridWrap);

  function togglePick(number) {
    const patterns = value('po32.patterns') || [];
    if (number > patterns.length) return;
    const { picks, refused } = pick(value('po32.picks') || [], number);
    if (refused) {
      display.show('PO-32 IMPORT', '12 patterns picked already');
      store.assume('po32.focus', number);
      renderImport();
      run('po32.focus', { pattern: number });
      return;
    }
    store.assume('po32.picks', picks);
    store.assume('po32.focus', number);
    renderImport();
    run('po32.pick', { pattern: number });
  }

  function patternMenu(number, x, y) {
    const patterns = value('po32.patterns') || [];
    if (number > patterns.length) return;
    const picks = value('po32.picks') || [];
    const picked = picks.some((p) => p.pattern === number);
    const items = [[`PO-32 pattern ${number}`, null], ['show its triggers', () => {
      store.assume('po32.focus', number);
      renderImport();
      run('po32.focus', { pattern: number });
    }], null];
    for (const { letter, holder, current } of letterChoices(picks, number)) {
      const text = `→ ${letter}${holder ? `  (swaps with ${holder})` : ''}`;
      items.push([text, () => {
        const next = pick(picks, number, { letter });
        if (next.refused) { display.show('PO-32 IMPORT', '12 patterns picked already'); return; }
        store.assume('po32.picks', next.picks);
        store.assume('po32.focus', number);
        renderImport();
        run('po32.pick', { pattern: number, letter });
      }, { current }]);
    }
    if (picked) {
      items.push(null, ['unpick', () => {
        store.assume('po32.picks', pick(picks, number, { picked: false }).picks);
        renderImport();
        run('po32.pick', { pattern: number, picked: false });
      }]);
    }
    ctx.openMenu(items, x, y, { className: 'po-letters' });
  }

  first.addEventListener('click', () => run('po32.pick_first', {}));
  clear.addEventListener('click', () => {
    store.assume('po32.picks', []);
    renderImport();
    run('po32.pick_clear', {});
  });
  preview.addEventListener('click', async () => {
    const on = !value('po32.previewing');
    const result = await run('po32.preview', { on }, { label: 'PO-32 PREVIEW', failTitle: on ? 'Could not preview' : null });
    if (result && on) display.show('PO-32 PREVIEW', `pattern ${result.pattern}`);
  });

  // 4 import
  const summary = el('div', 'po-note po-summary', '');
  summary.id = 'po32-summary';
  const importButton = button('on', 'import', { id: 'po32-import', 'data-verb': 'po32.import',
    'data-address': 'po32.imported' });
  const importError = el('div', 'po-error');
  importError.dataset.address = 'po32.error';
  importError.id = 'po32-error';
  importStage.body.append(summary, importButton, importError);
  importButton.addEventListener('click', async () => {
    const result = await run('po32.import', {}, { label: 'PO-32 IMPORT', failTitle: 'Could not import from the PO-32' });
    if (result) {
      const n = result.patterns.length;
      display.show('PO-32 IMPORT', `${result.drums} sounds, ${n} pattern${n === 1 ? '' : 's'}`, 3000);
    }
  });

  let peak = null;
  function renderMeter() {
    const readout = store.readout('po32') || {};
    const listening = !!value('po32.listening');
    const level = listening ? Math.max(0, Math.min(1, Number(readout.level) || 0)) : 0;
    peak = listening ? holdPeak(peak, level) : null;
    meterFill.style.width = `${Math.round(level * 1000) / 10}%`;
    meterPeak.style.left = `${Math.round((peak ? peak.peak : 0) * 1000) / 10}%`;
    meterPeak.hidden = !peak || peak.peak < 0.01;
    meter.dataset.tone = levelTone(level);
    meterDb.textContent = listening ? levelText(level) : '-∞ dB';
    meterDb.dataset.tone = meter.dataset.tone;
    if (value('po32.recording')) {
      sourceTextEl.textContent = sourceText({ recording: true, seconds: readout.recorded_seconds });
    }
    const step = Number.isInteger(readout.preview_step) && value('po32.previewing') ? readout.preview_step : -1;
    gridCells.forEach((row) => row.forEach((cell, s) => cell.classList.toggle('ph', s === step)));
  }

  function renderImport() {
    const decoded = value('po32.decoded') || null;
    const picks = value('po32.picks') || [];
    const patterns = value('po32.patterns') || [];
    const focus = value('po32.focus') || 0;
    const recording = !!value('po32.recording');
    const listening = !!value('po32.listening');
    const states = importStages({ decoded: !!decoded, picks, imported: !!value('po32.imported') });
    [listen, bankStage, pickStage, importStage].forEach((s, i) => s.set(states[i]));

    // listen
    deviceBox.textContent = (listening && value('po32.input')) || chosenDevice() || '(no input devices)';
    monitor.classList.toggle('on', listening && !recording);
    record.classList.toggle('on', recording);
    record.textContent = recording ? '■ stop' : '● record';
    sourceTextEl.textContent = sourceText({ decode: value('po32.decode'), decoded, recording, listening,
      seconds: (store.readout('po32') || {}).recorded_seconds, error: value('po32.error') });
    source.dataset.state = value('po32.decode') || 'none';
    keepNote.textContent = value('pref.po32.save_recordings') ? `in ${value('po32.recordings_folder') || ''}` : '';

    // bank
    const banks = value('po32.banks') || [false, false];
    const bank = value('po32.bank') || 0;
    decodedBanks.forEach((b, i) => {
      b.disabled = !decoded || !banks[i];
      b.classList.toggle('on', !!decoded && i === bank);
    });
    const sounds = value('po32.sounds') || [];
    soundItems.forEach((li, i) => {
      const s = sounds[i];
      li.textContent = decoded ? (s ? String(s).split(',')[0] : '(empty)') : '—';
      li.title = s || '';
      li.classList.toggle('none', !s);
    });

    // pick patterns
    patButtons.forEach((b, i) => {
      const view = patternButton(i + 1, { patterns, picks, focus });
      b.textContent = view.text;
      b.disabled = !view.usable;
      b.classList.toggle('on', view.picked);
      b.classList.toggle('focus', view.focused);
      b.classList.toggle('empty', view.empty);
    });
    count.textContent = `${picks.length} / 12 picked`;
    first.disabled = !decoded;
    clear.disabled = !picks.length;
    const previewing = !!value('po32.previewing');
    preview.disabled = !decoded || !focus;
    preview.classList.toggle('on', previewing);
    preview.textContent = previewing ? '■ stop' : '▶ preview';
    const focused = focus ? patterns[focus - 1] : null;
    const letter = (picks.find((p) => p.pattern === focus) || {}).letter;
    focusLine.textContent = focused ? `pattern ${focus}${letter ? ` → ${letter}` : ''} · ${focused.summary}`
      : 'right-click a pattern for its letter';
    const g = value('po32.grid') || [];
    gridCells.forEach((row, d) => row.forEach((cell, s) => cell.classList.toggle('on', !!(g[d] && g[d][s]))));
    gridLabels.forEach((label, d) => {
      const s = sounds[d];
      label.textContent = `D${d + 1}${decoded && s ? ` ${String(s).split(/[ ,]/)[0]}` : ''}`;
    });

    // import
    summary.textContent = importText({ decoded, bank, picks, imported: !!value('po32.imported') });
    importButton.textContent = importButtonText(picks);
    importButton.disabled = !decoded;
    const error = value('po32.error');
    importError.textContent = error && value('po32.decode') !== 'error' ? error : '';
    renderMeter();
  }

  // ------------------------------------------------------------ tabs, open, close
  function renderHead() {
    back.textContent = `◀ ch ${value('global.channel') || 1} edit`;
    tabButtons.forEach((b) => b.classList.toggle('on', b.dataset.tab === tab));
    root.dataset.tab = tab;
  }

  function render() {
    if (!shown) return;
    renderHead();
    renderTransfer();
    renderImport();
  }

  function showTab(name) {
    if (!TABS.includes(name)) return;
    const was = tab;
    tab = name;
    renderHead();
    if (shown && name === 'transfer' && (was !== 'transfer' || settings.slots === null)) prepareSignal();
  }

  async function open({ tab: wanted = null } = {}) {
    if (shown && !wanted) { hide(); return null; } // the PO-32 button again
    if (wanted) tab = TABS.includes(wanted) ? wanted : tab;
    if (shown) { showTab(tab); return root; }
    // The channels sent start from the face mutes each time the page opens
    settings.channels = new Set(unmutedChannels(CHANNELS.map((n) => !!value(`ch${n}.mute`))));
    settings.slots = null;
    savedLine.textContent = '';
    shown = true;
    drawer.show('po32', root, { onHide });
    render();
    store.seed(await client.get(VOLATILE));
    render();
    if (tab === 'transfer') prepareSignal();
    return root;
  }

  function onHide() {
    shown = false;
    ctx.closeMenus();
    if (value('po32.transfer') === 'sending') client.act('po32.cancel', {});
    if (value('po32.previewing')) client.act('po32.preview', { on: false });
    if (value('po32.listening') || value('po32.recording')) client.act('po32.listen', { on: false });
  }

  function hide() { return drawer.hide('po32'); }

  // The PO-32 button is lit while the page shows
  const po32Button = panel.slot('po32');
  offs.push(drawer.onChange((name) => { if (po32Button) po32Button.classList.toggle('on', name === 'po32'); }));

  for (const address of [...PO32_ADDRESSES, 'global.channel', 'preset.name', 'pref.po32.save_recordings',
    ...CHANNELS.flatMap((n) => [`ch${n}.name`, `ch${n}.mute`])]) {
    offs.push(store.watch(address, render, { now: false }));
  }
  offs.push(store.watchReadout('po32', () => {
    if (!shown) return;
    renderProgress();
    renderMeter();
  }, { now: false }));
  offs.push(panel.registerPage('po32', open));

  return {
    open,
    hide,
    get tab() { return tab; },
    get shown() { return shown; },
    element: root,
    /** The transfer settings (sounds to, chain, channels). */
    get settings() { return { ...transferArgs() }; },
    destroy() {
      if (shown) hide();
      offs.forEach((off) => off());
    },
  };
}
