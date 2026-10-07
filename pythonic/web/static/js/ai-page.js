// The AI drum generator page (map decisions #13, #21, #22): a page of the
// edit rack drawer that tries candidates on the live face.
//
//   left (under the left column)   patch and pattern models (load…), patch
//                                  temperature, candidates, seed, generate all 8
//   middle, one lane per strip     drum type, gen, ‹ i/n ›, the candidate name,
//                                  try / trying ✓ (or, without the ML extras,
//                                  the pip command with copy and install now)
//   right (under the right column) keep current / generate new patterns,
//                                  pattern temperature and the bank, ▶ loop,
//                                  ▶ bank A→L, keep tried, replace patterns,
//                                  revert all
//
// Generating a lane tries its candidate 1 on the channel (the core does);
// the strip tab of a channel trying a candidate shows its name in italics,
// knob edits there go into the tried sound, and keep tried is one undo step
// (ai.keep). Leaving the page with tried sounds asks keep or revert: ✕ and
// ◀ ch n edit ask first (cancel stays), any other way out (another drawer
// page, closing the rack) asks once the page is gone; a preset load asks
// first too (panel.guardPresetLoad). The AI settings are the pref.ai.*
// addresses the setup sheet's ai tab shows as well (#22).
//
//   const ai = mountAiPage({ panel, store, client, stage });
//   ai.open() / ai.leave() / ai.element / ai.patternMode / ai.destroy()

import {
  bankText, canReplacePatterns, CHANNELS, clampCandidates, DEFAULT_CANDIDATES, fileName,
  keptText, laneAddress, laneView, LANE_FIELDS, leaveQuestion, modelView, parseSeed, previewArgs,
  randomSeed, stateText,
} from './ai-logic.js';

const KINDS = [['patch', 'patch model', 'Load a drum patch model'], ['pattern', 'pattern model', 'Load a pattern model']];

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function button(text, { id = null, cls = '', verb = null, address = null } = {}) {
  const b = el('button', `btn sq${cls ? ` ${cls}` : ''}`, text);
  b.type = 'button';
  if (id) b.id = id;
  if (verb) b.dataset.verb = verb;
  if (address) b.dataset.address = address;
  return b;
}

function control(tag, address, attrs = {}) {
  const node = document.createElement(tag);
  node.dataset.address = address;
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
}

/** Copy text to the clipboard (a selected hidden textarea, else the async API). */
async function copyText(text) {
  const area = el('textarea');
  area.value = text;
  area.style.cssText = 'position:fixed;left:-1000px;top:0';
  document.body.append(area);
  area.select();
  let ok = false;
  try { ok = document.execCommand('copy'); } catch (_) { ok = false; }
  area.remove();
  if (ok) return true;
  try {
    await Promise.race([navigator.clipboard.writeText(text),
      new Promise((_, reject) => setTimeout(() => reject(new Error('timeout')), 1000))]);
    return true;
  } catch (_) {
    return false;
  }
}

export function mountAiPage({ panel, store, client, stage }) {
  const { display, drawer, sheets } = panel;
  const offs = [];
  const watch = (address, fn, options) => { offs.push(store.watch(address, fn, options)); };
  const value = (address) => store.value(address);
  let patternMode = 'keep';
  let candidates = DEFAULT_CANDIDATES;
  let leaving = false; // our own ✕ / ◀ asked already
  let settling = null; // the leave question in progress

  /** Run a verb; an error shows on the display, and on a red alert when it needs reading. */
  const run = async (verb, args = {}, failTitle = null) => {
    const event = await panel.act(verb, args);
    if (event.status === 'error' && failTitle) sheets.alert({ title: failTitle, text: event.error, tone: 'error' });
    return event.status === 'done' ? (event.result ?? {}) : null;
  };

  // ------------------------------------------------------------ the page
  const page = el('div', 'aipage');
  page.dataset.address = 'ai.state';
  const head = el('div', 'ai-head');
  const back = button('◀ ch 1 edit', { id: 'ai-back' });
  const stateNote = el('span', 'ai-state');
  const triedNote = el('span', 'ai-tried');
  triedNote.dataset.address = 'ai.tried';
  const close = button('✕', { id: 'ai-close' });
  head.append(back, el('span', 'ai-title', 'AI drum generator'), stateNote, el('span', 'sp'), triedNote, close);

  // left: models and generation
  const left = el('section', 'ai-sec ai-left');
  left.append(el('h4', '', 'generate'));
  const models = el('div', 'ai-models');
  models.dataset.address = 'ai.models';
  const modelLines = {};
  for (const [kind, label, title] of KINDS) {
    const line = el('div', 'ai-model');
    line.dataset.kind = kind;
    line.dataset.address = `pref.ai.${kind}_model`;
    const status = el('span', 'st');
    const load = button('load…', { cls: 'ai-load', verb: 'ai.load_model' });
    load.dataset.kind = kind;
    load.addEventListener('click', () => loadModel(kind, title));
    line.append(el('span', 'k', label), status, load);
    models.append(line);
    modelLines[kind] = status;
  }
  const candDown = button('‹', { id: 'ai-cand-down', cls: 'arr' });
  const candUp = button('›', { id: 'ai-cand-up', cls: 'arr' });
  const candValue = el('span', 'ai-cands', String(candidates));
  candValue.id = 'ai-cands';
  const candRow = el('div', 'fl');
  candRow.append(el('span', 'k', 'candidates'), candDown, candValue, candUp);
  const seed = el('input', 'ai-seed');
  seed.id = 'ai-seed';
  seed.placeholder = 'random';
  seed.spellcheck = false;
  seed.inputMode = 'numeric';
  const reseed = button('⟳', { id: 'ai-reseed', cls: 'arr' });
  reseed.title = 'new random seed';
  const seedRow = el('div', 'fl');
  seedRow.append(el('span', 'k', 'seed'), seed, reseed);
  const settings = el('div', 'ai-settings');
  const pickers = el('div', 'ai-pickers');
  pickers.append(candRow, seedRow);
  settings.append(control('px-knob', 'pref.ai.patch_temperature', { label: 'patch temp', name: 'patch temperature', size: '34' }), pickers);
  const generateAllButton = button('generate all 8', { id: 'ai-generate-all', cls: 'on', verb: 'ai.generate' });
  left.append(models, settings, generateAllButton);

  // middle: one lane under each strip, or the install box
  const middle = el('section', 'ai-sec ai-mid');
  middle.dataset.address = 'ai.available';
  const lanesBox = el('div', 'ai-lanes');
  const lanes = CHANNELS.map((n) => {
    const lane = el('div', 'ai-lane');
    lane.dataset.channel = String(n);
    lane.dataset.address = laneAddress(n, 'candidates');
    lane.style.setProperty('--cc', `var(--ch${n})`);
    const top = el('div', 'lrow');
    const type = control('px-list', laneAddress(n, 'type'), { name: `ch${n} AI drum type` });
    const gen = button('gen', { cls: 'gen', verb: 'ai.generate' });
    top.append(type, gen);
    const nav = el('div', 'lrow nav');
    nav.dataset.address = laneAddress(n, 'candidate');
    const prev = button('‹', { cls: 'arr prev', verb: 'ai.try' });
    const counter = el('span', 'cnt', '–/–');
    const next = button('›', { cls: 'arr next', verb: 'ai.try' });
    nav.append(prev, counter, next);
    const name = el('div', 'cname');
    name.dataset.address = laneAddress(n, 'name');
    const tryButton = button('try', { cls: 'try', verb: 'ai.try', address: laneAddress(n, 'trying') });
    const note = el('div', 'lnote');
    note.dataset.address = laneAddress(n, 'generating');
    const error = el('span', 'lerr');
    error.dataset.address = laneAddress(n, 'error');
    note.append(error);
    lane.append(top, nav, name, tryButton, note);
    gen.addEventListener('click', () => generateLane(n));
    prev.addEventListener('click', () => step(n, -1));
    next.addEventListener('click', () => step(n, 1));
    tryButton.addEventListener('click', () => toggleTry(n));
    lanesBox.append(lane);
    return { lane, gen, prev, next, counter, name, tryButton, note, error };
  });
  const install = el('div', 'ai-install');
  const command = el('code', 'ai-command');
  command.dataset.address = 'ai.install_command';
  const copy = button('copy command', { id: 'ai-copy' });
  const installNow = button('install now', { id: 'ai-install', cls: 'on', verb: 'ai.install', address: 'ai.installing' });
  const installNote = el('span', 'note', 'pip runs in the background');
  const installRow = el('div', 'row');
  installRow.append(copy, installNow, installNote);
  const installText = el('div', 'note');
  installText.append('Install them with ', command, ' (PyTorch, a large download).');
  install.append(el('div', 'ai-install-title', 'The AI generator needs the machine-learning extras'), installText, installRow);
  middle.append(el('h4', '', 'lanes: tried sounds play on the face'), lanesBox, install);

  // right: patterns and apply
  const right = el('section', 'ai-sec ai-right');
  right.append(el('h4', '', 'patterns and apply'));
  const modes = el('div', 'ai-pmodes');
  const keepMode = button('keep current', { id: 'ai-pmode-keep' });
  const newMode = button('generate new', { id: 'ai-pmode-new' });
  modes.append(keepMode, newMode);
  const bankRow = el('div', 'ai-bankrow');
  const patternTemp = control('px-knob', 'pref.ai.pattern_temperature', { label: 'pattern temp', name: 'pattern temperature', size: '30' });
  const bankNote = el('span', 'note ai-banknote');
  bankNote.dataset.address = 'ai.bank';
  const newBank = button('↻', { id: 'ai-new-bank', cls: 'arr', verb: 'ai.generate_patterns' });
  newBank.title = 'generate a new bank of 12 patterns for the drum patches on the face';
  bankRow.append(patternTemp, bankNote, newBank);
  const actions = el('div', 'ai-actions');
  const loop = button('▶ loop', { id: 'ai-loop', verb: 'ai.pattern_try', address: 'ai.preview' });
  const bank = button('▶ bank A→L', { id: 'ai-bank', verb: 'ai.pattern_try' });
  const keep = button('keep tried', { id: 'ai-keep', cls: 'on', verb: 'ai.keep' });
  const replace = button('replace patterns', { id: 'ai-replace', verb: 'ai.replace_patterns' });
  const revert = button('revert all', { id: 'ai-revert', cls: 'red', verb: 'ai.revert' });
  actions.append(loop, bank, keep, replace, revert);
  right.append(modes, bankRow, actions);

  const body = el('div', 'ai-body');
  body.append(left, middle, right);
  page.append(head, body);

  // ------------------------------------------------------------ rendering
  const tabs = CHANNELS.map((n) => stage.querySelector(`.strip[data-channel="${n}"] .tab`));
  const lane = (n) => Object.fromEntries(LANE_FIELDS.map((f) => [f, value(laneAddress(n, f))]));
  const renderLane = (n) => {
    const v = laneView(lane(n));
    const w = lanes[n - 1];
    w.lane.dataset.state = v.state;
    w.counter.textContent = v.counter;
    w.name.textContent = v.name;
    w.name.title = v.name;
    w.prev.disabled = w.next.disabled = !v.canStep;
    w.tryButton.disabled = !v.canTry;
    w.tryButton.textContent = v.tryLabel;
    w.tryButton.classList.toggle('on', v.trying);
    w.error.textContent = v.note; // the note (generating…, error: …, who plays)
    w.note.classList.toggle('busy', v.state === 'generating');
    w.note.classList.toggle('bad', v.state === 'error');
    w.lane.classList.toggle('trying', v.trying);
    // The strip tab: the tried candidate's name in italics
    if (tabs[n - 1]) {
      tabs[n - 1].classList.toggle('aitry', v.trying);
      if (v.trying && v.name) tabs[n - 1].title = `trying ${v.name}`; else tabs[n - 1].removeAttribute('title');
    }
  };
  const renderModels = () => {
    const available = !!value('ai.available');
    const all = value('ai.models') || {};
    for (const [kind] of KINDS) {
      const v = modelView(all[kind], available);
      modelLines[kind].textContent = v.text;
      modelLines[kind].title = all[kind] && all[kind].path ? all[kind].path : '';
      modelLines[kind].dataset.tone = v.tone;
    }
  };
  const renderAvailable = () => {
    const available = !!value('ai.available');
    page.classList.toggle('unavailable', !available);
    const installing = !!value('ai.installing');
    installNow.disabled = installing;
    installNow.textContent = installing ? 'installing…' : 'install now';
    installNote.textContent = installing ? 'pip is running in the background…' : 'pip runs in the background';
    generateAllButton.disabled = !available;
    renderModels();
  };
  const renderState = () => {
    const state = value('ai.state');
    stateNote.textContent = stateText(state);
    stateNote.dataset.state = state || '';
  };
  const renderTried = () => {
    const tried = value('ai.tried') || [];
    triedNote.textContent = tried.length ? `trying on ch ${tried.join(', ')}` : '';
    keep.disabled = !tried.length;
  };
  const renderPatterns = () => {
    const bankState = value('ai.bank');
    const preview = value('ai.preview') || 'off';
    keepMode.classList.toggle('on', patternMode === 'keep');
    newMode.classList.toggle('on', patternMode === 'generate');
    right.classList.toggle('generate', patternMode === 'generate');
    bankNote.textContent = bankText(patternMode, bankState);
    newBank.disabled = bankState === 'generating';
    replace.disabled = !canReplacePatterns(patternMode, bankState);
    loop.textContent = preview === 'loop' ? '■ stop loop' : '▶ loop';
    bank.textContent = preview === 'bank' ? '■ stop bank' : '▶ bank A→L';
    loop.classList.toggle('on', preview === 'loop');
    bank.classList.toggle('on', preview === 'bank');
  };
  const renderBack = (n) => { back.textContent = `◀ ch ${n || 1} edit`; };

  CHANNELS.forEach((n) => LANE_FIELDS.forEach((f) => watch(laneAddress(n, f), () => renderLane(n))));
  watch('ai.available', renderAvailable);
  watch('ai.installing', renderAvailable);
  watch('ai.models', renderModels);
  watch('ai.install_command', (cmd) => { command.textContent = cmd || 'pip install "pythonic[ml]"'; });
  watch('ai.state', renderState);
  watch('ai.tried', renderTried);
  watch('ai.bank', renderPatterns);
  watch('ai.preview', renderPatterns);
  watch('global.channel', renderBack);

  // ------------------------------------------------------------ models
  async function loadModel(kind, title) {
    if (!value('ai.available')) {
      sheets.alert({ title: 'The ML extras are missing', text: `Install them first: ${value('ai.install_command') || ''}`, tone: 'error' });
      return;
    }
    const current = (value('ai.models') || {})[kind];
    const path = await client.openFile({ title, filters: ['PyTorch checkpoints (*.pt)', 'All files (*)'],
      folder: current && current.path ? String(current.path).replace(/[\\/][^\\/]*$/, '') : null });
    if (!path) return;
    display.show(`${kind.toUpperCase()} MODEL`, `loading ${fileName(path)}`);
    const r = await run('ai.load_model', { kind, path }, 'Could not load the model');
    if (r) display.show(`${kind.toUpperCase()} MODEL`, `${fileName(r.path)} loaded`);
  }

  /** Load the saved (or bundled) models when the page opens (tkinter). */
  function loadModels() {
    if (!value('ai.available')) return;
    const all = value('ai.models') || {};
    for (const [kind] of KINDS) {
      if (all[kind] && all[kind].status === 'unloaded') panel.act('ai.load_model', { kind });
    }
  }

  // ------------------------------------------------------------ generation
  const generationArgs = () => {
    const args = { candidates };
    const s = parseSeed(seed.value);
    if (s !== null) args.seed = s;
    return args;
  };
  /** The drum patches changed: drop the AI pattern bank (tkinter). */
  const invalidateBank = () => {
    if (value('ai.bank') && value('ai.bank') !== 'none') panel.act('ai.clear_patterns');
  };
  const hit = (n) => {
    const transport = store.readout('transport');
    if (!transport || !transport.playing) client.trigger(n, 127).catch(() => {});
  };

  async function generateLane(n) {
    invalidateBank();
    display.show(`CH${n} AI`, 'generating…');
    const r = await run('ai.generate', { channel: n, type: value(laneAddress(n, 'type')), ...generationArgs() },
      'Generation failed');
    if (!r) return;
    if (r.failed && r.failed.length) display.alert(`CH${n} AI`, r.failed[0].error);
    else {
      display.show(`CH${n} ${String(value(laneAddress(n, 'name')) || '').toUpperCase()}`, `1 / ${value(laneAddress(n, 'candidates'))}`);
      hit(n);
    }
  }

  async function generateAll() {
    display.show('AI GENERATOR', 'generating 8 lanes…');
    const r = await run('ai.generate', generationArgs(), 'Generation failed');
    if (!r) return;
    const failed = r.failed || [];
    display.show('AI GENERATOR', failed.length ? `${r.channels.length} ready, ${failed.length} failed` : '8 lanes ready');
    if (patternMode === 'generate') generatePatterns();
  }

  async function generatePatterns() {
    const args = {};
    const s = parseSeed(seed.value);
    if (s !== null) args.seed = s;
    const r = await run('ai.generate_patterns', args, 'Could not generate the patterns');
    if (r) display.show('AI PATTERNS', `bank of ${r.patterns} ready`);
  }

  async function step(n, direction) {
    invalidateBank();
    const r = await run('ai.try', { channel: n, step: direction });
    if (!r) return;
    display.show(`CH${n} ${String(r.name || '').toUpperCase()}`, `${r.candidate} / ${value(laneAddress(n, 'candidates'))}`);
    hit(n);
  }

  async function toggleTry(n) {
    if (value(laneAddress(n, 'trying'))) {
      const r = await run('ai.untry', { channel: n });
      if (r) display.show(`CH${n}`, 'old sound back');
      return;
    }
    const r = await run('ai.try', { channel: n });
    if (!r) return;
    display.show(`CH${n} ${String(r.name || '').toUpperCase()}`, 'trying');
    hit(n);
  }

  generateAllButton.addEventListener('click', generateAll);
  newBank.addEventListener('click', () => {
    if (patternMode === 'generate') generatePatterns();
  });
  candDown.addEventListener('click', () => setCandidates(candidates - 1));
  candUp.addEventListener('click', () => setCandidates(candidates + 1));
  candValue.addEventListener('wheel', (e) => {
    e.preventDefault();
    if (e.deltaY) setCandidates(candidates + (e.deltaY < 0 ? 1 : -1));
  }, { passive: false });
  function setCandidates(n) {
    candidates = clampCandidates(n);
    candValue.textContent = String(candidates);
    display.show('AI CANDIDATES', String(candidates));
  }
  reseed.addEventListener('click', () => {
    seed.value = String(randomSeed());
    display.show('AI SEED', seed.value);
  });
  seed.addEventListener('change', () => {
    const s = parseSeed(seed.value);
    if (s === null) seed.value = '';
    display.show('AI SEED', s === null ? 'random' : String(s));
  });
  seed.addEventListener('keydown', (e) => {
    e.stopPropagation();
    if (e.key === 'Enter') seed.blur();
  });

  // ------------------------------------------------------------ patterns and apply
  const setPatternMode = (mode) => {
    if (mode === patternMode) return;
    patternMode = mode;
    invalidateBank();
    renderPatterns();
    display.show('AI PATTERNS', mode === 'generate' ? 'generate new' : 'keep current');
  };
  keepMode.addEventListener('click', () => setPatternMode('keep'));
  newMode.addEventListener('click', () => setPatternMode('generate'));

  const preview = async (mode) => {
    const args = previewArgs(value('ai.preview') || 'off', mode, patternMode);
    const r = await run('ai.pattern_try', args);
    if (!r) return;
    if (!args.mode) display.show('AI PREVIEW', 'stopped, restored');
    else display.show('AI PREVIEW', mode === 'bank' ? 'bank A→L' : `loop pattern ${r.pattern || ''}`.trim());
  };
  loop.addEventListener('click', () => preview('loop'));
  bank.addEventListener('click', () => preview('bank'));
  keep.addEventListener('click', async () => {
    const r = await run('ai.keep', {}, 'Could not keep the sounds');
    if (r) display.show('AI GENERATOR', keptText(r));
  });
  replace.addEventListener('click', async () => {
    const r = await run('ai.replace_patterns', {}, 'Could not replace the patterns');
    if (r) display.show('AI GENERATOR', `${keptText(r)} + 12 patterns`);
  });
  revert.addEventListener('click', async () => {
    const r = await run('ai.revert');
    if (r) display.show('AI GENERATOR', r.reverted && r.reverted.length ? `reverted ch ${r.reverted.join(', ')}` : 'reverted');
  });

  // ------------------------------------------------------------ install
  copy.addEventListener('click', async () => {
    const ok = await copyText(String(value('ai.install_command') || ''));
    display.show('INSTALL COMMAND', ok ? 'copied' : 'could not copy');
  });
  installNow.addEventListener('click', async () => {
    const cmd = String(value('ai.install_command') || '');
    const yes = await sheets.ask('Install the ML extras?',
      `This runs:\n${cmd}\n\nin this Python environment (PyTorch, a large download).`, { yes: 'install' });
    if (!yes) return;
    display.show('AI GENERATOR', 'installing…');
    const event = await panel.act('ai.install');
    if (event.status === 'done') {
      sheets.alert({ title: 'ML extras installed', text: 'The AI generator is ready.', tone: 'ok' });
      loadModels();
    } else {
      sheets.alert({ title: 'The install failed', text: event.error, tone: 'error' });
    }
  });

  // ------------------------------------------------------------ open and leave
  const triedNames = () => Object.fromEntries((value('ai.tried') || []).map((n) => [n, value(laneAddress(n, 'name'))]));

  /**
   * Settle the tried sounds before leaving: keep or revert (and cancel when
   * the page is still showing). Resolves false when the user stays.
   */
  function settle(cancellable) {
    if (settling) return settling;
    settling = (async () => {
      const tried = value('ai.tried') || [];
      if (!tried.length) {
        if ((value('ai.preview') || 'off') !== 'off') await panel.act('ai.pattern_try', { mode: null });
        return true;
      }
      const q = leaveQuestion(tried, triedNames());
      const buttons = [
        ...(cancellable ? [{ label: 'cancel', value: 'cancel' }] : []),
        { label: 'revert', value: 'revert' }, { label: 'keep', value: 'keep', primary: true },
      ];
      const answer = await sheets.alert({ title: q.title, text: q.text, tone: 'ok', buttons });
      if (answer !== 'keep' && answer !== 'revert') return false;
      if (answer === 'keep') {
        const r = await run('ai.keep', {}, 'Could not keep the sounds');
        if (r) display.show('AI GENERATOR', keptText(r));
        if ((value('ai.preview') || 'off') !== 'off') await panel.act('ai.pattern_try', { mode: null });
      } else {
        const r = await run('ai.revert');
        if (r) display.show('AI GENERATOR', 'reverted');
      }
      return true;
    })().finally(() => { settling = null; });
    return settling;
  }

  /** Leave from the page's own buttons: ask first, then hide. */
  async function leave() {
    if (drawer.current !== 'ai') return true;
    if (!(await settle(true))) return false;
    leaving = true;
    try { drawer.hide('ai'); } finally { leaving = false; }
    return true;
  }
  back.addEventListener('click', leave);
  close.addEventListener('click', leave);

  function open() {
    if (drawer.current !== 'ai') {
      drawer.show('ai', page, { onHide: () => { if (!leaving) settle(false); } });
    }
    renderAvailable();
    renderPatterns();
    loadModels();
    return page;
  }

  offs.push(panel.registerPage('ai', open));
  // A preset load replaces every sound: settle the tried ones first (#21)
  offs.push(panel.guardPresetLoad(() => settle(true)));

  renderAvailable();
  renderState();
  renderTried();
  renderPatterns();
  renderBack(value('global.channel'));
  CHANNELS.forEach(renderLane);

  return {
    element: page,
    open,
    leave,
    settle,
    get patternMode() { return patternMode; },
    get candidates() { return candidates; },
    destroy() {
      offs.forEach((off) => off());
      if (drawer.current === 'ai') { leaving = true; drawer.hide('ai'); }
    },
  };
}
