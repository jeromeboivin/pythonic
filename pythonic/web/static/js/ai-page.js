// The AI generator page (map decisions #13, #21, #22): a page of the edit
// rack drawer that makes new sounds and patterns and plays them on the live
// face, one click each.
//
//   left (under the left column)   sound temperature, generate 8 sounds
//   middle, one lane per strip     drum type, gen, the new sound's name,
//                                  old | new (or, without the ML extras, the
//                                  pip command with copy and install now)
//   right (under the right column) pattern temperature, generate patterns,
//                                  old | new, keep, revert
//
// gen makes one new sound for its channel and plays it there (the core tries
// it); generate patterns makes 12 (A-L) for the sounds on the face and plays
// them (ai.pattern_try 'try': the transport starts, any pattern picked plays
// its new version). old | new switches back and forth. The strip tab of a
// channel playing a new sound shows its name in italics, knob edits there go
// into the new sound. keep commits what plays, sounds and patterns, as one
// undo step (ai.keep or ai.replace_patterns), revert puts the old ones back;
// both then forget the new ones (ai.clear). Leaving the page with something
// new asks keep or revert: ✕ and ◀ ch n edit ask first (cancel stays), any
// other way out (another drawer page, closing the rack) asks once the page
// is gone; a preset load asks first too (panel.guardPresetLoad). models…
// opens the setup sheet's ai tab, where the models are chosen (#22).
//
//   const ai = mountAiPage({ panel, store, client, stage });
//   ai.open() / ai.leave() / ai.element / ai.destroy()

import {
  AI_EDITION_TEXT, CHANNELS, hasTrial, keptText, laneAddress, laneView, LANE_FIELDS, leaveQuestion,
  modelProblem, patternView, stateText, trialText,
} from './ai-logic.js';

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

/** An old | new switch: two buttons, the one playing lit. */
function abSwitch(cls = '') {
  const box = el('div', `ab${cls ? ` ${cls}` : ''}`);
  const old = button('old', { cls: 'old' });
  const fresh = button('new', { cls: 'new' });
  box.append(old, fresh);
  const render = (canSwitch, trying) => {
    old.disabled = fresh.disabled = !canSwitch;
    old.classList.toggle('on', canSwitch && !trying);
    fresh.classList.toggle('on', canSwitch && trying);
  };
  return { box, old, fresh, render };
}

export function mountAiPage({ panel, store, client, stage }) {
  const { display, drawer, sheets } = panel;
  const offs = [];
  const watch = (address, fn, options) => { offs.push(store.watch(address, fn, options)); };
  const value = (address) => store.value(address);
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
  const trialNote = el('span', 'ai-tried');
  trialNote.dataset.address = 'ai.tried';
  const models = button('models…', { id: 'ai-models' });
  models.title = 'choose the AI models (setup ▸ ai)';
  const close = button('✕', { id: 'ai-close' });
  head.append(back, el('span', 'ai-title', 'AI generator'), stateNote, el('span', 'sp'), trialNote, models, close);

  // left: new sounds
  const left = el('section', 'ai-sec ai-left');
  const generateAllButton = button('generate 8 sounds', { id: 'ai-generate-all', cls: 'on big', verb: 'ai.generate' });
  const modelNote = el('div', 'note ai-model-note');
  modelNote.dataset.address = 'ai.models';
  left.append(el('h4', '', 'new sounds'),
    control('px-knob', 'pref.ai.patch_temperature', { label: 'temperature', name: 'sound temperature', size: '34' }),
    generateAllButton, el('div', 'note', 'or gen one drum in its lane'), modelNote);

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
    gen.title = `a new ch${n} sound of this drum type`;
    top.append(type, gen);
    const name = el('div', 'cname');
    name.dataset.address = laneAddress(n, 'name');
    const ab = abSwitch();
    ab.box.dataset.address = laneAddress(n, 'trying');
    const note = el('div', 'lnote');
    note.dataset.address = laneAddress(n, 'generating');
    const noteText = el('span', 'lerr');
    noteText.dataset.address = laneAddress(n, 'error');
    note.append(noteText);
    lane.append(top, name, ab.box, note);
    gen.addEventListener('click', () => generateLane(n));
    ab.old.addEventListener('click', () => useOld(n));
    ab.fresh.addEventListener('click', () => useNew(n));
    lanesBox.append(lane);
    return { lane, gen, name, ab, note, noteText };
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
  const installTitle = el('div', 'ai-install-title');
  install.append(installTitle, installText, installRow);
  middle.append(el('h4', '', 'one new sound per drum: it plays on the face'), lanesBox, install);

  // right: new patterns, keep and revert
  const right = el('section', 'ai-sec ai-right');
  const patternRow = el('div', 'ai-patrow');
  const generatePatternsButton = button('generate patterns', { id: 'ai-gen-patterns', cls: 'on big', verb: 'ai.generate_patterns' });
  patternRow.append(
    control('px-knob', 'pref.ai.pattern_temperature', { label: 'temperature', name: 'pattern temperature', size: '30' }),
    generatePatternsButton);
  const patternAb = abSwitch('ai-patab');
  patternAb.box.dataset.address = 'ai.preview';
  patternAb.old.id = 'ai-pat-old';
  patternAb.fresh.id = 'ai-pat-new';
  const patternNote = el('div', 'note ai-patnote');
  patternNote.dataset.address = 'ai.bank';
  const actions = el('div', 'ai-actions');
  const keep = button('keep', { id: 'ai-keep', cls: 'on', verb: 'ai.keep' });
  keep.title = 'keep the new sounds and patterns that play (one undo step)';
  const revert = button('revert', { id: 'ai-revert', cls: 'red', verb: 'ai.clear' });
  revert.title = 'put the old sounds and patterns back';
  actions.append(keep, revert);
  right.append(el('h4', '', 'new patterns'), patternRow, patternAb.box, patternNote, actions);

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
    w.name.textContent = v.name;
    w.name.title = v.name;
    w.ab.render(v.canSwitch, v.trying);
    w.gen.disabled = v.state === 'generating' || !value('ai.available');
    w.noteText.textContent = v.note; // generating…, error: …, which sound plays
    w.note.classList.toggle('busy', v.state === 'generating');
    w.note.classList.toggle('bad', v.state === 'error');
    w.lane.classList.toggle('trying', v.trying);
    // The strip tab: the new sound's name in italics
    if (tabs[n - 1]) {
      tabs[n - 1].classList.toggle('aitry', v.trying);
      if (v.trying && v.name) tabs[n - 1].title = `new sound ${v.name}`; else tabs[n - 1].removeAttribute('title');
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
    modelNote.textContent = available ? modelProblem(value('ai.models')) : '';
    CHANNELS.forEach(renderLane);
    renderPatterns();
  };
  const renderState = () => {
    const state = value('ai.state');
    stateNote.textContent = stateText(state);
    stateNote.dataset.state = state || '';
  };
  const renderTrial = () => {
    const tried = value('ai.tried') || [];
    const preview = value('ai.preview') || 'off';
    trialNote.textContent = trialText(tried, preview);
    keep.disabled = revert.disabled = !hasTrial(tried, preview);
  };
  const renderPatterns = () => {
    const v = patternView(value('ai.bank') || 'none', value('ai.preview') || 'off');
    patternAb.render(v.canSwitch, v.trying);
    patternNote.textContent = v.note;
    patternNote.classList.toggle('busy', v.generating);
    generatePatternsButton.disabled = v.generating || !value('ai.available');
    generatePatternsButton.textContent = v.generating ? 'generating…' : 'generate patterns';
    renderTrial();
  };
  const renderBack = (n) => { back.textContent = `◀ ch ${n || 1} edit`; };

  CHANNELS.forEach((n) => LANE_FIELDS.forEach((f) => watch(laneAddress(n, f), () => renderLane(n))));
  watch('ai.available', renderAvailable);
  watch('ai.installing', renderAvailable);
  watch('ai.models', renderAvailable);
  watch('ai.install_command', (cmd) => {
    // Empty in the standalone app: no pip there, the AI comes with its own edition
    const edition = cmd === '';
    command.textContent = cmd || 'pip install "pythonic[ml]"';
    installTitle.textContent = edition ? 'This edition of Pythonic has no AI generator'
      : 'The AI generator needs the machine-learning extras';
    installText.replaceChildren(...(edition ? [AI_EDITION_TEXT]
      : ['Install them with ', command, ' (PyTorch, a large download).']));
    installRow.style.display = edition ? 'none' : '';
  });
  watch('ai.state', renderState);
  watch('ai.tried', renderTrial);
  watch('ai.bank', renderPatterns);
  watch('ai.preview', renderPatterns);
  watch('global.channel', renderBack);

  // ------------------------------------------------------------ models
  /** Load the saved (or bundled) models when the page opens, so the first click is quick. */
  function loadModels() {
    if (!value('ai.available')) return;
    const all = value('ai.models') || {};
    for (const kind of ['patch', 'pattern']) {
      if (all[kind] && all[kind].status === 'unloaded') panel.act('ai.load_model', { kind });
    }
  }
  models.addEventListener('click', () => panel.openPage('setup', { tab: 'ai' }));

  // ------------------------------------------------------------ sounds
  const hit = (n) => {
    const transport = store.readout('transport');
    if (!transport || !transport.playing) client.trigger(n, 127).catch(() => {});
  };

  async function generateLane(n) {
    display.show(`CH${n} AI`, 'generating…');
    const r = await run('ai.generate', { channel: n, type: value(laneAddress(n, 'type')) }, 'Generation failed');
    if (!r) return;
    if (r.failed && r.failed.length) display.alert(`CH${n} AI`, r.failed[0].error);
    else {
      display.show(`CH${n} ${String(value(laneAddress(n, 'name')) || '').toUpperCase()}`, 'new sound');
      hit(n);
    }
  }

  async function generateAll() {
    display.show('AI GENERATOR', 'generating 8 sounds…');
    const r = await run('ai.generate', {}, 'Generation failed');
    if (!r) return;
    const failed = r.failed || [];
    display.show('AI GENERATOR', failed.length ? `${r.channels.length} new, ${failed.length} failed` : '8 new sounds');
  }

  async function useOld(n) {
    if (!value(laneAddress(n, 'trying'))) return;
    if (await run('ai.untry', { channel: n })) display.show(`CH${n}`, 'old sound');
  }

  async function useNew(n) {
    if (value(laneAddress(n, 'trying'))) return;
    const r = await run('ai.try', { channel: n });
    if (!r) return;
    display.show(`CH${n} ${String(r.name || '').toUpperCase()}`, 'new sound');
    hit(n);
  }

  generateAllButton.addEventListener('click', generateAll);

  // ------------------------------------------------------------ patterns
  async function generatePatterns() {
    display.show('AI PATTERNS', 'generating…');
    const r = await run('ai.generate_patterns', {}, 'Could not generate the patterns');
    if (!r) return;
    if (await run('ai.pattern_try', { mode: 'try' }, 'Could not play the patterns')) {
      display.show('AI PATTERNS', `${r.patterns} new patterns`);
    }
  }

  generatePatternsButton.addEventListener('click', generatePatterns);
  patternAb.old.addEventListener('click', async () => {
    if (value('ai.preview') !== 'try') return;
    if (await run('ai.pattern_try', { mode: null })) display.show('AI PATTERNS', 'old patterns');
  });
  patternAb.fresh.addEventListener('click', async () => {
    if (value('ai.preview') === 'try') return;
    if (await run('ai.pattern_try', { mode: 'try' }, 'Could not play the patterns')) display.show('AI PATTERNS', 'new patterns');
  });

  // ------------------------------------------------------------ keep and revert
  /** Keep what plays (one undo step), then forget the new ones. */
  async function keepAll() {
    const patterns = value('ai.preview') === 'try';
    const r = await run(patterns ? 'ai.replace_patterns' : 'ai.keep', {}, 'Could not keep the new sounds');
    if (!r) return false;
    await panel.act('ai.clear');
    display.show('AI GENERATOR', keptText(r));
    return true;
  }

  /** Put the old sounds and patterns back and forget the new ones. */
  async function revertAll() {
    const r = await run('ai.clear');
    if (r) display.show('AI GENERATOR', 'reverted');
    return !!r;
  }

  keep.addEventListener('click', keepAll);
  revert.addEventListener('click', revertAll);

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
   * Settle what is new before leaving: keep or revert (and cancel when the
   * page is still showing). Resolves false when the user stays.
   */
  function settle(cancellable) {
    if (settling) return settling;
    settling = (async () => {
      const tried = value('ai.tried') || [];
      const preview = value('ai.preview') || 'off';
      if (!hasTrial(tried, preview)) {
        if (preview !== 'off') await panel.act('ai.pattern_try', { mode: null });
        return true;
      }
      const q = leaveQuestion(tried, triedNames(), preview === 'try');
      const buttons = [
        ...(cancellable ? [{ label: 'cancel', value: 'cancel' }] : []),
        { label: 'revert', value: 'revert' }, { label: 'keep', value: 'keep', primary: true },
      ];
      const answer = await sheets.alert({ title: q.title, text: q.text, tone: 'ok', buttons });
      if (answer === 'keep') await keepAll();
      else if (answer === 'revert') await revertAll();
      else return false;
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
    loadModels();
    return page;
  }

  offs.push(panel.registerPage('ai', open));
  // A preset load replaces every sound: settle the new ones first (#21)
  offs.push(panel.guardPresetLoad(() => settle(true)));

  renderAvailable();
  renderState();
  renderBack(value('global.channel'));

  return {
    element: page,
    open,
    leave,
    settle,
    destroy() {
      offs.forEach((off) => off());
      if (drawer.current === 'ai') { leaving = true; drawer.hide('ai'); }
    },
  };
}
