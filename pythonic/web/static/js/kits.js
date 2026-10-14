// KIT, INST and WRITE, as on a TR-8S (owner's manual: "Editing a Kit or
// Instrument", "Saving ... (WRITE)"). Kits are a library of their own
// (kit.list: the factory kits, then the user kits), apart from presets.
//
// KIT lights and turns the display into the kit selector: `KIT 004*` (the
// number, * when the kit's sounds are edited) over the kit's name. INST lights
// and turns it into the drum patch selector of the selected channel: `INST 1
// BD 3/12` (the channel, the category, the place in it) over the drum patch's
// name. The VALUE controls step through the list and every step plays at
// once: ◀ ▶ under the display, the wheel over the display, the arrow keys;
// SHIFT with any of them jumps to the previous or next category (INST) or
// between the factory and user kits (KIT). In INST the channel buttons change
// the channel, as BD-RC do; while stopped each new drum patch is heard. The
// lit button again or Escape leaves; the display shows the pattern again.
//
// Kit 000 is the preset kit: the sounds the preset was loaded with (a preset
// that names no kit plays it). Switching kits keeps the edits of the kit left
// (the core keeps them for the session); WRITE stores them: over the current
// user kit, or as a new user kit (a factory kit is only written under a new
// name), and saves the preset (its patterns and the kit it plays). A
// pattern's menu sets the kit the pattern recalls (pattern.<P>.kit).
//
// Bound addresses: kit.list, kit.current, kit.name, kit.edited,
// drum_patch.library, pattern.<P>.kit, program.occupied (WRITE's programs
// item).

import { drumCategory } from './drum-type.js';
import { PATTERNS } from './steps-logic.js';

export const PRESET_KIT = '';

/** The kits the KIT selector steps through: kit 000, the preset kit, then the library. */
export function kitEntries(list, presetName) {
  const kits = Array.isArray(list) ? list.filter((k) => k && typeof k.id === 'string') : [];
  return [{ id: PRESET_KIT, name: String(presetName || 'preset'), factory: false, preset: true }, ...kits];
}

/** A kit's number on the display: 000 for the preset kit. */
export const kitNumber = (index) => String(Math.max(0, index)).padStart(3, '0');

/** The two display lines of the KIT selector. */
export function kitLines(entries, index, edited) {
  const kit = entries[index];
  if (!kit) return ['KIT ---', 'no kit'];
  const group = kit.preset ? 'PRESET' : kit.factory ? 'FACTORY' : 'USER';
  return [`KIT ${kitNumber(index)}${edited ? '*' : ' '} ${group}`, kit.name.toUpperCase()];
}

/** `index` moved by `delta` within a list of `length` (clamped, as a VALUE knob). */
export const stepIndex = (length, index, delta) => Math.max(0, Math.min(length - 1, index + delta));

/**
 * The index SHIFT + a step goes to: the first entry of the next group
 * (`delta` > 0), or of the current group, or of the previous one when
 * already at the start of the current one (`delta` < 0). `groupOf(i)` names
 * entry i's group.
 */
export function groupJump(length, index, delta, groupOf) {
  if (length <= 0) return 0;
  const group = groupOf(index);
  let i = index;
  if (delta > 0) {
    while (i < length && groupOf(i) === group) i += 1;
    return i < length ? i : index;
  }
  while (i > 0 && groupOf(i - 1) === group) i -= 1;
  if (i < index) return i;
  if (i === 0) return 0;
  const previous = groupOf(i - 1);
  while (i > 0 && groupOf(i - 1) === previous) i -= 1;
  return i;
}

/**
 * Where the INST selector starts for a channel: the drum patch it last
 * loaded there (`lastId`), else the first one named as the channel's sound,
 * else the first of the sound's category, else the first of all.
 */
export function instStart(library, channelName, lastId = null) {
  const list = Array.isArray(library) ? library : [];
  if (!list.length) return -1;
  const byId = lastId ? list.findIndex((e) => e[0] === lastId && e[1] === channelName) : -1;
  if (byId >= 0) return byId;
  const byName = list.findIndex((e) => e[1] === channelName);
  if (byName >= 0) return byName;
  const category = drumCategory(channelName);
  const byCategory = list.findIndex((e) => e[2] === category);
  return byCategory >= 0 ? byCategory : 0;
}

/** The two display lines of the INST selector. */
export function instLines(library, index, channel) {
  const entry = (Array.isArray(library) ? library : [])[index];
  if (!entry) return [`INST ${channel}`, 'no drum patches'];
  const category = entry[2];
  let first = index;
  while (first > 0 && library[first - 1][2] === category) first -= 1;
  let count = index - first + 1;
  while (first + count < library.length && library[first + count][2] === category) count += 1;
  return [`INST ${channel} ${category} ${index - first + 1}/${count}`, String(entry[1]).toUpperCase()];
}

/** A default name for "write kit as": the kit's own for a user kit, else "<name> copy". */
export function writeName(entry) {
  if (!entry) return 'New kit';
  return entry.factory || entry.preset ? `${entry.name} copy` : entry.name;
}

export function mountKits({ store, client, ctx, display, act, stage, slot, sheets, presets: getPresets,
  patterns, onBase = () => {} }) {
  const offs = [];
  const state = { mode: null, index: 0, shift: false };
  const lastInst = {}; // channel -> library id loaded there through INST
  const value = (address) => store.value(address);
  const channel = () => Number(value('global.channel')) || 1;
  const library = () => (Array.isArray(value('drum_patch.library')) ? value('drum_patch.library') : []);
  const entries = () => kitEntries(value('kit.list'), value('preset.name'));
  const currentKitIndex = () => Math.max(0, entries().findIndex((k) => k.id === (value('kit.current') ?? '')));

  // ------------------------------------------------------------ buttons
  const kitButton = slot('kit');
  const instButton = slot('inst');
  const writeButton = slot('write');
  for (const [b, id] of [[kitButton, 'kit-button'], [instButton, 'inst-button'], [writeButton, 'write-button']]) {
    b.disabled = false;
    b.classList.remove('slot');
    b.id = id;
  }

  /** The display lines while a selector is on (null: the panel's own). */
  function baseLines() {
    if (state.mode === 'kit') return kitLines(entries(), state.index, state.index === currentKitIndex() && value('kit.edited'));
    if (state.mode === 'inst') return instLines(library(), state.index, channel());
    return null;
  }

  function render(fresh = false) {
    kitButton.classList.toggle('on', state.mode === 'kit');
    instButton.classList.toggle('on', state.mode === 'inst');
    writeButton.classList.toggle('edited', !!value('kit.edited'));
    display.classList.toggle('selector', !!state.mode);
    const presets = getPresets();
    if (presets) presets.showButtons();
    onBase();
    if (fresh && state.mode) display.clear(); // the selector's lines at once, over any message
  }

  function setMode(mode) {
    state.mode = mode;
    if (mode === 'kit') state.index = currentKitIndex();
    if (mode === 'inst') syncInst();
    render(true);
  }

  function syncInst() {
    const n = channel();
    state.index = Math.max(0, instStart(library(), String(value(`ch${n}.name`) || ''), lastInst[n]));
  }

  // ------------------------------------------------------------ loading (the latest step wins)
  let busy = false;
  let pending = null;
  async function send(job) {
    pending = job;
    if (busy) return;
    busy = true;
    while (pending) {
      const next = pending;
      pending = null;
      try { await next(); } catch (err) { console.warn('kits', err); }
    }
    busy = false;
  }

  const playing = () => !!(store.readout('transport') || {}).playing;

  function loadKit(index) {
    const kit = entries()[index];
    if (!kit) return;
    send(() => act('kit.select', { kit: kit.id, browse: true }));
  }

  function loadInst(index) {
    const entry = library()[index];
    if (!entry) return;
    const n = channel();
    lastInst[n] = entry[0];
    send(() => act('drum_patch.load', { library: entry[0], channel: n, browse: true }).then((event) => {
      if (event.status === 'done' && !playing() && !pending) client.trigger(n, 100).catch(() => {});
    }));
  }

  /** VALUE: step the selector (`jump`: to the next / previous group). False when no selector is on. */
  function step(delta, jump = false) {
    if (!state.mode) return false;
    if (state.mode === 'kit') {
      const list = entries();
      const group = (i) => (list[i].preset ? 0 : list[i].factory ? 1 : 2);
      const next = jump ? groupJump(list.length, state.index, delta, group) : stepIndex(list.length, state.index, delta);
      if (next !== state.index) { state.index = next; loadKit(next); }
    } else {
      const list = library();
      if (!list.length) return true;
      const next = jump ? groupJump(list.length, state.index, delta, (i) => list[i][2])
        : stepIndex(list.length, state.index, delta);
      if (next !== state.index) { state.index = next; loadInst(next); }
    }
    render(true);
    return true;
  }

  kitButton.addEventListener('click', () => setMode(state.mode === 'kit' ? null : 'kit'));
  instButton.addEventListener('click', () => setMode(state.mode === 'inst' ? null : 'inst'));

  // The wheel over the display and the arrow keys are VALUE too
  const onWheel = (e) => {
    if (!state.mode || !e.deltaY) return;
    e.preventDefault();
    step(e.deltaY > 0 ? 1 : -1, e.shiftKey);
  };
  display.addEventListener('wheel', onWheel, { passive: false });
  offs.push(() => display.removeEventListener('wheel', onWheel));
  const onKey = (e) => {
    if (!state.mode || e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey) return;
    const target = e.target;
    if (target && (target.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName))) return;
    if (stage.querySelector(':scope > .sheet-layer')) return;
    const delta = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
    if (delta) {
      e.preventDefault();
      step(delta, e.shiftKey);
    } else if (e.key === 'Escape' || e.key === 'Enter') {
      e.preventDefault();
      setMode(null);
    }
  };
  document.addEventListener('keydown', onKey);
  offs.push(() => document.removeEventListener('keydown', onKey));

  // The selectors follow the core: another kit (a pattern recalled it, undo),
  // another channel, the lists
  const follow = (fresh) => {
    if (state.mode === 'kit' && !busy) state.index = currentKitIndex();
    if (state.mode === 'inst' && !busy) syncInst();
    render(fresh);
  };
  for (const address of ['kit.current', 'kit.list', 'kit.edited', 'preset.name', 'drum_patch.library']) {
    offs.push(store.watch(address, () => follow(false), { now: false }));
  }
  // Another channel in INST: its drum patch shows at once (as BD-RC on a TR-8S)
  offs.push(store.watch('global.channel', () => follow(state.mode === 'inst'), { now: false }));
  offs.push(store.watch('kit.name', () => onBase(), { now: false }));

  // ------------------------------------------------------------ WRITE
  async function writeAs() {
    const kit = entries()[currentKitIndex()];
    let name = await sheets.prompt('Write the kit as', 'The sounds playing become a user kit of this name.',
      { value: writeName(kit), yes: 'write' });
    if (name === null) return;
    name = name.trim();
    if (!name) return;
    let event = await act('kit.write', { name });
    if (event.status !== 'done') return;
    if (event.result && event.result.exists && !event.result.saved) {
      const sure = await sheets.ask('Replace the kit?', `A user kit “${event.result.name}” exists: write over it?`,
        { yes: 'replace', no: 'cancel' });
      if (!sure) return;
      event = await act('kit.write', { name, overwrite: true });
      if (event.status !== 'done') return;
    }
    display.show('WRITE KIT', String(event.result.name || name).toUpperCase());
  }

  async function writeOver() {
    const name = String(value('kit.name') || '');
    const sure = await sheets.ask('Write the kit?', `“${name}” gets the sounds playing.`, { yes: 'write' });
    if (!sure) return;
    const event = await act('kit.write');
    if (event.status === 'done') display.show('WRITE KIT', name.toUpperCase());
  }

  async function programsToKits() {
    const event = await act('kit.from_programs');
    if (event.status === 'done') {
      const n = (event.result.kits || []).length;
      display.show('WRITE KITS', `${n} kit${n === 1 ? '' : 's'} from programs`);
    }
  }

  function writeItems() {
    const current = String(value('kit.current') ?? '');
    const user = current.startsWith('user:');
    const presets = getPresets();
    const hasPrograms = (value('program.occupied') || []).some(Boolean);
    return [
      ['write', null],
      [user ? `kit “${value('kit.name')}”` : 'kit (a new name: write as)', user ? writeOver : null],
      ['kit as…', writeAs],
      null,
      ['preset', () => presets.save()],
      ['preset as…', () => presets.saveAs()],
      ...(hasPrograms ? [null, ['programs as kits', programsToKits]] : []),
    ];
  }

  writeButton.addEventListener('click', (e) => {
    e.stopPropagation();
    const r = writeButton.getBoundingClientRect();
    ctx.openMenu(writeItems(), r.left, r.bottom + 2);
  });

  // ------------------------------------------------------------ patterns recall kits
  const kitName = (id) => {
    if (!id) return '';
    const kit = (value('kit.list') || []).find((k) => k.id === id);
    return kit ? kit.name : id.split(':').slice(1).join(':');
  };
  // A K on a pattern button that recalls a kit
  const kitMarks = {};
  for (const letter of PATTERNS) {
    const button = stage.querySelector(`.pbtn[data-pattern="${letter}"]`);
    if (!button) continue;
    const mark = document.createElement('small');
    mark.className = 'kitmark';
    mark.dataset.address = `pattern.${letter}.kit`;
    button.insertBefore(mark, button.querySelector('.len'));
    kitMarks[letter] = mark;
  }
  const showPatternKits = () => {
    for (const [letter, mark] of Object.entries(kitMarks)) {
      const id = value(`pattern.${letter}.kit`) || '';
      mark.textContent = id ? 'K' : '';
      mark.title = id ? `recalls kit ${kitName(id)}` : '';
      mark.parentElement.classList.toggle('has-kit', !!id);
    }
  };
  for (const letter of PATTERNS) offs.push(store.watch(`pattern.${letter}.kit`, showPatternKits));
  offs.push(store.watch('kit.list', showPatternKits, { now: false }));
  if (patterns) {
    offs.push(patterns.addMenuItems((letter) => {
      const id = value(`pattern.${letter}.kit`) || '';
      const current = String(value('kit.current') ?? '');
      const items = [];
      if (current && current !== id) {
        items.push([`recall kit “${kitName(current)}”`, () => {
          ctx.set(`pattern.${letter}.kit`, current);
          display.show(`PATTERN ${letter}`, `kit ${kitName(current).toUpperCase()}`);
        }]);
      }
      if (id) {
        items.push([`recall no kit (now “${kitName(id)}”)`, () => {
          ctx.set(`pattern.${letter}.kit`, '');
          display.show(`PATTERN ${letter}`, 'no kit');
        }]);
      }
      return items;
    }));
  }

  render();
  return {
    state,
    setMode,
    step,
    baseLines,
    /** Is a selector (KIT or INST) on? ◀ ▶ step it then. */
    get active() { return !!state.mode; },
    destroy() { offs.forEach((off) => off()); },
  };
}
