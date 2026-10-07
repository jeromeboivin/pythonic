// The pattern controls of the left column (map decisions #7, #12, #13): the
// 12 pattern buttons A-L (selected lit, playing red, queued blinking, empty
// dim, chains as links, the length when it is not 16), chain ◀◀ ▶▶, the
// pattern menu, and copy / paste of the selected channel's lane.
//
// A click selects the pattern (`pattern.select`: while playing it is queued);
// right-click opens the menu of that pattern. The menu runs the pattern ops
// as verbs on the selected pattern (or the one right-clicked) and shows the
// result on the display; failures alert there (panel act). Later slices add
// entries (export) with patterns.addMenuItems((letter, index) => items).

import { chainOf, chainText, PATTERNS, patternStates } from './steps-logic.js';

const OPS = [
  [['cut', 'pattern.cut'], ['copy', 'pattern.copy'], ['paste', 'pattern.paste'],
    ['exchange', 'pattern.exchange'], ['clear', 'pattern.clear']],
  [['shift left', 'pattern.shift_left'], ['shift right', 'pattern.shift_right'],
    ['reverse', 'pattern.reverse']],
  [['randomize', 'pattern.randomize'], ['alter', 'pattern.alter'],
    ['randomize accents + fills', 'pattern.randomize_accents_fills']],
];

/** What the display says after a pattern op (null: the op's name). */
function resultText(verb, result) {
  if (verb === 'pattern.paste' && result && result.pasted === false) return 'nothing to paste';
  if (verb === 'pattern.exchange' && result && result.exchanged === false) return 'nothing to exchange';
  if (verb === 'pattern.paste_lane' && result && result.pasted === false) return 'nothing to paste';
  return null;
}

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

export function mountPatterns({ store, ctx, display, act, slot, selected }) {
  const offs = [];
  const extraItems = [];
  const root = slot('patterns');
  root.classList.remove('slot-mark');
  const grid = el('div', 'pgrid');
  grid.dataset.address = 'pattern.selected';
  const buttons = [...PATTERNS].map((letter, index) => {
    const wrap = el('div', 'pat');
    wrap.dataset.address = `pattern.${letter}.chained`;
    const b = el('button', 'btn sq pbtn', letter);
    b.type = 'button';
    b.dataset.verb = 'pattern.select';
    b.dataset.pattern = letter;
    b.dataset.address = `pattern.${letter}.empty`;
    const len = el('small', 'len');
    len.dataset.address = `pattern.${letter}.length`;
    b.append(len);
    b.addEventListener('click', () => {
      const t = store.readout('transport') || {};
      const next = t.playing && t.playing_pattern !== index;
      act('pattern.select', { pattern: letter });
      display.show(`PATTERN ${letter}`, next ? 'plays next' : 'selected');
    });
    b.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      openMenu(index, e.clientX, e.clientY);
    });
    wrap.append(b);
    grid.append(wrap);
    return b;
  });

  const tools = el('div', 'ptools');
  const tool = (label, verb, id) => {
    const b = el('button', 'btn sq', label);
    b.type = 'button';
    b.id = id;
    if (verb) b.dataset.verb = verb;
    tools.append(b);
    return b;
  };
  const prev = tool('◀◀', 'pattern.chain_prev', 'chain-prev');
  const next = tool('▶▶', 'pattern.chain_next', 'chain-next');
  const menu = tool('menu', null, 'pattern-menu');
  const copy = tool('copy', 'pattern.copy_lane', 'lane-copy');
  const paste = tool('paste', 'pattern.paste_lane', 'lane-paste');
  root.replaceChildren(grid, tools);

  const letterOf = (index) => PATTERNS[index];
  const channel = () => store.value('global.channel') || 1;
  const chained = () => [...PATTERNS].map((l) => !!store.value(`pattern.${l}.chained`));

  // ------------------------------------------------------------ rendering
  function render() {
    const empty = [...PATTERNS].map((l) => store.value(`pattern.${l}.empty`) !== false);
    const t = { ...(store.readout('transport') || {}), selected_pattern: selected() };
    patternStates(t, empty, chained()).forEach((s, i) => {
      const b = buttons[i];
      b.classList.toggle('on', s.selected);
      b.classList.toggle('playing', s.playing);
      b.classList.toggle('queued', s.queued);
      b.classList.toggle('empty', s.empty);
      b.classList.toggle('chain-in', s.chainIn);
      b.classList.toggle('chain-out', s.chainOut);
      const n = store.value(`pattern.${s.letter}.length`);
      b.lastChild.textContent = Number.isInteger(n) && n !== 16 ? String(n) : '';
    });
    const index = selected();
    prev.disabled = index === 0;
    next.disabled = index === PATTERNS.length - 1;
  }
  for (const l of PATTERNS) {
    for (const a of [`pattern.${l}.empty`, `pattern.${l}.chained`, `pattern.${l}.length`]) {
      offs.push(store.watch(a, render, { now: false }));
    }
  }
  offs.push(store.watch('pattern.selected', render, { now: false }));
  offs.push(store.watch('ai.available', () => {}, { now: false })); // read by the menu
  offs.push(store.watchReadout('transport', render));

  // ------------------------------------------------------------ verbs
  /** Run a pattern verb and show its result on the display. */
  function run(verb, args, label, done = null) {
    return act(verb, args).then((event) => {
      if (event.status !== 'done') return event;
      const text = resultText(verb, event.result);
      if (done) done(event.result, text);
      else display.show(label, text || verb.split('.').pop().replace(/_/g, ' '));
      return event;
    });
  }

  const showChain = (index) => display.show('CHAIN', chainText(chainOf(index, chained())));
  prev.addEventListener('click', () => {
    const index = selected();
    run('pattern.chain_prev', { pattern: letterOf(index) }, '', () => showChain(index));
  });
  next.addEventListener('click', () => {
    const index = selected();
    run('pattern.chain_next', { pattern: letterOf(index) }, '', () => showChain(index));
  });
  copy.addEventListener('click', () => {
    const ch = channel();
    run('pattern.copy_lane', { pattern: letterOf(selected()), channel: ch }, `LANE CH${ch}`,
      () => display.show(`LANE CH${ch}`, 'copied'));
  });
  paste.addEventListener('click', () => {
    const ch = channel();
    run('pattern.paste_lane', { pattern: letterOf(selected()), channel: ch }, `LANE CH${ch}`,
      (_r, text) => display.show(`LANE CH${ch}`, text || 'pasted'));
  });
  menu.addEventListener('click', (e) => {
    e.stopPropagation();
    const r = menu.getBoundingClientRect();
    openMenu(selected(), r.left, r.bottom + 2);
  });

  /** The pattern menu of a pattern (the selected one from MENU, any from a right-click). */
  function openMenu(index, x, y) {
    const letter = letterOf(index);
    const t = store.readout('transport') || {};
    const ch = channel();
    const label = `PATTERN ${letter}`;
    const items = [[label, null]];
    if (t.playing && t.playing_pattern !== index) {
      items.push(['play next (queue)', () => run('pattern.queue', { pattern: letter }, label,
        () => display.show(label, 'plays next'))]);
    }
    for (const group of OPS) {
      items.push(null);
      for (const [text, verb] of group) items.push([text, () => run(verb, { pattern: letter }, label)]);
    }
    items.push(null);
    const ai = store.value('ai.available') ? (args, what) => () => run('ai.randomize_pattern', args, label,
      () => display.show(label, `${what} randomized (AI)`)) : () => null;
    items.push(['randomize (AI)', ai({ pattern: letter }, 'pattern')]);
    items.push([`randomize ch${ch} (AI)`, ai({ pattern: letter, channel: ch }, `ch${ch}`)]);
    items.push(null);
    items.push([`copy lane ch${ch}`, () => run('pattern.copy_lane', { pattern: letter, channel: ch }, label,
      () => display.show(`LANE CH${ch}`, 'copied'))]);
    items.push([`paste lane ch${ch}`, () => run('pattern.paste_lane', { pattern: letter, channel: ch }, label,
      (_r, text) => display.show(`LANE CH${ch}`, text || 'pasted'))]);
    items.push(['clear all chains', () => run('pattern.chain_clear', {}, 'CHAIN', () => display.show('CHAIN', 'all cleared'))]);
    for (const fn of extraItems) {
      const more = fn(letter, index) || [];
      if (more.length) items.push(null, ...more);
    }
    ctx.openMenu(items, x, y, { className: 'pmenu' });
  }

  render();
  return {
    render,
    openMenu,
    /** Add entries to the pattern menu: fn(letter, index) -> [[label, action], ...]. */
    addMenuItems(fn) { extraItems.push(fn); return () => extraItems.splice(extraItems.indexOf(fn), 1); },
    destroy() { offs.forEach((off) => off()); },
  };
}
