// Placeholder panel of the first web slice: the stage with the wordmark,
// the green display, the master tempo and START/STOP beside the 16 pads,
// wired end to end through the store and the core client. Later slices
// replace it with the real face.
//
// Controls carry data-address (the address they show and edit) or
// data-verb (the verb they start); the parity guard reads them.

const PATTERNS = 'ABCDEFGHIJKL';
const TEMPO = 'global.tempo';

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') node.className = v; else if (k === 'text') node.textContent = v; else node.setAttribute(k, v);
  }
  for (const child of children) node.append(child);
  return node;
}

/** The addresses the panel shows (main.js reads their values and metadata). */
export const PANEL_ADDRESSES = [TEMPO];

/** Build the panel in the stage; meta maps addresses to describe() metadata. */
export function mountPanel(stage, { store, client, meta = {} }) {
  const display1 = el('div', { class: 'line', id: 'display-line1' });
  const display2 = el('div', { class: 'line', id: 'display-line2', text: 'PYTHONIC' });
  const tempoValue = el('div', { class: 'seg7', id: 'tempo-value', text: '---' });
  const tempoDown = el('button', { class: 'btn sq', id: 'tempo-down', title: 'tempo -1', text: '−' });
  const tempoUp = el('button', { class: 'btn sq', id: 'tempo-up', title: 'tempo +1', text: '+' });
  const tempo = el('div', { class: 'tempo', 'data-address': TEMPO }, [
    tempoValue, el('div', { class: 'row' }, [tempoDown, el('span', { class: 'lbl', text: 'tempo' }), tempoUp]),
  ]);
  const startStop = el('button', { class: 'ss', id: 'start-stop', 'data-verb': 'transport.toggle', title: 'start / stop' });
  const pads = [];
  const numbers = [];
  for (let i = 0; i < 16; i += 1) {
    pads.push(el('div', { class: `pad g${Math.floor(i / 4) + 1}`, 'data-step': String(i + 1) }));
    numbers.push(el('div', { text: String(i + 1) }));
  }

  stage.append(
    el('div', { class: 'wordmark' }, ['PYTHON', el('b', { text: 'IC' })]),
    el('div', { class: 'display', id: 'display' }, [display1, display2]),
    tempo,
    el('div', { class: 'transport' }, [startStop, el('span', { class: 'wtab', text: 'start / stop' })]),
    el('div', { class: 'steps' }, [el('div', { class: 'pnums' }, numbers), el('div', { class: 'pads' }, pads)]),
  );

  // Display: pattern and step on line 1; the touched control replaces line 2 for 1.5 s
  let touchTimer = null;
  const touched = (text) => {
    display2.textContent = text;
    clearTimeout(touchTimer);
    touchTimer = setTimeout(() => { display2.textContent = 'PYTHONIC'; }, 1500);
  };

  const offs = [];
  offs.push(store.watch(TEMPO, (bpm) => { tempoValue.textContent = String(bpm); }));
  offs.push(store.watchReadout('transport', (t) => {
    startStop.classList.toggle('on', !!t.playing);
    const step = t.playing ? t.position % 16 : -1;
    pads.forEach((pad, i) => pad.classList.toggle('ph', i === step));
    const letter = PATTERNS[t.playing ? t.playing_pattern : t.selected_pattern] || '?';
    display1.textContent = `PATTERN ${letter}${t.playing ? `  STEP ${String(t.position + 1).padStart(2, '0')}` : ''}`;
  }));

  const nudgeTempo = (delta, burst) => {
    const desc = meta[TEMPO] || { minimum: 1, maximum: 300 };
    const next = Math.min(desc.maximum, Math.max(desc.minimum, (store.value(TEMPO) ?? 120) + delta));
    store.assume(TEMPO, next);
    client.set(TEMPO, next, { burst });
    touched(`TEMPO ${next} BPM`);
  };
  tempoDown.addEventListener('click', () => nudgeTempo(-1, false));
  tempoUp.addEventListener('click', () => nudgeTempo(1, false));
  tempo.addEventListener('wheel', (e) => {
    e.preventDefault();
    if (e.deltaY) nudgeTempo(e.deltaY < 0 ? 1 : -1, true);
  }, { passive: false });
  startStop.addEventListener('click', () => { client.act(startStop.dataset.verb); });

  return { destroy() { offs.forEach((off) => off()); stage.replaceChildren(); } };
}
