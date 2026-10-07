// Pattern exports from the pattern menu (map decision #13): export to MIDI
// goes straight to the native save dialog; export to audio first opens a
// popover on the pattern MENU button with the tail choice (cut, add 2 s,
// loop +1 pass), then the save dialog. Both ask before replacing a file
// (files.js); the audio render's progress shows on the display.

import { fileName } from './files.js';

/** The tails of export.wav: [value, button text, note]. */
export const TAILS = [
  ['cut', 'cut', 'ends where the pattern loops'],
  ['append', 'add 2 s', 'adds 2 s for the last hits to ring out'],
  ['loop', 'loop +1 pass', 'plays one more pass, cut where it loops'],
];

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

export function mountExports({ client, ctx, display, stage, patterns, files }) {
  let tail = 'cut'; // the last choice, for this session

  async function exportMidi(letter) {
    const path = await client.saveFile({ title: `Export pattern ${letter} to MIDI`,
      filters: ['MIDI files (*.mid)'], name: `pythonic_pattern_${letter}.mid`, suffix: 'mid' });
    if (!path) return null;
    return files.save('export.midi', { path, pattern: letter }, { label: `PATTERN ${letter} MIDI`,
      failTitle: 'Could not export the MIDI file', done: (r) => `saved ${fileName(r.path)}` });
  }

  async function exportAudio(letter, chosen = tail) {
    tail = chosen;
    const path = await client.saveFile({ title: `Export pattern ${letter} to audio`,
      filters: ['WAV audio (*.wav)'], name: `pythonic_pattern_${letter}.wav`, suffix: 'wav' });
    if (!path) return null;
    display.show(`PATTERN ${letter} WAV`, 'rendering…', 3000);
    return files.save('export.wav', { path, pattern: letter, tail: chosen }, { label: `PATTERN ${letter} WAV`,
      failTitle: 'Could not export the audio file', done: (r) => `saved ${fileName(r.path)}` });
  }

  /** The tail popover, on the pattern MENU button. */
  function openTailPopover(letter) {
    ctx.closeMenus();
    const pop = el('div', 'px-menu tail-pop');
    pop.id = 'tail-pop';
    pop.addEventListener('pointerdown', (e) => e.stopPropagation());
    const opts = el('div', 'tp-opts');
    const note = el('div', 'tp-note');
    let choice = tail;
    const buttons = TAILS.map(([value, text, about]) => {
      const b = el('button', 'btn sq', text);
      b.type = 'button';
      b.dataset.tail = value;
      b.addEventListener('click', (e) => {
        e.stopPropagation();
        choice = value;
        for (const other of buttons) other.classList.toggle('on', other === b);
        note.textContent = about;
      });
      return b;
    });
    opts.append(...buttons);
    buttons.find((b) => b.dataset.tail === choice).click();
    const go = el('div', 'tp-go');
    const save = el('button', 'btn sq on', 'save wav…');
    save.type = 'button';
    save.id = 'tail-save';
    save.addEventListener('click', (e) => {
      e.stopPropagation();
      ctx.closeMenus();
      exportAudio(letter, choice);
    });
    go.append(save);
    pop.append(el('div', 'tp-title', `export pattern ${letter} to audio · tail`), opts, note, go);
    stage.append(pop);
    const anchor = stage.querySelector('#pattern-menu');
    const scale = Number(stage.dataset.scale) || 1;
    const s = stage.getBoundingClientRect();
    const r = anchor ? anchor.getBoundingClientRect() : s;
    const x = (r.left - s.left) / scale;
    const y = (r.bottom - s.top) / scale + 4;
    pop.style.left = `${Math.max(4, Math.min(x, stage.offsetWidth - pop.offsetWidth - 4))}px`;
    pop.style.top = `${Math.max(4, Math.min(y, stage.offsetHeight - pop.offsetHeight - 4))}px`;
    return pop;
  }

  const off = patterns.addMenuItems((letter) => [
    ['export to MIDI…', () => exportMidi(letter)],
    ['export to audio…', () => openTailPopover(letter)],
  ]);

  return {
    exportMidi,
    exportAudio,
    openTailPopover,
    get tail() { return tail; },
    destroy() { off(); },
  };
}
