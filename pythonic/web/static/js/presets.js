// The PRESET buttons of the right column (map decisions #12, #13, #14):
// ◀ ▶ load the previous / next file of the preset folder (of the factory
// presets while one of them is loaded), PRESET opens the preset menu: the
// factory presets (read-only) and the folder's presets as in-panel lists (a
// click loads one), the recent files, and the commands: open / save / save
// as (native dialogs, a save asks before replacing a file; a factory preset
// is only saved as another file), reload the last preset, the preset
// clipboard, initialize, randomize all, restore the factory kits into
// programs 1-7 (asks first), the selected channel's drum patch
// (the edit rack's menu, also here so it works with the rack closed), every
// drum as WAV files, the preset folder, and the pages (PO-32, AI drum
// generator, setup) through panel.openPage.
//
// Bound addresses: preset.name, preset.path, preset.files, preset.clipboard,
// preset.factory, factory.presets, pref.preset_folder (a folder dialog sets
// it; preset.files follows) and pref.recent_files.

import { fileName, folderOf, samePath } from './files.js';

export const OPEN_FILTERS = ['Presets (*.mtpreset *.json)', 'All files (*)'];
export const SAVE_FILTERS = ['JSON presets (*.json)'];

/** Is the current preset the file `name` of the folder? */
function isCurrent(path, folder, name) {
  return !!path && fileName(path) === name && (!folder || samePath(folderOf(path), folder));
}

/**
 * The file of the preset folder `step` (-1 / +1) away from the current
 * preset, as tkinter's ◀ ▶ (no wrap): null at either end. Without a current
 * preset in the folder, ▶ gives the first file and ◀ nothing.
 */
export function neighbourFile(files, path, folder, step) {
  const list = Array.isArray(files) ? files : [];
  const i = list.findIndex((name) => isCurrent(path, folder, name));
  const next = i + step;
  if (i < 0 && step < 0) return null;
  return next >= 0 && next < list.length ? list[next] : null;
}

/** The preset can be saved over its own file (a JSON preset; .mtpreset files and factory presets are only read). */
export const canSaveInPlace = (path, factory = false) => !factory && typeof path === 'string' && /\.json$/i.test(path);

/** A preset's name in a list: the file name without its extension. */
export const presetLabel = (name) => String(name).replace(/\.(mtpreset|json)$/i, '');

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

export function mountPresets({ store, client, ctx, display, act, stage, slot, files, openPage, patchItems,
  ask = async () => true, beforeLoad = async () => true }) {
  const offs = [];
  const prev = slot('preset-prev');
  const next = slot('preset-next');
  const button = slot('preset');
  for (const [b, id] of [[prev, 'preset-prev'], [next, 'preset-next'], [button, 'preset-button']]) {
    b.disabled = false;
    b.classList.remove('slot');
    b.id = id;
  }
  const value = (address) => store.value(address);
  const folder = () => value('pref.preset_folder');
  const list = () => (Array.isArray(value('preset.files')) ? value('preset.files') : []);
  const factoryList = () => (Array.isArray(value('factory.presets')) ? value('factory.presets') : []);
  const isFactory = () => value('preset.factory') === true;
  /** The list ◀ ▶ walk: the factory presets while one is loaded, else the folder. */
  const walked = () => (isFactory() ? { files: factoryList(), folder: null } : { files: list(), folder: folder() });

  // ------------------------------------------------------------ load and save
  async function loadWith(args, label) {
    if (!(await beforeLoad())) return null;
    const r = await files.run('preset.load', args, { label, failTitle: 'Could not load the preset' });
    if (r) display.show(label, String(r.name || fileName(r.path)).toUpperCase());
    return r;
  }
  const load = (path, label = 'PRESET') => loadWith({ path }, label);
  const loadFactory = (name) => loadWith({ factory: name }, 'FACTORY PRESET');

  async function open() {
    const path = await client.openFile({ title: 'Open preset', filters: OPEN_FILTERS, folder: folder() });
    if (path) await load(path);
  }

  async function saveAs() {
    const name = `${value('preset.name') || 'preset'}.json`;
    const path = await client.saveFile({ title: 'Save preset as', filters: SAVE_FILTERS, folder: folder(),
      name, suffix: 'json' });
    if (path) await files.save('preset.save', { path }, { label: 'SAVE PRESET', failTitle: 'Could not save the preset' });
  }

  /** Save over the current file (the document you have open: no question); else save as. */
  async function save() {
    const path = value('preset.path');
    if (!canSaveInPlace(path, isFactory())) return saveAs();
    return files.save('preset.save', { path, overwrite: true }, { label: 'SAVE PRESET', failTitle: 'Could not save the preset' });
  }

  async function step(direction) {
    const { files: names, folder: within } = walked();
    const name = neighbourFile(names, value('preset.path'), within, direction);
    if (!name) {
      display.show('PRESET', names.length ? (direction < 0 ? 'first preset' : 'last preset') : 'no presets in the folder');
      return;
    }
    await (isFactory() ? loadFactory(name) : load(name));
  }

  async function restoreKits() {
    const current = Number(value('program.current')) || 1;
    const playing = current <= 7
      ? ` Program ${current} is one of them: the sounds playing change too (undo brings them back).` : '';
    const sure = await ask('Restore the factory kits?',
      `Programs 1-7 get the 505, 707, 808, 909, DMX, LM2 and TR-8 kits back; the other programs and the patterns stay.${playing}`,
      { yes: 'restore', no: 'cancel' });
    if (!sure) return;
    const event = await act('program.restore_factory');
    if (event.status === 'done') display.show('PROGRAMS 1-7', 'factory kits');
  }

  async function chooseFolder() {
    const path = await client.chooseFolder({ title: 'Select preset folder', folder: folder() });
    if (!path) return;
    client.set('pref.preset_folder', path);
    const errors = await client.flush();
    if (errors['pref.preset_folder']) {
      display.alert('PRESET FOLDER', errors['pref.preset_folder']);
      return;
    }
    display.show('PRESET FOLDER', fileName(path));
  }

  async function exportDrums() {
    const path = await client.chooseFolder({ title: 'Export every drum as WAV files into', folder: folder() });
    if (path) {
      await files.save('export.drum_wavs', { folder: path }, { label: 'DRUM WAVS',
        failTitle: 'Could not export the drums', done: (r) => `${(r.paths || []).length} files saved` });
    }
  }

  const quick = (verb, text) => () => act(verb).then((event) => {
    if (event.status === 'done') display.show('PRESET', text);
  });

  async function loadLast() {
    if (!(await beforeLoad())) return;
    const r = await files.run('preset.load_last', {}, { label: 'PRESET', failTitle: 'Could not load the preset' });
    if (r) display.show('PRESET', r.loaded ? String(r.name || '').toUpperCase() : 'no last preset');
  }

  // ------------------------------------------------------------ the menu
  /** The command column: [label, action] items, null = a separator. */
  function commands() {
    const path = value('preset.path');
    return [
      ['open preset…', open],
      [canSaveInPlace(path, isFactory()) ? `save “${fileName(path)}”` : 'save preset', save],
      ['save preset as…', saveAs],
      ['reload last preset', loadLast],
      null,
      ['copy preset', quick('preset.copy', 'copied')],
      ['cut preset', quick('preset.cut', 'cut')],
      ['paste preset', value('preset.clipboard') ? quick('preset.paste', 'pasted') : null],
      ['initialize preset', quick('preset.initialize', 'initialized')],
      ['randomize all', quick('preset.randomize_all', 'randomized')],
      ['restore factory kits…', restoreKits],
      null,
      ...patchItems(),
      ['export every drum as WAV…', exportDrums],
      null,
      ['preset folder…', chooseFolder],
      ['refresh list', () => act('preset.refresh').then(() => display.show('PRESET', 'list refreshed'))],
      null,
      ['transfer to PO-32…', () => openPage('po32', { tab: 'transfer' })],
      ['import from PO-32…', () => openPage('po32', { tab: 'import' })],
      ['AI drum generator…', () => openPage('ai')],
      ['setup…', () => openPage('setup')],
    ];
  }

  const item = (label, action, cls = '') => {
    const row = el('div', `it${action ? '' : ' dis'}${cls ? ` ${cls}` : ''}`, label);
    if (action) {
      row.addEventListener('pointerdown', (e) => e.stopPropagation());
      row.addEventListener('click', (e) => { e.stopPropagation(); ctx.closeMenus(); action(); });
    }
    return row;
  };

  function buildMenu() {
    const menu = el('div', 'px-menu preset-menu');
    menu.id = 'preset-menu';
    const side = el('div', 'pm-list');
    const head = el('div', 'pm-h');
    head.append(el('span', '', 'folder'), el('span', 'pm-folder', fileName(folder()) || '—'));
    const refresh = el('button', 'btn sq', '↻');
    refresh.type = 'button';
    refresh.title = 'refresh list';
    refresh.addEventListener('pointerdown', (e) => e.stopPropagation());
    refresh.addEventListener('click', (e) => { e.stopPropagation(); act('preset.refresh'); });
    head.append(refresh);
    head.title = folder() || '';
    const filesBox = el('div', 'pm-files');
    const path = value('preset.path');
    const factoryBox = el('div', 'pm-factory');
    for (const name of factoryList()) {
      factoryBox.append(item(presetLabel(name), () => loadFactory(name),
        isFactory() && fileName(path) === name ? 'cur' : ''));
    }
    const factoryHead = el('div', 'pm-h', 'factory');
    factoryHead.append(el('span', 'pm-lock', 'read-only'));
    for (const name of list()) {
      filesBox.append(item(presetLabel(name), () => load(name),
        !isFactory() && isCurrent(path, folder(), name) ? 'cur' : ''));
    }
    if (!list().length) filesBox.append(el('div', 'pm-empty', 'no presets in this folder'));
    const recent = el('div', 'pm-recent');
    for (const p of (Array.isArray(value('pref.recent_files')) ? value('pref.recent_files') : [])) {
      const row = item(fileName(p), () => load(p), samePath(p, path) ? 'cur' : '');
      row.title = p;
      row.append(el('small', '', fileName(folderOf(p))));
      recent.append(row);
    }
    if (!recent.childElementCount) recent.append(el('div', 'pm-empty', 'none yet'));
    if (factoryList().length) side.append(factoryHead, factoryBox);
    side.append(head, filesBox, el('div', 'pm-h', 'recent'), recent);
    const cmds = el('div', 'pm-cmds');
    for (const entry of commands()) {
      if (!entry) { cmds.append(el('hr')); continue; }
      cmds.append(item(entry[0], entry[1]));
    }
    menu.append(side, cmds);
    return menu;
  }

  function openMenu() {
    ctx.closeMenus();
    const menu = buildMenu();
    stage.append(menu);
    // Under the PRESET button, right-aligned with it, inside the stage
    const scale = Number(stage.dataset.scale) || 1;
    const s = stage.getBoundingClientRect();
    const r = button.getBoundingClientRect();
    const w = menu.offsetWidth;
    const h = menu.offsetHeight;
    const x = (r.right - s.left) / scale - w;
    const y = (r.bottom - s.top) / scale + 4;
    menu.style.left = `${Math.max(4, Math.min(x, stage.offsetWidth - w - 4))}px`;
    menu.style.top = `${Math.max(4, Math.min(y, stage.offsetHeight - h - 4))}px`;
    menu.querySelector('.pm-files .cur, .pm-factory .cur')?.scrollIntoView({ block: 'nearest' });
    return menu;
  }
  const isOpen = () => !!stage.querySelector(':scope > #preset-menu');
  // An open menu follows the folder, the recent files and the clipboard
  const refreshOpen = () => { if (isOpen()) openMenu(); };

  // ------------------------------------------------------------ buttons
  const showButtons = () => {
    const { files: names, folder: within } = walked();
    prev.disabled = !neighbourFile(names, value('preset.path'), within, -1);
    next.disabled = !neighbourFile(names, value('preset.path'), within, 1);
  };
  for (const address of ['preset.files', 'preset.path', 'pref.preset_folder', 'preset.factory', 'factory.presets']) {
    offs.push(store.watch(address, showButtons));
  }
  for (const address of ['preset.files', 'pref.recent_files', 'preset.clipboard', 'pref.preset_folder',
    'preset.factory', 'factory.presets']) {
    offs.push(store.watch(address, refreshOpen, { now: false }));
  }
  prev.addEventListener('click', () => step(-1));
  next.addEventListener('click', () => step(1));
  // PRESET toggles the menu: a press closes any menu (controls.js) before
  // the click, so note whether this one was open before that
  let wasOpen = false;
  const notePress = (e) => { if (e.target === button) wasOpen = isOpen(); };
  document.addEventListener('pointerdown', notePress, true);
  offs.push(() => document.removeEventListener('pointerdown', notePress, true));
  button.addEventListener('click', (e) => {
    e.stopPropagation();
    if (wasOpen || isOpen()) ctx.closeMenus(); else openMenu();
    wasOpen = false;
  });

  return {
    openMenu,
    load,
    loadFactory,
    restoreKits,
    save,
    saveAs,
    open,
    chooseFolder,
    exportDrums,
    destroy() { offs.forEach((off) => off()); },
  };
}
