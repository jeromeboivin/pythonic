// File verbs of the panel (map decisions #13, #14): the page picks a path
// with a native dialog (client.openFile / saveFile / chooseFolder), the core
// reads or writes it. A failure shows on the display and on the alert sheet
// (it needs reading); progress (exports) shows on the display; a save the
// core refused because the file exists asks on the alert sheet, then saves
// again with overwrite=true.
//
//   const files = createFileFlows({ client, display, sheets });
//   const result = await files.run(verb, args, { label, failTitle });   // result or null
//   const saved = await files.save(verb, args, { label, failTitle });   // result or null

/** The last part of a path (either separator). */
export const fileName = (path) => String(path || '').split(/[\\/]/).pop();

/** The folder of a path ('' without one). */
export function folderOf(path) {
  const p = String(path || '');
  const i = Math.max(p.lastIndexOf('/'), p.lastIndexOf('\\'));
  return i > 0 ? p.slice(0, i) : '';
}

const norm = (p) => String(p || '').replace(/\\/g, '/').replace(/\/+$/, '');

/** Two paths name the same place (separators and a trailing one aside). */
export const samePath = (a, b) => !!a && !!b && norm(a) === norm(b);

/** A 0..1 fraction as a percentage for the display. */
export const percent = (fraction) => `${Math.round(Math.max(0, Math.min(1, Number(fraction) || 0)) * 100)} %`;

/** The question asked before replacing files: [title, text]. */
export function replaceQuestion(result) {
  if (Array.isArray(result.paths) && result.paths.length) {
    const names = result.paths.map(fileName);
    const n = names.length;
    return [`Replace ${n} file${n > 1 ? 's' : ''}?`,
      `${names.join(', ')} already exist${n > 1 ? '' : 's'} in ${result.folder || folderOf(result.paths[0])}. `
      + 'Replacing overwrites them.'];
  }
  return [`Replace “${fileName(result.path)}”?`,
    `A file with this name already exists in ${folderOf(result.path) || 'this folder'}. Replacing it overwrites it.`];
}

export function createFileFlows({ client, display, sheets }) {
  /** Run a file verb: its result, or null when it failed (shown) or was cancelled. */
  async function run(verb, args, { label = verb.toUpperCase(), failTitle = 'Something went wrong' } = {}) {
    const event = await client.act(verb, args, {
      onProgress: (fraction) => display.show(label, percent(fraction), 3000),
    });
    if (event.status === 'error') {
      display.alert(label, event.error);
      sheets.alert({ title: failTitle, text: event.error, tone: 'error' });
      return null;
    }
    if (event.status !== 'done') return null;
    return event.result || {};
  }

  /** Run a save verb; when the file exists, ask and save again with overwrite. */
  async function save(verb, args, options = {}) {
    const { label = verb.toUpperCase(), done = 'saved' } = options;
    const result = await run(verb, args, options);
    if (!result) return null;
    if (result.saved === false && result.exists) {
      const [title, text] = replaceQuestion(result);
      const yes = await sheets.ask(title, text, { yes: 'replace', no: 'cancel' });
      if (!yes) {
        display.show(label, 'not saved');
        return null;
      }
      return save(verb, { ...args, overwrite: true }, options);
    }
    if (done) display.show(label, typeof done === 'function' ? done(result) : done);
    return result;
  }

  return { run, save };
}
