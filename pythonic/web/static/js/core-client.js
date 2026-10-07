// The app core as the page sees it: addresses, verbs and frames over a
// bridge (bridge.js). Sets are coalesced per animation frame (latest value
// wins per address, one bridge call per frame); verbs resolve with their
// poll event; file dialogs resolve with the chosen path or null.

const defaultSchedule = (fn) => (typeof requestAnimationFrame === 'function'
  ? requestAnimationFrame(() => fn()) : setTimeout(fn, 0));

export function createCoreClient(bridge, { schedule = defaultSchedule } = {}) {
  let pending = new Map(); // address -> change
  let flushQueued = false;
  let flushing = Promise.resolve();
  const waiting = new Map(); // action id -> resolve
  const early = new Map(); // events that arrived before their id came back
  const dialogs = new Map(); // dialog id -> resolve
  const earlyDialogs = new Map();
  const frameListeners = new Set();

  bridge.onFrame((frame) => {
    for (const event of frame.events || []) {
      if (event.id == null) continue;
      const resolve = waiting.get(event.id);
      if (resolve) { waiting.delete(event.id); resolve(event); } else {
        early.set(event.id, event);
        if (early.size > 64) early.delete(early.keys().next().value);
      }
    }
    for (const fn of [...frameListeners]) fn(frame);
  });
  bridge.onDialog(({ id, path }) => {
    const resolve = dialogs.get(id);
    if (resolve) { dialogs.delete(id); resolve(path); } else earlyDialogs.set(id, path);
  });
  // Frames only carry what changed: ask for every readout now that we listen
  bridge.call('resync', null);

  function flush() {
    flushQueued = false;
    if (!pending.size) return flushing;
    const changes = [...pending.values()];
    pending = new Map();
    flushing = bridge.call('set', changes).then((reply) => {
      const errors = (reply && reply.errors) || {};
      for (const [address, message] of Object.entries(errors)) {
        console.warn(`set ${address}: ${message}`);
      }
      return errors;
    });
    return flushing;
  }

  function dialog(mode, options) {
    return bridge.call('fileDialog', { ...options, mode }).then(({ id }) => {
      if (earlyDialogs.has(id)) { const p = earlyDialogs.get(id); earlyDialogs.delete(id); return p; }
      return new Promise((resolve) => dialogs.set(id, resolve));
    });
  }

  return {
    bridge,
    /** Values of addresses: Promise<{address: value}> (unknown ones left out, with a warning). */
    async get(addresses) {
      const reply = await bridge.call('get', [].concat(addresses));
      for (const [a, m] of Object.entries(reply.errors || {})) console.warn(`get ${a}: ${m}`);
      return reply.values;
    },
    /** Queue a change; sent with the others of this animation frame. */
    set(address, value, { burst = false, editAll } = {}) {
      const change = { address, value };
      if (burst) change.burst = true;
      if (editAll !== undefined) change.edit_all = editAll;
      pending.delete(address); // keep the order of the latest changes
      pending.set(address, change);
      if (!flushQueued) { flushQueued = true; schedule(flush); }
    },
    /** Send the queued changes now; Promise<{address: error}>. */
    flush,
    /** Start a verb; Promise of its poll event ({status, result} or {status: 'error', error}). */
    async act(verb, args = {}) {
      await flush();
      const { id } = await bridge.call('act', { verb, args });
      if (early.has(id)) { const e = early.get(id); early.delete(id); return e; }
      return new Promise((resolve) => waiting.set(id, resolve));
    },
    /** Metadata of addresses (a list) or of every registered address under a prefix. */
    async describe(query) {
      const reply = await bridge.call('describe', query);
      return reply.addresses;
    },
    /** Bracket a drag: the sets in between are one undo step. */
    beginGesture() { flush(); return bridge.call('gesture', 'begin'); },
    endGesture() { flush(); return bridge.call('gesture', 'end'); },
    openFile: (options = {}) => dialog('open', options),
    saveFile: (options = {}) => dialog('save', options),
    chooseFolder: (options = {}) => dialog('folder', options),
    /** Every frame (poll result) after the client has resolved its events. */
    onFrame(fn) { frameListeners.add(fn); return () => frameListeners.delete(fn); },
  };
}
