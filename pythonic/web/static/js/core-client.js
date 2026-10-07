// The app core as the page sees it: addresses, verbs and frames over a
// bridge (bridge.js). Sets are coalesced per animation frame (latest value
// wins per address, one bridge call per frame); verbs resolve with their
// poll event; file dialogs resolve with the chosen path or null.
//
// Error events nobody waits for (errors not tied to an action: audio callback,
// stalled stream, MIDI, AI worker; or an action whose caller is gone) go to
// the onUnclaimedError listeners (the panel's global error handler).

const defaultSchedule = (fn) => (typeof requestAnimationFrame === 'function'
  ? requestAnimationFrame(() => fn()) : setTimeout(fn, 0));

export function createCoreClient(bridge, { schedule = defaultSchedule } = {}) {
  let pending = new Map(); // address -> change
  let flushQueued = false;
  let flushing = Promise.resolve();
  const waiting = new Map(); // action id -> resolve
  const early = new Map(); // id -> {event, frames}: events that arrived before their id came back
  const progress = new Map(); // action id -> fn(fraction, event)
  const unclaimed = new Set(); // fn(event): error events nobody waits for
  let asking = 0; // act calls waiting for their id
  const dialogs = new Map(); // dialog id -> resolve
  const earlyDialogs = new Map();
  const frameListeners = new Set();

  const reportUnclaimed = (event) => { for (const fn of [...unclaimed]) fn(event); };
  bridge.onFrame((frame) => {
    for (const event of frame.events || []) {
      if (event.id == null) { // not tied to an action
        if (event.status === 'error') reportUnclaimed(event);
        continue;
      }
      if (event.status === 'progress') { // not an end
        const fn = progress.get(event.id);
        if (fn) fn(event.progress, event);
        continue;
      }
      progress.delete(event.id);
      const resolve = waiting.get(event.id);
      if (resolve) { waiting.delete(event.id); resolve(event); } else {
        early.set(event.id, { event, frames: 0 });
        if (early.size > 64) early.delete(early.keys().next().value);
      }
    }
    // An error still unclaimed two frames later, with no act waiting for
    // its id, belongs to nobody
    if (!asking) {
      for (const [id, entry] of [...early]) {
        entry.frames += 1;
        if (entry.frames > 2) {
          early.delete(id);
          if (entry.event.status === 'error') reportUnclaimed(entry.event);
        }
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
    /** Start a verb; Promise of its poll event ({status, result} or {status: 'error', error}).
     * `onProgress(fraction, event)` gets its progress events (exports). */
    async act(verb, args = {}, { onProgress = null } = {}) {
      await flush();
      asking += 1;
      let id;
      try {
        ({ id } = await bridge.call('act', { verb, args }));
      } finally {
        asking -= 1;
      }
      if (early.has(id)) { const { event } = early.get(id); early.delete(id); return event; }
      if (onProgress) progress.set(id, onProgress);
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
    /** The stage changed height (the edit rack drawer): the window follows by
     * the difference at the panel's scale (Python side: PanelWindow.fit_stage_height). */
    resizeWindow: (from, to) => bridge.call('resizeWindow', { from, to }),
    openFile: (options = {}) => dialog('open', options),
    saveFile: (options = {}) => dialog('save', options),
    chooseFolder: (options = {}) => dialog('folder', options),
    /** Hit a channel (1..8) now, as a pad would (the core's trigger). */
    trigger: (channel, velocity = 127) => bridge.call('trigger', { channel, velocity }),
    /** Error events nobody waits for: fn(event) -> off. */
    onUnclaimedError(fn) { unclaimed.add(fn); return () => unclaimed.delete(fn); },
    /** Every frame (poll result) after the client has resolved its events. */
    onFrame(fn) { frameListeners.add(fn); return () => frameListeners.delete(fn); },
  };
}
