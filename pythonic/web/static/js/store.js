// The page's copy of core state, rebuilt from get() and the poll frames.
// Pure module (no DOM, no bridge): values by address, the readouts of the
// latest frame (transport, modulation, audio, midi, po32) and the action events.

export const READOUTS = ['transport', 'modulation', 'audio', 'midi', 'po32'];

const same = (a, b) => a === b || JSON.stringify(a) === JSON.stringify(b);

export function createStore() {
  const values = new Map();
  const watchers = new Map(); // address -> Set(fn)
  const readouts = {};
  const readoutWatchers = new Map(); // name -> Set(fn)
  const eventListeners = new Set();
  let version = 0;

  function notify(address, value) {
    const set = watchers.get(address);
    if (set) for (const fn of [...set]) fn(value, address);
  }

  function put(address, value) {
    if (values.has(address) && same(values.get(address), value)) return;
    values.set(address, value);
    notify(address, value);
  }

  return {
    get version() { return version; },
    has: (address) => values.has(address),
    value: (address) => values.get(address),
    /** Values read with get(): {address: value}. */
    seed(entries) { for (const [a, v] of Object.entries(entries)) put(a, v); },
    /** Local echo of a set the core has not reported yet. */
    assume(address, value) { put(address, value); },
    /** Apply one frame (a poll result): changes by address, readouts, events. */
    apply(frame) {
      if (frame.version !== undefined) version = frame.version;
      for (const [a, v] of Object.entries(frame.changes || {})) put(a, v);
      for (const name of READOUTS) {
        if (!(name in frame) || same(readouts[name], frame[name])) continue;
        readouts[name] = frame[name];
        const set = readoutWatchers.get(name);
        if (set) for (const fn of [...set]) fn(frame[name]);
      }
      for (const event of frame.events || []) for (const fn of [...eventListeners]) fn(event);
    },
    /** Call fn(value, address) on every change of an address (and now, when known). */
    watch(address, fn, { now = true } = {}) {
      if (!watchers.has(address)) watchers.set(address, new Set());
      watchers.get(address).add(fn);
      if (now && values.has(address)) fn(values.get(address), address);
      return () => watchers.get(address).delete(fn);
    },
    /** Addresses something watches (what the page needs from get()). */
    watched: () => [...watchers.keys()].filter((a) => watchers.get(a).size > 0),
    readout: (name) => readouts[name],
    /** Call fn(readout) whenever a readout changes (and now, when known). */
    watchReadout(name, fn, { now = true } = {}) {
      if (!readoutWatchers.has(name)) readoutWatchers.set(name, new Set());
      readoutWatchers.get(name).add(fn);
      if (now && readouts[name] !== undefined) fn(readouts[name]);
      return () => readoutWatchers.get(name).delete(fn);
    },
    /** Call fn(event) for every action event a frame carries. */
    onEvent(fn) { eventListeners.add(fn); return () => eventListeners.delete(fn); },
  };
}
