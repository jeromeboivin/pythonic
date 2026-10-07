// The transport to the app core, kept behind one small interface so it can
// be swapped (QWebChannel in the app, a fake in specs and in a plain browser).
//
// A bridge is { call(slot, payload) -> Promise<reply>, onFrame(fn) -> off,
// onDialog(fn) -> off }: payloads and replies are plain values (JSON text
// stays inside this module); frames are poll results, dialog results are
// { id, path }.
//
// Slots (see pythonic/web/bridge.py): get(addresses), set(changes),
// act({verb, args}), describe(prefix | addresses), gesture('begin'|'end'),
// resync(null), fileDialog(options), resizeWindow({from, to}),
// trigger({channel, velocity}).

/** Connect to the Python bridge over QWebChannel; null outside the app. */
export function connectBridge(scope = globalThis) {
  if (typeof scope.qt === 'undefined' || !scope.qt.webChannelTransport || !scope.QWebChannel) {
    return Promise.resolve(null);
  }
  return new Promise((resolve) => {
    new scope.QWebChannel(scope.qt.webChannelTransport, (channel) => {
      resolve(webChannelBridge(channel.objects.bridge));
    });
  });
}

function listeners() {
  const set = new Set();
  return {
    add(fn) { set.add(fn); return () => set.delete(fn); },
    emit(value) { for (const fn of [...set]) fn(value); },
  };
}

/** Wrap the published Python object. */
export function webChannelBridge(remote) {
  const frames = listeners();
  const dialogs = listeners();
  remote.frame.connect((text) => frames.emit(JSON.parse(text)));
  remote.dialog.connect((text) => dialogs.emit(JSON.parse(text)));
  return {
    kind: 'qt',
    call(slot, payload) {
      return new Promise((resolve, reject) => {
        const method = remote[slot];
        if (typeof method !== 'function') { reject(new Error(`no bridge slot ${slot}`)); return; }
        const arg = slot === 'gesture' ? String(payload) : JSON.stringify(payload ?? null);
        method(arg, (reply) => {
          if (reply === undefined || reply === null || reply === '') { resolve(null); return; }
          const value = JSON.parse(reply);
          if (value && value.error) reject(new Error(value.error)); else resolve(value);
        });
      });
    },
    onFrame: (fn) => frames.add(fn),
    onDialog: (fn) => dialogs.add(fn),
  };
}

/**
 * A stand-in bridge with the same interface, for specs and for opening the
 * page in a plain browser. `describe` maps addresses to metadata, `values`
 * holds their values. Sets apply at once and are reported by the next
 * pushFrame(); `act` answers with an id and `actions[verb](args, fake, id)` may
 * return a result (reported as a done event on the next frame); `post(event)`
 * queues any other event (progress, errors without an action).
 * `calls` records every call as [slot, payload].
 */
export function createFakeBridge({ describe = {}, values = {}, actions = {} } = {}) {
  const frames = listeners();
  const dialogs = listeners();
  let version = 0;
  let nextId = 1;
  let changes = {};
  let events = [];
  const fake = {
    kind: 'fake',
    calls: [],
    values: { ...values },
    describeTable: { ...describe },
    transport: { playing: false, position: 0, playing_pattern: 0, selected_pattern: 0,
      queued_pattern: null, chain: [] },
    dialogAnswers: [],
    async call(slot, payload) {
      fake.calls.push([slot, payload]);
      switch (slot) {
        case 'get': {
          const out = { values: {}, errors: {} };
          for (const a of [].concat(payload)) {
            if (a in fake.values) out.values[a] = fake.values[a]; else out.errors[a] = `unknown address: ${a}`;
          }
          return out;
        }
        case 'set': {
          const errors = {};
          for (const c of [].concat(payload)) {
            if (!(c.address in fake.values)) { errors[c.address] = `unknown address: ${c.address}`; continue; }
            fake.values[c.address] = c.value;
            changes[c.address] = c.value;
          }
          return { errors };
        }
        case 'act': {
          const id = nextId++;
          const handler = actions[payload.verb];
          let event;
          try {
            event = { id, verb: payload.verb, status: 'done', result: handler ? handler(payload.args || {}, fake, id) ?? null : null };
          } catch (err) {
            event = { id, verb: payload.verb, status: 'error', error: String(err.message || err) };
          }
          events.push(event);
          return { id };
        }
        case 'describe': {
          const names = typeof payload === 'string'
            ? Object.keys(fake.describeTable).filter((a) => a.startsWith(payload)).sort()
            : payload;
          const out = { addresses: {}, errors: {} };
          for (const a of names) {
            if (a in fake.describeTable) out.addresses[a] = fake.describeTable[a]; else out.errors[a] = `unknown address: ${a}`;
          }
          return out;
        }
        case 'gesture':
        case 'resync':
          return null;
        case 'resizeWindow':
          return { size: null };
        case 'trigger':
          return {};
        case 'fileDialog': {
          const id = nextId++;
          const path = fake.dialogAnswers.length ? fake.dialogAnswers.shift() : null;
          Promise.resolve().then(() => dialogs.emit({ id, path }));
          return { id };
        }
        default:
          throw new Error(`no bridge slot ${slot}`);
      }
    },
    onFrame: (fn) => frames.add(fn),
    onDialog: (fn) => dialogs.add(fn),
    /** Queue an event for the next frame (progress, errors without an action). */
    post(event) { events.push(event); },
    /** Emit a frame with the changes and events since the last one. */
    pushFrame(extra = {}) {
      version += 1;
      for (const e of events) e.version = version;
      const frame = { version, changes, events, transport: { ...fake.transport }, modulation: { channel: 0, offsets: {} },
        audio: {}, midi: { activity: 0, notes: [0, 0, 0, 0, 0, 0, 0, 0], pickup: {} }, ...extra };
      changes = {};
      events = [];
      frames.emit(frame);
      return frame;
    },
  };
  return fake;
}
