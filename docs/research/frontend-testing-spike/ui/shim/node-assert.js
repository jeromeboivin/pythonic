// Browser stand-in for the subset of `node:assert/strict` the specs use.
function fail(msg) { throw new Error(msg); }
const show = v => JSON.stringify(v);
const assert = (v, msg) => { if (!v) fail(msg ?? 'assertion failed'); };
assert.ok = assert;
assert.equal = (a, b, msg) => { if (!Object.is(a, b)) fail(msg ?? `${show(a)} !== ${show(b)}`); };
assert.deepEqual = (a, b, msg) => { if (show(a) !== show(b)) fail(msg ?? `${show(a)} != ${show(b)}`); };
assert.throws = (fn, msg) => { try { fn(); } catch { return; } fail(msg ?? 'did not throw'); };
export default assert;
