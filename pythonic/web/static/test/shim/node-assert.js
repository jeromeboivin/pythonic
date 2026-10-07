// Browser stand-in for the subset of `node:assert/strict` the specs use.
function fail(message) { throw new Error(message); }
const show = (v) => JSON.stringify(v);
const assert = (value, message) => { if (!value) fail(message ?? `expected truthy, got ${show(value)}`); };
assert.ok = assert;
assert.equal = (a, b, message) => { if (!Object.is(a, b)) fail(message ?? `${show(a)} !== ${show(b)}`); };
assert.notEqual = (a, b, message) => { if (Object.is(a, b)) fail(message ?? `${show(a)} === ${show(b)}`); };
assert.deepEqual = (a, b, message) => { if (show(a) !== show(b)) fail(message ?? `${show(a)} != ${show(b)}`); };
assert.throws = (fn, message) => {
  try { fn(); } catch { return; }
  fail(message ?? 'did not throw');
};
assert.rejects = async (promise, message) => {
  try { await (typeof promise === 'function' ? promise() : promise); } catch { return; }
  fail(message ?? 'did not reject');
};
export default assert;
