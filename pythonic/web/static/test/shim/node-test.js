// Browser stand-in for the subset of `node:test` the specs use (mapped in by
// runner.html's import map). Specs register; the runner runs them in order.
export const registered = [];
let prefix = '';
export function test(name, fn) { registered.push({ name: prefix + name, fn }); }
export const it = test;
export function describe(name, fn) {
  const outer = prefix;
  prefix = `${outer}${name} > `;
  try { fn(); } finally { prefix = outer; }
}
export default test;
