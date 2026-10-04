// Browser stand-in for the subset of `node:test` the specs use, mapped in by test.html's import map.
export const registered = [];
export function test(name, fn) { registered.push({ name, fn }); }
export const it = test;
export function describe(name, fn) { fn(); }
export default test;
