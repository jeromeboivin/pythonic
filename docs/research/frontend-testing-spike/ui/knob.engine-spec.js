// Component test: needs a real DOM + customElements, so it runs in the engine, not in Node.
import './knob.js';
const assert = (c, m) => { if (!c) throw new Error(m); };
test('knob emits change with its address', () => {
  const k = document.createElement('pd-knob');
  k.setAttribute('address', 'ch3.decay');
  document.body.append(k);
  let got = null;
  k.addEventListener('change', e => (got = e.detail));
  k.drag(-200);
  assert(got && got.address === 'ch3.decay' && got.value === 24, JSON.stringify(got));
  assert(k.textContent === '24.0', k.textContent);
});
test('deliberately failing test is reported', () => assert(false, 'boom'));
