import './knob.js';
const ready = new Promise(res => {
  if (typeof qt === 'undefined') return res(null);
  new QWebChannel(qt.webChannelTransport, ch => res(ch.objects.core));
});
const core = await ready;
document.addEventListener('change', e => core && core.set(e.detail.address, e.detail.value));
if (core) core.frame.connect(f => { document.getElementById('step').textContent = String(f.step); });
window.__ready = true;
