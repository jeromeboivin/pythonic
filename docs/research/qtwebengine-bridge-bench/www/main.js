import { hello } from './mod.js';

const s = window.__status = {
  module: hello(),
  origin: location.origin,
  qwebchannelGlobal: typeof QWebChannel,
  transport: typeof qt !== 'undefined' && !!qt.webChannelTransport,
};
try {
  const r = await fetch('./data.json');
  s.fetch = (await r.json()).ok;
} catch (e) {
  s.fetch = 'ERR ' + e;
}

const params = new URLSearchParams(location.search);

function report(b, kind, data) { b.report(kind, JSON.stringify(data)); }

if (s.transport && typeof QWebChannel === 'function') {
  new QWebChannel(qt.webChannelTransport, ch => {
    const b = ch.objects.bridge;
    window.bridge = b;
    b.tick.connect((seq, payload) => b.ack(seq));
    let propCount = 0;
    b.levelChanged.connect(() => { propCount++; });
    window.getPropCount = () => propCount;
    window.onTick = (seq, payload) => b.ack(seq);
    window.chFlood = n => {
      const t0 = performance.now();
      for (let i = 0; i < n; i++) b.setParam(i % 8, i * 0.001);
      return performance.now() - t0;
    };
    window.chRtt = async n => {
      const out = [];
      for (let i = 0; i < n; i++) {
        const t = performance.now();
        await b.echo(i, 0.5);
        out.push(performance.now() - t);
      }
      report(b, 'chRtt', out);
    };

    const wsPort = params.get('ws');
    if (wsPort) {
      const ws = new WebSocket('ws://127.0.0.1:' + wsPort);
      const pending = new Map();
      ws.onmessage = e => {
        const m = JSON.parse(e.data);
        if (m.t === 'tick') ws.send(JSON.stringify({ t: 'ack', seq: m.seq }));
        else if (m.t === 'echo') { const r = pending.get(m.i); pending.delete(m.i); r(); }
      };
      ws.onopen = () => b.ready('ws-open');
      ws.onerror = () => b.ready('ws-error');
      window.wsFlood = n => {
        const t0 = performance.now();
        for (let i = 0; i < n; i++) ws.send(JSON.stringify({ t: 'param', i: i % 8, v: i * 0.001 }));
        return performance.now() - t0;
      };
      window.wsRtt = async n => {
        const out = [];
        for (let i = 0; i < n; i++) {
          const t = performance.now();
          await new Promise(res => { pending.set(i, res); ws.send(JSON.stringify({ t: 'echo', i })); });
          out.push(performance.now() - t);
        }
        report(b, 'wsRtt', out);
      };
    } else {
      b.ready('channel');
    }
  });
}
