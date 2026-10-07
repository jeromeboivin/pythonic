// Overlay sheets over the panel (map decisions #13, #22). A sheet sits on a
// layer that dims the whole stage: the panel keeps playing and animating
// behind it (frames still arrive) but takes no input until the sheet closes.
// The setup sheet (later slice) and the alert sheet are sheets.
//
//   const sheets = createSheets(stage, { onShow });
//   sheets.show('setup', element, { width, dismissable, onHide, className });
//   sheets.hide('setup');                // or hide() for the top one
//   sheets.current                       // the top sheet's name, or null
//   sheets.isOpen('setup')
//   await sheets.alert({ title, text, tone, buttons }) // the clicked button's value
//   await sheets.ask(title, text, { yes, no, tone })   // true / false
//
// Sheets stack (an alert shows over the setup sheet). A dismissable sheet
// closes with a click outside it (on the dimmed panel); the alert sheet is
// 460 wide with a red (error) or green (question, success) top edge and
// closes only by its buttons, the primary one lit on the right. Alerts come
// one at a time, in order; an error alert equal to one showing or waiting is
// not shown twice. `onShow` runs before a sheet shows (the panel closes its
// menus and ends click to assign).

export const ALERT_WIDTH = 460;

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

export function createSheets(stage, { onShow = null } = {}) {
  const layers = []; // { name, layer, onHide }
  const queue = []; // alerts waiting: { options, resolve, key }
  let alerting = null; // the alert showing: { key, promise }

  const find = (name) => layers.find((s) => s.name === name);

  function show(name, element, { width = null, dismissable = false, onHide = null, className = '' } = {}) {
    if (find(name)) hide(name);
    if (onShow) onShow(name);
    const layer = el('div', 'sheet-layer');
    layer.dataset.sheet = name;
    layer.style.zIndex = String(40 + layers.length);
    const card = el('div', `sheet ${className}`.trim());
    if (width) card.style.width = `${width}px`;
    card.append(element);
    layer.append(card);
    // The panel behind takes no input: nothing reaches its controls, and a
    // click on the dimmed panel closes a dismissable sheet
    layer.addEventListener('wheel', (e) => { e.preventDefault(); e.stopPropagation(); }, { passive: false });
    layer.addEventListener('contextmenu', (e) => { e.preventDefault(); e.stopPropagation(); });
    layer.addEventListener('click', (e) => {
      if (e.target === layer && dismissable) hide(name);
    });
    stage.append(layer);
    layers.push({ name, layer, onHide });
    return card;
  }

  function hide(name = null) {
    const entry = name ? find(name) : layers.at(-1);
    if (!entry) return false;
    layers.splice(layers.indexOf(entry), 1);
    entry.layer.remove();
    if (entry.onHide) entry.onHide();
    return true;
  }

  function showNextAlert() {
    if (alerting || !queue.length) return;
    const { options, resolve, key } = queue.shift();
    const { title = '', text = '', tone = 'error',
      buttons = [{ label: 'OK', value: true, primary: true }] } = options;
    const box = el('div', `alert-sheet tone-${tone === 'error' ? 'error' : 'ok'}`);
    box.append(el('div', 'alert-title', title));
    if (text) box.append(el('div', 'alert-text', text));
    const row = el('div', 'alert-buttons');
    for (const b of buttons) {
      const button = el('button', `btn sq${b.primary ? ' on primary' : ''}`, b.label);
      button.type = 'button';
      button.addEventListener('click', (e) => {
        e.stopPropagation();
        hide('alert');
        alerting = null;
        resolve(b.value);
        showNextAlert();
      });
      row.append(button);
    }
    box.append(row);
    alerting = { key };
    show('alert', box, { width: ALERT_WIDTH, className: 'alert' });
  }

  /** Show the alert sheet; resolves with the clicked button's value. */
  function alert(options = {}) {
    const key = (options.tone || 'error') === 'error' ? `${options.title}\n${options.text}` : null;
    if (key) {
      const same = (alerting && alerting.key === key) || queue.some((q) => q.key === key);
      if (same) return Promise.resolve(null);
    }
    const promise = new Promise((resolve) => queue.push({ options, resolve, key }));
    showNextAlert();
    return promise;
  }

  /** A yes / no question on the alert sheet (green): Promise<boolean>. */
  function ask(title, text, { yes = 'yes', no = 'cancel', tone = 'ok' } = {}) {
    return alert({ title, text, tone, buttons: [{ label: no, value: false }, { label: yes, value: true, primary: true }] })
      .then((v) => v === true);
  }

  return {
    show,
    hide,
    alert,
    ask,
    get current() { return layers.length ? layers.at(-1).name : null; },
    isOpen: (name) => !!find(name),
    destroy() {
      while (layers.length) layers.pop().layer.remove();
      queue.length = 0;
      alerting = null;
    },
  };
}
