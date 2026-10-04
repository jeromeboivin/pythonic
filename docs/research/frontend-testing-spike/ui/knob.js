import { dragToValue } from './model.js';
export class PdKnob extends HTMLElement {
  connectedCallback() { this.value = 0; this.textContent = '0'; }
  drag(dy) {
    this.value = dragToValue(this.value, dy, -24, 24);
    this.textContent = this.value.toFixed(1);
    this.dispatchEvent(new CustomEvent('change', { detail: { address: this.getAttribute('address'), value: this.value }, bubbles: true }));
  }
}
customElements.define('pd-knob', PdKnob);
