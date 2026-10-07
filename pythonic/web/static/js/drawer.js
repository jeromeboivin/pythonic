// The edit rack drawer under the face (map decisions #7, #8, #13): it shows
// one page at a time. Its base page is the selected channel's edit rack (the
// edit rack slice sets it); other pages (the ⊞ matrix, later the PO-32 and AI
// pages) replace it until they are hidden:
//
//   const drawer = createDrawer(slotElement);
//   drawer.setBase(rackElement);                  // the edit rack slice
//   drawer.show('matrix', element, { onHide });   // replaces the current page
//   drawer.hide('matrix');                        // back to the base page
//   drawer.toggle('matrix', () => element, opts); // show, or hide when shown
//   drawer.current                                // 'matrix' or null (base)
//   drawer.onChange((name) => ...)                // after every show / hide
//
// Opening a page opens the drawer when it was closed, and hiding the page
// restores the closed state. The drawer's open / closed state belongs to the
// edit rack slice (window shrink, saved preference): it plugs in with
// drawer.setOpener({ isOpen: () => bool, setOpen: (open) => void }). Until
// then the drawer is always open.

export function createDrawer(slot) {
  let base = null;
  let page = null; // { name, element, onHide, opened }
  let opener = { isOpen: () => true, setOpen: () => {} };
  const listeners = new Set();
  const emit = () => { for (const fn of [...listeners]) fn(page ? page.name : null); };

  const showBase = () => {
    slot.replaceChildren(...(base ? [base] : []));
    delete slot.dataset.page;
  };

  const drawer = {
    get current() { return page ? page.name : null; },
    get open() { return !!opener.isOpen(); },
    /** The base page (the edit rack). */
    setBase(element) {
      base = element;
      if (!page) showBase();
    },
    /** Plug in the drawer's open / closed state (the edit rack slice). */
    setOpener(next) { opener = next; },
    /** Show a page in place of the current one; opens the drawer if closed. */
    show(name, element, { onHide = null } = {}) {
      // A page replacing another keeps the state from before the first one
      const opened = page ? page.opened : !opener.isOpen();
      if (page) drawer.hide(page.name, { restore: false });
      page = { name, element, onHide, opened };
      slot.replaceChildren(element);
      slot.dataset.page = name;
      if (!opener.isOpen()) opener.setOpen(true);
      emit();
      return element;
    },
    /** Hide a page (the current one when no name): back to the base page. */
    hide(name = null, { restore = true } = {}) {
      if (!page || (name && page.name !== name)) return false;
      const done = page;
      page = null;
      showBase();
      if (restore && done.opened) opener.setOpen(false);
      if (done.onHide) done.onHide();
      if (restore) emit();
      return true;
    },
    /** Show a page, or hide it when it is the current one; returns whether it shows. */
    toggle(name, build, options) {
      if (page && page.name === name) { drawer.hide(name); return false; }
      drawer.show(name, build(), options);
      return true;
    },
    onChange(fn) { listeners.add(fn); return () => listeners.delete(fn); },
  };
  return drawer;
}
