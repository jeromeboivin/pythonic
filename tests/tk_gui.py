"""Closing the tkinter GUI a test built."""


def close(gui):
    """End the GUI as closing its window does: gui.run() returns, the core closes.

    The timers still pending are cancelled first (a destroyed window's would
    fire in the next test's event loop). They are only cancelled: the widget
    that set a timer deletes its callback as it is destroyed, and deleting it
    here too made the destroy of a dialog left open by a failed test fail, so
    the event loop never ended. Should the destroy fail anyway, the loop still
    ends and the error is raised here.
    """
    errors = []

    def stop():
        try:
            for job in gui.root.tk.splitlist(gui.root.tk.call('after', 'info')):
                gui.root.tk.call('after', 'cancel', job)
            gui.root.destroy()
        except Exception as exc:
            errors.append(exc)
            gui.root.quit()

    gui.root.after(10, stop)
    gui.run()
    if errors:
        raise errors[0]
