"""
The dialog the tkinter interface shows when the web interface was asked for
but PySide6 with QtWebEngine is missing: it explains why, and (except on
Windows ARM64, where no QtWebEngine exists) installs it with pip, then asks
for a restart.
"""

import threading
import tkinter as tk
from tkinter import messagebox

from pythonic.install import EXTRAS, install_extra, windows_arm64


def explanation(reason, arm64=None):
    arm64 = windows_arm64() if arm64 is None else arm64
    text = ("The new interface runs in a Qt web view (PySide6 with QtWebEngine), "
            f"which could not be loaded:\n\n  {reason}\n\n"
            "Pythonic started the classic interface instead.")
    if arm64:
        text += ("\n\nQtWebEngine is not available for Windows on ARM, so this "
                 "machine uses the classic interface.")
    else:
        text += ("\n\nInstall runs:\n  pip install " + ' '.join(EXTRAS['qt'])
                 + "\n\nin the current Python environment (about 650 MB). "
                 "Restart Pythonic afterwards.")
    return text


def show_qt_missing(root, reason, arm64=None):
    """Open the dialog over root; returns the Toplevel."""
    arm64 = windows_arm64() if arm64 is None else arm64
    dialog = tk.Toplevel(root)
    dialog.title('Web interface unavailable')
    dialog.transient(root)
    dialog.resizable(False, False)
    tk.Label(dialog, text=explanation(reason, arm64), justify='left', wraplength=460,
             padx=16, pady=12).pack(fill='x')
    buttons = tk.Frame(dialog, padx=12, pady=10)
    buttons.pack(fill='x')
    close = tk.Button(buttons, text='Close', width=10, command=dialog.destroy)
    close.pack(side='right')

    if not arm64:
        install = tk.Button(buttons, text='Install', width=10)
        install.pack(side='right', padx=6)

        def run_install():
            install.config(state='disabled', text='Installing...')
            lines = []

            def work():
                ok = install_extra('qt', lines.append)
                dialog.after(0, lambda: finished(ok))

            threading.Thread(target=work, daemon=True).start()

        def finished(ok):
            if ok:
                install.config(text='Installed')
                messagebox.showinfo('Installed', 'PySide6 is installed. Restart Pythonic '
                                    'to use the new interface.', parent=dialog)
            else:
                install.config(text='Install', state='normal')
                messagebox.showerror('Install failed', 'Installation failed.\n\n'
                                     + '\n'.join(lines[-10:]), parent=dialog)

        install.config(command=run_install)
    return dialog
