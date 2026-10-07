"""
PO-32 Transfer Dialog for Pythonic
Sends the sounds to a PO-32 Tonic over its audio modem, through the app
core's PO-32 module (``po32.*`` verbs): the core renders the signal and plays
it through its output stream.

Transfer window:
- Bank selection (1-8 or 9-16)
- Pattern chain selection (those PO-32 pattern slots are sent empty)
- Channel selection
- Transfer / stop with progress, save WAV
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


class PO32TransferDialog:
    """
    PO-32 Tonic Transfer Window
    
    Provides a GUI dialog for transferring sounds and patterns to the
    PO-32 via FSK audio modem signal.
    """
    
    # Dialog colors matching Pythonic theme
    COLORS = {
        'bg': '#2a2a3a',
        'bg_medium': '#3a3a4a',
        'bg_light': '#4a4a5a',
        'accent': '#5566aa',
        'accent_light': '#7788cc',
        'text': '#ccccee',
        'text_dim': '#8888aa',
        'highlight': '#4488ff',
        'green': '#44ff88',
        'red': '#ff4444',
        'orange': '#ffaa44',
        'white': '#ffffff',
    }
    
    def __init__(self, parent, core, when_done=None):
        """
        Args:
            parent: Parent tkinter window
            core: the AppCore (its ``po32.*`` verbs render and send)
            when_done: ``when_done(action_id, callback(event))`` runs a
                callback once the core reports an action (the main window's
                poll tick)
        """
        self.parent = parent
        self.core = core
        self.when_done = when_done or (lambda action_id, callback: None)
        self.preset_name = core.get('preset.name') or "Untitled"
        
        # Transfer state
        self.is_transferring = False
        self.seconds = None  # length of the prepared signal
        self._progress_job = None
        self._closed = False
        
        # Settings
        self.bank = 0  # 0 = instruments 1-8, 1 = instruments 9-16
        
        # The channel checkboxes start from the face mutes
        self.mute_mask = [core.get(f'ch{i + 1}.mute') for i in range(8)]
        
        # Create dialog
        self._create_dialog()
    
    def _create_dialog(self):
        """Build the transfer dialog window"""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("PO-32 Tonic Transfer")
        self.dialog.configure(bg=self.COLORS['bg'])
        self.dialog.geometry("420x550")
        self.dialog.resizable(True, True)
        self.dialog.minsize(380, 400)
        self.dialog.transient(self.parent)
        
        # Make modal
        self.dialog.grab_set()
        
        # Main container with padding
        main = tk.Frame(self.dialog, bg=self.COLORS['bg'])
        main.pack(fill='both', expand=True, padx=15, pady=15)
        
        # --- Header ---
        header = tk.Frame(main, bg=self.COLORS['bg'])
        header.pack(fill='x', pady=(0, 10))
        
        tk.Label(header, text="PO-32 Tonic Transfer",
                font=('Segoe UI', 16, 'bold'),
                fg=self.COLORS['accent_light'],
                bg=self.COLORS['bg']).pack()
        
        # --- Preset name ---
        preset_frame = tk.Frame(main, bg=self.COLORS['bg_medium'],
                               relief='groove', bd=1)
        preset_frame.pack(fill='x', pady=(0, 10))
        
        tk.Label(preset_frame, text=self.preset_name,
                font=('Segoe UI', 11),
                fg=self.COLORS['white'],
                bg=self.COLORS['bg_medium'],
                pady=5).pack()
        
        # --- Settings Frame ---
        settings = tk.LabelFrame(main, text="Transfer Settings",
                                font=('Segoe UI', 9),
                                fg=self.COLORS['text'],
                                bg=self.COLORS['bg'],
                                bd=1, relief='groove')
        settings.pack(fill='x', pady=(0, 10))
        
        # Bank selection
        bank_row = tk.Frame(settings, bg=self.COLORS['bg'])
        bank_row.pack(fill='x', padx=10, pady=5)
        
        tk.Label(bank_row, text="Transfer sounds to:",
                font=('Segoe UI', 9),
                fg=self.COLORS['text'],
                bg=self.COLORS['bg']).pack(side='left')
        
        self.bank_var = tk.StringVar(value="1 - 8")
        bank_combo = ttk.Combobox(bank_row, textvariable=self.bank_var,
                                  values=["1 - 8", "9 - 16"],
                                  width=8, state='readonly')
        bank_combo.pack(side='right')
        bank_combo.bind('<<ComboboxSelected>>', self._on_bank_change)
        
        # Pattern chain selection
        pattern_row = tk.Frame(settings, bg=self.COLORS['bg'])
        pattern_row.pack(fill='x', padx=10, pady=5)
        
        tk.Label(pattern_row, text="Pattern (chain) to:",
                font=('Segoe UI', 9),
                fg=self.COLORS['text'],
                bg=self.COLORS['bg']).pack(side='left')
        
        # Build pattern chain options
        pattern_options = self._get_pattern_chain_options()
        self.pattern_var = tk.StringVar(value=pattern_options[0] if pattern_options else "A - B")
        pattern_combo = ttk.Combobox(pattern_row, textvariable=self.pattern_var,
                                     values=pattern_options,
                                     width=12, state='readonly')
        pattern_combo.pack(side='right')
        pattern_combo.bind('<<ComboboxSelected>>', self._on_pattern_change)
        
        # --- Channel List ---
        channels_frame = tk.LabelFrame(main, text="Channels",
                                       font=('Segoe UI', 9),
                                       fg=self.COLORS['text'],
                                       bg=self.COLORS['bg'],
                                       bd=1, relief='groove')
        channels_frame.pack(fill='x', pady=(0, 10))
        
        # Show 8 channel names with checkboxes
        self.channel_vars = []
        for i in range(8):
            name = self.core.get(f'ch{i + 1}.name') or f'Drum {i+1}'
            
            var = tk.BooleanVar(value=not self.mute_mask[i])
            self.channel_vars.append(var)
            
            ch_row = tk.Frame(channels_frame, bg=self.COLORS['bg'])
            ch_row.pack(fill='x', padx=10, pady=1)
            
            cb = tk.Checkbutton(ch_row, variable=var,
                               bg=self.COLORS['bg'],
                               fg=self.COLORS['text'],
                               selectcolor=self.COLORS['bg_medium'],
                               activebackground=self.COLORS['bg'],
                               activeforeground=self.COLORS['text'],
                               command=self._on_channel_toggle)
            cb.pack(side='left')
            
            tk.Label(ch_row, text=f"{i+1}:",
                    font=('Segoe UI', 9, 'bold'),
                    fg=self.COLORS['accent_light'],
                    bg=self.COLORS['bg'],
                    width=2).pack(side='left')
            
            tk.Label(ch_row, text=name,
                    font=('Segoe UI', 9),
                    fg=self.COLORS['text'],
                    bg=self.COLORS['bg']).pack(side='left', padx=(5, 0))
        
        # --- Instructions ---
        instr_frame = tk.Frame(main, bg=self.COLORS['bg_medium'],
                              relief='groove', bd=1)
        instr_frame.pack(fill='x', pady=(0, 10))
        
        tk.Label(instr_frame, 
                text="Put your PO-32 in receive mode:\nhold [ write ] + press [ sound ]",
                font=('Segoe UI', 9),
                fg=self.COLORS['orange'],
                bg=self.COLORS['bg_medium'],
                pady=8, justify='center').pack()
        
        # --- Progress ---
        self.progress_frame = tk.Frame(main, bg=self.COLORS['bg'])
        self.progress_frame.pack(fill='x', pady=(0, 5))
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(self.progress_frame,
                                            variable=self.progress_var,
                                            maximum=100,
                                            length=380)
        self.progress_bar.pack(fill='x')
        
        self.status_label = tk.Label(self.progress_frame,
                                     text="Ready to transfer",
                                     font=('Segoe UI', 8),
                                     fg=self.COLORS['text_dim'],
                                     bg=self.COLORS['bg'])
        self.status_label.pack(pady=(2, 0))
        
        # --- Buttons ---
        btn_frame = tk.Frame(main, bg=self.COLORS['bg'])
        btn_frame.pack(fill='x', pady=(5, 0))
        
        # Transfer / Stop button
        self.transfer_btn = tk.Button(
            btn_frame, text="▶  Transfer",
            font=('Segoe UI', 11, 'bold'),
            bg=self.COLORS['accent'],
            fg=self.COLORS['white'],
            activebackground=self.COLORS['accent_light'],
            activeforeground=self.COLORS['white'],
            width=15, height=1,
            command=self._on_transfer_click,
            relief='raised', bd=2
        )
        self.transfer_btn.pack(side='left', padx=(0, 5))
        
        # Save WAV button
        self.save_btn = tk.Button(
            btn_frame, text="💾  Save WAV",
            font=('Segoe UI', 10),
            bg=self.COLORS['bg_light'],
            fg=self.COLORS['text'],
            activebackground=self.COLORS['bg_medium'],
            width=12,
            command=self._on_save_wav
        )
        self.save_btn.pack(side='left', padx=5)
        
        # Close button
        tk.Button(
            btn_frame, text="Close",
            font=('Segoe UI', 10),
            bg=self.COLORS['bg_light'],
            fg=self.COLORS['text'],
            activebackground=self.COLORS['bg_medium'],
            width=8,
            command=self._on_close
        ).pack(side='right')
        
        # Handle window close
        self.dialog.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Pre-generate audio
        self._generate_audio()
    
    def _get_pattern_chain_options(self) -> list:
        """The pattern chain choices (the core's chain groups, 'A - B', 'C', ...)."""
        return self.core.get('po32.chain_options') or ["A"]
    
    def _settings(self):
        """The transfer settings for the core's po32 verbs."""
        return {
            'bank': self.bank,
            'chain': self.pattern_var.get(),
            'channels': [i + 1 for i, muted in enumerate(self.mute_mask) if not muted],
        }
    
    def _act(self, verb, on_done=None, **args):
        """Start a core verb; on_done(event) runs on the main window's tick."""
        action_id = self.core.act(verb, **args)
        if on_done is not None:
            self.when_done(action_id, on_done)
        return action_id
    
    def _on_bank_change(self, event=None):
        """Handle bank selection change"""
        self.bank = 0 if "1 - 8" in self.bank_var.get() else 1
        self._generate_audio()
    
    def _on_pattern_change(self, event=None):
        """Handle pattern chain selection change"""
        self._generate_audio()
    
    def _on_channel_toggle(self):
        """Handle channel enable/disable toggle"""
        self.mute_mask = [not var.get() for var in self.channel_vars]
        self._generate_audio()
    
    def _generate_audio(self):
        """Have the core render the signal (it reports its length)"""
        self.status_label.config(text="Generating audio signal...")
        
        def done(event):
            if self._closed or self.is_transferring:
                return
            if event['status'] == 'done':
                self.seconds = event['result']['seconds']
                self.status_label.config(
                    text=f"Ready to transfer ({self.seconds:.1f}s audio)",
                    fg=self.COLORS['text_dim'])
            else:
                self.seconds = None
                self.status_label.config(text=f"Error: {event.get('error')}",
                                         fg=self.COLORS['red'])
        self._act('po32.prepare', done, **self._settings())
    
    def _on_transfer_click(self):
        """Handle Transfer/Stop button click"""
        if self.is_transferring:
            self._stop_transfer()
        else:
            self._start_transfer()
    
    def _start_transfer(self):
        """Send the signal through the core's audio output"""
        if not self.core.get('audio.running'):
            messagebox.showwarning("Audio Not Available",
                                   "The audio output is not running.\n"
                                   "Use 'Save WAV' to save the transfer audio.",
                                   parent=self.dialog)
            return
        
        self.is_transferring = True
        
        # Update UI
        self.transfer_btn.config(text="■  Stop", bg=self.COLORS['red'])
        self.save_btn.config(state='disabled')
        self.status_label.config(text="TRANSFERRING...",
                                fg=self.COLORS['green'])
        
        self._act('po32.send', self._transfer_finished, **self._settings())
        
        # Start progress update timer
        self._update_progress()
    
    def _transfer_finished(self, event):
        if self._closed:
            return
        status = event['status']
        if status == 'done':
            self._transfer_complete()
        elif status == 'cancelled':
            self._transfer_stopped()
        else:
            self._transfer_error(event.get('error'))
    
    def _update_progress(self):
        """Update progress bar during transfer"""
        if not self.is_transferring or self._closed:
            self._progress_job = None
            return
        
        self.progress_var.set(self.core.get('po32.progress') * 100)
        self._progress_job = self.dialog.after(100, self._update_progress)
    
    def _transfer_complete(self):
        """Called when transfer finishes successfully"""
        self.is_transferring = False
        self.progress_var.set(100)
        
        self.transfer_btn.config(text="▶  Transfer", bg=self.COLORS['accent'])
        self.save_btn.config(state='normal')
        self.status_label.config(text="Transfer complete!",
                                fg=self.COLORS['green'])
        
        # Reset progress after a moment
        def _reset_status():
            if self._closed or self.is_transferring:
                return
            self.progress_var.set(0)
            if self.seconds is not None:
                self.status_label.config(
                    text=f"Ready to transfer ({self.seconds:.1f}s audio)",
                    fg=self.COLORS['text_dim']
                )
        self.dialog.after(2000, _reset_status)
    
    def _transfer_stopped(self):
        """Called when transfer is stopped by user"""
        self.is_transferring = False
        self.progress_var.set(0)
        
        self.transfer_btn.config(text="▶  Transfer", bg=self.COLORS['accent'])
        self.save_btn.config(state='normal')
        self.status_label.config(text="Transfer stopped",
                                fg=self.COLORS['orange'])
    
    def _transfer_error(self, error_msg):
        """Called when transfer encounters an error"""
        self.is_transferring = False
        self.progress_var.set(0)
        
        self.transfer_btn.config(text="▶  Transfer", bg=self.COLORS['accent'])
        self.save_btn.config(state='normal')
        self.status_label.config(text=f"Error: {error_msg}",
                                fg=self.COLORS['red'])
    
    def _stop_transfer(self):
        """Stop ongoing transfer"""
        self._act('po32.cancel')
    
    def _on_save_wav(self):
        """Save transfer audio as WAV file"""
        if self.seconds is None:
            messagebox.showwarning("No Audio",
                                   "Generate audio first.",
                                   parent=self.dialog)
            return
        
        # Default filename from preset name
        default_name = f"{self.preset_name} (PO-32 transfer).wav"
        default_name = "".join(c for c in default_name if c not in '<>:"/\\|?*')
        
        filepath = filedialog.asksaveasfilename(
            parent=self.dialog,
            title="Save PO-32 Transfer Audio",
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
            initialfile=default_name
        )
        
        if filepath:
            def done(event):
                if event['status'] != 'done':
                    messagebox.showerror("Save Error", str(event.get('error')),
                                         parent=self.parent if self._closed else self.dialog)
                elif not self._closed:
                    self.status_label.config(
                        text=f"Saved: {os.path.basename(event['result']['path'])}",
                        fg=self.COLORS['green']
                    )
            # Tk's save dialog has asked before replacing a file
            self._act('po32.save_wav', done, path=filepath, overwrite=True,
                      **self._settings())
    
    def _on_close(self):
        """Handle dialog close"""
        self._closed = True
        
        if self.is_transferring:
            self._stop_transfer()
        if self._progress_job is not None:
            self.dialog.after_cancel(self._progress_job)
        
        try:
            self.dialog.grab_release()
        except Exception:
            pass
        self.dialog.destroy()
