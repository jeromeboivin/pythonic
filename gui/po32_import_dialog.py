"""
PO-32 Import Dialog
Imports drum patches and patterns from PO-32 modem audio (WAV files or live
recording) through the app core's PO-32 module (``po32.*`` verbs and
addresses): the core listens to and records the input, decodes, keeps the
bank and pattern picks, previews and imports. This dialog only shows them.

Supports both Pythonic→PO-32 transfers (8 drums, 2 patterns) and
PO-32→PO-32 transfers (16 drums, 16 patterns).
"""

import math
import os
import subprocess
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

PATTERN_LETTERS = 'ABCDEFGHIJKL'
NO_DEVICES = "(no input devices)"


class PO32ImportDialog:
    """
    Dialog for importing PO-32 modem audio data.
    
    Provides:
    - WAV file import or live audio recording
    - Bank selection (drums 1-8 or 9-16)
    - Multi-pattern selection with assignable destination letters (A-L)
    - Pre-listen (play the focused pattern with the decoded sounds)
    - Import up to 12 patterns + 8 drum patches
    """
    
    COLORS = {
        'bg_dark': '#2a2a3a',
        'bg_medium': '#3a3a4a',
        'bg_light': '#4a4a5a',
        'accent': '#5566aa',
        'accent_light': '#7788cc',
        'text': '#ccccee',
        'text_dim': '#8888aa',
        'highlight': '#4488ff',
        'led_on': '#44ff88',
        'led_off': '#333344',
        'trigger_on': '#ff8844',
        'trigger_off': '#333344',
    }
    
    def __init__(self, parent, core, when_done=None):
        """
        Args:
            parent: Parent tk window
            core: the AppCore (its ``po32.*`` verbs do the work)
            when_done: ``when_done(action_id, callback(event))`` runs a
                callback once the core reports an action (the main window's
                poll tick)
        """
        self.parent = parent
        self.core = core
        self.when_done = when_done or (lambda action_id, callback: None)
        
        # Display state (the decode, picks, recording and preview live in the core)
        self._closed = False
        self._tick_job = None
        self._vu_peak = 0.0   # Peak hold level
        self._vu_peak_decay = 0  # Counter for peak hold decay
        self._debug_save = bool(core.get('pref.po32.save_recordings'))
        
        # Build dialog
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Import from PO-32")
        self.dialog.configure(bg=self.COLORS['bg_dark'])
        self.dialog.geometry("720x680")
        self.dialog.minsize(600, 480)
        self.dialog.resizable(True, True)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        self.dialog.protocol("WM_DELETE_WINDOW", self._on_cancel)
        
        self._build_ui()
        self._show_decoded()
        self._tick()
    
    def _build_ui(self):
        """Build the import dialog UI."""
        # Scrollable container
        outer = tk.Frame(self.dialog, bg=self.COLORS['bg_dark'])
        outer.pack(fill='both', expand=True)
        
        canvas = tk.Canvas(outer, bg=self.COLORS['bg_dark'], highlightthickness=0)
        vscroll = tk.Scrollbar(outer, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)
        vscroll.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)
        
        main = tk.Frame(canvas, bg=self.COLORS['bg_dark'])
        canvas_win = canvas.create_window((0, 0), window=main, anchor='nw')
        
        def _on_frame_cfg(e):
            canvas.configure(scrollregion=canvas.bbox('all'))
        main.bind('<Configure>', _on_frame_cfg)
        
        def _on_canvas_cfg(e):
            canvas.itemconfig(canvas_win, width=e.width)
        canvas.bind('<Configure>', _on_canvas_cfg)
        
        def _on_mousewheel(e):
            canvas.yview_scroll(int(-1 * (e.delta / 120)), 'units')
        canvas.bind_all('<MouseWheel>', _on_mousewheel)
        
        # === Source section (top) ===
        source_frame = tk.LabelFrame(main, text="Source", font=('Segoe UI', 9),
                                     fg=self.COLORS['text_dim'],
                                     bg=self.COLORS['bg_medium'],
                                     labelanchor='nw')
        source_frame.pack(fill='x', pady=(0, 8))
        
        # Input device selector row
        device_row = tk.Frame(source_frame, bg=self.COLORS['bg_medium'])
        device_row.pack(fill='x', padx=8, pady=(8, 4))
        
        tk.Label(device_row, text="Input device:",
                font=('Segoe UI', 9),
                fg=self.COLORS['text'],
                bg=self.COLORS['bg_medium']).pack(side='left', padx=(0, 6))
        
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(device_row, textvariable=self.device_var,
                                         width=40, state='readonly')
        self.device_combo.pack(side='left', fill='x', expand=True, padx=(0, 6))
        self.device_combo.bind('<<ComboboxSelected>>', self._on_device_changed)
        self._populate_input_devices()
        
        self.monitor_btn = tk.Button(device_row, text="🔊 Monitor",
                                     font=('Segoe UI', 8), width=10,
                                     bg=self.COLORS['bg_light'],
                                     fg=self.COLORS['text'],
                                     command=self._on_toggle_monitor)
        self.monitor_btn.pack(side='right')
        
        # VU meter row
        vu_frame = tk.Frame(source_frame, bg=self.COLORS['bg_medium'])
        vu_frame.pack(fill='x', padx=8, pady=(0, 4))
        
        tk.Label(vu_frame, text="Level:",
                font=('Segoe UI', 8),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack(side='left', padx=(0, 4))
        
        self.vu_canvas = tk.Canvas(vu_frame, height=18, bg='#1a1a2a',
                                   highlightthickness=1,
                                   highlightbackground=self.COLORS['bg_light'])
        self.vu_canvas.pack(side='left', fill='x', expand=True, padx=(0, 6))
        
        self.vu_db_label = tk.Label(vu_frame, text="-∞ dB",
                                    font=('Segoe UI', 8, 'bold'),
                                    fg=self.COLORS['text_dim'],
                                    bg=self.COLORS['bg_medium'],
                                    width=8, anchor='e')
        self.vu_db_label.pack(side='right')
        
        # Draw initial empty VU meter
        self.vu_canvas.update_idletasks()
        self._draw_vu_meter(0.0, 0.0)
        
        # Debug save checkbox
        debug_row = tk.Frame(source_frame, bg=self.COLORS['bg_medium'])
        debug_row.pack(fill='x', padx=8, pady=(0, 4))
        
        self.debug_save_var = tk.BooleanVar(value=self._debug_save)
        debug_check = tk.Checkbutton(debug_row, text="💾 Save recorded audio to file (debug)",
                                    variable=self.debug_save_var,
                                    font=('Segoe UI', 8),
                                    fg=self.COLORS['text_dim'],
                                    bg=self.COLORS['bg_medium'],
                                    selectcolor=self.COLORS['bg_dark'],
                                    activebackground=self.COLORS['bg_medium'],
                                    activeforeground=self.COLORS['text'],
                                    command=self._on_debug_save_changed)
        debug_check.pack(side='left')
        
        tk.Button(debug_row, text="📂 Open Folder",
                 font=('Segoe UI', 8),
                 bg=self.COLORS['bg_light'],
                 fg=self.COLORS['text_dim'],
                 command=self._open_debug_folder).pack(side='left', padx=(8, 0))
        
        # Buttons row
        source_row = tk.Frame(source_frame, bg=self.COLORS['bg_medium'])
        source_row.pack(fill='x', padx=8, pady=(0, 8))
        
        tk.Button(source_row, text="Import WAV File...",
                 font=('Segoe UI', 9), width=16,
                 bg=self.COLORS['accent'], fg=self.COLORS['text'],
                 command=self._on_import_wav).pack(side='left', padx=(0, 8))
        
        self.record_btn = tk.Button(source_row, text="● Record",
                                    font=('Segoe UI', 9), width=12,
                                    bg=self.COLORS['bg_light'],
                                    fg=self.COLORS['text'],
                                    command=self._on_toggle_record)
        self.record_btn.pack(side='left', padx=(0, 8))
        
        self.source_label = tk.Label(source_row, text="No data loaded",
                                     font=('Segoe UI', 8),
                                     fg=self.COLORS['text_dim'],
                                     bg=self.COLORS['bg_medium'])
        self.source_label.pack(side='left', fill='x', expand=True)
        
        # === Bank selector ===
        bank_frame = tk.LabelFrame(main, text="Drum Bank", font=('Segoe UI', 9),
                                   fg=self.COLORS['text_dim'],
                                   bg=self.COLORS['bg_medium'],
                                   labelanchor='nw')
        bank_frame.pack(fill='x', pady=(0, 8))
        
        bank_row = tk.Frame(bank_frame, bg=self.COLORS['bg_medium'])
        bank_row.pack(fill='x', padx=8, pady=8)
        
        self.bank_var = tk.IntVar(value=0)
        
        self.bank0_radio = tk.Radiobutton(bank_row, text="Bank 0 (Drums 1-8)",
                                          variable=self.bank_var, value=0,
                                          font=('Segoe UI', 9),
                                          fg=self.COLORS['text'],
                                          bg=self.COLORS['bg_medium'],
                                          selectcolor=self.COLORS['bg_dark'],
                                          activebackground=self.COLORS['bg_medium'],
                                          activeforeground=self.COLORS['text'],
                                          command=self._on_bank_change)
        self.bank0_radio.pack(side='left', padx=(0, 20))
        
        self.bank1_radio = tk.Radiobutton(bank_row, text="Bank 1 (Drums 9-16)",
                                          variable=self.bank_var, value=1,
                                          font=('Segoe UI', 9),
                                          fg=self.COLORS['text'],
                                          bg=self.COLORS['bg_medium'],
                                          selectcolor=self.COLORS['bg_dark'],
                                          activebackground=self.COLORS['bg_medium'],
                                          activeforeground=self.COLORS['text'],
                                          command=self._on_bank_change)
        self.bank1_radio.pack(side='left')
        
        self.bank_info_label = tk.Label(bank_row, text="",
                                        font=('Segoe UI', 8),
                                        fg=self.COLORS['text_dim'],
                                        bg=self.COLORS['bg_medium'])
        self.bank_info_label.pack(side='right')
        
        # === Pattern selector with grid ===
        pattern_frame = tk.LabelFrame(main, text="Patterns", font=('Segoe UI', 9),
                                      fg=self.COLORS['text_dim'],
                                      bg=self.COLORS['bg_medium'],
                                      labelanchor='nw')
        pattern_frame.pack(fill='both', expand=True, pady=(0, 8))
        
        # Pattern button grid (2 rows of 8)
        btn_frame = tk.Frame(pattern_frame, bg=self.COLORS['bg_medium'])
        btn_frame.pack(fill='x', padx=8, pady=(8, 4))
        
        self.pattern_buttons = []
        for row in range(2):
            row_frame = tk.Frame(btn_frame, bg=self.COLORS['bg_medium'])
            row_frame.pack(fill='x')
            for col in range(8):
                idx = row * 8 + col
                btn = tk.Button(row_frame, text=str(idx + 1), width=5, height=1,
                               font=('Segoe UI', 8),
                               bg=self.COLORS['bg_light'],
                               fg=self.COLORS['text_dim'],
                               state='disabled',
                               command=lambda i=idx: self._on_pattern_select(i))
                btn.pack(side='left', padx=1, pady=1)
                self.pattern_buttons.append(btn)
        
        # Select / Clear buttons
        sel_row = tk.Frame(pattern_frame, bg=self.COLORS['bg_medium'])
        sel_row.pack(fill='x', padx=8, pady=(2, 4))
        
        tk.Button(sel_row, text="Select First 12", font=('Segoe UI', 8),
                  bg=self.COLORS['bg_light'], fg=self.COLORS['text'],
                  command=self._on_select_all).pack(side='left', padx=(0, 4))
        tk.Button(sel_row, text="Clear All", font=('Segoe UI', 8),
                  bg=self.COLORS['bg_light'], fg=self.COLORS['text'],
                  command=self._on_clear_selection).pack(side='left')
        
        self.selection_count_label = tk.Label(sel_row, text="0/12 patterns selected",
                                              font=('Segoe UI', 8),
                                              fg=self.COLORS['text_dim'],
                                              bg=self.COLORS['bg_medium'])
        self.selection_count_label.pack(side='right')
        
        # Pattern → Destination letter mapping table
        self.mapping_frame = tk.Frame(pattern_frame, bg=self.COLORS['bg_medium'])
        self.mapping_frame.pack(fill='x', padx=8, pady=(0, 4))
        
        # Pattern detail display
        detail_frame = tk.Frame(pattern_frame, bg=self.COLORS['bg_dark'])
        detail_frame.pack(fill='both', expand=True, padx=8, pady=4)
        
        # Step grid visualization
        grid_frame = tk.Frame(detail_frame, bg=self.COLORS['bg_dark'])
        grid_frame.pack(fill='both', expand=True, pady=4)
        
        # Header row: step numbers
        header_row = tk.Frame(grid_frame, bg=self.COLORS['bg_dark'])
        header_row.pack(fill='x')
        tk.Label(header_row, text="", width=6,
                font=('Segoe UI', 7), bg=self.COLORS['bg_dark']).pack(side='left')
        for s in range(16):
            tk.Label(header_row, text=str(s + 1), width=2,
                    font=('Segoe UI', 7), fg=self.COLORS['text_dim'],
                    bg=self.COLORS['bg_dark']).pack(side='left', padx=1)
        
        # 8 drum rows with trigger indicators
        self.grid_cells = []  # [drum][step] -> label widget
        self.drum_labels = []
        for d in range(8):
            drum_row = tk.Frame(grid_frame, bg=self.COLORS['bg_dark'])
            drum_row.pack(fill='x')
            
            label = tk.Label(drum_row, text=f"D{d+1}", width=6,
                           font=('Segoe UI', 7), fg=self.COLORS['text_dim'],
                           bg=self.COLORS['bg_dark'], anchor='w')
            label.pack(side='left')
            self.drum_labels.append(label)
            
            cells = []
            for s in range(16):
                cell = tk.Label(drum_row, text="", width=2, height=1,
                              font=('Segoe UI', 5),
                              bg=self.COLORS['trigger_off'],
                              relief='flat')
                cell.pack(side='left', padx=1, pady=1)
                cells.append(cell)
            self.grid_cells.append(cells)
        
        # Pattern info + preview
        info_row = tk.Frame(detail_frame, bg=self.COLORS['bg_dark'])
        info_row.pack(fill='x', pady=4)
        
        self.pattern_info_label = tk.Label(info_row, text="Select a pattern to preview",
                                           font=('Segoe UI', 9),
                                           fg=self.COLORS['text'],
                                           bg=self.COLORS['bg_dark'],
                                           anchor='w')
        self.pattern_info_label.pack(side='left', fill='x', expand=True)
        
        self.preview_btn = tk.Button(info_row, text="▶ Preview",
                                     font=('Segoe UI', 9), width=10,
                                     bg=self.COLORS['bg_light'],
                                     fg=self.COLORS['text'],
                                     state='disabled',
                                     command=self._on_toggle_preview)
        self.preview_btn.pack(side='right', padx=4)
        
        # === Drum patches summary ===
        drums_frame = tk.LabelFrame(main, text="Drum Patches", font=('Segoe UI', 9),
                                    fg=self.COLORS['text_dim'],
                                    bg=self.COLORS['bg_medium'],
                                    labelanchor='nw')
        drums_frame.pack(fill='x', pady=(0, 8))
        
        self.drums_text = tk.Label(drums_frame, text="No drums loaded",
                                   font=('Segoe UI', 8),
                                   fg=self.COLORS['text_dim'],
                                   bg=self.COLORS['bg_medium'],
                                   justify='left', anchor='w',
                                   wraplength=650)
        self.drums_text.pack(fill='x', padx=8, pady=8)
        
        # === Action buttons (bottom) ===
        action_frame = tk.Frame(main, bg=self.COLORS['bg_dark'])
        action_frame.pack(fill='x')
        
        self.import_btn = tk.Button(action_frame, text="Import Drums + Patterns",
                                    font=('Segoe UI', 10, 'bold'), width=30,
                                    bg=self.COLORS['accent'], fg=self.COLORS['text'],
                                    state='disabled',
                                    command=self._on_import)
        self.import_btn.pack(side='left', padx=(0, 8))
        
        tk.Button(action_frame, text="Cancel",
                 font=('Segoe UI', 9), width=8,
                 bg=self.COLORS['bg_light'], fg=self.COLORS['text'],
                 command=self._on_cancel).pack(side='right')
    
    # ============================================================
    # Core helpers
    # ============================================================
    
    def _act(self, verb, on_done=None, error_title=None, **args):
        """Start a core verb; on_done(result) runs on the main window's tick,
        an error is shown under ``error_title`` (or on the source line)."""
        def done(event):
            if self._closed:
                return
            if event['status'] == 'done':
                if on_done is not None:
                    on_done(event['result'])
            elif event['status'] == 'error':
                if error_title:
                    messagebox.showerror(error_title, str(event.get('error')),
                                         parent=self.dialog)
                self._show_decoded()
        action_id = self.core.act(verb, **args)
        self.when_done(action_id, done)
        return action_id
    
    def _tick(self):
        """The dialog's own timer: the level meter, the record, monitor and
        preview buttons (a preview ends when the panel starts playing)."""
        if self._closed:
            return
        core = self.core
        listening = core.get('po32.listening')
        recording = core.get('po32.recording')
        if listening:
            self._update_vu_display(core.get('po32.level'))
        elif self._vu_peak:
            self._vu_peak = 0.0
            self._draw_vu_meter(0.0, 0.0)
            self.vu_db_label.config(text="-∞ dB", fg=self.COLORS['text_dim'])
        if recording:
            self.record_btn.config(text="■ Stop", bg='#884444')
            self.source_label.config(
                text=f"Recording {core.get('po32.recorded_seconds'):.1f}s... "
                     f"Play the PO-32 modem signal now.")
        else:
            self.record_btn.config(text="● Record", bg=self.COLORS['bg_light'])
        if listening and not recording:
            self.monitor_btn.config(text="■ Stop", bg='#884444')
        else:
            self.monitor_btn.config(text="🔊 Monitor", bg=self.COLORS['bg_light'])
        if core.get('po32.previewing'):
            self.preview_btn.config(text="■ Stop", bg='#884444')
        else:
            self.preview_btn.config(text="▶ Preview", bg=self.COLORS['bg_light'])
        self._tick_job = self.dialog.after(50, self._tick)  # ~20 fps
    
    # ============================================================
    # Input Device Management
    # ============================================================
    
    def _populate_input_devices(self):
        """Populate the input device dropdown with the core's input devices."""
        names = list(self.core.get('audio.input_devices') or [])
        if not names:
            self.device_combo['values'] = [NO_DEVICES]
            self.device_var.set(NO_DEVICES)
            return
        self.device_combo['values'] = names
        # Prefer the saved preference, then the default, then the first
        for preferred in (self.core.get('pref.audio.input_device'),
                          self.core.get('audio.default_input')):
            if preferred in names:
                self.device_combo.current(names.index(preferred))
                return
        self.device_combo.current(0)
    
    def _selected_device(self):
        """The name of the selected input device (None: the core's choice)."""
        name = self.device_var.get()
        return name if name and name != NO_DEVICES else None
    
    def _on_device_changed(self, event=None):
        """Handle input device selection change — reopen the monitor if active."""
        if self.core.get('po32.listening') and not self.core.get('po32.recording'):
            self._act('po32.listen', error_title="Monitor Error", device=self._selected_device())
    
    def _on_debug_save_changed(self):
        """Handle debug save checkbox change."""
        self._debug_save = self.debug_save_var.get()
        self.core.set('pref.po32.save_recordings', self._debug_save)
    
    def _open_debug_folder(self):
        """Open the debug recordings folder in the system file manager."""
        try:
            debug_folder = self.core.get('po32.recordings_folder')
            os.makedirs(debug_folder, exist_ok=True)
            if os.name == 'nt':
                os.startfile(debug_folder)
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', debug_folder])
            else:
                subprocess.Popen(['xdg-open', debug_folder])
        except Exception as e:
            print(f"[DEBUG] Failed to open folder: {e}", flush=True)
    
    # ============================================================
    # VU Meter & Monitoring
    # ============================================================
    
    def _on_toggle_monitor(self):
        """Toggle live input monitoring (VU meter without recording)."""
        listening = self.core.get('po32.listening') and not self.core.get('po32.recording')
        self._act('po32.listen', error_title="Monitor Error", on=not listening,
                  device=self._selected_device())
    
    def _update_vu_display(self, level):
        """Update the VU meter canvas and dB label from the input level."""
        # Peak hold with decay
        if level >= self._vu_peak:
            self._vu_peak = level
            self._vu_peak_decay = 15  # Hold for ~750ms at 20fps
        else:
            if self._vu_peak_decay > 0:
                self._vu_peak_decay -= 1
            else:
                self._vu_peak *= 0.92  # Slow decay
        
        self._draw_vu_meter(level, self._vu_peak)
        
        # dB display
        if level > 1e-6:
            db = 20 * math.log10(level)
            db_text = f"{db:+.1f} dB"
            if level > 0.95:
                color = '#ff4444'  # Clipping
            elif level > 0.5:
                color = '#ffaa44'  # Hot
            elif level > 0.05:
                color = '#44ff88'  # Good
            else:
                color = self.COLORS['text_dim']  # Low
            self.vu_db_label.config(text=db_text, fg=color)
        else:
            self.vu_db_label.config(text="-∞ dB", fg=self.COLORS['text_dim'])
    
    def _draw_vu_meter(self, level, peak):
        """Draw the VU meter bar on the canvas."""
        self.vu_canvas.delete('all')
        w = self.vu_canvas.winfo_width()
        h = self.vu_canvas.winfo_height()
        if w < 10:
            w = 300  # Default before first layout
        
        # Background segments for reference
        green_end = int(w * 0.6)
        yellow_end = int(w * 0.85)
        
        # Draw background gradient segments
        self.vu_canvas.create_rectangle(0, 0, green_end, h, fill='#0a2a0a', outline='')
        self.vu_canvas.create_rectangle(green_end, 0, yellow_end, h, fill='#2a2a0a', outline='')
        self.vu_canvas.create_rectangle(yellow_end, 0, w, h, fill='#2a0a0a', outline='')
        
        # Draw level bar
        bar_w = int(w * min(level, 1.0))
        if bar_w > 0:
            # Green portion
            g_end = min(bar_w, green_end)
            if g_end > 0:
                self.vu_canvas.create_rectangle(0, 2, g_end, h - 2, fill='#44ff88', outline='')
            # Yellow portion
            if bar_w > green_end:
                y_end = min(bar_w, yellow_end)
                self.vu_canvas.create_rectangle(green_end, 2, y_end, h - 2, fill='#ffcc44', outline='')
            # Red portion
            if bar_w > yellow_end:
                self.vu_canvas.create_rectangle(yellow_end, 2, bar_w, h - 2, fill='#ff4444', outline='')
        
        # Peak indicator line
        peak_x = int(w * min(peak, 1.0))
        if peak_x > 2:
            if peak > 0.85:
                peak_color = '#ff4444'
            elif peak > 0.6:
                peak_color = '#ffcc44'
            else:
                peak_color = '#44ff88'
            self.vu_canvas.create_line(peak_x, 1, peak_x, h - 1, fill=peak_color, width=2)
        
        # Scale markers
        for db_mark in [-40, -20, -12, -6, -3, 0]:
            lin = 10 ** (db_mark / 20.0)
            x = int(w * lin)
            if 0 < x < w:
                self.vu_canvas.create_line(x, 0, x, 3, fill='#666688', width=1)
                self.vu_canvas.create_line(x, h - 3, x, h, fill='#666688', width=1)
    
    # ============================================================
    # Source Loading
    # ============================================================
    
    def _on_import_wav(self):
        """Import a WAV file containing PO-32 modem audio."""
        filename = filedialog.askopenfilename(
            parent=self.dialog,
            title="Import PO-32 Modem Audio",
            filetypes=[
                ('WAV files', '*.wav'),
                ('All files', '*.*'),
            ]
        )
        if not filename:
            return
        
        self.source_label.config(text=f"Decoding: {os.path.basename(filename)}...")
        self._act('po32.decode', self._on_decode_complete, error_title="Decode Error",
                  path=filename)
    
    def _on_toggle_record(self):
        """Toggle audio recording from the selected input device."""
        if self.core.get('po32.recording'):
            self.source_label.config(text="Decoding the recorded audio...")
            self._act('po32.stop', self._on_decode_complete, error_title="Decode Error")
        else:
            self._act('po32.record', error_title="Recording Error",
                      device=self._selected_device())
    
    def _on_decode_complete(self, result):
        """Handle decode completion."""
        if not result.get('drums') and not result.get('patterns'):
            self.source_label.config(text="No audio recorded")
            return
        self._show_decoded()
    
    def _show_decoded(self):
        """Show the core's decode, bank, picks and focus."""
        core = self.core
        summary = core.get('po32.decoded')
        state = core.get('po32.decode')
        if summary is None:
            if state == 'error':
                self.source_label.config(text=f"Error: {core.get('po32.error')}")
            elif not core.get('po32.recording'):
                self.source_label.config(text="No data loaded")
        else:
            card_type = "PO-32 card" if summary['card'] else "Pythonic"
            self.source_label.config(
                text=f"Decoded {card_type}: {summary['drums']} drums, "
                     f"{summary['patterns']} patterns — {summary['source']}")
        banks = core.get('po32.banks')
        if summary is None:
            banks = [True, True]
        self.bank0_radio.config(state='normal' if banks[0] else 'disabled')
        self.bank1_radio.config(state='normal' if banks[1] else 'disabled')
        self.bank_var.set(core.get('po32.bank'))
        self._update_bank_info()
        self._update_drums_display()
        self._update_pattern_buttons()
        self._rebuild_mapping_ui()
        self._update_pattern_grid()
        self._update_import_button_state()
        focus = core.get('po32.focus')
        patterns = core.get('po32.patterns')
        if focus and focus <= len(patterns):
            self.pattern_info_label.config(text=patterns[focus - 1]['summary'])
            self.preview_btn.config(state='normal')
        else:
            self.pattern_info_label.config(text="Select a pattern to preview")
            self.preview_btn.config(state='disabled')
    
    # ============================================================
    # Bank / Pattern Selection
    # ============================================================
    
    def _on_bank_change(self):
        """Handle bank selection change."""
        self._act('po32.select_bank', lambda result: self._show_decoded(),
                  bank=self.bank_var.get())
    
    def _update_bank_info(self):
        """Update bank info label."""
        if self.core.get('po32.decoded') is None:
            self.bank_info_label.config(text="")
            return
        n_patches = sum(1 for sound in self.core.get('po32.sounds') if sound is not None)
        self.bank_info_label.config(text=f"{n_patches}/8 drum patches in this bank")
    
    def _update_pattern_buttons(self):
        """Update pattern button states based on decoded data and picks."""
        patterns = self.core.get('po32.patterns')
        letters = {p['pattern']: p['letter'] for p in self.core.get('po32.picks')}
        focus = self.core.get('po32.focus')
        for i, btn in enumerate(self.pattern_buttons):
            number = i + 1
            if i < len(patterns):
                is_empty = patterns[i]['empty']
                is_focused = number == focus
                if number in letters:
                    btn.config(
                        state='normal',
                        text=f"{number}\u2192{letters[number]}",
                        bg=self.COLORS['highlight'] if is_focused else self.COLORS['accent'],
                        fg=self.COLORS['text'],
                    )
                else:
                    btn.config(
                        state='normal',
                        text=str(number),
                        bg=self.COLORS['highlight'] if is_focused else self.COLORS['bg_light'],
                        fg=self.COLORS['text'] if not is_empty else self.COLORS['text_dim'],
                    )
            else:
                btn.config(state='disabled', text=str(number),
                          fg=self.COLORS['text_dim'], bg=self.COLORS['bg_light'])
    
    def _on_pattern_select(self, idx):
        """Handle pattern button click - toggles the pick and focuses the pattern."""
        self._act('po32.pick', lambda result: self._show_decoded(), pattern=idx + 1)
    
    def _update_pattern_grid(self):
        """Update the step trigger grid of the focused pattern."""
        grid = self.core.get('po32.grid')
        sounds = self.core.get('po32.sounds')
        for d in range(8):
            if sounds[d] is not None:
                short_name = sounds[d].split(',')[0][:10]
                self.drum_labels[d].config(text=f"D{d+1} {short_name}",
                                          fg=self.COLORS['text'])
            else:
                self.drum_labels[d].config(text=f"D{d+1}", fg=self.COLORS['text_dim'])
            steps = grid[d] if grid else [False] * 16
            for s in range(16):
                self.grid_cells[d][s].config(
                    bg=self.COLORS['trigger_on'] if steps[s] else self.COLORS['trigger_off'])
    
    def _update_drums_display(self):
        """Update the drums summary display."""
        if self.core.get('po32.decoded') is None:
            self.drums_text.config(text="No drums loaded")
            return
        lines = [f"Ch{d+1}: {sound if sound is not None else '(empty)'}"
                 for d, sound in enumerate(self.core.get('po32.sounds'))]
        self.drums_text.config(text="  |  ".join(lines))
    
    # ============================================================
    # Multi-Pattern Selection & Mapping
    # ============================================================
    
    def _rebuild_mapping_ui(self):
        """Rebuild the pattern → destination mapping table."""
        for widget in self.mapping_frame.winfo_children():
            widget.destroy()
        
        picks = self.core.get('po32.picks')
        self.selection_count_label.config(text=f"{len(picks)}/12 patterns selected")
        
        cols = 4  # mapping entries per row
        for i, pick in enumerate(picks):
            row_num = i // cols
            col_num = i % cols
            
            cell = tk.Frame(self.mapping_frame, bg=self.COLORS['bg_medium'])
            cell.grid(row=row_num, column=col_num, padx=(0, 10), pady=1, sticky='w')
            
            tk.Label(cell, text=f"#{pick['pattern']} \u2192",
                     font=('Segoe UI', 8), fg=self.COLORS['text'],
                     bg=self.COLORS['bg_medium']).pack(side='left')
            
            dest_var = tk.StringVar(value=pick['letter'])
            menu = tk.OptionMenu(cell, dest_var, *PATTERN_LETTERS,
                                 command=lambda val, n=pick['pattern']:
                                 self._on_destination_change(n, val))
            menu.config(font=('Segoe UI', 7), bg=self.COLORS['bg_light'],
                       fg=self.COLORS['text'], width=2, highlightthickness=0)
            menu.pack(side='left', padx=2)
    
    def _on_destination_change(self, pattern, new_letter):
        """Handle manual change of destination letter — the core swaps on conflict."""
        self._act('po32.pick', lambda result: self._show_decoded(), pattern=pattern,
                  letter=new_letter)
    
    def _on_select_all(self):
        """Select first 12 patterns (preferring non-empty)."""
        if self.core.get('po32.decoded') is None:
            return
        self._act('po32.pick_first', lambda result: self._show_decoded())
    
    def _on_clear_selection(self):
        """Clear pattern selection."""
        self._act('po32.pick_clear', lambda result: self._show_decoded())
    
    def _update_import_button_state(self):
        """Enable/disable the import button and update its label."""
        n = len(self.core.get('po32.picks'))
        if self.core.get('po32.decoded') is not None:
            self.import_btn.config(
                state='normal',
                text=f"Import Drums + {n} Pattern{'s' if n != 1 else ''}"
            )
        else:
            self.import_btn.config(state='disabled', text="Import Drums + Patterns")
    
    # ============================================================
    # Preview Playback
    # ============================================================
    
    def _on_toggle_preview(self):
        """Toggle pattern preview (the core stops the panel's transport)."""
        self._act('po32.preview', error_title="Preview Error",
                  on=not self.core.get('po32.previewing'))
    
    # ============================================================
    # Import
    # ============================================================
    
    def _on_import(self):
        """Import the bank's sounds and the picked patterns (one undo step)."""
        if self.core.get('po32.decoded') is None:
            return
        
        def done(result):
            imported = [f"#{p['pattern']}\u2192{p['letter']}" for p in result['patterns']]
            pattern_desc = ', '.join(imported) if imported else 'none'
            messagebox.showinfo(
                "Import Complete",
                f"Imported {result['drums']} drum patches and "
                f"{len(imported)} pattern{'s' if len(imported) != 1 else ''}: {pattern_desc}",
                parent=self.dialog
            )
            self._on_cancel()
        self._act('po32.import', done, error_title="Import Error")
    
    def _on_cancel(self):
        """Close the dialog: the core closes the input and ends the preview."""
        if self._closed:
            return
        self._closed = True
        if self._tick_job is not None:
            self.dialog.after_cancel(self._tick_job)
        if self.core.get('po32.previewing'):
            self.core.act('po32.preview', on=False)
        if self.core.get('po32.listening') or self.core.get('po32.recording'):
            self.core.act('po32.listen', on=False)
        self.dialog.destroy()
