"""
PO-32 Import Dialog
Imports drum patches and patterns from PO-32 modem audio (WAV files or live recording).

Supports both Pythonic→PO-32 transfers (8 drums, 2 patterns) and
PO-32→PO-32 transfers (16 drums, 16 patterns).
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import numpy as np
import time
import os
import wave
from datetime import datetime

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

from pythonic.po32_decoder import (
    decode_wav_file, decode_audio_samples, DecodedPreset, DecodedPattern,
    get_pattern_triggers_for_bank, get_pattern_summary, get_patch_summary,
    normalized_to_synth_params, SAMPLE_RATE,
)


class PO32ImportDialog:
    """
    Dialog for importing PO-32 modem audio data.
    
    Provides:
    - WAV file import or live audio recording
    - Bank selection (drums 1-8 or 9-16)
    - Multi-pattern selection with assignable destination letters (A-L)
    - Pre-listen (synthesize and play pattern)
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
    
    def __init__(self, parent, synth, pattern_manager, on_import_callback=None, preferences_manager=None):
        """
        Args:
            parent: Parent tk window
            synth: PythonicSynthesizer instance (for preview playback)
            pattern_manager: PatternManager instance (for import target)
            on_import_callback: Called after successful import to refresh UI
            preferences_manager: PreferencesManager instance (for input device preference)
        """
        self.parent = parent
        self.synth = synth
        self.pattern_manager = pattern_manager
        self.on_import_callback = on_import_callback
        self.preferences_manager = preferences_manager
        
        # State
        self.decoded_preset = None
        self.selected_bank = 0  # 0 or 1
        self.selected_pattern_idx = 0  # Focused pattern for grid/preview
        self.selected_patterns = set()  # Indices selected for import
        self.pattern_destinations = {}  # {po32_idx: letter} e.g. {0: 'A', 2: 'B'}
        self.preview_playing = False
        self.preview_stop_flag = threading.Event()
        self._preview_stream = None
        
        # Recording state
        self.recording = False
        self.recorded_samples = []
        self.record_stream = None
        self._input_device_ids = []
        self._vu_level = 0.0  # Current VU meter level (0.0 - 1.0)
        self._vu_peak = 0.0   # Peak hold level
        self._vu_peak_decay = 0  # Counter for peak hold decay
        self._vu_monitoring = False  # Whether VU monitor stream is active
        self._vu_monitor_stream = None
        self._debug_save = preferences_manager.get('po32_debug_save_recordings', False) if preferences_manager else False
        
        # Build dialog
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Import from PO-32")
        self.dialog.configure(bg=self.COLORS['bg_dark'])
        self.dialog.geometry("720x680")
        self.dialog.minsize(600, 480)
        self.dialog.resizable(True, True)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self._build_ui()
    
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
        if AUDIO_AVAILABLE:
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
        if AUDIO_AVAILABLE:
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
        if AUDIO_AVAILABLE:
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
        
        if AUDIO_AVAILABLE:
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
    # Input Device Management
    # ============================================================
    
    def _populate_input_devices(self):
        """Populate the input device dropdown with available audio input devices."""
        if not AUDIO_AVAILABLE:
            self.device_combo['values'] = ["(sounddevice not available)"]
            self.device_var.set("(sounddevice not available)")
            return
        
        try:
            devices = sd.query_devices()
            input_devices = []
            default_idx = None
            preferred_idx = None
            
            # Get preferred device from preferences
            preferred_name = None
            if self.preferences_manager:
                preferred_name = self.preferences_manager.get('audio_input_device')
            
            default_input = sd.query_devices(kind='input')
            
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    name = dev['name']
                    input_devices.append((i, name))
                    if dev == default_input:
                        default_idx = len(input_devices) - 1
                    if preferred_name and name == preferred_name:
                        preferred_idx = len(input_devices) - 1
            
            if input_devices:
                self._input_device_ids = [d[0] for d in input_devices]
                names = [d[1] for d in input_devices]
                self.device_combo['values'] = names
                # Prefer the saved preference, then default, then first
                if preferred_idx is not None:
                    self.device_combo.current(preferred_idx)
                elif default_idx is not None:
                    self.device_combo.current(default_idx)
                else:
                    self.device_combo.current(0)
            else:
                self._input_device_ids = []
                self.device_combo['values'] = ["(no input devices)"]
                self.device_var.set("(no input devices)")
        except Exception:
            self._input_device_ids = []
            self.device_combo['values'] = ["(error listing devices)"]
            self.device_var.set("(error listing devices)")
    
    def _get_selected_device_id(self):
        """Get the sounddevice ID for the selected input device."""
        idx = self.device_combo.current()
        if idx >= 0 and idx < len(self._input_device_ids):
            return self._input_device_ids[idx]
        return None
    
    def _on_device_changed(self, event=None):
        """Handle input device selection change — restart monitor if active."""
        if self._vu_monitoring:
            self._stop_monitor()
            self._start_monitor()
    
    def _on_debug_save_changed(self):
        """Handle debug save checkbox change."""
        self._debug_save = self.debug_save_var.get()
        if self.preferences_manager:
            self.preferences_manager.set('po32_debug_save_recordings', self._debug_save)
    
    def _get_debug_recordings_folder(self):
        """Get or create the debug recordings folder.
        
        Uses the Documents folder to avoid Windows MSIX/Store Python
        filesystem virtualization that makes AppData writes invisible.
        """
        from pathlib import Path
        documents = os.path.join(Path.home(), 'Documents')
        debug_folder = os.path.join(documents, 'Pythonic Debug Recordings')
        os.makedirs(debug_folder, exist_ok=True)
        return debug_folder
    
    def _save_debug_recording(self, samples, sample_rate):
        """Save recorded audio to a WAV file for debugging."""
        try:
            debug_folder = self._get_debug_recordings_folder()
            print(f"[DEBUG] Debug recordings folder: {debug_folder}", flush=True)
            print(f"[DEBUG] Folder exists: {os.path.exists(debug_folder)}", flush=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'po32_recording_{timestamp}.wav'
            filepath = os.path.join(debug_folder, filename)
            
            # Clamp to [-1, 1] then convert to int16 for WAV
            samples_clamped = np.clip(samples, -1.0, 1.0)
            samples_int16 = (samples_clamped * 32767).astype(np.int16)
            frames_data = samples_int16.tobytes()
            
            print(f"[DEBUG] Writing {len(samples_int16)} samples ({len(frames_data)} bytes) to: {filepath}", flush=True)
            
            wf = wave.open(filepath, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(int(sample_rate))
            wf.writeframes(frames_data)
            wf.close()
            
            # Verify the file actually exists
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                print(f"[DEBUG] Verified: {filepath} ({file_size} bytes)", flush=True)
                self._last_debug_filepath = filepath
                return filepath
            else:
                print(f"[DEBUG] ERROR: File was not created at {filepath}", flush=True)
                return None
        except Exception as e:
            import traceback
            print(f"[DEBUG] Failed to save recording: {e}", flush=True)
            traceback.print_exc()
            return None
    
    def _open_debug_folder(self):
        """Open the debug recordings folder in the system file manager."""
        try:
            debug_folder = self._get_debug_recordings_folder()
            if os.name == 'nt':
                os.startfile(debug_folder)
            elif os.sys.platform == 'darwin':
                import subprocess
                subprocess.Popen(['open', debug_folder])
            else:
                import subprocess
                subprocess.Popen(['xdg-open', debug_folder])
        except Exception as e:
            print(f"[DEBUG] Failed to open folder: {e}", flush=True)
    
    # ============================================================
    # VU Meter & Monitoring
    # ============================================================
    
    def _on_toggle_monitor(self):
        """Toggle live input monitoring (VU meter without recording)."""
        if self._vu_monitoring:
            self._stop_monitor()
        else:
            self._start_monitor()
    
    def _start_monitor(self):
        """Start monitoring the selected input device (VU meter only, no recording)."""
        device_id = self._get_selected_device_id()
        if device_id is None:
            return
        
        self._vu_monitoring = True
        self.monitor_btn.config(text="■ Stop", bg='#884444')
        
        def monitor_callback(indata, frames, time_info, status):
            # Compute RMS level for VU meter
            rms = np.sqrt(np.mean(indata[:, 0] ** 2))
            peak = np.max(np.abs(indata[:, 0]))
            self._vu_level = float(peak)
        
        try:
            self._vu_monitor_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='float32',
                callback=monitor_callback,
                blocksize=2048,
                device=device_id,
            )
            self._vu_monitor_stream.start()
            self._schedule_vu_update()
        except Exception as e:
            self._vu_monitoring = False
            self.monitor_btn.config(text="🔊 Monitor", bg=self.COLORS['bg_light'])
            messagebox.showerror("Monitor Error",
                               f"Failed to open input device:\n{e}",
                               parent=self.dialog)
    
    def _stop_monitor(self):
        """Stop input monitoring."""
        self._vu_monitoring = False
        self.monitor_btn.config(text="🔊 Monitor", bg=self.COLORS['bg_light'])
        
        if self._vu_monitor_stream:
            try:
                self._vu_monitor_stream.stop()
                self._vu_monitor_stream.close()
            except Exception:
                pass
            self._vu_monitor_stream = None
        
        self._vu_level = 0.0
        self._vu_peak = 0.0
        self._draw_vu_meter(0.0, 0.0)
        self.vu_db_label.config(text="-∞ dB", fg=self.COLORS['text_dim'])
    
    def _schedule_vu_update(self):
        """Schedule periodic VU meter UI updates."""
        if self._vu_monitoring or self.recording:
            self._update_vu_display()
            self.dialog.after(50, self._schedule_vu_update)  # ~20 fps
    
    def _update_vu_display(self):
        """Update the VU meter canvas and dB label from current levels."""
        level = self._vu_level
        
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
            db = 20 * np.log10(level)
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
        self.dialog.update()
        
        # Decode in a thread to avoid blocking UI
        def decode():
            preset = decode_wav_file(filename)
            self.dialog.after(0, lambda: self._on_decode_complete(preset, filename))
        
        threading.Thread(target=decode, daemon=True).start()
    
    def _on_toggle_record(self):
        """Toggle audio recording from selected input device."""
        if not AUDIO_AVAILABLE:
            messagebox.showerror("Error", "Audio recording requires the 'sounddevice' library.",
                               parent=self.dialog)
            return
        
        if self.recording:
            self._stop_recording()
        else:
            self._start_recording()
    
    def _start_recording(self):
        """Start recording audio from the selected input device."""
        device_id = self._get_selected_device_id()
        if device_id is None:
            messagebox.showwarning("No Device",
                                   "No input device selected.",
                                   parent=self.dialog)
            return
        
        # Stop monitoring if active (we'll use the recording stream for VU)
        if self._vu_monitoring:
            self._stop_monitor()
        
        self.recording = True
        self.recorded_samples = []
        self.record_btn.config(text="■ Stop", bg='#884444')
        self.source_label.config(text="Recording... Play the PO-32 modem signal now.")
        
        def audio_callback(indata, frames, time_info, status):
            if self.recording:
                self.recorded_samples.append(indata[:, 0].copy())
                # Update VU meter level
                peak = float(np.max(np.abs(indata[:, 0])))
                self._vu_level = peak
        
        try:
            self.record_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='float32',
                callback=audio_callback,
                blocksize=4096,
                device=device_id,
            )
            self.record_stream.start()
            self._schedule_vu_update()
        except Exception as e:
            self.recording = False
            self.record_btn.config(text="● Record", bg=self.COLORS['bg_light'])
            messagebox.showerror("Recording Error", f"Failed to start recording:\n{e}",
                               parent=self.dialog)
    
    def _stop_recording(self):
        """Stop recording and decode the captured audio."""
        self.recording = False
        self.record_btn.config(text="● Record", bg=self.COLORS['bg_light'])
        
        if self.record_stream:
            self.record_stream.stop()
            self.record_stream.close()
            self.record_stream = None
        
        # Reset VU meter
        self._vu_level = 0.0
        self._vu_peak = 0.0
        if hasattr(self, 'vu_canvas'):
            self._draw_vu_meter(0.0, 0.0)
            self.vu_db_label.config(text="-∞ dB", fg=self.COLORS['text_dim'])
        
        if not self.recorded_samples:
            self.source_label.config(text="No audio recorded")
            return
        
        # Concatenate all recorded buffers
        samples = np.concatenate(self.recorded_samples).astype(np.float64)
        duration = len(samples) / SAMPLE_RATE
        
        # Save debug recording if enabled
        saved_path = None
        if self._debug_save:
            saved_path = self._save_debug_recording(samples, SAMPLE_RATE)
            if saved_path:
                status_msg = f"Saved to {os.path.basename(saved_path)} | Decoding {duration:.1f}s..."
            else:
                status_msg = f"Decoding {duration:.1f}s of recorded audio..."
        else:
            status_msg = f"Decoding {duration:.1f}s of recorded audio..."
        
        self.source_label.config(text=status_msg)
        self.dialog.update()
        
        def decode():
            preset = decode_audio_samples(samples, SAMPLE_RATE)
            source_name = f"recorded audio ({os.path.basename(saved_path)})" if saved_path else "recorded audio"
            self.dialog.after(0, lambda: self._on_decode_complete(preset, source_name))
        
        threading.Thread(target=decode, daemon=True).start()
    
    def _on_decode_complete(self, preset: DecodedPreset, source_name: str):
        """Handle decode completion."""
        self.decoded_preset = preset
        
        if preset.error:
            self.source_label.config(text=f"Error: {preset.error}")
            messagebox.showerror("Decode Error", 
                               f"Failed to decode PO-32 data:\n{preset.error}",
                               parent=self.dialog)
            return
        
        n_drums = len(preset.left_patches)
        n_patterns = len(preset.decoded_patterns)
        base_name = os.path.basename(source_name) if os.path.sep in source_name or '/' in source_name else source_name
        
        is_po32_card = n_drums > 8 or n_patterns > 2
        card_type = "PO-32 card" if is_po32_card else "Pythonic"
        
        self.source_label.config(
            text=f"Decoded {card_type}: {n_drums} drums, {n_patterns} patterns — {base_name}"
        )
        
        # Update bank availability
        has_bank0 = any(d < 8 for d in preset.left_patches.keys())
        has_bank1 = any(d >= 8 for d in preset.left_patches.keys())
        
        self.bank0_radio.config(state='normal' if has_bank0 else 'disabled')
        self.bank1_radio.config(state='normal' if has_bank1 else 'disabled')
        
        if has_bank0:
            self.bank_var.set(0)
        elif has_bank1:
            self.bank_var.set(1)
        
        self._update_bank_info()
        self._update_drums_display()
        
        # Auto-select first 12 non-empty patterns with default letter assignments
        self.selected_patterns = set()
        self.pattern_destinations = {}
        non_empty = [i for i, dp in enumerate(preset.decoded_patterns)
                     if any(t != 0 for t in dp.triggers)]
        to_select = non_empty[:12] if non_empty else list(range(min(12, len(preset.decoded_patterns))))
        self.selected_patterns = set(to_select)
        self._auto_assign_destinations()
        self._update_pattern_buttons()
        self._rebuild_mapping_ui()
        self._update_import_button_state()
        
        # Focus the first selected pattern for grid display
        if to_select:
            self.selected_pattern_idx = to_select[0]
            self._update_pattern_grid()
            dp = preset.decoded_patterns[to_select[0]]
            self.pattern_info_label.config(text=get_pattern_summary(dp))
            self.preview_btn.config(state='normal')
    
    # ============================================================
    # Bank / Pattern Selection
    # ============================================================
    
    def _on_bank_change(self):
        """Handle bank selection change."""
        self.selected_bank = self.bank_var.get()
        self._update_bank_info()
        self._update_drums_display()
        self._update_pattern_grid()
    
    def _update_bank_info(self):
        """Update bank info label."""
        if not self.decoded_preset:
            return
        bank = self.bank_var.get()
        bank_offset = bank * 8
        n_patches = sum(1 for d in range(bank_offset, bank_offset + 8) 
                       if d in self.decoded_preset.left_patches)
        self.bank_info_label.config(text=f"{n_patches}/8 drum patches in this bank")
    
    def _update_pattern_buttons(self):
        """Update pattern button states based on decoded data and selection."""
        if not self.decoded_preset:
            return
        
        for i, btn in enumerate(self.pattern_buttons):
            if i < len(self.decoded_preset.decoded_patterns):
                dp = self.decoded_preset.decoded_patterns[i]
                is_empty = all(t == 0 for t in dp.triggers)
                is_selected = i in self.selected_patterns
                is_focused = i == self.selected_pattern_idx
                
                if is_selected:
                    letter = self.pattern_destinations.get(i, '?')
                    btn.config(
                        state='normal',
                        text=f"{i+1}\u2192{letter}",
                        bg=self.COLORS['highlight'] if is_focused else self.COLORS['accent'],
                        fg=self.COLORS['text'],
                    )
                else:
                    btn.config(
                        state='normal',
                        text=str(i + 1),
                        bg=self.COLORS['highlight'] if is_focused else self.COLORS['bg_light'],
                        fg=self.COLORS['text'] if not is_empty else self.COLORS['text_dim'],
                    )
            else:
                btn.config(state='disabled', text=str(i + 1),
                          fg=self.COLORS['text_dim'], bg=self.COLORS['bg_light'])
    
    def _on_pattern_select(self, idx):
        """Handle pattern button click - toggles selection and focuses the pattern."""
        if not self.decoded_preset or idx >= len(self.decoded_preset.decoded_patterns):
            return
        
        # Stop any preview when switching
        if self.preview_playing:
            self._stop_preview()
        
        # Toggle selection
        if idx in self.selected_patterns:
            self.selected_patterns.discard(idx)
            if idx in self.pattern_destinations:
                del self.pattern_destinations[idx]
        else:
            if len(self.selected_patterns) >= 12:
                return  # Max 12 patterns (Pythonic limit)
            self.selected_patterns.add(idx)
            # Assign next available letter
            used = set(self.pattern_destinations.values())
            for letter in self.pattern_manager.PATTERN_NAMES:
                if letter not in used:
                    self.pattern_destinations[idx] = letter
                    break
        
        # Always set as focused pattern for grid / preview
        self.selected_pattern_idx = idx
        
        # Update all UI
        self._update_pattern_buttons()
        self._rebuild_mapping_ui()
        self._update_pattern_grid()
        self._update_import_button_state()
        
        dp = self.decoded_preset.decoded_patterns[idx]
        self.pattern_info_label.config(text=get_pattern_summary(dp))
        self.preview_btn.config(state='normal')
    
    def _update_pattern_grid(self):
        """Update the step trigger grid visualization."""
        if not self.decoded_preset or self.selected_pattern_idx >= len(self.decoded_preset.decoded_patterns):
            # Clear grid
            for d in range(8):
                for s in range(16):
                    self.grid_cells[d][s].config(bg=self.COLORS['trigger_off'])
            return
        
        dp = self.decoded_preset.decoded_patterns[self.selected_pattern_idx]
        bank = self.bank_var.get()
        triggers = get_pattern_triggers_for_bank(dp, bank)
        bank_offset = bank * 8
        
        for d in range(8):
            drum_idx = d + bank_offset
            # Update drum label with patch summary if available
            if self.decoded_preset and drum_idx in self.decoded_preset.left_patches:
                patch = self.decoded_preset.left_patches[drum_idx]
                short_name = get_patch_summary(patch).split(',')[0][:10]
                self.drum_labels[d].config(text=f"D{d+1} {short_name}",
                                          fg=self.COLORS['text'])
            else:
                self.drum_labels[d].config(text=f"D{d+1}", fg=self.COLORS['text_dim'])
            
            # Update step cells
            step_triggers = triggers.get(d, [False] * 16)
            for s in range(16):
                if step_triggers[s]:
                    self.grid_cells[d][s].config(bg=self.COLORS['trigger_on'])
                else:
                    self.grid_cells[d][s].config(bg=self.COLORS['trigger_off'])
    
    def _update_drums_display(self):
        """Update the drums summary display."""
        if not self.decoded_preset:
            self.drums_text.config(text="No drums loaded")
            return
        
        bank = self.bank_var.get()
        bank_offset = bank * 8
        lines = []
        for d in range(8):
            drum_idx = d + bank_offset
            if drum_idx in self.decoded_preset.left_patches:
                patch = self.decoded_preset.left_patches[drum_idx]
                summary = get_patch_summary(patch)
                lines.append(f"Ch{d+1}: {summary}")
            else:
                lines.append(f"Ch{d+1}: (empty)")
        
        self.drums_text.config(text="  |  ".join(lines))
    
    # ============================================================
    # Multi-Pattern Selection & Mapping
    # ============================================================
    
    def _auto_assign_destinations(self):
        """Auto-assign Pythonic letters A-L to selected patterns in order."""
        letters = self.pattern_manager.PATTERN_NAMES  # ['A', ..., 'L']
        sorted_selected = sorted(self.selected_patterns)
        self.pattern_destinations = {}
        for i, pat_idx in enumerate(sorted_selected):
            if i < len(letters):
                self.pattern_destinations[pat_idx] = letters[i]
    
    def _rebuild_mapping_ui(self):
        """Rebuild the pattern → destination mapping table."""
        for widget in self.mapping_frame.winfo_children():
            widget.destroy()
        
        n = len(self.selected_patterns)
        self.selection_count_label.config(text=f"{n}/12 patterns selected")
        
        if not self.selected_patterns:
            return
        
        sorted_selected = sorted(self.selected_patterns)
        letters = self.pattern_manager.PATTERN_NAMES  # ['A', ..., 'L']
        
        cols = 4  # mapping entries per row
        for i, pat_idx in enumerate(sorted_selected):
            row_num = i // cols
            col_num = i % cols
            
            cell = tk.Frame(self.mapping_frame, bg=self.COLORS['bg_medium'])
            cell.grid(row=row_num, column=col_num, padx=(0, 10), pady=1, sticky='w')
            
            tk.Label(cell, text=f"#{pat_idx+1} \u2192",
                     font=('Segoe UI', 8), fg=self.COLORS['text'],
                     bg=self.COLORS['bg_medium']).pack(side='left')
            
            dest_var = tk.StringVar(value=self.pattern_destinations.get(pat_idx, '?'))
            menu = tk.OptionMenu(cell, dest_var, *letters,
                                 command=lambda val, idx=pat_idx: self._on_destination_change(idx, val))
            menu.config(font=('Segoe UI', 7), bg=self.COLORS['bg_light'],
                       fg=self.COLORS['text'], width=2, highlightthickness=0)
            menu.pack(side='left', padx=2)
    
    def _on_destination_change(self, pat_idx, new_letter):
        """Handle manual change of destination letter — swaps on conflict."""
        old_letter = self.pattern_destinations.get(pat_idx, '')
        # Find conflict and swap
        for other_idx, letter in self.pattern_destinations.items():
            if other_idx != pat_idx and letter == new_letter:
                if old_letter:
                    self.pattern_destinations[other_idx] = old_letter
                break
        self.pattern_destinations[pat_idx] = new_letter
        self._update_pattern_buttons()
        self._rebuild_mapping_ui()
    
    def _on_select_all(self):
        """Select first 12 patterns (preferring non-empty)."""
        if not self.decoded_preset:
            return
        non_empty = [i for i, dp in enumerate(self.decoded_preset.decoded_patterns)
                     if any(t != 0 for t in dp.triggers)]
        all_indices = list(range(len(self.decoded_preset.decoded_patterns)))
        to_select = non_empty[:12]
        if len(to_select) < 12:
            remaining = [i for i in all_indices if i not in to_select]
            to_select.extend(remaining[:12 - len(to_select)])
        self.selected_patterns = set(to_select)
        self._auto_assign_destinations()
        self._update_pattern_buttons()
        self._rebuild_mapping_ui()
        self._update_import_button_state()
    
    def _on_clear_selection(self):
        """Clear pattern selection."""
        self.selected_patterns = set()
        self.pattern_destinations = {}
        self._update_pattern_buttons()
        self._rebuild_mapping_ui()
        self._update_import_button_state()
    
    def _update_import_button_state(self):
        """Enable/disable the import button and update its label."""
        n = len(self.selected_patterns)
        if n > 0 or (self.decoded_preset and self.decoded_preset.left_patches):
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
        """Toggle pattern preview playback."""
        if self.preview_playing:
            self._stop_preview()
        else:
            self._start_preview()
    
    def _start_preview(self):
        """Start playing preview of the selected pattern using a real-time
        audio callback, matching the architecture of the main pattern player."""
        if not self.decoded_preset or not AUDIO_AVAILABLE:
            return

        if self.selected_pattern_idx >= len(self.decoded_preset.decoded_patterns):
            return

        dp = self.decoded_preset.decoded_patterns[self.selected_pattern_idx]
        bank = self.bank_var.get()
        triggers = get_pattern_triggers_for_bank(dp, bank)
        bank_offset = bank * 8

        # Create a dedicated preview synth (avoid touching the main synth)
        from pythonic.synthesizer import PythonicSynthesizer
        preview_synth = PythonicSynthesizer(SAMPLE_RATE)

        # Apply decoded patches
        for d in range(8):
            drum_idx = d + bank_offset
            if drum_idx in self.decoded_preset.left_patches:
                patch = self.decoded_preset.left_patches[drum_idx]
                if patch.synth_params:
                    preview_synth.channels[d].set_parameters(patch.synth_params)

        # Pre-compute flat trigger table: triggers_table[step] = list of drum indices
        triggers_table = []
        for step in range(16):
            active = []
            for d in range(8):
                step_triggers = triggers.get(d, [False] * 16)
                if step_triggers[step]:
                    active.append(d)
            triggers_table.append(active)

        # Timing
        bpm = self.pattern_manager.bpm
        step_duration_s = 60.0 / bpm / 4.0  # 16th notes
        step_samples = int(step_duration_s * SAMPLE_RATE)

        # Shared mutable state for the callback (no locks needed – single writer)
        cb_state = {
            'synth': preview_synth,
            'step': 0,
            'frame_in_step': 0,
            'step_samples': step_samples,
            'triggers': triggers_table,
        }

        # Trigger step 0 immediately so the first beat is audible
        for d in triggers_table[0]:
            preview_synth.trigger_drum(d, velocity=100)

        def _preview_callback(outdata, frames, time_info, status):
            """Real-time audio callback – runs on the audio thread."""
            synth = cb_state['synth']
            s_step = cb_state['step']
            s_frame = cb_state['frame_in_step']
            s_step_samples = cb_state['step_samples']
            t_table = cb_state['triggers']

            written = 0
            while written < frames:
                remaining_in_step = s_step_samples - s_frame
                chunk = min(remaining_in_step, frames - written)

                audio = synth.process_audio(chunk)
                outdata[written:written + chunk] = audio

                s_frame += chunk
                written += chunk

                # Advance to next step?
                if s_frame >= s_step_samples:
                    s_frame = 0
                    s_step = (s_step + 1) % 16
                    for d in t_table[s_step]:
                        synth.trigger_drum(d, velocity=100)

            cb_state['step'] = s_step
            cb_state['frame_in_step'] = s_frame

        # Open a real-time output stream (same buffer size as main player)
        try:
            self._preview_stream = sd.OutputStream(
                channels=2,
                callback=_preview_callback,
                samplerate=SAMPLE_RATE,
                blocksize=1050,
                dtype=np.float32,
            )
            self._preview_stream.start()
        except Exception as e:
            print(f"Preview audio error: {e}", flush=True)
            return

        self.preview_playing = True
        self.preview_stop_flag.clear()
        self.preview_btn.config(text="■ Stop", bg='#884444')

    def _stop_preview(self):
        """Stop preview playback and release the audio stream."""
        self.preview_playing = False
        self.preview_stop_flag.set()
        self.preview_btn.config(text="▶ Preview", bg=self.COLORS['bg_light'])

        if hasattr(self, '_preview_stream') and self._preview_stream is not None:
            try:
                self._preview_stream.stop()
                self._preview_stream.close()
            except Exception:
                pass
            self._preview_stream = None
    
    # ============================================================
    # Import
    # ============================================================
    
    def _on_import(self):
        """Import selected patterns and drum patches into Pythonic."""
        if not self.decoded_preset:
            return
        
        # Stop preview if playing
        if self.preview_playing:
            self._stop_preview()
        
        bank = self.bank_var.get()
        bank_offset = bank * 8
        
        # Import drum patches (8 drums from selected bank)
        imported_drums = 0
        for d in range(8):
            drum_idx = d + bank_offset
            if drum_idx in self.decoded_preset.left_patches:
                patch = self.decoded_preset.left_patches[drum_idx]
                if patch.synth_params:
                    self.synth.channels[d].set_parameters(patch.synth_params)
                    imported_drums += 1
        
        # Import morph endpoints if right patches (morph B) are available
        # Access morph_manager from the main window if available
        main_window = getattr(self.parent, 'master', None) or self.parent
        morph_manager = None
        # Walk up the widget tree to find the PythonicGUI instance
        if hasattr(main_window, 'morph_manager'):
            morph_manager = main_window.morph_manager
        else:
            # Try to find it via parent's attributes
            for attr_name in dir(self.parent):
                obj = getattr(self.parent, attr_name, None)
                if hasattr(obj, 'morph_manager'):
                    morph_manager = obj.morph_manager
                    break
        
        if morph_manager is not None:
            # Set endpoint A from left patches (current state)
            morph_manager.capture_endpoint_a()
            
            # If right patches exist, set them as endpoint B
            has_right = any(
                (d + bank_offset) in self.decoded_preset.right_patches 
                for d in range(8)
            )
            if has_right:
                # Temporarily apply right patches, capture as B, then restore A
                right_params_applied = False
                for d in range(8):
                    drum_idx = d + bank_offset
                    if drum_idx in self.decoded_preset.right_patches:
                        patch = self.decoded_preset.right_patches[drum_idx]
                        if patch.synth_params:
                            self.synth.channels[d].set_parameters(patch.synth_params)
                            right_params_applied = True
                
                if right_params_applied:
                    morph_manager.capture_endpoint_b()
                    # Restore left patches (endpoint A)
                    for d in range(8):
                        drum_idx = d + bank_offset
                        if drum_idx in self.decoded_preset.left_patches:
                            patch = self.decoded_preset.left_patches[drum_idx]
                            if patch.synth_params:
                                self.synth.channels[d].set_parameters(patch.synth_params)
            else:
                # No right patches - both endpoints are the same
                morph_manager.capture_endpoint_b()
        
        # Import all selected patterns to their assigned destinations
        # First, clear ALL 12 Pythonic patterns so non-imported ones are empty
        letters = self.pattern_manager.PATTERN_NAMES
        for i in range(len(letters)):
            pat = self.pattern_manager.get_pattern(i)
            pat.clear()
        
        imported_patterns = []
        for pat_idx in sorted(self.selected_patterns):
            if pat_idx >= len(self.decoded_preset.decoded_patterns):
                continue
            dest_letter = self.pattern_destinations.get(pat_idx)
            if not dest_letter or dest_letter not in letters:
                continue
            
            dest_index = letters.index(dest_letter)
            dp = self.decoded_preset.decoded_patterns[pat_idx]
            triggers = get_pattern_triggers_for_bank(dp, bank)
            
            target_pattern = self.pattern_manager.get_pattern(dest_index)
            for d in range(8):
                channel = target_pattern.get_channel(d)
                step_triggers = triggers.get(d, [False] * 16)
                for s in range(16):
                    channel.steps[s].trigger = step_triggers[s]
                    channel.steps[s].accent = False
                    channel.steps[s].fill = False
                    channel.steps[s].probability = 100
                    channel.steps[s].substeps = ""
            
            imported_patterns.append(f"#{pat_idx+1}\u2192{dest_letter}")
        
        # Callback to refresh UI
        if self.on_import_callback:
            self.on_import_callback()
        
        pattern_desc = ', '.join(imported_patterns) if imported_patterns else 'none'
        messagebox.showinfo(
            "Import Complete",
            f"Imported {imported_drums} drum patches and "
            f"{len(imported_patterns)} pattern{'s' if len(imported_patterns) != 1 else ''}: {pattern_desc}",
            parent=self.dialog
        )
        
        # Stop monitoring before closing
        if self._vu_monitoring:
            self._stop_monitor()
        
        self.dialog.destroy()
    
    def _on_cancel(self):
        """Cancel and close dialog."""
        if self.preview_playing:
            self._stop_preview()
        if self.recording:
            self._stop_recording()
        if self._vu_monitoring:
            self._stop_monitor()
        self.dialog.destroy()
