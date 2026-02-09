"""
PO-32 Import Dialog for Pythonic
Imports patches and patterns from PO-32 Tonic via FSK audio decoding.

Supports:
- Recording from audio input device
- Loading from a local WAV file
- Preview of decoded patches
- Selective channel import
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import numpy as np
import os

from pythonic.po32_codec import (
    decode_wav_file, decode_audio_samples, get_patch_summary,
    DecodedPreset, DecodedPatch, SAMPLE_RATE,
)

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False


# =============================================================================
# Import Dialog
# =============================================================================

class PO32ImportDialog:
    """
    PO-32 Tonic Import Window
    
    GUI dialog for importing sounds and patterns from PO-32 via FSK audio,
    either by recording from an audio input device or loading a WAV file.
    """
    
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
    
    RECORD_DURATION = 12.0  # seconds — enough for a full PO-32 transfer (~9s + margin)
    
    def __init__(self, parent, synth, on_import_callback=None):
        """
        Args:
            parent: Parent tkinter window
            synth: PythonicSynthesizer instance
            on_import_callback: Called with (channel_index, synth_params_dict) for each imported channel
        """
        self.parent = parent
        self.synth = synth
        self.on_import_callback = on_import_callback
        
        # State
        self.decoded_preset: DecodedPreset = None
        self.is_recording = False
        self.record_thread = None
        self._stop_event = threading.Event()
        self.recorded_samples = None
        
        # Create dialog
        self._create_dialog()
    
    def _create_dialog(self):
        """Build the import dialog window."""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("PO-32 Tonic Import")
        self.dialog.configure(bg=self.COLORS['bg'])
        self.dialog.geometry("480x620")
        self.dialog.resizable(False, False)
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        
        main = tk.Frame(self.dialog, bg=self.COLORS['bg'])
        main.pack(fill='both', expand=True, padx=15, pady=15)
        
        # --- Header ---
        tk.Label(main, text="PO-32 Tonic Import",
                font=('Segoe UI', 16, 'bold'),
                fg=self.COLORS['accent_light'],
                bg=self.COLORS['bg']).pack(pady=(0, 10))
        
        # --- Source Selection ---
        source_frame = tk.LabelFrame(main, text="Audio Source",
                                     font=('Segoe UI', 9),
                                     fg=self.COLORS['text'],
                                     bg=self.COLORS['bg'],
                                     bd=1, relief='groove')
        source_frame.pack(fill='x', pady=(0, 10))
        
        # Bank selection row
        bank_row = tk.Frame(source_frame, bg=self.COLORS['bg'])
        bank_row.pack(fill='x', padx=10, pady=(5, 0))

        tk.Label(bank_row, text="Import to sounds:",
                font=('Segoe UI', 9),
                fg=self.COLORS['text'],
                bg=self.COLORS['bg']).pack(side='left')

        self.bank_var = tk.StringVar(value="1 - 8")
        bank_combo = ttk.Combobox(bank_row, textvariable=self.bank_var,
                                  values=["1 - 8", "9 - 16"],
                                  width=8, state='readonly')
        bank_combo.pack(side='right')
        bank_combo.bind('<<ComboboxSelected>>', self._on_bank_change)

        # Instructions
        tk.Label(source_frame,
                text="Put your PO-32 in send mode:\nhold [ write ] + press [ play ]",
                font=('Segoe UI', 9),
                fg=self.COLORS['orange'],
                bg=self.COLORS['bg'],
                pady=5, justify='center').pack(fill='x', padx=10)
        
        # Input device selector
        device_row = tk.Frame(source_frame, bg=self.COLORS['bg'])
        device_row.pack(fill='x', padx=10, pady=5)
        
        tk.Label(device_row, text="Input device:",
                font=('Segoe UI', 9),
                fg=self.COLORS['text'],
                bg=self.COLORS['bg']).pack(side='left')
        
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(device_row, textvariable=self.device_var,
                                         width=30, state='readonly')
        self.device_combo.pack(side='right')
        self._populate_input_devices()
        
        # Buttons row
        btn_row = tk.Frame(source_frame, bg=self.COLORS['bg'])
        btn_row.pack(fill='x', padx=10, pady=(5, 10))
        
        self.record_btn = tk.Button(
            btn_row, text="⏺  Record",
            font=('Segoe UI', 10, 'bold'),
            bg=self.COLORS['red'] if AUDIO_AVAILABLE else self.COLORS['bg_light'],
            fg=self.COLORS['white'],
            activebackground='#ff6666',
            activeforeground=self.COLORS['white'],
            width=14,
            command=self._on_record_click,
            state='normal' if AUDIO_AVAILABLE else 'disabled',
        )
        self.record_btn.pack(side='left', padx=(0, 5))
        
        self.browse_btn = tk.Button(
            btn_row, text="📂  Load WAV",
            font=('Segoe UI', 10),
            bg=self.COLORS['bg_light'],
            fg=self.COLORS['text'],
            activebackground=self.COLORS['bg_medium'],
            width=14,
            command=self._on_browse_wav,
        )
        self.browse_btn.pack(side='left', padx=5)
        
        # --- Progress ---
        progress_frame = tk.Frame(main, bg=self.COLORS['bg'])
        progress_frame.pack(fill='x', pady=(0, 5))
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame,
                                            variable=self.progress_var,
                                            maximum=100, length=440)
        self.progress_bar.pack(fill='x')
        
        self.status_label = tk.Label(progress_frame,
                                     text="Ready — record or load a WAV file",
                                     font=('Segoe UI', 8),
                                     fg=self.COLORS['text_dim'],
                                     bg=self.COLORS['bg'])
        self.status_label.pack(pady=(2, 0))
        
        # --- Decoded Channels Preview ---
        preview_frame = tk.LabelFrame(main, text="Decoded Channels",
                                      font=('Segoe UI', 9),
                                      fg=self.COLORS['text'],
                                      bg=self.COLORS['bg'],
                                      bd=1, relief='groove')
        preview_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        # Scrollable channel list
        canvas = tk.Canvas(preview_frame, bg=self.COLORS['bg'],
                          highlightthickness=0, height=200)
        scrollbar = ttk.Scrollbar(preview_frame, orient='vertical',
                                  command=canvas.yview)
        self.channels_inner = tk.Frame(canvas, bg=self.COLORS['bg'])
        
        self.channels_inner.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.channels_inner, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        scrollbar.pack(side='right', fill='y')
        
        # Channel checkboxes (populated after decode)
        self.channel_vars = []
        self.channel_labels = []
        self._populate_empty_channels()
        
        # --- Import Buttons ---
        import_frame = tk.Frame(main, bg=self.COLORS['bg'])
        import_frame.pack(fill='x', pady=(5, 0))
        
        self.import_btn = tk.Button(
            import_frame, text="⬇  Import Selected",
            font=('Segoe UI', 11, 'bold'),
            bg=self.COLORS['accent'],
            fg=self.COLORS['white'],
            activebackground=self.COLORS['accent_light'],
            activeforeground=self.COLORS['white'],
            width=18,
            command=self._on_import,
            state='disabled',
        )
        self.import_btn.pack(side='left', padx=(0, 5))
        
        # Select All / None
        sel_frame = tk.Frame(import_frame, bg=self.COLORS['bg'])
        sel_frame.pack(side='left', padx=5)
        
        tk.Button(sel_frame, text="All", font=('Segoe UI', 8),
                 bg=self.COLORS['bg_light'], fg=self.COLORS['text'],
                 command=self._select_all, width=4).pack(side='left', padx=1)
        tk.Button(sel_frame, text="None", font=('Segoe UI', 8),
                 bg=self.COLORS['bg_light'], fg=self.COLORS['text'],
                 command=self._select_none, width=4).pack(side='left', padx=1)
        
        tk.Button(
            import_frame, text="Close",
            font=('Segoe UI', 10),
            bg=self.COLORS['bg_light'],
            fg=self.COLORS['text'],
            activebackground=self.COLORS['bg_medium'],
            width=8,
            command=self._on_close,
        ).pack(side='right')
        
        self.dialog.protocol("WM_DELETE_WINDOW", self._on_close)
    
    # -------------------------------------------------------------------------
    # Input Device Management
    # -------------------------------------------------------------------------
    
    def _populate_input_devices(self):
        """Populate the input device dropdown."""
        if not AUDIO_AVAILABLE:
            self.device_combo['values'] = ["(sounddevice not available)"]
            self.device_var.set("(sounddevice not available)")
            return
        
        try:
            devices = sd.query_devices()
            input_devices = []
            default_idx = None
            
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    name = f"{dev['name']}"
                    input_devices.append((i, name))
                    if dev == sd.query_devices(kind='input'):
                        default_idx = len(input_devices) - 1
            
            if input_devices:
                self._input_device_ids = [d[0] for d in input_devices]
                names = [d[1] for d in input_devices]
                self.device_combo['values'] = names
                if default_idx is not None:
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
        if idx >= 0 and hasattr(self, '_input_device_ids') and idx < len(self._input_device_ids):
            return self._input_device_ids[idx]
        return None
    
    # -------------------------------------------------------------------------
    # Recording
    # -------------------------------------------------------------------------
    
    def _on_record_click(self):
        """Handle Record / Stop button click."""
        if self.is_recording:
            self._stop_recording()
        else:
            self._start_recording()
    
    def _start_recording(self):
        """Start recording from the selected input device."""
        device_id = self._get_selected_device_id()
        if device_id is None:
            messagebox.showwarning("No Device",
                                   "No input device selected.",
                                   parent=self.dialog)
            return
        
        self.is_recording = True
        self._stop_event.clear()
        self.recorded_samples = None
        
        # Update UI
        self.record_btn.config(text="■  Stop", bg='#cc4444')
        self.browse_btn.config(state='disabled')
        self.import_btn.config(state='disabled')
        self.status_label.config(text="Recording... Play PO-32 transfer now",
                                fg=self.COLORS['orange'])
        self.progress_var.set(0)
        
        self.record_thread = threading.Thread(
            target=self._record_thread,
            args=(device_id,),
            daemon=True
        )
        self.record_thread.start()
        self._update_record_progress()
    
    def _record_thread(self, device_id):
        """Background recording thread."""
        try:
            total_samples = int(self.RECORD_DURATION * SAMPLE_RATE)
            recording = sd.rec(
                total_samples,
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='float32',
                device=device_id,
            )
            
            # Wait for recording to finish or stop event
            elapsed = 0.0
            check_interval = 0.1
            while elapsed < self.RECORD_DURATION and not self._stop_event.is_set():
                import time
                time.sleep(check_interval)
                elapsed += check_interval
                self._record_progress = min(elapsed / self.RECORD_DURATION * 100, 100)
            
            if self._stop_event.is_set():
                sd.stop()
                # Use what we have so far
                samples_so_far = int(elapsed * SAMPLE_RATE)
                self.recorded_samples = recording[:samples_so_far].flatten()
            else:
                sd.wait()
                self.recorded_samples = recording.flatten()
            
            # Decode
            self.dialog.after(0, self._on_recording_complete)
            
        except Exception as e:
            self.dialog.after(0, lambda: self._on_record_error(str(e)))
    
    def _update_record_progress(self):
        """Update progress bar during recording."""
        if not self.is_recording:
            return
        
        progress = getattr(self, '_record_progress', 0)
        self.progress_var.set(progress)
        
        if progress < 100:
            self.dialog.after(100, self._update_record_progress)
    
    def _on_recording_complete(self):
        """Called when recording finishes."""
        self.is_recording = False
        self.record_btn.config(text="⏺  Record", bg=self.COLORS['red'])
        self.browse_btn.config(state='normal')
        
        if self.recorded_samples is not None and len(self.recorded_samples) > 0:
            self.status_label.config(text="Decoding recorded audio...",
                                    fg=self.COLORS['text'])
            self.dialog.update_idletasks()
            self._decode_samples(self.recorded_samples)
        else:
            self.status_label.config(text="No audio recorded",
                                    fg=self.COLORS['orange'])
    
    def _on_record_error(self, error_msg):
        """Called on recording error."""
        self.is_recording = False
        self.record_btn.config(text="⏺  Record", bg=self.COLORS['red'])
        self.browse_btn.config(state='normal')
        self.status_label.config(text=f"Record error: {error_msg}",
                                fg=self.COLORS['red'])
    
    def _stop_recording(self):
        """Stop ongoing recording."""
        self._stop_event.set()
    
    # -------------------------------------------------------------------------
    # WAV File Loading
    # -------------------------------------------------------------------------
    
    def _on_browse_wav(self):
        """Open file browser for WAV file."""
        filepath = filedialog.askopenfilename(
            parent=self.dialog,
            title="Open PO-32 Transfer Audio",
            filetypes=[
                ("WAV files", "*.wav"),
                ("All audio files", "*.wav *.WAV"),
                ("All files", "*.*"),
            ],
        )
        
        if filepath:
            self.status_label.config(text=f"Loading {os.path.basename(filepath)}...",
                                    fg=self.COLORS['text'])
            self.progress_var.set(50)
            self.dialog.update_idletasks()
            
            preset = decode_wav_file(filepath)
            self._on_decode_result(preset, os.path.basename(filepath))
    
    # -------------------------------------------------------------------------
    # Decode Pipeline
    # -------------------------------------------------------------------------
    
    def _decode_samples(self, samples: np.ndarray):
        """Decode audio samples and update the UI."""
        preset = decode_audio_samples(samples, SAMPLE_RATE)
        self._on_decode_result(preset, "recorded audio")
    
    def _on_decode_result(self, preset: DecodedPreset, source_name: str):
        """Handle decode result."""
        self.decoded_preset = preset
        self.progress_var.set(100)
        
        if preset.error:
            self.status_label.config(
                text=f"Decode failed: {preset.error}",
                fg=self.COLORS['red']
            )
            self.import_btn.config(state='disabled')
            self._populate_empty_channels()
        else:
            n_patches = len(preset.left_patches)
            n_patterns = len(preset.patterns)
            self.status_label.config(
                text=f"Decoded {n_patches} patches, {n_patterns} patterns from {source_name}",
                fg=self.COLORS['green']
            )
            self.import_btn.config(state='normal')
            self._populate_decoded_channels(preset)
    
    # -------------------------------------------------------------------------
    # Channel Preview
    # -------------------------------------------------------------------------
    
    def _populate_empty_channels(self):
        """Show empty channel list."""
        for widget in self.channels_inner.winfo_children():
            widget.destroy()
        self.channel_vars = []
        self.channel_labels = []
        
        for i in range(8):
            row = tk.Frame(self.channels_inner, bg=self.COLORS['bg'])
            row.pack(fill='x', padx=5, pady=1)
            
            var = tk.BooleanVar(value=False)
            self.channel_vars.append(var)
            
            cb = tk.Checkbutton(row, variable=var,
                               bg=self.COLORS['bg'],
                               fg=self.COLORS['text_dim'],
                               selectcolor=self.COLORS['bg_medium'],
                               activebackground=self.COLORS['bg'],
                               state='disabled')
            cb.pack(side='left')
            
            lbl = tk.Label(row, text=f"{i+1}:  —",
                          font=('Segoe UI', 9),
                          fg=self.COLORS['text_dim'],
                          bg=self.COLORS['bg'])
            lbl.pack(side='left', padx=(5, 0))
            self.channel_labels.append(lbl)
    
    def _get_bank_offset(self) -> int:
        """Return 0 for bank 1-8, 8 for bank 9-16."""
        return 0 if self.bank_var.get() == "1 - 8" else 8

    def _on_bank_change(self, event=None):
        """Re-populate the channel list when bank selection changes."""
        if self.decoded_preset and not self.decoded_preset.error:
            self._populate_decoded_channels(self.decoded_preset)

    def _populate_decoded_channels(self, preset: DecodedPreset):
        """Populate channel list with decoded data for the selected bank."""
        for widget in self.channels_inner.winfo_children():
            widget.destroy()
        self.channel_vars = []
        self.channel_labels = []

        bank_offset = self._get_bank_offset()

        for i in range(8):
            po32_idx = i + bank_offset
            row = tk.Frame(self.channels_inner, bg=self.COLORS['bg'])
            row.pack(fill='x', padx=5, pady=2)

            has_patch = po32_idx in preset.left_patches
            var = tk.BooleanVar(value=has_patch)
            self.channel_vars.append(var)

            cb = tk.Checkbutton(row, variable=var,
                               bg=self.COLORS['bg'],
                               fg=self.COLORS['text'],
                               selectcolor=self.COLORS['bg_medium'],
                               activebackground=self.COLORS['bg'],
                               state='normal' if has_patch else 'disabled')
            cb.pack(side='left')

            # Channel number (synth channel) ← PO-32 sound number
            label_text = f"{i+1}:"
            tk.Label(row, text=label_text,
                    font=('Segoe UI', 9, 'bold'),
                    fg=self.COLORS['accent_light'] if has_patch else self.COLORS['text_dim'],
                    bg=self.COLORS['bg'],
                    width=2).pack(side='left')

            # Patch summary
            if has_patch:
                summary = get_patch_summary(preset.left_patches[po32_idx])
                fg = self.COLORS['text']
            else:
                summary = "— (no data)"
                fg = self.COLORS['text_dim']

            lbl = tk.Label(row, text=summary,
                          font=('Segoe UI', 9),
                          fg=fg,
                          bg=self.COLORS['bg'])
            lbl.pack(side='left', padx=(5, 0))
            self.channel_labels.append(lbl)
    
    # -------------------------------------------------------------------------
    # Import
    # -------------------------------------------------------------------------
    
    def _on_import(self):
        """Apply decoded patches to the synth."""
        if self.decoded_preset is None or self.decoded_preset.error:
            return

        bank_offset = self._get_bank_offset()
        imported = 0
        for i in range(8):
            if i < len(self.channel_vars) and self.channel_vars[i].get():
                po32_idx = i + bank_offset
                if po32_idx in self.decoded_preset.left_patches:
                    patch = self.decoded_preset.left_patches[po32_idx]
                    if patch.synth_params:
                        # Apply PO-32 sound (po32_idx) to synth channel i
                        channel = self.synth.channels[i]
                        channel.set_parameters(patch.synth_params, immediate=True)

                        if self.on_import_callback:
                            self.on_import_callback(i, patch.synth_params)

                        imported += 1
        
        if imported > 0:
            self.status_label.config(
                text=f"Imported {imported} patch{'es' if imported != 1 else ''} successfully!",
                fg=self.COLORS['green']
            )
            # Signal success to parent
            self.dialog.after(1500, self._on_close)
        else:
            self.status_label.config(
                text="No channels selected for import",
                fg=self.COLORS['orange']
            )
    
    def _select_all(self):
        """Select all available channels."""
        bank_offset = self._get_bank_offset()
        for i, var in enumerate(self.channel_vars):
            po32_idx = i + bank_offset
            if self.decoded_preset and po32_idx in self.decoded_preset.left_patches:
                var.set(True)
    
    def _select_none(self):
        """Deselect all channels."""
        for var in self.channel_vars:
            var.set(False)
    
    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------
    
    def _on_close(self):
        """Handle dialog close."""
        if self.is_recording:
            self._stop_recording()
            if self.record_thread:
                self.record_thread.join(timeout=2.0)
        
        self.dialog.grab_release()
        self.dialog.destroy()
