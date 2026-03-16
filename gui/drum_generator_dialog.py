"""
AI Drum Generator Dialog

TR-8-inspired 8-lane interface for generating drum patches using a CVAE model.
Supports per-slot generation, one-shot preview, pattern-loop preview, and
selective apply back to the live kit.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from queue import Empty, Full, Queue
import numpy as np

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False

from pythonic.drum_generator import (
    PatchGenerator, SLOT_MAP, DRUM_TYPES,
    is_torch_available, install_ml_dependencies,
)
from pythonic.pattern_generator import PatternGenerator, PATTERN_NAMES
from pythonic.preset_manager import (
    convert_drum_patch_data, apply_drum_patch_to_channel, channel_to_raw_patch,
)


SAMPLE_RATE = 44100
PREVIEW_BLOCK_SIZE = 1050
PREVIEW_QUEUE_BLOCKS = 8

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
    'orange': '#ffaa44',
    'slot_bg': '#333348',
    'slot_active': '#3a3a5a',
    'generate_btn': '#446644',
    'apply_btn': '#664444',
}


class BufferedPatternPreviewSource:
    """Render preview audio off the real-time callback and feed the main stream."""

    def __init__(self, synth, trigger_tables: list, bpm: float,
                 sample_rate: int = SAMPLE_RATE,
                 block_size: int = PREVIEW_BLOCK_SIZE,
                 queue_blocks: int = PREVIEW_QUEUE_BLOCKS):
        self.synth = synth
        self.trigger_tables = trigger_tables or [[[] for _ in range(16)]]
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.step_samples = max(1, int((60.0 / bpm / 4.0) * sample_rate))

        self._queue = Queue(maxsize=max(2, queue_blocks))
        self._stop_event = threading.Event()
        self._thread = None
        self._current_block = None
        self._current_offset = 0
        self._read_buffer = np.zeros((block_size, 2), dtype=np.float32)

        self._step = 0
        self._frame_in_step = 0
        self._table_idx = 0
        self._needs_initial_trigger = True

    def start(self, prefill_blocks: int = 2):
        """Prime a few blocks synchronously, then keep rendering in the background."""
        target_blocks = min(prefill_blocks, self._queue.maxsize)
        while self._queue.qsize() < target_blocks:
            self._queue.put_nowait(self._render_block())

        self._thread = threading.Thread(
            target=self._render_loop,
            name="drum-preview-render",
            daemon=True,
        )
        self._thread.start()

    def read(self, num_samples: int) -> np.ndarray:
        """Return the next preview block without blocking the audio callback."""
        if num_samples > len(self._read_buffer):
            self._read_buffer = np.zeros((num_samples, 2), dtype=np.float32)

        output = self._read_buffer[:num_samples]
        output.fill(0)

        written = 0
        while written < num_samples:
            if self._current_block is None or self._current_offset >= len(self._current_block):
                try:
                    self._current_block = self._queue.get_nowait()
                    self._current_offset = 0
                except Empty:
                    break

            available = len(self._current_block) - self._current_offset
            chunk = min(available, num_samples - written)
            output[written:written + chunk] = self._current_block[
                self._current_offset:self._current_offset + chunk
            ]
            written += chunk
            self._current_offset += chunk

            if self._current_offset >= len(self._current_block):
                self._current_block = None
                self._current_offset = 0

        return output

    def stop(self):
        """Stop the background renderer and release queued audio blocks."""
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self._thread = None
        self._current_block = None
        self._current_offset = 0

        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                break

    def _render_loop(self):
        while not self._stop_event.is_set():
            try:
                block = self._render_block()
                self._queue.put(block, timeout=0.05)
            except Full:
                continue
            except Exception as exc:
                print(f"Preview render error: {exc}", flush=True)
                self._stop_event.set()

    def _render_block(self) -> np.ndarray:
        output = np.zeros((self.block_size, 2), dtype=np.float32)

        if self._needs_initial_trigger:
            self._trigger_current_step()
            self._needs_initial_trigger = False

        written = 0
        while written < self.block_size:
            remaining_in_step = self.step_samples - self._frame_in_step
            chunk = min(remaining_in_step, self.block_size - written)
            output[written:written + chunk] = self.synth.process_audio(chunk)
            self._frame_in_step += chunk
            written += chunk

            if self._frame_in_step >= self.step_samples:
                self._frame_in_step = 0
                self._step = (self._step + 1) % 16
                if self._step == 0 and len(self.trigger_tables) > 1:
                    self._table_idx = (self._table_idx + 1) % len(self.trigger_tables)
                self._trigger_current_step()

        return output

    def _trigger_current_step(self):
        table = self.trigger_tables[self._table_idx]
        for drum_idx in table[self._step]:
            self.synth.trigger_drum(drum_idx, velocity=100)


class DrumGeneratorDialog:
    """TR-8-inspired 8-lane AI drum generator dialog."""

    def __init__(self, parent, synth, pattern_manager, preferences_manager,
                 on_apply_callback=None):
        self.parent = parent
        self.synth = synth
        self.pattern_manager = pattern_manager
        self.preferences_manager = preferences_manager
        self.on_apply_callback = on_apply_callback

        self.generator = PatchGenerator()
        self.pattern_gen = PatternGenerator()

        # Per-slot state: list of 8 dicts
        # Each: {candidates: [raw_patch_dict, ...], selected_idx: int, type_override: str|None}
        self.slot_state = [
            {"candidates": [], "selected_idx": 0, "type_override": None}
            for _ in range(8)
        ]

        # Pattern handling mode: 'keep' or 'generate'
        self._pattern_mode = 'keep'
        # Cached generated pattern bank {name: pattern_data}
        self._cached_pattern_bank = None

        # Preview state
        self.preview_playing = False
        self._preview_renderer = None
        self.preview_stop_flag = threading.Event()

        self._build_dialog()
        self._update_model_status()

    # ================================================================
    # Dialog Layout
    # ================================================================

    def _build_dialog(self):
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("AI Drum Generator")
        self.dialog.geometry("920x680")
        self.dialog.resizable(True, True)
        self.dialog.transient(self.parent)
        self.dialog.configure(bg=COLORS['bg_dark'])
        self.dialog.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── Top bar: model status + global controls ──
        self._build_top_bar()

        # ── 8-lane slot area ──
        self._build_slot_area()

        # ── Bottom bar: preview + apply ──
        self._build_bottom_bar()

        # Center on parent
        self.dialog.update_idletasks()
        x = self.parent.winfo_x() + (self.parent.winfo_width() - self.dialog.winfo_width()) // 2
        y = self.parent.winfo_y() + (self.parent.winfo_height() - self.dialog.winfo_height()) // 2
        self.dialog.geometry(f"+{x}+{y}")

    # ── Top bar ──────────────────────────────────────────────────────

    def _build_top_bar(self):
        top = tk.Frame(self.dialog, bg=COLORS['bg_dark'])
        top.pack(fill='x', padx=10, pady=(10, 5))

        # Title
        tk.Label(top, text="AI Drum Generator", font=('Segoe UI', 13, 'bold'),
                 fg=COLORS['text'], bg=COLORS['bg_dark']).pack(side='left')

        # Right side controls
        right = tk.Frame(top, bg=COLORS['bg_dark'])
        right.pack(side='right')

        self.install_btn = tk.Button(right, text="Install ML Support",
                                     command=self._on_install_ml,
                                     bg=COLORS['bg_light'], fg=COLORS['text'],
                                     font=('Segoe UI', 8), relief='flat', padx=6)
        self.install_btn.pack(side='left', padx=2)

        # ── Model row: patch + pattern ──
        model_row = tk.Frame(self.dialog, bg=COLORS['bg_dark'])
        model_row.pack(fill='x', padx=10, pady=(0, 2))

        # Patch model
        tk.Label(model_row, text="Patch Model:", font=('Segoe UI', 8),
                 fg=COLORS['text_dim'], bg=COLORS['bg_dark']).pack(side='left')
        self.model_status_label = tk.Label(model_row, text="not loaded",
                                           font=('Segoe UI', 8),
                                           fg=COLORS['text_dim'], bg=COLORS['bg_dark'])
        self.model_status_label.pack(side='left', padx=(2, 4))
        tk.Button(model_row, text="Load...", command=self._on_load_model,
                  bg=COLORS['bg_light'], fg=COLORS['text'],
                  font=('Segoe UI', 8), relief='flat', padx=4).pack(side='left', padx=(0, 12))

        # Pattern model
        tk.Label(model_row, text="Pattern Model:", font=('Segoe UI', 8),
                 fg=COLORS['text_dim'], bg=COLORS['bg_dark']).pack(side='left')
        self.pattern_model_status_label = tk.Label(model_row, text="not loaded",
                                                   font=('Segoe UI', 8),
                                                   fg=COLORS['text_dim'],
                                                   bg=COLORS['bg_dark'])
        self.pattern_model_status_label.pack(side='left', padx=(2, 4))
        tk.Button(model_row, text="Load...", command=self._on_load_pattern_model,
                  bg=COLORS['bg_light'], fg=COLORS['text'],
                  font=('Segoe UI', 8), relief='flat', padx=4).pack(side='left')

        # ── Controls row ──
        ctrl = tk.Frame(self.dialog, bg=COLORS['bg_dark'])
        ctrl.pack(fill='x', padx=10, pady=(0, 5))

        # Patch temperature
        tk.Label(ctrl, text="Patch Temp:", font=('Segoe UI', 9),
                 fg=COLORS['text'], bg=COLORS['bg_dark']).pack(side='left')
        saved_patch_temp = self.preferences_manager.get('drum_generator_patch_temperature', 1.0)
        self.temp_var = tk.DoubleVar(value=saved_patch_temp)
        temp_spin = tk.Spinbox(ctrl, from_=0.1, to=3.0, increment=0.1,
                               textvariable=self.temp_var, width=5,
                               font=('Segoe UI', 9),
                               bg=COLORS['bg_medium'], fg=COLORS['text'],
                               buttonbackground=COLORS['bg_light'],
                               insertbackground=COLORS['text'])
        temp_spin.pack(side='left', padx=(2, 12))

        # Candidates per slot
        tk.Label(ctrl, text="Candidates:", font=('Segoe UI', 9),
                 fg=COLORS['text'], bg=COLORS['bg_dark']).pack(side='left')
        self.candidates_var = tk.IntVar(value=8)
        cand_spin = tk.Spinbox(ctrl, from_=1, to=32, increment=1,
                               textvariable=self.candidates_var, width=4,
                               font=('Segoe UI', 9),
                               bg=COLORS['bg_medium'], fg=COLORS['text'],
                               buttonbackground=COLORS['bg_light'],
                               insertbackground=COLORS['text'])
        cand_spin.pack(side='left', padx=(2, 12))

        # Random seed
        tk.Label(ctrl, text="Seed:", font=('Segoe UI', 9),
                 fg=COLORS['text'], bg=COLORS['bg_dark']).pack(side='left')
        self.seed_var = tk.StringVar(value="")
        seed_entry = tk.Entry(ctrl, textvariable=self.seed_var, width=8,
                              font=('Segoe UI', 9),
                              bg=COLORS['bg_medium'], fg=COLORS['text'],
                              insertbackground=COLORS['text'])
        seed_entry.pack(side='left', padx=(2, 4))
        tk.Button(ctrl, text="Reseed", command=self._on_reseed,
                  bg=COLORS['bg_light'], fg=COLORS['text'],
                  font=('Segoe UI', 8), relief='flat', padx=4).pack(side='left', padx=(0, 12))

        # Generate All button
        tk.Button(ctrl, text="Generate All 8", command=self._on_generate_all,
                  bg=COLORS['generate_btn'], fg=COLORS['text'],
                  font=('Segoe UI', 9, 'bold'), relief='flat',
                  padx=8).pack(side='right')

        # ── Pattern handling row ──
        pat_row = tk.Frame(self.dialog, bg=COLORS['bg_dark'])
        pat_row.pack(fill='x', padx=10, pady=(0, 5))

        tk.Label(pat_row, text="Patterns:", font=('Segoe UI', 9),
                 fg=COLORS['text'], bg=COLORS['bg_dark']).pack(side='left')
        self._pattern_mode_var = tk.StringVar(value='keep')
        self._pattern_mode_var.trace_add('write', self._on_pattern_mode_changed)
        tk.Radiobutton(pat_row, text="Keep Current", variable=self._pattern_mode_var,
                       value='keep', font=('Segoe UI', 9),
                       fg=COLORS['text'], bg=COLORS['bg_dark'],
                       selectcolor=COLORS['bg_medium'],
                       activebackground=COLORS['bg_dark'],
                       activeforeground=COLORS['text']).pack(side='left', padx=(4, 8))
        tk.Radiobutton(pat_row, text="Generate New AI Patterns", variable=self._pattern_mode_var,
                       value='generate', font=('Segoe UI', 9),
                       fg=COLORS['text'], bg=COLORS['bg_dark'],
                       selectcolor=COLORS['bg_medium'],
                       activebackground=COLORS['bg_dark'],
                       activeforeground=COLORS['text']).pack(side='left', padx=(0, 12))

        # Pattern temperature (only visible in generate mode)
        self._pat_temp_frame = tk.Frame(pat_row, bg=COLORS['bg_dark'])
        self._pat_temp_frame.pack(side='left')
        tk.Label(self._pat_temp_frame, text="Pattern Temp:", font=('Segoe UI', 9),
                 fg=COLORS['text'], bg=COLORS['bg_dark']).pack(side='left')
        saved_pat_temp = self.preferences_manager.get('drum_generator_pattern_temperature', 0.7)
        self.pattern_temp_var = tk.DoubleVar(value=saved_pat_temp)
        tk.Spinbox(self._pat_temp_frame, from_=0.1, to=3.0, increment=0.1,
                   textvariable=self.pattern_temp_var, width=5,
                   font=('Segoe UI', 9),
                   bg=COLORS['bg_medium'], fg=COLORS['text'],
                   buttonbackground=COLORS['bg_light'],
                   insertbackground=COLORS['text']).pack(side='left', padx=(2, 8))

        # Pattern bank status
        self._pat_bank_label = tk.Label(pat_row, text="", font=('Segoe UI', 8),
                                        fg=COLORS['text_dim'], bg=COLORS['bg_dark'])
        self._pat_bank_label.pack(side='left')

        # Initially hide pattern temp controls
        self._pat_temp_frame.pack_forget()

    # ── 8-lane slot area ─────────────────────────────────────────────

    def _build_slot_area(self):
        container = tk.Frame(self.dialog, bg=COLORS['bg_dark'])
        container.pack(fill='both', expand=True, padx=10, pady=5)

        self.slot_frames = []
        self.slot_widgets = []

        for i in range(8):
            label, allowed = SLOT_MAP[i]
            sf = self._build_slot_lane(container, i, label, allowed)
            sf.pack(side='left', fill='both', expand=True, padx=2)

    def _build_slot_lane(self, parent, slot_idx, label, allowed_types):
        frame = tk.Frame(parent, bg=COLORS['slot_bg'], bd=1, relief='groove')

        widgets = {}

        # Slot header: number + label
        header = tk.Frame(frame, bg=COLORS['slot_bg'])
        header.pack(fill='x', padx=4, pady=(6, 2))
        tk.Label(header, text=f"{slot_idx + 1}", font=('Segoe UI', 11, 'bold'),
                 fg=COLORS['orange'], bg=COLORS['slot_bg']).pack(side='left')
        tk.Label(header, text=label, font=('Segoe UI', 8),
                 fg=COLORS['text_dim'], bg=COLORS['slot_bg']).pack(side='left', padx=4)

        # Type override dropdown
        type_frame = tk.Frame(frame, bg=COLORS['slot_bg'])
        type_frame.pack(fill='x', padx=4, pady=2)
        type_var = tk.StringVar(value=allowed_types[0])
        widgets['type_var'] = type_var
        type_menu = ttk.Combobox(type_frame, textvariable=type_var,
                                 values=allowed_types, state='readonly',
                                 width=8, font=('Segoe UI', 8))
        type_menu.pack(fill='x')

        # Generate button for this slot
        tk.Button(frame, text="Generate", command=lambda i=slot_idx: self._on_generate_slot(i),
                  bg=COLORS['generate_btn'], fg=COLORS['text'],
                  font=('Segoe UI', 8), relief='flat').pack(fill='x', padx=4, pady=4)

        # Candidate navigator: < idx/total >
        nav_frame = tk.Frame(frame, bg=COLORS['slot_bg'])
        nav_frame.pack(fill='x', padx=4)
        prev_btn = tk.Button(nav_frame, text="<", width=2,
                             command=lambda i=slot_idx: self._on_prev_candidate(i),
                             bg=COLORS['bg_light'], fg=COLORS['text'],
                             font=('Segoe UI', 8), relief='flat')
        prev_btn.pack(side='left')
        idx_label = tk.Label(nav_frame, text="- / -", font=('Segoe UI', 8),
                             fg=COLORS['text'], bg=COLORS['slot_bg'])
        idx_label.pack(side='left', expand=True)
        next_btn = tk.Button(nav_frame, text=">", width=2,
                             command=lambda i=slot_idx: self._on_next_candidate(i),
                             bg=COLORS['bg_light'], fg=COLORS['text'],
                             font=('Segoe UI', 8), relief='flat')
        next_btn.pack(side='right')
        widgets['idx_label'] = idx_label

        # Candidate name
        name_label = tk.Label(frame, text="", font=('Segoe UI', 8),
                              fg=COLORS['accent_light'], bg=COLORS['slot_bg'],
                              wraplength=100)
        name_label.pack(fill='x', padx=4, pady=2)
        widgets['name_label'] = name_label

        # One-shot preview button
        tk.Button(frame, text="Preview", command=lambda i=slot_idx: self._on_preview_slot(i),
                  bg=COLORS['bg_light'], fg=COLORS['text'],
                  font=('Segoe UI', 8), relief='flat').pack(fill='x', padx=4, pady=2)

        # Apply checkbox
        apply_var = tk.BooleanVar(value=False)
        widgets['apply_var'] = apply_var
        apply_cb = tk.Checkbutton(frame, text="Apply", variable=apply_var,
                                  font=('Segoe UI', 8),
                                  fg=COLORS['text'], bg=COLORS['slot_bg'],
                                  selectcolor=COLORS['bg_medium'],
                                  activebackground=COLORS['slot_bg'],
                                  activeforeground=COLORS['text'])
        apply_cb.pack(fill='x', padx=4, pady=(2, 6))

        self.slot_frames.append(frame)
        self.slot_widgets.append(widgets)
        return frame

    # ── Bottom bar ───────────────────────────────────────────────────

    def _build_bottom_bar(self):
        bottom = tk.Frame(self.dialog, bg=COLORS['bg_dark'])
        bottom.pack(fill='x', padx=10, pady=(5, 10))

        # Preview controls (left)
        preview_frame = tk.Frame(bottom, bg=COLORS['bg_dark'])
        preview_frame.pack(side='left')

        self.preview_loop_btn = tk.Button(
            preview_frame, text="Loop Preview",
            command=self._on_toggle_loop_preview,
            bg=COLORS['bg_light'], fg=COLORS['text'],
            font=('Segoe UI', 9), relief='flat', padx=6)
        self.preview_loop_btn.pack(side='left', padx=(0, 4))

        self.preview_bank_btn = tk.Button(
            preview_frame, text="Preview Bank",
            command=self._on_toggle_bank_preview,
            bg=COLORS['bg_light'], fg=COLORS['text'],
            font=('Segoe UI', 9), relief='flat', padx=6)
        self.preview_bank_btn.pack(side='left', padx=(0, 8))

        # Apply controls (right)
        apply_frame = tk.Frame(bottom, bg=COLORS['bg_dark'])
        apply_frame.pack(side='right')

        self.replace_patterns_btn = tk.Button(
            apply_frame, text="Replace All Patterns From AI",
            command=self._on_replace_patterns,
            bg='#664422', fg=COLORS['text'],
            font=('Segoe UI', 9, 'bold'), relief='flat',
            padx=8, state='disabled')
        self.replace_patterns_btn.pack(side='right', padx=(8, 0))

        tk.Button(apply_frame, text="Apply Selected",
                  command=self._on_apply_selected,
                  bg=COLORS['apply_btn'], fg=COLORS['text'],
                  font=('Segoe UI', 9, 'bold'), relief='flat',
                  padx=8).pack(side='right', padx=(8, 0))

        tk.Button(apply_frame, text="Close",
                  command=self._on_close,
                  bg=COLORS['bg_light'], fg=COLORS['text'],
                  font=('Segoe UI', 9), relief='flat',
                  padx=8).pack(side='right')

    # ================================================================
    # Model loading
    # ================================================================

    def _update_model_status(self):
        if not is_torch_available():
            self.model_status_label.config(text="PyTorch not installed",
                                           fg='#ff8888')
            self.pattern_model_status_label.config(text="PyTorch not installed",
                                                   fg='#ff8888')
            self.install_btn.config(state='normal')
            return

        self.install_btn.config(state='disabled')

        if self.generator.is_loaded:
            self.model_status_label.config(
                text=f"loaded ({self.generator.sampling_summary})",
                fg=COLORS['led_on']
            )
        else:
            saved_path = self.preferences_manager.get('drum_generator_model_path', None)
            if saved_path and self._try_load_model(saved_path):
                pass
            else:
                self.model_status_label.config(text="not loaded",
                                               fg=COLORS['text_dim'])

        if self.pattern_gen.is_loaded:
            self.pattern_model_status_label.config(text="loaded",
                                                   fg=COLORS['led_on'])
        else:
            saved_path = self.preferences_manager.get(
                'drum_generator_pattern_model_path', None)
            if saved_path and self._try_load_pattern_model(saved_path):
                pass
            else:
                self.pattern_model_status_label.config(text="not loaded",
                                                       fg=COLORS['text_dim'])

        self._update_pattern_controls()

    def _try_load_model(self, path: str) -> bool:
        try:
            self.generator.load_model(path)
            self.preferences_manager.set('drum_generator_model_path', path)
            self.model_status_label.config(
                text=f"loaded ({self.generator.sampling_summary})",
                fg=COLORS['led_on']
            )
            return True
        except Exception as e:
            self.model_status_label.config(text=f"error: {e}",
                                           fg='#ff8888')
            return False

    def _try_load_pattern_model(self, path: str) -> bool:
        try:
            self.pattern_gen.load_model(path)
            self.preferences_manager.set('drum_generator_pattern_model_path', path)
            self.pattern_model_status_label.config(text="loaded",
                                                   fg=COLORS['led_on'])
            self._update_pattern_controls()
            return True
        except Exception as e:
            self.pattern_model_status_label.config(text=f"error: {e}",
                                                   fg='#ff8888')
            return False

    def _on_load_model(self):
        if not is_torch_available():
            messagebox.showerror("PyTorch Required",
                                 "PyTorch is not installed.\n"
                                 "Click 'Install ML Support' to install it.",
                                 parent=self.dialog)
            return

        initial_dir = None
        saved_path = self.preferences_manager.get('drum_generator_model_path', None)
        if saved_path:
            import os
            initial_dir = os.path.dirname(saved_path)

        path = filedialog.askopenfilename(
            parent=self.dialog,
            title="Select CVAE Checkpoint",
            filetypes=[("PyTorch Checkpoint", "*.pt"), ("All Files", "*.*")],
            initialdir=initial_dir,
        )
        if path:
            self._try_load_model(path)

    def _on_load_pattern_model(self):
        if not is_torch_available():
            messagebox.showerror("PyTorch Required",
                                 "PyTorch is not installed.\n"
                                 "Click 'Install ML Support' to install it.",
                                 parent=self.dialog)
            return

        initial_dir = None
        saved_path = self.preferences_manager.get(
            'drum_generator_pattern_model_path', None)
        if saved_path:
            import os
            initial_dir = os.path.dirname(saved_path)

        path = filedialog.askopenfilename(
            parent=self.dialog,
            title="Select Pattern CVAE Checkpoint",
            filetypes=[("PyTorch Checkpoint", "*.pt"), ("All Files", "*.*")],
            initialdir=initial_dir,
        )
        if path:
            self._try_load_pattern_model(path)

    def _on_pattern_mode_changed(self, *_args):
        self._pattern_mode = self._pattern_mode_var.get()
        self._cached_pattern_bank = None
        self._update_pattern_controls()

    def _update_pattern_controls(self):
        """Show/hide pattern controls based on mode and model availability."""
        if self._pattern_mode == 'generate':
            self._pat_temp_frame.pack(side='left')
            if self._cached_pattern_bank:
                self._pat_bank_label.config(
                    text="Bank ready (12 patterns)",
                    fg=COLORS['led_on'])
            else:
                self._pat_bank_label.config(text="(generate patches first)",
                                            fg=COLORS['text_dim'])
        else:
            self._pat_temp_frame.pack_forget()
            self._pat_bank_label.config(text="")

        # Enable/disable replace button
        can_replace = (self._pattern_mode == 'generate'
                       and self._cached_pattern_bank is not None)
        self.replace_patterns_btn.config(
            state='normal' if can_replace else 'disabled')

    def _invalidate_pattern_bank(self):
        """Invalidate the cached pattern bank when parameters change."""
        self._cached_pattern_bank = None
        self._update_pattern_controls()

    def _on_install_ml(self):
        if is_torch_available():
            messagebox.showinfo("Already Installed",
                                "PyTorch is already available.",
                                parent=self.dialog)
            return

        if not messagebox.askyesno(
            "Install ML Dependencies",
            "This will run:\n  pip install -r requirements-ml.txt\n\n"
            "in the current Python environment. Proceed?",
            parent=self.dialog
        ):
            return

        self.install_btn.config(state='disabled', text="Installing...")
        self.dialog.update_idletasks()

        def _do_install():
            lines = []

            def on_output(line):
                lines.append(line)

            success = install_ml_dependencies(on_output=on_output)

            def _finish():
                if success:
                    self.install_btn.config(text="Installed", state='disabled')
                    self._update_model_status()
                    messagebox.showinfo("Success",
                                        "ML dependencies installed successfully.\n"
                                        "You can now load a model.",
                                        parent=self.dialog)
                else:
                    self.install_btn.config(text="Install ML Support", state='normal')
                    messagebox.showerror("Install Failed",
                                         "Installation failed.\n\n" +
                                         "\n".join(lines[-10:]),
                                         parent=self.dialog)

            self.dialog.after(0, _finish)

        threading.Thread(target=_do_install, daemon=True).start()

    # ================================================================
    # Generation
    # ================================================================

    def _get_seed(self):
        """Return int seed or None if blank."""
        s = self.seed_var.get().strip()
        if not s:
            return None
        try:
            return int(s)
        except ValueError:
            return None

    def _on_reseed(self):
        import random
        self.seed_var.set(str(random.randint(0, 2**31 - 1)))

    def _on_generate_slot(self, slot_idx):
        if not self.generator.is_loaded:
            messagebox.showwarning("No Model", "Load a patch model first.",
                                   parent=self.dialog)
            return

        w = self.slot_widgets[slot_idx]
        drum_type = w['type_var'].get()
        n = self.candidates_var.get()
        temp = self.temp_var.get()
        seed = self._get_seed()

        try:
            candidates = self.generator.generate(drum_type, n=n,
                                                  temperature=temp, seed=seed)
        except Exception as e:
            messagebox.showerror("Generation Error", str(e), parent=self.dialog)
            return

        self.slot_state[slot_idx]['candidates'] = candidates
        self.slot_state[slot_idx]['selected_idx'] = 0
        w['apply_var'].set(True)
        self._update_slot_display(slot_idx)
        self._invalidate_pattern_bank()

    def _on_generate_all(self):
        if not self.generator.is_loaded:
            messagebox.showwarning("No Model", "Load a patch model first.",
                                   parent=self.dialog)
            return

        n = self.candidates_var.get()
        temp = self.temp_var.get()
        seed = self._get_seed()

        for i in range(8):
            w = self.slot_widgets[i]
            drum_type = w['type_var'].get()
            try:
                candidates = self.generator.generate(drum_type, n=n,
                                                      temperature=temp, seed=seed)
            except Exception as e:
                print(f"Slot {i} generation error: {e}", flush=True)
                candidates = []

            self.slot_state[i]['candidates'] = candidates
            self.slot_state[i]['selected_idx'] = 0
            w['apply_var'].set(True)
            self._update_slot_display(i)

        # Auto-generate pattern bank when in generate mode
        self._maybe_generate_pattern_bank()

    def _on_prev_candidate(self, slot_idx):
        st = self.slot_state[slot_idx]
        if st['candidates']:
            st['selected_idx'] = (st['selected_idx'] - 1) % len(st['candidates'])
            self._update_slot_display(slot_idx)
            self._invalidate_pattern_bank()

    def _on_next_candidate(self, slot_idx):
        st = self.slot_state[slot_idx]
        if st['candidates']:
            st['selected_idx'] = (st['selected_idx'] + 1) % len(st['candidates'])
            self._update_slot_display(slot_idx)
            self._invalidate_pattern_bank()

    def _update_slot_display(self, slot_idx):
        w = self.slot_widgets[slot_idx]
        st = self.slot_state[slot_idx]
        if st['candidates']:
            total = len(st['candidates'])
            idx = st['selected_idx']
            w['idx_label'].config(text=f"{idx + 1} / {total}")
            patch = st['candidates'][idx]
            w['name_label'].config(text=patch.get('Name', ''))
        else:
            w['idx_label'].config(text="- / -")
            w['name_label'].config(text="")

    # ================================================================
    # Preview-state builder
    # ================================================================

    def _get_tentative_raw_patches(self) -> list[dict]:
        """Build the tentative kit: live channels with generated patches overlaid."""
        patches = []
        for i in range(8):
            st = self.slot_state[i]
            if st['candidates']:
                patches.append(st['candidates'][st['selected_idx']])
            else:
                patches.append(channel_to_raw_patch(self.synth.channels[i]))
        return patches

    def _build_preview_synth(self):
        """Create a preview synth with the tentative kit overlaid."""
        from pythonic.synthesizer import PythonicSynthesizer
        preview_synth = PythonicSynthesizer(
            SAMPLE_RATE,
            parallel_channel_processing=True,
        )
        kit_data = self.synth.get_preset_data()
        preview_synth.load_preset_data(kit_data)

        for i in range(8):
            st = self.slot_state[i]
            if st['candidates']:
                patch = st['candidates'][st['selected_idx']]
                channel_data = convert_drum_patch_data(patch)
                apply_drum_patch_to_channel(preview_synth.channels[i], channel_data)
        return preview_synth

    def _build_trigger_table_from_pattern(self, pattern) -> list[list[int]]:
        """Build a 16-step trigger table from a Pattern object."""
        triggers_table = [[] for _ in range(16)]
        for ch_idx in range(min(8, len(pattern.channels))):
            for step_idx, step in enumerate(pattern.channels[ch_idx].steps):
                if step.trigger and step_idx < 16:
                    triggers_table[step_idx].append(ch_idx)
        return triggers_table

    def _build_trigger_table_from_dict(self, pat_data: dict) -> list[list[int]]:
        """Build a 16-step trigger table from a generated pattern dict."""
        triggers_table = [[] for _ in range(16)]
        for ch_idx in range(8):
            ch_key = str(ch_idx + 1)
            ch_data = pat_data.get(ch_key, {})
            if isinstance(ch_data, dict):
                triggers_str = ch_data.get("Triggers", "")
                for step in range(min(16, len(triggers_str))):
                    if triggers_str[step] == "#":
                        triggers_table[step].append(ch_idx)
        return triggers_table

    def _get_preview_trigger_table(self) -> list[list[int]]:
        """Get the trigger table for the current preview scope."""
        if self._pattern_mode == 'generate' and self._cached_pattern_bank:
            # Use the generated version of the current selected/playing pattern
            pm = self.pattern_manager
            pat_name = PATTERN_NAMES[pm.playing_pattern_index
                                     if pm.playing_pattern_index >= 0
                                     else pm.selected_pattern_index]
            pat_data = self._cached_pattern_bank.get(pat_name)
            if pat_data:
                return self._build_trigger_table_from_dict(pat_data)

        # Keep Current mode: use the real pattern from PatternManager
        pm = self.pattern_manager
        try:
            pattern = pm.get_playing_pattern()
        except Exception:
            pattern = pm.get_selected_pattern()
        return self._build_trigger_table_from_pattern(pattern)

    # ================================================================
    # Pattern bank generation
    # ================================================================

    def _maybe_generate_pattern_bank(self):
        """Generate a pattern bank if in generate mode and pattern model is loaded."""
        if self._pattern_mode != 'generate' or not self.pattern_gen.is_loaded:
            self._invalidate_pattern_bank()
            return

        raw_patches = self._get_tentative_raw_patches()
        pm = self.pattern_manager
        seed = self._get_seed()
        pat_temp = self.pattern_temp_var.get()

        try:
            self._cached_pattern_bank = self.pattern_gen.generate_bank(
                raw_patches,
                tempo=pm.bpm,
                swing=0.0,
                fill_rate=pm.fill_rate,
                step_rate=pm.step_rate,
                temperature=pat_temp,
                seed=seed,
            )
            self._pat_bank_label.config(text="Bank ready (12 patterns)",
                                        fg=COLORS['led_on'])
        except Exception as e:
            print(f"Pattern bank generation error: {e}", flush=True)
            self._cached_pattern_bank = None
            self._pat_bank_label.config(text=f"error: {e}",
                                        fg='#ff8888')
        self._update_pattern_controls()

    # ================================================================
    # Preview: one-shot
    # ================================================================

    def _on_preview_slot(self, slot_idx):
        """Trigger a one-shot preview for the selected candidate on this slot."""
        st = self.slot_state[slot_idx]
        if not st['candidates'] or not AUDIO_AVAILABLE:
            return

        patch = st['candidates'][st['selected_idx']]
        channel_data = convert_drum_patch_data(patch)

        from pythonic.synthesizer import PythonicSynthesizer
        preview_synth = PythonicSynthesizer(
            SAMPLE_RATE,
            parallel_channel_processing=True,
        )

        apply_drum_patch_to_channel(preview_synth.channels[slot_idx], channel_data)
        preview_synth.trigger_drum(slot_idx, velocity=100)

        duration_samples = int(SAMPLE_RATE * 2.0)
        audio = preview_synth.process_audio(duration_samples)

        try:
            sd.play(audio, samplerate=SAMPLE_RATE)
        except Exception as e:
            print(f"One-shot preview error: {e}", flush=True)

    # ================================================================
    # Preview: pattern loop (current pattern)
    # ================================================================

    def _on_toggle_loop_preview(self):
        if self.preview_playing:
            self._stop_loop_preview()
        else:
            self._start_loop_preview()

    def _start_loop_preview(self):
        """Loop the current selected/playing pattern with the tentative kit."""
        if not AUDIO_AVAILABLE:
            return

        preview_synth = self._build_preview_synth()
        triggers_table = self._get_preview_trigger_table()

        self._start_preview_stream(preview_synth, [triggers_table])
        self.preview_loop_btn.config(text="Stop Loop", bg='#884444')

    # ================================================================
    # Preview: bank sequence
    # ================================================================

    def _on_toggle_bank_preview(self):
        if self.preview_playing:
            self._stop_loop_preview()
        else:
            self._start_bank_preview()

    def _start_bank_preview(self):
        """Sequence through generated pattern bank (or live patterns if keeping)."""
        if not AUDIO_AVAILABLE:
            return

        preview_synth = self._build_preview_synth()

        # Build trigger tables for all 12 patterns
        if self._pattern_mode == 'generate' and self._cached_pattern_bank:
            tables = []
            for name in PATTERN_NAMES:
                pat_data = self._cached_pattern_bank.get(name)
                if pat_data:
                    tables.append(self._build_trigger_table_from_dict(pat_data))
                else:
                    tables.append([[] for _ in range(16)])
        else:
            tables = []
            for pat in self.pattern_manager.patterns:
                tables.append(self._build_trigger_table_from_pattern(pat))

        self._start_preview_stream(preview_synth, tables)
        self.preview_bank_btn.config(text="Stop Bank", bg='#884444')

    # ================================================================
    # Shared preview stream
    # ================================================================

    def _start_preview_stream(self, preview_synth, trigger_tables: list):
        """Start buffered preview playback through the main audio stream."""
        self._stop_loop_preview()
        renderer = BufferedPatternPreviewSource(
            preview_synth,
            trigger_tables,
            bpm=self.pattern_manager.bpm,
        )

        try:
            renderer.start()
        except Exception as e:
            print(f"Preview error: {e}", flush=True)
            renderer.stop()
            return

        self.synth.set_preview_source(renderer)
        self._preview_renderer = renderer
        self.preview_playing = True
        self.preview_stop_flag.clear()

    def _stop_loop_preview(self):
        self.preview_playing = False
        self.preview_stop_flag.set()
        self.preview_loop_btn.config(text="Loop Preview", bg=COLORS['bg_light'])
        self.preview_bank_btn.config(text="Preview Bank", bg=COLORS['bg_light'])

        if self._preview_renderer is not None:
            self.synth.clear_preview_source(self._preview_renderer)
            self._preview_renderer = None

    # ================================================================
    # Apply
    # ================================================================

    def _on_apply_selected(self):
        """Apply checked patch slots to the live synth (patterns unchanged)."""
        applied = []
        for i in range(8):
            st = self.slot_state[i]
            w = self.slot_widgets[i]
            if w['apply_var'].get() and st['candidates']:
                patch = st['candidates'][st['selected_idx']]
                channel_data = convert_drum_patch_data(patch)
                apply_drum_patch_to_channel(self.synth.channels[i], channel_data)
                applied.append(i)

        if applied and self.on_apply_callback:
            self.on_apply_callback('patches')

        if applied:
            names = ", ".join(f"{i+1}" for i in applied)
            self.model_status_label.config(
                text=f"Applied to slot{'s' if len(applied) > 1 else ''} {names}",
                fg=COLORS['led_on'])

    def _on_replace_patterns(self):
        """Apply checked patches + replace all 12 patterns from the cached AI bank."""
        if not self._cached_pattern_bank:
            messagebox.showwarning("No Pattern Bank",
                                   "Generate patterns first by setting the pattern\n"
                                   "mode to 'Generate New AI Patterns' and\n"
                                   "generating patches.",
                                   parent=self.dialog)
            return

        # Apply patches first
        applied = []
        for i in range(8):
            st = self.slot_state[i]
            w = self.slot_widgets[i]
            if w['apply_var'].get() and st['candidates']:
                patch = st['candidates'][st['selected_idx']]
                channel_data = convert_drum_patch_data(patch)
                apply_drum_patch_to_channel(self.synth.channels[i], channel_data)
                applied.append(i)

        # Replace all patterns
        self.pattern_manager.apply_pattern_bank(self._cached_pattern_bank)

        if self.on_apply_callback:
            self.on_apply_callback('patches_and_patterns')

        self.model_status_label.config(
            text=f"Applied patches + 12 patterns",
            fg=COLORS['led_on'])

    # ================================================================
    # Cleanup
    # ================================================================

    def _on_close(self):
        self._stop_loop_preview()
        # Save temperature preferences
        self.preferences_manager.set('drum_generator_patch_temperature',
                                     self.temp_var.get())
        self.preferences_manager.set('drum_generator_pattern_temperature',
                                     self.pattern_temp_var.get())
        try:
            sd.stop()
        except Exception:
            pass
        self.dialog.destroy()
