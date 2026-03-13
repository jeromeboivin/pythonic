"""
AI Drum Generator Dialog

TR-8-inspired 8-lane interface for generating drum patches using a CVAE model.
Supports per-slot generation, one-shot preview, pattern-loop preview, and
selective apply back to the live kit.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
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
from pythonic.preset_manager import convert_drum_patch_data, apply_drum_patch_to_channel


SAMPLE_RATE = 44100

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

        # Per-slot state: list of 8 dicts
        # Each: {candidates: [raw_patch_dict, ...], selected_idx: int, type_override: str|None}
        self.slot_state = [
            {"candidates": [], "selected_idx": 0, "type_override": None}
            for _ in range(8)
        ]

        # Preview state
        self.preview_playing = False
        self._preview_stream = None
        self._preview_synth = None
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

        # Model status
        self.model_status_label = tk.Label(right, text="No model loaded",
                                           font=('Segoe UI', 8),
                                           fg=COLORS['text_dim'], bg=COLORS['bg_dark'])
        self.model_status_label.pack(side='left', padx=(0, 8))

        tk.Button(right, text="Load Model...", command=self._on_load_model,
                  bg=COLORS['bg_light'], fg=COLORS['text'],
                  font=('Segoe UI', 8), relief='flat', padx=6).pack(side='left', padx=2)

        self.install_btn = tk.Button(right, text="Install ML Support",
                                     command=self._on_install_ml,
                                     bg=COLORS['bg_light'], fg=COLORS['text'],
                                     font=('Segoe UI', 8), relief='flat', padx=6)
        self.install_btn.pack(side='left', padx=2)

        # ── Controls row ──
        ctrl = tk.Frame(self.dialog, bg=COLORS['bg_dark'])
        ctrl.pack(fill='x', padx=10, pady=(0, 5))

        # Temperature
        tk.Label(ctrl, text="Temperature:", font=('Segoe UI', 9),
                 fg=COLORS['text'], bg=COLORS['bg_dark']).pack(side='left')
        self.temp_var = tk.DoubleVar(value=1.0)
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
        self.preview_loop_btn.pack(side='left', padx=(0, 8))

        # Apply controls (right)
        apply_frame = tk.Frame(bottom, bg=COLORS['bg_dark'])
        apply_frame.pack(side='right')

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
            self.install_btn.config(state='normal')
            return

        self.install_btn.config(state='disabled')

        if self.generator.is_loaded:
            self.model_status_label.config(
                text=f"Model loaded", fg=COLORS['led_on'])
        else:
            # Try auto-loading from preferences
            saved_path = self.preferences_manager.get('drum_generator_model_path', None)
            if saved_path and self._try_load_model(saved_path):
                return
            self.model_status_label.config(text="No model loaded",
                                           fg=COLORS['text_dim'])

    def _try_load_model(self, path: str) -> bool:
        try:
            self.generator.load_model(path)
            self.preferences_manager.set('drum_generator_model_path', path)
            self.model_status_label.config(text="Model loaded",
                                           fg=COLORS['led_on'])
            return True
        except Exception as e:
            self.model_status_label.config(text=f"Load failed: {e}",
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
            messagebox.showwarning("No Model", "Load a model first.",
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

    def _on_generate_all(self):
        if not self.generator.is_loaded:
            messagebox.showwarning("No Model", "Load a model first.",
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

    def _on_prev_candidate(self, slot_idx):
        st = self.slot_state[slot_idx]
        if st['candidates']:
            st['selected_idx'] = (st['selected_idx'] - 1) % len(st['candidates'])
            self._update_slot_display(slot_idx)

    def _on_next_candidate(self, slot_idx):
        st = self.slot_state[slot_idx]
        if st['candidates']:
            st['selected_idx'] = (st['selected_idx'] + 1) % len(st['candidates'])
            self._update_slot_display(slot_idx)

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
    # Preview: one-shot
    # ================================================================

    def _on_preview_slot(self, slot_idx):
        """Trigger a one-shot preview for the selected candidate on this slot."""
        st = self.slot_state[slot_idx]
        if not st['candidates'] or not AUDIO_AVAILABLE:
            return

        patch = st['candidates'][st['selected_idx']]
        channel_data = convert_drum_patch_data(patch)

        # Create a temporary single-use synth
        from pythonic.synthesizer import PythonicSynthesizer
        preview_synth = PythonicSynthesizer(SAMPLE_RATE)

        # Apply only to the target channel
        apply_drum_patch_to_channel(preview_synth.channels[slot_idx], channel_data)
        preview_synth.trigger_drum(slot_idx, velocity=100)

        # Render and play
        duration_samples = int(SAMPLE_RATE * 2.0)  # 2 seconds max
        audio = preview_synth.process_audio(duration_samples)

        try:
            sd.play(audio, samplerate=SAMPLE_RATE)
        except Exception as e:
            print(f"One-shot preview error: {e}", flush=True)

    # ================================================================
    # Preview: pattern loop
    # ================================================================

    def _on_toggle_loop_preview(self):
        if self.preview_playing:
            self._stop_loop_preview()
        else:
            self._start_loop_preview()

    def _start_loop_preview(self):
        if not AUDIO_AVAILABLE:
            return

        # Build a preview synth cloned from the current live kit
        from pythonic.synthesizer import PythonicSynthesizer
        preview_synth = PythonicSynthesizer(SAMPLE_RATE)
        kit_data = self.synth.get_preset_data()
        preview_synth.load_preset_data(kit_data)

        # Overlay generated candidates for slots that have them
        for i in range(8):
            st = self.slot_state[i]
            if st['candidates']:
                patch = st['candidates'][st['selected_idx']]
                channel_data = convert_drum_patch_data(patch)
                apply_drum_patch_to_channel(preview_synth.channels[i], channel_data)

        # Build trigger table from the current pattern
        triggers_table = self._build_trigger_table()

        bpm = self.pattern_manager.bpm
        step_duration_s = 60.0 / bpm / 4.0
        step_samples = int(step_duration_s * SAMPLE_RATE)

        cb_state = {
            'synth': preview_synth,
            'step': 0,
            'frame_in_step': 0,
            'step_samples': step_samples,
            'triggers': triggers_table,
        }

        # Trigger step 0 immediately
        for d in triggers_table[0]:
            preview_synth.trigger_drum(d, velocity=100)

        def _preview_callback(outdata, frames, time_info, status):
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

                if s_frame >= s_step_samples:
                    s_frame = 0
                    s_step = (s_step + 1) % 16
                    for d in t_table[s_step]:
                        synth.trigger_drum(d, velocity=100)

            cb_state['step'] = s_step
            cb_state['frame_in_step'] = s_frame

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
            print(f"Loop preview error: {e}", flush=True)
            return

        self._preview_synth = preview_synth
        self.preview_playing = True
        self.preview_stop_flag.clear()
        self.preview_loop_btn.config(text="Stop Loop", bg='#884444')

    def _stop_loop_preview(self):
        self.preview_playing = False
        self.preview_stop_flag.set()
        self.preview_loop_btn.config(text="Loop Preview", bg=COLORS['bg_light'])

        if self._preview_stream is not None:
            try:
                self._preview_stream.stop()
                self._preview_stream.close()
            except Exception:
                pass
            self._preview_stream = None
        self._preview_synth = None

    def _build_trigger_table(self):
        """Build a 16-step trigger table from the current pattern."""
        triggers_table = [[] for _ in range(16)]
        try:
            pattern = self.pattern_manager.get_current_pattern()
            for ch_idx in range(8):
                ch_triggers = pattern.get('triggers', {}).get(ch_idx, [])
                for step, active in enumerate(ch_triggers):
                    if active and step < 16:
                        triggers_table[step].append(ch_idx)
        except Exception:
            # Fallback: four-on-the-floor kick
            for step in range(0, 16, 4):
                triggers_table[step].append(0)
        return triggers_table

    # ================================================================
    # Apply
    # ================================================================

    def _on_apply_selected(self):
        """Apply checked slots to the live synth."""
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
            self.on_apply_callback()

        if applied:
            names = ", ".join(f"{i+1}" for i in applied)
            self.model_status_label.config(
                text=f"Applied to slot{'s' if len(applied) > 1 else ''} {names}",
                fg=COLORS['led_on'])

    # ================================================================
    # Cleanup
    # ================================================================

    def _on_close(self):
        self._stop_loop_preview()
        try:
            sd.stop()
        except Exception:
            pass
        self.dialog.destroy()
