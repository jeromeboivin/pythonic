"""
AI Drum Generator Dialog

TR-8-inspired 8-lane interface for the app core's AI generators (``ai.*``):
per-slot generation, candidates tried on the live channels, one-shot,
pattern-loop and bank previews, and selective keep back into the preset. The
models run in the core's AI worker process; this dialog only reads the
``ai.*`` addresses on its own timer and starts ``ai.*`` verbs.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from pythonic.drum_generator import SLOT_MAP

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

    def __init__(self, parent, core, when_done=None, on_close=None):
        """``when_done(action_id, callback(event))`` runs a callback once the
        core reports an action (the main window's poll tick); ``on_close()``
        runs after the dialog has been closed."""
        self.parent = parent
        self.core = core
        self.when_done = when_done or (lambda action_id, callback: None)
        self.on_close = on_close

        # Pattern handling mode: 'keep' or 'generate'
        self._pattern_mode = 'keep'
        self._refresh_job = None
        self._closed = False

        self._build_dialog()
        self._load_models()
        self._refresh()

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
        self.dialog.grab_set()
        self.dialog.focus_set()

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
        saved_patch_temp = self.core.get('pref.ai.patch_temperature')
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
        saved_pat_temp = self.core.get('pref.ai.pattern_temperature')
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
    # Core helpers
    # ================================================================

    def _act(self, verb, message=None, on_done=None, **args):
        """Start a core verb; show its error, or call on_done(result)."""
        def done(event):
            if event['status'] != 'done':
                if message:
                    messagebox.showerror(message, str(event.get('error')),
                                         parent=self._parent_window())
            elif on_done is not None:
                on_done(event.get('result'))
        action_id = self.core.act(verb, **args)
        self.when_done(action_id, done)
        return action_id

    def _parent_window(self):
        return self.parent if self._closed else self.dialog

    def _lane(self, slot_idx, field):
        return self.core.get(f'ai.ch{slot_idx + 1}.{field}')

    # ================================================================
    # Model loading
    # ================================================================

    def _load_models(self):
        """Load the saved (or bundled) models in the AI worker at once, so
        the status shows them; they load in the background."""
        if not self.core.get('ai.available'):
            return
        for kind, models in self.core.get('ai.models').items():
            if models['status'] == 'unloaded':
                self._act('ai.load_model', kind=kind)

    def _update_model_status(self):
        if self.core.get('ai.installing'):
            self.install_btn.config(state='disabled', text="Installing...")
        elif not self.core.get('ai.available'):
            self.install_btn.config(state='normal', text="Install ML Support")
        else:
            self.install_btn.config(state='disabled')
        if not self.core.get('ai.available'):
            for label in (self.model_status_label, self.pattern_model_status_label):
                label.config(text="PyTorch not installed", fg='#ff8888')
            return
        models = self.core.get('ai.models')
        for kind, label in (('patch', self.model_status_label),
                            ('pattern', self.pattern_model_status_label)):
            model = models[kind]
            status = model['status']
            if status == 'loaded':
                text = f"loaded ({model['sampling']})" if model['sampling'] else "loaded"
                label.config(text=text, fg=COLORS['led_on'])
            elif status == 'loading':
                label.config(text="loading...", fg=COLORS['orange'])
            elif status == 'error':
                label.config(text=f"error: {model['error']}", fg='#ff8888')
            else:
                label.config(text="not loaded", fg=COLORS['text_dim'])

    def _choose_model(self, kind, title):
        if not self.core.get('ai.available'):
            messagebox.showerror("PyTorch Required",
                                 "PyTorch is not installed.\n"
                                 "Click 'Install ML Support' to install it.",
                                 parent=self.dialog)
            return
        import os
        current = self.core.get('ai.models')[kind]['path']
        path = filedialog.askopenfilename(
            parent=self.dialog,
            title=title,
            filetypes=[("PyTorch Checkpoint", "*.pt"), ("All Files", "*.*")],
            initialdir=os.path.dirname(current) if current else None,
        )
        if path:
            # The core saves the path once the model has loaded
            self._act('ai.load_model', None, kind=kind, path=path)

    def _on_load_model(self):
        self._choose_model('patch', "Select CVAE Checkpoint")

    def _on_load_pattern_model(self):
        self._choose_model('pattern', "Select Pattern CVAE Checkpoint")

    def _on_pattern_mode_changed(self, *_args):
        self._pattern_mode = self._pattern_mode_var.get()
        self._invalidate_pattern_bank()

    def _update_pattern_controls(self):
        """Show/hide pattern controls based on mode and the core's bank."""
        bank = self.core.get('ai.bank')
        if self._pattern_mode == 'generate':
            self._pat_temp_frame.pack(side='left')
            if bank == 'ready':
                self._pat_bank_label.config(text="Bank ready (12 patterns)",
                                            fg=COLORS['led_on'])
            elif bank == 'generating':
                self._pat_bank_label.config(text="generating...", fg=COLORS['orange'])
            else:
                self._pat_bank_label.config(text="(generate patches first)",
                                            fg=COLORS['text_dim'])
        else:
            self._pat_temp_frame.pack_forget()
            self._pat_bank_label.config(text="")

        can_replace = self._pattern_mode == 'generate' and bank == 'ready'
        self.replace_patterns_btn.config(state='normal' if can_replace else 'disabled')

    def _invalidate_pattern_bank(self):
        """Drop the AI pattern bank when the drum patches or the mode change."""
        if self.core.get('ai.bank') != 'none':
            self._act('ai.clear_patterns')

    def _on_install_ml(self):
        if self.core.get('ai.available'):
            messagebox.showinfo("Already Installed",
                                "PyTorch is already available.",
                                parent=self.dialog)
            return

        if not messagebox.askyesno(
            "Install ML Dependencies",
            f"This will run:\n  {self.core.get('ai.install_command')}\n\n"
            "in the current Python environment. Proceed?",
            parent=self.dialog
        ):
            return

        def done(result):
            messagebox.showinfo("Success",
                                "ML dependencies installed successfully.\n"
                                "You can now load a model.",
                                parent=self._parent_window())
            if not self._closed:
                self._load_models()
        self._act('ai.install', "Install Failed", on_done=done)

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

    def _generation_args(self):
        return {'temperature': self.temp_var.get(), 'candidates': self.candidates_var.get(),
                'seed': self._get_seed()}

    def _on_generate_slot(self, slot_idx):
        w = self.slot_widgets[slot_idx]
        # The core tries candidate 1 on the live channel once it arrives
        self._act('ai.generate', "Generation Error", channel=slot_idx + 1,
                  type=w['type_var'].get(), **self._generation_args())
        w['apply_var'].set(True)
        self._invalidate_pattern_bank()

    def _on_generate_all(self):
        for i in range(8):
            w = self.slot_widgets[i]
            self.core.set(f'ai.ch{i + 1}.type', w['type_var'].get())
            w['apply_var'].set(True)
        self._act('ai.generate', "Generation Error",
                  on_done=lambda result: self._maybe_generate_pattern_bank(),
                  **self._generation_args())

    def _on_prev_candidate(self, slot_idx):
        if self._lane(slot_idx, 'candidates'):
            self._act('ai.try', None, channel=slot_idx + 1, step=-1)
            self._invalidate_pattern_bank()

    def _on_next_candidate(self, slot_idx):
        if self._lane(slot_idx, 'candidates'):
            self._act('ai.try', None, channel=slot_idx + 1, step=1)
            self._invalidate_pattern_bank()

    def _update_slot_display(self, slot_idx):
        w = self.slot_widgets[slot_idx]
        total = self._lane(slot_idx, 'candidates')
        if self._lane(slot_idx, 'generating'):
            w['idx_label'].config(text="generating...")
        elif total:
            w['idx_label'].config(text=f"{self._lane(slot_idx, 'candidate')} / {total}")
        else:
            w['idx_label'].config(text="- / -")
        error = self._lane(slot_idx, 'error')
        name = self._lane(slot_idx, 'name') if total else ''
        w['name_label'].config(text=f"error: {error}" if error and not total else name)

    # ================================================================
    # Pattern bank generation
    # ================================================================

    def _maybe_generate_pattern_bank(self):
        """Generate a pattern bank for the tried drum patches in generate mode."""
        if self._closed or self._pattern_mode != 'generate':
            return
        self._act('ai.generate_patterns', None, temperature=self.pattern_temp_var.get(),
                  seed=self._get_seed())

    # ================================================================
    # Preview
    # ================================================================

    def _on_preview_slot(self, slot_idx):
        """Try the slot's candidate on its live channel and hit it at
        velocity 127, as keys 1-8 do in the main window."""
        if not self._lane(slot_idx, 'candidates'):
            return
        self._act('ai.try', None, channel=slot_idx + 1,
                  on_done=lambda result: self.core.trigger(slot_idx, 127))

    def _on_toggle_loop_preview(self):
        self._toggle_preview('loop')

    def _on_toggle_bank_preview(self):
        self._toggle_preview('bank')

    def _toggle_preview(self, mode):
        """Loop the playing pattern (or chain all 12 from A) with the tried drum
        patches, on the main transport; stopping puts the preset's patterns back."""
        if self.core.get('ai.preview') != 'off':
            self._act('ai.pattern_try', "Preview", mode=None)
        else:
            self._act('ai.pattern_try', "Preview", mode=mode,
                      bank=self._pattern_mode == 'generate')

    def _update_preview_buttons(self):
        preview = self.core.get('ai.preview')
        self.preview_loop_btn.config(
            text="Stop Loop" if preview == 'loop' else "Loop Preview",
            bg='#884444' if preview == 'loop' else COLORS['bg_light'])
        self.preview_bank_btn.config(
            text="Stop Bank" if preview == 'bank' else "Preview Bank",
            bg='#884444' if preview == 'bank' else COLORS['bg_light'])

    # ================================================================
    # Keep
    # ================================================================

    def _checked_slots(self):
        return [i + 1 for i in range(8)
                if self.slot_widgets[i]['apply_var'].get() and self._lane(i, 'candidates')]

    def _on_apply_selected(self):
        """Keep the checked slots' candidates (one undo step; patterns unchanged)."""
        channels = self._checked_slots()
        if not channels:
            return

        def done(result):
            kept = result['kept']
            if kept and not self._closed:
                names = ", ".join(str(c) for c in kept)
                self.model_status_label.config(
                    text=f"Applied to slot{'s' if len(kept) > 1 else ''} {names}",
                    fg=COLORS['led_on'])
        self._act('ai.keep', "Apply Failed", on_done=done, channels=channels)

    def _on_replace_patterns(self):
        """Keep the checked slots and replace all 12 patterns from the AI bank."""
        if self.core.get('ai.bank') != 'ready':
            messagebox.showwarning("No Pattern Bank",
                                   "Generate patterns first by setting the pattern\n"
                                   "mode to 'Generate New AI Patterns' and\n"
                                   "generating patches.",
                                   parent=self.dialog)
            return

        def done(result):
            if not self._closed:
                self.model_status_label.config(text="Applied patches + 12 patterns",
                                               fg=COLORS['led_on'])
        self._act('ai.replace_patterns', "Apply Failed", on_done=done,
                  channels=self._checked_slots())

    # ================================================================
    # Refresh and cleanup
    # ================================================================

    def _refresh(self):
        """Show the core's AI state (on the dialog's own timer)."""
        if self._closed:
            return
        self._update_model_status()
        for i in range(8):
            self._update_slot_display(i)
        self._update_pattern_controls()
        self._update_preview_buttons()
        self._refresh_job = self.dialog.after(100, self._refresh)

    def _on_close(self):
        if self._closed:
            return
        self._closed = True
        if self._refresh_job is not None:
            self.dialog.after_cancel(self._refresh_job)
        # Save temperature preferences
        for name, var in (('patch', self.temp_var), ('pattern', self.pattern_temp_var)):
            try:
                self.core.set(f'pref.ai.{name}_temperature', var.get())
            except (tk.TclError, ValueError):
                pass  # not a number: keep the saved one
        # Slots not kept go back to their sounds, the preset's patterns stay
        self._act('ai.clear')
        self.dialog.destroy()
        if self.on_close is not None:
            self.on_close()
