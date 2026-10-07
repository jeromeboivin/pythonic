"""
Main GUI Window for Pythonic
Visual interface of the drum synthesizer
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import json
import os
import random
import copy

# Import our synthesizer
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pythonic.sequencer import StepSequencer, STEP_TICKS
from pythonic.pattern_manager import PatternManager
from pythonic.lfo import ModTarget, MOD_TARGET_GROUPS, MOD_TARGET_LABELS
from gui.widgets import (
    RotaryKnob, VerticalSlider, ChannelButton,
    WaveformSelector, ModeSelector, ToggleButton, PatternEditor, MatrixEditor,
    STEPS_PER_PAGE, MAX_PAGES
)
from gui.po32_transfer import PO32TransferDialog
from gui.po32_import_dialog import PO32ImportDialog
from gui.drum_generator_dialog import DrumGeneratorDialog
from pythonic.drum_generator import infer_drum_type
from pythonic.pattern_generator import PatternGenerator
from pythonic.preset_manager import channel_to_raw_patch

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: sounddevice not available. Audio playback disabled.")

try:
    import mido
    MIDI_AVAILABLE = True
except ImportError:
    MIDI_AVAILABLE = False
    print("Warning: mido not available. MIDI export disabled.")

from pythonic.app import AppCore
from pythonic.app.export import pattern_midi_file
from pythonic.app.midi import cc_name
from pythonic.app.sound import cc_parameter_target

class PythonicGUI:
    """
    Main GUI application for Pythonic
    """
    
    # Color scheme matching Pythonic
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
    }
    
    def __init__(self, core=None):
        # Create main window
        self.root = tk.Tk()
        self.root.title("Pythonic Drum Synthesizer")
        self.root.configure(bg=self.COLORS['bg_dark'])
        self.root.resizable(True, True)
        self.root.minsize(800, 480)  # Compact minimum window size
        self.root.geometry("960x600")  # Compact default window size

        # The app core builds the preferences, synth, patterns, morph and
        # preset manager, and owns the audio stream. The GUI reads transport,
        # play position, modulation and action results through core.poll().
        self.core = core if core is not None else AppCore()
        self._poll_version = 0
        self._action_callbacks = {}  # action id -> callback(event), run by the UI tick
        self._audio_start_action = None

        # MIDI input runs in the core; the GUI shows what poll reports
        self._midi_activity = 0  # message counter last shown on the LED
        self._midi_notes = [0] * 8  # note counters last shown on the channel buttons
        self._midi_learn_action = None  # id of the running learn action
        self._midi_learn_widget = None
        self._midi_learn_flash_id = None
        self._last_transport = None  # (playing, selected, queued, playing pattern) last shown
        # Transport as last reported by poll; the editors show pattern _pattern
        self._transport = self.core.poll()['transport']
        self._pattern = self._transport['selected_pattern']
        self._dirty_lanes = set()  # channels whose lane refresh waits for a drag to end

        # UI state. The selected channel is a copy of the core's
        # global.channel (0-based here), updated from poll.
        self.selected_channel = self.core.get('global.channel') - 1
        self.updating_ui = False  # Prevent feedback loops
        
        # Undo/redo state management
        self._undo_stack = []  # List of (synth_data, pattern_data) snapshots
        self._redo_stack = []
        self._max_undo = 50  # Maximum undo levels
        self._undo_pending = False  # Debounce flag for coalescing rapid changes
        self._restore_pending = False  # A snapshot restore is queued in the core
        
        # Controls offered for MIDI learn: parameter name -> widget. The core
        # maps CCs to their targets (cc_parameter_target(name)).
        self._cc_widgets = {}
        
        # Playback display state
        self.button_flash_state = False  # For flashing playing pattern button
        self.ui_update_timer = None  # Timer for UI updates

        # Build the interface
        self._build_ui()
        self._build_address_widget_tables()
        
        # Offer the controls for MIDI learn (right-click menu)
        self._register_cc_parameters()
        
        # Register undo/redo on all knobs and sliders
        self._register_undo_on_widgets()
        
        # Bind keyboard events
        self.root.bind('<Key>', self._on_key_press)
        self.root.bind('<Control-z>', lambda e: self._on_undo())
        self.root.bind('<Control-y>', lambda e: self._on_redo())
        
        # Update UI with current channel
        self._update_ui_from_channel()
        self._update_morph_ui()

        # Start button state updates
        self.root.after(250, self._toggle_button_flash)

        # Start UI position update timer (separate from audio thread)
        self._start_ui_update_timer()

        # Load the last preset if available
        self._load_last_preset()

        # Start-up is done: the core freezes the GC heap and opens the stream
        self._audio_start_action = self.core.start()

    # ============== App core services ==============
    # Code not yet moved into the core still reaches these engine objects
    # directly; they are looked up on the core because the synth is replaced
    # when the synth rate changes.

    @property
    def synth(self):
        return self.core.synth

    @property
    def pattern_manager(self):
        return self.core.pattern_manager

    @property
    def preset_manager(self):
        return self.core.preset_manager

    @property
    def morph_manager(self):
        return self.core.morph_manager

    @property
    def preferences_manager(self):
        return self.core.preferences

    @property
    def synth_sample_rate(self):
        return self.core.get('audio.synth_rate')

    def _when_action_done(self, action_id, callback):
        """Run callback(event) on the UI tick once the core reports the action."""
        self._action_callbacks[action_id] = callback

    def _build_ui(self):
        """Build the complete user interface
        
        Layout follows this structure (top to bottom):
        1. Toolbar (program selector, morph, undo/redo, options)
        2. Preset Section (preset name, channel buttons 1-8 in 2 rows, mute buttons)
        3. Drum Patch Section (mixing, oscillator, noise, velocity controls)
        4. Pattern Section (pattern buttons A-L, step editor, play controls)
        5. Global Section (transport, swing, fill rate, master volume)
        """
        # Scrollable main container
        outer_frame = tk.Frame(self.root, bg=self.COLORS['bg_dark'])
        outer_frame.pack(fill='both', expand=True)
        
        self._canvas = tk.Canvas(outer_frame, bg=self.COLORS['bg_dark'],
                                 highlightthickness=0)
        self._vscrollbar = tk.Scrollbar(outer_frame, orient='vertical',
                                        command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._vscrollbar.set)
        
        self._vscrollbar.pack(side='right', fill='y')
        self._canvas.pack(side='left', fill='both', expand=True)
        
        main_frame = tk.Frame(self._canvas, bg=self.COLORS['bg_dark'])
        self._canvas_window = self._canvas.create_window((0, 0), window=main_frame,
                                                          anchor='nw')
        
        # Resize canvas scroll region when content changes
        def _on_frame_configure(event):
            self._canvas.configure(scrollregion=self._canvas.bbox('all'))
        main_frame.bind('<Configure>', _on_frame_configure)
        
        # Make canvas width follow window width
        def _on_canvas_configure(event):
            self._canvas.itemconfig(self._canvas_window, width=event.width)
        self._canvas.bind('<Configure>', _on_canvas_configure)
        
        # Bind mouse wheel to scroll
        def _on_mousewheel(event):
            self._canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
        self._canvas.bind_all('<MouseWheel>', _on_mousewheel)
        
        # Toolbar (top bar with program selector, morph slider)
        self._build_toolbar(main_frame)
        
        # Preset section (preset name, channel buttons in 2 rows)
        self._build_preset_section(main_frame)
        
        # Drum Patch section (main controls)
        self._build_drum_patch_section(main_frame)
        
        # Pattern section
        self._build_pattern_section(main_frame)
        
        # Global section (transport, master volume)
        self._build_global_section(main_frame)
    
    def _build_toolbar(self, parent):
        """Build toolbar with program selector, sound morph, and options
        
        Toolbar layout:
        - Left: Program selector (1-16)
        - Center: Sound Morph slider
        - Right: Undo/Redo, option buttons, master volume
        """
        toolbar = tk.Frame(parent, bg=self.COLORS['bg_medium'], height=32)
        toolbar.pack(fill='x', pady=(0, 2))
        toolbar.pack_propagate(False)
        
        # Left side: Program selector
        program_frame = tk.Frame(toolbar, bg=self.COLORS['bg_medium'])
        program_frame.pack(side='left', padx=3, pady=2)
        
        tk.Label(program_frame, text="program:", 
                font=('Segoe UI', 7), fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack(side='left', padx=(0, 3))
        
        self.program_var = tk.StringVar(value="1")
        self.program_combo = ttk.Combobox(program_frame, textvariable=self.program_var,
                                         values=[str(i) for i in range(1, 17)],
                                         width=3, state='readonly')
        self.program_combo.pack(side='left')
        self.program_combo.bind('<<ComboboxSelected>>', self._on_program_select)
        
        # Center: Sound Morph slider with A/B learn buttons
        morph_frame = tk.Frame(toolbar, bg=self.COLORS['bg_medium'])
        morph_frame.pack(side='left', padx=10, pady=2, expand=True)
        
        tk.Label(morph_frame, text="sound morph:", 
                font=('Segoe UI', 7), fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack(side='left', padx=(0, 3))
        
        # Learn A button (small square)
        self.morph_learn_a_btn = tk.Button(
            morph_frame, text="A", width=2, height=1,
            font=('Segoe UI', 7, 'bold'),
            bg='#444455', fg=self.COLORS['text_dim'],
            activebackground='#555566',
            relief='flat', bd=1,
            command=self._on_morph_learn_a)
        self.morph_learn_a_btn.pack(side='left', padx=(0, 2))
        
        self.morph_slider = tk.Scale(morph_frame, from_=0, to=100, 
                                    orient='horizontal', length=120,
                                    bg=self.COLORS['bg_medium'], 
                                    fg=self.COLORS['text'],
                                    highlightthickness=0,
                                    troughcolor=self.COLORS['bg_dark'],
                                    command=self._on_morph_change)
        self.morph_slider.pack(side='left')
        
        # Learn B button (small square)
        self.morph_learn_b_btn = tk.Button(
            morph_frame, text="B", width=2, height=1,
            font=('Segoe UI', 7, 'bold'),
            bg='#444455', fg=self.COLORS['text_dim'],
            activebackground='#555566',
            relief='flat', bd=1,
            command=self._on_morph_learn_b)
        self.morph_learn_b_btn.pack(side='left', padx=(2, 0))
        
        # Right side: Undo/Redo, options, master volume
        right_frame = tk.Frame(toolbar, bg=self.COLORS['bg_medium'])
        right_frame.pack(side='right', padx=3, pady=2)
        
        # Undo/Redo buttons
        self.undo_btn = tk.Button(right_frame, text="↶", width=2, height=1,
                                 font=('Segoe UI', 10),
                                 bg=self.COLORS['bg_light'],
                                 fg=self.COLORS['text'],
                                 state='disabled',
                                 command=self._on_undo)
        self.undo_btn.pack(side='left', padx=1)
        
        self.redo_btn = tk.Button(right_frame, text="↷", width=2, height=1,
                                 font=('Segoe UI', 10),
                                 bg=self.COLORS['bg_light'],
                                 fg=self.COLORS['text'],
                                 state='disabled',
                                 command=self._on_redo)
        self.redo_btn.pack(side='left', padx=1)
        
        # Separator
        tk.Frame(right_frame, width=10, bg=self.COLORS['bg_medium']).pack(side='left')
        
        # Master volume
        tk.Label(right_frame, text="master", 
                font=('Segoe UI', 7), fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack(side='left', padx=(5, 2))
        
        self.master_knob = RotaryKnob(right_frame, size=28, 
                                      min_val=-60, max_val=10, default=0,
                                      command=self._on_master_volume_change)
        self.master_knob.pack(side='left')
        
        # MIDI activity indicator
        tk.Frame(right_frame, width=5, bg=self.COLORS['bg_medium']).pack(side='left')
        self.midi_indicator = tk.Canvas(right_frame, width=12, height=12,
                                        bg=self.COLORS['bg_medium'], 
                                        highlightthickness=0)
        self.midi_indicator.pack(side='left', padx=(5, 2))
        self._midi_indicator_id = self.midi_indicator.create_oval(
            2, 2, 10, 10, fill=self.COLORS['led_off'], outline='')
        # Bind click to open MIDI preferences
        self.midi_indicator.bind('<Button-1>', lambda e: self._show_midi_preferences())
    
    def _build_header(self, parent):
        """Build header with title and program selector - DEPRECATED, use _build_toolbar"""
        # This method kept for backwards compatibility but not used
        pass
    
    def _build_preset_section(self, parent):
        """Build the preset/channel selection section
        
        Toolbar layout:
        - Left: Logo "PYTHONIC" 
        - Center: Preset name display with prev/next buttons
        - Right: Channel buttons 1-8 in SINGLE row with small mute buttons, patch name below
        """
        preset_section = tk.Frame(parent, bg=self.COLORS['bg_medium'])
        preset_section.pack(fill='x', pady=(0, 2))
        
        # Left: Logo/Title
        logo_frame = tk.Frame(preset_section, bg=self.COLORS['bg_medium'])
        logo_frame.pack(side='left', padx=5, pady=2)
        
        tk.Label(logo_frame, text="PYTHONIC", 
                font=('Segoe UI', 11, 'bold'), fg=self.COLORS['accent_light'],
                bg=self.COLORS['bg_medium']).pack()
        
        # PO-32 Transfer button
        self.po32_btn = tk.Button(logo_frame, text="PO-32",
                                  font=('Segoe UI', 7, 'bold'),
                                  bg=self.COLORS['bg_light'],
                                  fg=self.COLORS['orange'],
                                  activebackground=self.COLORS['bg_medium'],
                                  width=5, height=1,
                                  command=self._show_po32_transfer,
                                  relief='raised', bd=1)
        self.po32_btn.pack(pady=(2, 0))
        
        # Center-left: Preset name display with navigation
        preset_nav_frame = tk.Frame(preset_section, bg=self.COLORS['bg_medium'])
        preset_nav_frame.pack(side='left', padx=5, pady=2)
        
        tk.Button(preset_nav_frame, text="◀", width=2, height=1,
                 font=('Segoe UI', 8),
                 bg=self.COLORS['bg_light'],
                 fg=self.COLORS['text'],
                 command=self._on_preset_prev).pack(side='left', padx=1)
        
        self.preset_combo = ttk.Combobox(preset_nav_frame, width=25, state='readonly')
        self.preset_combo.pack(side='left', padx=2)
        self.preset_combo.bind('<<ComboboxSelected>>', self._on_preset_combo_select)
        
        tk.Button(preset_nav_frame, text="▶", width=2, height=1,
                 font=('Segoe UI', 8),
                 bg=self.COLORS['bg_light'],
                 fg=self.COLORS['text'],
                 command=self._on_preset_next).pack(side='left', padx=1)
        
        tk.Button(preset_nav_frame, text="▼", width=2, height=1,
                 font=('Segoe UI', 7),
                 bg=self.COLORS['accent'],
                 fg=self.COLORS['text'],
                 command=self._on_preset_menu).pack(side='left', padx=2)
        
        self._refresh_preset_list()
        
        # Right side: Channel buttons 1-8 in SINGLE ROW
        channels_container = tk.Frame(preset_section, bg=self.COLORS['bg_medium'])
        channels_container.pack(side='right', padx=5, pady=2)
        
        # Patch name display ABOVE channels
        self.patch_name_label = tk.Label(channels_container,
                                         text="SC BD Schmack",
                                         font=('Segoe UI', 8),
                                         fg=self.COLORS['text'],
                                         bg=self.COLORS['bg_dark'],
                                         width=18, anchor='center',
                                         relief='sunken', padx=3)
        self.patch_name_label.pack(pady=(0, 1))
        
        # Single row of channels 1-8 with number, button, mute
        channels_row = tk.Frame(channels_container, bg=self.COLORS['bg_medium'])
        channels_row.pack()
        
        self.channel_buttons = []
        self.mute_buttons = []
        self.channel_type_labels = []
        
        for i in range(8):
            ch_frame = tk.Frame(channels_row, bg=self.COLORS['bg_medium'])
            ch_frame.pack(side='left', padx=1)
            
            # Channel number label on top
            tk.Label(ch_frame, text=str(i + 1), font=('Segoe UI', 7),
                    fg=self.COLORS['text_dim'],
                    bg=self.COLORS['bg_medium']).pack()
            
            # Channel button and mute in a row
            btn_row = tk.Frame(ch_frame, bg=self.COLORS['bg_medium'])
            btn_row.pack()
            
            btn = ChannelButton(btn_row, i, size=24,
                               command=self._on_channel_select)
            btn.pack(side='left')
            self.channel_buttons.append(btn)
            
            # Small mute button
            mute_btn = ToggleButton(btn_row, text="m", width=14, height=14,
                                   command=lambda en, ch=i: self._on_mute_toggle(ch, en))
            mute_btn.pack(side='left', padx=1)
            self.mute_buttons.append(mute_btn)
            
            # Drum type label under channel (TR-8 style)
            type_lbl = tk.Label(ch_frame, text="", font=('Segoe UI', 6),
                               fg=self.COLORS['orange'],
                               bg=self.COLORS['bg_medium'],
                               width=5, anchor='center')
            type_lbl.pack()
            self.channel_type_labels.append(type_lbl)
        
        self.channel_buttons[self.selected_channel].set_selected(True)
    
    def _build_pattern_section(self, parent):
        """Build the pattern editor section
        Left side: Pattern buttons (A-L), matrix toggle, chain controls
        Right side: Pattern editor (trig/acc/fill/len lanes)
        """
        pattern_frame = tk.LabelFrame(parent, text="pattern", 
                                     font=('Segoe UI', 7),
                                     fg=self.COLORS['text_dim'],
                                     bg=self.COLORS['bg_medium'],
                                     labelanchor='nw')
        pattern_frame.pack(fill='x', pady=(0, 2))
        
        # Main horizontal layout: controls on left, editor on right
        main_row = tk.Frame(pattern_frame, bg=self.COLORS['bg_medium'])
        main_row.pack(fill='both', expand=True, padx=3, pady=2)
        
        # LEFT SIDE: Pattern controls
        left_panel = tk.Frame(main_row, bg=self.COLORS['bg_medium'])
        left_panel.pack(side='left', fill='y', padx=(0, 10))
        
        # Matrix Editor toggle button
        self.matrix_toggle_btn = tk.Button(left_panel, text="⊞", width=2, height=1,
                                          font=('Segoe UI', 8),
                                          bg=self.COLORS['bg_light'],
                                          fg=self.COLORS['text'],
                                          command=self._on_matrix_toggle)
        self.matrix_toggle_btn.pack(anchor='w', pady=2)
        
        # Pattern selection buttons (A-L) in 3 rows of 4
        btn_container = tk.Frame(left_panel, bg=self.COLORS['bg_medium'])
        btn_container.pack(pady=5)
        
        self.pattern_buttons = []
        for row in range(3):
            row_frame = tk.Frame(btn_container, bg=self.COLORS['bg_medium'])
            row_frame.pack()
            for col in range(4):
                idx = row * 4 + col
                name = PatternManager.PATTERN_NAMES[idx]
                btn = tk.Button(row_frame, text=name, width=2, height=1,
                               font=('Segoe UI', 7),
                               bg=self.COLORS['bg_light'],
                               fg=self.COLORS['text'],
                               command=lambda i=idx: self._on_pattern_select(i))
                btn.pack(side='left', padx=1, pady=1)
                btn.bind('<Button-3>', lambda e, i=idx: self._on_pattern_right_click(i, e))
                self.pattern_buttons.append(btn)
        
        self.pattern_buttons[0].config(bg=self.COLORS['highlight'])
        
        # Chain controls
        chain_frame = tk.Frame(left_panel, bg=self.COLORS['bg_medium'])
        chain_frame.pack(pady=5)
        
        tk.Label(chain_frame, text="- chain -", 
                font=('Segoe UI', 6), fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack()
        
        chain_btn_frame = tk.Frame(chain_frame, bg=self.COLORS['bg_medium'])
        chain_btn_frame.pack()
        
        tk.Button(chain_btn_frame, text="◀◀", width=3,
                 font=('Segoe UI', 7),
                 bg=self.COLORS['bg_light'],
                 fg=self.COLORS['text_dim'],
                 command=self._on_chain_previous).pack(side='left', padx=1)
        
        tk.Button(chain_btn_frame, text="▶▶", width=3,
                 font=('Segoe UI', 7),
                 bg=self.COLORS['bg_light'],
                 fg=self.COLORS['text_dim'],
                 command=self._on_chain_next).pack(side='left', padx=1)
        
        # RIGHT SIDE: Pattern editor and controls
        right_panel = tk.Frame(main_row, bg=self.COLORS['bg_medium'])
        right_panel.pack(side='left', fill='both', expand=True)
        
        # Top row: Menu/Copy/Paste only (fill rate, step rate, swing are in bottom bar)
        top_controls = tk.Frame(right_panel, bg=self.COLORS['bg_medium'])
        top_controls.pack(fill='x', pady=(0, 5))
        
        # Page selector (left side): the editors show 16 steps of a pattern
        # of up to 64; follow makes them track the playhead
        page_frame = tk.Frame(top_controls, bg=self.COLORS['bg_medium'])
        page_frame.pack(side='left', padx=2)
        tk.Label(page_frame, text="page", font=('Segoe UI', 6),
                 fg=self.COLORS['text_dim'], bg=self.COLORS['bg_medium']).pack(side='left', padx=(0, 2))
        self._page = 0
        self._page_follow = True
        self._playing_page = None
        self.page_buttons = []
        for page in range(MAX_PAGES):
            first = page * STEPS_PER_PAGE
            btn = tk.Button(page_frame, text=f"{first + 1}-{first + STEPS_PER_PAGE}", width=5,
                            font=('Segoe UI', 7),
                            bg=self.COLORS['bg_light'], fg=self.COLORS['text'],
                            command=lambda p=page: self._on_page_select(p))
            btn.pack(side='left', padx=1)
            self.page_buttons.append(btn)
        self.page_follow_btn = tk.Button(page_frame, text="follow", width=5,
                                         font=('Segoe UI', 7),
                                         bg=self.COLORS['bg_light'], fg=self.COLORS['text_dim'],
                                         command=self._on_page_follow_toggle)
        self.page_follow_btn.pack(side='left', padx=(4, 1))

        # Menu/Copy/Paste (right side)
        clipboard_frame = tk.Frame(top_controls, bg=self.COLORS['bg_medium'])
        clipboard_frame.pack(side='right', padx=2)

        tk.Button(clipboard_frame, text="Menu", width=4,
                 font=('Segoe UI', 7),
                 bg=self.COLORS['accent'],
                 fg=self.COLORS['text'],
                 command=self._on_pattern_menu).pack(side='left', padx=1)

        tk.Button(clipboard_frame, text="Copy", width=4,
                 font=('Segoe UI', 7),
                 bg=self.COLORS['bg_light'],
                 fg=self.COLORS['text_dim'],
                 command=self._on_pattern_copy).pack(side='left', padx=1)

        tk.Button(clipboard_frame, text="Paste", width=4,
                 font=('Segoe UI', 7),
                 bg=self.COLORS['bg_light'],
                 fg=self.COLORS['text_dim'],
                 command=self._on_pattern_paste).pack(side='left', padx=1)
        
        # Probability mode toggle button
        self.prob_mode_btn = tk.Button(clipboard_frame, text="Prob", width=4,
                                      font=('Segoe UI', 7),
                                      bg=self.COLORS['bg_light'],
                                      fg=self.COLORS['text_dim'],
                                      command=self._on_toggle_prob_mode)
        self.prob_mode_btn.pack(side='left', padx=1)
        self.probability_mode_active = False

        # Pattern editor area
        self.editors_container = tk.Frame(right_panel, bg=self.COLORS['bg_dark'])
        self.editors_container.pack(fill='both', expand=True, pady=2)
        
        # Single channel pattern editor frame
        self.single_editor_frame = tk.Frame(self.editors_container, bg=self.COLORS['bg_dark'])
        self.single_editor_frame.pack(fill='both', expand=True)
        
        # Channel label + pattern editor
        self.current_editor_frame = tk.Frame(self.single_editor_frame, bg=self.COLORS['bg_dark'])
        self.current_editor_frame.pack(fill='both', expand=True)
        
        self.pattern_channel_label = tk.Label(self.current_editor_frame, text="ch1",
                                             font=('Segoe UI', 8), fg=self.COLORS['text_dim'],
                                             bg=self.COLORS['bg_dark'], anchor='w')
        self.pattern_channel_label.pack(side='left', padx=2)
        
        # Create pattern editors for all 8 channels
        self.pattern_editors = []
        for ch in range(8):
            editor = PatternEditor(self.current_editor_frame, channel_id=ch, pattern_length=16,
                                  num_steps=16,
                                  command=self._on_pattern_edit,
                                  all_channels_command=self._on_pattern_edit_all,
                                  length_change_callback=self._on_pattern_length_change)
            self.pattern_editors.append(editor)
        
        # Show first channel's editor
        self.pattern_editors[0].pack(side='left', fill='both', expand=True)
        self.current_pattern_editor_index = 0
        
        # Matrix editor (hidden by default)
        self.matrix_editor = MatrixEditor(self.editors_container, num_channels=8, 
                                         num_steps=16,
                                         command=self._on_matrix_edit)
        
        self.matrix_view_active = False

    def _build_drum_patch_section(self, parent):
        """Build the main drum patch editing section"""
        patch_frame = tk.Frame(parent, bg=self.COLORS['bg_medium'])
        patch_frame.pack(fill='x', expand=True, pady=(0, 2))
        
        # Seven subsections — lfo1, lfo2, pump racks to the right of vel
        self._build_mixing_section(patch_frame)
        self._build_oscillator_section(patch_frame)
        self._build_noise_section(patch_frame)
        self._build_fx_section(patch_frame)
        self._build_velocity_section(patch_frame)
        self._build_modulation_section(patch_frame)
    
    def _build_mixing_section(self, parent):
        """Build the mixing controls section"""
        section = tk.LabelFrame(parent, text="mixing", 
                               font=('Segoe UI', 7),
                               fg=self.COLORS['text_dim'],
                               bg=self.COLORS['bg_medium'],
                               labelanchor='n')
        section.pack(side='left', padx=2, pady=2, fill='both', expand=True)
        
        # Row 1: osc/noise HORIZONTAL mix slider
        mix_row = tk.Frame(section, bg=self.COLORS['bg_medium'])
        mix_row.pack(pady=2, fill='x')
        
        tk.Label(mix_row, text="osc", font=('Segoe UI', 7),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack(side='left', padx=2)
        
        # Horizontal mix slider
        # Show oscillator on the left and noise on the right.
        # Internal parameter semantics remain 0=noise, 100=oscillator.
        self.mix_slider = tk.Scale(mix_row, from_=100, to=0, 
                                  orient='horizontal', length=60,
                                  showvalue=False,
                                  bg=self.COLORS['bg_medium'], 
                                  fg=self.COLORS['text'],
                                  highlightthickness=0,
                                  troughcolor=self.COLORS['bg_dark'],
                                  command=self._on_mix_change)
        self.mix_slider.set(50)
        self.mix_slider.pack(side='left', padx=2)
        
        tk.Label(mix_row, text="noise", font=('Segoe UI', 7),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack(side='left', padx=2)
        
        # Row 2: EQ Freq knob with frequency labels
        freq_row = tk.Frame(section, bg=self.COLORS['bg_medium'])
        freq_row.pack(pady=1)
        
        tk.Label(freq_row, text="20Hz", font=('Segoe UI', 6),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack(side='left')
        
        self.eq_freq_knob = RotaryKnob(freq_row, size=35,
                                       min_val=20, max_val=20000, default=632,
                                       label="eq freq",
                                       logarithmic=True,
                                       command=self._on_eq_freq_change)
        self.eq_freq_knob.pack(side='left', padx=2)
        
        tk.Label(freq_row, text="20kHz", font=('Segoe UI', 6),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack(side='left')
        
        # Edit All button
        self.edit_all_btn = ToggleButton(section, text="edit all", width=45, height=16,
                                        command=self._on_edit_all_toggle)
        self.edit_all_btn.pack(pady=1)
        
        # Row 3: Distortion and EQ Gain
        row3 = tk.Frame(section, bg=self.COLORS['bg_medium'])
        row3.pack(pady=1)
        
        self.distort_knob = RotaryKnob(row3, size=32,
                                       min_val=0, max_val=100, default=0,
                                       label="distort",
                                       command=self._on_distort_change)
        self.distort_knob.pack(side='left', padx=2)
        
        self.eq_gain_knob = RotaryKnob(row3, size=32,
                                       min_val=-40, max_val=40, default=0,
                                       label="eq gain",
                                       command=self._on_eq_gain_change)
        self.eq_gain_knob.pack(side='left', padx=2)
        
        # Row 4: Level and Pan
        row4 = tk.Frame(section, bg=self.COLORS['bg_medium'])
        row4.pack(pady=1)
        
        self.level_knob = RotaryKnob(row4, size=32,
                                     min_val=-60, max_val=10, default=0,
                                     label="level",
                                     command=self._on_level_change)
        self.level_knob.pack(side='left', padx=2)
        
        self.pan_knob = RotaryKnob(row4, size=32,
                                   min_val=-100, max_val=100, default=0,
                                   label="pan",
                                   command=self._on_pan_change)
        self.pan_knob.pack(side='left', padx=2)
        
        # Row 5: Choke and Output A/B
        row5 = tk.Frame(section, bg=self.COLORS['bg_medium'])
        row5.pack(pady=1)
        
        self.choke_btn = ToggleButton(row5, text="choke", width=40, height=16,
                                     command=self._on_choke_toggle)
        self.choke_btn.pack(side='left', padx=2)
        
        self.output_selector = ModeSelector(row5, options=['A', 'B'], width=45,
                                           command=self._on_output_change)
        self.output_selector.pack(side='left', padx=2)
    
    def _build_oscillator_section(self, parent):
        """Build the oscillator controls section"""
        section = tk.LabelFrame(parent, text="oscillator",
                               font=('Segoe UI', 7),
                               fg=self.COLORS['text_dim'],
                               bg=self.COLORS['bg_medium'],
                               labelanchor='n')
        section.pack(side='left', padx=2, pady=2, fill='both', expand=True)
        
        # Waveform selector with label
        waveform_frame = tk.Frame(section, bg=self.COLORS['bg_medium'])
        waveform_frame.pack(pady=1)
        
        tk.Label(waveform_frame, text="waveform", font=('Segoe UI', 7),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack()
        
        self.waveform_selector = WaveformSelector(waveform_frame,
                                                  command=self._on_waveform_change)
        self.waveform_selector.pack()
        
        # Oscillator Frequency with labels
        freq_frame = tk.Frame(section, bg=self.COLORS['bg_medium'])
        freq_frame.pack(pady=1)
        
        tk.Label(freq_frame, text="20Hz", font=('Segoe UI', 6),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack(side='left')
        
        self.osc_freq_knob = RotaryKnob(freq_frame, size=38,
                                        min_val=20, max_val=20000, default=440,
                                        label="osc freq",
                                        logarithmic=True,
                                        command=self._on_osc_freq_change)
        self.osc_freq_knob.pack(side='left', padx=2)
        
        tk.Label(freq_frame, text="20kHz", font=('Segoe UI', 6),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack(side='left')
        
        # Pitch (tune) offset in semitones
        pitch_frame = tk.Frame(section, bg=self.COLORS['bg_medium'])
        pitch_frame.pack(pady=1)
        
        self.pitch_knob = RotaryKnob(pitch_frame, size=35,
                                     min_val=-24, max_val=24, default=0,
                                     label="pitch", unit="st",
                                     command=self._on_pitch_change)
        self.pitch_knob.pack(side='left', padx=2)
        
        # Pitch modulation mode
        pitch_mod_frame = tk.Frame(section, bg=self.COLORS['bg_medium'])
        pitch_mod_frame.pack(pady=1)
        
        tk.Label(pitch_mod_frame, text="pitch mod", font=('Segoe UI', 7),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack()
        
        self.pitch_mod_mode = ModeSelector(pitch_mod_frame, 
                                          options=['Decay', 'Sine', 'Rand'],
                                          command=self._on_pitch_mod_mode_change)
        self.pitch_mod_mode.pack()
        
        # Pitch mod amount and rate
        pitch_knobs_frame = tk.Frame(section, bg=self.COLORS['bg_medium'])
        pitch_knobs_frame.pack(pady=1)
        
        self.pitch_amount_knob = RotaryKnob(pitch_knobs_frame, size=32,
                                            min_val=-120, max_val=120, default=0,
                                            label="amount",
                                            command=self._on_pitch_amount_change)
        self.pitch_amount_knob.pack(side='left', padx=2)
        
        self.pitch_rate_knob = RotaryKnob(pitch_knobs_frame, size=32,
                                          min_val=1, max_val=2000, default=100,
                                          label="rate",
                                          logarithmic=True,
                                          command=self._on_pitch_rate_change)
        self.pitch_rate_knob.pack(side='left', padx=2)
        
        # Attack and Decay knobs
        env_frame = tk.Frame(section, bg=self.COLORS['bg_medium'])
        env_frame.pack(pady=1)
        
        self.osc_attack_knob = RotaryKnob(env_frame, size=32,
                                          min_val=0, max_val=10000, default=0,
                                          label="attack",
                                          logarithmic=True,
                                          command=self._on_osc_attack_change)
        self.osc_attack_knob.pack(side='left', padx=2)
        
        self.osc_decay_knob = RotaryKnob(env_frame, size=32,
                                         min_val=10, max_val=10000, default=316,
                                         label="decay",
                                         logarithmic=True,
                                         command=self._on_osc_decay_change)
        self.osc_decay_knob.pack(side='left', padx=2)
    
    def _build_noise_section(self, parent):
        """Build the noise generator controls section"""
        section = tk.LabelFrame(parent, text="noise",
                               font=('Segoe UI', 7),
                               fg=self.COLORS['text_dim'],
                               bg=self.COLORS['bg_medium'],
                               labelanchor='n')
        section.pack(side='left', padx=2, pady=2, fill='both', expand=True)
        
        # Filter mode selector (LP/BP/HP)
        filter_mode_frame = tk.Frame(section, bg=self.COLORS['bg_medium'])
        filter_mode_frame.pack(pady=1)
        
        tk.Label(filter_mode_frame, text="filter mode", font=('Segoe UI', 7),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack()
        
        self.noise_filter_mode = ModeSelector(filter_mode_frame,
                                             options=['LP', 'BP', 'HP'],
                                             command=self._on_noise_filter_mode_change)
        self.noise_filter_mode.pack()
        
        # Filter freq with frequency labels
        freq_frame = tk.Frame(section, bg=self.COLORS['bg_medium'])
        freq_frame.pack(pady=1)
        
        tk.Label(freq_frame, text="20Hz", font=('Segoe UI', 6),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack(side='left')
        
        self.noise_freq_knob = RotaryKnob(freq_frame, size=38,
                                          min_val=20, max_val=20000, default=20000,
                                          label="filter freq",
                                          logarithmic=True,
                                          command=self._on_noise_freq_change)
        self.noise_freq_knob.pack(side='left', padx=2)
        
        tk.Label(freq_frame, text="20kHz", font=('Segoe UI', 6),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack(side='left')
        
        # Filter Q knob
        q_frame = tk.Frame(section, bg=self.COLORS['bg_medium'])
        q_frame.pack(pady=1)
        
        self.noise_q_knob = RotaryKnob(q_frame, size=32,
                                       min_val=0.5, max_val=20.0, default=0.707,
                                       label="filter q",
                                       logarithmic=True,
                                       command=self._on_noise_q_change)
        self.noise_q_knob.pack(side='left', padx=2)
        
        # Stereo toggle button
        self.stereo_btn = ToggleButton(q_frame, text="stereo", width=40, height=16,
                                      command=self._on_stereo_toggle)
        self.stereo_btn.pack(side='left', padx=2)
        
        # Envelope mode selector
        env_mode_frame = tk.Frame(section, bg=self.COLORS['bg_medium'])
        env_mode_frame.pack(pady=1)
        
        tk.Label(env_mode_frame, text="envelope", font=('Segoe UI', 7),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack()
        
        self.noise_env_mode = ModeSelector(env_mode_frame,
                                          options=['Exp', 'Lin', 'Mod'],
                                          command=self._on_noise_env_mode_change)
        self.noise_env_mode.pack()
        
        # Attack and Decay as VERTICAL SLIDERS
        env_sliders_frame = tk.Frame(section, bg=self.COLORS['bg_medium'])
        env_sliders_frame.pack(pady=1)
        
        self.noise_attack_slider = VerticalSlider(env_sliders_frame, width=22, height=55,
                                                 min_val=0, max_val=10000, default=0,
                                                 label="attack",
                                                 logarithmic=True,
                                                 command=self._on_noise_attack_change)
        self.noise_attack_slider.pack(side='left', padx=3)
        
        self.noise_decay_slider = VerticalSlider(env_sliders_frame, width=22, height=55,
                                                min_val=10, max_val=10000, default=316,
                                                label="decay",
                                                logarithmic=True,
                                                command=self._on_noise_decay_change)
        self.noise_decay_slider.pack(side='left', padx=3)
    
    def _build_velocity_section(self, parent):
        """Build the velocity sensitivity section"""
        section = tk.LabelFrame(parent, text="vel",
                               font=('Segoe UI', 7),
                               fg=self.COLORS['text_dim'],
                               bg=self.COLORS['bg_medium'],
                               labelanchor='n')
        section.pack(side='left', padx=2, pady=2, fill='both', expand=True)
        
        # Oscillator velocity
        self.osc_vel_slider = VerticalSlider(section, width=22, height=55,
                                            min_val=0, max_val=200, default=0,
                                            label="osc",
                                            command=self._on_osc_vel_change)
        self.osc_vel_slider.pack(pady=2)
        
        # Noise velocity
        self.noise_vel_slider = VerticalSlider(section, width=22, height=55,
                                              min_val=0, max_val=200, default=0,
                                              label="noise",
                                              command=self._on_noise_vel_change)
        self.noise_vel_slider.pack(pady=2)
        
        # Mod velocity
        self.mod_vel_slider = VerticalSlider(section, width=22, height=55,
                                            min_val=0, max_val=200, default=0,
                                            label="mod",
                                            command=self._on_mod_vel_change)
        self.mod_vel_slider.pack(pady=2)
        
    def _build_modulation_section(self, parent):
        """Build three vertical modulation racks (lfo 1, lfo 2, pump) side-by-side,
        packed to the right of the velocity section."""
        # Build destination option list once (short labels for narrow comboboxes)
        self._mod_target_options = ['Off']
        self._mod_target_values = [ModTarget.NONE]
        for group_name, targets in MOD_TARGET_GROUPS.items():
            for t in targets:
                self._mod_target_options.append(MOD_TARGET_LABELS[t])
                self._mod_target_values.append(t)
        
        # Wave / sync option lists for comboboxes
        self._lfo_wave_options = ['Sin', 'Tri', 'Saw▲', 'Saw▼', 'Sq', 'S&H']
        self._lfo_sync_options = ['Free', '1/1', '1/2', '1/4', '1/8', '1/16',
                                  '1/4.', '1/8.', '1/4T', '1/8T', '2bar', '4bar']
        
        self._build_lfo_panel(parent, 'lfo1', 'lfo 1')
        self._build_lfo_panel(parent, 'lfo2', 'lfo 2')
        self._build_pump_panel(parent)
    
    def _build_lfo_panel(self, parent, lfo_id: str, title: str):
        """Build one vertical LFO rack (same height as other sections)."""
        section = tk.LabelFrame(parent, text=title,
                               font=('Segoe UI', 7),
                               fg=self.COLORS['text_dim'],
                               bg=self.COLORS['bg_medium'],
                               labelanchor='n')
        section.pack(side='left', padx=2, pady=2, fill='both')
        
        # Enable toggle
        enable_btn = ToggleButton(section, text="on", width=30, height=16,
                                  command=lambda v, lid=lfo_id: self._on_lfo_enable(lid, v))
        enable_btn.pack(pady=(2, 4))
        
        # Waveform combobox
        wave_var = tk.StringVar(value=self._lfo_wave_options[0])
        wave_combo = ttk.Combobox(section, textvariable=wave_var,
                                  values=self._lfo_wave_options,
                                  state='readonly', width=8,
                                  font=('Segoe UI', 7))
        wave_combo.pack(padx=2, pady=1)
        wave_combo.bind('<<ComboboxSelected>>',
                        lambda e, lid=lfo_id, wv=wave_var: self._on_lfo_waveform(lid, wv))
        
        # Rate knob
        rate_knob = RotaryKnob(section, size=32,
                               min_val=0.01, max_val=50, default=1.0,
                               label="rate",
                               logarithmic=True,
                               command=lambda v, lid=lfo_id: self._on_lfo_rate(lid, v))
        rate_knob.pack(pady=2)
        
        # Depth knob
        depth_knob = RotaryKnob(section, size=32,
                                min_val=0, max_val=100, default=0,
                                label="depth",
                                command=lambda v, lid=lfo_id: self._on_lfo_depth(lid, v))
        depth_knob.pack(pady=2)
        
        # Sync combobox (tempo division)
        sync_var = tk.StringVar(value=self._lfo_sync_options[0])
        sync_combo = ttk.Combobox(section, textvariable=sync_var,
                                  values=self._lfo_sync_options,
                                  state='readonly', width=8,
                                  font=('Segoe UI', 7))
        sync_combo.pack(padx=2, pady=1)
        sync_combo.bind('<<ComboboxSelected>>',
                        lambda e, lid=lfo_id, sv=sync_var: self._on_lfo_sync(lid, sv))
        
        # Retrigger + Polarity toggles side-by-side
        toggle_row = tk.Frame(section, bg=self.COLORS['bg_medium'])
        toggle_row.pack(pady=2)
        
        retrig_btn = ToggleButton(toggle_row, text="re", width=24, height=16,
                                  command=lambda v, lid=lfo_id: self._on_lfo_retrigger(lid, v))
        retrig_btn.set_value(True)
        retrig_btn.pack(side='left', padx=1)
        
        polar_btn = ToggleButton(toggle_row, text="uni", width=24, height=16,
                                 command=lambda v, lid=lfo_id: self._on_lfo_polarity(lid, v))
        polar_btn.pack(side='left', padx=1)
        
        # Destination combobox
        dest_var = tk.StringVar(value='Off')
        dest_combo = ttk.Combobox(section, textvariable=dest_var,
                                  values=self._mod_target_options,
                                  state='readonly', width=14,
                                  font=('Segoe UI', 7))
        dest_combo.pack(padx=2, pady=(2, 2))
        dest_combo.bind('<<ComboboxSelected>>',
                        lambda e, lid=lfo_id, dv=dest_var: self._on_lfo_dest(lid, dv))
        
        # Store widget refs
        setattr(self, f'{lfo_id}_enable_btn', enable_btn)
        setattr(self, f'{lfo_id}_wave_var', wave_var)
        setattr(self, f'{lfo_id}_wave_combo', wave_combo)
        setattr(self, f'{lfo_id}_rate_knob', rate_knob)
        setattr(self, f'{lfo_id}_depth_knob', depth_knob)
        setattr(self, f'{lfo_id}_sync_var', sync_var)
        setattr(self, f'{lfo_id}_sync_combo', sync_combo)
        setattr(self, f'{lfo_id}_retrig_btn', retrig_btn)
        setattr(self, f'{lfo_id}_polar_btn', polar_btn)
        setattr(self, f'{lfo_id}_dest_var', dest_var)
        setattr(self, f'{lfo_id}_dest_combo', dest_combo)
    
    def _build_pump_panel(self, parent):
        """Build the pump / sidechain vertical rack."""
        section = tk.LabelFrame(parent, text="pump",
                               font=('Segoe UI', 7),
                               fg=self.COLORS['text_dim'],
                               bg=self.COLORS['bg_medium'],
                               labelanchor='n')
        section.pack(side='left', padx=2, pady=2, fill='both')
        
        # Enable toggle
        self.pump_enable_btn = ToggleButton(section, text="on", width=30, height=16,
                                            command=self._on_pump_enable)
        self.pump_enable_btn.pack(pady=(2, 4))
        
        # Amount knob
        self.pump_amount_knob = RotaryKnob(section, size=32,
                                           min_val=0, max_val=100, default=0,
                                           label="amount",
                                           command=self._on_pump_amount)
        self.pump_amount_knob.pack(pady=2)
        
        # Attack knob
        self.pump_attack_knob = RotaryKnob(section, size=32,
                                           min_val=0.1, max_val=100, default=1,
                                           label="attack",
                                           logarithmic=True,
                                           command=self._on_pump_attack)
        self.pump_attack_knob.pack(pady=2)
        
        # Release knob
        self.pump_release_knob = RotaryKnob(section, size=32,
                                            min_val=1, max_val=1000, default=100,
                                            label="release",
                                            logarithmic=True,
                                            command=self._on_pump_release)
        self.pump_release_knob.pack(pady=2)
        
        # Curve knob
        self.pump_curve_knob = RotaryKnob(section, size=32,
                                          min_val=0, max_val=100, default=50,
                                          label="curve",
                                          command=self._on_pump_curve)
        self.pump_curve_knob.pack(pady=2)
        
        # Destination combobox
        self.pump_dest_var = tk.StringVar(value='Off')
        self.pump_dest_combo = ttk.Combobox(section, textvariable=self.pump_dest_var,
                                            values=self._mod_target_options,
                                            state='readonly', width=14,
                                            font=('Segoe UI', 7))
        self.pump_dest_combo.pack(padx=2, pady=(2, 2))
        self.pump_dest_combo.bind('<<ComboboxSelected>>', self._on_pump_dest)
    
    # ---- LFO callbacks ----
    # Wave and sync list positions are the engine enum positions, which the
    # core's enum addresses also take.

    def _on_lfo_enable(self, lfo_id, value):
        if not self.updating_ui:
            self._set_sound(f'{lfo_id}.on', bool(value))
    
    def _on_lfo_waveform(self, lfo_id, wave_var):
        if not self.updating_ui:
            try:
                idx = self._lfo_wave_options.index(wave_var.get())
            except ValueError:
                idx = 0
            self._set_sound(f'{lfo_id}.wave', idx)
    
    def _on_lfo_rate(self, lfo_id, value):
        if not self.updating_ui:
            self._set_sound(f'{lfo_id}.rate', float(value))
    
    def _on_lfo_depth(self, lfo_id, value):
        if not self.updating_ui:
            self._set_sound(f'{lfo_id}.depth', float(value))
    
    def _on_lfo_sync(self, lfo_id, sync_var):
        if not self.updating_ui:
            try:
                idx = self._lfo_sync_options.index(sync_var.get())
            except ValueError:
                idx = 0
            self._set_sound(f'{lfo_id}.sync', idx)
    
    def _on_lfo_retrigger(self, lfo_id, value):
        if not self.updating_ui:
            self._set_sound(f'{lfo_id}.retrig', bool(value))
    
    def _on_lfo_polarity(self, lfo_id, value):
        if not self.updating_ui:
            self._set_sound(f'{lfo_id}.unipolar', bool(value))
    
    def _on_lfo_dest(self, lfo_id, dest_var):
        if not self.updating_ui:
            self._set_sound(f'{lfo_id}.target', self._mod_target_name(dest_var.get()))
    
    def _mod_target_name(self, option):
        """Address value of a destination list option ('none' if unknown)."""
        try:
            return self._mod_target_values[self._mod_target_options.index(option)].value
        except ValueError:
            return ModTarget.NONE.value
    
    def _mod_target_option(self, name):
        """Destination list option of an address value."""
        for option, target in zip(self._mod_target_options, self._mod_target_values):
            if target.value == name:
                return option
        return 'Off'
    
    # ---- Pump callbacks ----
    
    def _on_pump_enable(self, value):
        if not self.updating_ui:
            self._set_sound('pump.on', bool(value))
    
    def _on_pump_amount(self, value):
        if not self.updating_ui:
            self._set_sound('pump.amount', value / 100.0)
    
    def _on_pump_attack(self, value):
        if not self.updating_ui:
            self._set_sound('pump.attack', float(value))
    
    def _on_pump_release(self, value):
        if not self.updating_ui:
            self._set_sound('pump.release', float(value))
    
    def _on_pump_curve(self, value):
        if not self.updating_ui:
            self._set_sound('pump.curve', value / 100.0)
    
    def _on_pump_dest(self, event=None):
        if not self.updating_ui:
            self._set_sound('pump.target', self._mod_target_name(self.pump_dest_var.get()))
    
    def _build_fx_section(self, parent):
        """Build the effects section (reverb, delay, vintage)"""
        section = tk.LabelFrame(parent, text="fx",
                               font=('Segoe UI', 7),
                               fg=self.COLORS['text_dim'],
                               bg=self.COLORS['bg_medium'],
                               labelanchor='n')
        section.pack(side='left', padx=2, pady=2, fill='both', expand=True)
        
        # Vintage (analog simulation)
        self.vintage_knob = RotaryKnob(section, size=30,
                                       min_val=0, max_val=100, default=0,
                                       label="vintage",
                                       command=self._on_vintage_change)
        self.vintage_knob.pack(pady=1)
        
        # Reverb controls
        reverb_row = tk.Frame(section, bg=self.COLORS['bg_medium'])
        reverb_row.pack(pady=1)
        
        self.reverb_decay_knob = RotaryKnob(reverb_row, size=30,
                                            min_val=0, max_val=100, default=0,
                                            label="rvb time",
                                            command=self._on_reverb_decay_change)
        self.reverb_decay_knob.pack(side='left', padx=1)
        
        self.reverb_mix_knob = RotaryKnob(reverb_row, size=30,
                                          min_val=0, max_val=100, default=0,
                                          label="rvb mix",
                                          command=self._on_reverb_mix_change)
        self.reverb_mix_knob.pack(side='left', padx=1)
        
        self.reverb_width_knob = RotaryKnob(reverb_row, size=30,
                                            min_val=0, max_val=200, default=100,
                                            label="rvb wide",
                                            command=self._on_reverb_width_change)
        self.reverb_width_knob.pack(side='left', padx=1)
        
        # Delay time selector (musical divisions)
        delay_row = tk.Frame(section, bg=self.COLORS['bg_medium'])
        delay_row.pack(pady=1)
        
        self.delay_time_options = ['1/4', '1/8', '1/16', '1/8T', '1/4.']
        # Address values (DelayTime names) of the five options
        self.delay_time_names = ['quarter', 'eighth', 'sixteenth', 'eighth_t', 'quarter_d']
        self.delay_time_selector = ModeSelector(delay_row,
                                                options=self.delay_time_options,
                                                width=100,
                                                command=self._on_delay_time_change)
        self.delay_time_selector.pack(side='left', padx=1)
        
        delay_knobs_row = tk.Frame(section, bg=self.COLORS['bg_medium'])
        delay_knobs_row.pack(pady=1)
        
        self.delay_feedback_knob = RotaryKnob(delay_knobs_row, size=30,
                                              min_val=0, max_val=95, default=30,
                                              label="dly fdbk",
                                              command=self._on_delay_feedback_change)
        self.delay_feedback_knob.pack(side='left', padx=1)
        
        self.delay_mix_knob = RotaryKnob(delay_knobs_row, size=30,
                                         min_val=0, max_val=100, default=0,
                                         label="dly mix",
                                         command=self._on_delay_mix_change)
        self.delay_mix_knob.pack(side='left', padx=1)
        
        self.delay_pingpong_btn = ToggleButton(delay_knobs_row, text="P.P",
                                               width=28, height=16,
                                               command=self._on_delay_pingpong_toggle)
        self.delay_pingpong_btn.pack(side='left', padx=1)
    
    def _build_global_section(self, parent):
        """Build the global controls section (bottom bar)
        
        Bottom bar layout:
        - Left: Stop/Play buttons with icons
        - Center-left: Step rate selector (1/8, 1/8T, 1/16, 1/16T, 1/32)
        - Center: Swing slider (0% to 100%)
        - Center-right: Fill rate (2x to 8x)
        - Right: Master volume knob
        """
        global_frame = tk.Frame(parent, bg=self.COLORS['bg_medium'], height=50)
        global_frame.pack(fill='x', pady=(5, 0))
        global_frame.pack_propagate(False)
        
        # Left: Transport controls (Stop/Play) - circular buttons
        transport_frame = tk.Frame(global_frame, bg=self.COLORS['bg_medium'])
        transport_frame.pack(side='left', padx=10, pady=8)
        
        from gui.widgets import CircularButton
        
        self.stop_btn = CircularButton(transport_frame, text="■", size=35,
                                       command=self._on_pattern_stop,
                                       bg_color='#4a4a5a', fg_color='#ccccee')
        self.stop_btn.pack(side='left', padx=2)
        
        self.play_btn = CircularButton(transport_frame, text="▶", size=35,
                                       command=self._on_pattern_play,
                                       bg_color='#446644', fg_color='#88ff88')
        self.play_btn.pack(side='left', padx=2)
        
        # BPM control
        bpm_frame = tk.Frame(global_frame, bg=self.COLORS['bg_medium'])
        bpm_frame.pack(side='left', padx=10, pady=8)
        
        tk.Label(bpm_frame, text="BPM", 
                font=('Segoe UI', 7), fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack(side='left', padx=(0, 3))
        
        self.bpm_var = tk.StringVar(value=str(self.core.get('global.tempo')))
        self.bpm_entry = tk.Entry(bpm_frame, textvariable=self.bpm_var, 
                                  width=4, font=('Segoe UI', 9),
                                  bg=self.COLORS['bg_dark'], fg=self.COLORS['text'],
                                  insertbackground=self.COLORS['text'],
                                  justify='center')
        self.bpm_entry.pack(side='left')
        self.bpm_entry.bind('<Return>', self._on_bpm_change)
        self.bpm_entry.bind('<FocusOut>', self._on_bpm_change)
        
        # Step rate buttons (1/8 to 1/32 selector)
        rate_frame = tk.Frame(global_frame, bg=self.COLORS['bg_medium'])
        rate_frame.pack(side='left', padx=8, pady=4)
        
        self.step_rate_buttons = []
        for rate in ['1/8', '1/8T', '1/16', '1/16T', '1/32']:
            is_selected = (rate == self.core.get('global.step_rate'))
            btn = tk.Button(rate_frame, text=rate, width=4, height=1,
                           font=('Segoe UI', 7),
                           bg=self.COLORS['highlight'] if is_selected else self.COLORS['bg_light'],
                           fg=self.COLORS['text'],
                           command=lambda r=rate: self._on_step_rate_button(r))
            btn.pack(side='left', padx=1)
            self.step_rate_buttons.append((rate, btn))
        
        # Swing slider (center)
        swing_frame = tk.Frame(global_frame, bg=self.COLORS['bg_medium'])
        swing_frame.pack(side='left', padx=8, pady=4)
        
        tk.Label(swing_frame, text="0%", 
                font=('Segoe UI', 7), fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack(side='left')
        
        self.global_swing_slider = tk.Scale(swing_frame, from_=0, to=100, 
                                           orient='horizontal', length=80,
                                           showvalue=False,
                                           bg=self.COLORS['bg_medium'], 
                                           fg=self.COLORS['text'],
                                           highlightthickness=0,
                                           troughcolor=self.COLORS['bg_dark'],
                                           command=self._on_global_swing_change)
        self.global_swing_slider.pack(side='left')
        
        tk.Label(swing_frame, text="swing", 
                font=('Segoe UI', 7), fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack(side='left', padx=(3, 0))
        
        tk.Label(swing_frame, text="100%", 
                font=('Segoe UI', 7), fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack(side='left', padx=(5, 0))
        
        # Fill rate (center-right)
        fill_frame = tk.Frame(global_frame, bg=self.COLORS['bg_medium'])
        fill_frame.pack(side='left', padx=8, pady=4)
        
        tk.Label(fill_frame, text="fill rate", 
                font=('Segoe UI', 7), fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack(side='left', padx=(0, 5))
        
        self.fill_rate_buttons = []
        for rate in ['2x', '3x', '4x', '5x', '6x', '7x', '8x']:
            rate_val = int(rate[0])
            is_selected = (rate_val == self.core.get('global.fill_rate'))
            btn = tk.Button(fill_frame, text=rate, width=2, height=1,
                           font=('Segoe UI', 7),
                           bg=self.COLORS['highlight'] if is_selected else self.COLORS['bg_light'],
                           fg=self.COLORS['text'],
                           command=lambda r=rate_val: self._on_fill_rate_button(r))
            btn.pack(side='left', padx=1)
            self.fill_rate_buttons.append((rate_val, btn))
        
        # Right: Keyboard hint and trigger info
        info_frame = tk.Frame(global_frame, bg=self.COLORS['bg_medium'])
        info_frame.pack(side='right', padx=5, pady=4)
        
        tk.Label(info_frame, 
                text="Keys 1-8: Trigger channels",
                font=('Segoe UI', 7),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack()
        
        # Bind keyboard
        self.root.bind('<Key>', self._on_key_press)
    
    def _on_step_rate_button(self, rate):
        """Handle step rate button click (the highlight follows poll)"""
        self.core.set('global.step_rate', rate)
    
    def _show_step_rate(self, rate):
        for r, btn in self.step_rate_buttons:
            btn.config(bg=self.COLORS['highlight'] if r == rate else self.COLORS['bg_light'])
    
    def _on_fill_rate_button(self, rate):
        """Handle fill rate button click (the highlight follows poll)"""
        self.core.set('global.fill_rate', rate)
    
    def _show_fill_rate(self, rate):
        for r, btn in self.fill_rate_buttons:
            btn.config(bg=self.COLORS['highlight'] if r == round(rate) else self.COLORS['bg_light'])
    
    def _on_global_swing_change(self, value):
        """Handle global swing slider change (0-100 %)"""
        if not self.updating_ui:
            self.core.set('global.swing', int(float(value)) / 100.0)
    
    def _on_bpm_change(self, event=None):
        """Handle BPM entry change (the core clamps to 1-300)"""
        try:
            bpm = int(self.bpm_var.get())
        except ValueError:
            # Restore current BPM if invalid input
            self.bpm_var.set(str(self.core.get('global.tempo')))
            return
        self.core.set('global.tempo', bpm)
        self.bpm_var.set(str(max(1, min(300, bpm))))  # Show the clamped value at once
    
    # ============== Event Handlers ==============
    
    def _on_program_select(self, event=None):
        """Handle program selection (1-16 slots)
        
        Stores the current synth state into the previously selected slot,
        then recalls the newly selected slot. If the new slot is empty,
        the current state is copied into it so every first visit
        captures a snapshot.
        """
        try:
            program_num = int(self.program_var.get())
            new_slot = program_num - 1  # 0-indexed internally
            old_slot = self.synth.get_current_program()
            
            if new_slot == old_slot:
                return  # No change
            
            # Save current state into the old slot before switching
            self.synth.store_program(old_slot)
            
            # Try to recall the new slot
            if self.synth.recall_program(new_slot):
                # Slot had data – UI needs to reflect the loaded state
                self._update_ui_from_channel()
            else:
                # Empty slot – capture current state into it
                self.synth.store_program(new_slot)
                self.synth._current_program = new_slot
        except ValueError:
            pass
    
    def _on_morph_change(self, value):
        """Handle sound morph slider change
        
        Sound morph interpolates all drum patch parameters
        between two end-points using this single slider.
        During learn mode the slider stores the position but does not
        affect the synth – the synth stays pinned to the learned endpoint.
        """
        if self.updating_ui:
            return
        # The knobs follow once poll reports the new position
        self.core.set('morph.position', float(value) / 100.0)
    
    def _on_morph_learn_a(self):
        """Toggle learn mode for morph endpoint A."""
        current = self.morph_manager.get_learn_mode()
        if current == 'a':
            # Stop learning A - capture current state
            self.morph_manager.stop_learn()
            # Re-apply the actual slider position now that learn is off
            self.morph_manager.apply_effective_position()
            self._update_ui_from_channel()
            self._update_morph_ui()
        else:
            # Start learning A (stop B if active)
            if current == 'b':
                self.morph_manager.stop_learn()
            self.morph_manager.start_learn_a()
            # Apply effective position (0.0) so user hears endpoint A
            self.morph_manager.apply_effective_position()
            self._update_ui_from_channel()
            self._update_morph_ui()
    
    def _on_morph_learn_b(self):
        """Toggle learn mode for morph endpoint B."""
        current = self.morph_manager.get_learn_mode()
        if current == 'b':
            # Stop learning B - capture current state
            self.morph_manager.stop_learn()
            # Re-apply the actual slider position now that learn is off
            self.morph_manager.apply_effective_position()
            self._update_ui_from_channel()
            self._update_morph_ui()
        else:
            # Start learning B (stop A if active)
            if current == 'a':
                self.morph_manager.stop_learn()
            self.morph_manager.start_learn_b()
            # Apply effective position (1.0) so user hears endpoint B
            self.morph_manager.apply_effective_position()
            self._update_ui_from_channel()
            self._update_morph_ui()
    
    def _update_morph_ui(self):
        """Update morph learn button colors and slider enabled state."""
        mode = self.morph_manager.get_learn_mode()
        has_morph = self.morph_manager.has_different_endpoints()
        
        # A button: green when learning A, dark gray otherwise
        if mode == 'a':
            self.morph_learn_a_btn.config(
                bg='#22cc55', fg='#000000',
                activebackground='#33dd66')
        else:
            self.morph_learn_a_btn.config(
                bg='#444455', fg=self.COLORS['text_dim'],
                activebackground='#555566')
        
        # B button: green when learning B, dark gray otherwise
        if mode == 'b':
            self.morph_learn_b_btn.config(
                bg='#22cc55', fg='#000000',
                activebackground='#33dd66')
        else:
            self.morph_learn_b_btn.config(
                bg='#444455', fg=self.COLORS['text_dim'],
                activebackground='#555566')
        
        # Slider: enabled only when endpoints differ or learn is active
        if has_morph or mode is not None:
            self.morph_slider.config(
                state='normal',
                fg=self.COLORS['text'],
                troughcolor=self.COLORS['bg_dark'])
        else:
            self.morph_slider.config(
                state='disabled',
                fg=self.COLORS['text_dim'],
                troughcolor=self.COLORS['bg_medium'])
    
    def _get_full_state_snapshot(self):
        """Capture a deep copy of all synth + pattern + morph state for undo/redo"""
        return self.core.legacy_snapshot()

    def _restore_state_snapshot(self, snapshot, revert=None):
        """Restore synth + pattern + morph state from a snapshot.

        The core applies the snapshot at the next audio block start; the
        widgets are refreshed by the UI tick once the core reports it, and
        revert() puts the undo/redo stacks back if the core reports a failure.
        (Temporary core verb until the undo journal moves into the core.)
        """
        has_morph = len(snapshot) == 3 and bool(snapshot[2])
        action_id = self.core.act('legacy.restore_snapshot', snapshot=snapshot)
        self._restore_pending = True
        self._update_undo_redo_buttons()

        def refresh(event):
            self._restore_pending = False
            if event['status'] != 'done':
                print(f"Undo/redo failed: {event.get('error')}", flush=True)
                if revert is not None:
                    revert()
                self._update_undo_redo_buttons()
                return
            if has_morph:
                # Update morph slider position
                self.morph_slider.set(int(self.morph_manager.position * 100))
            self._update_ui_from_channel()
            self._update_pattern_editors()
            self._update_matrix_editor()
            self._update_undo_redo_buttons()
            self._update_morph_ui()

        self._when_action_done(action_id, refresh)

    def _push_undo_state(self):
        """Push current state onto the undo stack (call BEFORE making a change)"""
        snapshot = self._get_full_state_snapshot()
        self._undo_stack.append(snapshot)
        if len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)
        # Any new action clears the redo stack
        self._redo_stack.clear()
        self._update_undo_redo_buttons()

    def _push_undo_state_deferred(self, phase=None):
        """Push undo state — used as command_end callback on knobs/sliders.
        Called with 'start' when drag begins and 'end' when drag ends."""
        if phase == 'start':
            # Capture state before changes begin
            self._pre_drag_snapshot = self._get_full_state_snapshot()
        elif phase == 'end':
            # Commit the pre-drag snapshot to undo stack
            if hasattr(self, '_pre_drag_snapshot') and self._pre_drag_snapshot is not None:
                self._undo_stack.append(self._pre_drag_snapshot)
                if len(self._undo_stack) > self._max_undo:
                    self._undo_stack.pop(0)
                self._redo_stack.clear()
                self._pre_drag_snapshot = None
                self._update_undo_redo_buttons()

    def _push_undo_state_now(self):
        """Actually push the undo state (for discrete actions like pattern edits)"""
        self._undo_pending = False
        self._push_undo_state()

    def _update_undo_redo_buttons(self):
        """Update undo/redo button enabled state"""
        if hasattr(self, 'undo_btn'):
            state = 'normal' if self._undo_stack else 'disabled'
            self.undo_btn.config(state=state)
        if hasattr(self, 'redo_btn'):
            state = 'normal' if self._redo_stack else 'disabled'
            self.redo_btn.config(state=state)

    def _on_undo(self):
        """Handle undo button click"""
        if not self._undo_stack or self._restore_pending:
            return
        # Save current state to redo stack
        self._redo_stack.append(self._get_full_state_snapshot())
        # Pop and restore previous state
        snapshot = self._undo_stack.pop()

        def revert():
            if self._redo_stack:
                self._redo_stack.pop()
            self._undo_stack.append(snapshot)
        self._restore_state_snapshot(snapshot, revert)

    def _on_redo(self):
        """Handle redo button click"""
        if not self._redo_stack or self._restore_pending:
            return
        # Save current state to undo stack
        self._undo_stack.append(self._get_full_state_snapshot())
        # Pop and restore next state
        snapshot = self._redo_stack.pop()

        def revert():
            if self._undo_stack:
                self._undo_stack.pop()
            self._redo_stack.append(snapshot)
        self._restore_state_snapshot(snapshot, revert)
    
    def _on_preset_prev(self):
        """Navigate to previous preset in the list"""
        current = self.preset_combo.current()
        values = self.preset_combo['values']
        if values and current > 0:
            self.preset_combo.current(current - 1)
            self._on_preset_combo_select()
    
    def _on_preset_next(self):
        """Navigate to next preset in the list"""
        current = self.preset_combo.current()
        values = self.preset_combo['values']
        if values and current < len(values) - 1:
            self.preset_combo.current(current + 1)
            self._on_preset_combo_select()
    
    def _on_preset_menu(self):
        """Show preset menu"""
        menu = tk.Menu(self.root, tearoff=0)
        
        menu.add_command(label="Open Preset...", command=self._load_preset)
        menu.add_command(label="Save Preset As...", command=self._save_preset)
        menu.add_separator()
        menu.add_command(label="Load Drum Patch (.mtdrum)...", command=self._load_drum_patch)
        menu.add_command(label="Save Drum Patch (.mtdrum)...", command=self._save_drum_patch)
        menu.add_command(label="Export Drum to WAV...", command=self._export_current_drum)
        menu.add_command(label="Export All Drums to WAV...", command=self._export_all_wavs)
        menu.add_separator()
        menu.add_command(label="Cut Preset", command=self._cut_preset)
        menu.add_command(label="Copy Preset", command=self._copy_preset)
        menu.add_command(label="Paste Preset", command=self._paste_preset,
                         state='normal' if getattr(self, '_preset_clipboard', None) else 'disabled')
        menu.add_separator()
        menu.add_command(label="Initialize Preset", command=self._init_preset)
        menu.add_command(label="Randomize All", command=self._randomize_all)
        menu.add_separator()
        menu.add_command(label="Select Preset Folder...", command=self._select_preset_folder)
        menu.add_command(label="Refresh Preset List", command=self._refresh_preset_list)
        menu.add_separator()
        menu.add_command(label="Transfer to PO-32...", command=self._show_po32_transfer)
        menu.add_command(label="Import from PO-32...", command=self._show_po32_import)
        menu.add_separator()
        menu.add_command(label="AI Drum Generator...", command=self._show_drum_generator)
        menu.add_separator()
        menu.add_command(label="Audio Settings...", command=self._show_audio_preferences)
        menu.add_command(label="MIDI Settings...", command=self._show_midi_preferences)
        menu.add_command(label="Synthesis Settings...", command=self._show_synthesis_preferences)
        menu.add_command(label="AI Settings...", command=self._show_ai_preferences)
        
        try:
            menu.tk_popup(self.root.winfo_pointerx(), self.root.winfo_pointery())
        finally:
            menu.grab_release()
    
    def _cut_preset(self):
        """Cut preset to clipboard"""
        self._copy_preset()
        self._init_preset()
    
    def _copy_preset(self):
        """Copy current preset to clipboard"""
        # Store preset data in memory for paste
        self._preset_clipboard = self.preset_manager.export_preset_to_dict(self.pattern_manager)
        self._preset_clipboard['morph'] = copy.deepcopy(self.morph_manager.to_dict())
    
    def _paste_preset(self):
        """Paste preset from clipboard"""
        if hasattr(self, '_preset_clipboard') and self._preset_clipboard:
            self._push_undo_state()
            self.preset_manager.import_preset_from_dict(self._preset_clipboard, self.pattern_manager)
            self.morph_manager.from_dict(copy.deepcopy(self._preset_clipboard['morph']))
            self.morph_slider.set(int(self.morph_manager.position * 100))
            self._after_preset_replaced()
    
    def _init_preset(self):
        """Initialize/reset preset to defaults"""
        self._push_undo_state()
        for channel in self.synth.channels:
            channel.reset_to_defaults()
        self.pattern_manager.reset_all_patterns()
        self.morph_manager._init_endpoints()
        self._after_preset_replaced()
    
    def _randomize_all(self):
        """Randomize all drum patches and patterns"""
        self._push_undo_state()
        for channel in self.synth.channels:
            channel.randomize()
        self.core.act('pattern.randomize', pattern=self._pattern)
        self.morph_manager._init_endpoints()
        self._after_preset_replaced()

    def _after_preset_replaced(self):
        """Refresh every view after the whole preset changed in place"""
        self._update_ui_from_channel()
        self._update_pattern_ui()
        self._update_matrix_editor()
        self._update_morph_ui()
    
    def _on_channel_select(self, channel_idx, event=None):
        """Handle channel selection
        
        If clicking on an already-selected channel, trigger the drum to preview it.
        Hold Ctrl for accented trigger (velocity 127), otherwise normal (velocity 64).
        Otherwise the core selects the channel; the editors follow from poll.
        """
        # Check if clicking on already-selected channel -> trigger preview
        if channel_idx == self.selected_channel:
            # Check if Ctrl is held for accented trigger
            if event and (event.state & 0x4):  # Control key mask
                velocity = 127  # Accented
            else:
                velocity = 64   # Normal
            self._trigger_channel(channel_idx, velocity)
            return
        self.core.set('global.channel', channel_idx + 1)
    
    def _show_selected_channel(self, channel_idx):
        """Show the channel the core reports as selected (from poll)."""
        for i, btn in enumerate(self.channel_buttons):
            btn.set_selected(i == channel_idx)
        self.selected_channel = channel_idx
        
        # Switch pattern editor to show the selected channel
        if hasattr(self, 'pattern_editors') and hasattr(self, 'current_pattern_editor_index'):
            # Hide current editor
            self.pattern_editors[self.current_pattern_editor_index].pack_forget()
            # Show new editor
            self.pattern_editors[channel_idx].pack(side='left', fill='both', expand=True, padx=2)
            self.current_pattern_editor_index = channel_idx
            # Update channel label
            if hasattr(self, 'pattern_channel_label'):
                self.pattern_channel_label.config(text=f"ch{channel_idx + 1}")
        
        # Update UI to reflect selected channel's parameters
        self._update_ui_from_channel()
    
    def _on_mute_toggle(self, channel_idx, enabled):
        """Handle mute toggle (the channel LED follows poll)"""
        if not self.updating_ui:
            self.core.set(f'ch{channel_idx + 1}.mute', bool(enabled))
    
    def _show_mute(self, channel_idx, muted):
        self.channel_buttons[channel_idx].set_muted(muted)
        self.mute_buttons[channel_idx].set_value(muted)
    
    def _on_master_volume_change(self, value):
        """Handle master volume change"""
        if not self.updating_ui:
            self.core.set('global.master', value)
    
    def _on_edit_all_toggle(self, enabled):
        """Edit all: the core applies sound changes to every unmuted channel"""
        self.core.set('global.edit_all', bool(enabled))
    
    def _set_sound(self, suffix, value):
        """Write a sound parameter of the selected channel through the core
        (with Edit all on, the core also writes the unmuted channels)."""
        self.core.set(f'ch{self.selected_channel + 1}.{suffix}', value)
    
    # Sound parameter handlers: widget units -> engine units
    
    def _on_mix_change(self, value):
        """Handle osc/noise mix change (value comes from tk.Scale as string)"""
        if not self.updating_ui:
            self._set_sound('mix.osc_noise', float(value) / 100.0)
    
    def _on_eq_freq_change(self, value):
        if not self.updating_ui:
            self._set_sound('eq.freq', value)
    
    def _on_eq_gain_change(self, value):
        if not self.updating_ui:
            self._set_sound('eq.gain', value)
    
    def _on_distort_change(self, value):
        if not self.updating_ui:
            self._set_sound('mix.distortion', value / 100.0)
    
    def _on_vintage_change(self, value):
        """Vintage analog simulation: pitch drift, noise floor, saturation, HF roll-off"""
        if not self.updating_ui:
            self._set_sound('fx.vintage', value / 100.0)
    
    def _on_reverb_decay_change(self, value):
        """Reverb time: 0-100 % maps to about 0.1 s to 4 s RT60"""
        if not self.updating_ui:
            self._set_sound('fx.reverb_decay', value / 100.0)
    
    def _on_reverb_mix_change(self, value):
        if not self.updating_ui:
            self._set_sound('fx.reverb_mix', value / 100.0)
    
    def _on_reverb_width_change(self, value):
        """Reverb width: 0 % mono, 100 % stereo, 200 % extra wide"""
        if not self.updating_ui:
            self._set_sound('fx.reverb_width', value / 100.0)
    
    def _on_delay_time_change(self, index):
        """Tempo-synced delay time (1/4, 1/8, 1/16, 1/8T, 1/4.)"""
        if not self.updating_ui:
            self._push_undo_state()
            self._set_sound('fx.delay_time', self.delay_time_names[index])
    
    def _on_delay_feedback_change(self, value):
        if not self.updating_ui:
            self._set_sound('fx.delay_feedback', value / 100.0)
    
    def _on_delay_mix_change(self, value):
        if not self.updating_ui:
            self._set_sound('fx.delay_mix', value / 100.0)
    
    def _on_delay_pingpong_toggle(self, enabled):
        """Ping-pong: echoes alternate between left and right"""
        if not self.updating_ui:
            self._push_undo_state()
            self._set_sound('fx.delay_pingpong', bool(enabled))
    
    def _on_level_change(self, value):
        if not self.updating_ui:
            self._set_sound('mix.level', value)
    
    def _on_pan_change(self, value):
        if not self.updating_ui:
            self._set_sound('mix.pan', value)
    
    def _on_choke_toggle(self, enabled):
        if not self.updating_ui:
            self._push_undo_state()
            self._set_sound('mix.choke', bool(enabled))
    
    def _on_output_change(self, value):
        if not self.updating_ui:
            self._push_undo_state()
            self._set_sound('mix.output', 'A' if value == 0 else 'B')
    
    def _on_waveform_change(self, value):
        if not self.updating_ui:
            self._push_undo_state()
            self._set_sound('osc.wave', value)  # selector position = enum position
    
    def _on_pitch_change(self, value):
        """Pitch (tune) offset in semitones"""
        if not self.updating_ui:
            self._set_sound('osc.pitch', value)
    
    def _on_osc_freq_change(self, value):
        if not self.updating_ui:
            self._set_sound('osc.freq', value)
    
    def _on_pitch_mod_mode_change(self, value):
        if not self.updating_ui:
            self._push_undo_state()
            self._set_sound('osc.mod_mode', value)
    
    def _on_pitch_amount_change(self, value):
        if not self.updating_ui:
            self._set_sound('osc.mod_amount', value)
    
    def _on_pitch_rate_change(self, value):
        if not self.updating_ui:
            self._set_sound('osc.mod_rate', value)
    
    def _on_osc_attack_change(self, value):
        if not self.updating_ui:
            self._set_sound('osc.attack', value)
    
    def _on_osc_decay_change(self, value):
        if not self.updating_ui:
            self._set_sound('osc.decay', value)
    
    def _on_noise_filter_mode_change(self, value):
        if not self.updating_ui:
            self._push_undo_state()
            self._set_sound('noise.filter', value)
    
    def _on_noise_freq_change(self, value):
        if not self.updating_ui:
            self._set_sound('noise.freq', value)
    
    def _on_noise_q_change(self, value):
        if not self.updating_ui:
            self._set_sound('noise.q', value)
    
    def _on_stereo_toggle(self, enabled):
        if not self.updating_ui:
            self._push_undo_state()
            self._set_sound('noise.stereo', bool(enabled))
    
    def _on_noise_env_mode_change(self, value):
        if not self.updating_ui:
            self._push_undo_state()
            self._set_sound('noise.env', value)
    
    def _on_noise_attack_change(self, value):
        if not self.updating_ui:
            self._set_sound('noise.attack', value)
    
    def _on_noise_decay_change(self, value):
        if not self.updating_ui:
            self._set_sound('noise.decay', value)
    
    def _on_osc_vel_change(self, value):
        if not self.updating_ui:
            self._set_sound('vel.osc', value / 100.0)
    
    def _on_noise_vel_change(self, value):
        if not self.updating_ui:
            self._set_sound('vel.noise', value / 100.0)
    
    def _on_mod_vel_change(self, value):
        if not self.updating_ui:
            self._set_sound('vel.mod', value / 100.0)
    
    # ============ Pattern Callbacks ============
    
    def _on_pattern_select(self, pattern_index):
        """Pattern button: the core selects the pattern (and queues it while
        playing); the buttons and editors follow from poll."""
        self.core.act('pattern.select', pattern=pattern_index)
    
    def _step_address(self, channel_id, step, lane_type):
        """Address of a step of the pattern the editors show (lane types are
        the address fields: trig, acc, fill, prob, sub)."""
        name = PatternManager.PATTERN_NAMES[self._pattern]
        return f'pattern.{name}.ch{channel_id + 1}.step{step + 1}.{lane_type}'
    
    def _on_pattern_edit(self, channel_id, step, lane_type, value):
        """Handle pattern editor edits (the core applies them at block start)"""
        self._push_undo_state()
        self.core.set(self._step_address(channel_id, step, lane_type), value)
    
    def _on_toggle_prob_mode(self):
        """Toggle probability editing mode for pattern editor"""
        self.probability_mode_active = not self.probability_mode_active
        
        # Update button appearance
        if self.probability_mode_active:
            self.prob_mode_btn.config(bg='#44aa66', fg='#ffffff')
        else:
            self.prob_mode_btn.config(bg=self.COLORS['bg_light'], fg=self.COLORS['text_dim'])
        
        # Update all pattern editors
        for editor in self.pattern_editors:
            editor.set_probability_mode(self.probability_mode_active)
    
    def _on_pattern_edit_all(self, step, lane_type, value, muted_channels):
        """Shift+click: the same step on every channel not in muted_channels
        (the editor passes none, so muted channels are included)"""
        self._push_undo_state()
        for ch_idx in range(8):
            if ch_idx not in muted_channels:
                self.core.set(self._step_address(ch_idx, step, lane_type), value)
    
    def _on_pattern_length_change(self, new_length):
        """Handle pattern length change (the editors follow from poll)"""
        name = PatternManager.PATTERN_NAMES[self._pattern]
        self.core.set(f'pattern.{name}.length', new_length)
    
    def _on_pattern_play(self):
        """Play the selected pattern from its first step (the play and stop
        buttons, playhead and pattern buttons follow from poll)"""
        self.core.act('transport.play')
    
    def _on_pattern_stop(self):
        """Stop playback; the playhead goes back to step 1 (from poll)"""
        self.core.act('transport.stop')
    
    def _on_pattern_menu(self, idx=None):
        """Show the pattern menu for pattern idx (default: the shown pattern)"""
        menu = tk.Menu(self.root, tearoff=0)
        if idx is None:
            idx = self._pattern
        
        menu.add_command(label="Cut Pattern", 
                        command=lambda: self._pattern_menu_action('cut_pattern', idx))
        menu.add_command(label="Copy Pattern",
                        command=lambda: self._pattern_menu_action('copy_pattern', idx))
        menu.add_command(label="Paste Pattern",
                        command=lambda: self._pattern_menu_action('paste_pattern', idx))
        menu.add_separator()
        menu.add_command(label="Exchange Pattern",
                        command=lambda: self._pattern_menu_action('exchange_pattern', idx))
        menu.add_separator()
        menu.add_command(label="Shift Left",
                        command=lambda: self._pattern_menu_action('shift_left', idx))
        menu.add_command(label="Shift Right",
                        command=lambda: self._pattern_menu_action('shift_right', idx))
        menu.add_separator()
        menu.add_command(label="Reverse",
                        command=lambda: self._pattern_menu_action('reverse', idx))
        menu.add_command(label="Randomize",
                        command=lambda: self._pattern_menu_action('randomize', idx))
        menu.add_command(label="Alter Pattern",
                        command=lambda: self._pattern_menu_action('alter', idx))
        menu.add_command(label="Randomize Accents/Fills",
                        command=lambda: self._pattern_menu_action('rand_accents', idx))
        menu.add_separator()
        menu.add_command(label="Randomize Pattern (AI)",
                        command=lambda: self._pattern_menu_action('ai_randomize_pattern', idx))
        menu.add_command(label="Randomize Channel (AI)",
                        command=lambda: self._pattern_menu_action('ai_randomize_channel', idx))
        menu.add_separator()
        menu.add_command(label="Export Pattern to MIDI File...",
                        command=lambda: self._pattern_menu_action('export_midi', idx))
        menu.add_command(label="Export Pattern to Audio File...",
                        command=lambda: self._pattern_menu_action('export_audio', idx))
        
        # Show menu at button location
        try:
            menu.tk_popup(self.root.winfo_pointerx(), self.root.winfo_pointery())
        finally:
            menu.grab_release()
    
    def _on_pattern_right_click(self, pattern_idx, event):
        """Handle right-click on pattern button - show pattern menu for that pattern"""
        # First select the pattern
        self._on_pattern_select(pattern_idx)
        
        # Then show the menu (for that pattern: the selection lands later)
        self._on_pattern_menu(pattern_idx)
    
    # Pattern menu entries run as core verbs on the pattern they were opened for
    _PATTERN_MENU_VERBS = {
        'cut_pattern': 'pattern.cut', 'copy_pattern': 'pattern.copy',
        'paste_pattern': 'pattern.paste', 'exchange_pattern': 'pattern.exchange',
        'shift_left': 'pattern.shift_left', 'shift_right': 'pattern.shift_right',
        'reverse': 'pattern.reverse', 'randomize': 'pattern.randomize',
        'alter': 'pattern.alter', 'rand_accents': 'pattern.randomize_accents_fills',
    }
    
    def _pattern_menu_action(self, action, pattern_idx):
        """Handle pattern menu actions (the editors follow from poll)"""
        verb = self._PATTERN_MENU_VERBS.get(action)
        if verb is not None:
            self._act_or_warn(verb, "Pattern operation failed", pattern=pattern_idx)
            return
        try:
            if action == 'ai_randomize_pattern':
                self._ai_randomize_pattern(pattern_idx)
            elif action == 'ai_randomize_channel':
                self._ai_randomize_channel(pattern_idx, self.selected_channel)
            elif action == 'export_midi':
                self._export_pattern_to_midi(pattern_idx)
            elif action == 'export_audio':
                self._export_pattern_to_audio(pattern_idx)
        except Exception as e:
            messagebox.showerror("Error", f"Pattern operation failed: {e}")
    
    def _act_or_warn(self, verb, message, on_done=None, **args):
        """Run a core verb; show its error (from poll) in a message box."""
        def done(event):
            if event['status'] != 'done':
                messagebox.showerror("Error", f"{message}: {event.get('error')}")
            elif on_done is not None:
                on_done(event.get('result'))
        self._when_action_done(self.core.act(verb, **args), done)
    
    # ── AI pattern randomization ─────────────────────────────────────

    def _get_ai_pattern_generator(self) -> 'PatternGenerator | None':
        """Return a loaded PatternGenerator or None with user feedback."""
        if not hasattr(self, '_pattern_gen'):
            self._pattern_gen = PatternGenerator()
        gen = self._pattern_gen
        if gen.ensure_loaded(self.preferences_manager):
            return gen
        messagebox.showwarning(
            "No Pattern Model",
            "No AI pattern model is available.\n\n"
            "Set one via the preset menu → AI Settings,\n"
            "or place 'pattern_cvae_best.pt' in the\n"
            "drum_patterns/ folder.",
        )
        return None

    def _get_raw_patches_from_synth(self) -> list:
        """Build 8 raw patch dicts from the current live synth channels."""
        return [channel_to_raw_patch(ch) for ch in self.synth.channels[:8]]

    def _ai_randomize_pattern(self, pattern_idx: int):
        """Replace the selected pattern with an AI-generated one."""
        gen = self._get_ai_pattern_generator()
        if gen is None:
            return
        try:
            self._push_undo_state()
            raw_patches = self._get_raw_patches_from_synth()
            pm = self.pattern_manager
            temp = self.preferences_manager.get(
                'drum_generator_pattern_temperature', 0.7)
            patterns = gen.generate(
                raw_patches,
                tempo=pm.bpm,
                swing=pm.swing,
                fill_rate=pm.fill_rate,
                step_rate=pm.step_rate,
                n=1,
                temperature=temp,
            )
            pm.apply_single_pattern(pattern_idx, patterns[0])
            self._update_pattern_editors()
        except Exception as e:
            messagebox.showerror("Error",
                                 f"AI pattern generation failed: {e}")

    def _ai_randomize_channel(self, pattern_idx: int, channel_id: int):
        """Replace the selected channel with AI-generated data."""
        gen = self._get_ai_pattern_generator()
        if gen is None:
            return
        try:
            self._push_undo_state()
            raw_patches = self._get_raw_patches_from_synth()
            pm = self.pattern_manager
            temp = self.preferences_manager.get(
                'drum_generator_pattern_temperature', 0.7)
            patterns = gen.generate(
                raw_patches,
                tempo=pm.bpm,
                swing=pm.swing,
                fill_rate=pm.fill_rate,
                step_rate=pm.step_rate,
                n=1,
                temperature=temp,
            )
            pm.apply_single_channel(pattern_idx, channel_id, patterns[0])
            self._update_pattern_editors()
        except Exception as e:
            messagebox.showerror("Error",
                                 f"AI channel generation failed: {e}")

    def _export_pattern_to_midi(self, pattern_idx):
        """Export pattern to MIDI file"""
        if not MIDI_AVAILABLE:
            messagebox.showerror("Error", "MIDI export requires the 'mido' library. Install it with: pip install mido")
            return
        
        pattern = self.pattern_manager.get_pattern(pattern_idx)
        pattern_name = self.pattern_manager.PATTERN_NAMES[pattern_idx]
        
        # Ask for filename
        filename = filedialog.asksaveasfilename(
            title=f"Export Pattern {pattern_name} to MIDI",
            defaultextension=".mid",
            filetypes=[("MIDI files", "*.mid"), ("All files", "*.*")],
            initialfile=f"pythonic_pattern_{pattern_name}.mid"
        )
        
        if not filename:
            return
        
        try:
            pm = self.pattern_manager
            pattern_midi_file(pattern, pm.bpm, pm.step_rate, pm.swing).save(filename)
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export MIDI file:\\n{e}")
    
    def _export_pattern_to_audio(self, pattern_idx):
        """Export pattern to WAV file"""
        if not AUDIO_AVAILABLE:
            messagebox.showerror("Error", "Audio export requires the 'sounddevice' library.")
            return
        
        pattern = self.pattern_manager.get_pattern(pattern_idx)
        pattern_name = self.pattern_manager.PATTERN_NAMES[pattern_idx]
        
        # Ask for tail handling option
        tail_dialog = tk.Toplevel(self.root)
        tail_dialog.title("Export Audio Options")
        tail_dialog.geometry("300x150")
        tail_dialog.transient(self.root)
        tail_dialog.grab_set()
        
        tail_option = tk.StringVar(value="none")
        
        tk.Label(tail_dialog, text="Tail Handling:", font=('Segoe UI', 10)).pack(pady=10)
        tk.Radiobutton(tail_dialog, text="None (truncate)", variable=tail_option, value="none").pack(anchor='w', padx=20)
        tk.Radiobutton(tail_dialog, text="Append (add silence)", variable=tail_option, value="append").pack(anchor='w', padx=20)
        tk.Radiobutton(tail_dialog, text="Loop (repeat pattern)", variable=tail_option, value="loop").pack(anchor='w', padx=20)
        
        def on_ok():
            tail_dialog.destroy()
        
        tk.Button(tail_dialog, text="OK", command=on_ok).pack(pady=10)
        
        self.root.wait_window(tail_dialog)
        
        # Ask for filename
        filename = filedialog.asksaveasfilename(
            title=f"Export Pattern {pattern_name} to Audio",
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
            initialfile=f"pythonic_pattern_{pattern_name}.wav"
        )
        
        if not filename:
            return
        
        try:
            # Import wave for WAV file writing
            import wave
            
            # Calculate total samples needed
            # Use synth sample rate for offline rendering
            render_sr = self.synth_sample_rate
            pm = self.pattern_manager
            ticks = pattern.length * STEP_TICKS.get(pm.step_rate, 480)
            pattern_duration_samples = int(np.ceil(ticks / (pm.bpm * 32.0 / render_sr)))
            
            # Add tail handling
            if tail_option.get() == "append":
                # Add 2 seconds of tail for reverb/decay
                tail_samples = render_sr * 2
            elif tail_option.get() == "loop":
                # Add one more loop iteration
                tail_samples = pattern_duration_samples
            else:
                tail_samples = 0
            
            total_samples = pattern_duration_samples + tail_samples
            
            # Temporarily enable playback and render
            old_playing_state = pm.is_playing
            old_playing_idx = pm.playing_pattern_index
            
            pm.playing_pattern_index = pattern_idx
            pm.is_playing = True
            pm.play_position = 0
            
            seq = StepSequencer(pm, render_sr)
            seq.start(None, synth_clock=self.synth.sample_clock)
            # Sequence one pass (two with "loop"), then let the tail ring out
            seq_limit = pattern_duration_samples * (2 if tail_option.get() == "loop" else 1)
            chunks = []
            pos = 0
            while pos < total_samples:
                n = min(1024, total_samples - pos)
                if pos < seq_limit:
                    n = min(n, seq_limit - pos)
                    events = seq.advance(n)
                else:
                    events = []
                chunks.append(self.synth.process_audio_events(n, events))
                pos += n
            audio_buffer = np.concatenate(chunks, axis=0)
            
            # Restore playback state
            self.pattern_manager.is_playing = old_playing_state
            self.pattern_manager.playing_pattern_index = old_playing_idx
            
            # Convert to int16 for WAV file
            is_mono = self.synth.mono
            if is_mono:
                audio_out = audio_buffer[:, 0]  # L=R in mono, take one channel
                audio_int16 = (audio_out * 32767).astype(np.int16)
            else:
                audio_int16 = (audio_buffer * 32767).astype(np.int16)
            
            # Write WAV file
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(1 if is_mono else 2)
                wav_file.setsampwidth(2)
                wav_file.setframerate(render_sr)
                wav_file.writeframes(audio_int16.tobytes())
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export audio file:\\n{e}")
    
    def _on_chain_previous(self):
        """Toggle chain from previous pattern to current (the buttons follow from poll)"""
        self.core.act('pattern.chain_prev', pattern=self._pattern)
    
    def _on_chain_next(self):
        """Toggle chain from current pattern to next (the buttons follow from poll)"""
        self.core.act('pattern.chain_next', pattern=self._pattern)
    
    def _on_matrix_toggle(self):
        """Toggle between lane and matrix editor views"""
        if self.matrix_view_active:
            # Switch to lane view (single channel editor)
            self.matrix_editor.pack_forget()
            self.single_editor_frame.pack(fill='both', expand=True)
            self.matrix_toggle_btn.config(bg=self.COLORS['bg_light'])
            self.matrix_view_active = False
        else:
            # Switch to matrix view (all channels)
            self.single_editor_frame.pack_forget()
            self.matrix_editor.pack(fill='both', padx=5, pady=5)
            self.matrix_toggle_btn.config(bg=self.COLORS['highlight'])
            self._update_matrix_editor()
            self.matrix_view_active = True
    
    def _on_pattern_copy(self):
        """Copy the selected channel's lane to the core's lane clipboard"""
        self.core.act('pattern.copy_lane', pattern=self._pattern,
                      channel=self.selected_channel + 1)
    
    def _on_pattern_paste(self):
        """Paste the lane clipboard into the selected channel"""
        def pasted(result):
            if not result['pasted']:
                messagebox.showwarning("Paste", "Nothing in clipboard")
        self._act_or_warn('pattern.paste_lane', "Paste failed", pasted, pattern=self._pattern,
                          channel=self.selected_channel + 1)
    
    def _on_matrix_edit(self, channel_id, step, value):
        """Handle matrix editor edits"""
        self._push_undo_state()
        self.core.set(self._step_address(channel_id, step, 'trig'), value)
    
    def _lane(self, channel_id, field):
        name = PatternManager.PATTERN_NAMES[self._pattern]
        return self.core.get(f'pattern.{name}.ch{channel_id + 1}.{field}')
    
    def _update_matrix_editor(self):
        """Update matrix editor with current pattern data"""
        self.matrix_editor.set_matrix_data([self._lane(ch, 'trig') for ch in range(8)])
    
    def _update_pattern_editors(self):
        """Show the selected pattern on all lane editors (and the matrix)"""
        self._pattern = PatternManager.PATTERN_NAMES.index(self.core.get('pattern.selected'))
        self._dirty_lanes.clear()
        self._show_lanes(range(8))
    
    def _show_lanes(self, channels):
        """Refresh the lane editors of some channels from the core. An editor
        being dragged keeps its own state until the drag ends."""
        length = self.core.get(f'pattern.{PatternManager.PATTERN_NAMES[self._pattern]}.length')
        for ch_id in channels:
            editor = self.pattern_editors[ch_id]
            if editor.dragging:
                self._dirty_lanes.add(ch_id)
                continue
            self._dirty_lanes.discard(ch_id)
            editor.pattern_length = length
            editor.set_pattern_data(*(self._lane(ch_id, f)
                                      for f in ('trig', 'acc', 'fill', 'prob', 'sub', 'vel')))
        if self.matrix_view_active and not self.matrix_editor.dragging:
            self._update_matrix_editor()
        self._show_pages(length)
    
    # ── pages ────────────────────────────────────────────────────────
    
    def _on_page_select(self, page):
        """A page button: show that page; picking a page by hand turns follow off"""
        self._page_follow = False
        self._set_page(page)
    
    def _on_page_follow_toggle(self):
        """Follow: the editors track the playhead's page while playing"""
        self._page_follow = not self._page_follow
        transport = getattr(self, '_transport', None)
        if self._page_follow and transport and transport['playing']:
            self._set_page(transport['position'] // STEPS_PER_PAGE)
        else:
            self._show_pages()
    
    def _set_page(self, page):
        """Show page `page` (0-3) on the lane editors and the matrix"""
        self._page = max(0, min(MAX_PAGES - 1, page))
        for editor in self.pattern_editors:
            editor.set_page(self._page)
        self.matrix_editor.set_page(self._page)
        self._show_pages()
    
    def _show_pages(self, length=None):
        """Page buttons: the shown page highlighted, pages past the length
        dimmed, the playing page lit; the follow button"""
        if not hasattr(self, 'page_buttons'):
            return
        if length is None:
            length = self.pattern_editors[0].pattern_length
        for page, btn in enumerate(self.page_buttons):
            inside = page * STEPS_PER_PAGE < length
            bg = self.COLORS['highlight'] if page == self._page else (
                self.COLORS['bg_light'] if inside else self.COLORS['bg_dark'])
            fg = self.COLORS['led_on'] if page == self._playing_page else (
                self.COLORS['text'] if inside else self.COLORS['text_dim'])
            btn.config(bg=bg, fg=fg)
        if self._page_follow:
            self.page_follow_btn.config(bg='#44aa66', fg='#ffffff')
        else:
            self.page_follow_btn.config(bg=self.COLORS['bg_light'], fg=self.COLORS['text_dim'])
    
    def _show_playhead(self, transport):
        """The playhead on the editors (from poll); follow turns the page"""
        playing = transport['playing'] and transport['playing_pattern'] == self._pattern
        position = transport['position'] if transport['playing'] else 0
        page = position // STEPS_PER_PAGE if playing else None
        if playing and self._page_follow and page != self._page:
            self._set_page(page)
        if page != self._playing_page:
            self._playing_page = page
            self._show_pages()
        for editor in self.pattern_editors:
            editor.set_current_position(position)
        if self.matrix_view_active:
            self.matrix_editor.set_current_position(position)
    
    def _show_pattern_changes(self, addresses):
        """Pattern addresses reported by poll: refresh the lanes of the shown
        pattern and the pattern buttons."""
        prefix = f'pattern.{PatternManager.PATTERN_NAMES[self._pattern]}.'
        channels = set()
        for address in addresses:
            if not address.startswith(prefix):
                continue
            rest = address[len(prefix):]
            if rest == 'length':
                channels.update(range(8))
            elif rest.startswith('ch') and rest.count('.') == 1:
                channels.add(int(rest[2]) - 1)
        if channels:
            self._show_lanes(sorted(channels))
        self._update_pattern_button_states()
    
    def _update_pattern_ui(self):
        """Update all pattern UI elements (buttons and editors) after loading patterns"""
        self._update_pattern_editors()
        self._update_pattern_button_states()
    
    def _on_key_press(self, event):
        """Handle keyboard input"""
        key = (event.char or '').lower()

        # Number keys 1-8 trigger drums — ensure `key` is a single digit
        if len(key) == 1 and key in '12345678':
            channel = int(key) - 1
            self._trigger_channel(channel)

        # S to save preset
        elif key == 's':
            self._save_preset()

        # L to load preset
        elif key == 'l':
            self._load_preset()
    
    def _trigger_channel(self, channel_idx, velocity=127):
        """Trigger a drum channel (the core places the hit at its arrival time)"""
        self.core.trigger(channel_idx, velocity)
        
        # Flash the channel button
        self.channel_buttons[channel_idx].set_triggered(True)
        self.root.after(100, lambda: self.channel_buttons[channel_idx].set_triggered(False))
    
    def _update_ui_from_channel(self):
        """Update all UI elements from the selected channel's addresses"""
        was_updating = self.updating_ui
        self.updating_ui = True
        try:
            self._do_update_ui_from_channel()
        finally:
            self.updating_ui = was_updating
    
    def _do_update_ui_from_channel(self):
        """Internal: perform all widget updates (called inside the updating_ui guard)"""
        core = self.core
        prefix = f'ch{self.selected_channel + 1}.'
        
        # Update patch name - use fallback if empty
        name = core.get(prefix + 'name')
        self.patch_name_label.config(text=name if name else f"Channel {self.selected_channel + 1}")
        
        # Update drum type labels for all 8 channels
        for i, lbl in enumerate(self.channel_type_labels):
            lbl.config(text=infer_drum_type(core.get(f'ch{i + 1}.name')))
        
        for suffix, show in self._sound_widgets.items():
            show(core.get(prefix + suffix))
    
    def _build_address_widget_tables(self):
        """Address -> function showing its value (engine units) on the widgets.
        
        The UI tick calls these for the changes the core reports in poll;
        _update_ui_from_channel calls the sound ones for the whole channel.
        """
        def position(suffix):
            labels = self.core.describe('ch1.' + suffix)['labels']
            return labels.index
        
        def percent(widget):
            return lambda v: widget.set_value(v * 100)
        
        def lfo(lfo_id):
            def widget(name):
                return getattr(self, f'{lfo_id}_{name}')
            wave_pos = position(f'{lfo_id}.wave')
            sync_pos = position(f'{lfo_id}.sync')
            return {
                f'{lfo_id}.on': widget('enable_btn').set_value,
                f'{lfo_id}.wave': lambda v: widget('wave_var').set(
                    self._lfo_wave_options[min(wave_pos(v), len(self._lfo_wave_options) - 1)]),
                f'{lfo_id}.rate': widget('rate_knob').set_value,
                f'{lfo_id}.depth': widget('depth_knob').set_value,
                f'{lfo_id}.sync': lambda v: widget('sync_var').set(
                    self._lfo_sync_options[min(sync_pos(v), len(self._lfo_sync_options) - 1)]),
                f'{lfo_id}.retrig': widget('retrig_btn').set_value,
                f'{lfo_id}.unipolar': widget('polar_btn').set_value,
                f'{lfo_id}.target': lambda v: widget('dest_var').set(self._mod_target_option(v)),
            }
        
        def delay_time(name):
            # Delay times without a button show as 1/8
            index = self.delay_time_names.index(name) if name in self.delay_time_names else 1
            self.delay_time_selector.set_value(index)
        
        self._sound_widgets = {
            # Mixing
            'mix.osc_noise': lambda v: self.mix_slider.set(v * 100),
            'eq.freq': self.eq_freq_knob.set_value,
            'eq.gain': self.eq_gain_knob.set_value,
            'mix.distortion': percent(self.distort_knob),
            'mix.level': self.level_knob.set_value,
            'mix.pan': self.pan_knob.set_value,
            'mix.choke': self.choke_btn.set_value,
            'mix.output': lambda v: self.output_selector.set_value(0 if v == 'A' else 1),
            # FX
            'fx.vintage': percent(self.vintage_knob),
            'fx.reverb_decay': percent(self.reverb_decay_knob),
            'fx.reverb_mix': percent(self.reverb_mix_knob),
            'fx.reverb_width': percent(self.reverb_width_knob),
            'fx.delay_time': delay_time,
            'fx.delay_feedback': percent(self.delay_feedback_knob),
            'fx.delay_mix': percent(self.delay_mix_knob),
            'fx.delay_pingpong': self.delay_pingpong_btn.set_value,
            # Oscillator
            'osc.wave': lambda v, pos=position('osc.wave'): self.waveform_selector.set_value(pos(v)),
            'osc.freq': self.osc_freq_knob.set_value,
            'osc.pitch': self.pitch_knob.set_value,
            'osc.mod_mode': lambda v, pos=position('osc.mod_mode'):
                self.pitch_mod_mode.set_value(pos(v)),
            'osc.mod_amount': self.pitch_amount_knob.set_value,
            'osc.mod_rate': self.pitch_rate_knob.set_value,
            'osc.attack': self.osc_attack_knob.set_value,
            'osc.decay': self.osc_decay_knob.set_value,
            # Noise
            'noise.filter': lambda v, pos=position('noise.filter'):
                self.noise_filter_mode.set_value(pos(v)),
            'noise.freq': self.noise_freq_knob.set_value,
            'noise.q': self.noise_q_knob.set_value,
            'noise.stereo': self.stereo_btn.set_value,
            'noise.env': lambda v, pos=position('noise.env'): self.noise_env_mode.set_value(pos(v)),
            'noise.attack': self.noise_attack_slider.set_value,
            'noise.decay': self.noise_decay_slider.set_value,
            # Velocity
            'vel.osc': percent(self.osc_vel_slider),
            'vel.noise': percent(self.noise_vel_slider),
            'vel.mod': percent(self.mod_vel_slider),
            # Modulation
            **lfo('lfo1'),
            **lfo('lfo2'),
            'pump.on': self.pump_enable_btn.set_value,
            'pump.amount': percent(self.pump_amount_knob),
            'pump.attack': self.pump_attack_knob.set_value,
            'pump.release': self.pump_release_knob.set_value,
            'pump.curve': percent(self.pump_curve_knob),
            'pump.target': lambda v: self.pump_dest_var.set(self._mod_target_option(v)),
        }
        
        self._global_widgets = {
            'global.tempo': lambda v: self.bpm_var.set(str(v)),
            'global.step_rate': self._show_step_rate,
            'global.fill_rate': self._show_fill_rate,
            'global.swing': lambda v: self.global_swing_slider.set(int(round(v * 100))),
            'global.master': self.master_knob.set_value,
            'global.edit_all': self.edit_all_btn.set_value,
            'morph.position': self._show_morph_position,
        }
    
    def _show_morph_position(self, position):
        """A new morph position (from poll): the knobs show the blended sound."""
        self.morph_slider.set(int(round(position * 100)))
        self._update_ui_from_channel()
    
    def _show_changes(self, changes):
        """Refresh the widgets of the addresses the core reports as changed."""
        self.updating_ui = True
        try:
            channel = changes.get('global.channel')
            if channel is not None and channel - 1 != self.selected_channel:
                self._show_selected_channel(channel - 1)
            prefix = f'ch{self.selected_channel + 1}.'
            patterns = [a for a in changes if a.startswith('pattern.')]
            if patterns:
                self._show_pattern_changes(patterns)
            for address, value in changes.items():
                if address.startswith('ch'):
                    head, _, suffix = address.partition('.')
                    if suffix == 'mute':
                        self._show_mute(int(head[2:]) - 1, value)
                    elif address.startswith(prefix):
                        show = self._sound_widgets.get(suffix)
                        if show is not None:
                            show(value)
                else:
                    show = self._global_widgets.get(address)
                    if show is not None:
                        show(value)
        finally:
            self.updating_ui = False
    
    def _save_preset(self):
        """Save current preset to file"""
        preset_folder = self.preferences_manager.get_preset_folder()
        filename = filedialog.asksaveasfilename(
            initialdir=preset_folder,
            defaultextension='.json',
            filetypes=[('JSON files', '*.json'), ('All files', '*.*')],
            title='Save Preset'
        )
        
        if filename:
            # Get synth data
            data = self.synth.get_preset_data()
            # Add pattern data (includes substeps via PatternChannel.to_dict)
            data['patterns'] = self.pattern_manager.to_dict()
            # Add global settings
            data['tempo'] = self.pattern_manager.bpm
            data['step_rate'] = self.pattern_manager.step_rate
            data['swing'] = self.pattern_manager.swing
            data['fill_rate'] = self.pattern_manager.fill_rate
            # Add morph data
            data['morph'] = self.morph_manager.to_dict()
            # Add program bank data
            data['programs'] = self.synth.get_programs_data()
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            self.preferences_manager.add_recent_file(filename)
            self._refresh_preset_list()
    
    def _load_preset(self):
        """Load preset from file"""
        preset_folder = self.preferences_manager.get_preset_folder()
        filename = filedialog.askopenfilename(
            initialdir=preset_folder,
            filetypes=[
                ('Pythonic Preset', '*.mtpreset'),
                ('JSON files', '*.json'),
                ('All files', '*.*')
            ],
            title='Load Preset'
        )
        
        if filename:
            self._load_preset_file(filename)
    
    def _load_drum_patch(self):
        """Load a single drum patch (.mtdrum) into the currently selected channel"""
        preset_folder = self.preferences_manager.get_preset_folder()
        filename = filedialog.askopenfilename(
            initialdir=preset_folder,
            filetypes=[
                ('Drum Patch', '*.mtdrum'),
                ('All files', '*.*')
            ],
            title=f'Load Drum Patch into Channel {self.selected_channel + 1}'
        )
        
        if filename:
            try:
                # Load the drum patch into the currently selected channel
                self.preset_manager.load_drum_patch(filename, self.selected_channel)
                
                # Update UI to reflect the new drum parameters
                self._update_ui_from_channel()
                
                # Get the patch name from the file
                import os
                patch_name = os.path.splitext(os.path.basename(filename))[0]
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load drum patch: {e}")
    
    def _save_drum_patch(self):
        """Save the currently selected drum to a .mtdrum file"""
        preset_folder = self.preferences_manager.get_preset_folder()
        
        # Get current channel name as default filename
        channel = self.synth.channels[self.selected_channel]
        default_name = getattr(channel, 'name', f'Drum_{self.selected_channel + 1}')
        
        filename = filedialog.asksaveasfilename(
            initialdir=preset_folder,
            defaultextension='.mtdrum',
            initialfile=f'{default_name}.mtdrum',
            filetypes=[('Drum Patch', '*.mtdrum'), ('All files', '*.*')],
            title=f'Save Drum {self.selected_channel + 1} Patch'
        )
        
        if filename:
            try:
                # Save the drum patch
                self.preset_manager.save_drum_patch(self.selected_channel, filename)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save drum patch: {e}")
    
    def _export_all_wavs(self):
        """Export all drums to WAV files"""
        folder = filedialog.askdirectory(title='Select folder for WAV export')
        if folder:
            try:
                exported = self.preset_manager.export_all_drums_to_wav(
                    self.synth, folder,
                    sample_rate=self.synth_sample_rate,
                    mono=self.synth.mono
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {e}")
    
    def _export_current_drum(self):
        """Export the currently selected drum to WAV"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[('WAV files', '*.wav')],
            title=f'Export Drum {self.selected_channel + 1}'
        )
        if filename:
            try:
                self.preset_manager.export_drum_to_wav(
                    self.synth.channels[self.selected_channel],
                    filename,
                    sample_rate=self.synth_sample_rate,
                    mono=self.synth.mono
                )
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {e}")
    
    def _load_preset_file(self, filename, show_message=True):
        """Load a preset file (internal helper)"""
        self._push_undo_state()
        try:
            if filename.lower().endswith('.mtpreset'):
                # Load native Pythonic preset format
                preset_data = self.preset_manager.load_mtpreset(filename)
                if preset_data and preset_data.get('drums'):
                    for i, drum_params in enumerate(preset_data['drums']):
                        if i < 8 and drum_params:
                            channel = self.synth.channels[i]
                            channel.set_parameters(drum_params)
                    
                    # Load patterns if available
                    if preset_data.get('patterns'):
                        self.pattern_manager.load_from_preset_data(preset_data['patterns'])
                        # Update pattern UI to reflect loaded patterns
                        self._update_pattern_ui()
                    
                    # Load tempo if available
                    if 'tempo' in preset_data:
                        self.pattern_manager.set_bpm(int(preset_data['tempo']))
                        if hasattr(self, 'bpm_var'):
                            self.bpm_var.set(str(self.pattern_manager.bpm))
                    
                    # Load step rate if available
                    if 'step_rate' in preset_data:
                        self.pattern_manager.set_step_rate(preset_data['step_rate'])
                        # Update step rate button states
                        for r, btn in self.step_rate_buttons:
                            if r == preset_data['step_rate']:
                                btn.config(bg=self.COLORS['highlight'])
                            else:
                                btn.config(bg=self.COLORS['bg_light'])

                    # Swing, fill rate and master volume are part of the sound
                    if 'swing' in preset_data:
                        self.pattern_manager.set_swing(float(preset_data['swing']))
                        if hasattr(self, 'global_swing_slider'):
                            self.global_swing_slider.set(int(round(self.pattern_manager.swing * 100)))
                    if 'fill_rate' in preset_data:
                        rate = float(preset_data['fill_rate'])
                        self.pattern_manager.set_fill_rate(rate)
                        if hasattr(self, 'fill_rate_buttons'):
                            for r, btn in self.fill_rate_buttons:
                                btn.config(bg=self.COLORS['highlight'] if r == round(rate)
                                           else self.COLORS['bg_light'])
                    if 'master_volume_db' in preset_data:
                        self.synth.set_master_volume(float(preset_data['master_volume_db']))
                        if hasattr(self, 'master_knob'):
                            self.master_knob.set_value(self.synth.master_volume_db)
                    for i, muted in enumerate(preset_data.get('mutes') or []):
                        if i < len(self.synth.channels):
                            self.synth.mute_channel(i, bool(muted))
                            if hasattr(self, 'channel_buttons'):
                                self.channel_buttons[i].set_muted(bool(muted))
                            if hasattr(self, 'mute_buttons') and i < len(self.mute_buttons):
                                self.mute_buttons[i].set_value(bool(muted))
                    
                    # Initialize morph endpoints from loaded state
                    # (mtpreset Morph block has Time/AB but we use the loaded
                    # drum patches as both endpoints since the format doesn't
                    # store full A/B parameter snapshots)
                    self.morph_manager._init_endpoints()
                    # Set morph position from mtpreset if available
                    morph_pos = preset_data.get('morph_position')
                    if morph_pos is not None:
                        self.morph_slider.set(int(float(morph_pos) * 100))
                    else:
                        self.morph_slider.set(50)  # Center morph after loading
                    self._update_morph_ui()
                    
                    self._update_ui_from_channel()
                    self.preferences_manager.add_recent_file(filename)
                    self.preferences_manager.set('last_preset', filename)
                    self._refresh_preset_list()
                else:
                    messagebox.showerror("Error", "Failed to parse preset file")
            else:
                # Load JSON format
                with open(filename, 'r') as f:
                    data = json.load(f)
                self.synth.load_preset_data(data)
                
                # Load patterns if available (includes substeps)
                if 'patterns' in data:
                    self.pattern_manager.from_dict(data['patterns'])
                    self._update_pattern_ui()
                
                # Load global settings
                if 'tempo' in data:
                    self.pattern_manager.set_bpm(int(data['tempo']))
                    if hasattr(self, 'bpm_var'):
                        self.bpm_var.set(str(self.pattern_manager.bpm))
                
                if 'step_rate' in data:
                    self.pattern_manager.set_step_rate(data['step_rate'])
                    if hasattr(self, 'step_rate_buttons'):
                        for r, btn in self.step_rate_buttons:
                            if r == data['step_rate']:
                                btn.config(bg=self.COLORS['highlight'])
                            else:
                                btn.config(bg=self.COLORS['bg_light'])
                
                if 'swing' in data:
                    self.pattern_manager.set_swing(data['swing'])
                
                if 'fill_rate' in data:
                    self.pattern_manager.set_fill_rate(int(data['fill_rate']))
                
                # Load morph data if available
                if 'morph' in data:
                    self.morph_manager.from_dict(data['morph'])
                    self.morph_slider.set(int(self.morph_manager.position * 100))
                else:
                    # No morph data - initialize fresh endpoints
                    self.morph_manager._init_endpoints()
                    self.morph_slider.set(50)
                self._update_morph_ui()
                
                # Load program bank if available
                if 'programs' in data:
                    self.synth.load_programs_data(data['programs'])
                    current = self.synth.get_current_program()
                    self.program_var.set(str(current + 1))
                else:
                    # No program data — reset bank
                    self.synth._programs = [None] * self.synth.NUM_PROGRAMS
                    self.synth._current_program = 0
                    self.program_var.set("1")
                
                self._update_ui_from_channel()
                self.preferences_manager.add_recent_file(filename)
                self.preferences_manager.set('last_preset', filename)
                self._refresh_preset_list()
        except Exception as e:
            if show_message:
                messagebox.showerror("Error", f"Failed to load preset: {e}")
    
    def _select_preset_folder(self):
        """Select a new preset folder"""
        current_folder = self.preferences_manager.get_preset_folder()
        folder = filedialog.askdirectory(
            initialdir=current_folder,
            title='Select Preset Folder'
        )
        
        if folder:
            self.preferences_manager.set_preset_folder(folder)
            self._refresh_preset_list()
    
    # ============ MIDI (routed by the core) ============
    # The core opens the MIDI input and routes notes, program change,
    # transport, clock, CC (with pickup) and pitch bend itself. The GUI only
    # offers the controls for learn and shows what poll reports.

    def _register_cc_parameter(self, param_name: str, widget):
        """Offer a control for MIDI learn under its saved parameter name."""
        self._cc_widgets[param_name] = widget
        self._add_midi_learn_context_menu(widget, param_name)

    def _add_midi_learn_context_menu(self, widget, param_name: str):
        """Add right-click context menu with MIDI Learn to a widget"""
        target = cc_parameter_target(param_name)

        def show_context_menu(event):
            menu = tk.Menu(self.root, tearoff=0)
            
            # Check if this parameter already has a CC mapping
            mapped = [cc for cc, t in sorted(self.core.get('midi.cc_map').items()) if t == target]
            if mapped:
                menu.add_command(label=f"Mapped to {cc_name(mapped[0])}", state='disabled')
                menu.add_command(label="Remove CC Mapping", 
                               command=lambda: self._remove_cc_mapping(param_name))
                menu.add_separator()
            
            # Check if this parameter is the pitch bend target
            pitchbend_target = self.core.get('midi.pitchbend_target')
            if pitchbend_target == target:
                menu.add_command(label="Pitch Bend → This Parameter", state='disabled')
                menu.add_command(label="Remove Pitch Bend Mapping", 
                               command=self._remove_pitchbend_mapping)
                menu.add_separator()
            
            if self.core.get('midi.learning') is not None:
                menu.add_command(label="Cancel MIDI Learn", 
                               command=self._cancel_midi_learn)
            else:
                menu.add_command(label="MIDI Learn (CC)", 
                               command=lambda: self._start_midi_learn(param_name, widget))
                # Only show "Assign Pitch Bend" if not already assigned to this param
                if pitchbend_target != target:
                    menu.add_command(label="Assign Pitch Bend", 
                                   command=lambda: self._assign_pitchbend(param_name))
            
            menu.add_separator()
            menu.add_command(label="MIDI Settings...", 
                           command=self._show_cc_mapping_dialog)
            
            try:
                menu.tk_popup(event.x_root, event.y_root)
            finally:
                menu.grab_release()
        
        widget.bind('<Button-3>', show_context_menu)
    
    def _start_midi_learn(self, param_name: str, widget):
        """Ask the core to map the next CC to this control; the widget
        flashes until the learn action ends (learned or cancelled)."""
        self._flash_midi_learn_widget(False)  # a running learn is replaced
        action_id = self.core.act('midi.learn', target=cc_parameter_target(param_name))
        self._midi_learn_action = action_id
        self._midi_learn_widget = widget
        self._flash_midi_learn_widget(True)
        
        def on_done(event):
            if self._midi_learn_action == action_id:
                self._midi_learn_action = None
                self._flash_midi_learn_widget(False)
                self._midi_learn_widget = None
            if event['status'] == 'error':
                print(f"MIDI Learn failed: {event.get('error')}", flush=True)
        
        self._when_action_done(action_id, on_done)
    
    def _flash_midi_learn_widget(self, flashing: bool):
        """Flash the learning widget (orange every 300 ms) or restore it"""
        widget = self._midi_learn_widget
        if self._midi_learn_flash_id:
            self.root.after_cancel(self._midi_learn_flash_id)
            self._midi_learn_flash_id = None
        if widget is None:
            return
        
        if flashing:
            def flash():
                # Toggle between normal and highlight color
                current = widget.cget('bg') if hasattr(widget, 'cget') else '#3a3a4a'
                new_color = '#ff8844' if current != '#ff8844' else '#3a3a4a'
                if hasattr(widget, 'configure'):
                    try:
                        widget.configure(bg=new_color)
                    except tk.TclError:
                        pass
                self._midi_learn_flash_id = self.root.after(300, flash)
            flash()
        elif hasattr(widget, 'configure'):
            # Restore original color
            try:
                widget.configure(bg='#3a3a4a')
            except tk.TclError:
                pass
    
    def _cancel_midi_learn(self):
        """Cancel MIDI learn mode (the learn action then ends as cancelled)"""
        self.core.act('midi.learn_cancel')
    
    def _remove_cc_mapping(self, param_name: str):
        """Remove the CC mapping of a control"""
        target = cc_parameter_target(param_name)
        mapping = self.core.get('midi.cc_map')
        self.core.set('midi.cc_map', {cc: t for cc, t in mapping.items() if t != target})
        print(f"Removed MIDI mapping for {param_name}")
    
    def _assign_pitchbend(self, param_name: str):
        """Assign pitch bend wheel to control a parameter"""
        self.core.set('midi.pitchbend_target', cc_parameter_target(param_name))
        print(f"Pitch bend wheel assigned to: {param_name}")
    
    def _remove_pitchbend_mapping(self):
        """Remove pitch bend mapping"""
        self.core.set('midi.pitchbend_target', None)
        print("Pitch bend mapping removed")
    
    def _get_available_parameters(self) -> list:
        """Get list of available parameters for CC mapping"""
        return sorted(self._cc_widgets)
    
    def _show_midi(self, state):
        """MIDI activity from poll: the LED and the channel buttons of played notes."""
        if state['activity'] != self._midi_activity:
            self._midi_activity = state['activity']
            self._update_midi_indicator()
        notes = state['notes']
        for channel, count in enumerate(notes):
            if count != self._midi_notes[channel]:
                self._flash_channel_button(channel)
        self._midi_notes = notes
    
    def _flash_channel_button(self, channel: int):
        """Flash a channel button to show it was triggered"""
        if hasattr(self, 'channel_buttons') and channel < len(self.channel_buttons):
            self.channel_buttons[channel].set_triggered(True)
            self.root.after(100, lambda: self.channel_buttons[channel].set_triggered(False))
    
    def _push_cc_burst(self, event):
        """A MIDI CC burst the core closed (400 ms idle): one undo step, with
        the snapshot the core took before it."""
        self._undo_stack.append(event['snapshot'])
        if len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self._update_undo_redo_buttons()
    
    def _register_undo_on_widgets(self):
        """Register undo/redo callbacks on all knobs and sliders"""
        undo_cb = self._push_undo_state_deferred
        widgets = [
            self.eq_freq_knob, self.distort_knob, self.eq_gain_knob,
            self.level_knob, self.pan_knob,
            self.osc_freq_knob, self.pitch_knob, self.pitch_amount_knob, self.pitch_rate_knob,
            self.osc_attack_knob, self.osc_decay_knob,
            self.noise_freq_knob, self.noise_q_knob,
            self.noise_attack_slider, self.noise_decay_slider,
            self.osc_vel_slider, self.noise_vel_slider, self.mod_vel_slider,
            self.vintage_knob,
            self.reverb_decay_knob, self.reverb_mix_knob, self.reverb_width_knob,
            self.delay_feedback_knob, self.delay_mix_knob,
        ]
        for w in widgets:
            w.command_end = undo_cb
        # Initial snapshot so the first undo has a baseline
        self._push_undo_state()

    def _register_cc_parameters(self):
        """Offer the knobs and sliders for MIDI learn (ranges come from the core's describe)"""
        # Mixing section
        self._register_cc_parameter('level', self.level_knob)
        self._register_cc_parameter('pan', self.pan_knob)
        self._register_cc_parameter('distortion', self.distort_knob)
        self._register_cc_parameter('eq_freq', self.eq_freq_knob)
        self._register_cc_parameter('eq_gain', self.eq_gain_knob)
        self._register_cc_parameter('vintage', self.vintage_knob)
        
        # Reverb section
        self._register_cc_parameter('reverb_decay', self.reverb_decay_knob)
        self._register_cc_parameter('reverb_mix', self.reverb_mix_knob)
        self._register_cc_parameter('reverb_width', self.reverb_width_knob)
        
        # Delay section
        self._register_cc_parameter('delay_feedback', self.delay_feedback_knob)
        self._register_cc_parameter('delay_mix', self.delay_mix_knob)
        
        # Oscillator section
        self._register_cc_parameter('osc_freq', self.osc_freq_knob)
        self._register_cc_parameter('pitch', self.pitch_knob)
        self._register_cc_parameter('pitch_amount', self.pitch_amount_knob)
        self._register_cc_parameter('pitch_rate', self.pitch_rate_knob)
        self._register_cc_parameter('osc_attack', self.osc_attack_knob)
        self._register_cc_parameter('osc_decay', self.osc_decay_knob)
        
        # Noise section
        self._register_cc_parameter('noise_freq', self.noise_freq_knob)
        self._register_cc_parameter('noise_q', self.noise_q_knob)
        self._register_cc_parameter('noise_attack', self.noise_attack_slider)
        self._register_cc_parameter('noise_decay', self.noise_decay_slider)
        
        # Velocity section
        self._register_cc_parameter('osc_vel', self.osc_vel_slider)
        self._register_cc_parameter('noise_vel', self.noise_vel_slider)
        self._register_cc_parameter('mod_vel', self.mod_vel_slider)
        
        # Mix slider
        self._register_cc_parameter('osc_noise_mix', self.mix_slider)
        
        # Master volume
        self._register_cc_parameter('master_volume', self.master_knob)
        
        # Morph slider (global parameter)
        self._register_cc_parameter('sound_morph', self.morph_slider)
        
        # LFO 1
        self._register_cc_parameter('lfo1_rate', self.lfo1_rate_knob)
        self._register_cc_parameter('lfo1_depth', self.lfo1_depth_knob)
        
        # LFO 2
        self._register_cc_parameter('lfo2_rate', self.lfo2_rate_knob)
        self._register_cc_parameter('lfo2_depth', self.lfo2_depth_knob)
        
        # Pump
        self._register_cc_parameter('pump_amount', self.pump_amount_knob)
        self._register_cc_parameter('pump_attack', self.pump_attack_knob)
        self._register_cc_parameter('pump_release', self.pump_release_knob)
        self._register_cc_parameter('pump_curve', self.pump_curve_knob)

    def _update_midi_indicator(self):
        """Update the MIDI activity indicator LED"""
        if hasattr(self, 'midi_indicator') and hasattr(self, '_midi_indicator_id'):
            # Show green for activity
            self.midi_indicator.itemconfig(self._midi_indicator_id, fill=self.COLORS['led_on'])
            # Schedule turning it off after 100ms
            self.root.after(100, self._reset_midi_indicator)
    
    def _reset_midi_indicator(self):
        """Reset the MIDI indicator to off state"""
        if hasattr(self, 'midi_indicator') and hasattr(self, '_midi_indicator_id'):
            self.midi_indicator.itemconfig(self._midi_indicator_id, fill=self.COLORS['led_off'])
    
    def _show_po32_transfer(self):
        """Open the PO-32 Tonic transfer dialog"""
        preset_name = getattr(self.preset_manager, 'current_preset_name', 'Untitled')
        PO32TransferDialog(
            self.root,
            self.synth,
            self.pattern_manager,
            preset_name=preset_name
        )
    
    def _show_po32_import(self):
        """Open the PO-32 import dialog (with pattern & bank support)."""
        def on_import_complete():
            """Refresh all UI after import."""
            self._update_ui_from_channel()
            self._update_pattern_editors()
            # Reset morph slider and update learn buttons after import
            self.morph_slider.set(0)
            self._update_morph_ui()
        
        dialog = PO32ImportDialog(
            parent=self.root,
            synth=self.synth,
            pattern_manager=self.pattern_manager,
            on_import_callback=on_import_complete,
            preferences_manager=self.preferences_manager,
        )
        # Store morph_manager on root so PO32ImportDialog can find it
        self.root.morph_manager = self.morph_manager
        # Wait for dialog to close, then refresh UI
        self.root.wait_window(dialog.dialog)
        self._update_ui_from_channel()
        self._update_pattern_editors()
        self._update_morph_ui()
    
    def _show_drum_generator(self):
        """Open the AI Drum Generator dialog (modal).

        Stops the main transport while the dialog is open and restores
        playback state on close.  The dialog reuses the live synth and
        transport so all previews sound identical to main-window playback.
        """
        # Save and stop transport
        state = self.core.poll()['transport']
        was_playing = state['playing']
        saved_pattern_idx = state['playing_pattern']
        if was_playing:
            self.core.act('transport.stop')

        def on_apply(mode='patches'):
            self._push_undo_state()
            self._update_ui_from_channel()
            if mode == 'patches_and_patterns':
                self._update_pattern_editors()
            self._update_morph_ui()

        def start_transport():
            self.core.act('transport.play')

        def stop_transport():
            self.core.act('transport.stop')

        dialog = DrumGeneratorDialog(
            parent=self.root,
            synth=self.synth,
            pattern_manager=self.pattern_manager,
            preferences_manager=self.preferences_manager,
            on_apply_callback=on_apply,
            start_transport=start_transport,
            stop_transport=stop_transport,
        )
        self.root.wait_window(dialog.dialog)

        # Restore transport state (the buttons follow from poll)
        if was_playing:
            self.core.act('transport.play', pattern=saved_pattern_idx)

        self._update_ui_from_channel()
        self._update_pattern_editors()
    
    def _current_audio_device_text(self):
        """'Currently using: ...' line of the audio settings dialog."""
        if not self.core.get('audio.running'):
            return "Audio stream not running"
        name = self.core.get('audio.device')
        if self.core.get('audio.device_is_default'):
            return f"Currently using: {name} (default)"
        return f"Currently using: {name}"
    
    def _show_audio_preferences(self):
        """Show audio settings dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Audio Settings")
        dialog.geometry("450x810")
        dialog.resizable(True, True)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.configure(bg=self.COLORS['bg_dark'])
        
        # Title
        tk.Label(dialog, text="Audio Settings", 
                font=('Segoe UI', 12, 'bold'),
                fg=self.COLORS['text'],
                bg=self.COLORS['bg_dark']).pack(pady=(15, 10))
        
        # Audio Device selection
        device_frame = tk.Frame(dialog, bg=self.COLORS['bg_dark'])
        device_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Label(device_frame, text="Audio Output Device:", 
                font=('Segoe UI', 9),
                fg=self.COLORS['text'],
                bg=self.COLORS['bg_dark']).pack(anchor='w')
        
        # Get available devices
        device_names = self.core.get('audio.output_devices')
        
        # Get current setting
        current_device = self.preferences_manager.get('audio_output_device')
        if current_device is None:
            current_display = "(System Default)"
        else:
            current_display = current_device if current_device in device_names else "(System Default)"
        
        device_var = tk.StringVar(value=current_display)
        device_options = ["(System Default)"] + device_names
        device_combo = ttk.Combobox(device_frame, textvariable=device_var, 
                                    values=device_options, width=50, state='readonly')
        device_combo.pack(fill='x', pady=(2, 0))
        
        # Show current device info
        current_info = self._current_audio_device_text()
            
        info_label = tk.Label(device_frame, text=current_info,
                             font=('Segoe UI', 8),
                             fg=self.COLORS['text_dim'],
                             bg=self.COLORS['bg_dark'])
        info_label.pack(anchor='w', pady=(5, 0))
        
        # Note about restart
        note_label = tk.Label(device_frame, 
                             text="Use 'Apply Now' to apply changes immediately, or 'OK' to save for next launch.",
                             font=('Segoe UI', 8, 'italic'),
                             fg=self.COLORS['accent'],
                             bg=self.COLORS['bg_dark'])
        note_label.pack(anchor='w', pady=(10, 0))
        
        # --- Audio Input Device section ---
        input_frame = tk.Frame(dialog, bg=self.COLORS['bg_dark'])
        input_frame.pack(fill='x', padx=20, pady=(10, 0))
        
        tk.Label(input_frame, text="Audio Input Device (for PO-32 recording):", 
                font=('Segoe UI', 9),
                fg=self.COLORS['text'],
                bg=self.COLORS['bg_dark']).pack(anchor='w')
        
        input_device_names = self.core.get('audio.input_devices')
        
        current_input_device = self.preferences_manager.get('audio_input_device')
        if current_input_device is None:
            current_input_display = "(System Default)"
        else:
            current_input_display = current_input_device if current_input_device in input_device_names else "(System Default)"
        
        input_device_var = tk.StringVar(value=current_input_display)
        input_device_options = ["(System Default)"] + input_device_names
        input_device_combo = ttk.Combobox(input_frame, textvariable=input_device_var, 
                                          values=input_device_options, width=50, state='readonly')
        input_device_combo.pack(fill='x', pady=(2, 0))
        
        try:
            default_input = sd.query_devices(kind='input')
            input_info_text = f"Default input: {default_input['name']}"
        except Exception:
            input_info_text = "No default input device detected"
        
        tk.Label(input_frame, text=input_info_text,
                font=('Segoe UI', 8),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_dark']).pack(anchor='w', pady=(5, 0))
        
        # --- Audio Buffer Size section ---
        buffer_frame = tk.Frame(dialog, bg=self.COLORS['bg_dark'])
        buffer_frame.pack(fill='x', padx=20, pady=(15, 0))
        
        tk.Label(buffer_frame, text="Audio Buffer Size:", 
                font=('Segoe UI', 9),
                fg=self.COLORS['text'],
                bg=self.COLORS['bg_dark']).pack(anchor='w')
        
        buffer_options = [
            ("2 ms  (88 samples) — extreme", 2.0),
            ("5 ms  (220 samples) — ultra low latency", 5.0),
            ("10 ms  (441 samples) — low latency", 10.0),
            ("15 ms  (661 samples) — balanced", 15.0),
            ("23.8 ms  (1050 samples) — default", 23.8),
            ("30 ms  (1323 samples) — relaxed", 30.0),
            ("50 ms  (2205 samples) — safe", 50.0),
            ("75 ms  (3307 samples) — very safe", 75.0),
            ("100 ms  (4410 samples) — maximum", 100.0),
        ]
        current_buffer_ms = self.preferences_manager.get('audio_buffer_ms', 23.8)
        # Find closest match
        closest_label = buffer_options[4][0]  # default (23.8 ms)
        for label, val in buffer_options:
            if abs(val - current_buffer_ms) < 0.5:
                closest_label = label
                break
        
        buffer_var = tk.StringVar(value=closest_label)
        buffer_combo = ttk.Combobox(buffer_frame, textvariable=buffer_var, 
                                    values=[label for label, _ in buffer_options],
                                    width=50, state='readonly')
        buffer_combo.pack(fill='x', pady=(2, 0))
        
        buffer_info = tk.Label(buffer_frame, 
                              text="Lower = less latency but more CPU. Higher = more stable.",
                              font=('Segoe UI', 8),
                              fg=self.COLORS['text_dim'],
                              bg=self.COLORS['bg_dark'])
        buffer_info.pack(anchor='w', pady=(5, 0))
        
        # --- Output Sample Rate section ---
        sr_frame = tk.Frame(dialog, bg=self.COLORS['bg_dark'])
        sr_frame.pack(fill='x', padx=20, pady=(15, 0))
        
        tk.Label(sr_frame, text="Output Sample Rate:", 
                font=('Segoe UI', 9),
                fg=self.COLORS['text'],
                bg=self.COLORS['bg_dark']).pack(anchor='w')
        
        sr_options = [
            ("96000 Hz — studio quality", 96000),
            ("48000 Hz — high quality", 48000),
            ("44100 Hz — CD quality (default)", 44100),
            ("32000 Hz — broadcast", 32000),
            ("22050 Hz — low quality", 22050),
            ("11025 Hz — very low", 11025),
            ("8000 Hz — telephone", 8000),
        ]
        
        def _get_supported_rates(device_name):
            """Query which sample rates the selected device supports."""
            dev_idx = None
            if device_name and device_name != "(System Default)":
                try:
                    for i, dev in enumerate(sd.query_devices()):
                        if dev['max_output_channels'] > 0 and dev['name'] == device_name:
                            dev_idx = i
                            break
                except Exception:
                    pass
            supported = []
            for label, rate in sr_options:
                try:
                    sd.check_output_settings(device=dev_idx, channels=2, samplerate=rate)
                    supported.append((label, rate))
                except Exception:
                    pass
            return supported if supported else sr_options  # fallback to all if query fails
        
        def _update_sr_combo(*_args):
            """Update sample rate combo to show only device-supported rates."""
            dev_name = device_var.get()
            if dev_name == "(System Default)":
                dev_name = None
            supported = _get_supported_rates(dev_name)
            sr_combo['values'] = [label for label, _ in supported]
            # If current selection is not supported, pick closest supported
            current_sel = sr_var.get()
            supported_labels = [label for label, _ in supported]
            if current_sel not in supported_labels:
                # Default to 44100 if available, else first supported
                for label, val in supported:
                    if val == 44100:
                        sr_var.set(label)
                        return
                sr_var.set(supported_labels[0])
        
        current_sr = self.preferences_manager.get('audio_sample_rate', 44100)
        current_sr_label = sr_options[2][0]  # default 44100
        for label, val in sr_options:
            if val == current_sr:
                current_sr_label = label
                break
        
        sr_var = tk.StringVar(value=current_sr_label)
        sr_combo = ttk.Combobox(sr_frame, textvariable=sr_var, 
                                values=[label for label, _ in sr_options],
                                width=50, state='readonly')
        sr_combo.pack(fill='x', pady=(2, 0))
        
        sr_status_label = tk.Label(sr_frame, 
                text="",
                font=('Segoe UI', 8),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_dark'])
        sr_status_label.pack(anchor='w', pady=(5, 0))
        
        # Populate supported rates for current device and show status
        def _refresh_sr_status():
            dev_name = device_var.get()
            if dev_name == "(System Default)":
                dev_name = None
            supported = _get_supported_rates(dev_name)
            n = len(supported)
            total = len(sr_options)
            if n < total:
                sr_status_label.config(text=f"Device supports {n} of {total} sample rates. Unsupported rates are hidden.")
            else:
                sr_status_label.config(text="Device supports all sample rates.")
        
        _update_sr_combo()
        _refresh_sr_status()
        
        # Re-filter when output device changes
        device_var.trace_add('write', lambda *a: (_update_sr_combo(), _refresh_sr_status()))
        
        # --- Internal Synth Rate section ---
        synth_sr_frame = tk.Frame(dialog, bg=self.COLORS['bg_dark'])
        synth_sr_frame.pack(fill='x', padx=20, pady=(15, 0))
        
        tk.Label(synth_sr_frame, text="Internal Synth Rate:", 
                font=('Segoe UI', 9),
                fg=self.COLORS['text'],
                bg=self.COLORS['bg_dark']).pack(anchor='w')
        
        synth_sr_options = [
            ("Same as output (default)", 0),
            ("22050 Hz — lo-fi", 22050),
            ("11025 Hz — crunchy lo-fi", 11025),
            ("8000 Hz — telephone / 8-bit", 8000),
        ]
        
        current_synth_sr = self.preferences_manager.get('synth_sample_rate', 44100)
        current_synth_sr_label = synth_sr_options[0][0]  # default "Same as output"
        for label, val in synth_sr_options:
            if val == current_synth_sr:
                current_synth_sr_label = label
                break
        
        synth_sr_var = tk.StringVar(value=current_synth_sr_label)
        synth_sr_combo = ttk.Combobox(synth_sr_frame, textvariable=synth_sr_var, 
                                      values=[label for label, _ in synth_sr_options],
                                      width=50, state='readonly')
        synth_sr_combo.pack(fill='x', pady=(2, 0))
        
        tk.Label(synth_sr_frame,
                text="Lower = less CPU + lo-fi character. Audio is upsampled to output rate.",
                font=('Segoe UI', 8),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_dark']).pack(anchor='w', pady=(5, 0))
        
        def _get_selected_synth_rate():
            """Extract synth rate from combo selection. 0 means same as output."""
            sel = synth_sr_var.get()
            for label, val in synth_sr_options:
                if label == sel:
                    return val
            return 0
        
        # --- Mono / Stereo section ---
        mono_frame = tk.Frame(dialog, bg=self.COLORS['bg_dark'])
        mono_frame.pack(fill='x', padx=20, pady=(15, 0))
        
        current_mono = self.preferences_manager.get('audio_mono', False)
        mono_var = tk.BooleanVar(value=current_mono)
        mono_check = tk.Checkbutton(mono_frame, text="Mono output",
                                    variable=mono_var,
                                    font=('Segoe UI', 9),
                                    fg=self.COLORS['text'],
                                    bg=self.COLORS['bg_dark'],
                                    selectcolor=self.COLORS['bg_medium'],
                                    activebackground=self.COLORS['bg_dark'],
                                    activeforeground=self.COLORS['text'])
        mono_check.pack(anchor='w')
        
        tk.Label(mono_frame,
                text="Disables pan, stereo noise, and reverb width. WAV exports in mono.",
                font=('Segoe UI', 8),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_dark']).pack(anchor='w', pady=(2, 0))
        
        # Buttons
        button_frame = tk.Frame(dialog, bg=self.COLORS['bg_dark'])
        button_frame.pack(fill='x', padx=20, pady=20)
        
        def refresh_devices():
            device_names = self.core.get('audio.output_devices')
            device_combo['values'] = ["(System Default)"] + device_names
            input_names = self.core.get('audio.input_devices')
            input_device_combo['values'] = ["(System Default)"] + input_names
            _update_sr_combo()
            _refresh_sr_status()
        
        def _get_selected_buffer_ms():
            """Extract buffer ms value from combo selection."""
            sel = buffer_var.get()
            for label, val in buffer_options:
                if label == sel:
                    return val
            return 23.8
        
        def _get_selected_sample_rate():
            """Extract sample rate from combo selection."""
            sel = sr_var.get()
            for label, val in sr_options:
                if label == sel:
                    return val
            return 44100
        
        def save_and_close():
            selected = device_var.get()
            if selected == "(System Default)":
                self.preferences_manager.set('audio_output_device', None)
            else:
                self.preferences_manager.set('audio_output_device', selected)
            selected_input = input_device_var.get()
            if selected_input == "(System Default)":
                self.preferences_manager.set('audio_input_device', None)
            else:
                self.preferences_manager.set('audio_input_device', selected_input)
            self.preferences_manager.set('audio_buffer_ms', _get_selected_buffer_ms())
            self.preferences_manager.set('audio_sample_rate', _get_selected_sample_rate())
            synth_sr = _get_selected_synth_rate()
            self.preferences_manager.set('synth_sample_rate', synth_sr if synth_sr > 0 else _get_selected_sample_rate())
            self.preferences_manager.set('audio_mono', mono_var.get())
            print(f"Audio output device preference saved: {selected}", flush=True)
            print(f"Audio input device preference saved: {selected_input}", flush=True)
            print(f"Audio buffer size preference saved: {_get_selected_buffer_ms()} ms", flush=True)
            print(f"Audio sample rate preference saved: {_get_selected_sample_rate()} Hz", flush=True)
            print(f"Synth rate preference saved: {'same as output' if synth_sr == 0 else str(synth_sr) + ' Hz'}", flush=True)
            print(f"Mono mode preference saved: {mono_var.get()}", flush=True)
            dialog.destroy()
        
        def apply_now():
            """Apply changes: the core saves the audio preferences, rebuilds the
            synth if its rate changed and restarts the stream."""
            selected = device_var.get()
            selected_input = input_device_var.get()
            # The input device is only a preference (read by the PO-32 dialogs)
            if selected_input == "(System Default)":
                self.preferences_manager.set('audio_input_device', None)
            else:
                self.preferences_manager.set('audio_input_device', selected_input)
            
            action_id = self.core.act(
                'audio.apply',
                device=None if selected == "(System Default)" else selected,
                sample_rate=_get_selected_sample_rate(),
                synth_rate=_get_selected_synth_rate(),
                buffer_ms=_get_selected_buffer_ms(),
                mono=mono_var.get())
            
            def on_applied(event):
                # Update info label
                if info_label.winfo_exists():
                    info_label.config(text=self._current_audio_device_text())
                if event['status'] == 'done':
                    print(f"Audio output device changed to: {selected}", flush=True)
                    print(f"Audio input device changed to: {selected_input}", flush=True)
            
            self._when_action_done(action_id, on_applied)
        
        tk.Button(button_frame, text="Refresh", 
                 command=refresh_devices,
                 bg=self.COLORS['bg_light'],
                 fg=self.COLORS['text']).pack(side='left')
        
        tk.Button(button_frame, text="Apply Now", 
                 command=apply_now,
                 bg=self.COLORS['bg_light'],
                 fg=self.COLORS['text']).pack(side='left', padx=(10, 0))
        
        tk.Button(button_frame, text="OK", 
                 command=save_and_close,
                 bg=self.COLORS['accent'],
                 fg=self.COLORS['text']).pack(side='right')
        
        tk.Button(button_frame, text="Cancel", 
                 command=dialog.destroy,
                 bg=self.COLORS['bg_light'],
                 fg=self.COLORS['text']).pack(side='right', padx=(0, 5))
        
        # Center dialog on parent
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - dialog.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")
    
    def _show_synthesis_preferences(self):
        """Show synthesis settings dialog (smoothing, etc.)"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Synthesis Settings")
        dialog.geometry("400x200")
        dialog.resizable(True, True)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.configure(bg=self.COLORS['bg_dark'])
        
        # Title
        tk.Label(dialog, text="Synthesis Settings", 
                font=('Segoe UI', 12, 'bold'),
                fg=self.COLORS['text'],
                bg=self.COLORS['bg_dark']).pack(pady=(15, 10))
        
        # Smoothing time frame
        smooth_frame = tk.Frame(dialog, bg=self.COLORS['bg_dark'])
        smooth_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Label(smooth_frame, text="Parameter Smoothing Time:", 
                font=('Segoe UI', 9),
                fg=self.COLORS['text'],
                bg=self.COLORS['bg_dark']).pack(anchor='w')
        
        # Current smoothing value
        current_smoothing = self.preferences_manager.get('param_smoothing_ms', 30.0)
        smoothing_var = tk.DoubleVar(value=current_smoothing)
        
        # Slider for smoothing time (5-100ms)
        slider_frame = tk.Frame(smooth_frame, bg=self.COLORS['bg_dark'])
        slider_frame.pack(fill='x', pady=(5, 0))
        
        tk.Label(slider_frame, text="5ms", font=('Segoe UI', 8),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_dark']).pack(side='left')
        
        smoothing_slider = tk.Scale(slider_frame, from_=5, to=100, 
                                    orient='horizontal', 
                                    variable=smoothing_var,
                                    resolution=1,
                                    length=250,
                                    bg=self.COLORS['bg_medium'],
                                    fg=self.COLORS['text'],
                                    highlightthickness=0,
                                    troughcolor=self.COLORS['bg_dark'])
        smoothing_slider.pack(side='left', padx=5)
        
        tk.Label(slider_frame, text="100ms", font=('Segoe UI', 8),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_dark']).pack(side='left')
        
        # Explanation
        tk.Label(smooth_frame, 
                text="Controls how smoothly parameter changes are applied.\nLower = faster response, Higher = smoother transitions.",
                font=('Segoe UI', 8),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_dark'],
                justify='left').pack(anchor='w', pady=(10, 0))
        
        # Buttons
        button_frame = tk.Frame(dialog, bg=self.COLORS['bg_dark'])
        button_frame.pack(fill='x', padx=20, pady=20)
        
        def apply_and_close():
            smoothing_ms = smoothing_var.get()
            self.preferences_manager.set('param_smoothing_ms', smoothing_ms)
            # Apply to all channels
            for channel in self.synth.channels:
                channel.set_smoothing_time(smoothing_ms)
            print(f"Parameter smoothing set to {smoothing_ms}ms", flush=True)
            dialog.destroy()
        
        def apply_now():
            smoothing_ms = smoothing_var.get()
            self.preferences_manager.set('param_smoothing_ms', smoothing_ms)
            # Apply to all channels
            for channel in self.synth.channels:
                channel.set_smoothing_time(smoothing_ms)
            print(f"Parameter smoothing set to {smoothing_ms}ms", flush=True)
        
        tk.Button(button_frame, text="Apply", 
                 command=apply_now,
                 bg=self.COLORS['bg_light'],
                 fg=self.COLORS['text']).pack(side='left')
        
        tk.Button(button_frame, text="OK", 
                 command=apply_and_close,
                 bg=self.COLORS['accent'],
                 fg=self.COLORS['text']).pack(side='right')
        
        tk.Button(button_frame, text="Cancel", 
                 command=dialog.destroy,
                 bg=self.COLORS['bg_light'],
                 fg=self.COLORS['text']).pack(side='right', padx=(0, 5))
        
        # Center dialog on parent
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - dialog.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")
    
    def _show_ai_preferences(self):
        """Show AI settings dialog (pattern model path, temperature)."""
        from pythonic.pattern_generator import PatternGenerator, _BUNDLED_CHECKPOINT

        dialog = tk.Toplevel(self.root)
        dialog.title("AI Settings")
        dialog.geometry("500x260")
        dialog.resizable(True, True)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.configure(bg=self.COLORS['bg_dark'])

        tk.Label(dialog, text="AI Settings",
                 font=('Segoe UI', 12, 'bold'),
                 fg=self.COLORS['text'],
                 bg=self.COLORS['bg_dark']).pack(pady=(15, 10))

        # ── Pattern model path ───────────────────────────────────────
        path_frame = tk.Frame(dialog, bg=self.COLORS['bg_dark'])
        path_frame.pack(fill='x', padx=20, pady=5)

        tk.Label(path_frame, text="Pattern Model Checkpoint:",
                 font=('Segoe UI', 9),
                 fg=self.COLORS['text'],
                 bg=self.COLORS['bg_dark']).pack(anchor='w')

        saved_path = self.preferences_manager.get(
            'drum_generator_pattern_model_path', None) or ''
        path_var = tk.StringVar(value=saved_path)
        entry_row = tk.Frame(path_frame, bg=self.COLORS['bg_dark'])
        entry_row.pack(fill='x', pady=(2, 0))
        path_entry = tk.Entry(entry_row, textvariable=path_var,
                              font=('Segoe UI', 9),
                              bg=self.COLORS['bg_medium'],
                              fg=self.COLORS['text'],
                              insertbackground=self.COLORS['text'])
        path_entry.pack(side='left', fill='x', expand=True, padx=(0, 4))

        def browse():
            initial_dir = None
            cur = path_var.get()
            if cur:
                d = os.path.dirname(cur)
                if os.path.isdir(d):
                    initial_dir = d
            p = filedialog.askopenfilename(
                parent=dialog,
                title="Select Pattern CVAE Checkpoint",
                filetypes=[("PyTorch Checkpoint", "*.pt"),
                           ("All Files", "*.*")],
                initialdir=initial_dir,
            )
            if p:
                path_var.set(p)

        tk.Button(entry_row, text="Browse...", command=browse,
                  bg=self.COLORS['bg_light'], fg=self.COLORS['text'],
                  font=('Segoe UI', 8), relief='flat',
                  padx=4).pack(side='left')

        def clear_path():
            path_var.set('')

        tk.Button(entry_row, text="Clear", command=clear_path,
                  bg=self.COLORS['bg_light'], fg=self.COLORS['text'],
                  font=('Segoe UI', 8), relief='flat',
                  padx=4).pack(side='left', padx=(2, 0))

        # Fallback info
        fallback_exists = os.path.isfile(_BUNDLED_CHECKPOINT)
        fallback_text = ("Bundled checkpoint will be used as fallback."
                         if fallback_exists
                         else "No bundled checkpoint found.")
        tk.Label(path_frame, text=fallback_text,
                 font=('Segoe UI', 8),
                 fg=self.COLORS['text_dim'],
                 bg=self.COLORS['bg_dark']).pack(anchor='w', pady=(4, 0))

        # ── Pattern temperature ──────────────────────────────────────
        temp_frame = tk.Frame(dialog, bg=self.COLORS['bg_dark'])
        temp_frame.pack(fill='x', padx=20, pady=(10, 5))

        tk.Label(temp_frame, text="Pattern Temperature:",
                 font=('Segoe UI', 9),
                 fg=self.COLORS['text'],
                 bg=self.COLORS['bg_dark']).pack(side='left')

        saved_temp = self.preferences_manager.get(
            'drum_generator_pattern_temperature', 0.7)
        temp_var = tk.DoubleVar(value=saved_temp)
        tk.Spinbox(temp_frame, from_=0.1, to=3.0, increment=0.1,
                   textvariable=temp_var, width=5,
                   font=('Segoe UI', 9),
                   bg=self.COLORS['bg_medium'],
                   fg=self.COLORS['text'],
                   buttonbackground=self.COLORS['bg_light'],
                   insertbackground=self.COLORS['text']).pack(
                       side='left', padx=(4, 0))

        # ── Buttons ──────────────────────────────────────────────────
        btn_frame = tk.Frame(dialog, bg=self.COLORS['bg_dark'])
        btn_frame.pack(fill='x', padx=20, pady=15)

        def apply_and_close():
            new_path = path_var.get().strip() or None
            self.preferences_manager.set(
                'drum_generator_pattern_model_path', new_path)
            self.preferences_manager.set(
                'drum_generator_pattern_temperature', temp_var.get())
            # Invalidate cached generator so next use picks up new path
            if hasattr(self, '_pattern_gen'):
                del self._pattern_gen
            dialog.destroy()

        tk.Button(btn_frame, text="OK", command=apply_and_close,
                  bg=self.COLORS['accent'], fg=self.COLORS['text'],
                  font=('Segoe UI', 9), padx=8).pack(side='right')
        tk.Button(btn_frame, text="Cancel", command=dialog.destroy,
                  bg=self.COLORS['bg_light'], fg=self.COLORS['text'],
                  font=('Segoe UI', 9), padx=8).pack(
                      side='right', padx=(0, 5))

        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - dialog.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")

    def _show_midi_preferences(self):
        """Show MIDI settings dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("MIDI Settings")
        dialog.geometry("400x420")
        dialog.resizable(True, True)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.configure(bg=self.COLORS['bg_dark'])
        
        # Title
        tk.Label(dialog, text="MIDI Input Settings", 
                font=('Segoe UI', 12, 'bold'),
                fg=self.COLORS['text'],
                bg=self.COLORS['bg_dark']).pack(pady=(15, 10))
        
        # MIDI Device selection
        device_frame = tk.Frame(dialog, bg=self.COLORS['bg_dark'])
        device_frame.pack(fill='x', padx=20, pady=5)
        
        tk.Label(device_frame, text="MIDI Input Device:", 
                font=('Segoe UI', 9),
                fg=self.COLORS['text'],
                bg=self.COLORS['bg_dark']).pack(anchor='w')
        
        # The core's last device scan; a rescan below refreshes the list
        available_ports = self.core.describe('midi.device')['labels']
        current_port = self.core.get('midi.device') or "(Auto-detect)"
        
        device_var = tk.StringVar(value=current_port)
        device_options = ["(Auto-detect)"] + available_ports
        device_combo = ttk.Combobox(device_frame, textvariable=device_var, 
                                    values=device_options, width=40, state='readonly')
        device_combo.pack(fill='x', pady=(2, 0))
        
        def connection_status():
            device = self.core.get('midi.device')
            return f"Status: {'Connected to ' + device if device else 'Not connected'}"
        
        # Connection status
        status_text = connection_status()
        status_label = tk.Label(device_frame, text=status_text,
                               font=('Segoe UI', 8),
                               fg=self.COLORS['text_dim'],
                               bg=self.COLORS['bg_dark'])
        status_label.pack(anchor='w', pady=(2, 0))
        
        # Base note selection
        note_frame = tk.Frame(dialog, bg=self.COLORS['bg_dark'])
        note_frame.pack(fill='x', padx=20, pady=15)
        
        tk.Label(note_frame, text="Base Note for Drum Mapping:", 
                font=('Segoe UI', 9),
                fg=self.COLORS['text'],
                bg=self.COLORS['bg_dark']).pack(anchor='w')
        
        current_base = self.core.get('midi.base_note')
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Generate note options (C-1 to C8)
        note_options = []
        for octave in range(-1, 9):
            for i, name in enumerate(note_names):
                midi_note = (octave + 1) * 12 + i
                if 0 <= midi_note <= 120:  # Leave room for 8 channels
                    note_options.append(f"{name}{octave} (note {midi_note})")
        
        # Find current selection
        current_octave = (current_base // 12) - 1
        current_name = note_names[current_base % 12]
        current_note_str = f"{current_name}{current_octave} (note {current_base})"
        
        note_var = tk.StringVar(value=current_note_str)
        note_combo = ttk.Combobox(note_frame, textvariable=note_var,
                                  values=note_options, width=40, state='readonly')
        note_combo.pack(fill='x', pady=(2, 0))
        
        # Show mapping info
        mapping_text = f"Channels 1-8 will respond to notes {current_base} - {current_base + 7}"
        mapping_label = tk.Label(note_frame, text=mapping_text,
                                font=('Segoe UI', 8),
                                fg=self.COLORS['text_dim'],
                                bg=self.COLORS['bg_dark'])
        mapping_label.pack(anchor='w', pady=(2, 0))
        
        def update_mapping_text(*args):
            selection = note_var.get()
            # Extract note number from selection
            try:
                note_num = int(selection.split('note ')[1].rstrip(')'))
                mapping_label.config(text=f"Channels 1-8 will respond to notes {note_num} - {note_num + 7}")
            except (IndexError, ValueError):
                pass
        
        note_var.trace('w', update_mapping_text)
        
        # Clock sync option
        sync_frame = tk.Frame(dialog, bg=self.COLORS['bg_dark'])
        sync_frame.pack(fill='x', padx=20, pady=10)
        
        clock_sync_var = tk.BooleanVar(value=self.core.get('midi.clock_sync'))
        clock_sync_cb = tk.Checkbutton(sync_frame, text="Sync BPM to MIDI Clock", 
                                       variable=clock_sync_var,
                                       font=('Segoe UI', 9),
                                       fg=self.COLORS['text'],
                                       bg=self.COLORS['bg_dark'],
                                       selectcolor=self.COLORS['bg_medium'],
                                       activebackground=self.COLORS['bg_dark'],
                                       activeforeground=self.COLORS['text'])
        clock_sync_cb.pack(anchor='w')
        
        def sync_status_text():
            synced_bpm = self.core.get('midi.synced_tempo')
            return f"Current synced BPM: {synced_bpm:.1f}" if synced_bpm > 0 else "Not receiving MIDI clock"
        
        # Show current synced BPM if available
        sync_status = sync_status_text()
        sync_status_label = tk.Label(sync_frame, text=sync_status,
                                     font=('Segoe UI', 8),
                                     fg=self.COLORS['text_dim'],
                                     bg=self.COLORS['bg_dark'])
        sync_status_label.pack(anchor='w', pady=(2, 0))
        
        # Info section
        info_frame = tk.Frame(dialog, bg=self.COLORS['bg_dark'])
        info_frame.pack(fill='x', padx=20, pady=10)
        
        info_text = """MIDI Control:
• Note On → Trigger drum channels (velocity sensitive)
• Program Change 0-11 → Select patterns A-L
• MIDI Start → Play from beginning
• MIDI Stop → Stop playback
• MIDI Continue → Resume playback
• MIDI Clock → Sync BPM (when enabled)"""
        
        tk.Label(info_frame, text=info_text,
                font=('Segoe UI', 8),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_dark'],
                justify='left').pack(anchor='w')
        
        # Buttons
        button_frame = tk.Frame(dialog, bg=self.COLORS['bg_dark'])
        button_frame.pack(fill='x', padx=20, pady=15)
        
        def apply_settings():
            # Get selected device
            selected_device = device_var.get()
            if selected_device == "(Auto-detect)":
                selected_device = None
            
            # Get selected base note
            selection = note_var.get()
            try:
                base_note = int(selection.split('note ')[1].rstrip(')'))
            except (IndexError, ValueError):
                base_note = 36  # Default to C1
            
            # Get clock sync setting
            clock_sync = clock_sync_var.get()
            
            # Apply settings (the core saves them in the preferences)
            self.core.set('midi.base_note', base_note)
            self.core.set('midi.clock_sync', clock_sync)
            
            def show_status(event=None):
                if not dialog.winfo_exists():
                    return
                if event is not None and event['status'] == 'error':
                    status_label.config(text=f"Status: {event.get('error')}")
                else:
                    status_label.config(text=connection_status())
                sync_status_label.config(text=sync_status_text())
            
            # Reconnect if device changed (the core opens it and saves the choice)
            if selected_device != self.core.get('midi.device'):
                status_label.config(text="Status: connecting...")
                self._when_action_done(self.core.act('midi.open', device=selected_device),
                                       show_status)
            else:
                show_status()
            
            print(f"MIDI settings updated: base note {base_note}, clock sync: {clock_sync}, "
                  f"device: {selected_device or 'auto'}")
        
        def on_ok():
            apply_settings()
            dialog.destroy()
        
        def refresh_devices():
            def show(event):
                if event['status'] == 'done' and device_combo.winfo_exists():
                    device_combo['values'] = ["(Auto-detect)"] + event['result']['devices']
            self._when_action_done(self.core.act('midi.rescan'), show)
        
        refresh_devices()
        
        tk.Button(button_frame, text="Refresh Devices", 
                 command=refresh_devices,
                 bg=self.COLORS['bg_light'],
                 fg=self.COLORS['text']).pack(side='left')
        
        tk.Button(button_frame, text="CC Mappings...", 
                 command=lambda: [dialog.destroy(), self._show_cc_mapping_dialog()],
                 bg=self.COLORS['bg_light'],
                 fg=self.COLORS['text']).pack(side='left', padx=(10, 0))
        
        tk.Button(button_frame, text="Apply", 
                 command=apply_settings,
                 bg=self.COLORS['bg_light'],
                 fg=self.COLORS['text']).pack(side='right', padx=(5, 0))
        
        tk.Button(button_frame, text="OK", 
                 command=on_ok,
                 bg=self.COLORS['accent'],
                 fg=self.COLORS['text']).pack(side='right')
        
        # Center dialog on parent
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - dialog.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")
    
    def _show_cc_mapping_dialog(self):
        """Show MIDI CC mapping configuration dialog (the core's midi.cc_map)"""
        dialog = tk.Toplevel(self.root)
        dialog.title("MIDI CC Mappings")
        dialog.geometry("500x450")
        dialog.resizable(True, True)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.configure(bg=self.COLORS['bg_dark'])
        
        # Title
        tk.Label(dialog, text="MIDI CC Parameter Mappings", 
                font=('Segoe UI', 12, 'bold'),
                fg=self.COLORS['text'],
                bg=self.COLORS['bg_dark']).pack(pady=(15, 5))
        
        tk.Label(dialog, text="Map MIDI Control Change messages to synth parameters", 
                font=('Segoe UI', 8),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_dark']).pack(pady=(0, 10))
        
        # Frame for mappings list
        list_frame = tk.Frame(dialog, bg=self.COLORS['bg_dark'])
        list_frame.pack(fill='both', expand=True, padx=20, pady=5)
        
        # Header
        header = tk.Frame(list_frame, bg=self.COLORS['bg_medium'])
        header.pack(fill='x', pady=(0, 5))
        tk.Label(header, text="CC #", width=15, anchor='w',
                font=('Segoe UI', 9, 'bold'),
                fg=self.COLORS['text'],
                bg=self.COLORS['bg_medium']).pack(side='left', padx=5, pady=3)
        tk.Label(header, text="Parameter", width=25, anchor='w',
                font=('Segoe UI', 9, 'bold'),
                fg=self.COLORS['text'],
                bg=self.COLORS['bg_medium']).pack(side='left', padx=5, pady=3)
        tk.Label(header, text="", width=8,
                bg=self.COLORS['bg_medium']).pack(side='left', padx=5, pady=3)
        
        # Scrollable frame for mappings
        canvas = tk.Canvas(list_frame, bg=self.COLORS['bg_dark'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.COLORS['bg_dark'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Controls are shown by their parameter names; a target no tkinter
        # control offers (mapped by another front-end) is shown as its address
        current_mappings = self.core.get('midi.cc_map')
        target_of = {name: cc_parameter_target(name) for name in self._get_available_parameters()}
        name_of = {target: name for name, target in target_of.items()}
        available_params = sorted(target_of) + sorted(
            {t for t in current_mappings.values() if t not in name_of})
        
        # Any CC 0-127
        cc_options = ["(None)"] + [cc_name(cc) for cc in range(128)]
        
        # Store mapping widgets for later access
        mapping_widgets = []
        
        def create_mapping_row(idx):
            row = tk.Frame(scrollable_frame, bg=self.COLORS['bg_dark'])
            row.pack(fill='x', pady=2)
            
            # CC selector
            cc_var = tk.StringVar(value="(None)")
            cc_combo = ttk.Combobox(row, textvariable=cc_var, values=cc_options, 
                                   width=18, state='readonly')
            cc_combo.pack(side='left', padx=5)
            
            # Parameter selector
            param_var = tk.StringVar(value="(None)")
            param_options = ["(None)"] + available_params
            param_combo = ttk.Combobox(row, textvariable=param_var, values=param_options,
                                      width=25, state='readonly')
            param_combo.pack(side='left', padx=5)
            
            # Clear button
            clear_btn = tk.Button(row, text="Clear", width=6,
                                 bg=self.COLORS['bg_light'],
                                 fg=self.COLORS['text'],
                                 command=lambda: [cc_var.set("(None)"), param_var.set("(None)")])
            clear_btn.pack(side='left', padx=5)
            
            mapping_widgets.append((cc_var, param_var))
            return row
        
        # One row per mapping (no limit), plus empty rows to add more
        for i in range(max(8, len(current_mappings) + 2)):
            create_mapping_row(i)
        
        # Populate existing mappings
        for (cc_num, target), (cc_var, param_var) in zip(sorted(current_mappings.items()),
                                                         mapping_widgets):
            cc_var.set(cc_name(cc_num))
            param_var.set(name_of.get(target, target))
        
        # Info text
        info_frame = tk.Frame(dialog, bg=self.COLORS['bg_dark'])
        info_frame.pack(fill='x', padx=20, pady=10)
        
        tk.Label(info_frame, 
                text="Tip: Right-click any knob in the synth for quick MIDI Learn",
                font=('Segoe UI', 8),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_dark']).pack(anchor='w')
        
        # Buttons
        button_frame = tk.Frame(dialog, bg=self.COLORS['bg_dark'])
        button_frame.pack(fill='x', padx=20, pady=15)
        
        def apply_mappings():
            # The rows replace the whole map
            mapping = {}
            for cc_var, param_var in mapping_widgets:
                cc_str = cc_var.get()
                param = param_var.get()
                
                if cc_str == "(None)" or param == "(None)":
                    continue
                
                # Extract CC number from string like "CC1 (Mod Wheel)"
                try:
                    cc_num = int(cc_str.split('(')[0].replace('CC', '').strip())
                except (ValueError, IndexError):
                    continue
                mapping[cc_num] = target_of.get(param, param)
            
            # The core saves the map in the preferences
            self.core.set('midi.cc_map', mapping)
            print(f"Applied {len(mapping)} CC mappings")
        
        def on_ok():
            apply_mappings()
            dialog.destroy()
        
        def clear_all():
            for cc_var, param_var in mapping_widgets:
                cc_var.set("(None)")
                param_var.set("(None)")
        
        tk.Button(button_frame, text="Clear All", 
                 command=clear_all,
                 bg=self.COLORS['bg_light'],
                 fg=self.COLORS['text']).pack(side='left')
        
        tk.Button(button_frame, text="Apply", 
                 command=apply_mappings,
                 bg=self.COLORS['bg_light'],
                 fg=self.COLORS['text']).pack(side='right', padx=(5, 0))
        
        tk.Button(button_frame, text="OK", 
                 command=on_ok,
                 bg=self.COLORS['accent'],
                 fg=self.COLORS['text']).pack(side='right')
        
        # Center dialog on parent
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - dialog.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")

    def _refresh_preset_list(self):
        """Refresh the preset combo box with files from the preset folder"""
        preset_folder = self.preferences_manager.get_preset_folder()
        
        # Get all preset files from the folder
        preset_files = []
        if os.path.exists(preset_folder):
            for file in os.listdir(preset_folder):
                if file.lower().endswith(('.mtpreset', '.json')):
                    preset_files.append(file)
        
        # Sort alphabetically
        preset_files.sort()
        
        # Update combo box
        self.preset_combo['values'] = preset_files
        
        # Try to select the currently loaded preset in the combo box
        last_preset = self.preferences_manager.get('last_preset')
        if last_preset:
            preset_filename = os.path.basename(last_preset)
            if preset_filename in preset_files:
                self.preset_combo.set(preset_filename)
            else:
                self.preset_combo.set('')
        elif preset_files:
            self.preset_combo.set('')
    
    def _on_preset_combo_select(self, event=None):
        """Handle preset selection from combo box"""
        selected = self.preset_combo.get()
        if selected:
            preset_folder = self.preferences_manager.get_preset_folder()
            filepath = os.path.join(preset_folder, selected)
            if os.path.exists(filepath):
                self._load_preset_file(filepath, show_message=False)
    
    def _load_last_preset(self):
        """Load the last loaded preset if it exists"""
        last_preset = self.preferences_manager.get('last_preset')
        if last_preset and os.path.exists(last_preset):
            try:
                self._load_preset_file(last_preset, show_message=False)
            except Exception as e:
                # Silently fail if last preset can't be loaded
                print(f"Warning: Could not load last preset: {e}")
    
    # ============== UI tick ==============
    
    def _start_ui_update_timer(self):
        """Start timer for UI updates (runs on main thread)"""
        # Build target → widget mapping for modulation visual feedback.
        # Only RotaryKnob / VerticalSlider support set_mod_offset.
        self._mod_target_widget_map = {target.value: widget for target, widget in {
            ModTarget.OSC_FREQUENCY: self.osc_freq_knob,
            ModTarget.PITCH_SEMITONES: self.pitch_knob,
            ModTarget.PITCH_MOD_AMOUNT: self.pitch_amount_knob,
            ModTarget.PITCH_MOD_RATE: self.pitch_rate_knob,
            ModTarget.OSC_ATTACK: self.osc_attack_knob,
            ModTarget.OSC_DECAY: self.osc_decay_knob,
            ModTarget.NOISE_FILTER_FREQ: self.noise_freq_knob,
            ModTarget.NOISE_FILTER_Q: self.noise_q_knob,
            ModTarget.NOISE_ATTACK: self.noise_attack_slider,
            ModTarget.NOISE_DECAY: self.noise_decay_slider,
            ModTarget.LEVEL_DB: self.level_knob,
            ModTarget.PAN: self.pan_knob,
            ModTarget.DISTORTION: self.distort_knob,
            ModTarget.EQ_FREQUENCY: self.eq_freq_knob,
            ModTarget.EQ_GAIN_DB: self.eq_gain_knob,
            ModTarget.VINTAGE_AMOUNT: self.vintage_knob,
            ModTarget.REVERB_DECAY: self.reverb_decay_knob,
            ModTarget.REVERB_MIX: self.reverb_mix_knob,
            ModTarget.REVERB_WIDTH: self.reverb_width_knob,
            ModTarget.DELAY_FEEDBACK: self.delay_feedback_knob,
            ModTarget.DELAY_MIX: self.delay_mix_knob,
        }.items()}
        self._mod_active_targets = set()  # Track which widgets have active mod indicators
        self._ui_update_tick()
    
    def _ui_update_tick(self):
        """Periodic UI update tick - runs on main thread only.

        Reads the transport, play position, modulation readouts and action
        results from the app core's poll; it never touches the audio state.
        """
        state = self.core.poll(self._poll_version)
        self._poll_version = state['version']
        
        if state['changes']:
            self._show_changes(state['changes'])
        
        for event in state['events']:
            if event.get('kind') == 'cc_burst':
                self._push_cc_burst(event)
                continue
            callback = self._action_callbacks.pop(event.get('id'), None)
            if callback is not None:
                callback(event)

        self._show_midi(state['midi'])

        transport = self._transport = state['transport']
        shown = (transport['playing'], transport['selected_pattern'],
                 transport['queued_pattern'], transport['playing_pattern'])
        if shown != self._last_transport:
            previous, self._last_transport = self._last_transport, shown
            if previous is not None:
                self._show_transport(transport, previous)
        if self._dirty_lanes:
            self._show_lanes(sorted(self._dirty_lanes))
        if transport['playing'] and hasattr(self, 'pattern_editors'):
            # The selection follows the playing pattern (chains, queue) in the core
            self._show_playhead(transport)
        
        # Update modulation visual indicators on knobs/sliders
        self._update_mod_indicators(state['modulation']['offsets'])
        
        # Schedule next update (every 50ms to reduce load)
        self.ui_update_timer = self.root.after(50, self._ui_update_tick)
    
    def _show_transport(self, transport, previous):
        """The transport, the selected, queued or playing pattern changed
        (buttons, MIDI, a chain or the queue moving on)."""
        playing = transport['playing']
        if transport['selected_pattern'] != self._pattern:
            self._update_pattern_editors()
        if not playing and hasattr(self, 'pattern_editors'):
            self._show_playhead(transport)
        self._update_pattern_button_states()
        if hasattr(self, 'play_btn') and hasattr(self.play_btn, 'set_active'):
            self.play_btn.set_active(playing)
            self.stop_btn.set_active(not playing)

    def _update_mod_indicators(self, offsets):
        """Show the selected channel's modulation offsets (from poll) on the knobs."""
        # Update widgets that have active modulation
        new_active = set()
        for target, offset in offsets.items():
            widget = self._mod_target_widget_map.get(target)
            if widget is not None and offset != 0.0:
                widget.set_mod_offset(offset)
                new_active.add(target)
        
        # Clear indicators on widgets that are no longer modulated
        for target in self._mod_active_targets - new_active:
            widget = self._mod_target_widget_map.get(target)
            if widget is not None:
                widget.set_mod_offset(0.0)
        
        self._mod_active_targets = new_active
    
    def _update_pattern_button_states(self):
        """Update visual states of pattern buttons (transport from the last poll)"""
        transport = self._transport
        playing = transport['playing']
        selected_idx = self._pattern
        playing_idx = transport['playing_pattern'] if playing else -1
        queued_idx = transport['queued_pattern'] if playing else None
        names = PatternManager.PATTERN_NAMES
        
        for i, btn in enumerate(self.pattern_buttons):
            # Determine background color
            if i == playing_idx and self.button_flash_state:
                # Playing pattern - flash green
                bg_color = self.COLORS['led_on']
            elif queued_idx is not None and i == queued_idx and self.button_flash_state:
                # Queued pattern - flash blue
                bg_color = self.COLORS['highlight']
            elif i == selected_idx:
                # Selected pattern - blue highlight
                bg_color = self.COLORS['highlight']
            else:
                # Normal state
                bg_color = self.COLORS['bg_light']
            
            # Determine text color
            if self.core.get(f'pattern.{names[i]}.empty'):
                # Empty pattern - gray text
                fg_color = self.COLORS['text_dim']
            elif (self.core.get(f'pattern.{names[i]}.chained')
                  or (i > 0 and self.core.get(f'pattern.{names[i - 1]}.chained'))):
                # Chained pattern - blue text
                fg_color = self.COLORS['highlight']
            else:
                # Normal text
                fg_color = self.COLORS['text']
            
            btn.config(bg=bg_color, fg=fg_color)
    
    def _toggle_button_flash(self):
        """Toggle flash state and update buttons (one 250 ms chain, started at init)"""
        if self._transport['playing']:
            self.button_flash_state = not self.button_flash_state
            self._update_pattern_button_states()
        elif self.button_flash_state:
            # Reset flash state when not playing
            self.button_flash_state = False
            self._update_pattern_button_states()
        self.root.after(250, self._toggle_button_flash)
    
    def run(self):
        """Run the application"""
        try:
            self.root.mainloop()
        finally:
            # Stop UI update timer
            if self.ui_update_timer:
                try:
                    self.root.after_cancel(self.ui_update_timer)
                except tk.TclError:
                    pass  # the window is already gone
                self.ui_update_timer = None
            # Stop the audio stream and MIDI input, release the synth
            self.core.close()


def main():
    """Main entry point"""
    app = PythonicGUI()
    app.run()


if __name__ == '__main__':
    main()
