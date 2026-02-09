"""
Main GUI Window for Pythonic
Visual interface closely matching the original application
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import numpy as np
import json
import os
import time
import random
from collections import deque

# Import our synthesizer
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pythonic.synthesizer import PythonicSynthesizer
from pythonic.oscillator import WaveformType, PitchModMode
from pythonic.noise import NoiseFilterMode, NoiseEnvelopeMode
from pythonic.preset_manager import PresetManager
from pythonic.preferences_manager import PreferencesManager
from pythonic.pattern_manager import PatternManager
from gui.widgets import (
    RotaryKnob, VerticalSlider, ChannelButton,
    WaveformSelector, ModeSelector, ToggleButton, PatternEditor, MatrixEditor
)
from gui.po32_transfer import PO32TransferDialog
from gui.po32_import import PO32ImportDialog

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

# Import MIDI input manager
from pythonic.midi_manager import MidiManager

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
    
    def __init__(self):
        # Create main window
        self.root = tk.Tk()
        self.root.title("Pythonic Drum Synthesizer")
        self.root.configure(bg=self.COLORS['bg_dark'])
        self.root.resizable(True, True)
        self.root.minsize(900, 700)  # Minimum window size
        self.root.geometry("1000x780")  # Default window size
        
        # Initialize synthesizer
        self.sample_rate = 44100
        self.synth = PythonicSynthesizer(self.sample_rate)
        
        # Preferences manager
        self.preferences_manager = PreferencesManager()
        
        # Apply saved parameter smoothing time to all channels
        smoothing_ms = self.preferences_manager.get('param_smoothing_ms', 30.0)
        for channel in self.synth.channels:
            channel.set_smoothing_time(smoothing_ms)
        
        # Preset manager
        self.preset_manager = PresetManager(self.synth)
        
        # Pattern manager
        self.pattern_manager = PatternManager(num_channels=8, pattern_length=16)
        
        # MIDI input manager
        self.midi_manager = MidiManager()
        self._midi_activity_time = 0  # For activity indicator
        self._midi_learn_target = None  # Parameter name being learned
        self._init_midi()
        
        # Audio state
        self.audio_stream = None
        self.audio_buffer = np.zeros((2048, 2), dtype=np.float32)
        self.buffer_lock = threading.Lock()
        
        # UI state
        self.selected_channel = 0
        self.updating_ui = False  # Prevent feedback loops
        self.edit_all_mode = False  # When True, knob changes affect all unmuted channels
        
        # Parameter registry for MIDI CC mapping
        # Maps parameter_name -> (widget, min_val, max_val, setter_function)
        self._cc_parameter_registry = {}
        
        # Pitch bend state tracking
        self._pitchbend_original_value = None  # Original param value before pitch bend
        self._pitchbend_active = False  # True when pitch bend is away from center
        self._pitchbend_center_threshold = 0.02  # Consider centered if within this range
        
        # Playback state (thread-safe)
        self.last_triggered_step = -1  # Track last triggered step to avoid double triggers
        self.frames_since_last_step = 0
        self.button_flash_state = False  # For flashing playing pattern button
        self.current_play_position = 0  # Atomic position for UI updates
        self.position_lock = threading.Lock()  # Protect position updates
        self.ui_update_timer = None  # Timer for UI updates
        
        # Fill tracking: list of (frames_until_trigger, channel_id, velocity)
        # Fills are triggered at sub-step intervals within the current step
        self.pending_fills = []
        
        # Performance monitoring
        self.callback_times = deque(maxlen=100)  # Last 100 callback times
        self.trigger_times = deque(maxlen=100)  # Time spent triggering
        self.process_times = deque(maxlen=100)  # Time spent processing audio
        self.underrun_count = 0
        self.callback_count = 0
        self.last_perf_report = time.time()
        
        # Adaptive processing - drop samples to prevent underruns
        self.enable_sample_dropping = True  # Enable adaptive processing
        self.dropped_callback_count = 0
        self._last_good_audio = np.zeros((1050, 2), dtype=np.float32)
        
        # Build the interface
        self._build_ui()
        
        # Register parameters for MIDI CC control
        self._register_cc_parameters()
        
        # Bind keyboard events
        self.root.bind('<Key>', self._on_key_press)
        
        # Start audio if available
        if AUDIO_AVAILABLE:
            self._start_audio()
        
        # Initialize synth BPM from pattern manager (for tempo-synced delay)
        self.synth.set_bpm(self.pattern_manager.bpm)
        
        # Update UI with current channel
        self._update_ui_from_channel()
        
        # Start button state updates
        self.root.after(250, self._toggle_button_flash)
        
        # Start UI position update timer (separate from audio thread)
        self._start_ui_update_timer()
        
        # Load the last preset if available
        self._load_last_preset()
    
    def _build_ui(self):
        """Build the complete user interface
        
        Layout follows this structure (top to bottom):
        1. Toolbar (program selector, morph, undo/redo, options)
        2. Preset Section (preset name, channel buttons 1-8 in 2 rows, mute buttons)
        3. Drum Patch Section (mixing, oscillator, noise, velocity controls)
        4. Pattern Section (pattern buttons A-L, step editor, play controls)
        5. Global Section (transport, swing, fill rate, master volume)
        """
        # Main container - fill entire window
        main_frame = tk.Frame(self.root, bg=self.COLORS['bg_dark'])
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
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
        toolbar = tk.Frame(parent, bg=self.COLORS['bg_medium'], height=45)
        toolbar.pack(fill='x', pady=(0, 5))
        toolbar.pack_propagate(False)
        
        # Left side: Program selector
        program_frame = tk.Frame(toolbar, bg=self.COLORS['bg_medium'])
        program_frame.pack(side='left', padx=5, pady=5)
        
        tk.Label(program_frame, text="program:", 
                font=('Segoe UI', 7), fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack(side='left', padx=(0, 3))
        
        self.program_var = tk.StringVar(value="1")
        self.program_combo = ttk.Combobox(program_frame, textvariable=self.program_var,
                                         values=[str(i) for i in range(1, 17)],
                                         width=3, state='readonly')
        self.program_combo.pack(side='left')
        self.program_combo.bind('<<ComboboxSelected>>', self._on_program_select)
        
        # Center: Sound Morph slider
        morph_frame = tk.Frame(toolbar, bg=self.COLORS['bg_medium'])
        morph_frame.pack(side='left', padx=20, pady=5, expand=True)
        
        tk.Label(morph_frame, text="sound morph:", 
                font=('Segoe UI', 7), fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack(side='left', padx=(0, 5))
        
        self.morph_slider = tk.Scale(morph_frame, from_=0, to=100, 
                                    orient='horizontal', length=150,
                                    bg=self.COLORS['bg_medium'], 
                                    fg=self.COLORS['text'],
                                    highlightthickness=0,
                                    troughcolor=self.COLORS['bg_dark'],
                                    command=self._on_morph_change)
        self.morph_slider.pack(side='left')
        
        # Right side: Undo/Redo, options, master volume
        right_frame = tk.Frame(toolbar, bg=self.COLORS['bg_medium'])
        right_frame.pack(side='right', padx=5, pady=5)
        
        # Undo/Redo buttons
        self.undo_btn = tk.Button(right_frame, text="↶", width=2, height=1,
                                 font=('Segoe UI', 10),
                                 bg=self.COLORS['bg_light'],
                                 fg=self.COLORS['text'],
                                 command=self._on_undo)
        self.undo_btn.pack(side='left', padx=1)
        
        self.redo_btn = tk.Button(right_frame, text="↷", width=2, height=1,
                                 font=('Segoe UI', 10),
                                 bg=self.COLORS['bg_light'],
                                 fg=self.COLORS['text'],
                                 command=self._on_redo)
        self.redo_btn.pack(side='left', padx=1)
        
        # Separator
        tk.Frame(right_frame, width=10, bg=self.COLORS['bg_medium']).pack(side='left')
        
        # Master volume
        tk.Label(right_frame, text="master", 
                font=('Segoe UI', 7), fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack(side='left', padx=(5, 2))
        
        self.master_knob = RotaryKnob(right_frame, size=35, 
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
        preset_section.pack(fill='x', pady=(0, 5))
        
        # Left: Logo/Title
        logo_frame = tk.Frame(preset_section, bg=self.COLORS['bg_medium'])
        logo_frame.pack(side='left', padx=10, pady=5)
        
        tk.Label(logo_frame, text="PYTHONIC", 
                font=('Segoe UI', 14, 'bold'), fg=self.COLORS['accent_light'],
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
        preset_nav_frame.pack(side='left', padx=10, pady=5)
        
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
        channels_container.pack(side='right', padx=10, pady=5)
        
        # Patch name display ABOVE channels
        self.patch_name_label = tk.Label(channels_container,
                                         text="SC BD Schmack",
                                         font=('Segoe UI', 9),
                                         fg=self.COLORS['text'],
                                         bg=self.COLORS['bg_dark'],
                                         width=20, anchor='center',
                                         relief='sunken', padx=5)
        self.patch_name_label.pack(pady=(0, 3))
        
        # Single row of channels 1-8 with number, button, mute
        channels_row = tk.Frame(channels_container, bg=self.COLORS['bg_medium'])
        channels_row.pack()
        
        self.channel_buttons = []
        self.mute_buttons = []
        
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
            
            btn = ChannelButton(btn_row, i, size=28,
                               command=self._on_channel_select)
            btn.pack(side='left')
            self.channel_buttons.append(btn)
            
            # Small mute button
            mute_btn = ToggleButton(btn_row, text="m", width=16, height=16,
                                   command=lambda en, ch=i: self._on_mute_toggle(ch, en))
            mute_btn.pack(side='left', padx=1)
            self.mute_buttons.append(mute_btn)
        
        self.channel_buttons[0].set_selected(True)
    
    def _build_pattern_section(self, parent):
        """Build the pattern editor section
        Left side: Pattern buttons (A-L), matrix toggle, chain controls
        Right side: Pattern editor (trig/acc/fill/len lanes)
        """
        pattern_frame = tk.LabelFrame(parent, text="pattern", 
                                     font=('Segoe UI', 8),
                                     fg=self.COLORS['text_dim'],
                                     bg=self.COLORS['bg_medium'],
                                     labelanchor='nw')
        pattern_frame.pack(fill='x', pady=(0, 5))
        
        # Main horizontal layout: controls on left, editor on right
        main_row = tk.Frame(pattern_frame, bg=self.COLORS['bg_medium'])
        main_row.pack(fill='both', expand=True, padx=5, pady=5)
        
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
        patch_frame.pack(fill='x', expand=True, pady=(0, 5))
        
        # Four main subsections
        self._build_mixing_section(patch_frame)
        self._build_oscillator_section(patch_frame)
        self._build_noise_section(patch_frame)
        self._build_velocity_section(patch_frame)
    
    def _build_mixing_section(self, parent):
        """Build the mixing controls section"""
        section = tk.LabelFrame(parent, text="mixing", 
                               font=('Segoe UI', 8),
                               fg=self.COLORS['text_dim'],
                               bg=self.COLORS['bg_medium'],
                               labelanchor='n')
        section.pack(side='left', padx=5, pady=5, fill='both', expand=True)
        
        # Row 1: osc/noise HORIZONTAL mix slider
        mix_row = tk.Frame(section, bg=self.COLORS['bg_medium'])
        mix_row.pack(pady=5, fill='x')
        
        tk.Label(mix_row, text="osc", font=('Segoe UI', 7),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack(side='left', padx=2)
        
        # Horizontal mix slider
        self.mix_slider = tk.Scale(mix_row, from_=0, to=100, 
                                  orient='horizontal', length=80,
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
        freq_row.pack(pady=3)
        
        tk.Label(freq_row, text="20Hz", font=('Segoe UI', 6),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack(side='left')
        
        self.eq_freq_knob = RotaryKnob(freq_row, size=45,
                                       min_val=20, max_val=20000, default=632,
                                       label="eq freq",
                                       logarithmic=True,
                                       command=self._on_eq_freq_change)
        self.eq_freq_knob.pack(side='left', padx=3)
        
        tk.Label(freq_row, text="20kHz", font=('Segoe UI', 6),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack(side='left')
        
        # Edit All button
        self.edit_all_btn = ToggleButton(section, text="edit all", width=50, height=18,
                                        command=self._on_edit_all_toggle)
        self.edit_all_btn.pack(pady=3)
        
        # Row 3: Distortion and EQ Gain
        row3 = tk.Frame(section, bg=self.COLORS['bg_medium'])
        row3.pack(pady=3)
        
        self.distort_knob = RotaryKnob(row3, size=42,
                                       min_val=0, max_val=100, default=0,
                                       label="distort",
                                       command=self._on_distort_change)
        self.distort_knob.pack(side='left', padx=3)
        
        self.eq_gain_knob = RotaryKnob(row3, size=42,
                                       min_val=-40, max_val=40, default=0,
                                       label="eq gain",
                                       command=self._on_eq_gain_change)
        self.eq_gain_knob.pack(side='left', padx=3)
        
        # Row 3b: Vintage (analog simulation)
        row3b = tk.Frame(section, bg=self.COLORS['bg_medium'])
        row3b.pack(pady=3)
        
        self.vintage_knob = RotaryKnob(row3b, size=42,
                                       min_val=0, max_val=100, default=0,
                                       label="vintage",
                                       command=self._on_vintage_change)
        self.vintage_knob.pack(side='left', padx=3)
        
        # Row 3c: Reverb controls (per-channel stereo reverb/decay effect)
        reverb_row = tk.Frame(section, bg=self.COLORS['bg_medium'])
        reverb_row.pack(pady=3)
        
        self.reverb_decay_knob = RotaryKnob(reverb_row, size=38,
                                            min_val=0, max_val=100, default=0,
                                            label="rvb time",
                                            command=self._on_reverb_decay_change)
        self.reverb_decay_knob.pack(side='left', padx=2)
        
        self.reverb_mix_knob = RotaryKnob(reverb_row, size=38,
                                          min_val=0, max_val=100, default=0,
                                          label="rvb mix",
                                          command=self._on_reverb_mix_change)
        self.reverb_mix_knob.pack(side='left', padx=2)
        
        self.reverb_width_knob = RotaryKnob(reverb_row, size=38,
                                            min_val=0, max_val=200, default=100,
                                            label="rvb wide",
                                            command=self._on_reverb_width_change)
        self.reverb_width_knob.pack(side='left', padx=2)
        
        # Row 3d: Delay/Echo controls (tempo-synced)
        delay_row = tk.Frame(section, bg=self.COLORS['bg_medium'])
        delay_row.pack(pady=3)
        
        # Delay time selector (musical divisions)
        self.delay_time_options = ['1/4', '1/8', '1/16', '1/8T', '1/4.']
        self.delay_time_indices = [2, 3, 4, 8, 11]  # Map to DelayTime enum values
        self.delay_time_selector = ModeSelector(delay_row, 
                                                options=self.delay_time_options,
                                                width=120,
                                                command=self._on_delay_time_change)
        self.delay_time_selector.pack(side='left', padx=2)
        
        delay_knobs_row = tk.Frame(section, bg=self.COLORS['bg_medium'])
        delay_knobs_row.pack(pady=2)
        
        self.delay_feedback_knob = RotaryKnob(delay_knobs_row, size=36,
                                              min_val=0, max_val=95, default=30,
                                              label="dly fdbk",
                                              command=self._on_delay_feedback_change)
        self.delay_feedback_knob.pack(side='left', padx=2)
        
        self.delay_mix_knob = RotaryKnob(delay_knobs_row, size=36,
                                         min_val=0, max_val=100, default=0,
                                         label="dly mix",
                                         command=self._on_delay_mix_change)
        self.delay_mix_knob.pack(side='left', padx=2)
        
        self.delay_pingpong_btn = ToggleButton(delay_knobs_row, text="P.P", 
                                               width=32, height=20,
                                               command=self._on_delay_pingpong_toggle)
        self.delay_pingpong_btn.pack(side='left', padx=2)
        
        # Row 4: Level and Pan
        row4 = tk.Frame(section, bg=self.COLORS['bg_medium'])
        row4.pack(pady=3)
        
        self.level_knob = RotaryKnob(row4, size=42,
                                     min_val=-60, max_val=10, default=0,
                                     label="level",
                                     command=self._on_level_change)
        self.level_knob.pack(side='left', padx=3)
        
        self.pan_knob = RotaryKnob(row4, size=42,
                                   min_val=-100, max_val=100, default=0,
                                   label="pan",
                                   command=self._on_pan_change)
        self.pan_knob.pack(side='left', padx=3)
        
        # Row 5: Choke and Output A/B
        row5 = tk.Frame(section, bg=self.COLORS['bg_medium'])
        row5.pack(pady=3)
        
        self.choke_btn = ToggleButton(row5, text="choke", width=45, height=20,
                                     command=self._on_choke_toggle)
        self.choke_btn.pack(side='left', padx=3)
        
        self.output_selector = ModeSelector(row5, options=['A', 'B'], width=50,
                                           command=self._on_output_change)
        self.output_selector.pack(side='left', padx=3)
    
    def _build_oscillator_section(self, parent):
        """Build the oscillator controls section"""
        section = tk.LabelFrame(parent, text="oscillator",
                               font=('Segoe UI', 8),
                               fg=self.COLORS['text_dim'],
                               bg=self.COLORS['bg_medium'],
                               labelanchor='n')
        section.pack(side='left', padx=5, pady=5, fill='both', expand=True)
        
        # Waveform selector with label
        waveform_frame = tk.Frame(section, bg=self.COLORS['bg_medium'])
        waveform_frame.pack(pady=3)
        
        tk.Label(waveform_frame, text="waveform", font=('Segoe UI', 7),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack()
        
        self.waveform_selector = WaveformSelector(waveform_frame,
                                                  command=self._on_waveform_change)
        self.waveform_selector.pack()
        
        # Oscillator Frequency with labels
        freq_frame = tk.Frame(section, bg=self.COLORS['bg_medium'])
        freq_frame.pack(pady=3)
        
        tk.Label(freq_frame, text="20Hz", font=('Segoe UI', 6),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack(side='left')
        
        self.osc_freq_knob = RotaryKnob(freq_frame, size=50,
                                        min_val=20, max_val=20000, default=440,
                                        label="osc freq",
                                        logarithmic=True,
                                        command=self._on_osc_freq_change)
        self.osc_freq_knob.pack(side='left', padx=3)
        
        tk.Label(freq_frame, text="20kHz", font=('Segoe UI', 6),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack(side='left')
        
        # Pitch modulation mode
        pitch_mod_frame = tk.Frame(section, bg=self.COLORS['bg_medium'])
        pitch_mod_frame.pack(pady=3)
        
        tk.Label(pitch_mod_frame, text="pitch mod", font=('Segoe UI', 7),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack()
        
        self.pitch_mod_mode = ModeSelector(pitch_mod_frame, 
                                          options=['Decay', 'Sine', 'Rand'],
                                          command=self._on_pitch_mod_mode_change)
        self.pitch_mod_mode.pack()
        
        # Pitch mod amount and rate
        pitch_knobs_frame = tk.Frame(section, bg=self.COLORS['bg_medium'])
        pitch_knobs_frame.pack(pady=3)
        
        self.pitch_amount_knob = RotaryKnob(pitch_knobs_frame, size=42,
                                            min_val=-120, max_val=120, default=0,
                                            label="amount",
                                            command=self._on_pitch_amount_change)
        self.pitch_amount_knob.pack(side='left', padx=3)
        
        self.pitch_rate_knob = RotaryKnob(pitch_knobs_frame, size=42,
                                          min_val=1, max_val=2000, default=100,
                                          label="rate",
                                          logarithmic=True,
                                          command=self._on_pitch_rate_change)
        self.pitch_rate_knob.pack(side='left', padx=3)
        
        # Attack and Decay knobs
        env_frame = tk.Frame(section, bg=self.COLORS['bg_medium'])
        env_frame.pack(pady=3)
        
        self.osc_attack_knob = RotaryKnob(env_frame, size=42,
                                          min_val=0, max_val=10000, default=0,
                                          label="attack",
                                          logarithmic=True,
                                          command=self._on_osc_attack_change)
        self.osc_attack_knob.pack(side='left', padx=3)
        
        self.osc_decay_knob = RotaryKnob(env_frame, size=42,
                                         min_val=10, max_val=10000, default=316,
                                         label="decay",
                                         logarithmic=True,
                                         command=self._on_osc_decay_change)
        self.osc_decay_knob.pack(side='left', padx=3)
    
    def _build_noise_section(self, parent):
        """Build the noise generator controls section"""
        section = tk.LabelFrame(parent, text="noise",
                               font=('Segoe UI', 8),
                               fg=self.COLORS['text_dim'],
                               bg=self.COLORS['bg_medium'],
                               labelanchor='n')
        section.pack(side='left', padx=5, pady=5, fill='both', expand=True)
        
        # Filter mode selector (LP/BP/HP)
        filter_mode_frame = tk.Frame(section, bg=self.COLORS['bg_medium'])
        filter_mode_frame.pack(pady=3)
        
        tk.Label(filter_mode_frame, text="filter mode", font=('Segoe UI', 7),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack()
        
        self.noise_filter_mode = ModeSelector(filter_mode_frame,
                                             options=['LP', 'BP', 'HP'],
                                             command=self._on_noise_filter_mode_change)
        self.noise_filter_mode.pack()
        
        # Filter freq with frequency labels
        freq_frame = tk.Frame(section, bg=self.COLORS['bg_medium'])
        freq_frame.pack(pady=3)
        
        tk.Label(freq_frame, text="20Hz", font=('Segoe UI', 6),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack(side='left')
        
        self.noise_freq_knob = RotaryKnob(freq_frame, size=50,
                                          min_val=20, max_val=20000, default=20000,
                                          label="filter freq",
                                          logarithmic=True,
                                          command=self._on_noise_freq_change)
        self.noise_freq_knob.pack(side='left', padx=3)
        
        tk.Label(freq_frame, text="20kHz", font=('Segoe UI', 6),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack(side='left')
        
        # Filter Q knob
        q_frame = tk.Frame(section, bg=self.COLORS['bg_medium'])
        q_frame.pack(pady=3)
        
        self.noise_q_knob = RotaryKnob(q_frame, size=42,
                                       min_val=0.5, max_val=20.0, default=0.707,
                                       label="filter q",
                                       logarithmic=True,
                                       command=self._on_noise_q_change)
        self.noise_q_knob.pack(side='left', padx=3)
        
        # Stereo toggle button
        self.stereo_btn = ToggleButton(q_frame, text="stereo", width=45, height=20,
                                      command=self._on_stereo_toggle)
        self.stereo_btn.pack(side='left', padx=3)
        
        # Envelope mode selector
        env_mode_frame = tk.Frame(section, bg=self.COLORS['bg_medium'])
        env_mode_frame.pack(pady=3)
        
        tk.Label(env_mode_frame, text="envelope", font=('Segoe UI', 7),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack()
        
        self.noise_env_mode = ModeSelector(env_mode_frame,
                                          options=['Exp', 'Lin', 'Mod'],
                                          command=self._on_noise_env_mode_change)
        self.noise_env_mode.pack()
        
        # Attack and Decay as VERTICAL SLIDERS
        env_sliders_frame = tk.Frame(section, bg=self.COLORS['bg_medium'])
        env_sliders_frame.pack(pady=3)
        
        self.noise_attack_slider = VerticalSlider(env_sliders_frame, width=25, height=70,
                                                 min_val=0, max_val=10000, default=0,
                                                 label="attack",
                                                 logarithmic=True,
                                                 command=self._on_noise_attack_change)
        self.noise_attack_slider.pack(side='left', padx=5)
        
        self.noise_decay_slider = VerticalSlider(env_sliders_frame, width=25, height=70,
                                                min_val=10, max_val=10000, default=316,
                                                label="decay",
                                                logarithmic=True,
                                                command=self._on_noise_decay_change)
        self.noise_decay_slider.pack(side='left', padx=5)
    
    def _build_velocity_section(self, parent):
        """Build the velocity sensitivity section"""
        section = tk.LabelFrame(parent, text="vel",
                               font=('Segoe UI', 8),
                               fg=self.COLORS['text_dim'],
                               bg=self.COLORS['bg_medium'],
                               labelanchor='n')
        section.pack(side='left', padx=5, pady=5, fill='both', expand=True)
        
        # Oscillator velocity
        self.osc_vel_slider = VerticalSlider(section, width=25, height=80,
                                            min_val=0, max_val=200, default=0,
                                            label="osc",
                                            command=self._on_osc_vel_change)
        self.osc_vel_slider.pack(pady=5)
        
        # Noise velocity
        self.noise_vel_slider = VerticalSlider(section, width=25, height=80,
                                              min_val=0, max_val=200, default=0,
                                              label="noise",
                                              command=self._on_noise_vel_change)
        self.noise_vel_slider.pack(pady=5)
        
        # Mod velocity
        self.mod_vel_slider = VerticalSlider(section, width=25, height=80,
                                            min_val=0, max_val=200, default=0,
                                            label="mod",
                                            command=self._on_mod_vel_change)
        self.mod_vel_slider.pack(pady=5)
        
        # Probability (0-100%, controls chance of triggering during pattern playback)
        self.prob_slider = VerticalSlider(section, width=25, height=80,
                                         min_val=0, max_val=100, default=100,
                                         label="prob",
                                         command=self._on_probability_change)
        self.prob_slider.pack(pady=5)
    
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
        
        self.bpm_var = tk.StringVar(value=str(self.pattern_manager.bpm))
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
        rate_frame.pack(side='left', padx=15, pady=8)
        
        self.step_rate_buttons = []
        for rate in ['1/8', '1/8T', '1/16', '1/16T', '1/32']:
            is_selected = (rate == self.pattern_manager.step_rate)
            btn = tk.Button(rate_frame, text=rate, width=4, height=1,
                           font=('Segoe UI', 7),
                           bg=self.COLORS['highlight'] if is_selected else self.COLORS['bg_light'],
                           fg=self.COLORS['text'],
                           command=lambda r=rate: self._on_step_rate_button(r))
            btn.pack(side='left', padx=1)
            self.step_rate_buttons.append((rate, btn))
        
        # Swing slider (center)
        swing_frame = tk.Frame(global_frame, bg=self.COLORS['bg_medium'])
        swing_frame.pack(side='left', padx=15, pady=8)
        
        tk.Label(swing_frame, text="0%", 
                font=('Segoe UI', 7), fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack(side='left')
        
        self.global_swing_slider = tk.Scale(swing_frame, from_=0, to=100, 
                                           orient='horizontal', length=100,
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
        fill_frame.pack(side='left', padx=15, pady=8)
        
        tk.Label(fill_frame, text="fill rate", 
                font=('Segoe UI', 7), fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack(side='left', padx=(0, 5))
        
        self.fill_rate_buttons = []
        for rate in ['2x', '3x', '4x', '5x', '6x', '7x', '8x']:
            rate_val = int(rate[0])
            is_selected = (rate_val == self.pattern_manager.fill_rate)
            btn = tk.Button(fill_frame, text=rate, width=2, height=1,
                           font=('Segoe UI', 7),
                           bg=self.COLORS['highlight'] if is_selected else self.COLORS['bg_light'],
                           fg=self.COLORS['text'],
                           command=lambda r=rate_val: self._on_fill_rate_button(r))
            btn.pack(side='left', padx=1)
            self.fill_rate_buttons.append((rate_val, btn))
        
        # Right: Keyboard hint and trigger info
        info_frame = tk.Frame(global_frame, bg=self.COLORS['bg_medium'])
        info_frame.pack(side='right', padx=10, pady=8)
        
        tk.Label(info_frame, 
                text="Keys 1-8: Trigger channels",
                font=('Segoe UI', 7),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack()
        
        # Bind keyboard
        self.root.bind('<Key>', self._on_key_press)
    
    def _on_step_rate_button(self, rate):
        """Handle step rate button click"""
        self.pattern_manager.set_step_rate(rate)
        # Update button states
        for r, btn in self.step_rate_buttons:
            if r == rate:
                btn.config(bg=self.COLORS['highlight'])
            else:
                btn.config(bg=self.COLORS['bg_light'])
    
    def _on_fill_rate_button(self, rate):
        """Handle fill rate button click"""
        self.pattern_manager.set_fill_rate(rate)
        # Update button states
        for r, btn in self.fill_rate_buttons:
            if r == rate:
                btn.config(bg=self.COLORS['highlight'])
            else:
                btn.config(bg=self.COLORS['bg_light'])
    
    def _on_global_swing_change(self, value):
        """Handle global swing slider change"""
        swing_val = int(value)
        self.pattern_manager.swing = swing_val / 100.0  # Convert to 0-1 range
    
    def _on_bpm_change(self, event=None):
        """Handle BPM entry change"""
        try:
            bpm = int(self.bpm_var.get())
            bpm = max(1, min(300, bpm))  # Clamp to valid range
            self.pattern_manager.set_bpm(bpm)
            self.synth.set_bpm(bpm)  # Update synth for tempo-synced delay
            self.bpm_var.set(str(bpm))  # Update display with clamped value
        except ValueError:
            # Restore current BPM if invalid input
            self.bpm_var.set(str(self.pattern_manager.bpm))
    
    # ============== Event Handlers ==============
    
    def _on_program_select(self, event=None):
        """Handle program selection (1-16 slots)"""
        try:
            program_num = int(self.program_var.get())
            # Future: implement program bank switching
            # For now, just log the selection
            print(f"Selected program {program_num}")
        except ValueError:
            pass
    
    def _on_morph_change(self, value):
        """Handle sound morph slider change
        
        Sound morph interpolates all drum patch parameters
        between two end-points using this single slider.
        """
        morph_value = float(value) / 100.0  # Normalize to 0-1
        # Future: implement morph interpolation between two preset endpoints
        # For now, this is a placeholder
        pass
    
    def _on_undo(self):
        """Handle undo button click"""
        # Future: implement undo stack
        print("Undo clicked")
    
    def _on_redo(self):
        """Handle redo button click"""
        # Future: implement redo stack
        print("Redo clicked")
    
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
        menu.add_separator()
        menu.add_command(label="Cut Preset", command=self._cut_preset)
        menu.add_command(label="Copy Preset", command=self._copy_preset)
        menu.add_command(label="Paste Preset", command=self._paste_preset)
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
        menu.add_command(label="Audio Settings...", command=self._show_audio_preferences)
        menu.add_command(label="MIDI Settings...", command=self._show_midi_preferences)
        menu.add_command(label="Synthesis Settings...", command=self._show_synthesis_preferences)
        
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
        self._preset_clipboard = self.preset_manager.export_preset_to_dict()
    
    def _paste_preset(self):
        """Paste preset from clipboard"""
        if hasattr(self, '_preset_clipboard') and self._preset_clipboard:
            self.preset_manager.import_preset_from_dict(self._preset_clipboard)
            self._update_ui_from_channel()
    
    def _init_preset(self):
        """Initialize/reset preset to defaults"""
        for channel in self.synth.channels:
            channel.reset_to_defaults()
        self.pattern_manager.reset_all_patterns()
        self._update_ui_from_channel()
        self._update_pattern_editors()
    
    def _randomize_all(self):
        """Randomize all drum patches and patterns"""
        import random
        for channel in self.synth.channels:
            channel.randomize()
        self.pattern_manager.randomize_pattern(self.pattern_manager.selected_pattern_index)
        self._update_ui_from_channel()
        self._update_pattern_editors()
    
    def _on_channel_select(self, channel_idx, event=None):
        """Handle channel selection
        
        If clicking on an already-selected channel, trigger the drum to preview it.
        Hold Ctrl for accented trigger (velocity 127), otherwise normal (velocity 64).
        """
        # Check if clicking on already-selected channel -> trigger preview
        if channel_idx == self.selected_channel:
            # Trigger the drum to preview
            # Check if Ctrl is held for accented trigger
            if event and (event.state & 0x4):  # Control key mask
                velocity = 127  # Accented
            else:
                velocity = 64   # Normal
            self.synth.trigger_drum(channel_idx, velocity)
            # Flash the button
            self.channel_buttons[channel_idx].set_triggered(True)
            self.root.after(100, lambda: self.channel_buttons[channel_idx].set_triggered(False))
            return
        
        # Update button states
        for i, btn in enumerate(self.channel_buttons):
            btn.set_selected(i == channel_idx)
        
        self.selected_channel = channel_idx
        self.synth.select_channel(channel_idx)
        
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
        """Handle mute toggle"""
        self.synth.mute_channel(channel_idx, enabled)
        self.channel_buttons[channel_idx].set_muted(enabled)
    
    def _on_master_volume_change(self, value):
        """Handle master volume change"""
        self.synth.set_master_volume(value)
    
    def _on_mix_change(self, value):
        """Handle osc/noise mix change (value comes from tk.Scale as string)"""
        if not self.updating_ui:
            channel = self.synth.get_selected_channel()
            channel.osc_noise_mix = float(value) / 100.0
    
    def _on_eq_freq_change(self, value):
        """Handle EQ frequency change"""
        if not self.updating_ui:
            channel = self.synth.get_selected_channel()
            channel.eq_frequency = value
    
    def _on_eq_gain_change(self, value):
        """Handle EQ gain change"""
        if not self.updating_ui:
            channel = self.synth.get_selected_channel()
            channel.eq_gain_db = value
    
    def _on_distort_change(self, value):
        """Handle distortion change"""
        if not self.updating_ui:
            channel = self.synth.get_selected_channel()
            channel.distortion = value / 100.0
    
    def _on_vintage_change(self, value):
        """Handle vintage analog simulation change
        
        The vintage knob simulates analog circuit behavior:
        - Oscillator pitch drift (random walk instability)
        - Thermal noise floor
        - Soft saturation for harmonic warmth
        - High-frequency roll-off (capacitor loading)
        """
        if not self.updating_ui:
            channel = self.synth.get_selected_channel()
            channel.vintage_amount = value / 100.0
    
    def _on_reverb_decay_change(self, value):
        """Handle reverb decay time change
        
        Controls the RT60 (time for reverb to decay by 60dB).
        Range: 0-100% maps to approximately 0.1s to 4s decay time.
        """
        if not self.updating_ui:
            if self.edit_all_mode:
                for ch in self.synth.channels:
                    if not ch.muted:
                        ch.reverb_decay = value / 100.0
            else:
                channel = self.synth.get_selected_channel()
                channel.reverb_decay = value / 100.0
    
    def _on_reverb_mix_change(self, value):
        """Handle reverb dry/wet mix change
        
        Controls the balance between dry (original) and wet (reverb) signal.
        0% = fully dry, 100% = fully wet.
        """
        if not self.updating_ui:
            if self.edit_all_mode:
                for ch in self.synth.channels:
                    if not ch.muted:
                        ch.reverb_mix = value / 100.0
            else:
                channel = self.synth.get_selected_channel()
                channel.reverb_mix = value / 100.0
    
    def _on_reverb_width_change(self, value):
        """Handle reverb stereo width change
        
        Controls the stereo width of the reverb effect.
        0% = mono reverb, 100% = normal stereo, 200% = extra wide.
        """
        if not self.updating_ui:
            if self.edit_all_mode:
                for ch in self.synth.channels:
                    if not ch.muted:
                        ch.reverb_width = value / 100.0
            else:
                channel = self.synth.get_selected_channel()
                channel.reverb_width = value / 100.0
    
    def _on_delay_time_change(self, index):
        """Handle delay time selector change
        
        Sets the tempo-synced delay time (1/4, 1/8, 1/16, triplets, dotted).
        """
        if not self.updating_ui:
            from pythonic.delay import DelayTime
            delay_time_value = self.delay_time_indices[index]
            if self.edit_all_mode:
                for ch in self.synth.channels:
                    if not ch.muted:
                        ch.delay_time = DelayTime(delay_time_value)
            else:
                channel = self.synth.get_selected_channel()
                channel.delay_time = DelayTime(delay_time_value)
    
    def _on_delay_feedback_change(self, value):
        """Handle delay feedback change
        
        Controls how many echoes/repeats occur.
        0% = single echo, 95% = many repeating echoes.
        """
        if not self.updating_ui:
            if self.edit_all_mode:
                for ch in self.synth.channels:
                    if not ch.muted:
                        ch.delay_feedback = value / 100.0
            else:
                channel = self.synth.get_selected_channel()
                channel.delay_feedback = value / 100.0
    
    def _on_delay_mix_change(self, value):
        """Handle delay dry/wet mix change
        
        Controls the balance between dry (original) and wet (delayed) signal.
        0% = no delay, 100% = full delay effect.
        """
        if not self.updating_ui:
            if self.edit_all_mode:
                for ch in self.synth.channels:
                    if not ch.muted:
                        ch.delay_mix = value / 100.0
            else:
                channel = self.synth.get_selected_channel()
                channel.delay_mix = value / 100.0
    
    def _on_delay_pingpong_toggle(self, enabled):
        """Handle delay ping-pong mode toggle
        
        When enabled, echoes alternate between left and right channels.
        """
        if not self.updating_ui:
            if self.edit_all_mode:
                for ch in self.synth.channels:
                    if not ch.muted:
                        ch.delay_ping_pong = enabled
            else:
                channel = self.synth.get_selected_channel()
                channel.delay_ping_pong = enabled
    
    def _on_level_change(self, value):
        """Handle level change"""
        if not self.updating_ui:
            channel = self.synth.get_selected_channel()
            channel.level_db = value
    
    def _on_pan_change(self, value):
        """Handle pan change"""
        if not self.updating_ui:
            channel = self.synth.get_selected_channel()
            channel.pan = value
    
    def _on_edit_all_toggle(self, enabled):
        """Handle edit all toggle - when enabled, changes affect all unmuted channels"""
        self.edit_all_mode = enabled
    
    def _on_choke_toggle(self, enabled):
        """Handle choke toggle"""
        if not self.updating_ui:
            channel = self.synth.get_selected_channel()
            channel.choke_enabled = enabled
    
    def _on_output_change(self, value):
        """Handle output selector change"""
        if not self.updating_ui:
            channel = self.synth.get_selected_channel()
            channel.output_pair = 'A' if value == 0 else 'B'
    
    def _on_waveform_change(self, value):
        """Handle waveform change"""
        if not self.updating_ui:
            channel = self.synth.get_selected_channel()
            channel.set_osc_waveform(WaveformType(value))
    
    def _on_osc_freq_change(self, value):
        """Handle oscillator frequency change"""
        if not self.updating_ui:
            channel = self.synth.get_selected_channel()
            channel.set_osc_frequency(value)
    
    def _on_pitch_mod_mode_change(self, value):
        """Handle pitch mod mode change"""
        if not self.updating_ui:
            channel = self.synth.get_selected_channel()
            channel.set_pitch_mod_mode(PitchModMode(value))
    
    def _on_pitch_amount_change(self, value):
        """Handle pitch mod amount change"""
        if not self.updating_ui:
            channel = self.synth.get_selected_channel()
            channel.set_pitch_mod_amount(value)
    
    def _on_pitch_rate_change(self, value):
        """Handle pitch mod rate change"""
        if not self.updating_ui:
            channel = self.synth.get_selected_channel()
            channel.set_pitch_mod_rate(value)
    
    def _on_osc_attack_change(self, value):
        """Handle oscillator attack change"""
        if not self.updating_ui:
            channel = self.synth.get_selected_channel()
            channel.set_osc_attack(value)
    
    def _on_osc_decay_change(self, value):
        """Handle oscillator decay change"""
        if not self.updating_ui:
            channel = self.synth.get_selected_channel()
            channel.set_osc_decay(value)
    
    def _on_noise_filter_mode_change(self, value):
        """Handle noise filter mode change"""
        if not self.updating_ui:
            channel = self.synth.get_selected_channel()
            channel.set_noise_filter_mode(NoiseFilterMode(value))
    
    def _on_noise_freq_change(self, value):
        """Handle noise filter frequency change"""
        if not self.updating_ui:
            channel = self.synth.get_selected_channel()
            channel.set_noise_filter_freq(value)
    
    def _on_noise_q_change(self, value):
        """Handle noise filter Q change"""
        if not self.updating_ui:
            channel = self.synth.get_selected_channel()
            channel.set_noise_filter_q(value)
    
    def _on_stereo_toggle(self, enabled):
        """Handle stereo toggle"""
        if not self.updating_ui:
            channel = self.synth.get_selected_channel()
            channel.set_noise_stereo(enabled)
    
    def _on_noise_env_mode_change(self, value):
        """Handle noise envelope mode change"""
        if not self.updating_ui:
            channel = self.synth.get_selected_channel()
            channel.set_noise_envelope_mode(NoiseEnvelopeMode(value))
    
    def _on_noise_attack_change(self, value):
        """Handle noise attack change"""
        if not self.updating_ui:
            channel = self.synth.get_selected_channel()
            channel.set_noise_attack(value)
    
    def _on_noise_decay_change(self, value):
        """Handle noise decay change"""
        if not self.updating_ui:
            channel = self.synth.get_selected_channel()
            channel.set_noise_decay(value)
    
    def _on_osc_vel_change(self, value):
        """Handle oscillator velocity sensitivity change"""
        if not self.updating_ui:
            channel = self.synth.get_selected_channel()
            channel.osc_vel_sensitivity = value / 100.0
    
    def _on_noise_vel_change(self, value):
        """Handle noise velocity sensitivity change"""
        if not self.updating_ui:
            channel = self.synth.get_selected_channel()
            channel.noise_vel_sensitivity = value / 100.0
    
    def _on_mod_vel_change(self, value):
        """Handle mod velocity sensitivity change"""
        if not self.updating_ui:
            channel = self.synth.get_selected_channel()
            channel.mod_vel_sensitivity = value / 100.0
    
    def _on_probability_change(self, value):
        """Handle probability change (0-100%)"""
        if not self.updating_ui:
            channel = self.synth.get_selected_channel()
            channel.probability = int(value)
    
    # ============ Pattern Callbacks ============
    
    def _on_pattern_select(self, pattern_index):
        """Handle pattern selection"""
        self.pattern_manager.select_pattern(pattern_index)
        
        # Reset playback position to beginning when switching patterns
        self.pattern_manager.play_position = 0
        self.pattern_manager.current_step = 0
        self.frames_since_last_step = 0
        
        # Update button states
        self._update_pattern_button_states()
        
        # Update editors and show position at beginning
        self._update_pattern_editors()
        if hasattr(self, 'pattern_editors'):
            for editor in self.pattern_editors:
                editor.set_current_position(0)
    
    def _on_pattern_edit(self, channel_id, step, lane_type, value):
        """Handle pattern editor edits"""
        pattern = self.pattern_manager.get_selected_pattern()
        channel = pattern.get_channel(channel_id)
        
        if lane_type == 'trig':
            channel.set_trigger(step, value)
        elif lane_type == 'acc':
            channel.set_accent(step, value)
        elif lane_type == 'fill':
            channel.set_fill(step, value)
        elif lane_type == 'prob':
            channel.set_probability(step, value)
        elif lane_type == 'sub':
            # Update substeps pattern for the step
            if step < len(channel.steps):
                channel.steps[step].substeps = value
    
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
        """Handle pattern edit applied to all unmuted channels (Shift+Click)"""
        pattern = self.pattern_manager.get_selected_pattern()
        
        for ch_idx in range(8):
            if ch_idx not in muted_channels:
                channel = pattern.get_channel(ch_idx)
                
                if lane_type == 'trig':
                    channel.set_trigger(step, value)
                elif lane_type == 'acc':
                    channel.set_accent(step, value)
                elif lane_type == 'fill':
                    channel.set_fill(step, value)
                elif lane_type == 'prob':
                    channel.set_probability(step, value)
                elif lane_type == 'sub':
                    # Update substeps pattern for the step
                    if step < len(channel.steps):
                        channel.steps[step].substeps = value
                
                # Update that channel's editor with substeps
                substeps = [s.substeps for s in channel.steps]
                self.pattern_editors[ch_idx].set_pattern_data(
                    channel.get_triggers(),
                    channel.get_accents(),
                    channel.get_fills(),
                    channel.get_probabilities(),
                    substeps
                )
    
    def _on_pattern_length_change(self, new_length):
        """Handle pattern length change"""
        pattern = self.pattern_manager.get_selected_pattern()
        pattern.set_length(new_length)
        # Update all editors to reflect new length
        for editor in self.pattern_editors:
            editor.pattern_length = new_length
            editor._draw()
    
    def _on_pattern_play(self):
        """Start pattern playback"""
        selected_idx = self.pattern_manager.selected_pattern_index
        self.pattern_manager.start_playback(selected_idx)
        self.frames_since_last_step = 0  # Reset frame counter
        self._last_playing_pattern_idx = selected_idx  # Track for chaining detection
        self._update_pattern_button_states()
        # Update circular transport buttons
        if hasattr(self, 'play_btn') and hasattr(self.play_btn, 'set_active'):
            self.play_btn.set_active(True)
            self.stop_btn.set_active(False)
    
    def _on_pattern_stop(self):
        """Stop pattern playback and reset to beginning"""
        self.pattern_manager.stop_playback()
        self.frames_since_last_step = 0
        self._last_playing_pattern_idx = -1  # Reset tracking
        self._update_pattern_button_states()
        # Reset position display to beginning (step 0)
        if hasattr(self, 'pattern_editors'):
            for editor in self.pattern_editors:
                editor.set_current_position(0)
        # Update circular transport buttons
        if hasattr(self, 'play_btn') and hasattr(self.play_btn, 'set_active'):
            self.play_btn.set_active(False)
            self.stop_btn.set_active(True)
    
    def _on_pattern_menu(self):
        """Show pattern menu"""
        menu = tk.Menu(self.root, tearoff=0)
        
        # Get current pattern index
        idx = self.pattern_manager.selected_pattern_index
        
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
        
        # Then show the menu
        self._on_pattern_menu()
    
    def _pattern_menu_action(self, action, pattern_idx):
        """Handle pattern menu actions"""
        try:
            if action == 'cut_pattern':
                self.pattern_manager.cut_pattern(pattern_idx)
            elif action == 'copy_pattern':
                self.pattern_manager.copy_pattern(pattern_idx)
            elif action == 'paste_pattern':
                self.pattern_manager.paste_pattern(pattern_idx)
            elif action == 'exchange_pattern':
                self.pattern_manager.exchange_pattern(pattern_idx)
            elif action == 'shift_left':
                self.pattern_manager.shift_pattern_left(pattern_idx)
            elif action == 'shift_right':
                self.pattern_manager.shift_pattern_right(pattern_idx)
            elif action == 'reverse':
                self.pattern_manager.reverse_pattern(pattern_idx)
            elif action == 'randomize':
                self.pattern_manager.randomize_pattern(pattern_idx)
            elif action == 'alter':
                self.pattern_manager.alter_pattern(pattern_idx)
            elif action == 'rand_accents':
                self.pattern_manager.randomize_accents_fills(pattern_idx)
            elif action == 'export_midi':
                self._export_pattern_to_midi(pattern_idx)
                return  # Don't refresh editors or show message
            elif action == 'export_audio':
                self._export_pattern_to_audio(pattern_idx)
                return  # Don't refresh editors or show message
            
            # Refresh editors
            self._update_pattern_editors()
            messagebox.showinfo("Pattern", f"Pattern operation '{action}' completed!")
        except Exception as e:
            messagebox.showerror("Error", f"Pattern operation failed: {e}")
    
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
            # Create MIDI file
            mid = mido.MidiFile()
            track = mido.MidiTrack()
            mid.tracks.append(track)
            
            # Set tempo based on pattern manager BPM
            tempo = mido.bpm2tempo(self.pattern_manager.bpm)
            track.append(mido.MetaMessage('set_tempo', tempo=tempo))
            
            # Calculate ticks per step (assume 480 ticks per quarter note)
            ticks_per_beat = mid.ticks_per_beat
            ticks_per_step = ticks_per_beat  # Each step is a quarter note
            
            # Process each channel
            for channel_id, channel in enumerate(pattern.channels):
                # Use different MIDI channels for each drum (or use channel 10 for drums)
                midi_channel = 9  # Channel 10 (0-indexed = 9) is standard for drums
                
                # Standard GM drum note mapping (typical values)
                drum_notes = [36, 38, 42, 46, 45, 41, 39, 37]  # Kick, Snare, CHH, OHH, TomH, TomL, Clap, Rim
                note = drum_notes[channel_id] if channel_id < len(drum_notes) else 36
                
                # Add note events for each triggered step
                for step_idx in range(pattern.length):
                    step = channel.get_step(step_idx)
                    
                    if step.trigger:
                        # Calculate time offset
                        time_offset = step_idx * ticks_per_step
                        
                        # Velocity based on accent
                        velocity = 127 if step.accent else 64
                        
                        # Note on
                        track.append(mido.Message('note_on', 
                                                 channel=midi_channel, 
                                                 note=note, 
                                                 velocity=velocity, 
                                                 time=time_offset if step_idx == 0 else 0))
                        
                        # Note off after short duration (10 ticks)
                        track.append(mido.Message('note_off', 
                                                 channel=midi_channel, 
                                                 note=note, 
                                                 velocity=0, 
                                                 time=10))
            
            # Sort track by time
            track = sorted(track, key=lambda msg: getattr(msg, 'time', 0))
            
            # Convert absolute times to delta times
            cumulative_time = 0
            for msg in track:
                if hasattr(msg, 'time'):
                    abs_time = msg.time
                    msg.time = abs_time - cumulative_time
                    cumulative_time = abs_time
            
            # Save MIDI file
            mid.save(filename)
            messagebox.showinfo("Export Success", f"Pattern exported to:\\n{filename}")
            
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
            step_duration_samples = int((self.sample_rate * self.pattern_manager.step_duration_ms) / 1000.0)
            pattern_duration_samples = step_duration_samples * pattern.length
            
            # Add tail handling
            if tail_option.get() == "append":
                # Add 2 seconds of tail for reverb/decay
                tail_samples = self.sample_rate * 2
            elif tail_option.get() == "loop":
                # Add one more loop iteration
                tail_samples = pattern_duration_samples
            else:
                tail_samples = 0
            
            total_samples = pattern_duration_samples + tail_samples
            
            # Render audio
            audio_buffer = np.zeros((total_samples, 2), dtype=np.float32)
            
            # Temporarily enable playback and render
            old_playing_state = self.pattern_manager.is_playing
            old_playing_idx = self.pattern_manager.playing_pattern_index
            
            self.pattern_manager.playing_pattern_index = pattern_idx
            self.pattern_manager.is_playing = True
            self.pattern_manager.play_position = 0
            
            sample_position = 0
            for step_idx in range(pattern.length + (1 if tail_option.get() == "loop" else 0)):
                # Trigger step
                self._trigger_pattern_step(step_idx % pattern.length)
                
                # Render audio for this step
                step_audio = self.synth.process_audio(step_duration_samples)
                
                end_pos = min(sample_position + step_duration_samples, total_samples)
                chunk_size = end_pos - sample_position
                audio_buffer[sample_position:end_pos] = step_audio[:chunk_size]
                sample_position = end_pos
            
            # Restore playback state
            self.pattern_manager.is_playing = old_playing_state
            self.pattern_manager.playing_pattern_index = old_playing_idx
            
            # Convert to int16 for WAV file
            audio_int16 = (audio_buffer * 32767).astype(np.int16)
            
            # Write WAV file
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(2)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            messagebox.showinfo("Export Success", f"Pattern exported to:\\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export audio file:\\n{e}")
    
    def _on_chain_previous(self):
        """Toggle chain from previous pattern to current"""
        idx = self.pattern_manager.selected_pattern_index
        if idx > 0:
            new_state = self.pattern_manager.toggle_chain_from_prev(idx)
            status = "chained" if new_state else "unchained"
            self._update_pattern_button_states()
    
    def _on_chain_next(self):
        """Toggle chain from current pattern to next"""
        idx = self.pattern_manager.selected_pattern_index
        if idx < 11:
            new_state = self.pattern_manager.toggle_chain_to_next(idx)
            status = "chained" if new_state else "unchained"
            self._update_pattern_button_states()
    
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
    
    def _on_swing_change(self, value):
        """Handle swing slider change
        
        Swing delays the 16th notes that fall between the 8ths,
        creating a looser, more human feel (also known as shuffle).
        0% = no swing (straight timing)
        100% = maximum swing (triplet feel)
        """
        swing_percent = int(value)
        self.pattern_manager.swing = swing_percent
    
    def _on_pattern_copy(self):
        """Copy current pattern/channel to clipboard"""
        pattern = self.pattern_manager.get_selected_pattern()
        channel = pattern.get_channel(self.selected_channel)
        
        # Store in clipboard (using root variable)
        self.root.clipboard_data = {
            'type': 'pattern_channel',
            'channel_id': self.selected_channel,
            'triggers': channel.get_triggers(),
            'accents': channel.get_accents(),
            'fills': channel.get_fills(),
            'probabilities': channel.get_probabilities(),
        }
        messagebox.showinfo("Copy", "Pattern channel copied to clipboard")
    
    def _on_pattern_paste(self):
        """Paste pattern/channel from clipboard"""
        if not hasattr(self.root, 'clipboard_data') or not self.root.clipboard_data:
            messagebox.showwarning("Paste", "Nothing in clipboard")
            return
        
        pattern = self.pattern_manager.get_selected_pattern()
        channel = pattern.get_channel(self.selected_channel)
        data = self.root.clipboard_data
        
        if data['type'] == 'pattern_channel':
            channel.set_triggers(data['triggers'])
            channel.set_accents(data['accents'])
            channel.set_fills(data['fills'])
            if 'probabilities' in data:
                channel.set_probabilities(data['probabilities'])
            self._update_pattern_editors()
            messagebox.showinfo("Paste", "Pattern channel pasted")
    
    def _on_matrix_edit(self, channel_id, step, value):
        """Handle matrix editor edits"""
        pattern = self.pattern_manager.get_selected_pattern()
        channel = pattern.get_channel(channel_id)
        channel.set_trigger(step, value)
    
    def _update_matrix_editor(self):
        """Update matrix editor with current pattern data"""
        pattern = self.pattern_manager.get_selected_pattern()
        matrix_data = []
        
        for ch in range(8):
            channel = pattern.get_channel(ch)
            matrix_data.append(channel.get_triggers())
        
        self.matrix_editor.set_matrix_data(matrix_data)
    
    def _update_pattern_editors(self):

        """Update all pattern editors from current pattern"""
        pattern = self.pattern_manager.get_selected_pattern()
        
        for ch_id, editor in enumerate(self.pattern_editors):
            channel = pattern.get_channel(ch_id)
            triggers = [step.trigger for step in channel.steps]
            accents = [step.accent for step in channel.steps]
            fills = [step.fill for step in channel.steps]
            probabilities = [step.probability for step in channel.steps]
            substeps = [step.substeps for step in channel.steps]
            editor.set_pattern_data(triggers, accents, fills, probabilities, substeps)
    
    def _update_pattern_ui(self):
        """Update all pattern UI elements (buttons and editors) after loading patterns"""
        self._update_pattern_button_states()
        self._update_pattern_editors()
    
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
    
    def _trigger_channel(self, channel_idx):
        """Trigger a drum channel"""
        self.synth.trigger_drum(channel_idx, velocity=127)
        
        # Flash the channel button
        self.channel_buttons[channel_idx].set_triggered(True)
        self.root.after(100, lambda: self.channel_buttons[channel_idx].set_triggered(False))
    
    def _update_ui_from_channel(self):
        """Update all UI elements from selected channel's parameters"""
        self.updating_ui = True
        
        channel = self.synth.get_selected_channel()
        
        # Update patch name - use fallback if empty
        patch_name = channel.name if channel.name else f"Channel {self.selected_channel + 1}"
        self.patch_name_label.config(text=patch_name)
        
        # Mixing section - use set() for tk.Scale
        self.mix_slider.set(channel.osc_noise_mix * 100)
        self.eq_freq_knob.set_value(channel.eq_frequency)
        self.eq_gain_knob.set_value(channel.eq_gain_db)
        self.distort_knob.set_value(channel.distortion * 100)
        self.vintage_knob.set_value(channel.vintage_amount * 100)
        self.reverb_decay_knob.set_value(channel.reverb_decay * 100)
        self.reverb_mix_knob.set_value(channel.reverb_mix * 100)
        self.reverb_width_knob.set_value(channel.reverb_width * 100)
        
        # Delay controls
        # Map DelayTime enum value back to selector index
        delay_time_val = channel.delay_time.value
        try:
            delay_selector_idx = self.delay_time_indices.index(delay_time_val)
        except ValueError:
            delay_selector_idx = 1  # Default to 1/8
        self.delay_time_selector.set_value(delay_selector_idx)
        self.delay_feedback_knob.set_value(channel.delay_feedback * 100)
        self.delay_mix_knob.set_value(channel.delay_mix * 100)
        self.delay_pingpong_btn.set_value(channel.delay_ping_pong)
        
        self.level_knob.set_value(channel.level_db)
        self.pan_knob.set_value(channel.pan)
        self.choke_btn.set_value(channel.choke_enabled)
        self.output_selector.set_value(0 if channel.output_pair == 'A' else 1)
        
        # Oscillator section
        self.waveform_selector.set_value(channel.oscillator.waveform.value)
        self.osc_freq_knob.set_value(channel.oscillator.frequency)
        self.pitch_mod_mode.set_value(channel.oscillator.pitch_mod_mode.value)
        self.pitch_amount_knob.set_value(channel.oscillator.pitch_mod_amount)
        self.pitch_rate_knob.set_value(channel.oscillator.pitch_mod_rate)
        self.osc_attack_knob.set_value(channel.osc_envelope.attack_ms)
        self.osc_decay_knob.set_value(channel.osc_envelope.decay_ms)
        
        # Noise section - use sliders for attack/decay
        self.noise_filter_mode.set_value(channel.noise_gen.filter_mode.value)
        self.noise_freq_knob.set_value(channel.noise_gen.filter_frequency)
        self.noise_q_knob.set_value(channel.noise_gen.filter_q)
        self.stereo_btn.set_value(channel.noise_gen.stereo)
        self.noise_env_mode.set_value(channel.noise_gen.envelope_mode.value)
        self.noise_attack_slider.set_value(channel.noise_gen.attack_ms)
        self.noise_decay_slider.set_value(channel.noise_gen.decay_ms)
        
        # Velocity section
        self.osc_vel_slider.set_value(channel.osc_vel_sensitivity * 100)
        self.noise_vel_slider.set_value(channel.noise_vel_sensitivity * 100)
        self.mod_vel_slider.set_value(channel.mod_vel_sensitivity * 100)
        self.prob_slider.set_value(channel.probability)
        
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
            
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            self.preferences_manager.add_recent_file(filename)
            self._refresh_preset_list()
            messagebox.showinfo("Saved", f"Preset saved to {filename}")
    
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
                messagebox.showinfo("Loaded", f"Drum patch loaded into channel {self.selected_channel + 1}:\n{patch_name}")
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
                messagebox.showinfo("Saved", f"Drum patch saved to:\n{filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save drum patch: {e}")
    
    def _export_all_wavs(self):
        """Export all drums to WAV files"""
        folder = filedialog.askdirectory(title='Select folder for WAV export')
        if folder:
            try:
                exported = self.preset_manager.export_all_drums_to_wav(self.synth, folder)
                messagebox.showinfo("Exported", f"Exported {len(exported)} drums to {folder}")
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
                    filename
                )
                messagebox.showinfo("Exported", f"Drum exported to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export: {e}")
    
    def _load_preset_file(self, filename, show_message=True):
        """Load a preset file (internal helper)"""
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
                    
                    self._update_ui_from_channel()
                    self.preferences_manager.add_recent_file(filename)
                    self.preferences_manager.set('last_preset', filename)
                    self._refresh_preset_list()
                    if show_message:
                        messagebox.showinfo("Loaded", f"Preset loaded: {preset_data.get('name', 'Unknown')}")
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
                
                self._update_ui_from_channel()
                self.preferences_manager.add_recent_file(filename)
                self.preferences_manager.set('last_preset', filename)
                self._refresh_preset_list()
                if show_message:
                    messagebox.showinfo("Loaded", f"Preset loaded from {filename}")
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
            messagebox.showinfo("Folder Selected", f"Preset folder set to:\n{folder}")
    
    # ============ MIDI Input Methods ============
    
    def _init_midi(self):
        """Initialize MIDI input system"""
        if not self.midi_manager.enabled:
            print("MIDI input not available (mido library not installed)")
            return
        
        # Load preferences
        midi_enabled = self.preferences_manager.get('midi_enabled', True)
        base_note = self.preferences_manager.get('midi_base_note', 36)
        preferred_device = self.preferences_manager.get('midi_input_device', None)
        clock_sync_enabled = self.preferences_manager.get('midi_clock_sync', True)
        cc_mappings_raw = self.preferences_manager.get('midi_cc_mappings', {})
        
        # Convert CC mappings from string keys to int keys
        cc_mappings = {int(k): v for k, v in cc_mappings_raw.items()}
        
        # Configure MIDI manager
        self.midi_manager.set_base_note(base_note)
        self.midi_manager.set_clock_sync_enabled(clock_sync_enabled)
        self.midi_manager.set_cc_mappings(cc_mappings)
        
        # Load pitch bend target (default to osc_freq)
        pitchbend_target = self.preferences_manager.get('midi_pitchbend_target', 'osc_freq')
        self.midi_manager.set_pitchbend_target(pitchbend_target)
        
        # Set up callbacks
        self.midi_manager.set_drum_trigger_callback(self._on_midi_drum_trigger)
        self.midi_manager.set_pattern_select_callback(self._on_midi_pattern_select)
        self.midi_manager.set_transport_callbacks(
            on_start=self._on_midi_transport_start,
            on_stop=self._on_midi_transport_stop,
            on_continue=self._on_midi_transport_continue
        )
        self.midi_manager.set_activity_callback(self._on_midi_activity)
        self.midi_manager.set_bpm_callback(self._on_midi_bpm_change)
        self.midi_manager.set_cc_callback(self._on_midi_cc_change)
        self.midi_manager.set_pitchbend_callback(self._on_midi_pitchbend_change)
        
        # Auto-connect if enabled
        if midi_enabled:
            if preferred_device:
                # Try preferred device first
                if not self.midi_manager.connect(preferred_device):
                    # Fall back to auto-connect
                    self.midi_manager.auto_connect()
            else:
                self.midi_manager.auto_connect()
            
            if self.midi_manager.is_connected:
                print(f"MIDI input connected: {self.midi_manager.port_name}")
                print(f"Note mapping: {self.midi_manager.get_mapping_description()}")
                print(f"MIDI clock sync: {'enabled' if clock_sync_enabled else 'disabled'}")
                if cc_mappings:
                    print(f"MIDI CC mappings: {len(cc_mappings)} active")
                if pitchbend_target:
                    print(f"Pitch bend mapped to: {pitchbend_target}")
    
    def _on_midi_drum_trigger(self, channel: int, velocity: int):
        """Handle MIDI note triggering a drum channel"""
        # Trigger the drum through the synth
        self.synth.trigger_drum(channel, velocity)
        
        # Visual feedback - flash the channel button (must be done on main thread)
        self.root.after(0, lambda: self._flash_channel_button(channel))
    
    def _flash_channel_button(self, channel: int):
        """Flash a channel button to show it was triggered"""
        if hasattr(self, 'channel_buttons') and channel < len(self.channel_buttons):
            self.channel_buttons[channel].set_triggered(True)
            self.root.after(100, lambda: self.channel_buttons[channel].set_triggered(False))
    
    def _on_midi_pattern_select(self, pattern_index: int):
        """Handle MIDI program change to select pattern"""
        if 0 <= pattern_index < 12:  # Patterns A-L
            # Must update UI on main thread
            self.root.after(0, lambda: self._select_pattern_by_index(pattern_index))
    
    def _select_pattern_by_index(self, pattern_index: int):
        """Select a pattern by index and update UI"""
        self.pattern_manager.selected_pattern_index = pattern_index
        
        # Update pattern button states
        if hasattr(self, 'pattern_buttons'):
            for i, btn in enumerate(self.pattern_buttons):
                btn.set_selected(i == pattern_index)
        
        # Update pattern editors
        self._update_pattern_editors()
    
    def _on_midi_transport_start(self):
        """Handle MIDI Start message"""
        self.root.after(0, self._midi_start_playback)
    
    def _midi_start_playback(self):
        """Start playback from beginning (called on main thread)"""
        selected_idx = self.pattern_manager.selected_pattern_index
        self.pattern_manager.stop_playback()  # Reset position
        self.pattern_manager.start_playback(selected_idx)
        self.frames_since_last_step = 0
        self._last_playing_pattern_idx = selected_idx
        self._update_pattern_button_states()
        if hasattr(self, 'play_btn') and hasattr(self.play_btn, 'set_active'):
            self.play_btn.set_active(True)
            self.stop_btn.set_active(False)
    
    def _on_midi_transport_stop(self):
        """Handle MIDI Stop message"""
        self.root.after(0, self._on_pattern_stop)
    
    def _on_midi_transport_continue(self):
        """Handle MIDI Continue message"""
        self.root.after(0, self._midi_continue_playback)
    
    def _midi_continue_playback(self):
        """Continue playback from current position (called on main thread)"""
        if not self.pattern_manager.is_playing:
            selected_idx = self.pattern_manager.selected_pattern_index
            # Don't reset position - continue from where we are
            self.pattern_manager.is_playing = True
            self._last_playing_pattern_idx = selected_idx
            self._update_pattern_button_states()
            if hasattr(self, 'play_btn') and hasattr(self.play_btn, 'set_active'):
                self.play_btn.set_active(True)
                self.stop_btn.set_active(False)
    
    def _on_midi_activity(self):
        """Handle MIDI activity for visual feedback"""
        self._midi_activity_time = time.time()
        # Update indicator on main thread
        self.root.after(0, self._update_midi_indicator)
    
    def _on_midi_bpm_change(self, bpm: float):
        """Handle BPM change from MIDI clock sync"""
        # Must update on main thread
        self.root.after(0, lambda: self._apply_midi_bpm(bpm))
    
    def _apply_midi_bpm(self, bpm: float):
        """Apply BPM from MIDI clock (called on main thread)"""
        bpm_int = int(round(bpm))
        bpm_int = max(1, min(300, bpm_int))  # Clamp to valid range
        
        # Update pattern manager and synth
        self.pattern_manager.set_bpm(bpm_int)
        self.synth.set_bpm(bpm_int)
        
        # Update BPM display
        if hasattr(self, 'bpm_var'):
            self.bpm_var.set(str(bpm_int))
    
    def _on_midi_cc_change(self, cc_number: int, value: int):
        """Handle MIDI CC change"""
        # Must update on main thread
        self.root.after(0, lambda: self._apply_midi_cc(cc_number, value))
    
    def _on_midi_pitchbend_change(self, value: float):
        """Handle MIDI pitch bend change"""
        # Must update on main thread
        self.root.after(0, lambda: self._apply_midi_pitchbend(value))
    
    def _apply_midi_pitchbend(self, value: float):
        """Apply MIDI pitch bend value to mapped parameter (called on main thread)
        
        Pitch bend acts as a temporary modulation - when the wheel returns to center,
        the original parameter value is restored.
        
        Args:
            value: Normalized pitch bend value (-1.0 to 1.0)
        """
        param_name = self.midi_manager.get_pitchbend_target()
        if not param_name or param_name not in self._cc_parameter_registry:
            return
        
        widget, min_val, max_val, setter_func = self._cc_parameter_registry[param_name]
        
        # Check if pitch bend is at center (released)
        is_centered = abs(value) < self._pitchbend_center_threshold
        
        if is_centered:
            # Wheel returned to center - restore original value
            if self._pitchbend_active and self._pitchbend_original_value is not None:
                if hasattr(widget, 'set_value'):
                    widget.set_value(self._pitchbend_original_value)
                elif hasattr(widget, 'set'):
                    widget.set(self._pitchbend_original_value)
            self._pitchbend_active = False
            self._pitchbend_original_value = None
            return
        
        # Pitch bend is active (away from center)
        if not self._pitchbend_active:
            # Just started moving - store the original value
            if hasattr(widget, 'get_value'):
                self._pitchbend_original_value = widget.get_value()
            elif hasattr(widget, 'get'):
                self._pitchbend_original_value = widget.get()
            self._pitchbend_active = True
        
        # Calculate modulated value based on original value and pitch bend
        if self._pitchbend_original_value is not None:
            # Get the range for modulation (use half the parameter range for pitch bend)
            param_range = max_val - min_val
            modulation_range = param_range * 0.5  # Pitch bend covers +/- 50% of range
            
            # Apply modulation to original value
            modulated_value = self._pitchbend_original_value + (value * modulation_range)
            
            # Clamp to valid range
            modulated_value = max(min_val, min(max_val, modulated_value))
            
            # Update the widget
            if hasattr(widget, 'set_value'):
                widget.set_value(modulated_value)
            elif hasattr(widget, 'set'):
                widget.set(modulated_value)
    
    def _apply_midi_cc(self, cc_number: int, value: int):
        """Apply MIDI CC value to mapped parameter (called on main thread)"""
        param_name = self.midi_manager.get_parameter_for_cc(cc_number)
        if not param_name or param_name not in self._cc_parameter_registry:
            return
        
        widget, min_val, max_val, setter_func = self._cc_parameter_registry[param_name]
        
        # Map CC value (0-127) to normalized 0-1
        normalized = value / 127.0
        
        # Check if widget has logarithmic scaling and use its conversion method
        if hasattr(widget, 'logarithmic') and widget.logarithmic and hasattr(widget, '_normalized_to_value'):
            # Use widget's log conversion for proper scaling
            param_value = widget._normalized_to_value(normalized)
        else:
            # Linear mapping
            param_value = min_val + normalized * (max_val - min_val)
        
        # Update the widget (this will trigger the setter via the widget's callback)
        if hasattr(widget, 'set_value'):
            widget.set_value(param_value)
        elif hasattr(widget, 'set'):
            # For tk.Scale widgets
            widget.set(param_value)
    
    def _register_cc_parameter(self, param_name: str, widget, min_val: float, max_val: float, 
                               setter_func=None):
        """
        Register a parameter for MIDI CC control.
        
        Args:
            param_name: Unique name for the parameter
            widget: The widget (knob/slider) controlling this parameter
            min_val: Minimum value
            max_val: Maximum value  
            setter_func: Optional setter function (if None, widget callback is used)
        """
        self._cc_parameter_registry[param_name] = (widget, min_val, max_val, setter_func)
        
        # Add context menu for MIDI Learn to the widget
        self._add_midi_learn_context_menu(widget, param_name)
    
    def _add_midi_learn_context_menu(self, widget, param_name: str):
        """Add right-click context menu with MIDI Learn to a widget"""
        def show_context_menu(event):
            menu = tk.Menu(self.root, tearoff=0)
            
            # Check if this parameter already has a CC mapping
            current_cc = self.midi_manager.get_cc_for_parameter(param_name)
            if current_cc is not None:
                from pythonic.midi_manager import get_cc_name
                menu.add_command(label=f"Mapped to {get_cc_name(current_cc)}", state='disabled')
                menu.add_command(label="Remove CC Mapping", 
                               command=lambda: self._remove_cc_mapping(param_name))
                menu.add_separator()
            
            # Check if this parameter is the pitch bend target
            pitchbend_target = self.midi_manager.get_pitchbend_target()
            if pitchbend_target == param_name:
                menu.add_command(label="Pitch Bend → This Parameter", state='disabled')
                menu.add_command(label="Remove Pitch Bend Mapping", 
                               command=self._remove_pitchbend_mapping)
                menu.add_separator()
            
            if self.midi_manager.is_midi_learn_active():
                menu.add_command(label="Cancel MIDI Learn", 
                               command=self._cancel_midi_learn)
            else:
                menu.add_command(label="MIDI Learn (CC)", 
                               command=lambda: self._start_midi_learn(param_name, widget))
                # Only show "Assign Pitch Bend" if not already assigned to this param
                if pitchbend_target != param_name:
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
        """Start MIDI learn mode for a parameter"""
        self._midi_learn_target = param_name
        
        # Visual feedback - change widget appearance
        if hasattr(widget, 'configure'):
            self._midi_learn_original_bg = widget.cget('bg') if hasattr(widget, 'cget') else None
        
        # Flash the widget to indicate learn mode
        self._flash_midi_learn_widget(widget, True)
        
        def on_cc_learned(cc_number):
            # Stop flashing
            self._flash_midi_learn_widget(widget, False)
            
            # Add the mapping
            self.midi_manager.add_cc_mapping(cc_number, param_name)
            
            # Save to preferences
            self._save_cc_mappings()
            
            from pythonic.midi_manager import get_cc_name
            print(f"MIDI Learn: {get_cc_name(cc_number)} -> {param_name}")
            
            self._midi_learn_target = None
        
        self.midi_manager.start_midi_learn(lambda cc: self.root.after(0, lambda: on_cc_learned(cc)))
    
    def _flash_midi_learn_widget(self, widget, flashing: bool):
        """Flash a widget to indicate MIDI learn mode"""
        if not hasattr(self, '_midi_learn_flash_id'):
            self._midi_learn_flash_id = None
        
        if flashing:
            def flash():
                if not self.midi_manager.is_midi_learn_active():
                    return
                # Toggle between normal and highlight color
                current = widget.cget('bg') if hasattr(widget, 'cget') else '#3a3a4a'
                new_color = '#ff8844' if current != '#ff8844' else '#3a3a4a'
                if hasattr(widget, 'configure'):
                    try:
                        widget.configure(bg=new_color)
                    except:
                        pass
                self._midi_learn_flash_id = self.root.after(300, flash)
            flash()
        else:
            if self._midi_learn_flash_id:
                self.root.after_cancel(self._midi_learn_flash_id)
                self._midi_learn_flash_id = None
            # Restore original color
            if hasattr(widget, 'configure'):
                try:
                    widget.configure(bg='#3a3a4a')
                except:
                    pass
    
    def _cancel_midi_learn(self):
        """Cancel MIDI learn mode"""
        self.midi_manager.stop_midi_learn()
        self._midi_learn_target = None
    
    def _remove_cc_mapping(self, param_name: str):
        """Remove CC mapping for a parameter"""
        cc = self.midi_manager.get_cc_for_parameter(param_name)
        if cc is not None:
            self.midi_manager.remove_cc_mapping(cc)
            self._save_cc_mappings()
            print(f"Removed MIDI mapping for {param_name}")
    
    def _save_cc_mappings(self):
        """Save CC mappings to preferences"""
        mappings = self.midi_manager.get_cc_mappings()
        # Convert int keys to string for JSON
        mappings_str = {str(k): v for k, v in mappings.items()}
        self.preferences_manager.set('midi_cc_mappings', mappings_str)
    
    def _assign_pitchbend(self, param_name: str):
        """Assign pitch bend wheel to control a parameter"""
        self.midi_manager.set_pitchbend_target(param_name)
        self._save_pitchbend_mapping()
        print(f"Pitch bend wheel assigned to: {param_name}")
    
    def _remove_pitchbend_mapping(self):
        """Remove pitch bend mapping"""
        self.midi_manager.set_pitchbend_target(None)
        self._save_pitchbend_mapping()
        print("Pitch bend mapping removed")
    
    def _save_pitchbend_mapping(self):
        """Save pitch bend target to preferences"""
        target = self.midi_manager.get_pitchbend_target()
        self.preferences_manager.set('midi_pitchbend_target', target)
    
    def _get_available_parameters(self) -> list:
        """Get list of available parameters for CC mapping"""
        return sorted(self._cc_parameter_registry.keys())
    
    def _register_cc_parameters(self):
        """Register all knobs and sliders for MIDI CC control"""
        # Mixing section
        self._register_cc_parameter('level', self.level_knob, -60, 10)
        self._register_cc_parameter('pan', self.pan_knob, -100, 100)
        self._register_cc_parameter('distortion', self.distort_knob, 0, 100)
        self._register_cc_parameter('eq_freq', self.eq_freq_knob, 100, 10000)
        self._register_cc_parameter('eq_gain', self.eq_gain_knob, -12, 12)
        self._register_cc_parameter('vintage', self.vintage_knob, 0, 100)
        
        # Reverb section
        self._register_cc_parameter('reverb_decay', self.reverb_decay_knob, 0, 100)
        self._register_cc_parameter('reverb_mix', self.reverb_mix_knob, 0, 100)
        self._register_cc_parameter('reverb_width', self.reverb_width_knob, 0, 100)
        
        # Delay section
        self._register_cc_parameter('delay_feedback', self.delay_feedback_knob, 0, 100)
        self._register_cc_parameter('delay_mix', self.delay_mix_knob, 0, 100)
        
        # Oscillator section
        self._register_cc_parameter('osc_freq', self.osc_freq_knob, 20, 2000)
        self._register_cc_parameter('pitch_amount', self.pitch_amount_knob, 0, 96)
        self._register_cc_parameter('pitch_rate', self.pitch_rate_knob, 0, 500)
        self._register_cc_parameter('osc_attack', self.osc_attack_knob, 0, 1000)
        self._register_cc_parameter('osc_decay', self.osc_decay_knob, 1, 5000)
        
        # Noise section
        self._register_cc_parameter('noise_freq', self.noise_freq_knob, 100, 15000)
        self._register_cc_parameter('noise_q', self.noise_q_knob, 0.5, 20)
        self._register_cc_parameter('noise_attack', self.noise_attack_slider, 0, 1000)
        self._register_cc_parameter('noise_decay', self.noise_decay_slider, 1, 5000)
        
        # Velocity section
        self._register_cc_parameter('osc_vel', self.osc_vel_slider, 0, 100)
        self._register_cc_parameter('noise_vel', self.noise_vel_slider, 0, 100)
        self._register_cc_parameter('mod_vel', self.mod_vel_slider, 0, 100)
        self._register_cc_parameter('probability', self.prob_slider, 0, 100)
        
        # Mix slider
        self._register_cc_parameter('osc_noise_mix', self.mix_slider, 0, 100)
        
        # Master volume
        self._register_cc_parameter('master_volume', self.master_knob, -60, 10)

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
        """Open the PO-32 Tonic import dialog"""
        def on_import_done(channel_idx, params):
            """Callback after a channel is imported — refresh UI"""
            pass  # UI refresh happens after dialog closes
        
        def on_dialog_close():
            """Refresh all UI after import dialog closes"""
            self._update_ui_from_channel()
            self._update_pattern_editors()
        
        dialog = PO32ImportDialog(
            self.root,
            self.synth,
            on_import_callback=on_import_done,
        )
        # Wait for dialog to close, then refresh UI
        self.root.wait_window(dialog.dialog)
        self._update_ui_from_channel()
    
    def _get_audio_output_devices(self):
        """Get list of available audio output devices"""
        devices = []
        try:
            all_devices = sd.query_devices()
            for i, dev in enumerate(all_devices):
                if dev['max_output_channels'] > 0:
                    devices.append((i, dev['name']))
        except Exception as e:
            print(f"Error querying audio devices: {e}", flush=True)
        return devices
    
    def _show_audio_preferences(self):
        """Show audio settings dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Audio Settings")
        dialog.geometry("450x250")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.configure(bg=self.COLORS['bg_dark'])
        
        # Title
        tk.Label(dialog, text="Audio Output Settings", 
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
        available_devices = self._get_audio_output_devices()
        device_names = [name for _, name in available_devices]
        
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
        try:
            if self.audio_stream:
                current_stream_device = self.audio_stream.device
                if current_stream_device is not None:
                    dev_info = sd.query_devices(current_stream_device)
                    current_info = f"Currently using: {dev_info['name']}"
                else:
                    default_dev = sd.query_devices(sd.default.device[1])
                    current_info = f"Currently using: {default_dev['name']} (default)"
            else:
                current_info = "Audio stream not running"
        except Exception:
            current_info = "Audio stream not running"
            
        info_label = tk.Label(device_frame, text=current_info,
                             font=('Segoe UI', 8),
                             fg=self.COLORS['text_dim'],
                             bg=self.COLORS['bg_dark'])
        info_label.pack(anchor='w', pady=(5, 0))
        
        # Note about restart
        note_label = tk.Label(device_frame, 
                             text="Note: Changes take effect after restarting the application.",
                             font=('Segoe UI', 8, 'italic'),
                             fg=self.COLORS['accent'],
                             bg=self.COLORS['bg_dark'])
        note_label.pack(anchor='w', pady=(10, 0))
        
        # Buttons
        button_frame = tk.Frame(dialog, bg=self.COLORS['bg_dark'])
        button_frame.pack(fill='x', padx=20, pady=20)
        
        def refresh_devices():
            available_devices = self._get_audio_output_devices()
            device_names = [name for _, name in available_devices]
            device_combo['values'] = ["(System Default)"] + device_names
        
        def save_and_close():
            selected = device_var.get()
            if selected == "(System Default)":
                self.preferences_manager.set('audio_output_device', None)
            else:
                self.preferences_manager.set('audio_output_device', selected)
            print(f"Audio output device preference saved: {selected}", flush=True)
            dialog.destroy()
        
        def apply_now():
            """Apply changes and restart audio stream"""
            selected = device_var.get()
            if selected == "(System Default)":
                self.preferences_manager.set('audio_output_device', None)
            else:
                self.preferences_manager.set('audio_output_device', selected)
            
            # Restart audio stream
            self._stop_audio()
            self._start_audio()
            
            # Update info label
            try:
                if self.audio_stream:
                    current_stream_device = self.audio_stream.device
                    if current_stream_device is not None:
                        dev_info = sd.query_devices(current_stream_device)
                        info_label.config(text=f"Currently using: {dev_info['name']}")
                    else:
                        default_dev = sd.query_devices(sd.default.device[1])
                        info_label.config(text=f"Currently using: {default_dev['name']} (default)")
            except Exception:
                pass
            
            print(f"Audio output device changed to: {selected}", flush=True)
        
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
    
    def _show_midi_preferences(self):
        """Show MIDI settings dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("MIDI Settings")
        dialog.geometry("400x420")
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
        
        available_ports = self.midi_manager.get_available_ports()
        current_port = self.midi_manager.port_name or "(Auto-detect)"
        
        device_var = tk.StringVar(value=current_port)
        device_options = ["(Auto-detect)"] + available_ports
        device_combo = ttk.Combobox(device_frame, textvariable=device_var, 
                                    values=device_options, width=40, state='readonly')
        device_combo.pack(fill='x', pady=(2, 0))
        
        # Connection status
        status_text = f"Status: {'Connected to ' + self.midi_manager.port_name if self.midi_manager.is_connected else 'Not connected'}"
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
        
        current_base = self.midi_manager.get_base_note()
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
        
        clock_sync_var = tk.BooleanVar(value=self.midi_manager.is_clock_sync_enabled())
        clock_sync_cb = tk.Checkbutton(sync_frame, text="Sync BPM to MIDI Clock", 
                                       variable=clock_sync_var,
                                       font=('Segoe UI', 9),
                                       fg=self.COLORS['text'],
                                       bg=self.COLORS['bg_dark'],
                                       selectcolor=self.COLORS['bg_medium'],
                                       activebackground=self.COLORS['bg_dark'],
                                       activeforeground=self.COLORS['text'])
        clock_sync_cb.pack(anchor='w')
        
        # Show current synced BPM if available
        synced_bpm = self.midi_manager.get_synced_bpm()
        sync_status = f"Current synced BPM: {synced_bpm:.1f}" if synced_bpm > 0 else "Not receiving MIDI clock"
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
            
            # Apply settings
            self.midi_manager.set_base_note(base_note)
            self.midi_manager.set_clock_sync_enabled(clock_sync)
            
            # Reconnect if device changed
            current_connected = self.midi_manager.port_name
            if selected_device != current_connected:
                self.midi_manager.disconnect()
                if selected_device:
                    self.midi_manager.connect(selected_device)
                else:
                    self.midi_manager.auto_connect()
            
            # Save preferences
            self.preferences_manager.set('midi_base_note', base_note)
            self.preferences_manager.set('midi_input_device', selected_device)
            self.preferences_manager.set('midi_clock_sync', clock_sync)
            self.preferences_manager.set('midi_enabled', True)
            
            # Update status
            status_text = f"Status: {'Connected to ' + self.midi_manager.port_name if self.midi_manager.is_connected else 'Not connected'}"
            status_label.config(text=status_text)
            
            # Update sync status
            synced_bpm = self.midi_manager.get_synced_bpm()
            sync_status = f"Current synced BPM: {synced_bpm:.1f}" if synced_bpm > 0 else "Not receiving MIDI clock"
            sync_status_label.config(text=sync_status)
            
            print(f"MIDI settings updated: base note {base_note}, clock sync: {clock_sync}, device: {self.midi_manager.port_name or 'None'}")
        
        def on_ok():
            apply_settings()
            dialog.destroy()
        
        def refresh_devices():
            available_ports = self.midi_manager.get_available_ports()
            device_combo['values'] = ["(Auto-detect)"] + available_ports
        
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
        """Show MIDI CC mapping configuration dialog"""
        from pythonic.midi_manager import get_cc_name, CC_NAMES
        
        dialog = tk.Toplevel(self.root)
        dialog.title("MIDI CC Mappings")
        dialog.geometry("500x450")
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
        
        # Get available parameters and current mappings
        available_params = self._get_available_parameters()
        current_mappings = self.midi_manager.get_cc_mappings()
        
        # Common CC options
        cc_options = ["(None)"]
        for cc in [1, 2, 4, 7, 10, 11, 12, 13, 16, 17, 18, 19, 71, 74]:
            cc_options.append(get_cc_name(cc))
        # Add any other CCs that are currently mapped
        for cc in current_mappings.keys():
            cc_name = get_cc_name(cc)
            if cc_name not in cc_options:
                cc_options.append(cc_name)
        
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
        
        # Create 8 mapping rows
        for i in range(8):
            create_mapping_row(i)
        
        # Populate existing mappings
        mapping_idx = 0
        for cc_num, param_name in current_mappings.items():
            if mapping_idx < len(mapping_widgets):
                cc_var, param_var = mapping_widgets[mapping_idx]
                cc_var.set(get_cc_name(cc_num))
                param_var.set(param_name)
                mapping_idx += 1
        
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
            # Clear existing mappings
            self.midi_manager.clear_cc_mappings()
            
            # Add new mappings
            for cc_var, param_var in mapping_widgets:
                cc_str = cc_var.get()
                param = param_var.get()
                
                if cc_str == "(None)" or param == "(None)":
                    continue
                
                # Extract CC number from string like "CC1 (Mod Wheel)"
                try:
                    cc_num = int(cc_str.split('(')[0].replace('CC', '').strip())
                    self.midi_manager.add_cc_mapping(cc_num, param)
                except (ValueError, IndexError):
                    pass
            
            # Save to preferences
            self._save_cc_mappings()
            
            print(f"Applied {len(self.midi_manager.get_cc_mappings())} CC mappings")
        
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
    
    # ============== Audio ==============
    
    def _audio_callback(self, outdata, frames, time_info, status):
        """Audio callback for sounddevice - adaptive processing with sample dropping"""
        callback_start = time.perf_counter()
        trigger_time = 0
        process_time = 0
        
        self.callback_count += 1
        
        if status:
            # Count underruns
            self.underrun_count += 1
            if self.underrun_count <= 5:  # Only print first 5
                print(f"[Callback #{self.callback_count}] UNDERRUN! Status: {status}", flush=True)
        
        # Adaptive: If we're consistently running behind, drop audio processing
        drop_this_callback = False
        if self.enable_sample_dropping and len(self.callback_times) > 10:
            avg_recent = np.mean(list(self.callback_times)[-10:])
            buffer_time_ms = (frames / self.sample_rate) * 1000
            # If average callback time exceeds 90% of buffer time, start dropping
            if avg_recent > buffer_time_ms * 0.9:
                drop_this_callback = True
                self.dropped_callback_count += 1
                if self.dropped_callback_count <= 3:
                    print(f"[Callback #{self.callback_count}] DROPPING audio processing to prevent cascade", flush=True)
        
        # Update pattern playback position (always do this, even when dropping)
        if self.pattern_manager.is_playing:
            pattern_length = self.pattern_manager.get_playing_pattern().length
            
            # Convert frames to milliseconds for swing calculation
            ms_per_frame = 1000.0 / self.sample_rate
            old_time_ms = self.frames_since_last_step * ms_per_frame
            self.frames_since_last_step += frames
            new_time_ms = self.frames_since_last_step * ms_per_frame
            
            # Calculate pattern duration with swing (last step time + step duration)
            pattern_duration_ms = self.pattern_manager.get_step_time_ms(pattern_length - 1) + self.pattern_manager.step_duration_ms
            
            # Check if pattern has finished first (before step changes)
            pattern_finished = new_time_ms >= pattern_duration_ms
            
            if pattern_finished and not drop_this_callback:
                # Pattern ended - check for chaining
                advanced = self.pattern_manager.advance_to_next_pattern()
                if advanced:
                    # Switched to a new pattern - reset frame counter
                    self.frames_since_last_step = 0
                else:
                    # Loop the current pattern - reset to start
                    self.frames_since_last_step = new_time_ms - pattern_duration_ms
                
                # Update time for step calculation
                new_time_ms = self.frames_since_last_step * ms_per_frame
                pattern_length = self.pattern_manager.get_playing_pattern().length
                
                # Trigger step 0 of new/looped pattern
                self._trigger_pattern_step(0)
                self.pattern_manager.play_position = 0
                
                # Update position for UI thread
                with self.position_lock:
                    self.current_play_position = 0
            elif not drop_this_callback:
                # Check which steps we've crossed (accounting for swing)
                # Find current step by checking swing-adjusted times
                old_step = -1
                new_step = -1
                
                for step_idx in range(pattern_length):
                    step_time = self.pattern_manager.get_step_time_ms(step_idx)
                    if step_time <= old_time_ms:
                        old_step = step_idx
                    if step_time <= new_time_ms:
                        new_step = step_idx
                
                # Only trigger on step change
                if new_step != old_step and new_step >= 0:
                    trigger_start = time.perf_counter()
                    
                    # Update pattern manager position
                    self.pattern_manager.play_position = new_step
                    
                    # Trigger channels at this step
                    self._trigger_pattern_step(new_step)
                    
                    # Update position for UI thread (thread-safe)
                    with self.position_lock:
                        self.current_play_position = new_step
                    
                    trigger_time = (time.perf_counter() - trigger_start) * 1000  # ms
        
        # Process any pending fill triggers
        if self.pending_fills and not drop_this_callback:
            self._process_pending_fills(frames)
        
        # Generate audio from synthesizer (skip if dropping)
        if drop_this_callback:
            # Return last good audio or silence
            outdata[:] = self._last_good_audio[:frames] if frames <= len(self._last_good_audio) else 0
            # Record minimal timing
            total_time = (time.perf_counter() - callback_start) * 1000
            self.callback_times.append(total_time)
            return
        
        process_start = time.perf_counter()
        audio = self.synth.process_audio(frames)
        process_time = (time.perf_counter() - process_start) * 1000  # ms
        
        # Debug: log if processing is unexpectedly slow when idle
        if self.callback_count % 100 == 0:
            active_count = sum(1 for ch in self.synth.channels if ch.is_active)
            if active_count == 0 and process_time > 5.0:
                print(f"DEBUG: No active channels but process_audio took {process_time:.1f}ms!", flush=True)
        
        # Copy to output buffer
        outdata[:]  = audio
        
        # Store as last good audio for potential reuse
        if frames <= len(self._last_good_audio):
            self._last_good_audio[:frames] = audio
        
        # Record timing
        total_time = (time.perf_counter() - callback_start) * 1000  # ms
        self.callback_times.append(total_time)
        if trigger_time > 0:
            self.trigger_times.append(trigger_time)
        self.process_times.append(process_time)
        
        # Report performance periodically or after first underrun
        now = time.perf_counter()
        if (now - self.last_perf_report > 5.0) or (self.underrun_count == 1 and self.callback_count > 50):
            # Force immediate output
            import sys
            sys.stdout.flush()
            self._report_performance()
    
    def _report_performance(self):
        """Report audio callback performance metrics"""
        self.last_perf_report = time.perf_counter()
        
        if len(self.callback_times) == 0:
            return
        
        avg_callback = np.mean(self.callback_times)
        max_callback = np.max(self.callback_times)
        avg_process = np.mean(self.process_times) if self.process_times else 0
        avg_trigger = np.mean(self.trigger_times) if self.trigger_times else 0
        
        buffer_time_ms = (1050 / self.sample_rate) * 1000  # Updated to 1050
        utilization = (avg_callback / buffer_time_ms) * 100
        
        # Count active channels
        active_count = sum(1 for ch in self.synth.channels if ch.is_active)
        
        print(f"\n=== Audio Performance (last {len(self.callback_times)} callbacks) ===", flush=True)
        print(f"Callbacks: {self.callback_count}, Underruns: {self.underrun_count}, Dropped: {self.dropped_callback_count}", flush=True)
        print(f"Buffer time available: {buffer_time_ms:.2f}ms", flush=True)
        print(f"Active channels: {active_count}/8", flush=True)
        print(f"Sample dropping: {'ENABLED' if self.enable_sample_dropping else 'DISABLED'}", flush=True)
        print(f"Avg callback time: {avg_callback:.3f}ms ({utilization:.1f}% utilization)", flush=True)
        print(f"Max callback time: {max_callback:.3f}ms", flush=True)
        print(f"Avg process_audio: {avg_process:.3f}ms", flush=True)
        print(f"Avg trigger_step: {avg_trigger:.3f}ms", flush=True)
        
        if max_callback > buffer_time_ms:
            print(f"WARNING: Max callback time ({max_callback:.3f}ms) exceeds buffer time ({buffer_time_ms:.2f}ms)!", flush=True)
        if avg_callback > buffer_time_ms * 0.8:
            print(f"WARNING: High CPU utilization ({utilization:.1f}%)", flush=True)
        print("="*60, flush=True)
    
    def _start_audio(self):
        """Start the audio stream"""
        try:
            # Get preferred audio device from preferences
            preferred_device = self.preferences_manager.get('audio_output_device')
            device_param = None  # None = system default
            
            if preferred_device:
                # Try to find the device by name
                try:
                    devices = sd.query_devices()
                    for i, dev in enumerate(devices):
                        if dev['max_output_channels'] > 0 and dev['name'] == preferred_device:
                            device_param = i
                            break
                    
                    if device_param is None:
                        print(f"Audio device '{preferred_device}' not found, using system default", flush=True)
                except Exception as e:
                    print(f"Error querying audio devices: {e}", flush=True)
            
            self.audio_stream = sd.OutputStream(
                device=device_param,
                channels=2,
                callback=self._audio_callback,
                samplerate=self.sample_rate,
                blocksize=1050,  # Reduced buffer size
                dtype=np.float32
            )
            self.audio_stream.start()
            
            # Show which device is being used
            if device_param is not None:
                device_name = sd.query_devices(device_param)['name']
                print(f"Audio stream started on '{device_name}' (buffer: 1050 samples, ~{1050/self.sample_rate*1000:.1f}ms latency)", flush=True)
            else:
                default_device = sd.query_devices(sd.default.device[1])
                print(f"Audio stream started on '{default_device['name']}' [default] (buffer: 1050 samples, ~{1050/self.sample_rate*1000:.1f}ms latency)", flush=True)
        except Exception as e:
            print(f"Failed to start audio: {e}", flush=True)
    
    def _stop_audio(self):
        """Stop the audio stream"""
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None
        
        # Stop UI update timer
        if self.ui_update_timer:
            self.root.after_cancel(self.ui_update_timer)
            self.ui_update_timer = None
    
    def _trigger_pattern_step(self, step_index):
        """Trigger all active channels at a pattern step"""
        pattern = self.pattern_manager.get_playing_pattern()
        
        for channel_id in range(self.pattern_manager.num_channels):
            channel = pattern.get_channel(channel_id)
            if channel:
                step = channel.get_step(step_index)
                
                if step.trigger:
                    # Calculate effective probability: step_prob * channel_prob / 100
                    # Both are 0-100, so combined prob = (step * channel) / 100
                    drum_channel = self.synth.channels[channel_id]
                    step_prob = step.probability  # Per-step probability (0-100)
                    channel_prob = drum_channel.probability  # Per-channel "master" probability (0-100)
                    effective_prob = (step_prob * channel_prob) // 100
                    
                    # Check probability (0-100, true random)
                    if effective_prob < 100:
                        if random.randint(1, 100) > effective_prob:
                            continue  # Skip this trigger
                    
                    # Calculate velocity based on accent
                    # Accent = 127, Normal = 64 (per spec)
                    velocity = 127 if step.accent else 64
                    
                    # Check if substeps are defined
                    if step.substeps and len(step.substeps) > 0:
                        # Use substeps to determine which subdivisions to trigger
                        self._schedule_substep_triggers(channel_id, step.substeps, velocity)
                    else:
                        # Normal single trigger at step start
                        self.synth.trigger_drum(channel_id, velocity)
                    
                    # Handle fills if enabled (fills work normally with substeps)
                    if step.fill:
                        self._schedule_fill_triggers(channel_id, step.accent)
    
    def _schedule_substep_triggers(self, channel_id: int, substep_pattern: str, velocity: int):
        """
        Schedule substep triggers for a channel.
        
        Substeps divide a step into multiple subdivisions where each can be on or off.
        Format: 'o' = trigger, '-' = no trigger
        Examples: 'oo-' = 3 subdivisions, first two trigger
                  'o-o-' = 4 subdivisions, alternating triggers
        
        Args:
            channel_id: Channel to trigger
            substep_pattern: String pattern like 'oo-' or 'o-o-'
            velocity: MIDI velocity for all substep triggers
        """
        if not substep_pattern:
            return
        
        num_substeps = len(substep_pattern)
        if num_substeps == 0:
            return
        
        step_duration_ms = self.pattern_manager.step_duration_ms
        substep_interval_ms = step_duration_ms / num_substeps
        substep_interval_frames = int((substep_interval_ms / 1000.0) * self.sample_rate)
        
        # Schedule triggers for each substep
        for i, substep_char in enumerate(substep_pattern):
            if substep_char == 'o' or substep_char == 'O':
                # This substep should trigger
                frames_until_trigger = substep_interval_frames * i
                
                if i == 0:
                    # First substep: trigger immediately
                    self.synth.trigger_drum(channel_id, velocity)
                else:
                    # Later substeps: schedule for future
                    self.pending_fills.append((frames_until_trigger, channel_id, velocity))
    
    def _schedule_fill_triggers(self, channel_id: int, accented: bool):
        """
        Schedule fill triggers for a channel.
        
        Fills create rapid drum rolls at the fill rate (2-8 hits per step).
        Velocities decay from the initial velocity to simulate natural rolling.
        
        Per spec:
        - Accented fills: velocity 127 -> 64 over the roll
        - Normal fills: velocity 64 -> 0 over the roll
        """
        fill_rate = self.pattern_manager.fill_rate
        step_duration_ms = self.pattern_manager.step_duration_ms
        
        # Calculate time between fill hits in frames
        hit_interval_ms = step_duration_ms / fill_rate
        hit_interval_frames = int((hit_interval_ms / 1000.0) * self.sample_rate)
        
        # Determine velocity range
        if accented:
            start_velocity = 127
            end_velocity = 64
        else:
            start_velocity = 64
            end_velocity = 0
        
        # Schedule fill hits (skip first one since it's already triggered)
        for i in range(1, fill_rate):
            # Calculate velocity with linear decay
            progress = i / (fill_rate - 1) if fill_rate > 1 else 0
            velocity = int(start_velocity - (start_velocity - end_velocity) * progress)
            velocity = max(1, velocity)  # Ensure at least velocity 1
            
            # Schedule the trigger (frames from now)
            frames_until_trigger = hit_interval_frames * i
            self.pending_fills.append((frames_until_trigger, channel_id, velocity))
    
    def _process_pending_fills(self, frames: int):
        """
        Process and trigger any pending fill hits that fall within this audio buffer.
        Updates remaining frames for fills that haven't triggered yet.
        """
        triggered = []
        remaining = []
        
        for frames_until, channel_id, velocity in self.pending_fills:
            if frames_until <= frames:
                # This fill should trigger in this buffer
                self.synth.trigger_drum(channel_id, velocity)
                triggered.append((frames_until, channel_id, velocity))
            else:
                # Still waiting - reduce frame count
                remaining.append((frames_until - frames, channel_id, velocity))
        
        self.pending_fills = remaining
    
    def _start_ui_update_timer(self):
        """Start timer for UI updates (runs on main thread)"""
        self._ui_update_tick()
    
    def _ui_update_tick(self):
        """Periodic UI update tick - runs on main thread only"""
        if self.pattern_manager.is_playing:
            # Get current position safely
            with self.position_lock:
                position = self.current_play_position
            
            # Check if playing pattern changed (due to chaining)
            current_playing_idx = self.pattern_manager.playing_pattern_index
            if not hasattr(self, '_last_playing_pattern_idx'):
                self._last_playing_pattern_idx = current_playing_idx
            
            if current_playing_idx != self._last_playing_pattern_idx:
                # Playing pattern changed - update button states and editors
                self._last_playing_pattern_idx = current_playing_idx
                
                # Auto-select the playing pattern so editors follow playback
                self.pattern_manager.select_pattern(current_playing_idx)
                self._update_pattern_button_states()
                self._update_pattern_editors()
            
            # Update UI with current position
            if hasattr(self, 'pattern_editors'):
                for editor in self.pattern_editors:
                    editor.set_current_position(position)
        
        # Schedule next update (every 50ms to reduce load)
        self.ui_update_timer = self.root.after(50, self._ui_update_tick)
    
    def _update_pattern_position_display(self):
        """Update the visual position indicator in pattern editors"""
        # This method is now handled by _ui_update_tick
        # Kept for compatibility but delegates to thread-safe version
        with self.position_lock:
            position = self.current_play_position
        
        if hasattr(self, 'pattern_editors'):
            for editor in self.pattern_editors:
                editor.set_current_position(position)
    
    def _update_pattern_button_states(self):
        """Update visual states of pattern buttons"""
        selected_idx = self.pattern_manager.selected_pattern_index
        playing_idx = self.pattern_manager.playing_pattern_index if self.pattern_manager.is_playing else -1
        
        for i, btn in enumerate(self.pattern_buttons):
            pattern = self.pattern_manager.get_pattern(i)
            
            # Determine background color
            if i == playing_idx and self.button_flash_state:
                # Playing pattern - flash green
                bg_color = self.COLORS['led_on']
            elif i == selected_idx:
                # Selected pattern - blue highlight
                bg_color = self.COLORS['highlight']
            else:
                # Normal state
                bg_color = self.COLORS['bg_light']
            
            # Determine text color
            if pattern.is_empty():
                # Empty pattern - gray text
                fg_color = self.COLORS['text_dim']
            elif pattern.chained_to_next or pattern.chained_from_prev:
                # Chained pattern - blue text
                fg_color = self.COLORS['highlight']
            else:
                # Normal text
                fg_color = self.COLORS['text']
            
            btn.config(bg=bg_color, fg=fg_color)
        
        # Schedule next update for flash animation
        self.root.after(250, self._toggle_button_flash)
    
    def _toggle_button_flash(self):
        """Toggle flash state and update buttons"""
        if self.pattern_manager.is_playing:
            self.button_flash_state = not self.button_flash_state
            self._update_pattern_button_states()
        else:
            # Reset flash state when not playing
            if self.button_flash_state:
                self.button_flash_state = False
                self._update_pattern_button_states()
            else:
                # Continue checking
                self.root.after(250, self._toggle_button_flash)
    
    def run(self):
        """Run the application"""
        try:
            self.root.mainloop()
        finally:
            self._stop_audio()
            # Cleanup MIDI
            if hasattr(self, 'midi_manager'):
                self.midi_manager.cleanup()
            # Cleanup synthesizer thread pool
            if hasattr(self.synth, 'cleanup'):
                self.synth.cleanup()


def main():
    """Main entry point"""
    app = PythonicGUI()
    app.run()


if __name__ == '__main__':
    main()
