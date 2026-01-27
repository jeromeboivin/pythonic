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

# Import our synthesizer
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pythonic.synthesizer import PythonicSynthesizer
from pythonic.oscillator import WaveformType, PitchModMode
from pythonic.noise import NoiseFilterMode, NoiseEnvelopeMode
from pythonic.preset_manager import PresetManager
from gui.widgets import (
    RotaryKnob, VerticalSlider, ChannelButton,
    WaveformSelector, ModeSelector, ToggleButton
)

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: sounddevice not available. Audio playback disabled.")


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
    }
    
    def __init__(self):
        # Create main window
        self.root = tk.Tk()
        self.root.title("Pythonic Drum Synthesizer")
        self.root.configure(bg=self.COLORS['bg_dark'])
        self.root.resizable(False, False)
        
        # Initialize synthesizer
        self.sample_rate = 44100
        self.synth = PythonicSynthesizer(self.sample_rate)
        
        # Preset manager
        self.preset_manager = PresetManager(self.synth)
        
        # Audio state
        self.audio_stream = None
        self.audio_buffer = np.zeros((1024, 2), dtype=np.float32)
        self.buffer_lock = threading.Lock()
        
        # UI state
        self.selected_channel = 0
        self.updating_ui = False  # Prevent feedback loops
        
        # Build the interface
        self._build_ui()
        
        # Start audio if available
        if AUDIO_AVAILABLE:
            self._start_audio()
        
        # Update UI with current channel
        self._update_ui_from_channel()
    
    def _build_ui(self):
        """Build the complete user interface"""
        # Main container
        main_frame = tk.Frame(self.root, bg=self.COLORS['bg_dark'])
        main_frame.pack(padx=10, pady=10)
        
        # Header/Title bar
        self._build_header(main_frame)
        
        # Preset section (channel buttons)
        self._build_preset_section(main_frame)
        
        # Drum Patch section (main controls)
        self._build_drum_patch_section(main_frame)
        
        # Global section (master controls)
        self._build_global_section(main_frame)
    
    def _build_header(self, parent):
        """Build header with title and program selector"""
        header = tk.Frame(parent, bg=self.COLORS['bg_medium'], height=50)
        header.pack(fill='x', pady=(0, 10))
        header.pack_propagate(False)
        
        # Logo/Title
        title_frame = tk.Frame(header, bg=self.COLORS['bg_medium'])
        title_frame.pack(side='left', padx=10, pady=5)
        
        tk.Label(title_frame, text="PYTHONIC", 
                font=('Segoe UI', 16, 'bold'), fg=self.COLORS['text'],
                bg=self.COLORS['bg_medium']).pack()
        
        # Preset name display
        preset_frame = tk.Frame(header, bg=self.COLORS['bg_dark'])
        preset_frame.pack(side='left', padx=20, pady=10)
        
        self.preset_label = tk.Label(preset_frame, 
                                    text="  Factory Kit  ",
                                    font=('Courier', 10),
                                    fg=self.COLORS['text'],
                                    bg=self.COLORS['bg_dark'],
                                    relief='sunken', padx=5, pady=2)
        self.preset_label.pack()
        
        # Master volume in header
        master_frame = tk.Frame(header, bg=self.COLORS['bg_medium'])
        master_frame.pack(side='right', padx=10, pady=5)
        
        tk.Label(master_frame, text="master", 
                font=('Segoe UI', 7), fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack()
        
        self.master_knob = RotaryKnob(master_frame, size=40, 
                                      min_val=-60, max_val=10, default=0,
                                      command=self._on_master_volume_change)
        self.master_knob.pack()
    
    def _build_preset_section(self, parent):
        """Build the preset/channel selection section"""
        preset_section = tk.Frame(parent, bg=self.COLORS['bg_medium'])
        preset_section.pack(fill='x', pady=(0, 10))
        
        # Drum patch selector label
        selector_frame = tk.Frame(preset_section, bg=self.COLORS['bg_medium'])
        selector_frame.pack(side='left', padx=10, pady=5)
        
        self.patch_name_label = tk.Label(selector_frame,
                                         text="Kick",
                                         font=('Segoe UI', 10),
                                         fg=self.COLORS['text'],
                                         bg=self.COLORS['bg_dark'],
                                         width=20, anchor='w',
                                         relief='sunken', padx=5)
        self.patch_name_label.pack()
        
        # Channel buttons
        channels_frame = tk.Frame(preset_section, bg=self.COLORS['bg_medium'])
        channels_frame.pack(side='left', padx=20, pady=5)
        
        self.channel_buttons = []
        for i in range(8):
            btn = ChannelButton(channels_frame, i, size=35,
                               command=self._on_channel_select)
            btn.pack(side='left', padx=2)
            self.channel_buttons.append(btn)
        
        # Select first channel
        self.channel_buttons[0].set_selected(True)
        
        # Mute buttons row
        mute_frame = tk.Frame(preset_section, bg=self.COLORS['bg_medium'])
        mute_frame.pack(side='left', padx=10, pady=5)
        
        tk.Label(mute_frame, text="m", font=('Segoe UI', 8),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack(side='left', padx=2)
        
        self.mute_buttons = []
        for i in range(8):
            btn = ToggleButton(mute_frame, text="", width=20, height=20,
                              command=lambda en, ch=i: self._on_mute_toggle(ch, en))
            btn.pack(side='left', padx=1)
            self.mute_buttons.append(btn)
    
    def _build_drum_patch_section(self, parent):
        """Build the main drum patch editing section"""
        patch_frame = tk.Frame(parent, bg=self.COLORS['bg_medium'])
        patch_frame.pack(fill='both', expand=True, pady=(0, 10))
        
        # Three main subsections
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
        section.pack(side='left', padx=5, pady=5, fill='y')
        
        # Row 1: Mix slider and labels
        row1 = tk.Frame(section, bg=self.COLORS['bg_medium'])
        row1.pack(pady=5)
        
        tk.Label(row1, text="osc", font=('Segoe UI', 7),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack(side='left')
        
        self.mix_slider = VerticalSlider(row1, width=25, height=60,
                                        min_val=0, max_val=100, default=50,
                                        command=self._on_mix_change)
        self.mix_slider.pack(side='left', padx=5)
        
        tk.Label(row1, text="noise", font=('Segoe UI', 7),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack(side='left')
        
        # Row 2: EQ Freq
        row2 = tk.Frame(section, bg=self.COLORS['bg_medium'])
        row2.pack(pady=5)
        
        self.eq_freq_knob = RotaryKnob(row2, size=45,
                                       min_val=20, max_val=20000, default=632,
                                       label="eq freq",
                                       command=self._on_eq_freq_change)
        self.eq_freq_knob.pack(side='left', padx=3)
        
        # Row 3: Distortion and EQ Gain
        row3 = tk.Frame(section, bg=self.COLORS['bg_medium'])
        row3.pack(pady=5)
        
        self.distort_knob = RotaryKnob(row3, size=45,
                                       min_val=0, max_val=100, default=0,
                                       label="distort",
                                       command=self._on_distort_change)
        self.distort_knob.pack(side='left', padx=3)
        
        self.eq_gain_knob = RotaryKnob(row3, size=45,
                                       min_val=-40, max_val=40, default=0,
                                       label="eq gain",
                                       command=self._on_eq_gain_change)
        self.eq_gain_knob.pack(side='left', padx=3)
        
        # Row 4: Level and Pan
        row4 = tk.Frame(section, bg=self.COLORS['bg_medium'])
        row4.pack(pady=5)
        
        self.level_knob = RotaryKnob(row4, size=45,
                                     min_val=-60, max_val=10, default=0,
                                     label="level",
                                     command=self._on_level_change)
        self.level_knob.pack(side='left', padx=3)
        
        self.pan_knob = RotaryKnob(row4, size=45,
                                   min_val=-100, max_val=100, default=0,
                                   label="pan",
                                   command=self._on_pan_change)
        self.pan_knob.pack(side='left', padx=3)
        
        # Row 5: Choke and Output
        row5 = tk.Frame(section, bg=self.COLORS['bg_medium'])
        row5.pack(pady=5)
        
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
        section.pack(side='left', padx=5, pady=5, fill='y')
        
        # Waveform selector
        row1 = tk.Frame(section, bg=self.COLORS['bg_medium'])
        row1.pack(pady=5)
        
        tk.Label(row1, text="waveform", font=('Segoe UI', 7),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack()
        
        self.waveform_selector = WaveformSelector(row1,
                                                  command=self._on_waveform_change)
        self.waveform_selector.pack()
        
        # Frequency
        row2 = tk.Frame(section, bg=self.COLORS['bg_medium'])
        row2.pack(pady=5)
        
        self.osc_freq_knob = RotaryKnob(row2, size=50,
                                        min_val=20, max_val=20000, default=440,
                                        label="osc freq",
                                        command=self._on_osc_freq_change)
        self.osc_freq_knob.pack()
        
        # Pitch modulation mode
        row3 = tk.Frame(section, bg=self.COLORS['bg_medium'])
        row3.pack(pady=5)
        
        tk.Label(row3, text="pitch mod", font=('Segoe UI', 7),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack()
        
        self.pitch_mod_mode = ModeSelector(row3, 
                                          options=['Decay', 'Sine', 'Rand'],
                                          command=self._on_pitch_mod_mode_change)
        self.pitch_mod_mode.pack()
        
        # Pitch mod amount and rate
        row4 = tk.Frame(section, bg=self.COLORS['bg_medium'])
        row4.pack(pady=5)
        
        self.pitch_amount_knob = RotaryKnob(row4, size=45,
                                            min_val=-120, max_val=120, default=0,
                                            label="amount",
                                            command=self._on_pitch_amount_change)
        self.pitch_amount_knob.pack(side='left', padx=3)
        
        self.pitch_rate_knob = RotaryKnob(row4, size=45,
                                          min_val=1, max_val=2000, default=100,
                                          label="rate",
                                          command=self._on_pitch_rate_change)
        self.pitch_rate_knob.pack(side='left', padx=3)
        
        # Attack and Decay
        row5 = tk.Frame(section, bg=self.COLORS['bg_medium'])
        row5.pack(pady=5)
        
        self.osc_attack_knob = RotaryKnob(row5, size=45,
                                          min_val=0, max_val=10000, default=0,
                                          label="attack",
                                          command=self._on_osc_attack_change)
        self.osc_attack_knob.pack(side='left', padx=3)
        
        self.osc_decay_knob = RotaryKnob(row5, size=45,
                                         min_val=10, max_val=10000, default=316,
                                         label="decay",
                                         command=self._on_osc_decay_change)
        self.osc_decay_knob.pack(side='left', padx=3)
    
    def _build_noise_section(self, parent):
        """Build the noise generator controls section"""
        section = tk.LabelFrame(parent, text="noise",
                               font=('Segoe UI', 8),
                               fg=self.COLORS['text_dim'],
                               bg=self.COLORS['bg_medium'],
                               labelanchor='n')
        section.pack(side='left', padx=5, pady=5, fill='y')
        
        # Filter mode
        row1 = tk.Frame(section, bg=self.COLORS['bg_medium'])
        row1.pack(pady=5)
        
        tk.Label(row1, text="filter mode", font=('Segoe UI', 7),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack()
        
        self.noise_filter_mode = ModeSelector(row1,
                                             options=['LP', 'BP', 'HP'],
                                             command=self._on_noise_filter_mode_change)
        self.noise_filter_mode.pack()
        
        # Filter freq
        row2 = tk.Frame(section, bg=self.COLORS['bg_medium'])
        row2.pack(pady=5)
        
        self.noise_freq_knob = RotaryKnob(row2, size=50,
                                          min_val=20, max_val=20000, default=20000,
                                          label="filter freq",
                                          command=self._on_noise_freq_change)
        self.noise_freq_knob.pack()
        
        # Filter Q and Stereo
        row3 = tk.Frame(section, bg=self.COLORS['bg_medium'])
        row3.pack(pady=5)
        
        self.noise_q_knob = RotaryKnob(row3, size=45,
                                       min_val=0.1, max_val=100, default=0.707,
                                       label="filter q",
                                       command=self._on_noise_q_change)
        self.noise_q_knob.pack(side='left', padx=3)
        
        self.stereo_btn = ToggleButton(row3, text="stereo", width=45, height=20,
                                      command=self._on_stereo_toggle)
        self.stereo_btn.pack(side='left', padx=3)
        
        # Envelope mode
        row4 = tk.Frame(section, bg=self.COLORS['bg_medium'])
        row4.pack(pady=5)
        
        tk.Label(row4, text="envelope", font=('Segoe UI', 7),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack()
        
        self.noise_env_mode = ModeSelector(row4,
                                          options=['Exp', 'Lin', 'Mod'],
                                          command=self._on_noise_env_mode_change)
        self.noise_env_mode.pack()
        
        # Attack and Decay
        row5 = tk.Frame(section, bg=self.COLORS['bg_medium'])
        row5.pack(pady=5)
        
        self.noise_attack_knob = RotaryKnob(row5, size=45,
                                            min_val=0, max_val=10000, default=0,
                                            label="attack",
                                            command=self._on_noise_attack_change)
        self.noise_attack_knob.pack(side='left', padx=3)
        
        self.noise_decay_knob = RotaryKnob(row5, size=45,
                                           min_val=10, max_val=10000, default=316,
                                           label="decay",
                                           command=self._on_noise_decay_change)
        self.noise_decay_knob.pack(side='left', padx=3)
    
    def _build_velocity_section(self, parent):
        """Build the velocity sensitivity section"""
        section = tk.LabelFrame(parent, text="vel",
                               font=('Segoe UI', 8),
                               fg=self.COLORS['text_dim'],
                               bg=self.COLORS['bg_medium'],
                               labelanchor='n')
        section.pack(side='left', padx=5, pady=5, fill='y')
        
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
    
    def _build_global_section(self, parent):
        """Build the global controls section"""
        global_frame = tk.Frame(parent, bg=self.COLORS['bg_medium'])
        global_frame.pack(fill='x', pady=(0, 5))
        
        # File operations buttons
        file_frame = tk.Frame(global_frame, bg=self.COLORS['bg_medium'])
        file_frame.pack(side='left', padx=10, pady=5)
        
        tk.Button(file_frame, text="Load Preset",
                 width=10, height=1,
                 bg=self.COLORS['accent'],
                 fg=self.COLORS['text'],
                 command=self._load_preset).pack(side='left', padx=2)
        
        tk.Button(file_frame, text="Save Preset",
                 width=10, height=1,
                 bg=self.COLORS['bg_light'],
                 fg=self.COLORS['text'],
                 command=self._save_preset).pack(side='left', padx=2)
        
        tk.Button(file_frame, text="Export WAVs",
                 width=10, height=1,
                 bg=self.COLORS['bg_light'],
                 fg=self.COLORS['text'],
                 command=self._export_all_wavs).pack(side='left', padx=2)
        
        tk.Button(file_frame, text="Export Drum",
                 width=10, height=1,
                 bg=self.COLORS['bg_light'],
                 fg=self.COLORS['text'],
                 command=self._export_current_drum).pack(side='left', padx=2)
        
        # Trigger buttons for testing
        trigger_frame = tk.Frame(global_frame, bg=self.COLORS['bg_medium'])
        trigger_frame.pack(side='left', padx=10, pady=5)
        
        tk.Label(trigger_frame, text="Trigger:", font=('Segoe UI', 8),
                fg=self.COLORS['text'],
                bg=self.COLORS['bg_medium']).pack(side='left', padx=5)
        
        for i in range(8):
            btn = tk.Button(trigger_frame, text=str(i + 1),
                           width=3, height=1,
                           bg=self.COLORS['bg_light'],
                           fg=self.COLORS['text'],
                           command=lambda ch=i: self._trigger_channel(ch))
            btn.pack(side='left', padx=2)
        
        # Keyboard instructions
        info_frame = tk.Frame(global_frame, bg=self.COLORS['bg_medium'])
        info_frame.pack(side='right', padx=10, pady=5)
        
        tk.Label(info_frame, 
                text="Keys 1-8: Trigger",
                font=('Segoe UI', 8),
                fg=self.COLORS['text_dim'],
                bg=self.COLORS['bg_medium']).pack()
        
        # Bind keyboard
        self.root.bind('<Key>', self._on_key_press)
    
    # ============== Event Handlers ==============
    
    def _on_channel_select(self, channel_idx):
        """Handle channel selection"""
        # Update button states
        for i, btn in enumerate(self.channel_buttons):
            btn.set_selected(i == channel_idx)
        
        self.selected_channel = channel_idx
        self.synth.select_channel(channel_idx)
        
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
        """Handle osc/noise mix change"""
        if not self.updating_ui:
            channel = self.synth.get_selected_channel()
            channel.osc_noise_mix = value / 100.0
    
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
        
        # Update patch name
        self.patch_name_label.config(text=channel.name)
        
        # Mixing section
        self.mix_slider.set_value(channel.osc_noise_mix * 100)
        self.eq_freq_knob.set_value(channel.eq_frequency)
        self.eq_gain_knob.set_value(channel.eq_gain_db)
        self.distort_knob.set_value(channel.distortion * 100)
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
        
        # Noise section
        self.noise_filter_mode.set_value(channel.noise_gen.filter_mode.value)
        self.noise_freq_knob.set_value(channel.noise_gen.filter_frequency)
        self.noise_q_knob.set_value(channel.noise_gen.filter_q)
        self.stereo_btn.set_value(channel.noise_gen.stereo)
        self.noise_env_mode.set_value(channel.noise_gen.envelope_mode.value)
        self.noise_attack_knob.set_value(channel.noise_gen.attack_ms)
        self.noise_decay_knob.set_value(channel.noise_gen.decay_ms)
        
        # Velocity section
        self.osc_vel_slider.set_value(channel.osc_vel_sensitivity * 100)
        self.noise_vel_slider.set_value(channel.noise_vel_sensitivity * 100)
        self.mod_vel_slider.set_value(channel.mod_vel_sensitivity * 100)
        
        self.updating_ui = False
    
    def _save_preset(self):
        """Save current preset to file"""
        filename = filedialog.asksaveasfilename(
            defaultextension='.json',
            filetypes=[('JSON files', '*.json'), ('All files', '*.*')],
            title='Save Preset'
        )
        
        if filename:
            data = self.synth.get_preset_data()
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            messagebox.showinfo("Saved", f"Preset saved to {filename}")
    
    def _load_preset(self):
        """Load preset from file"""
        filename = filedialog.askopenfilename(
            filetypes=[
                ('Pythonic Preset', '*.mtpreset'),
                ('JSON files', '*.json'),
                ('All files', '*.*')
            ],
            title='Load Preset'
        )
        
        if filename:
            try:
                if filename.lower().endswith('.mtpreset'):
                    # Load native Pythonic preset format
                    preset_data = self.preset_manager.load_mtpreset(filename)
                    if preset_data and preset_data.get('drums'):
                        for i, drum_params in enumerate(preset_data['drums']):
                            if i < 8 and drum_params:
                                channel = self.synth.channels[i]
                                channel.set_parameters(drum_params)
                        self._update_ui_from_channel()
                        messagebox.showinfo("Loaded", f"Preset loaded: {preset_data.get('name', 'Unknown')}")
                    else:
                        messagebox.showerror("Error", "Failed to parse preset file")
                else:
                    # Load JSON format
                    with open(filename, 'r') as f:
                        data = json.load(f)
                    self.synth.load_preset_data(data)
                    self._update_ui_from_channel()
                    messagebox.showinfo("Loaded", f"Preset loaded from {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load preset: {e}")
    
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
    
    # ============== Audio ==============
    
    def _audio_callback(self, outdata, frames, time, status):
        """Audio callback for sounddevice"""
        if status:
            print(f"Audio status: {status}")
        
        # Generate audio from synthesizer
        audio = self.synth.process_audio(frames)
        
        # Copy to output buffer
        outdata[:] = audio
    
    def _start_audio(self):
        """Start the audio stream"""
        try:
            self.audio_stream = sd.OutputStream(
                channels=2,
                callback=self._audio_callback,
                samplerate=self.sample_rate,
                blocksize=512,
                dtype=np.float32
            )
            self.audio_stream.start()
            print("Audio stream started")
        except Exception as e:
            print(f"Failed to start audio: {e}")
    
    def _stop_audio(self):
        """Stop the audio stream"""
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
            self.audio_stream = None
    
    def run(self):
        """Run the application"""
        try:
            self.root.mainloop()
        finally:
            self._stop_audio()


def main():
    """Main entry point"""
    app = PythonicGUI()
    app.run()


if __name__ == '__main__':
    main()
