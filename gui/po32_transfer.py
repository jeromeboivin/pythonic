"""
PO-32 Transfer Dialog for Pythonic
Generates FSK audio for transferring patches and patterns to PO-32 Tonic.

Transfer window:
- Transfer sounds + patterns via audio modem
- Bank selection (1-8 or 9-16)
- Pattern chain selection
- Audio playback through sounddevice
- Save WAV option
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import struct
import math
import numpy as np
import os
import time

from pythonic.po32_codec import (
    ModemEncoder, TAG_PATCH, TAG_PATTERN, TAG_STATE, TAG_TRAILER,
    PARAM_NAMES, DISCRETE_PARAMS,
    freq_to_norm, attack_to_norm, decay_to_norm,
    modrate_sine_to_norm, modrate_noise_to_norm, modrate_decay_to_norm,
    modamt_to_norm, level_to_norm,
    bit_reverse_bytes, generate_fsk_signal as generate_fsk_signal_v2, save_wav,
    default_right_patch as _default_right_patch,
    default_pattern as _default_pattern,
    default_state as _default_state,
    SAMPLE_RATE,
)

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False


# =============================================================================
# Synth-to-Encoder Bridge
# =============================================================================

# Maps synth enum values to PO-32 display strings
WAVEFORM_TO_STR = {0: 'Sine', 1: 'Triangle', 2: 'Saw'}
PITCHMOD_TO_STR = {0: 'Decay', 1: 'Sine', 2: 'Noise'}
FILTERMODE_TO_STR = {0: 'LP', 1: 'BP', 2: 'HP'}
ENVMODE_TO_STR = {0: 'Exp', 1: 'Linear', 2: 'Mod'}


def channel_to_display_params(channel) -> dict:
    """Extract display-value parameter strings from a live DrumChannel.
    
    Converts the synthesizer's internal parameter values into the display
    format strings that the po32_encoder's display_to_normalized() expects.
    """
    params = channel.get_parameters()
    
    display = {}
    
    # Discrete parameters
    display['OscWave'] = WAVEFORM_TO_STR.get(params['osc_waveform'], 'Sine')
    display['ModMode'] = PITCHMOD_TO_STR.get(params['pitch_mod_mode'], 'Decay')
    display['NFilMod'] = FILTERMODE_TO_STR.get(params['noise_filter_mode'], 'LP')
    display['NEnvMod'] = ENVMODE_TO_STR.get(params['noise_envelope_mode'], 'Exp')
    
    # Frequency parameters (Hz)
    display['OscFreq'] = f"{params['osc_frequency']:.2f} Hz"
    display['NFilFrq'] = f"{params['noise_filter_freq']:.2f} Hz"
    display['EQFreq'] = f"{params['eq_frequency']:.2f} Hz"
    
    # Attack/Decay times (ms)
    display['OscAtk'] = f"{params['osc_attack']:.2f} ms"
    display['OscDcy'] = f"{params['osc_decay']:.2f} ms"
    display['NEnvAtk'] = f"{params['noise_attack']:.2f} ms"
    display['NEnvDcy'] = f"{params['noise_decay']:.2f} ms"
    
    # ModRate: depends on mode
    mod_mode = display['ModMode']
    if mod_mode == 'Decay':
        display['ModRate'] = f"{params['pitch_mod_rate']:.2f} ms"
    else:
        display['ModRate'] = f"{params['pitch_mod_rate']:.2f} Hz"
    
    # ModAmt (semitones)
    display['ModAmt'] = f"{params['pitch_mod_amount']:.2f} sm"
    
    # NFilQ
    display['NFilQ'] = f"{params['noise_filter_q']:.4f}"
    
    # Mix: osc_noise_mix is 0=all noise, 1=all osc in synth
    # PO-32 Mix parameter is noise percentage
    # osc_noise_mix: 0.5 means 50/50, 0 means 100% noise, 1 means 100% osc
    osc_pct = params['osc_noise_mix'] * 100
    noise_pct = (1.0 - params['osc_noise_mix']) * 100
    display['Mix'] = f"{osc_pct:.2f} / {noise_pct:.2f}"
    
    # DistAmt (0-100%)
    display['DistAmt'] = f"{params['distortion'] * 100:.2f} %"
    
    # EQGain (dB)
    display['EQGain'] = f"{params['eq_gain_db']:.2f} dB"
    
    # Level (dB)
    display['Level'] = f"{params['level_db']:.2f} dB"
    
    # Velocity sensitivity (0-200%)
    display['OscVel'] = f"{params['osc_vel_sensitivity'] * 200:.2f} %"
    display['NVel'] = f"{params['noise_vel_sensitivity'] * 200:.2f} %"
    display['ModVel'] = f"{params['mod_vel_sensitivity'] * 200:.2f} %"
    
    return display


def channel_to_normalized(channel) -> dict:
    """Convert a DrumChannel directly to normalized 0-1 values for PO-32 encoding."""
    params = channel.get_parameters()
    
    norm = {}
    
    # Discrete parameters -> index / (num_options - 1)
    norm['OscWave'] = params['osc_waveform'] / 2.0
    norm['ModMode'] = params['pitch_mod_mode'] / 2.0
    norm['NFilMod'] = params['noise_filter_mode'] / 2.0
    norm['NEnvMod'] = params['noise_envelope_mode'] / 2.0
    
    # Frequencies
    norm['OscFreq'] = freq_to_norm(params['osc_frequency'])
    norm['NFilFrq'] = freq_to_norm(params['noise_filter_freq'])
    norm['EQFreq'] = freq_to_norm(params['eq_frequency'])
    
    # Attack times
    norm['OscAtk'] = attack_to_norm(params['osc_attack'])
    
    # NEnvAtk with Linear correction
    nenv_mode = ENVMODE_TO_STR.get(params['noise_envelope_mode'], 'Exp')
    atk_ms = params['noise_attack']
    if nenv_mode == 'Linear':
        atk_ms *= 1.5
    norm['NEnvAtk'] = attack_to_norm(atk_ms)
    
    # Decay times
    norm['OscDcy'] = decay_to_norm(params['osc_decay'])
    
    dcy_ms = params['noise_decay']
    if nenv_mode == 'Linear':
        dcy_ms *= 1.5
    norm['NEnvDcy'] = decay_to_norm(dcy_ms)
    
    # ModRate (mode-dependent)
    mod_mode_str = PITCHMOD_TO_STR.get(params['pitch_mod_mode'], 'Decay')
    rate = params['pitch_mod_rate']
    if mod_mode_str == 'Sine':
        norm['ModRate'] = modrate_sine_to_norm(rate)
    elif mod_mode_str == 'Noise':
        norm['ModRate'] = modrate_noise_to_norm(rate)
    else:  # Decay
        norm['ModRate'] = modrate_decay_to_norm(rate)
    
    # ModAmt
    norm['ModAmt'] = modamt_to_norm(params['pitch_mod_amount'], mod_mode_str)
    
    # NFilQ
    q = params['noise_filter_q']
    if q <= 0.1:
        norm['NFilQ'] = 0.0
    else:
        norm['NFilQ'] = max(0.0, min(1.0, math.log(q / 0.1) / math.log(10001.0 / 0.1)))
    
    # Mix: osc_noise_mix 0=all noise 1=all osc, PO-32 norm = noise_pct/100
    norm['Mix'] = 1.0 - params['osc_noise_mix']
    
    # DistAmt
    norm['DistAmt'] = params['distortion']
    
    # EQGain: -40 to +40 dB
    norm['EQGain'] = max(0.0, min(1.0, (params['eq_gain_db'] + 40) / 80))
    
    # Level
    norm['Level'] = level_to_norm(params['level_db'])
    
    # Velocity
    norm['OscVel'] = params['osc_vel_sensitivity']  # 0-1 in synth, = pct/200 for PO-32 (0-100% → 0-0.5)
    norm['NVel'] = params['noise_vel_sensitivity']
    norm['ModVel'] = params['mod_vel_sensitivity']
    
    return norm


def serialize_channel(channel) -> bytes:
    """Serialize a DrumChannel to 42 bytes (21 × uint16 LE) for PO-32 patch TLV."""
    norm = channel_to_normalized(channel)
    data = bytearray()
    for name in PARAM_NAMES:
        val = norm.get(name, 0.0)
        u16 = min(int(round(val * 65536)), 65535)
        data += struct.pack('<H', u16)
    return bytes(data)


def serialize_pattern(pattern, mute_mask=None) -> bytes:
    """Serialize a Pythonic Pattern to PO-32 pattern binary format.
    
    PO-32 pattern format (210 data bytes):
    The exact internal format is not fully documented, but from analysis:
    - 16 steps × 8 drums
    - Each step stores trigger, accent, fill info
    - We encode as: 8 channels × 16 steps, packed as nibbles or bytes
    
    Since the exact PO-32 pattern binary format is proprietary and not
    fully reverse-engineered, we use a best-effort encoding based on
    the observed 210-byte structure from reference files.
    
    Known structure from reference analysis:
    - Byte 0: pattern number (prefix, added separately)
    - Bytes 1-208: step data  
    - Format appears to be: for each of 16 steps, 13 bytes of data
      (16 × 13 = 208, close to 210)
    
    Since we don't have the exact format, we generate empty patterns
    and let the patches (sounds) be the main transfer content.
    """
    # For now, return empty pattern data (210 bytes of zeros = all steps off)
    # The prefix byte (pattern number) is added by the caller
    return bytes(210)


def encode_live_preset(synth, pattern_manager=None, bank=0, 
                       pattern_indices=None, mute_mask=None) -> bytes:
    """Encode current Pythonic synth state to a PO-32 modem packet.
    
    Args:
        synth: PythonicSynthesizer instance
        pattern_manager: PatternManager instance (for pattern data)
        bank: PO-32 bank (0 = instruments 1-8, 1 = instruments 9-16)
        pattern_indices: List of pattern indices to transfer (e.g., [0, 1] for A+B)
        mute_mask: List of 8 bools, True = muted (skip channel)
    """
    encoder = ModemEncoder()
    bank_offset = bank * 8
    
    # --- Patch TLVs (16 total: 8 drums × left + right) ---
    for drum_idx in range(8):
        channel = synth.channels[drum_idx]
        po32_idx = drum_idx + bank_offset
        
        # Left patch (current sound)
        if mute_mask and mute_mask[drum_idx]:
            # Muted channel: use silent defaults
            patch_bytes = b'\x00\x80' * 21
        else:
            patch_bytes = serialize_channel(channel)
        
        left_data = bytes([0x10 | po32_idx]) + patch_bytes
        encoder.append(TAG_PATCH, left_data)
        
        # Right patch (morph target - use defaults)
        right_data = bytes([0x20 | po32_idx]) + _default_right_patch(po32_idx)
        encoder.append(TAG_PATCH, right_data)
    
    # --- Pattern TLVs ---
    # Default: two empty patterns
    if pattern_indices:
        for i, pat_idx in enumerate(pattern_indices):
            encoder.append(TAG_PATTERN, _default_pattern(i))
    else:
        encoder.append(TAG_PATTERN, _default_pattern(0))
        encoder.append(TAG_PATTERN, _default_pattern(1))
    
    # --- State TLV ---
    state = build_state_data(pattern_manager)
    encoder.append(TAG_STATE, state)
    
    # --- Trailer ---
    encoder.append(TAG_TRAILER, b'')
    
    return encoder.get_packet()


def build_state_data(pattern_manager=None) -> bytes:
    """Build state TLV data from pattern manager settings.
    
    State data contains global settings like tempo, swing, step rate, etc.
    This is a 37-byte block. The exact format is partially known from
    reverse engineering.
    """
    # For now, use the default state (all zeros = factory defaults)
    # The PO-32 will use its own tempo/swing settings
    return _default_state()


def generate_transfer_audio(synth, pattern_manager=None, bank=0,
                           pattern_indices=None, mute_mask=None) -> np.ndarray:
    """Generate the complete FSK audio signal for PO-32 transfer.
    
    Returns:
        Float64 audio samples at 44100 Hz, mono
    """
    # Build modem packet
    packet = encode_live_preset(
        synth, pattern_manager, bank, pattern_indices, mute_mask
    )
    
    # Bit-reverse for FSK
    reversed_packet = bit_reverse_bytes(packet)
    
    # Generate FSK audio
    samples = generate_fsk_signal_v2(reversed_packet)
    
    return samples


# =============================================================================
# Transfer Dialog
# =============================================================================

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
    
    def __init__(self, parent, synth, pattern_manager, preset_name="Untitled"):
        """
        Args:
            parent: Parent tkinter window
            synth: PythonicSynthesizer instance
            pattern_manager: PatternManager instance
            preset_name: Current preset name for display
        """
        self.parent = parent
        self.synth = synth
        self.pattern_manager = pattern_manager
        self.preset_name = preset_name
        
        # Transfer state
        self.is_transferring = False
        self.transfer_thread = None
        self.audio_samples = None
        self.transfer_progress = 0.0
        self._stop_event = threading.Event()
        self._closed = False
        
        # Settings
        self.bank = 0  # 0 = instruments 1-8, 1 = instruments 9-16
        self.pattern_chain = [0, 1]  # Pattern indices (A=0, B=1, etc.)
        
        # Get mute states from synth channels
        self.mute_mask = [
            getattr(ch, 'muted', False) for ch in synth.channels
        ]
        
        # Create dialog
        self._create_dialog()
    
    def _create_dialog(self):
        """Build the transfer dialog window"""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title("PO-32 Tonic Transfer")
        self.dialog.configure(bg=self.COLORS['bg'])
        self.dialog.geometry("420x600")
        self.dialog.resizable(False, False)
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
            ch = self.synth.channels[i]
            name = getattr(ch, 'name', f'Drum {i+1}')
            
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
        """Build pattern chain dropdown options from PatternManager."""
        names = self.pattern_manager.PATTERN_NAMES  # ['A', 'B', ..., 'L']
        options = []
        
        # Find chained pattern groups
        i = 0
        while i < len(names):
            start = i
            pattern = self.pattern_manager.patterns[i]
            while i < len(names) - 1 and self.pattern_manager.patterns[i].chained_to_next:
                i += 1
            end = i
            
            if start == end:
                options.append(names[start])
            else:
                options.append(f"{names[start]} - {names[end]}")
            i += 1
        
        # Always ensure at least "A - B" as an option
        if not options:
            options = ["A - B"]
        
        return options
    
    def _parse_pattern_selection(self) -> list:
        """Parse the pattern chain dropdown value into pattern indices."""
        val = self.pattern_var.get()
        names = self.pattern_manager.PATTERN_NAMES
        
        if ' - ' in val:
            parts = val.split(' - ')
            start_name = parts[0].strip()
            end_name = parts[1].strip()
            try:
                start_idx = names.index(start_name)
                end_idx = names.index(end_name)
                return list(range(start_idx, end_idx + 1))
            except ValueError:
                return [0, 1]
        else:
            try:
                idx = names.index(val.strip())
                return [idx]
            except ValueError:
                return [0]
    
    def _on_bank_change(self, event=None):
        """Handle bank selection change"""
        self.bank = 0 if "1 - 8" in self.bank_var.get() else 1
        self._generate_audio()
    
    def _on_pattern_change(self, event=None):
        """Handle pattern chain selection change"""
        self.pattern_chain = self._parse_pattern_selection()
        self._generate_audio()
    
    def _on_channel_toggle(self):
        """Handle channel enable/disable toggle"""
        self.mute_mask = [not var.get() for var in self.channel_vars]
        self._generate_audio()
    
    def _generate_audio(self):
        """Generate the FSK audio signal in background"""
        self.status_label.config(text="Generating audio signal...")
        self.dialog.update_idletasks()
        
        try:
            self.audio_samples = generate_transfer_audio(
                self.synth,
                self.pattern_manager,
                self.bank,
                self.pattern_chain,
                self.mute_mask
            )
            duration = len(self.audio_samples) / SAMPLE_RATE
            self.status_label.config(
                text=f"Ready to transfer ({duration:.1f}s audio)"
            )
        except Exception as e:
            self.status_label.config(text=f"Error: {e}")
            self.audio_samples = None
    
    def _on_transfer_click(self):
        """Handle Transfer/Stop button click"""
        if self.is_transferring:
            self._stop_transfer()
        else:
            self._start_transfer()
    
    def _start_transfer(self):
        """Start audio playback for transfer"""
        if self.audio_samples is None:
            messagebox.showwarning("Transfer Error",
                                   "No audio generated. Check settings.",
                                   parent=self.dialog)
            return
        
        if not AUDIO_AVAILABLE:
            messagebox.showwarning("Audio Not Available",
                                   "sounddevice module not installed.\n"
                                   "Use 'Save WAV' to save the transfer audio.",
                                   parent=self.dialog)
            return
        
        self.is_transferring = True
        self._stop_event.clear()
        
        # Update UI
        self.transfer_btn.config(text="■  Stop", bg=self.COLORS['red'])
        self.save_btn.config(state='disabled')
        self.status_label.config(text="TRANSFERRING...",
                                fg=self.COLORS['green'])
        
        # Start playback in background thread
        self.transfer_thread = threading.Thread(
            target=self._transfer_audio_thread,
            daemon=True
        )
        self.transfer_thread.start()
        
        # Start progress update timer
        self._update_progress()
    
    def _transfer_audio_thread(self):
        """Background thread for audio playback"""
        try:
            samples = self.audio_samples.copy()
            
            # Normalize to ~80% volume for reliable PO-32 reception
            peak = np.max(np.abs(samples))
            if peak > 0:
                samples = samples / peak * 0.85
            
            # Convert to float32 for sounddevice
            samples_f32 = samples.astype(np.float32)
            
            total_samples = len(samples_f32)
            
            # Use a blocking playback with periodic stop checks
            block_size = SAMPLE_RATE  # 1 second blocks
            pos = 0
            
            stream = sd.OutputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype='float32',
                blocksize=1024
            )
            stream.start()
            
            try:
                while pos < total_samples and not self._stop_event.is_set():
                    end = min(pos + block_size, total_samples)
                    chunk = samples_f32[pos:end]
                    stream.write(chunk.reshape(-1, 1))
                    pos = end
                    self.transfer_progress = pos / total_samples * 100
            finally:
                stream.stop()
                stream.close()
            
            if self._closed:
                return
            if not self._stop_event.is_set():
                self.transfer_progress = 100
                self.dialog.after(0, self._transfer_complete)
            else:
                self.dialog.after(0, self._transfer_stopped)
                
        except Exception as e:
            if not self._closed:
                self.dialog.after(0, lambda: self._transfer_error(str(e)))
    
    def _update_progress(self):
        """Update progress bar during transfer"""
        if not self.is_transferring or self._closed:
            return
        
        self.progress_var.set(self.transfer_progress)
        
        if self.transfer_progress < 100:
            self.dialog.after(100, self._update_progress)
    
    def _transfer_complete(self):
        """Called when transfer finishes successfully"""
        self.is_transferring = False
        self.transfer_progress = 0
        self.progress_var.set(100)
        
        self.transfer_btn.config(text="▶  Transfer", bg=self.COLORS['accent'])
        self.save_btn.config(state='normal')
        self.status_label.config(text="Transfer complete!",
                                fg=self.COLORS['green'])
        
        # Reset progress after a moment
        def _reset_status():
            if self._closed:
                return
            self.progress_var.set(0)
            self.status_label.config(
                text=f"Ready to transfer ({len(self.audio_samples) / SAMPLE_RATE:.1f}s audio)",
                fg=self.COLORS['text_dim']
            )
        self.dialog.after(2000, _reset_status)
    
    def _transfer_stopped(self):
        """Called when transfer is stopped by user"""
        self.is_transferring = False
        self.transfer_progress = 0
        self.progress_var.set(0)
        
        self.transfer_btn.config(text="▶  Transfer", bg=self.COLORS['accent'])
        self.save_btn.config(state='normal')
        self.status_label.config(text="Transfer stopped",
                                fg=self.COLORS['orange'])
    
    def _transfer_error(self, error_msg):
        """Called when transfer encounters an error"""
        self.is_transferring = False
        self.transfer_progress = 0
        self.progress_var.set(0)
        
        self.transfer_btn.config(text="▶  Transfer", bg=self.COLORS['accent'])
        self.save_btn.config(state='normal')
        self.status_label.config(text=f"Error: {error_msg}",
                                fg=self.COLORS['red'])
    
    def _stop_transfer(self):
        """Stop ongoing transfer"""
        self._stop_event.set()
    
    def _on_save_wav(self):
        """Save transfer audio as WAV file"""
        if self.audio_samples is None:
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
            try:
                save_wav(self.audio_samples, filepath, SAMPLE_RATE)
                self.status_label.config(
                    text=f"Saved: {os.path.basename(filepath)}",
                    fg=self.COLORS['green']
                )
            except Exception as e:
                messagebox.showerror("Save Error", str(e),
                                    parent=self.dialog)
    
    def _on_close(self):
        """Handle dialog close"""
        self._closed = True
        
        if self.is_transferring:
            self._stop_transfer()
            # Wait briefly for thread to finish
            if self.transfer_thread:
                self.transfer_thread.join(timeout=1.0)
        
        try:
            self.dialog.grab_release()
        except Exception:
            pass
        self.dialog.destroy()
