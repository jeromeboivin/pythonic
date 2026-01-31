"""
Main Synthesizer for Pythonic
Manages 8 drum channels with audio output
"""

import numpy as np
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor
from .drum_channel import DrumChannel
from .oscillator import WaveformType, PitchModMode
from .noise import NoiseFilterMode, NoiseEnvelopeMode


class PythonicSynthesizer:
    """
    Main synthesizer class managing 8 drum channels
    """
    
    NUM_CHANNELS = 8
    
    def __init__(self, sample_rate: int = 44100):
        self.sr = sample_rate
        
        # Create 8 drum channels
        self.channels: List[DrumChannel] = [
            DrumChannel(i, sample_rate) for i in range(self.NUM_CHANNELS)
        ]
        
        # Master output
        self.master_volume_db = 0.0  # -inf to +10 dB
        
        # Pre-allocate buffers to avoid allocations in process_audio
        self._output_buffer = np.zeros((4096, 2), dtype=np.float32)
        
        # Thread pool for parallel channel processing (8 workers for 8 channels)
        self._thread_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="audio_")
        self._channel_buffers = [np.zeros((8192, 2), dtype=np.float32) for _ in range(self.NUM_CHANNELS)]
        
        # Current state
        self.selected_channel = 0
        
        # Choke group tracking
        self._choke_group: List[int] = []
        
        # Initialize with factory presets
        self._init_factory_sounds()
    
    def _init_factory_sounds(self):
        """Initialize channels with basic drum sounds"""
        # Channel 0: Kick
        self.channels[0].name = "Kick"
        self.channels[0].set_osc_frequency(55.0)
        self.channels[0].set_osc_waveform(WaveformType.SINE)
        self.channels[0].set_pitch_mod_mode(PitchModMode.DECAYING)
        self.channels[0].set_pitch_mod_amount(36.0)
        self.channels[0].set_pitch_mod_rate(80.0)
        self.channels[0].set_osc_decay(500.0)
        self.channels[0].set_noise_decay(50.0)
        self.channels[0].osc_noise_mix = 0.9
        self.channels[0].distortion = 0.15
        
        # Channel 1: Snare
        self.channels[1].name = "Snare"
        self.channels[1].set_osc_frequency(180.0)
        self.channels[1].set_osc_waveform(WaveformType.TRIANGLE)
        self.channels[1].set_pitch_mod_mode(PitchModMode.DECAYING)
        self.channels[1].set_pitch_mod_amount(12.0)
        self.channels[1].set_pitch_mod_rate(50.0)
        self.channels[1].set_osc_decay(150.0)
        self.channels[1].set_noise_filter_mode(NoiseFilterMode.BAND_PASS)
        self.channels[1].set_noise_filter_freq(2500.0)
        self.channels[1].set_noise_filter_q(2.0)
        self.channels[1].set_noise_decay(180.0)
        self.channels[1].osc_noise_mix = 0.35
        self.channels[1].distortion = 0.2
        
        # Channel 2: Closed Hi-Hat
        self.channels[2].name = "Closed HH"
        self.channels[2].set_osc_frequency(400.0)
        self.channels[2].set_osc_waveform(WaveformType.SAWTOOTH)
        self.channels[2].set_pitch_mod_mode(PitchModMode.RANDOM)
        self.channels[2].set_pitch_mod_amount(2000.0)
        self.channels[2].set_pitch_mod_rate(8000.0)
        self.channels[2].set_osc_decay(60.0)
        self.channels[2].set_noise_filter_mode(NoiseFilterMode.HIGH_PASS)
        self.channels[2].set_noise_filter_freq(6000.0)
        self.channels[2].set_noise_decay(60.0)
        self.channels[2].osc_noise_mix = 0.2
        self.channels[2].choke_enabled = True
        
        # Channel 3: Open Hi-Hat
        self.channels[3].name = "Open HH"
        self.channels[3].set_osc_frequency(400.0)
        self.channels[3].set_osc_waveform(WaveformType.SAWTOOTH)
        self.channels[3].set_pitch_mod_mode(PitchModMode.RANDOM)
        self.channels[3].set_pitch_mod_amount(2000.0)
        self.channels[3].set_pitch_mod_rate(8000.0)
        self.channels[3].set_osc_decay(400.0)
        self.channels[3].set_noise_filter_mode(NoiseFilterMode.HIGH_PASS)
        self.channels[3].set_noise_filter_freq(5000.0)
        self.channels[3].set_noise_stereo(True)
        self.channels[3].set_noise_decay(350.0)
        self.channels[3].osc_noise_mix = 0.15
        self.channels[3].choke_enabled = True
        
        # Channel 4: Tom High
        self.channels[4].name = "Tom Hi"
        self.channels[4].set_osc_frequency(200.0)
        self.channels[4].set_osc_waveform(WaveformType.SINE)
        self.channels[4].set_pitch_mod_mode(PitchModMode.DECAYING)
        self.channels[4].set_pitch_mod_amount(18.0)
        self.channels[4].set_pitch_mod_rate(60.0)
        self.channels[4].set_osc_decay(250.0)
        self.channels[4].set_noise_decay(40.0)
        self.channels[4].osc_noise_mix = 0.85
        
        # Channel 5: Tom Low
        self.channels[5].name = "Tom Lo"
        self.channels[5].set_osc_frequency(120.0)
        self.channels[5].set_osc_waveform(WaveformType.SINE)
        self.channels[5].set_pitch_mod_mode(PitchModMode.DECAYING)
        self.channels[5].set_pitch_mod_amount(18.0)
        self.channels[5].set_pitch_mod_rate(70.0)
        self.channels[5].set_osc_decay(300.0)
        self.channels[5].set_noise_decay(50.0)
        self.channels[5].osc_noise_mix = 0.85
        
        # Channel 6: Clap
        self.channels[6].name = "Clap"
        self.channels[6].set_osc_frequency(1000.0)
        self.channels[6].set_osc_waveform(WaveformType.TRIANGLE)
        self.channels[6].set_osc_decay(20.0)
        self.channels[6].set_noise_filter_mode(NoiseFilterMode.BAND_PASS)
        self.channels[6].set_noise_filter_freq(1500.0)
        self.channels[6].set_noise_filter_q(3.0)
        self.channels[6].set_noise_envelope_mode(NoiseEnvelopeMode.MODULATED)
        self.channels[6].set_noise_decay(50.0)  # Mod frequency
        self.channels[6].osc_noise_mix = 0.1
        
        # Channel 7: Rim/Click
        self.channels[7].name = "Rim"
        self.channels[7].set_osc_frequency(800.0)
        self.channels[7].set_osc_waveform(WaveformType.TRIANGLE)
        self.channels[7].set_pitch_mod_mode(PitchModMode.DECAYING)
        self.channels[7].set_pitch_mod_amount(-24.0)
        self.channels[7].set_pitch_mod_rate(10.0)
        self.channels[7].set_osc_decay(40.0)
        self.channels[7].set_noise_filter_mode(NoiseFilterMode.HIGH_PASS)
        self.channels[7].set_noise_filter_freq(3000.0)
        self.channels[7].set_noise_decay(30.0)
        self.channels[7].osc_noise_mix = 0.6
        self.channels[7].eq_frequency = 2000.0
        self.channels[7].eq_gain_db = 6.0
    
    def trigger_drum(self, channel_idx: int, velocity: int = 127):
        """
        Trigger a drum on the specified channel
        
        Args:
            channel_idx: Channel index (0-7)
            velocity: MIDI velocity (0-127)
        """
        if 0 <= channel_idx < self.NUM_CHANNELS:
            channel = self.channels[channel_idx]
            
            # Handle choke groups
            if channel.choke_enabled:
                self._apply_choke(channel_idx)
            
            # Trigger the drum
            channel.trigger(velocity)
    
    def _apply_choke(self, triggering_channel: int):
        """Apply choke to other choke-enabled channels"""
        for i, channel in enumerate(self.channels):
            if i != triggering_channel and channel.choke_enabled:
                channel.is_active = False
    
    def process_audio(self, num_samples: int) -> np.ndarray:
        """
        Generate audio from all channels (optimized sequential)
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Stereo audio array [num_samples, 2]
        """
        # Use pre-allocated buffer or create if needed
        if num_samples <= len(self._output_buffer):
            output = self._output_buffer[:num_samples]
            output.fill(0)  # Clear buffer
        else:
            output = np.zeros((num_samples, 2), dtype=np.float32)
        
        # Mix only active unmuted channels (sequential is faster for small ops)
        active_count = 0
        for channel in self.channels:
            if channel.is_active and not channel.muted:
                active_count += 1
                channel_output = channel.process(num_samples)
                np.add(output, channel_output, out=output)
        
        # If no active channels, return early
        if active_count == 0:
            return output
        
        # Apply master volume (optimized)
        if self.master_volume_db != 0.0 and self.master_volume_db > -60:
            master_gain = 10.0 ** (self.master_volume_db / 20.0)
            np.multiply(output, master_gain, out=output)
        elif self.master_volume_db <= -60:
            output.fill(0)
            return output
        
        # Note: Channel-level headroom is already applied in DrumChannel.INTERNAL_HEADROOM_LINEAR
        # No additional headroom reduction needed here for accurate reproduction
        
        # Allow peaks up to 1.0
        # Only apply soft clipping if we exceed 1.0 to prevent harsh digital clipping
        peak = np.max(np.abs(output))
        if peak > 1.0:
            # Soft clip using tanh for peaks above 1.0
            # This preserves more RMS energy than linear scaling
            np.tanh(output, out=output)
        
        return output
    
    def select_channel(self, channel_idx: int):
        """Select a channel for editing"""
        if 0 <= channel_idx < self.NUM_CHANNELS:
            self.selected_channel = channel_idx
    
    def get_selected_channel(self) -> DrumChannel:
        """Get the currently selected channel"""
        return self.channels[self.selected_channel]
    
    def set_master_volume(self, volume_db: float):
        """Set master volume in dB"""
        self.master_volume_db = np.clip(volume_db, -60.0, 10.0)
    
    def cleanup(self):
        """Cleanup thread pool resources (call on shutdown)"""
        if hasattr(self, '_thread_pool'):
            self._thread_pool.shutdown(wait=False)
    
    def mute_channel(self, channel_idx: int, muted: bool):
        """Mute or unmute a channel"""
        if 0 <= channel_idx < self.NUM_CHANNELS:
            self.channels[channel_idx].muted = muted
    
    def solo_channel(self, channel_idx: int):
        """Solo a channel (mute all others)"""
        for i, channel in enumerate(self.channels):
            channel.muted = (i != channel_idx)
    
    def unsolo_all(self):
        """Unmute all channels"""
        for channel in self.channels:
            channel.muted = False
    
    def get_preset_data(self) -> dict:
        """Get all synthesizer data as a dictionary for saving"""
        return {
            'version': '1.0',
            'master_volume_db': self.master_volume_db,
            'channels': [ch.get_parameters() for ch in self.channels]
        }
    
    def load_preset_data(self, data: dict):
        """Load synthesizer data from a dictionary"""
        if 'master_volume_db' in data:
            self.master_volume_db = data['master_volume_db']
        
        if 'channels' in data:
            for i, ch_data in enumerate(data['channels']):
                if i < self.NUM_CHANNELS:
                    self.channels[i].set_parameters(ch_data)
