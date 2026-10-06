"""
UI-free app core of Pythonic (ADR 0001): owns the synth, patterns, morph,
preferences and the audio stream, behind an address-based interface with a
command queue in and ``poll`` out.
"""

from .audio import AudioUnavailable
from .core import AppCore
from .registry import Address, Registry

__all__ = ['AppCore', 'Address', 'Registry', 'AudioUnavailable']
