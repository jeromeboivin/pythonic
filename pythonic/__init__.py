"""
Pythonic
A Python implementation of the Sonic Charge Pythonic drum synthesizer
"""

__version__ = "1.0.0"
__author__ = "Python Implementation"

from .synthesizer import PythonicSynthesizer
from .drum_channel import DrumChannel
from .oscillator import Oscillator, WaveformType, PitchModMode
from .noise import NoiseGenerator, NoiseFilterMode, NoiseEnvelopeMode
from .envelope import Envelope
from .filter import StateVariableFilter, FilterMode

__all__ = [
    'PythonicSynthesizer',
    'DrumChannel', 
    'Oscillator',
    'WaveformType',
    'PitchModMode',
    'NoiseGenerator',
    'NoiseFilterMode',
    'NoiseEnvelopeMode',
    'Envelope',
    'StateVariableFilter',
    'FilterMode'
]
