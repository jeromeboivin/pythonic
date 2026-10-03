"""
Envelope parameters for Pythonic.

The envelopes themselves run inside :mod:`pythonic.voice`; this object only
holds the attack / decay times that the GUI and presets read and write.
"""

import numpy as np


class Envelope:
    """Attack / decay times in milliseconds."""

    def __init__(self):
        self.attack_ms = 0.0
        self.decay_ms = 316.23

    def set_attack(self, attack_ms: float):
        self.attack_ms = float(np.clip(attack_ms, 0.0, 10000.0))

    def set_decay(self, decay_ms: float):
        self.decay_ms = float(np.clip(decay_ms, 1.0, 10000.0))
