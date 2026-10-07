"""
The sound morph of the app core (inventory cluster J): a blend of the sounds
of all eight channels between two morph endpoints, A and B.

- ``morph.position`` (0..1): where the morph sits between A (0) and B (1);
  setting it blends the channels (while learning, it only moves the position:
  the sound stays on the endpoint being learned). The blended channel values
  are not undo entries: undo restores the position and the sound follows.
- ``morph.learning`` (read-only: ``off``, ``a``, ``b``): morph learn.
- ``morph.differs`` (read-only): the endpoints hold different sounds (a
  front-end enables its morph control then, or while learning).
- ``morph.learn`` (endpoint ``'a'``, ``'b'`` or None): start learning an
  endpoint (the sound jumps to it, and edits are captured into it when learn
  stops) or stop. Stopping, or switching to the other endpoint, captures the
  sounds into the endpoint learned: one undo step.
- ``morph.capture`` (endpoint): store the current sounds as an endpoint, one
  undo step.

LFO and pump modulation with the Morph destination move the morph during
each block (the synth holds the core's morph manager); the position stays.
"""

from .registry import Address
from .undo import MORPH

LEARN_MODES = ('off', 'a', 'b')


def _endpoint(endpoint):
    if endpoint in ('a', 'b', 'A', 'B'):
        return endpoint.lower()
    raise ValueError(f'not a morph endpoint: {endpoint!r} (a or b)')


class Morph:
    """Morph addresses and verbs of an AppCore."""

    def __init__(self, core):
        self._core = core

    @property
    def manager(self):
        return self._core.morph_manager

    def register(self, registry):
        reg = registry.register

        def set_position(position):
            morph = self.manager
            if morph.is_learning():
                morph._position = position  # the synth stays on the learned endpoint
            else:
                morph.set_position(position)

        reg(Address('morph.position', get=lambda: float(self.manager.position),
                    set=set_position, minimum=0.0, maximum=1.0, default=0.0, unit='ratio'))
        reg(Address('morph.learning', get=lambda: self.manager.get_learn_mode() or 'off',
                    kind='enum', default='off', labels=LEARN_MODES))
        reg(Address('morph.differs', get=lambda: self.manager.has_different_endpoints(),
                    kind='bool'))

    def verbs(self):
        return {'morph.learn': self._verb_learn, 'morph.capture': self._verb_capture}

    def _verb_learn(self, endpoint=None):
        new = None if endpoint is None else _endpoint(endpoint)
        core = self._core
        current = self.manager.get_learn_mode()
        if new == current:
            return {'learning': new or 'off'}

        def switch():
            morph = self.manager
            if morph.is_learning():
                morph.stop_learn()  # captures the learned endpoint
            if new == 'a':
                morph.start_learn_a()
            elif new == 'b':
                morph.start_learn_b()
            morph.apply_effective_position()

        if current is None:
            core.at_block_start(switch)  # starting learn changes no saved value
        else:
            core.undo.snapshot_op((MORPH,), f'morph learn {current.upper()}', switch)
        core.note_changes(core.undo.part_addresses((MORPH,)))
        return {'learning': new or 'off'}

    def _verb_capture(self, endpoint):
        endpoint = _endpoint(endpoint)
        core = self._core

        def capture():
            if endpoint == 'a':
                self.manager.capture_endpoint_a()
            else:
                self.manager.capture_endpoint_b()
        core.undo.snapshot_op((MORPH,), f'morph capture {endpoint.upper()}', capture)
        core.note_changes(self._core.registry.names('morph.'))
        return {'endpoint': endpoint}
