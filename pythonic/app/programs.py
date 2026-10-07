"""
The program bank of the app core (inventory cluster I): sixteen stored sets
of the eight channel sounds inside the preset.

- ``program.current`` (read-only, 1..16): the program being played and edited.
- ``program.occupied`` (read-only): one bool per program, True once it holds
  sounds.
- ``program.select`` (program 1..16): store the sounds into the current
  program, then recall the new one; an empty program gets a copy of the
  current sounds. Applied at block start; one undo step. Patterns are not
  part of a program.
"""

from .registry import Address
from .undo import CHANNELS, PROGRAMS

NUM_PROGRAMS = 16


def program_index(program):
    """Index 0..15 of a program given as 1..16."""
    if isinstance(program, int) and not isinstance(program, bool) and 1 <= program <= NUM_PROGRAMS:
        return program - 1
    raise ValueError(f'not a program: {program!r} (1-{NUM_PROGRAMS})')


class Programs:
    """Program addresses and verbs of an AppCore."""

    def __init__(self, core):
        self._core = core

    def register(self, registry):
        reg = registry.register
        synth = lambda: self._core.synth  # noqa: E731 (replaced on a rate change)
        reg(Address('program.current', get=lambda: synth().get_current_program() + 1,
                    kind='int', minimum=1, maximum=NUM_PROGRAMS, default=1))
        reg(Address('program.occupied', kind='list',
                    get=lambda: [synth().is_program_occupied(i) for i in range(NUM_PROGRAMS)]))

    def verbs(self):
        return {'program.select': self._verb_select}

    def _verb_select(self, program):
        new = program_index(program)
        core = self._core
        if new == core.synth.get_current_program():
            return {'program': new + 1, 'recalled': False}

        def switch():
            synth = core.synth
            synth.store_program(synth.get_current_program())
            if synth.recall_program(new):
                return True
            synth.store_program(new)  # an empty program starts as a copy
            synth._current_program = new
            return False
        recalled = core.undo.snapshot_op((CHANNELS, PROGRAMS), f'program {new + 1}', switch)
        core.note_changes(core.undo.part_addresses((CHANNELS, PROGRAMS)))
        return {'program': new + 1, 'recalled': recalled}
