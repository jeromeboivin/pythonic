"""
The program bank of the app core (inventory cluster I): sixteen stored sets
of the eight channel sounds inside the preset.

- ``program.current`` (read-only, 1..16): the program being played and edited.
- ``program.occupied`` (read-only): one bool per program, True once it holds
  sounds.
- ``program.names`` (read-only): one name per program, the words its eight
  channel names start with ("808" for "808 BD", "808 SD", ...), the current
  program's from the sounds playing; "" for an empty program or names with
  nothing in common. Reported whenever a channel name is.
- ``program.select`` (program 1..16): store the sounds into the current
  program, then recall the new one; an empty program gets a copy of the
  current sounds. Applied at block start; one undo step. Patterns are not
  part of a program.
- ``program.restore_factory``: put the factory kits back into programs 1-7
  (505, 707, 808, 909, DMX, LM2, TR-8); when the current program is one of them its
  kit plays at once. Other programs and the patterns stay. One undo step.
"""

import copy

from pythonic import factory

from .registry import Address
from .undo import CHANNELS, PROGRAMS

NUM_PROGRAMS = 16


def program_index(program):
    """Index 0..15 of a program given as 1..16."""
    if isinstance(program, int) and not isinstance(program, bool) and 1 <= program <= NUM_PROGRAMS:
        return program - 1
    raise ValueError(f'not a program: {program!r} (1-{NUM_PROGRAMS})')


def kit_name(channel_names):
    """The words every channel name starts with: '808' for '808 BD', '808 SD'."""
    words = [str(name or '').split() for name in channel_names]
    common = []
    for column in zip(*words):
        if any(word != column[0] for word in column):
            break
        common.append(column[0])
    return ' '.join(common)


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
        reg(Address('program.names', kind='list', get=self.names))

    def verbs(self):
        return {'program.select': self._verb_select,
                'program.restore_factory': self._verb_restore_factory}

    def names(self):
        synth = self._core.synth
        current = synth.get_current_program()
        slots = synth.get_programs_data()['slots']
        result = []
        for i in range(NUM_PROGRAMS):
            if i == current:
                result.append(kit_name(ch.name for ch in synth.channels))
            elif str(i) in slots:
                result.append(kit_name(ch.get('name') for ch in slots[str(i)].get('channels', ())))
            else:
                result.append('')
        return result

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

    def _verb_restore_factory(self):
        kits = factory.kits()
        if not kits:
            raise ValueError('no factory kits installed')
        core = self._core

        def restore():
            synth = core.synth
            data = synth.get_programs_data()
            for i, kit in enumerate(kits):
                data['slots'][str(i)] = copy.deepcopy(kit)
            synth.load_programs_data(data)
            if synth.get_current_program() < len(kits):
                synth.recall_program(synth.get_current_program())
        core.undo.snapshot_op((CHANNELS, PROGRAMS), 'restore factory kits', restore)
        core.note_changes(core.undo.part_addresses((CHANNELS, PROGRAMS)))
        return {'programs': list(range(1, len(kits) + 1))}
