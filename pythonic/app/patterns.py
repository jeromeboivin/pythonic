"""
Patterns of the app core: step-lane addresses, pattern ops, selection, the
queued pattern and chains (inventory clusters E and F).

Addresses (patterns ``A``..``L``, channels 1..8, steps 1..64):

- ``pattern.<P>.ch<N>.step<S>.<field>``: one step. Fields: ``trig``, ``acc``,
  ``fill`` (bools), ``prob`` (0..100 %) and ``sub`` (substeps: ``o`` plays,
  ``-`` rests, ``''`` for none). Turning ``trig`` off clears ``acc`` and
  ``fill``. A step past the pattern length reads as empty and ignores sets.
- ``pattern.<P>.ch<N>.<field>``: the whole lane of that field, a list with one
  value per step of the pattern. A shorter list sets the first steps.
- ``pattern.<P>.length`` (1..64), ``pattern.<P>.chained`` (chained to the next
  pattern; read-only on L) and ``pattern.<P>.empty`` (read-only: no trigger at all).
- ``pattern.selected``: the selected pattern (a plain selection; the
  ``pattern.select`` verb is the pattern button, which also queues).

Every change of steps is reported by ``poll`` on the lane addresses (and on
``pattern.<P>.empty``); a set of one step also reports the step address.

Verbs take ``pattern`` as a letter (``'B'``) or an index (0..11), the selected
pattern when omitted, and ``channel`` as 1..8. They run at block start on the
audio thread, so they apply in order with the queued sets:

- ``pattern.cut|copy|paste|exchange`` (pattern clipboard), ``pattern.clear``,
  ``pattern.shift_left|shift_right|reverse``, ``pattern.randomize``,
  ``pattern.alter``, ``pattern.randomize_accents_fills``.
- ``pattern.copy_lane|paste_lane`` (pattern, channel): the lane clipboard,
  triggers, accents, fills and probabilities of one channel (not substeps).
- ``pattern.select`` (pattern): select it; while playing it is also queued
  (the playing pattern cancels the queue), while stopped the position goes
  back to step 1. ``pattern.queue`` (pattern or None) only queues.
- ``pattern.chain_prev|chain_next`` (pattern): toggle the chain from the
  previous pattern or to the next one; ``pattern.chain_clear`` unlinks all.

During playback the selection follows the playing pattern when a chain or the
queue moves on (the audio engine does this at the pattern change).
"""

import re
from dataclasses import dataclass
from typing import Any, Optional

from pythonic.pattern_manager import PatternManager

from .registry import Address

PATTERN_NAMES = tuple(PatternManager.PATTERN_NAMES)
NUM_CHANNELS = 8
MAX_STEPS = 64

_STEP_ADDRESS = re.compile(
    r'pattern\.([A-L])\.ch([1-8])\.(?:step([1-9][0-9]?)\.)?([a-z]+)$')


def _substeps(value):
    if not isinstance(value, str) or any(c not in 'oO-' for c in value):
        raise ValueError(f'substeps {value!r}: use o (play) and - (rest)')
    return value.lower()


@dataclass(frozen=True)
class StepField:
    """One property of a step: address suffix, PatternStep attribute, range."""

    name: str
    attr: str
    kind: str
    default: Any
    minimum: Optional[int] = None
    maximum: Optional[int] = None
    unit: str = ''

    def template(self, name='step'):
        return Address(name, get=lambda: self.default, kind=self.kind,
                       minimum=self.minimum, maximum=self.maximum, default=self.default,
                       unit=self.unit, convert=_substeps if self.kind == 'str' else None)


STEP_FIELDS = {f.name: f for f in (
    StepField('trig', 'trigger', 'bool', False),
    StepField('acc', 'accent', 'bool', False),
    StepField('fill', 'fill', 'bool', False),
    StepField('prob', 'probability', 'int', 100, 0, 100, '%'),
    StepField('sub', 'substeps', 'str', ''),
)}


def _write(step, field, value):
    """Write one field of a PatternStep (turning a trigger off clears its
    accent and fill)."""
    setattr(step, field.attr, value)
    if field.name == 'trig' and not value:
        step.accent = False
        step.fill = False


def pattern_index(pattern, selected=None):
    """Index 0..11 of a pattern given as a letter or an index (None: `selected`)."""
    if pattern is None and selected is not None:
        return selected
    if isinstance(pattern, str) and pattern.upper() in PATTERN_NAMES:
        return PATTERN_NAMES.index(pattern.upper())
    if isinstance(pattern, int) and not isinstance(pattern, bool) \
            and 0 <= pattern < len(PATTERN_NAMES):
        return pattern
    raise ValueError(f'not a pattern: {pattern!r} (A-L or 0-11)')


def channel_index(channel):
    if isinstance(channel, int) and not isinstance(channel, bool) \
            and 1 <= channel <= NUM_CHANNELS:
        return channel - 1
    raise ValueError(f'not a channel: {channel!r} (1-{NUM_CHANNELS})')


def lane_addresses(name):
    """Every lane address of pattern `name`."""
    return [f'pattern.{name}.ch{c}.{f}' for c in range(1, NUM_CHANNELS + 1) for f in STEP_FIELDS]


class Patterns:
    """Pattern addresses and verbs of an AppCore."""

    def __init__(self, core):
        self._core = core
        self._lane_clipboard = None  # the lane clipboard (copy_lane / paste_lane)

    @property
    def pm(self):
        return self._core.pattern_manager

    # ================================================================== addresses
    def register(self, registry):
        reg = registry.register
        pm = self.pm
        for index, name in enumerate(PATTERN_NAMES):
            def pattern(i=index):
                return self.pm.patterns[i]

            reg(Address(f'pattern.{name}.length', get=lambda p=pattern: p().length,
                        set=lambda v, p=pattern: p().set_length(v), kind='int', minimum=1,
                        maximum=MAX_STEPS, default=pm.pattern_length, unit='steps',
                        related=tuple(lane_addresses(name)) + (f'pattern.{name}.empty',)))
            reg(Address(f'pattern.{name}.empty', get=lambda p=pattern: p().is_empty(),
                        kind='bool'))
            last = index == len(PATTERN_NAMES) - 1  # nothing to chain L to
            reg(Address(f'pattern.{name}.chained',
                        get=lambda p=pattern: bool(p().chained_to_next),
                        set=None if last else lambda v, i=index: self._set_chained(i, v),
                        kind='bool', default=False))
        reg(Address('pattern.selected', get=lambda: PATTERN_NAMES[self.pm.selected_pattern_index],
                    set=lambda v: self.pm.select_pattern(PATTERN_NAMES.index(v)),
                    kind='enum', default='A', labels=PATTERN_NAMES))
        registry.add_resolver('pattern.', self._resolve)

    def _resolve(self, name):
        match = _STEP_ADDRESS.match(name)
        if match is None:
            return None
        letter, channel, step, field_name = match.groups()
        field = STEP_FIELDS.get(field_name)
        if field is None:
            return None
        p = PATTERN_NAMES.index(letter)
        c = int(channel) - 1
        lane = f'pattern.{letter}.ch{channel}.{field_name}'
        empty = f'pattern.{letter}.empty'
        if step is None:
            return self._lane_address(name, field, p, c, empty)
        s = int(step) - 1
        if s >= MAX_STEPS:
            return None
        related = [lane, empty]
        if field.name == 'trig':
            related += [f'pattern.{letter}.ch{channel}.step{step}.acc',
                        f'pattern.{letter}.ch{channel}.step{step}.fill',
                        f'pattern.{letter}.ch{channel}.acc',
                        f'pattern.{letter}.ch{channel}.fill']
        template = field.template()

        def get():
            steps = self.pm.patterns[p].channels[c].steps
            return getattr(steps[s], field.attr) if s < len(steps) else field.default

        def set_(value):
            steps = self.pm.patterns[p].channels[c].steps
            if s < len(steps):
                _write(steps[s], field, value)

        return Address(name, get=get, set=set_, kind=field.kind, minimum=field.minimum,
                       maximum=field.maximum, default=field.default, unit=field.unit,
                       convert=template.convert, related=tuple(related))

    def _lane_address(self, name, field, p, c, empty):
        template = field.template()

        def convert(values):
            if isinstance(values, (str, bytes)) or not hasattr(values, '__iter__'):
                raise ValueError(f'{name}: {values!r} is not a list')
            values = [template.coerce(v) for v in values]
            if len(values) > MAX_STEPS:
                raise ValueError(f'{name}: more than {MAX_STEPS} steps')
            return values

        def get():
            return [getattr(step, field.attr) for step in self.pm.patterns[p].channels[c].steps]

        def set_(values):
            for step, value in zip(self.pm.patterns[p].channels[c].steps, values):
                _write(step, field, value)

        related = [empty]
        if field.name == 'trig':
            prefix = name.rsplit('.', 1)[0]
            related += [f'{prefix}.acc', f'{prefix}.fill']
        return Address(name, get=get, set=set_, kind='list', default=None, convert=convert,
                       related=tuple(related))

    def _set_chained(self, index, chained):
        if chained:
            self.pm.chain_patterns(index, index + 1)
        else:
            self.pm.unchain_patterns(index, index + 1)

    # ================================================================== verbs
    def verbs(self):
        pm_ops = {
            'cut': lambda pm, i: pm.cut_pattern(i),
            'clear': lambda pm, i: pm.patterns[i].clear(),
            'shift_left': lambda pm, i: pm.shift_pattern_left(i),
            'shift_right': lambda pm, i: pm.shift_pattern_right(i),
            'reverse': lambda pm, i: pm.reverse_pattern(i),
            'randomize': lambda pm, i: pm.randomize_pattern(i),
            'alter': lambda pm, i: pm.alter_pattern(i),
            'randomize_accents_fills': lambda pm, i: pm.randomize_accents_fills(i),
            'paste': lambda pm, i: {'pasted': pm.paste_pattern(i)},
            'exchange': lambda pm, i: {'exchanged': pm.exchange_pattern(i)},
        }
        verbs = {f'pattern.{name}': self._pattern_op(op) for name, op in pm_ops.items()}
        verbs.update({
            'pattern.copy': self._verb_copy,
            'pattern.copy_lane': self._verb_copy_lane,
            'pattern.paste_lane': self._verb_paste_lane,
            'pattern.select': self._verb_select,
            'pattern.queue': self._verb_queue,
            'pattern.chain_prev': self._verb_chain_prev,
            'pattern.chain_next': self._verb_chain_next,
            'pattern.chain_clear': self._verb_chain_clear,
        })
        return verbs

    def _index(self, pattern):
        return pattern_index(pattern, self.pm.selected_pattern_index)

    def _pattern_op(self, op):
        def verb(pattern=None):
            index = self._index(pattern)
            result = self._core.at_block_start(lambda: op(self.pm, index))
            self._note_pattern(index)
            return result
        return verb

    def _note_pattern(self, index):
        """Report every lane, the length, empty and the chains around a pattern."""
        name = PATTERN_NAMES[index]
        names = lane_addresses(name) + [f'pattern.{name}.length', f'pattern.{name}.empty']
        names += [f'pattern.{PATTERN_NAMES[i]}.chained' for i in (index - 1, index) if i >= 0]
        self._core.note_changes(names)

    def _verb_copy(self, pattern=None):
        index = self._index(pattern)
        self._core.at_block_start(lambda: self.pm.copy_pattern(index))

    def _verb_copy_lane(self, pattern=None, channel=1):
        index, ch = self._index(pattern), channel_index(channel)

        def copy():
            lane = self.pm.patterns[index].channels[ch]
            self._lane_clipboard = {
                'triggers': lane.get_triggers(),
                'accents': lane.get_accents(),
                'fills': lane.get_fills(),
                'probabilities': lane.get_probabilities(),
            }
        self._core.at_block_start(copy)

    def _verb_paste_lane(self, pattern=None, channel=1):
        index, ch = self._index(pattern), channel_index(channel)

        def paste():
            data = self._lane_clipboard
            if not data:
                return False
            lane = self.pm.patterns[index].channels[ch]
            lane.set_triggers(data['triggers'])
            lane.set_accents(data['accents'])
            lane.set_fills(data['fills'])
            lane.set_probabilities(data['probabilities'])
            return True
        pasted = self._core.at_block_start(paste)
        if pasted:
            self._note_pattern(index)
        return {'pasted': pasted}

    # ------------------------------------------------------------------ selection
    def _apply_select(self, index):
        pm = self.pm
        pm.select_pattern(index)
        if pm.is_playing:
            pm.queued_pattern_index = index if index != pm.playing_pattern_index else None
        else:
            pm.play_position = 0
            pm.current_step = 0

    def _apply_queue(self, index):
        pm = self.pm
        if not pm.is_playing:
            return None
        pm.queued_pattern_index = (index if index is not None
                                   and index != pm.playing_pattern_index else None)
        return pm.queued_pattern_index

    def select(self, pattern):
        """The pattern button, from any thread (MIDI program change): queued
        to block start without waiting."""
        self._core.audio.submit_call(self._apply_select, pattern_index(pattern))

    def _verb_select(self, pattern):
        index = pattern_index(pattern)
        self._core.at_block_start(lambda: self._apply_select(index))
        return {'selected': PATTERN_NAMES[index]}

    def _verb_queue(self, pattern=None):
        index = None if pattern is None else pattern_index(pattern)
        queued = self._core.at_block_start(lambda: self._apply_queue(index))
        return {'queued': None if queued is None else PATTERN_NAMES[queued]}

    # ------------------------------------------------------------------ chains
    def _chain_verb(self, pattern, toggle, valid):
        index = pattern_index(pattern, self.pm.selected_pattern_index)
        if not valid(index):
            return {'chained': False}
        chained = self._core.at_block_start(lambda: toggle(index))
        self._core.note_changes([f'pattern.{PATTERN_NAMES[i]}.chained' for i in (index - 1, index)
                                 if i >= 0])
        return {'chained': bool(chained)}

    def _verb_chain_prev(self, pattern=None):
        return self._chain_verb(pattern, self.pm.toggle_chain_from_prev, lambda i: i > 0)

    def _verb_chain_next(self, pattern=None):
        return self._chain_verb(pattern, self.pm.toggle_chain_to_next,
                                lambda i: i < len(PATTERN_NAMES) - 1)

    def _verb_chain_clear(self):
        def clear():
            for pattern in self.pm.patterns:
                pattern.chained_to_next = False
                pattern.chained_from_prev = False
        self._core.at_block_start(clear)
        self._core.note_changes([f'pattern.{n}.chained' for n in PATTERN_NAMES])

    # ================================================================== poll
    def chain(self):
        """Indexes of the chain of the playing pattern (the selected one when
        stopped); empty when it is not chained."""
        pm = self.pm
        index = pm.playing_pattern_index if pm.is_playing else pm.selected_pattern_index
        if not pm.is_in_chain(index):
            return []
        return pm.get_chain_patterns(index)
