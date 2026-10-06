"""
Address registry of the app core.

Every value a front-end reads or writes has an address (``audio.sample_rate``,
later ``ch3.osc.decay`` or ``pattern.B.ch2.step5.prob``). An address knows how
to read its value, how to write it (on the audio thread, at block start) and
how to describe it to a front-end. Slices register their addresses here.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence


@dataclass
class Address:
    """One named value of the core, in engine units.

    ``get`` runs on the caller's thread. ``set`` is never called directly by
    the caller: the core queues it and the audio thread runs it at block start.
    An address without ``set`` is read-only. ``queued=False`` marks core state
    the audio thread never reads (a mode such as Edit all): its ``set`` runs at
    once on the caller's thread.

    Enum values are the names in ``labels``; ``coerce`` also takes an index.
    """

    name: str
    get: Callable[[], Any]
    set: Optional[Callable[[Any], None]] = None
    kind: str = 'float'  # float, int, bool, enum, str, list
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    default: Any = None
    unit: str = ''
    curve: str = 'linear'
    labels: Sequence[str] = field(default_factory=tuple)
    queued: bool = True

    @property
    def readonly(self) -> bool:
        return self.set is None

    def describe(self) -> Dict[str, Any]:
        return {
            'address': self.name,
            'kind': self.kind,
            'minimum': self.minimum,
            'maximum': self.maximum,
            'default': self.default,
            'unit': self.unit,
            'curve': self.curve,
            'labels': list(self.labels),
            'readonly': self.readonly,
        }

    def coerce(self, value):
        """The value this address accepts for `value`: numbers clamped to the
        range, enum indexes turned into names. Raises ValueError otherwise."""
        kind = self.kind
        if kind == 'bool':
            return bool(value)
        if kind == 'enum':
            labels = list(self.labels)
            if isinstance(value, str) and value in labels:
                return value
            if (isinstance(value, int) and not isinstance(value, bool)
                    and 0 <= value < len(labels)):
                return labels[value]
            raise ValueError(f'{self.name}: {value!r} is not one of {labels}')
        if kind in ('float', 'int'):
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise ValueError(f'{self.name}: {value!r} is not a number') from None
            if not math.isfinite(number):
                raise ValueError(f'{self.name}: {value!r} is not a finite number')
            if kind == 'int':
                number = int(round(number))
            if self.minimum is not None and number < self.minimum:
                number = self.minimum
            if self.maximum is not None and number > self.maximum:
                number = self.maximum
            return number
        return value


class Registry:
    """Name -> Address table. Lookups of unknown names raise KeyError."""

    def __init__(self):
        self._addresses: Dict[str, Address] = {}

    def register(self, address: Address) -> Address:
        if address.name in self._addresses:
            raise ValueError(f'address already registered: {address.name}')
        self._addresses[address.name] = address
        return address

    def __contains__(self, name: str) -> bool:
        return name in self._addresses

    def __getitem__(self, name: str) -> Address:
        try:
            return self._addresses[name]
        except KeyError:
            raise KeyError(f'unknown address: {name}') from None

    def names(self, prefix: str = '') -> List[str]:
        return sorted(n for n in self._addresses if n.startswith(prefix))
