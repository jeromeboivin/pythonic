"""
Address registry of the app core.

Every value a front-end reads or writes has an address (``audio.sample_rate``,
later ``ch3.osc.decay`` or ``pattern.B.ch2.step5.prob``). An address knows how
to read its value, how to write it (on the audio thread, at block start) and
how to describe it to a front-end. Slices register their addresses here.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence


@dataclass
class Address:
    """One named value of the core, in engine units.

    ``get`` runs on the caller's thread. ``set`` is never called directly by
    the caller: the core queues it and the audio thread runs it at block start.
    An address without ``set`` is read-only.
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
