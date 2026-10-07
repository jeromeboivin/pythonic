"""
The parity guard: every address the core's describe() lists is bound to a
control on the page (an element with ``data-address``) or listed as absent,
with a reason, in parity_absent.json. Slices shrink the list as they bind
controls.
"""

import fnmatch
import json
from pathlib import Path

MANIFEST = Path(__file__).with_name('parity_absent.json')


def absent_patterns():
    return json.loads(MANIFEST.read_text())['absent']


def matching(name, patterns):
    return [p for p in patterns if fnmatch.fnmatchcase(name, p)]


def test_every_address_is_bound_or_listed_absent(panel, core_table):
    bound = panel.bound_addresses()
    patterns = absent_patterns()
    names = set(core_table['describe'])

    unknown = sorted(bound - names)
    assert not unknown, f'controls bound to addresses the core does not have: {unknown}'
    missing = sorted(n for n in names - bound if not matching(n, patterns))
    assert not missing, ('addresses neither bound to a control nor listed in '
                         f'{MANIFEST.name}: {missing}')
    stale = sorted(f'{n} (listed as {matching(n, patterns)})' for n in bound
                   if matching(n, patterns))
    assert not stale, f'bound addresses still listed as absent: {stale}'
    unused = sorted(p for p in patterns if not any(fnmatch.fnmatchcase(n, p) for n in names))
    assert not unused, f'absent patterns that match no address: {unused}'


def test_the_shell_binds_the_tempo(panel):
    assert panel.bound_addresses() == {'global.tempo'}
    assert panel.js("[...document.querySelectorAll('[data-verb]')].map((e) => e.dataset.verb)") \
        == ['transport.toggle']


def test_every_absent_entry_has_a_reason():
    assert all(isinstance(reason, str) and reason.strip()
               for reason in absent_patterns().values())
