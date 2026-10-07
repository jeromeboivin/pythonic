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


def bound_on_every_channel(panel):
    """The addresses bound with each channel selected in turn (the edit rack
    binds the selected channel's sound)."""
    bound = set()
    for n in range(1, 9):
        panel.core.post_change('global.channel', n)
        panel.wait_js("document.querySelector('px-knob[data-suffix=\"osc.freq\"]').dataset.address"
                      f" === 'ch{n}.osc.freq'")
        bound |= panel.bound_addresses()
    return bound


def bound_in_setup(panel):
    """The addresses the setup sheet binds, tab by tab."""
    bound = set()
    for tab in ('audio', 'midi', 'synthesis', 'ai'):
        panel.run(f"pythonic.panel.openPage('setup', {{tab: '{tab}'}})")
        panel.wait_js(f"document.querySelector('.setup') && document.querySelector('.setup').dataset.tab === '{tab}'")
        bound |= panel.bound_addresses()
    panel.run("pythonic.setup.close()")
    return bound


def bound_in_pages(panel):
    """The addresses bound by each other registered page (PO-32, AI), opened in turn."""
    bound = set()
    for name in panel.js('pythonic.panel.pages'):
        if name == 'setup':
            continue  # tab by tab: bound_in_setup
        panel.run(f'pythonic.panel.openPage({name!r})')
        panel.wait_js(f'pythonic.panel.drawer.current === {name!r} || pythonic.panel.sheets.current === {name!r}')
        bound |= panel.bound_addresses()
        panel.run(f'if (pythonic.panel.drawer.current === {name!r}) pythonic.panel.drawer.hide({name!r});'
                  f' pythonic.panel.sheets.hide({name!r});')
    return bound


def test_every_address_is_bound_or_listed_absent(panel, core_table):
    bound = bound_on_every_channel(panel) | bound_in_setup(panel) | bound_in_pages(panel)
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


def test_the_rack_binds_every_sound_address_of_every_channel(panel, core_table):
    bound = bound_on_every_channel(panel)
    sound = {name for name in core_table['describe']
             if name.startswith('ch') and name.split('.')[1] in
             ('osc', 'noise', 'mix', 'eq', 'fx', 'vel', 'lfo1', 'lfo2', 'pump')}
    assert len(sound) == 8 * 59
    assert sound <= bound
    assert 'global.edit_all' in bound


def test_the_face_binds_its_controls(panel):
    bound = panel.bound_addresses()
    for n in range(1, 9):
        assert {f'ch{n}.osc.pitch', f'ch{n}.osc.decay', f'ch{n}.mix.level', f'ch{n}.mute',
                f'ch{n}.name', f'ch{n}.mix.pan'} <= bound  # pan: the CTRL knobs' default mode
    assert {'global.tempo', 'global.swing', 'global.step_rate', 'global.fill_rate',
            'global.master', 'global.channel', 'morph.position', 'morph.learning',
            'morph.differs', 'program.current', 'program.occupied', 'undo.can_undo',
            'undo.can_redo', 'midi.connected'} <= bound
    verbs = set(panel.js("[...document.querySelectorAll('[data-verb]')].map((e) => e.dataset.verb)"))
    assert verbs == {'transport.toggle', 'undo', 'redo', 'program.select', 'morph.learn',
                     'pattern.select', 'pattern.chain_prev', 'pattern.chain_next',
                     'pattern.copy_lane', 'pattern.paste_lane'}


def test_the_step_row_binds_every_pattern(panel):
    bound = panel.bound_addresses()
    for p in 'ABCDEFGHIJKL':
        assert {f'pattern.{p}.length', f'pattern.{p}.chained', f'pattern.{p}.empty'} <= bound
    assert 'pattern.selected' in bound


def test_every_absent_entry_has_a_reason():
    assert all(isinstance(reason, str) and reason.strip()
               for reason in absent_patterns().values())
