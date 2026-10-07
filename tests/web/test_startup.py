"""Start-up of the web interface as `pythonic --ui web --quit-after` runs it."""

from pythonic.app import AppCore
from tests.fake_audio import FakeAudioBackend


def test_run_starts_the_core_loads_the_page_and_closes_cleanly(qapp, prefs, capsys):
    from pythonic.web.app import run

    backend = FakeAudioBackend()
    core = AppCore(preferences=prefs, audio_backend=backend, midi_backend=None,
                   stall_timeout=None)
    assert run(quit_after=2.0, core=core) == 0
    assert backend.stream.aborted  # closing the window closed the core
    assert core.get('audio.running') is False
    report = capsys.readouterr().err
    assert 'UI frames: ' in report and 'over the 4 ms budget' in report
