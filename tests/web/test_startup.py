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



def test_rendering_turns_the_gpu_off_when_asked_and_offscreen():
    from pythonic.web.app import prepare_rendering

    environ = {'QTWEBENGINE_CHROMIUM_FLAGS': '--foo'}
    prepare_rendering(gpu=False, environ=environ)
    assert environ['QTWEBENGINE_CHROMIUM_FLAGS'] == '--foo --disable-gpu'
    prepare_rendering(gpu=False, environ=environ)  # never added twice
    assert environ['QTWEBENGINE_CHROMIUM_FLAGS'] == '--foo --disable-gpu'

    offscreen = {'QT_QPA_PLATFORM': 'offscreen'}
    prepare_rendering(gpu=True, environ=offscreen)
    assert offscreen['QTWEBENGINE_CHROMIUM_FLAGS'] == '--disable-gpu'

    display = {}
    prepare_rendering(gpu=True, environ=display)
    assert 'QTWEBENGINE_CHROMIUM_FLAGS' not in display
