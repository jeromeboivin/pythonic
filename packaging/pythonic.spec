# PyInstaller spec of the standalone app; packaging/build.py runs it.
#
#     PYTHONIC_EDITION=light|ai pyinstaller packaging/pythonic.spec
#
# light: the app with both interfaces, no PyTorch. ai: also PyTorch and the
# bundled models, laid out as in the source tree (drum_cvae_best.pt at the
# root, drum_patterns/pattern_cvae_best.pt) so pythonic/app/ai.py finds them.
#
# The pythonic and gui packages are collected as .py files, not bytecode in
# the archive: numba caches its kernels next to a real source file, and the
# AI worker loads drum_generator.py and pattern_generator.py by path.

import os
import re
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files

EDITION = os.environ.get('PYTHONIC_EDITION', 'light')
if EDITION not in ('light', 'ai'):
    raise SystemExit(f'PYTHONIC_EDITION must be light or ai, not {EDITION!r}')
ROOT = Path(SPECPATH).resolve().parent
MODELS = [('drum_cvae_best.pt', '.'), ('drum_patterns/pattern_cvae_best.pt', 'drum_patterns')]

datas = collect_data_files('pythonic', excludes=['**/__pycache__', 'web/static/test'])
binaries = []
# Imported by name or path at run time: mido's MIDI backend, and what the AI
# worker (run by path itself) loads by path
hiddenimports = ['mido.backends.rtmidi', 'pythonic.app.ai_worker', 'pythonic.drum_generator',
                 'pythonic.pattern_generator']
excludes = ['matplotlib', 'IPython', 'pytest', 'PyInstaller']

if EDITION == 'ai':
    for name, folder in MODELS:
        path = ROOT / name
        # A Git LFS pointer is a few hundred bytes: the models were not pulled
        if not path.is_file() or path.stat().st_size < 1_000_000:
            raise SystemExit(f'{name} is missing or a Git LFS pointer: run git lfs pull')
        datas.append((str(path), folder))
    hiddenimports.append('torch')
else:
    excludes += ['torch', 'torchvision', 'torchaudio']

if sys.platform.startswith('linux'):
    # PortAudio, for systems without one (packaging/pythonic_app.py falls back on it)
    for folder in ('/usr/lib/x86_64-linux-gnu', '/usr/lib64', '/usr/lib'):
        lib = Path(folder) / 'libportaudio.so.2'
        if lib.exists():
            binaries.append((str(lib), '.'))  # copied under its soname
            break
    else:
        raise SystemExit('libportaudio.so.2 not found: install libportaudio2')

a = Analysis(
    [str(ROOT / 'packaging' / 'pythonic_app.py')],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    module_collection_mode={'pythonic': 'py', 'gui': 'py'},
    noarchive=False,
)

# QtWebEngine needs QtQml and QtQuick, whose hook brings every QML module
# with its Qt libraries (3D, charts, multimedia, ...). The panel is a widget
# app: none of them load.
QML_ONLY = re.compile(
    r'(?:^|[/\\])(?:lib)?Qt6?(?:3D|Charts|DataVisualization|Graphs|Location|Multimedia|Quick3D'
    r'|QuickControls2|QuickTemplates2|QuickDialogs2|QuickParticles|QuickTimeline|QuickEffects'
    r'|QuickVectorImage|QuickLayouts|Labs|Sensors|TextToSpeech|RemoteObjects|Scxml|StateMachine'
    r'|Test|WebView|ShaderTools|VirtualKeyboard|Lottie|SpatialAudio|Pdf|5Compat|WebSockets)')


def needed(entry):
    dest = entry[0].replace('\\', '/')
    qml = dest.startswith(('PySide6/Qt/qml/', 'PySide6/qml/')) or '/qmltooling/' in dest
    return not (qml or QML_ONLY.search(dest))


a.binaries = [b for b in a.binaries if needed(b)]
a.datas = [d for d in a.datas if needed(d)]

if sys.platform.startswith('linux'):
    # Every desktop has ALSA, and the system's own library finds its plugins
    # (PipeWire, PulseAudio): never bundle it
    a.binaries = [b for b in a.binaries if not os.path.basename(b[0]).startswith('libasound.so')]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Pythonic',
    console=False,
    disable_windowed_traceback=False,
    upx=False,
)

coll = COLLECT(exe, a.binaries, a.datas, name='Pythonic', upx=False)

if sys.platform == 'darwin':
    VERSION = re.search(r'__version__ = [\'"]([^\'"]+)',
                        (ROOT / 'pythonic' / '__init__.py').read_text()).group(1)
    app = BUNDLE(
        coll,
        name='Pythonic.app',
        bundle_identifier='io.github.jeromeboivin.pythonic',
        version=VERSION,
        info_plist={
            'CFBundleShortVersionString': VERSION,
            'NSHighResolutionCapable': True,
            'LSMinimumSystemVersion': '14.0',
        },
    )
