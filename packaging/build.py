"""
Build the standalone app and pack it for download.

    python packaging/build.py [--edition light|ai]

Runs PyInstaller (pythonic.spec) in the current environment, which must
have the app installed (``pip install -e .``, plus ``.[ml]`` for the ai
edition) and ``pyinstaller``. The download lands in ``dist/release/``:

- Linux: ``Pythonic-<version>-linux-x86_64[-ai].tar.gz`` (a ``Pythonic``
  folder; run ``Pythonic/Pythonic``)
- Windows: ``Pythonic-<version>-windows-x64[-ai].zip`` (run ``Pythonic.exe``)
- macOS: ``Pythonic-<version>-macos-arm64[-ai].dmg`` (drag to Applications)

The script prints the archive's path last.
"""

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / 'packaging' / 'pythonic.spec'


def version():
    text = (ROOT / 'pythonic' / '__init__.py').read_text()
    return re.search(r'__version__ = [\'"]([^\'"]+)', text).group(1)


def platform_tag():
    machine = platform.machine().lower()
    if sys.platform == 'win32':
        return 'windows-' + ('arm64' if machine in ('arm64', 'aarch64') else 'x64')
    if sys.platform == 'darwin':
        return 'macos-' + ('arm64' if machine == 'arm64' else 'x86_64')
    return f'linux-{machine}'


def archive_name(edition):
    suffix = '-ai' if edition == 'ai' else ''
    return f'Pythonic-{version()}-{platform_tag()}{suffix}'


def pyinstaller(edition, dist, work):
    env = dict(os.environ, PYTHONIC_EDITION=edition)
    subprocess.run([sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean',
                    '--distpath', str(dist), '--workpath', str(work), str(SPEC)],
                   cwd=ROOT, env=env, check=True)


def pack_tar(folder, target):
    with tarfile.open(target, 'w:gz') as tar:
        tar.add(folder, arcname=folder.name)


def pack_zip(folder, target):
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(folder.rglob('*')):
            archive.write(path, path.relative_to(folder.parent))


def pack_dmg(app, target, staging):
    """A disk image holding the app and a link to /Applications."""
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    # ditto keeps the bundle's symlinks and signature intact
    subprocess.run(['ditto', str(app), str(staging / app.name)], check=True)
    os.symlink('/Applications', staging / 'Applications')
    subprocess.run(['hdiutil', 'create', '-volname', 'Pythonic', '-srcfolder', str(staging),
                    '-ov', '-format', 'UDZO', str(target)], check=True)
    shutil.rmtree(staging)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--edition', choices=('light', 'ai'), default='light')
    parser.add_argument('--version', action='store_true', help='print the version and exit')
    args = parser.parse_args(argv)
    if args.version:
        print(version())
        return 0

    dist = ROOT / 'dist' / args.edition
    work = ROOT / 'build' / args.edition
    pyinstaller(args.edition, dist, work)

    release = ROOT / 'dist' / 'release'
    release.mkdir(parents=True, exist_ok=True)
    name = archive_name(args.edition)
    if sys.platform == 'darwin':
        target = release / f'{name}.dmg'
        pack_dmg(dist / 'Pythonic.app', target, work / 'dmg')
    elif sys.platform == 'win32':
        target = release / f'{name}.zip'
        pack_zip(dist / 'Pythonic', target)
    else:
        target = release / f'{name}.tar.gz'
        pack_tar(dist / 'Pythonic', target)
    print(target)
    return 0


if __name__ == '__main__':
    sys.exit(main())
