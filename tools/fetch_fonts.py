"""
Download the web interface's bundled fonts and their licences into
pythonic/web/static/fonts/ (all SIL OFL 1.1; see SOURCES.txt there).

    python tools/fetch_fonts.py

Existing files are kept. Commit the results.
"""

import sys
import time
import urllib.request
from pathlib import Path

FONTS = Path(__file__).resolve().parent.parent / 'pythonic' / 'web' / 'static' / 'fonts'
CDN = 'https://cdn.jsdelivr.net'
# Each file from jsDelivr, else from a mirror (unpkg for npm packages, GitHub
# for the Google Fonts licences)
FILES = {
    'BarlowCondensed-Medium.woff2':
        'npm/@fontsource/barlow-condensed@5/files/barlow-condensed-latin-500-normal.woff2',
    'BarlowCondensed-Bold.woff2':
        'npm/@fontsource/barlow-condensed@5/files/barlow-condensed-latin-700-normal.woff2',
    'Doto-Black.woff2': 'npm/@fontsource/doto@5/files/doto-latin-900-normal.woff2',
    'DSEG7Classic-Bold.woff2': 'npm/dseg@0.46.0/fonts/DSEG7-Classic/DSEG7Classic-Bold.woff2',
    'OFL-BarlowCondensed.txt': 'gh/google/fonts@main/ofl/barlowcondensed/OFL.txt',
    'OFL-Doto.txt': 'gh/google/fonts@main/ofl/doto/OFL.txt',
    'OFL-DSEG.txt': 'npm/dseg@0.46.0/DSEG-LICENSE.txt',
}


def urls(path):
    """Where a FILES path can be downloaded from, the CDN first."""
    if path.startswith('npm/'):
        mirror = f'https://unpkg.com/{path[4:]}'
    else:
        repo, _, rest = path[3:].partition('@')
        branch, _, file = rest.partition('/')
        mirror = f'https://raw.githubusercontent.com/{repo}/{branch}/{file}'
    return [f'{CDN}/{path}', mirror]


def download(sources, attempts=3):
    """The file's bytes from the first source that answers; CDNs fail now and
    then, so each round tries them all, waiting longer between rounds."""
    for attempt in range(attempts):
        for url in sources:
            try:
                with urllib.request.urlopen(url, timeout=30) as response:
                    return response.read()
            except Exception as exc:
                error = f'{exc} ({url})'
        if attempt < attempts - 1:
            time.sleep(2 ** attempt * 3)
    raise OSError(error)


def main():
    failed = 0
    for name, path in FILES.items():
        target = FONTS / name
        if target.exists():
            print(f'kept     {name}')
            continue
        try:
            data = download(urls(path))
        except Exception as exc:  # report and go on with the others
            print(f'FAILED   {name}: {exc}')
            failed += 1
            continue
        target.write_bytes(data)
        print(f'fetched  {name} ({len(data)} bytes)')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
