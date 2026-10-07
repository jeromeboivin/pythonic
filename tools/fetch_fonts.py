"""
Download the web interface's bundled fonts and their licences into
pythonic/web/static/fonts/ (all SIL OFL 1.1; see SOURCES.txt there).

    python tools/fetch_fonts.py

Existing files are kept. Commit the results.
"""

import sys
import urllib.request
from pathlib import Path

FONTS = Path(__file__).resolve().parent.parent / 'pythonic' / 'web' / 'static' / 'fonts'
CDN = 'https://cdn.jsdelivr.net'
FILES = {
    'BarlowCondensed-Medium.woff2':
        f'{CDN}/npm/@fontsource/barlow-condensed@5/files/barlow-condensed-latin-500-normal.woff2',
    'BarlowCondensed-Bold.woff2':
        f'{CDN}/npm/@fontsource/barlow-condensed@5/files/barlow-condensed-latin-700-normal.woff2',
    'Doto-Black.woff2': f'{CDN}/npm/@fontsource/doto@5/files/doto-latin-900-normal.woff2',
    'DSEG7Classic-Bold.woff2': f'{CDN}/npm/dseg@0.46.0/fonts/DSEG7-Classic/DSEG7Classic-Bold.woff2',
    'OFL-BarlowCondensed.txt': f'{CDN}/gh/google/fonts@main/ofl/barlowcondensed/OFL.txt',
    'OFL-Doto.txt': f'{CDN}/gh/google/fonts@main/ofl/doto/OFL.txt',
    'OFL-DSEG.txt': f'{CDN}/npm/dseg@0.46.0/DSEG-LICENSE.txt',
}


def main():
    failed = 0
    for name, url in FILES.items():
        target = FONTS / name
        if target.exists():
            print(f'kept     {name}')
            continue
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read()
        except Exception as exc:  # report and go on with the others
            print(f'FAILED   {name}: {exc} ({url})')
            failed += 1
            continue
        target.write_bytes(data)
        print(f'fetched  {name} ({len(data)} bytes)')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
