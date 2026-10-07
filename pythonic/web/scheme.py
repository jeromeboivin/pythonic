"""
The ``app://`` URL scheme: serves the front-end files with the right MIME
types, so ES modules load with no build step and the page has a real origin.

``register_scheme()`` must run before the QApplication exists;
``install_handler(profile)`` once per profile (Qt refuses a second handler for
a scheme, and the profile does not own it, so the module keeps it alive).
"""

from pathlib import PurePosixPath

from PySide6.QtCore import QBuffer, QByteArray, QIODevice
from PySide6.QtWebEngineCore import (QWebEngineProfile, QWebEngineUrlRequestJob,
                                     QWebEngineUrlScheme, QWebEngineUrlSchemeHandler)

from . import STATIC_DIR

SCHEME = b'app'
HOST = 'ui'
INDEX_URL = f'app://{HOST}/index.html'

# Module scripts are checked strictly: .js must be text/javascript. The table
# is explicit because mimetypes varies by OS (and the registry on Windows).
MIME_TYPES = {
    '.html': b'text/html',
    '.js': b'text/javascript',
    '.mjs': b'text/javascript',
    '.css': b'text/css',
    '.json': b'application/json',
    '.svg': b'image/svg+xml',
    '.png': b'image/png',
    '.woff2': b'font/woff2',
    '.woff': b'font/woff',
    '.ttf': b'font/ttf',
    '.otf': b'font/otf',
    '.txt': b'text/plain',
}

_registered = False
_handlers = {}  # id(profile) -> (profile, handler), kept for the process lifetime


def register_scheme():
    """Declare the app:// scheme (idempotent; before QApplication exists)."""
    global _registered
    if _registered:
        return
    scheme = QWebEngineUrlScheme(SCHEME)
    scheme.setSyntax(QWebEngineUrlScheme.Syntax.Host)
    flags = QWebEngineUrlScheme.Flag
    scheme.setFlags(flags.SecureScheme | flags.LocalAccessAllowed | flags.CorsEnabled
                    | flags.FetchApiAllowed)
    QWebEngineUrlScheme.registerScheme(scheme)
    _registered = True


class StaticHandler(QWebEngineUrlSchemeHandler):
    """Serves ``app://ui/<path>`` from a folder."""

    def __init__(self, root=STATIC_DIR, parent=None):
        super().__init__(parent)
        self.root = root.resolve()

    def resolve(self, url_path):
        """The file for a URL path, or None (unknown, a folder, or outside)."""
        root = self.root
        path = (root / PurePosixPath(url_path.lstrip('/'))).resolve()
        if root not in path.parents or not path.is_file():
            return None
        return path

    def requestStarted(self, job):
        url = job.requestUrl()
        path = self.resolve(url.path()) if url.host() == HOST else None
        if path is None:
            job.fail(QWebEngineUrlRequestJob.Error.UrlNotFound)
            return
        buffer = QBuffer(parent=job)
        buffer.setData(QByteArray(path.read_bytes()))
        buffer.open(QIODevice.OpenModeFlag.ReadOnly)
        job.reply(MIME_TYPES.get(path.suffix.lower(), b'application/octet-stream'), buffer)


def install_handler(profile=None):
    """The app:// handler of a profile (default profile when None), installed
    on first use."""
    profile = profile or QWebEngineProfile.defaultProfile()
    entry = _handlers.get(id(profile))
    if entry is None:
        handler = StaticHandler()
        profile.installUrlSchemeHandler(SCHEME, handler)
        entry = _handlers[id(profile)] = (profile, handler)
    return entry[1]
