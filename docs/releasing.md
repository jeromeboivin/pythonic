# Releasing

A release is a version tag. Pushing `v<version>` runs
`.github/workflows/release.yml`, which builds the standalone app on Linux
(x86_64), Windows (x64) and macOS (Apple Silicon), each in two editions, and
publishes the six downloads as a GitHub Release.

| Edition | What is inside | Size (unpacked) |
|---|---|---|
| light | the app, both interfaces, no PyTorch | about 800 MB |
| ai | also PyTorch (CPU) and the bundled AI models | about 1.5 GB |

## Cut a release

1. Bump `__version__` in `pythonic/__init__.py` and commit.
2. Optional: run the workflow by hand (Actions ▸ Release ▸ Run workflow) and
   try the artifacts it uploads; a manual run builds but never publishes.
3. Tag and push:

   ```bash
   git tag v1.1.0 && git push origin v1.1.0
   ```

The workflow refuses a tag that does not match `__version__`. The release
notes are `packaging/release-notes.md` (how to download and start, per OS)
followed by GitHub's generated list of changes; edit them on the release
page afterwards if needed.

## How the build works

- `packaging/pythonic.spec`: the PyInstaller spec. `PYTHONIC_EDITION`
  (`light` or `ai`) picks the edition. The `pythonic` and `gui` packages are
  collected as source files: numba caches its kernels next to a source file,
  and the AI worker loads `drum_generator.py` and `pattern_generator.py` by
  path. The QML modules QtWebEngine drags in are left out.
- `packaging/pythonic_app.py`: the entry point. It is the `pythonic` command,
  or the AI worker with `--ai-worker` (the app core starts the worker that
  way when frozen). It keeps numba's cache in the user's cache folder and, on
  Linux, falls back on the bundled PortAudio.
- `packaging/build.py`: runs PyInstaller, then packs `dist/release/`
  (`.tar.gz` on Linux, `.zip` on Windows, `.dmg` on macOS).
- `packaging/smoke.py`: starts both interfaces of a build and asks its AI
  worker for a drum patch; the workflow runs it on every build.

The AI models are Git LFS files. The workflow fetches them once per run and
caches them, so releases do not use up the repository's LFS bandwidth.

## Build locally

In a virtual environment with the app installed (`pip install -e .`, or
`".[ml]"` with the CPU build of PyTorch for the ai edition):

```bash
pip install "pyinstaller>=6.10"
python tools/fetch_fonts.py
python packaging/build.py --edition light
python packaging/smoke.py dist/light
```

PyInstaller builds for the system it runs on only: Windows and macOS
downloads come from the workflow.

## Not done yet

- **Signing.** The apps are unsigned: Windows SmartScreen and macOS
  Gatekeeper ask once at first start (the release notes say how to get
  past them). Signing needs an Apple Developer ID (and notarization) and a
  Windows code-signing certificate.
- **Intel Macs and Windows/Linux ARM.** Not built; they run from source.
- **Updates.** The app does not check for new releases.
