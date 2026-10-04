# Native file and device pickers from the web front-end

Research for [#14](https://github.com/jeromeboivin/pythonic/issues/14) (map: [#1](https://github.com/jeromeboivin/pythonic/issues/1)).
Date: 2026-10-04. Versions: PySide6 6.11.2 (Qt 6.11.2, Chromium 140), Python 3.12.3, sounddevice 0.5.5, mido 1.3.3.
Test machine: Ubuntu 24.04, MATE on X11, xdg-desktop-portal 1.18.4 with the GTK backend (FileChooser portal version 3).

## Answer

- **Recommended: a `QFileDialog` opened from a bridge slot, using the instance API with `open()`, not the static functions.**
  - JS calls a slot such as `pickFile({kind: 'preset-save'})`. Python builds the dialog from a per-kind spec (title, name filters, default folder, suggested name, default suffix). It shows the dialog with `open()` and returns at once. When the dialog closes, the chosen path goes **straight to the core**, for example "save the preset to this path". The page gets the result through the normal poll/frame or a one-off signal.
  - `open()` is window-modal: it blocks input to the panel only. The 60 Hz frame timer keeps running (measured on Linux). On Windows this is the only way to keep it running, because the static functions and `exec()` run a blocking loop that "does not dispatch any QTimers". On macOS a window-modal dialog with a parent opens as a sheet.
  - The core already owns preset I/O by path (ADR 0001). This keeps it that way: no file bytes cross the web channel.
- **`<input type=file>` with `QWebEnginePage.chooseFiles` is the wrong fit.**
  - The page only ever gets a `File` (name + bytes), never a path. Saving would have to go through the web File System Access API.
  - `chooseFiles` is a synchronous virtual function, so any dialog shown from it must be modal and blocking. On Windows that freezes the frame timer.
  - The default implementation ignores filters for save, starts directory pickers nowhere in particular, and is not async. QML's async `FileDialogRequest` has no widgets equivalent.
  - Everything it can do, the slot approach does better.
- **An HTML-drawn file browser fed by the core is not a substitute** for open, save and folder pickers. It loses the OS dialog's places, recent files, mounts, overwrite prompt and accessibility. Inside Flatpak or Snap it cannot reach host files at all without the portal. It is still a good fit for an in-panel **preset list of the chosen preset folder** (core lists the folder, page shows names, click loads), as long as the native dialogs remain for everything else.
- **Device lists come from the core, not from web APIs.**
  - Audio: `sounddevice.query_devices()`. MIDI: `mido.get_input_names()` and `get_output_names()`.
  - The core exposes them as metadata on the device-choice addresses, plus a `rescan` verb. The page shows them in HTML selects.
  - The web equivalents are unusable:
    - `navigator.mediaDevices.enumerateDevices()` returns Chromium's own outputs with empty labels, unrelated to PortAudio's devices.
    - `navigator.requestMIDIAccess()` is rejected with `NotAllowedError`, and QtWebEngine has no MIDI permission type to grant.
- **Linux file dialog flavour depends on the platform theme Qt picks at start-up.** The PySide6 wheel ships the `gtk3` and `xdgdesktopportal` theme plugins.
  - **GTK desktops** (GNOME, MATE, XFCE, Cinnamon, …) get an in-process GTK dialog.
  - **Flatpak and Snap** get the portal automatically.
  - **KDE gets Qt's own widget dialog**, unless `QT_QPA_PLATFORMTHEME=xdgdesktopportal` is set.
  - The app can let users override this with the environment variable. It could also default to the portal on KDE when the portal is running. That is a small decision left open.

## 1. The dialogs the app needs

From the tkinter GUI (`gui/main_window.py`, `gui/po32_*.py`, `gui/drum_generator_dialog.py`):

| Dialog | Kind | Filters | Start folder | Suggested name / suffix |
|---|---|---|---|---|
| Load preset | open | `*.mtpreset`, `*.json`, all | preset folder | — |
| Save preset | save | `*.json`, all | preset folder | suffix `.json` |
| Load drum patch | open | `*.mtdrum`, all | preset folder | — |
| Save drum patch | save | `*.mtdrum`, all | preset folder | `<name>.mtdrum` |
| Select preset folder | directory | — | current preset folder | — |
| Export pattern to MIDI | save | `*.mid`, all | (none) | `pythonic_pattern_<X>.mid` |
| Export pattern to WAV | save | `*.wav`, all | (none) | `pythonic_pattern_<X>.wav` |
| Export drum to WAV | save | `*.wav` | (none) | suffix `.wav` |
| Export all drums to WAV | directory | — | (none) | — |
| AI checkpoint (drum, pattern) | open | `*.pt`, all | folder of current path | — |
| PO-32 import audio | open | `*.wav`, all | (none) | — |
| PO-32 save transfer audio | save | `*.wav`, all | (none) | `<name>.wav` |

All of these are single-file open, single-file save, or existing-directory pickers. None needs multi-select. Every one maps directly onto a `QFileDialog` spec: `fileMode`, `acceptMode`, `nameFilters`, `directory`, `selectFile`, `defaultSuffix`.

## 2. Approach A: `<input type=file>` and `QWebEnginePage.chooseFiles`

### How it works (source)

- Chromium's `RunFileChooser` creates a `FilePickerController` from the request's mode, `default_file_name` and accept types. It then calls the page **deferred via `QTimer::singleShot(0, …)`** so the dialog does not run inside Chromium's message handling ([web_contents_delegate_qt.cpp](https://code.qt.io/cgit/qt/qtwebengine.git/tree/src/core/web_contents_delegate_qt.cpp?h=6.11)).
- `QWebEnginePagePrivate::runFileChooser` calls the virtual `chooseFiles(mode, [defaultFileName], acceptedMimeTypes)` **synchronously**. A non-empty return is accepted, an empty one rejected ([qwebenginepage.cpp](https://code.qt.io/cgit/qt/qtwebengine.git/tree/src/core/api/qwebenginepage.cpp?h=6.11)). The override works from Python (verified).
- Modes: `FileSelectOpen`, `FileSelectOpenMultiple`, `FileSelectUploadFolder`, `FileSelectSave`. "*acceptedMimeTypes* is ignored by the default implementation, but might be used by overrides" ([QWebEnginePage](https://doc-snapshots.qt.io/qtwebengine/qwebenginepage.html)). In fact the widgets default does turn them into name filters for open (next item).
- **The default widgets implementation** is `QWebEngineViewPrivate::chooseFiles` ([qwebengineview.cpp](https://code.qt.io/cgit/qt/qtwebengine.git/tree/src/webenginewidgets/api/qwebengineview.cpp?h=6.11)):
  - **Open:** `QFileDialog::getOpenFileName(view, QString(), oldFiles.first(), filters, …, HideNameFilterDetails)`.
  - **Multiple:** `getOpenFileNames(view, …)`.
  - **Folder:** `getExistingDirectory(view, tr("Select folder to upload"))`, with **no start folder**.
  - **Save:** `getSaveFileName(view, QString(), DownloadLocation + oldFiles.first())`, with **no filters**. When the suggested path is absolute, as it is for `showSaveFilePicker` (below), that concatenation yields a nonsense path under `~/Downloads`. This is an inference from the code and the probe values.
  - All of these are the blocking static functions.
- There is **no async request object for widgets**. The Qt WebEngine Core class list has none ([module page](https://doc-snapshots.qt.io/qtwebengine/qtwebenginecore-module.html)). The async `FileDialogRequest` (`accepted = true`, then `dialogAccept()` or `dialogReject()`) is QML-only, on `WebEngineView.fileDialogRequested` ([FileDialogRequest](https://doc-snapshots.qt.io/qtwebengine/qml-qtwebengine-filedialogrequest.html)).

### What the probe saw

[`native-pickers-probe/probe.py`](native-pickers-probe/probe.py) overrides `chooseFiles` to log and return a prepared path. Its page is served from a secure `app://` scheme, and it clicks with real input events.

| Page action | `chooseFiles` mode | `oldFiles` | `acceptedMimeTypes` | JS gets |
|---|---|---|---|---|
| `<input type=file accept=".json,.mtpreset">` | `FileSelectOpen` | `[""]` | `[".json", ".mtpreset"]` | `File{name, size}`, **no path** |
| `<input multiple accept="audio/wav,.wav">` | `FileSelectOpenMultiple` | `[""]` | `["audio/wav", ".wav"]` | two `File`s |
| `<input webkitdirectory>` | `FileSelectUploadFolder` | `[""]` | `[]` | files with `webkitRelativePath` only |
| `showOpenFilePicker({types, startIn:'documents'})` | `FileSelectOpen` | `["~/Documents/"]` | `[".json", ".mtpreset"]` | a `FileSystemFileHandle` (name only) |
| `showSaveFilePicker({suggestedName, types})` | `FileSelectSave` | `["<last dir>/my kit.mtpreset"]` | `[".json", ".mtpreset"]` | a handle; `createWritable()` wrote the file |
| `showDirectoryPicker({mode:'readwrite'})` | `FileSelectUploadFolder` | `["<last dir>/"]` | `[]` | a directory handle, after a permission request |

- **The File System Access API works in QtWebEngine on a secure custom scheme** (`isSecureContext` is true on `app://`).
  - Its pickers go through the same `chooseFiles`: `SelectFileDialogQt` maps `SELECT_SAVEAS_FILE` to `Save`, folder types to `UploadFolder` and open to `Open`, and passes the type extensions ([select_file_dialog_factory_qt.cpp](https://code.qt.io/cgit/qt/qtwebengine.git/tree/src/core/select_file_dialog_factory_qt.cpp?h=6.11)).
  - Open and save grant read access, and save also grants write access, without asking ([file_system_access_permission_context_qt.cpp](https://code.qt.io/cgit/qt/qtwebengine.git/tree/src/core/file_system_access/file_system_access_permission_context_qt.cpp?h=6.11)).
  - A read-write directory raises `QWebEnginePage.fileSystemAccessRequested`, which the app must `accept()` (Qt 6.4+, [QWebEngineFileSystemAccessRequest](https://doc-snapshots.qt.io/qtwebengine/qwebenginefilesystemaccessrequest.html)). When it is **not handled, the picker fails with `AbortError`** (verified).
  - Chromium also blocks sensitive paths such as the home directory itself and system folders (same source).
- So a pure-web design could open and save files. But the page would hold handles with no paths, and file bytes would have to be shipped to the core through the web channel. The core's path-based preset, export and AI I/O would all need byte-based twins.

## 3. Approach B: `QFileDialog` from a bridge slot (recommended)

- **Static functions** (`getOpenFileName`, `getSaveFileName`, `getExistingDirectory`) each create "a modal file dialog with the given parent widget". "On Windows and macOS, this static function uses the native file dialog and not a QFileDialog". On Windows "the dialog spins a blocking modal event loop that does not dispatch any QTimers" ([QFileDialog](https://doc.qt.io/qt-6/qfiledialog.html)).
- **Instance plus `open()`** "shows the dialog, and connects the slot … to the signal that informs about selection changes". The dialog becomes window-modal and the call returns at once (same page). Everything the app needs is configurable on the instance:
  - `setFileMode(ExistingFile | AnyFile | Directory)` and `setAcceptMode(AcceptOpen | AcceptSave)`
  - `setNameFilters(["Pythonic Preset (*.mtpreset)", "JSON (*.json)", "All files (*)"])`
  - `setDirectory(preset_folder)`, `selectFile(suggested_name)`, `setDefaultSuffix("mtdrum")`
  - `setOption(ShowDirsOnly)`, and `DontConfirmOverwrite` if the core asks itself
- **Caveat on `defaultSuffix` with native dialogs:**
  - `QFileDialog` adds the suffix in `selectedFiles()` after the native dialog has returned (`addDefaultSuffixToUrls(selectedFiles_sys())` in [qfiledialog.cpp](https://code.qt.io/cgit/qt/qtbase.git/tree/src/widgets/dialogs/qfiledialog.cpp?h=6.11)).
  - So a native dialog's overwrite prompt checks the name as typed. If the user types `kit`, the dialog will not warn that `kit.mtdrum` exists.
  - The core's save verb should check for this case. This is an inference from the code.
- **Bridge shape:**
  - The slot `pick(kind)` returns nothing (or a request id). The `finished` and `fileSelected` signals call the core verb directly. The page learns the outcome (for example "preset loaded" or an error) through the usual poll or event path.
  - One dialog at a time: the slot ignores calls while a dialog is open. Window modality already blocks clicks on the view.
  - Last-used folders per kind are a core preference, as the preset folder is today.

## 4. Approach C: an HTML-drawn browser fed by the core

- The core would list directories (`os.scandir`) and the page would draw them in the panel's style.
- **Gains:** a uniform look, and no OS window in front of the panel.
- **Losses:**
  - The OS dialog's sidebar places, recent files, network and removable mounts, typing a path, the overwrite prompt and accessibility.
  - A save-name field, folder creation and filtering all have to be built and tested by hand.
  - **Sandboxing:** under Flatpak or Snap the process cannot see host files except through the FileChooser portal, which only a native dialog uses. An in-app browser would only show the sandbox.
- **Where it fits:** a list of presets and patches **in the configured preset folder**, shown in the panel display. This is browsing, not picking. It complements the native dialogs and needs only a core `list_presets()` read plus the existing load verbs.

## 5. Modality and the frame timer (measured)

Probe: [`native-pickers-probe/modal.py`](native-pickers-probe/modal.py). A 60 Hz `PreciseTimer` calls `runJavaScript('n++')` on a visible `QWebEngineView`. A dialog opens over the view and is closed after about 1.5 s through `QApplication.activeModalWidget().reject()`. The table counts timer ticks while the dialog was up (Linux, X11, MATE).

| Variant | Theme | Modality | Ticks in about 1.5 s | Note |
|---|---|---|---|---|
| static `getSaveFileName` | gtk3 | ApplicationModal | 61 | the static call blocked for 1.64 s, but timers kept firing |
| instance `exec()` | gtk3 | ApplicationModal | 80 | the same |
| instance `open()` | gtk3 | WindowModal | 77 | returned in 0.27 s |
| instance `open()`, `DontUseNativeDialog` | Qt widgets | WindowModal | 74 | |
| instance `open()` | xdgdesktopportal | WindowModal | 86 | returned in 0.2 s |
| static `getSaveFileName` | xdgdesktopportal | ApplicationModal | 88 by close time | `reject()` did not close the portal dialog, so this is a test artifact; timers ran |

- **Linux:** every variant keeps the Qt main loop, the frame timer and page delivery running, because the GTK and portal paths run inside Qt's glib-based event loop. A modal GTK `exec()` uses `gtk_dialog_run`, and a window-modal one uses a `QEventLoop` ([qgtk3dialoghelpers.cpp](https://code.qt.io/cgit/qt/qtbase.git/tree/src/plugins/platformthemes/gtk3/qgtk3dialoghelpers.cpp?h=6.11)).
- **Windows:** a modal native dialog is shown by an idle timer that starts a dialog thread. If `exec()` follows, it "is called directly" on the main thread instead ([qwindowsdialoghelpers.cpp](https://code.qt.io/cgit/qt/qtbase.git/tree/src/plugins/platforms/windows/qwindowsdialoghelpers.cpp?h=6.11)). This is the blocking case the docs warn about, so static functions and `exec()` stop the frame timer. **`open()` takes the thread path, so the main loop keeps running.** Not measured: no Windows machine.
- **macOS:** `WindowModal` with a parent uses `beginSheetModalForWindow`, a sheet attached to the panel window that runs asynchronously. `ApplicationModal` defers to `exec()` and `runModal` ([qcocoafiledialoghelper.mm](https://code.qt.io/cgit/qt/qtbase.git/tree/src/plugins/platforms/cocoa/qcocoafiledialoghelper.mm?h=6.11)). Not measured.
- **Audio is not affected either way.** The PortAudio callback runs on its own thread. Only the UI frame (playhead, meters, CC feedback) could stall, and only on Windows with a blocking dialog.
- The `chooseFiles` route (approach A) has to block, because it must return the list. It therefore always gets the Windows stall.

## 6. Platform behaviour

### Linux

- **Theme choice** happens in `QGuiApplication` at start-up ([qguiapplication.cpp](https://code.qt.io/cgit/qt/qtbase.git/tree/src/gui/kernel/qguiapplication.cpp?h=6.11), [qgenericunixtheme.cpp](https://code.qt.io/cgit/qt/qtbase.git/tree/src/gui/platform/unix/qgenericunixtheme.cpp?h=6.11)). Qt tries the candidates in this order:
  1. `QT_QPA_PLATFORMTHEME` or `-platformtheme`, if set.
  2. `xdgdesktopportal`, if `/.flatpak-info` exists or `SNAP` is set.
  3. From `XDG_CURRENT_DESKTOP`: `kde` for KDE; for GNOME, X-Cinnamon, Pantheon, Unity, MATE, XFCE and LXDE, `gtk3` with `gnome` as fallback.
- **Which themes give a native file dialog:**
  - `gtk3` creates `QGtk3FileDialogHelper`, an in-process GTK 3 dialog, when GTK ≥ 3.15.5 ([qgtk3theme.cpp](https://code.qt.io/cgit/qt/qtbase.git/tree/src/plugins/platformthemes/gtk3/qgtk3theme.cpp?h=6.11)).
  - `xdgdesktopportal` always uses the portal for file dialogs ([qxdgdesktopportaltheme.cpp](https://code.qt.io/cgit/qt/qtbase.git/tree/src/plugins/platformthemes/xdgdesktopportal/qxdgdesktopportaltheme.cpp?h=6.11)).
  - The built-in `kde` and `gnome` themes provide **no** file dialog helper, so `QFileDialog` falls back to Qt's widget dialog.
- **The PySide6 wheel ships `libqgtk3.so` and `libqxdgdesktopportal.so`** in `PySide6/Qt/plugins/platformthemes` (verified). It does not ship the Plasma integration theme.
- **This machine:** `QT_QPA_PLATFORMTHEME=gtk2` is set by the session. Qt has no gtk2 plugin, so it fell through to `gtk3`, which produced GTK dialogs. Setting `QT_QPA_PLATFORMTHEME=xdgdesktopportal` gave the portal dialog.
- **The portal dialog** ([qxdgdesktopportalfiledialog.cpp](https://code.qt.io/cgit/qt/qtbase.git/tree/src/plugins/platformthemes/xdgdesktopportal/qxdgdesktopportalfiledialog.cpp?h=6.11)):
  - It calls `org.freedesktop.portal.FileChooser.OpenFile` or `SaveFile`.
  - It passes `modal`, `multiple`, `directory`, `filters` (from the name filters), `current_filter`, `current_folder`, `current_name` (save), `current_file` (save, existing file) and the parent window identifier (X11 or Wayland), so the dialog attaches to the panel.
  - Directory mode needs FileChooser version ≥ 3. Older portals fall back to the base theme's dialog.
  - If the portal call fails, it falls back to the native dialog.
  - The portal and the GTK and widget dialogs all honour start folder, filters and suggested name. The portal accepts `current_folder` as a hint.
- **Wayland:** the portal parent identifier comes from `QDesktopUnixServices::portalWindowIdentifier`, so parenting works where the compositor supports it. Not measured here (X11 session).

### Windows

- Static functions use the native IFileDialog. Use `open()` (section 5).
- The native dialog is threaded only when the dialog is not `exec()`'d.
- Filters, start folder, suggested name and the overwrite prompt are native features.

### macOS

- `open()` with the panel as parent gives a sheet ("not all platforms display file dialogs with a title bar, so the caption text may not be visible", [QFileDialog](https://doc.qt.io/qt-6/qfiledialog.html)).
- Put the purpose in the dialog's label or prompt rather than relying on the title.

## 7. Device lists (audio and MIDI)

- **Audio:**
  - `sounddevice.query_devices()` returns a `DeviceList` of dicts: `name`, `index`, `hostapi`, `max_input_channels`, `max_output_channels`, `default_samplerate` and latency fields. `query_hostapis()` names the host APIs ([sounddevice docs](https://python-sounddevice.readthedocs.io/en/latest/api/checking-hardware.html)).
  - **Measured:** `query_devices()` takes 0.1–0.2 ms (29 devices), while `import sounddevice` (which runs `Pa_Initialize`) takes about 290 ms. So the list is the one PortAudio built at initialisation.
  - **Rescanning means terminating and re-initialising PortAudio** (`sounddevice._terminate()` then `_initialize()`, which are private helpers). That deallocates "all resources allocated by PortAudio" ([PortAudio API](https://files.portaudio.com/docs/v19-doxydocs/portaudio_8h.html)), so it is a core action done with the stream stopped, never a per-frame read.
- **MIDI:**
  - `mido.get_input_names()` and `get_output_names()` come from the backend (python-rtmidi) ([mido backends](https://mido.readthedocs.io/en/stable/backends/index.html)).
  - **Measured:** 4.5 ms the first time, about 1 ms after (3 inputs, 7 outputs). This is cheap enough for a rescan verb, but not for every frame.
- **How lists reach the page:**
  - They are core data, like any other described value. The audio-device and MIDI-port addresses carry their option lists in `describe()` metadata. A `rescan_devices` verb refreshes them, and the change arrives through the poll like any metadata change.
  - The page renders plain HTML selects in the preferences view.
  - The selection is stored by **name**, not by index, as the tkinter preferences already do (`audio_output_device`, `midi_input_device`), because indices move between runs. Several host APIs can expose the same device name, so the host API name could be stored with it.
- **Web device APIs are not an option:**
  - `enumerateDevices()` on `app://` returned one `audiooutput` with an empty label and no id, because media permission was never granted. Those are Chromium's audio sinks, not PortAudio's.
  - `requestMIDIAccess()` rejected with `NotAllowedError`. `QWebEnginePermission.PermissionType` lists MediaAudioCapture, MediaVideoCapture, MediaAudioVideoCapture, DesktopVideoCapture, DesktopAudioVideoCapture, MouseLock, Notifications, Geolocation, ClipboardReadWrite and LocalFontsAccess, with no MIDI (verified on 6.11.2).
  - Even if they worked, the page would open devices behind the core's back.

## 8. Comparison

| | A: `<input>` / `chooseFiles` | B: `QFileDialog` from a slot | C: HTML browser |
|---|---|---|---|
| Open | yes (page gets bytes, not a path) | yes (path to the core) | yes (path) |
| Save | only with `showSaveFilePicker` and a handle | yes, with overwrite prompt | hand-built |
| Directory | `webkitdirectory` (lists files) or `showDirectoryPicker` (handle; needs `fileSystemAccessRequested`) | `Directory` + `ShowDirsOnly` | hand-built |
| Filters | from `accept`; the default implementation drops them on save | `setNameFilters`, all platforms | hand-built |
| Default folder / name | `startIn` / `suggestedName` arrive in `oldFiles`; the default implementation mangles save paths | `setDirectory` / `selectFile` / `defaultSuffix` | yes |
| Modality | always blocking (synchronous virtual) | `open()`: window-modal, async; sheet on macOS | none (in page) |
| Frame timer during dialog | stalls on Windows | runs everywhere | runs |
| Portals / sandbox | through Qt's dialog, if the override uses one | yes, via the `xdgdesktopportal` theme | cannot see host files in a sandbox |
| Fits core path-based I/O | no | yes | yes |

## Sources

- Qt WebEngine: [QWebEnginePage](https://doc-snapshots.qt.io/qtwebengine/qwebenginepage.html), [QWebEngineFileSystemAccessRequest](https://doc-snapshots.qt.io/qtwebengine/qwebenginefilesystemaccessrequest.html), [FileDialogRequest (QML)](https://doc-snapshots.qt.io/qtwebengine/qml-qtwebengine-filedialogrequest.html), [Core class list](https://doc-snapshots.qt.io/qtwebengine/qtwebenginecore-module.html).
- Qt WebEngine source (6.11): `src/webenginewidgets/api/qwebengineview.cpp`, `src/core/api/qwebenginepage.cpp`, `src/core/web_contents_delegate_qt.cpp`, `src/core/select_file_dialog_factory_qt.cpp`, `src/core/file_picker_controller.cpp`, `src/core/file_system_access/*`.
- Qt base: [QFileDialog](https://doc.qt.io/qt-6/qfiledialog.html), [QGuiApplication](https://doc.qt.io/qt-6/qguiapplication.html). Source (6.11): `qguiapplication.cpp`, `qgenericunixtheme.cpp`, `qkdetheme.cpp`, `qgnometheme.cpp`, `platformthemes/gtk3/*`, `platformthemes/xdgdesktopportal/*`, `platforms/windows/qwindowsdialoghelpers.cpp`, `platforms/cocoa/qcocoafiledialoghelper.mm`, `widgets/dialogs/qfiledialog.cpp`.
- [python-sounddevice: checking hardware](https://python-sounddevice.readthedocs.io/en/latest/api/checking-hardware.html), [PortAudio API reference](https://files.portaudio.com/docs/v19-doxydocs/portaudio_8h.html), [mido backends](https://mido.readthedocs.io/en/stable/backends/index.html).
- Probes: [`native-pickers-probe/`](native-pickers-probe/). Run `probe.py` with `QT_QPA_PLATFORM=offscreen`; `modal.py <static|exec|open|open-nonnative>` needs a display.
