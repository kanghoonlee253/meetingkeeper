# PyInstaller build: .venv\Scripts\pyinstaller MeetingKeeper.spec --noconfirm
# Models are NOT bundled here; build.ps1 copies models\ next to the exe.
from PyInstaller.utils.hooks import collect_all, collect_data_files

datas, binaries, hiddenimports = [], [], []
for pkg in ("faster_whisper", "ctranslate2", "llama_cpp", "sherpa_onnx", "onnxruntime", "soxr",
            "pyaudiowpatch", "sentencepiece", "tokenizers"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h
datas += collect_data_files("docx")

a = Analysis(
    ["app.py"],
    datas=datas,
    binaries=binaries,
    hiddenimports=hiddenimports,
    excludes=["torch", "tensorflow", "matplotlib", "tkinter", "IPython", "pandas", "scipy",
              "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtQuick",
              "PySide6.QtQml", "PySide6.Qt3DCore", "PySide6.QtMultimedia", "PySide6.QtPdf"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="MeetingKeeper", console=False, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="MeetingKeeper", upx=False)
