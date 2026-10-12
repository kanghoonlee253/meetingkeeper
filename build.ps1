# Builds dist\MeetingKeeper (exe + models + 사용설명서) and dist\MeetingKeeper.zip for the company laptop.
# Usage (from the project folder):  powershell -ExecutionPolicy Bypass -File build.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

.\.venv\Scripts\pyinstaller.exe MeetingKeeper.spec --noconfirm
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

$m = "dist\MeetingKeeper\models"
New-Item -ItemType Directory -Force "$m\llm" | Out-Null
Copy-Item -Recurse -Force models\whisper, models\speaker $m
if (Test-Path models\nllb-1.3b-ct2) { Copy-Item -Recurse -Force models\nllb-1.3b-ct2 $m }
Copy-Item -Force models\llm\gemma-3-4b-it-q4_k_m.gguf "$m\llm\"
Copy-Item -Force 사용설명서.txt dist\MeetingKeeper\

$p = Start-Process -FilePath "dist\MeetingKeeper\MeetingKeeper.exe" -ArgumentList "--selftest" -PassThru -Wait
Get-Content "dist\MeetingKeeper\data\selftest.txt" -Encoding UTF8
if ($p.ExitCode -ne 0) { Write-Warning "Self-test reported a failure (see above)" }
Remove-Item -Recurse -Force "dist\MeetingKeeper\data"

if (Test-Path dist\MeetingKeeper.zip) { Remove-Item dist\MeetingKeeper.zip }
.\.venv\Scripts\python.exe -c "import os,zipfile; os.chdir('dist'); z=zipfile.ZipFile('MeetingKeeper.zip','w',zipfile.ZIP_STORED,allowZip64=True); [z.write(os.path.join(r,f)) for r,_,fs in os.walk('MeetingKeeper') for f in fs]; z.close()"
"Done: dist\MeetingKeeper.zip"
