@echo off
setlocal
rem Faceforge offload server for the iPhone app.
rem First run: creates a private Python environment and installs packages (about 2 GB with CUDA).
rem Usage: Start-Faceforge-Server.bat [cuda|directml]

cd /d "%~dp0.."
set "VARIANT=%~1"
if "%VARIANT%"=="" set "VARIANT=cuda"

where py >nul 2>nul
if errorlevel 1 (
  echo [Faceforge] Python 3.11 is required: https://www.python.org/downloads/release/python-3119/
  echo   Tick "Add python.exe to PATH" during install, then run this again.
  pause
  exit /b 1
)

if not exist ".venv-server-%VARIANT%\Scripts\python.exe" (
  echo [Faceforge] Creating Python environment for %VARIANT%...
  py -3.11 -m venv ".venv-server-%VARIANT%" || (echo Python 3.11 not found. & pause & exit /b 1)
  ".venv-server-%VARIANT%\Scripts\python.exe" -m pip install --upgrade pip
  ".venv-server-%VARIANT%\Scripts\python.exe" -m pip install -r "server\requirements-%VARIANT%.txt"
  if errorlevel 1 (
    echo [Faceforge] Package install failed.
    if "%VARIANT%"=="cuda" echo   Try: Start-Faceforge-Server.bat directml
    pause
    exit /b 1
  )
)

rem The model download needs your explicit agreement to InsightFace's licence:
rem non-commercial research use only. Remove --accept-model-license to be asked
rem to place the files yourself instead.
".venv-server-%VARIANT%\Scripts\python.exe" -m faceforge_server --accept-model-license %2 %3 %4 %5
pause
