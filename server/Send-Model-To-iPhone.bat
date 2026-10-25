@echo off
setlocal
rem Makes the iPhone's fast swap model (fp16, runs on the Neural Engine) from
rem your copy of inswapper_128 and sends it to the Faceforge app over Wi-Fi.
rem Uses the same Python environment as Start-Faceforge-Server.bat (run that once first).
rem Usage: Send-Model-To-iPhone.bat [cuda|directml]

cd /d "%~dp0.."
set "VARIANT=%~1"
if "%VARIANT%"=="" set "VARIANT=cuda"
set "PY=.venv-server-%VARIANT%\Scripts\python.exe"
if not exist "%PY%" (
  echo [Faceforge] Run server\Start-Faceforge-Server.bat %VARIANT% once first ^(it sets up Python^).
  pause
  exit /b 1
)

rem Same licence terms as the server: InsightFace models are non-commercial research only.
"%PY%" -m faceforge_server.ios_model --accept-model-license %2 %3 %4 %5
pause
