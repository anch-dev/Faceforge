@echo off
setlocal
cd /d "%~dp0"
set "PATH=%~dp0;%PATH%"
"%~dp0Faceforge.exe" --execution-provider __PROVIDER__ %*
if errorlevel 1 pause
