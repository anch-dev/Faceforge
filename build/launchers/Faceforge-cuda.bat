@echo off
setlocal
cd /d "%~dp0"

rem The CUDA build needs CUDA 12.x and cuDNN 9 installed system-wide.
rem Without them ONNX Runtime silently drops to CPU (single-digit FPS).
set "CUDA_DIR="
for /d %%D in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.*") do set "CUDA_DIR=%%D"
if not defined CUDA_DIR (
  echo [Faceforge] CUDA 12 is not installed.
  echo   Get it: https://developer.nvidia.com/cuda-12-8-0-download-archive
  echo   Or use the directml build instead, which needs no extra installs.
  pause
  exit /b 1
)
set "CUDNN_OK="
for /d %%D in ("C:\Program Files\NVIDIA\CUDNN\v9.*") do set "CUDNN_OK=1"
if not defined CUDNN_OK (
  echo [Faceforge] cuDNN 9 is not installed.
  echo   Get it: https://developer.nvidia.com/cudnn-downloads
  echo   Or use the directml build instead, which needs no extra installs.
  pause
  exit /b 1
)

set "CUDA_PATH=%CUDA_DIR%"
set "PATH=%CUDA_DIR%\bin;%~dp0;%PATH%"
"%~dp0Faceforge.exe" --execution-provider cuda %*
if errorlevel 1 pause
