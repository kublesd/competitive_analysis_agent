@echo off
setlocal

cd /d "%~dp0"
title AI Competitive Analysis Agent

echo Starting AI Competitive Analysis Agent...
echo The browser will open automatically after the service is ready.
echo Close this window or press Ctrl+C to stop the service.
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1"
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Startup failed with exit code %EXIT_CODE%.
    echo Review the error message above, then press any key to close.
    pause >nul
)

endlocal
