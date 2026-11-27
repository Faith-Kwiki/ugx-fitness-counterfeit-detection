@echo off
setlocal
title UGX Banknote Lab
cd /d "%~dp0"

set "PROJECT_DIR=%~dp0"
set "PYTHON=%PROJECT_DIR%.venv\Scripts\python.exe"

rem Reuse the existing desktop environment when this project is open as a Codex worktree.
if not exist "%PYTHON%" set "PYTHON=%USERPROFILE%\OneDrive\Desktop\ugx-fitness-counterfeit-detection\.venv\Scripts\python.exe"

rem On another PC, create a project-local environment the first time the launcher runs.
if not exist "%PYTHON%" (
    where py >nul 2>nul
    if errorlevel 1 (
        echo Python 3.12 is required. Install Python 3.12, then run this button again.
        pause
        exit /b 1
    )
    echo Creating the project environment with Python 3.12...
    py -3.12 -m venv "%PROJECT_DIR%.venv"
    if errorlevel 1 (
        echo Could not create the environment. Check that Python 3.12 is installed.
        pause
        exit /b 1
    )
    set "PYTHON=%PROJECT_DIR%.venv\Scripts\python.exe"
)

"%PYTHON%" -c "import streamlit" >nul 2>nul
if errorlevel 1 (
    echo Installing the project's pinned packages. This can take a while on first launch...
    "%PYTHON%" -m pip install --upgrade pip
    if errorlevel 1 goto install_failed
    "%PYTHON%" -m pip install -r "%PROJECT_DIR%requirements.txt"
    if errorlevel 1 goto install_failed
)

rem If this project's dashboard is already running, open it instead of starting a duplicate.
powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $r = Invoke-WebRequest -UseBasicParsing -Uri 'http://localhost:8501/_stcore/health' -TimeoutSec 2; if ($r.StatusCode -eq 200 -and $r.Content.Trim() -eq 'ok') { exit 0 } } catch {}; exit 1" >nul 2>nul
if not errorlevel 1 (
    start "" "http://localhost:8501"
    exit /b 0
)

echo Starting UGX Banknote Lab...
"%PYTHON%" -m streamlit run "%PROJECT_DIR%app.py" --server.port 8501
if errorlevel 1 (
    echo The dashboard stopped because of an error.
    pause
    exit /b 1
)
exit /b 0

:install_failed
echo Package installation failed. Check your internet connection and try again.
pause
exit /b 1
