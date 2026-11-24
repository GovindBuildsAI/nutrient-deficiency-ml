@echo off
REM One-time setup: creates a virtual environment and installs dependencies.
cd /d "%~dp0"
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
%PY% -m venv .venv || (echo Python 3.10+ not found. Install from python.org and tick "Add to PATH". & exit /b 1)
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt || exit /b 1
echo.
echo Setup done. Now double-click run_app_windows.bat
