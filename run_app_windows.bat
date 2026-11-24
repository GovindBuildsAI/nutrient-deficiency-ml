@echo off
REM Starts the app at http://127.0.0.1:8000
cd /d "%~dp0"
set PYTHONPATH=src
start "" http://127.0.0.1:8000
.venv\Scripts\python -m uvicorn app:app --host 127.0.0.1 --port 8000
