@echo off
echo Starting GridKey Optimizer Service...

:: Set PYTHONPATH to include current directory
set PYTHONPATH=%CD%

:: Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python not found in PATH.
    pause
    exit /b 1
)

:: Install dependencies if flag provided
if "%1"=="--install" (
    echo Installing dependencies...
    pip install -r requirements.txt
    pip install -r requirements-api.txt
)

:: Run Uvicorn server
echo Server running at http://localhost:8000
echo Documentation at http://localhost:8000/docs
echo Press Ctrl+C to stop.
echo.

uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
