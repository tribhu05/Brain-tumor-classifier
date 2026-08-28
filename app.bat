@echo off
echo ======================================================================
echo   Launching Brain Tumor Classifier Web UI (Flask Server)
echo   VIT Bhopal University -- School of Computing Science Engineering & AI
echo ======================================================================
echo.

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found. Please run setup.bat first.
    pause
    exit /b 1
)

echo Starting web server at http://localhost:5000 ...
echo Press Ctrl+C in this window to stop the server.
echo.
start http://localhost:5000
.\venv\Scripts\python.exe scripts/app.py
