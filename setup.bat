@echo off
REM ============================================================
REM  Brain Tumor Classifier - One-Click Windows Setup
REM  Double-click this file to set up everything automatically.
REM ============================================================
cd /d "%~dp0"

echo.
echo === Step 1: Checking Python 3.11/3.10 installation ===
py -3.11 --version >nul 2>&1
if not errorlevel 1 (
    set PY_CMD=py -3.11
    goto CREATE_VENV
)
py -3.10 --version >nul 2>&1
if not errorlevel 1 (
    set PY_CMD=py -3.10
    goto CREATE_VENV
)
python --version >nul 2>&1
if not errorlevel 1 (
    set PY_CMD=python
    goto CREATE_VENV
)

echo.
echo [ERROR] No suitable Python installation found.
echo Please install Python 3.11 from https://www.python.org/downloads/
echo (Ensure "Add Python to PATH" is checked during installation).
pause
exit /b 1

:CREATE_VENV
echo Using Python launcher: %PY_CMD%
echo.
echo === Step 2: Creating virtual environment in .\venv ===
%PY_CMD% -m venv venv
if errorlevel 1 (
    echo.
    echo [ERROR] Could not create virtual environment.
    pause
    exit /b 1
)

echo.
echo === Step 3: Activating virtual environment ===
call venv\Scripts\activate.bat

echo.
echo === Step 4: Installing dependencies ===
python -m pip install -r requirements.txt
python -m pip install -e .

echo.
echo ============================================================
echo   Setup complete! Virtual environment is ready.
echo ============================================================
echo.
echo Next steps:
echo   1. Double-click "predict.bat" to test MRI prediction on a sample scan.
echo   2. Place your training dataset into "data\raw\Training" and "data\raw\Testing".
echo   3. Double-click "train.bat" to train the model.
echo.
pause
