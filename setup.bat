@echo off
REM ============================================================
REM  Brain Tumor Classifier - One-Click Windows Setup
REM  Double-click this file to set up everything automatically.
REM ============================================================

echo.
echo === Step 1: Creating virtual environment ===
py -3.11 -m venv venv
if errorlevel 1 (
    echo.
    echo ERROR: Could not create venv. Make sure Python 3.11 is installed
    echo and available as "py -3.11" ^(check with: py --list^).
    pause
    exit /b 1
)

echo.
echo === Step 2: Activating virtual environment ===
call venv\Scripts\activate.bat

echo.
echo === Step 3: Installing dependencies (this can take a few minutes) ===
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

echo.
echo === Setup complete! ===
echo.
echo Next steps:
echo   1. Put your dataset in data\raw\Training and data\raw\Testing
echo      (or edit configs\config.yaml to point elsewhere)
echo   2. Run train.bat to train the model
echo.
pause
