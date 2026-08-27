@echo off
REM Brain Tumor Classifier - Model Training
cd /d "%~dp0"

if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else if exist ..\venv\Scripts\activate.bat (
    call ..\venv\Scripts\activate.bat
) else (
    echo [ERROR] Virtual environment not found. Please run setup.bat first.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Brain Tumor Classifier - Train Model
echo ============================================================
echo.
python scripts\train.py --config configs\config.yaml
echo.
pause
