@echo off
REM Brain Tumor Classifier - Model Evaluation
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

if exist artifacts\checkpoints\best_model.keras (
    set MODEL_PATH=artifacts\checkpoints\best_model.keras
) else (
    set MODEL_PATH=assets\best_model.keras
)

echo.
echo ============================================================
echo   Brain Tumor Classifier - Evaluation on Held-Out Test Set
echo ============================================================
echo Using model: %MODEL_PATH%
echo.
python scripts\evaluate.py --config configs\config.yaml --model-path "%MODEL_PATH%"
echo.
pause
