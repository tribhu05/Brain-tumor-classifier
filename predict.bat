@echo off
REM Brain Tumor Classifier - Prediction Tool
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
echo   Brain Tumor Classifier - Single Image MRI Inference
echo ============================================================
echo.
set /p IMAGE_PATH="Enter path to MRI image (press Enter to test sample scan): "
if "%IMAGE_PATH%"=="" (
    set IMAGE_PATH=data\sample\sample_mri.jpg
)

if exist artifacts\checkpoints\best_model.keras (
    set MODEL_PATH=artifacts\checkpoints\best_model.keras
) else (
    set MODEL_PATH=assets\best_model.keras
)

echo.
echo Running prediction for image: %IMAGE_PATH%
echo Using model: %MODEL_PATH%
echo.
python scripts\predict.py --config configs\config.yaml --model-path "%MODEL_PATH%" --image "%IMAGE_PATH%"
echo.
pause
