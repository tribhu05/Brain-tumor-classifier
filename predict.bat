@echo off
REM Double-click, then paste/type the path to an MRI image when prompted
call venv\Scripts\activate.bat
set /p IMAGE_PATH="Enter path to MRI image: "
python scripts\predict.py --config configs\config.yaml --model-path artifacts\checkpoints\best_model.keras --image "%IMAGE_PATH%"
pause
