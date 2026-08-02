@echo off
REM Double-click to evaluate the best trained model
call venv\Scripts\activate.bat
python scripts\evaluate.py --config configs\config.yaml --model-path artifacts\checkpoints\best_model.keras
pause
