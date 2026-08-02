@echo off
REM Double-click to train the model using the venv set up by setup.bat
call venv\Scripts\activate.bat
python scripts\train.py --config configs\config.yaml
pause
