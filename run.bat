@echo off
cd /d "%~dp0"
if not exist "venv\Scripts\python.exe" (
  echo [ERROR] venv\Scripts\python.exe not found. Folder: %~dp0
  pause
  exit /b 1
)
echo Starting Face Recognition app... browser will open automatically.
echo Keep this window open while using the app. Close it to stop.
"venv\Scripts\python.exe" face_app.py
echo.
echo [Program exited. If there is an error above, copy it and send it.]
pause
