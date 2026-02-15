@echo off
REM ========================================================
REM   Run MRI Brain Tumor Classification App
REM ========================================================

echo.
echo [1/3] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in your PATH.
    echo Please install Python 3.9+ and try again.
    pause
    exit /b
)

echo.
echo [2/3] Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error: Failed to install dependencies.
    pause
    exit /b
)

echo.
echo [3/3] Starting the application...
echo.
echo The app will open in your browser shortly...
echo Please look for "Serving Flask app" below.
echo.

REM Start the browser after a short delay (in a separate process)
start "" "http://localhost:5000"

python app.py

pause
