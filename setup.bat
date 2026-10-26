@echo off
REM ============================================================
REM  Rental Price Analyzer — Quick Setup Script
REM  Run this ONCE to install all dependencies in the venv.
REM ============================================================

echo.
echo ===================================================
echo   RENTAL PRICE ANALYZER - SETUP
echo ===================================================
echo.

REM Check if venv exists, create it if not
IF NOT EXIST "venv\Scripts\python.exe" (
    echo [1/3] Creating virtual environment...
    python -m venv venv
    echo       Done.
) ELSE (
    echo [1/3] Virtual environment already exists. Skipping.
)

echo.
echo [2/3] Installing Python dependencies into venv...
venv\Scripts\pip install -r requirements.txt --quiet
echo       Done.

echo.
echo [3/3] Verifying installations...
venv\Scripts\python -c "import flask, pandas, pyarrow, sklearn; print('  All core libraries OK!')"

echo.
echo ===================================================
echo   SETUP COMPLETE! Run start.bat to launch the app.
echo ===================================================
echo.
pause
