@echo off
REM ============================================================
REM  Rental Price Analyzer — One-Click Launch Script
REM  Opens the app at http://127.0.0.1:5000
REM ============================================================

echo.
echo ===================================================
echo   RENTAL PRICE ANALYZER - STARTING
echo ===================================================
echo.
echo   Web App:  http://127.0.0.1:5000
echo   Press Ctrl+C to stop the server.
echo.
echo ===================================================
echo.

REM Use venv Python to run the Flask app
venv\Scripts\python app.py
