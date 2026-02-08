@echo off
echo ========================================
echo Empathy System - Quick Setup Script
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.9 or 3.10
    pause
    exit /b 1
)

echo Step 1: Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo Virtual environment created.
) else (
    echo Virtual environment already exists.
)

echo.
echo Step 2: Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Step 3: Upgrading pip...
python -m pip install --upgrade pip --quiet

echo.
echo Step 4: Installing dependencies...
echo This may take 5-10 minutes...
pip install -r requirements.txt --quiet

echo.
echo Step 5: Installing test dependencies...
pip install pytest pytest-asyncio --quiet

echo.
echo Step 6: Creating directories...
if not exist "data\profiles" mkdir data\profiles
if not exist "logs" mkdir logs
if not exist "models\llm" mkdir models\llm

echo.
echo Step 7: Running quick test...
python main.py

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo   1. Run tests: pytest tests/ -v
echo   2. Try demos: python scripts/demo_vision.py
echo   3. See SETUP.md for full instructions
echo.
pause
