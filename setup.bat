@echo off
echo ===================================================
echo PDF Learning Assistant - Setup and Run Script
echo ===================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH!
    echo Please install Python 3.9 or higher from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b
)

echo Python is installed. Checking version...
for /f "tokens=2" %%I in ('python --version 2^>^&1') do set PYTHON_VERSION=%%I
echo Detected Python version: %PYTHON_VERSION%
echo.

REM Check if virtual environment exists and create it if it doesn't
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment.
        echo Please make sure you have the venv module installed.
        echo Try running: pip install virtualenv
        echo.
        pause
        exit /b
    )
    echo Virtual environment created successfully.
) else (
    echo Virtual environment already exists.
)
echo.

REM Check if activation script exists
if not exist venv\Scripts\activate.bat (
    echo ERROR: Virtual environment activation script not found.
    echo Trying to recreate the virtual environment...
    rmdir /s /q venv
    python -m venv venv
    if %errorlevel% neq 0 (
        echo ERROR: Failed to recreate virtual environment.
        pause
        exit /b
    )
    echo Virtual environment recreated successfully.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment.
    echo This could be due to security settings or script execution policies.
    echo Try running this script as administrator.
    pause
    exit /b
)
echo Virtual environment activated.
echo.

REM Install dependencies
echo Installing required packages...
echo This may take some time (5-10 minutes) as it will download necessary packages.
echo.
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b
)
echo.
echo All packages installed successfully.
echo.

REM Create directories if they don't exist
if not exist uploaded_pdfs (
    mkdir uploaded_pdfs
    echo Created uploaded_pdfs directory.
)

REM Download AI models
echo.
echo ===================================================
echo Downloading AI models...
echo This will take 5-10 minutes on first run.
echo ===================================================
echo.
python download_models.py
if %errorlevel% neq 0 (
    echo.
    echo WARNING: Failed to download AI models.
    echo The application will still run, but some features may not work correctly.
    echo You can try running 'python download_models.py' manually later.
    echo.
    pause
)

echo.
echo ===================================================
echo Setup completed successfully!
echo.
echo Starting the application...
echo.
echo When the application is ready, open your browser and go to:
echo http://localhost:8080
echo.
echo To stop the application, press CTRL+C in this window.
echo ===================================================
echo.

REM Run the application
uvicorn main:app --host 0.0.0.0 --port 8080 --reload

pause 