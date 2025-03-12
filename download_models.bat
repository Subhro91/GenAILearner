@echo off
echo ===================================================
echo PDF Learning Assistant - Download AI Models
echo ===================================================
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment.
    echo Please run setup.bat first to set up the application.
    pause
    exit /b
)
echo Virtual environment activated.
echo.

echo ===================================================
echo Downloading AI models...
echo This will take 5-10 minutes on first run.
echo ===================================================
echo.

python download_models.py
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to download AI models.
    echo Please check your internet connection and try again.
    echo.
    pause
    exit /b
)

echo.
echo ===================================================
echo AI models downloaded successfully!
echo You can now run the application using run_app.bat
echo ===================================================
echo.

pause 