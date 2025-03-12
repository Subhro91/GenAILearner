@echo off
echo ===================================================
echo PDF Learning Assistant - Run Application
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

REM Check if models exist
if not exist models\summarizer (
    echo WARNING: AI models not found in the models directory.
    echo You may need to run setup.bat or download_models.py first.
    echo The application will still run, but some features may not work correctly.
    echo.
    pause
)

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