@echo off
echo Starting MuseTalk Realtime API Server...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if OpenAI API key is set
if "%OPENAI_API_KEY%"=="" (
    echo Error: OPENAI_API_KEY environment variable is not set
    echo Please set your OpenAI API key:
    echo set OPENAI_API_KEY=your-api-key-here
    pause
    exit /b 1
)

REM Install requirements if needed
echo Installing/updating requirements...
pip install -r requirements_realtime.txt

REM Start the server
echo.
echo Starting WebSocket server...
echo Open realtime_frontend.html in your browser to test
echo Press Ctrl+C to stop the server
echo.

python start_realtime_server.py

pause