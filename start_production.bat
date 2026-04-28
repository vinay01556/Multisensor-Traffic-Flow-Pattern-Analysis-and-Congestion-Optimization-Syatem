@echo off
echo ========================================================
echo   IntelliTraffic Pro - Production Server Startup
echo ========================================================

:: Ensure we are in the script's directory
cd /d "%~dp0"

echo [1/3] Checking dependencies...
python -m pip install -r requirements.txt | findstr /V "already satisfied"

echo [2/3] Setting Production Environment Variables...
set PORT=5000
set HOST=0.0.0.0

echo [3/3] Launching Uvicorn ASGI Server...
cd backend
python -m uvicorn server:app --host %HOST% --port %PORT%

pause
