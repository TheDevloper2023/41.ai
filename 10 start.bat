@echo off
title TTS Server Launcher

echo ============================================
echo  Starting TTS API and Celery Worker
echo ============================================

REM --- Activate virtual environment ---
call venv\Scripts\activate

REM --- Start Celery worker ---
echo [1/2] Starting Celery worker...
celery -A worker worker --loglevel=info --pool=threads

REM --- Start FastAPI server ---
echo [2/2] Starting FastAPI (port 5000)...
start "FastAPI API" cmd /k uvicorn api:app --reload --port 5000

REM --- Open in browser ---
timeout /t 3 >nul
start "" http://127.0.0.1:5000/

echo.
echo ============================================
echo  All services started successfully!
echo ============================================
pause
