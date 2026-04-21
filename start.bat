@echo off
title NodeTorch
chcp 65001 >nul 2>&1

echo.
echo  NodeTorch
echo.

:: ─────────────────────────────────────────────
:: Check Python
:: ─────────────────────────────────────────────
set PYTHON=
for %%P in (python3 python py) do (
    where %%P >nul 2>&1 && (
        for /f "tokens=*" %%V in ('%%P -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2^>nul') do (
            for /f "tokens=1,2 delims=." %%A in ("%%V") do (
                if %%A==3 if %%B geq 10 (
                    set PYTHON=%%P
                    goto :python_found
                )
            )
        )
    )
)
echo [ERROR] Python 3.10+ not found
echo.
echo   Install Python from https://python.org/downloads
echo   Make sure to check "Add Python to PATH" during installation
echo.
pause
exit /b 1

:python_found
for /f "tokens=*" %%V in ('%PYTHON% --version 2^>^&1') do echo  [OK] %%V

:: ─────────────────────────────────────────────
:: Check Node.js
:: ─────────────────────────────────────────────
where node >nul 2>&1 || (
    echo [ERROR] Node.js not found
    echo.
    echo   Install from https://nodejs.org
    echo.
    pause
    exit /b 1
)
for /f "tokens=*" %%V in ('node --version 2^>^&1') do echo  [OK] Node.js %%V

where npm >nul 2>&1 || (
    echo [ERROR] npm not found
    pause
    exit /b 1
)

:: ─────────────────────────────────────────────
:: Setup Python venv (first run)
:: ─────────────────────────────────────────────
if not exist ".venv" (
    echo.
    echo  First run — setting up Python environment...

    %PYTHON% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )

    .venv\Scripts\pip install --upgrade pip -q

    :: Detect NVIDIA GPU
    echo   Detecting GPU...
    set HAS_CUDA=0
    where nvidia-smi >nul 2>&1 && (
        nvidia-smi >nul 2>&1 && set HAS_CUDA=1
    )

    if "%HAS_CUDA%"=="1" (
        echo   [OK] NVIDIA GPU detected — installing PyTorch with CUDA
        .venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 -q
    ) else (
        echo   No GPU detected — installing CPU-only PyTorch
        echo   (Training will be slower but everything works^)
        .venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q
    )

    if errorlevel 1 (
        echo [ERROR] Failed to install PyTorch
        echo   Try: .venv\Scripts\pip install torch torchvision
        pause
        exit /b 1
    )

    echo   Installing other dependencies...
    .venv\Scripts\pip install -r requirements.txt -q
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )

    echo   [OK] Python dependencies installed
)

:: ─────────────────────────────────────────────
:: Setup frontend (always runs to pick up new deps)
:: ─────────────────────────────────────────────
echo.
echo  Installing frontend dependencies...
call npm install --silent 2>nul
if errorlevel 1 (
    echo [ERROR] npm install failed
    pause
    exit /b 1
)
echo   [OK] Frontend dependencies installed

:: ─────────────────────────────────────────────
:: Start
:: ─────────────────────────────────────────────
echo.
echo  ========================================
echo    Frontend  -  http://localhost:5173
echo    Backend   -  http://localhost:8000
echo  ========================================
echo.
echo  Press Ctrl+C to stop.
echo.

:: Start backend
cd backend
start /b "" ..\".venv\Scripts\python" main.py
cd ..

:: Wait for backend
timeout /t 3 /nobreak >nul

:: Start frontend and open browser
start http://localhost:5173
call npm run dev

pause
