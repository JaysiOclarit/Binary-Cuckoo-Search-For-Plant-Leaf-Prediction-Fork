@echo off
title PhytoCuckoo - Extractor Environment Setup
color 0B

echo =========================================================================
echo   PhytoCuckoo - Setting up Python Virtual Environment for Feature Extractor
echo =========================================================================
echo.

:: Detect Python executable
set PYTHON_BIN=
where python >nul 2>&1 && set PYTHON_BIN=python
if not defined PYTHON_BIN (
    where py >nul 2>&1 && set PYTHON_BIN=py
)

if not defined PYTHON_BIN (
    echo [ERROR] No Python installation detected on your system.
    echo Please install Python 3.10+ from https://www.python.org/
    pause
    exit /b 1
)

echo [OK] Using Python: %PYTHON_BIN%
echo.
echo Creating virtual environment in backend\extractor\venv ...
%PYTHON_BIN% -m venv backend\extractor\venv

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

echo.
echo Installing PyTorch CPU and Extractor dependencies...
call backend\extractor\venv\Scripts\pip.exe install --upgrade pip
call backend\extractor\venv\Scripts\pip.exe install torch torchvision --index-url https://download.pytorch.org/whl/cpu
call backend\extractor\venv\Scripts\pip.exe install -r backend\extractor\requirements.txt

echo.
echo =========================================================================
echo   Feature Extractor environment is ready!
echo   The backend will now automatically discover and use this environment.
echo =========================================================================
pause
