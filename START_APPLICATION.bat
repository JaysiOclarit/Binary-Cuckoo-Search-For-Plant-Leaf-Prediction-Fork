@echo off
title PhytoCuckoo - Plant Leaf Prediction System Launcher
color 0A

echo =========================================================================
echo   PhytoCuckoo - Genetic Binary Cuckoo Search (GBCS) Thesis System
echo =========================================================================
echo.
echo Checking Java Environment...

where java >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Java is not detected in your system PATH.
    echo Please install Java 17 (JRE/JDK) or higher to run the executable JAR.
    echo Download: https://adoptium.net/temurin/releases/
    echo.
    pause
    exit /b 1
)

echo [OK] Java detected.
echo.
echo Starting PhytoCuckoo Backend Service ^& Interactive Web Interface...
echo.

:: Check if standalone executable jar exists in 01_Executable_Application or backend/target
set JAR_PATH=01_Executable_Application\PhytoCuckoo-Application.jar
if not exist "%JAR_PATH%" (
    set JAR_PATH=backend\target\FeatureSelection_BlackHole_Optimizer-1.0-SNAPSHOT.jar
)

if exist "%JAR_PATH%" (
    echo Launching: %JAR_PATH%
    start "" cmd /c "java -jar %JAR_PATH%"
    
    echo Waiting for application server to initialize...
    timeout /t 5 /nobreak >nul
    
    echo Opening PhytoCuckoo Dashboard in default browser...
    start http://localhost:8080
) else (
    echo [WARNING] Executable JAR not found.
    echo Please verify the contents of 01_Executable_Application or 02_Complete_Source_Code.
)

echo.
echo =========================================================================
echo   Application is running. Keep this window open while using the system.
echo   URL: http://localhost:8080 or http://localhost:3000
echo =========================================================================
echo.
pause
