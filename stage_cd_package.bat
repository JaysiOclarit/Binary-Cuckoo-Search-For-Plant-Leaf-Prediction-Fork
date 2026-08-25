@echo off
title Stage Academic CD/DVD Submission Package
color 0B

echo =========================================================================
echo   PhytoCuckoo - Staging Academic CD/DVD Submission Package
echo =========================================================================
echo.

set OUTPUT_DIR=CD_DISTRIBUTION_PACKAGE

if exist "%OUTPUT_DIR%" (
    echo [INFO] Removing previous staging folder...
    rmdir /s /q "%OUTPUT_DIR%"
)

echo [1/4] Creating Clean CD Directory Structure...
mkdir "%OUTPUT_DIR%"
mkdir "%OUTPUT_DIR%\01_Executable_Application"
mkdir "%OUTPUT_DIR%\01_Executable_Application\models"
mkdir "%OUTPUT_DIR%\01_Executable_Application\extractor"
mkdir "%OUTPUT_DIR%\02_Complete_Source_Code"
mkdir "%OUTPUT_DIR%\02_Complete_Source_Code\backend"
mkdir "%OUTPUT_DIR%\02_Complete_Source_Code\frontend"
mkdir "%OUTPUT_DIR%\02_Complete_Source_Code\scripts"
mkdir "%OUTPUT_DIR%\03_Datasets_and_Benchmarks"

echo.
echo [2/4] Building Frontend and Embedding into Spring Boot Static Resources...
cd frontend
call npm run build
cd ..
if not exist "backend\src\main\resources\static" mkdir "backend\src\main\resources\static"
xcopy /E /I /Y "frontend\dist\*" "backend\src\main\resources\static\"

echo.
echo [3/4] Packaging Unified Executable Spring Boot Fat JAR...
cd backend
call mvn clean package -DskipTests
cd ..

echo.
echo [4/4] Copying Software, Models, and Datasets into CD Package...

:: 1. Executable App + Serialized Models + Feature Extractor
copy /Y "backend\target\FeatureSelection_BlackHole_Optimizer-1.0-SNAPSHOT.jar" "%OUTPUT_DIR%\01_Executable_Application\PhytoCuckoo-Application.jar"
xcopy /E /I /Y "backend\models\*" "%OUTPUT_DIR%\01_Executable_Application\models\"
xcopy /E /I /Y "backend\extractor\*" "%OUTPUT_DIR%\01_Executable_Application\extractor\"

:: 2. Clean Source Code (Excluding heavy node_modules, target, .git, .idea)
robocopy "backend" "%OUTPUT_DIR%\02_Complete_Source_Code\backend" /E /XD "target" ".idea" /XF "*.log" /NFL /NDL /NJH /NJS
robocopy "frontend" "%OUTPUT_DIR%\02_Complete_Source_Code\frontend" /E /XD "node_modules" "dist" ".idea" /NFL /NDL /NJH /NJS
robocopy "scripts" "%OUTPUT_DIR%\02_Complete_Source_Code\scripts" /E /NFL /NDL /NJH /NJS
copy /Y "docker-compose.yml" "%OUTPUT_DIR%\02_Complete_Source_Code\"

:: 3. Datasets & Benchmarks
if exist "backend\Entire Data Folder" (
    robocopy "backend\Entire Data Folder" "%OUTPUT_DIR%\03_Datasets_and_Benchmarks\Entire Data Folder" /E /NFL /NDL /NJH /NJS
)

:: Root CD Launchers
copy /Y "START_APPLICATION.bat" "%OUTPUT_DIR%\"
copy /Y "autorun.inf" "%OUTPUT_DIR%\"

:: Create README.txt in CD root
(
echo =========================================================================
echo   PhytoCuckoo - System Software Disc (CD/DVD)
echo   Genetic Binary Cuckoo Search (GBCS) Plant Leaf Prediction System
echo =========================================================================
echo.
echo DIRECTORY OVERVIEW:
echo  - 01_Executable_Application   : Standalone pre-compiled executable JAR & models.
echo  - 02_Complete_Source_Code     : Full Java Spring Boot + React source code.
echo  - 03_Datasets_and_Benchmarks  : Swedish, Flavia, & Philippine leaf datasets.
echo.
echo HOW TO RUN THE APPLICATION:
echo  1. Ensure Java 17 or higher is installed on your computer.
echo  2. Double-click "START_APPLICATION.bat" in this disc directory.
echo  3. The browser will open automatically to: http://localhost:8080
echo.
echo SYSTEM REQUIREMENTS:
echo  - Windows 10/11, macOS, or Linux
echo  - Java Runtime Environment (JRE/JDK) 17+
echo  - Modern Web Browser (Google Chrome, Microsoft Edge, Safari, Firefox)
echo =========================================================================
) > "%OUTPUT_DIR%\README.txt"

echo.
echo [5/5] Calculating Staged CD Package Size...
powershell -Command "$size = (Get-ChildItem -Path '%OUTPUT_DIR%' -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1MB; Write-Host ('Total Staged CD Package Size: ' + [math]::Round($size, 2) + ' MB'); if ($size -le 700) { Write-Host '[SUCCESS] Fits within standard 700 MB CD-R!' -ForegroundColor Green } else { Write-Host '[NOTICE] Size exceeds 700 MB - please burn onto DVD-R (4.7 GB) or dual-layer disc.' -ForegroundColor Yellow }"

echo.
echo =========================================================================
echo   Staging Complete! All files are organized in: %OUTPUT_DIR%
echo   You can now burn the contents of %OUTPUT_DIR% directly to CD/DVD.
echo =========================================================================
echo.
pause
