@echo off
setlocal

:: Prevent self-recursion
set "THIS_SCRIPT=%~f0"

:: 1. Check if MAVEN_HOME or M2_HOME is set
if defined MAVEN_HOME if exist "%MAVEN_HOME%\bin\mvn.cmd" (
    "%MAVEN_HOME%\bin\mvn.cmd" %*
    exit /b %ERRORLEVEL%
)
if defined M2_HOME if exist "%M2_HOME%\bin\mvn.cmd" (
    "%M2_HOME%\bin\mvn.cmd" %*
    exit /b %ERRORLEVEL%
)

:: 2. Find any external mvn on PATH that is NOT this script
for /f "delims=" %%I in ('where mvn.cmd 2^>nul') do (
    if /i not "%%~fI"=="%THIS_SCRIPT%" (
        "%%~fI" %*
        exit /b %ERRORLEVEL%
    )
)
for /f "delims=" %%I in ('where mvn.exe 2^>nul') do (
    if /i not "%%~fI"=="%THIS_SCRIPT%" (
        "%%~fI" %*
        exit /b %ERRORLEVEL%
    )
)

:: 3. Check for IntelliJ IDEA bundled Maven across versions
for /f "delims=" %%I in ('dir /b /ad "%ProgramFiles%\JetBrains\IntelliJ*" 2^>nul') do (
    if exist "%ProgramFiles%\JetBrains\%%I\plugins\maven\lib\maven3\bin\mvn.cmd" (
        "%ProgramFiles%\JetBrains\%%I\plugins\maven\lib\maven3\bin\mvn.cmd" %*
        exit /b %ERRORLEVEL%
    )
)
for /f "delims=" %%I in ('dir /b /ad "%LOCALAPPDATA%\Programs\IntelliJ*" 2^>nul') do (
    if exist "%LOCALAPPDATA%\Programs\%%I\plugins\maven\lib\maven3\bin\mvn.cmd" (
        "%LOCALAPPDATA%\Programs\%%I\plugins\maven\lib\maven3\bin\mvn.cmd" %*
        exit /b %ERRORLEVEL%
    )
)

echo [ERROR] Apache Maven was not found on your system.
echo Please install Maven or ensure it is added to your PATH.
exit /b 1

