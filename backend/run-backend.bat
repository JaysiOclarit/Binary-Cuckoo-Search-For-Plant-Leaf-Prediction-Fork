@echo off
cd /d "%~dp0"
call mvn.cmd spring-boot:run
echo [INFO] Starting Spring Boot using IntelliJ Bundled Maven...
call "C:\Program Files\JetBrains\IntelliJ IDEA 2026.2.0.1\plugins\maven-plugin\lib\maven3\bin\mvn.cmd" spring-boot:run
