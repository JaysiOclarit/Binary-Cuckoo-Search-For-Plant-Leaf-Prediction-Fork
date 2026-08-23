@echo off
if exist "C:\Program Files\JetBrains\IntelliJ IDEA Community Edition 2025.1\plugins\maven\lib\maven3\bin\mvn.cmd" (
    "C:\Program Files\JetBrains\IntelliJ IDEA Community Edition 2025.1\plugins\maven\lib\maven3\bin\mvn.cmd" %*
) else (
    mvn %*
)
