#!/usr/bin/env bash

# PhytoCuckoo - Plant Leaf Prediction System Launcher (macOS & Linux)
echo "========================================================================="
echo "  PhytoCuckoo - Genetic Binary Cuckoo Search (GBCS) Thesis System"
echo "========================================================================="
echo ""
echo "Checking Java Environment..."

if ! command -v java &> /dev/null; then
    echo "[ERROR] Java is not detected in your system PATH."
    echo "Please install Java 17 (JRE/JDK) or higher to run the executable JAR."
    echo "Download: https://adoptium.net/temurin/releases/"
    echo ""
    exit 1
fi

echo "[OK] Java detected: $(java -version 2>&1 | head -n 1)"
echo ""
echo "Starting PhytoCuckoo Backend Service & Interactive Web Interface..."
echo ""

JAR_PATH="01_Executable_Application/PhytoCuckoo-Application.jar"
if [ ! -f "$JAR_PATH" ]; then
    JAR_PATH="backend/target/FeatureSelection_BlackHole_Optimizer-1.0-SNAPSHOT.jar"
fi

if [ -f "$JAR_PATH" ]; then
    echo "Launching: $JAR_PATH"
    java -jar "$JAR_PATH" &
    APP_PID=$!
    
    echo "Waiting for application server to initialize..."
    sleep 5
    
    echo "Opening PhytoCuckoo Dashboard in default browser..."
    if command -v xdg-open &> /dev/null; then
        xdg-open "http://localhost:8080"
    elif command -v open &> /dev/null; then
        open "http://localhost:8080"
    else
        echo "Please open your browser and navigate to: http://localhost:8080"
    fi
    
    echo ""
    echo "========================================================================="
    echo "  Application is running (PID: $APP_PID). Press Ctrl+C to stop."
    echo "  URL: http://localhost:8080 or http://localhost:3000"
    echo "========================================================================="
    wait $APP_PID
else
    echo "[WARNING] Executable JAR not found at $JAR_PATH."
    echo "Please build the project first using Maven or check your release package."
    exit 1
fi
