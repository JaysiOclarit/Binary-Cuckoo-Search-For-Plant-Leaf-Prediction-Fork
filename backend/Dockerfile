# Multi-stage Dockerfile for PhytoCuckoo Spring Boot Backend
FROM maven:3.9.6-eclipse-temurin-17 AS builder
WORKDIR /app

# Copy pom.xml and source code
COPY pom.xml .
COPY src ./src

# Package the Spring Boot application (skipping tests for fast build)
RUN mvn clean package -DskipTests

# Runtime stage with Java 17 and Python for Inception-V3 PyTorch feature extraction
FROM eclipse-temurin:17-jre-jammy
WORKDIR /app

# Install Python 3, pip, and lightweight dependencies for OpenCV / PyTorch CPU
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU and extractor requirements
COPY extractor/requirements.txt ./extractor/
RUN pip3 install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install --no-cache-dir -r extractor/requirements.txt

# Copy built JAR from builder
COPY --from=builder /app/target/FeatureSelection_BlackHole_Optimizer-1.0-SNAPSHOT.jar app.jar

# Copy serialized Tribuo ML models and extractor script
COPY models/ ./models/
COPY extractor/ ./extractor/

# Expose Spring Boot port
EXPOSE 8080

# Run Spring Boot application
ENTRYPOINT ["java", "-jar", "app.jar"]
