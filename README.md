# PhytoCuckoo: Plant Leaf Prediction & GBCS Feature Selection Platform 🍃

> **Thesis Title**: Optimizing Cuckoo Search using Genetic Operators and Correlation-Aware Fitness for Plant Leaf Classification
> 
> **Evaluated Datasets**: Swedish Leaf Dataset (15 Species), Flavia Leaf Dataset (32 Species), Philippine Native & Medicinal Leaf Dataset (40 Species)

---

## 🌟 Executive Summary

**PhytoCuckoo** is an interactive web platform and research pipeline built to present the **Genetic Binary Cuckoo Search (GBCS)** feature selection algorithm. The system combines:

1. **Inception-V3 Deep CNN Feature Extraction**: Processes uploaded leaf photos and extracts 2,048-dimensional feature vectors.
2. **Oracle Tribuo ML Engine**: Java Spring Boot backend executing Factorization Machines (FM) & Bagging Ensemble models trained on feature-selected datasets.
3. **Side-by-Side Benchmark Workbench**: Evaluates Baseline BCS vs. Proposed GBCS in real time.
4. **Interactive Cuckoo Simulator**: Animates Lévy flight dynamics and live fitness convergence curves ($f(x)$ over 30 iterations).
5. **Panel Defense Suite**: Guided step-by-step presentation mode for thesis defense day.

---

## 📁 Organized Project Directory Structure

```text
Binary-Cuckoo-Search-For-Plant-Leaf-Prediction-Fork/
│
├── 📄 .gitignore                             # Git exclusion rules (ignores target/, node_modules/, dist/)
├── 📄 README.md                              # Master project documentation & live launch instructions
├── 🚀 START_APPLICATION.bat                  # One-click Windows application launcher
├── 🚀 start_application.sh                   # One-click macOS / Linux application launcher
├── ⚙️ setup_extractor_env.bat                # Automated Python virtualenv & PyTorch setup (Windows)
├── ⚙️ setup_extractor_env.sh                 # Automated Python virtualenv & PyTorch setup (macOS/Linux)
├── 📦 stage_cd_package.bat                   # Academic CD/DVD submission package stager
├── 🐳 docker-compose.yml                     # Multi-container Docker orchestration (Spring Boot + React)
│
├── 🧠 backend/                               # Java Spring Boot Server & ML Core
│   ├── Dockerfile                            # Multi-stage Dockerfile (Java 17 JRE + PyTorch CPU)
│   ├── mvn.cmd                               # Universal Maven launcher with dynamic IDE discovery
│   ├── extractor/                            # Feature Extractor Module
│   │   ├── extract_features.py               # Inception-V3 Deep CNN Feature Extractor & Auto-Masker
│   │   └── requirements.txt                  # PyTorch, Torchvision, Pillow, OpenCV, NumPy, SciPy
│   ├── models/                               # Serialized Oracle Tribuo Models (.ser)
│   │   ├── Swedish_BCS_Model.ser
│   │   ├── Swedish_GBCS_Model.ser
│   │   ├── Flavia_BCS_Model.ser
│   │   ├── Flavia_GBCS_Model.ser
│   │   ├── Philippine_BCS_Model.ser
│   │   └── Philippine_GBCS_Model.ser
│   ├── pom.xml                               # Maven Dependency Manifest (Spring Boot 3.2, Tribuo 4.3)
│   ├── src/main/java/WrapperCuckooSearchForFS/
│   │   ├── API/
│   │   │   └── PlantPredictionController.java # REST API with universal cross-platform Python bridge
│   │   ├── Discreeting/                      # V2 Transfer Function
│   │   ├── Evaluation/                       # Fitness & Correlation-Aware Evaluators
│   │   ├── Main/                             # Batch Runners & ModelExporter.java
│   │   └── Optimizers/                       # CuckooSearchOptimizer & GeneticCuckooSearchOptimizer
│   └── Entire Data Folder/                   # Raw & Feature-Selected CSV Datasets
│
├── 🎨 frontend/                              # Vite + React 19 + TypeScript Web App
│   ├── Dockerfile                            # Production Nginx containerization
│   ├── nginx.conf                            # Nginx reverse proxy configuration
│   ├── src/
│   │   ├── components/                       # LeafClassifier, SideBySideBenchmark, CuckooSimulator, BotanicalEncyclopedia, ThesisAnalytics, DefenseWizard
│   │   ├── App.tsx                           # Main Controller & REST API Router
│   │   ├── index.css                         # Tailwind CSS v4 & Glassmorphism Design System
│   │   └── types.ts                          # TypeScript Data Interfaces
│   ├── index.html                            # HTML5 Template & SEO Metadata
│   ├── vite.config.ts                        # Vite Configuration & Backend API Proxy
│   └── dist/                                 # Production Web Bundle
│
├── 📊 Results/                               # Consolidated Experimental CSV/TXT Output Reports
│   ├── All_KFold_CrossValidation_Results.csv # Automated K-Fold CSV Output
│   ├── CV_Manual_Results.txt                 # Cross-Validation Text Log
│   ├── Class_Distribution_Analysis.csv       # EDA Class Imbalance Data
│   ├── Compiled Results.xlsx                 # Master Excel Results
│   └── EDA_Analysis_Report.txt               # EDA Text Report
│
├── 🛠️ scripts/                               # Python Preprocessing Utilities
│   └── clean_datasets.py                     # Dataset Label Cleaning Utility
│
└── 📚 Research Paper/                        # Literature & Reference PDFs (Baseline & Proposed)
```

---

## 📊 Empirical Performance Benchmark Summary (Ground Truth Results)

The metrics below are pulled directly from `Results/All_KFold_CrossValidation_Results.csv` and `Results/Compiled Results.xlsx`:

### 1. K-Fold Cross-Validation Metrics

| Dataset | Method | K-Folds | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Swedish Leaf** | **Proposed GBCS** | **5** | **96.30%** | **96.08%** | **96.09%** | **95.86%** |
| Swedish Leaf | Baseline BCS | 5 | 95.85% | 96.12% | 95.81% | 95.57% |
| **Swedish Leaf** | **Proposed GBCS** | **9** | **96.89%** | **96.92%** | **97.10%** | **96.63%** |
| Swedish Leaf | Baseline BCS | 9 | 96.30% | 96.66% | 96.45% | 96.04% |
| **Flavia Leaf** | **Proposed GBCS** | **5** | **97.20%** | **94.49%** | **94.43%** | **94.27%** |
| Flavia Leaf | Baseline BCS | 5 | 97.73% | 95.10% | 94.88% | 94.82% |
| **Flavia Leaf** | **Proposed GBCS** | **7** | **97.90%** | **94.38%** | **94.08%** | **93.97%** |
| Flavia Leaf | Baseline BCS | 7 | 97.81% | 93.97% | 94.28% | 93.87% |
| **Philippine Leaf** | **Proposed GBCS** | **5** | **97.55%** | **97.67%** | **97.44%** | **97.45%** |
| Philippine Leaf | Baseline BCS | 5 | 97.39% | 97.36% | 97.29% | 97.19% |
| **Philippine Leaf** | **Proposed GBCS** | **9** | **97.92%** | **98.01%** | **97.94%** | **97.81%** |
| Philippine Leaf | Baseline BCS | 9 | 97.69% | 97.80% | 97.64% | 97.55% |

### 2. Exploratory Data Analysis (EDA) Characteristics

| Dataset | Total Samples | Species Classes | Class Imbalance Ratio | Initial Features | Avg Correlation ($\bar{\rho}$) | Highly Correlated Pairs ($&#124;r&#124; > 0.85$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Swedish Leaf** | 1,125 | 15 | 1.00 : 1 | 2,048 | 0.1611 | 66 pairs |
| **Flavia Leaf** | 1,907 | 33 | 77.00 : 1 | 2,048 | 0.1489 | 148 pairs |
| **Philippine Leaf** | 4,971 | 40 | 1.71 : 1 | 2,048 | 0.1253 | 21 pairs |

---

## 💻 System Requirements & Prerequisites

Before running the project, verify that the following tools are installed on your system:

| Software | Version | Purpose | Required? |
| :--- | :---: | :--- | :---: |
| **Java JDK** | **17+** (JDK 17, 21, or 27) | Spring Boot backend, Oracle Tribuo ML engine, GBCS optimizer | **Yes** |
| **Node.js & npm** | **18+** (v20, v22, v24) | Vite + React web interface & interactive dashboard | **Yes** |
| **Orange Data Mining** | **3.x** | Exact Inception-V3 image embedder used for thesis datasets | **Recommended** |
| **Python** | **3.10 – 3.12** | Standalone PyTorch extractor (if not using Orange) | Optional |
| **Apache Maven** | **3.8+** | Java build tool (`backend/mvn.cmd` auto-detects extracted copies) | Optional |

> [!IMPORTANT]
> **Windows Environment Setup**:
> 1. **Set `JAVA_HOME`**: Maven requires the `JAVA_HOME` environment variable to point to your JDK root (e.g. `C:\Program Files\Java\jdk-27` or `jdk-17`).
>    * Quick PowerShell fix:
>      ```powershell
>      [Environment]::SetEnvironmentVariable("JAVA_HOME", "C:\Program Files\Java\jdk-27", "User")
>      ```
> 2. **Enable Script Execution in PowerShell**: Windows blocks `.ps1` scripts by default (which affects `npm`). Run this once in PowerShell:
>    ```powershell
>    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
>    ```

---

## 🍊 Orange Data Mining Setup (Thesis Standard Embedder)

All thesis datasets (`Swedish`, `Flavia`, `Philippine`) were engineered using **Orange Data Mining's Image Analytics (Inception-V3) add-on**, producing the 2,048 deep features labeled `Att0` to `Att2047`.

To ensure live uploaded images use the exact same feature extraction pipeline:

1. Download and install **[Orange Data Mining](https://orangedatamining.com/download/)** (default path: `C:\Program Files\Orange`).
2. Open the Orange desktop app.
3. In the top menu, go to **Options** -> **Add-ons...**.
4. Check **Orange3-ImageAnalytics** and click **OK** to install.
   *(Or via terminal: `& "C:\Program Files\Orange\python.exe" -m pip install Orange3-ImageAnalytics`)*

> [!NOTE]
> The backend server automatically auto-discovers Orange at `C:\Program Files\Orange\python.exe` or `%LOCALAPPDATA%\Programs\Orange\python.exe`.
> If Orange is not installed, the system automatically falls back to PyTorch Inception-V3.

---

## 🚀 How to Run the Application

Choose whichever launch method best fits your workflow:

### Option 1: One-Click Windows Launcher (Easiest)

Simply double-click **`START_APPLICATION.bat`** from the project root folder.
* It checks your Java environment.
* If a pre-packaged JAR exists, it runs it; otherwise, it automatically starts the Spring Boot backend server.
* It launches the application and opens your default browser at **`http://localhost:8080`**.

---

### Option 2: Developer Mode (Full Source Setup)

If you are developing or testing both the backend and frontend simultaneously:

#### Step 1: Initialize the Frontend (First time only)
In your terminal, navigate to the `frontend` folder and install packages:
```powershell
cd frontend
npm install
```

#### Step 2: Start the Backend Server (Terminal 1)
```powershell
cd backend
mvn.cmd spring-boot:run
```
*(On macOS/Linux, run `./mvnw spring-boot:run` or `mvn spring-boot:run`)*.
The backend API and Tribuo ML engine will start on **`http://localhost:8080`**.

#### Step 3: Start the Frontend Web UI (Terminal 2)
```powershell
cd frontend
npm run dev
```
Open **`http://localhost:5173`** (or `http://localhost:3000`) to interact with the live defense dashboard.

---

### Option 3: Universal Docker Container (Zero Dependencies)

If you have Docker installed and want to run the application without installing Java, Node, or Python:

```bash
docker compose up --build
```
Once built, open **`http://localhost:3000`** in your browser.

---

### Option 4: Standalone Python Extractor (Without Orange)

If you prefer using a local Python virtual environment instead of Orange Data Mining:
* **Windows**: Run `setup_extractor_env.bat`
* **macOS / Linux**: Run `./setup_extractor_env.sh`

This creates `backend/extractor/venv/` and installs PyTorch CPU and Torchvision automatically.

---

## 📦 Packaging for Academic Submission (CD/DVD / Flash Drive)

To generate a standalone academic submission bundle with an executable JAR:
```cmd
stage_cd_package.bat
```
This utility:
1. Builds the production frontend bundle (`npm run build`) and copies assets to Spring Boot static resources.
2. Packages a self-contained executable JAR (`PhytoCuckoo-Application.jar`).
3. Assembles models, datasets, launchers, and research papers into `CD_DISTRIBUTION_PACKAGE/`.

---

## ❓ Troubleshooting & Frequently Asked Questions (FAQ)

### 1. `npm : File ... npm.ps1 cannot be loaded because running scripts is disabled`
* **Cause**: Windows PowerShell restricts running scripts by default.
* **Fix**: Run this command once in PowerShell:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

### 2. `'tsc' is not recognized as an internal or external command`
* **Cause**: Frontend dependencies were not installed before running `npm run build`.
* **Fix**: Navigate into the frontend directory and install dependencies:
  ```powershell
  cd frontend
  npm install
  npm run build
  ```

### 3. `JAVA_HOME environment variable is not set`
* **Cause**: Maven cannot find your JDK directory.
* **Fix**: Set your `JAVA_HOME` to your JDK root folder (not the `bin` folder):
  ```powershell
  [Environment]::SetEnvironmentVariable("JAVA_HOME", "C:\Program Files\Java\jdk-27", "User")
  ```
  *(Replace `jdk-27` with your installed JDK version, e.g. `jdk-17` or `jdk-21`).*

### 4. `'mvn' is not recognized as an internal or external command`
* **Cause**: Apache Maven is not added to your system PATH.
* **Fix**: Use the bundled wrapper `.\mvn.cmd` inside the `backend/` directory:
  ```powershell
  cd backend
  .\mvn.cmd spring-boot:run
  ```
  The wrapper automatically searches for extracted Maven installations in `Program Files` and `AppData`.

### 5. `Unrecognized token 'Downloading': was expecting ...` on First Image Upload
* **Cause**: PyTorch or Inception-V3 is downloading its pre-trained weights (`.pth`) for the first time.
* **Fix**: This is normal and only happens once. The backend's fallback parser automatically catches the output. Subsequent predictions will be instantaneous.

---

## 🛠️ Technology Stack

* **Backend**: Java 17 / 27, Spring Boot 3.2, Oracle Tribuo 4.3.1, Apache Commons Math 3.6, oj! Algorithms
* **Feature Extraction**: Orange Data Mining 3 (Image Analytics) / Inception-V3 CNN, PyTorch CPU, Torchvision, Pillow
* **Frontend**: React 19, Vite 8, TypeScript, Tailwind CSS v4, Recharts, Lucide React, Canvas Confetti
* **Containerization**: Docker, Docker Compose, Nginx Alpine, Eclipse Temurin 17 JRE
* **Supported OS**: Windows 10/11, macOS (Apple Silicon / Intel), Ubuntu / Debian Linux

