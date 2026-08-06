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
│
├── 🧠 backend/                               # Java Spring Boot Server & ML Core
│   ├── extractor/                            # Feature Extractor Module
│   │   ├── extract_features.py               # Inception-V3 Deep CNN Feature Extractor
│   │   └── requirements.txt                  # Python Extractor Dependencies
│   ├── models/                               # Serialized Oracle Tribuo Models (.ser)
│   │   ├── Swedish_BCS_Model.ser
│   │   ├── Swedish_GBCS_Model.ser
│   │   ├── Flavia_BCS_Model.ser
│   │   ├── Flavia_GBCS_Model.ser
│   │   ├── Philippine_BCS_Model.ser
│   │   └── Philippine_GBCS_Model.ser
│   ├── pom.xml                               # Maven Dependency Manifest (Spring Boot, Tribuo 4.3)
│   ├── src/main/java/WrapperCuckooSearchForFS/
│   │   ├── API/
│   │   │   └── PlantPredictionController.java # REST Endpoints (predict, compare, analytics, catalog)
│   │   ├── Discreeting/                      # V2 Transfer Function
│   │   ├── Evaluation/                       # Fitness & Correlation-Aware Evaluators
│   │   ├── Main/                             # Batch Runners & ModelExporter.java
│   │   └── Optimizers/                       # CuckooSearchOptimizer & GeneticCuckooSearchOptimizer
│   └── Entire Data Folder/                   # Raw & Feature-Selected CSV Datasets
│
├── 🎨 frontend/                              # Vite + React + TypeScript Web App
│   ├── src/
│   │   ├── components/                       # LeafClassifier, SideBySideBenchmark, CuckooSimulator, BotanicalEncyclopedia, ThesisAnalytics, DefenseWizard
│   │   ├── App.tsx                           # Main Controller & REST API Router
│   │   ├── index.css                         # Tailwind CSS & Glassmorphism Design System
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

| Dataset | Total Samples | Species Classes | Class Imbalance Ratio | Initial Features | Avg Correlation ($\bar{\rho}$) | Highly Correlated Pairs ($|r| > 0.85$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Swedish Leaf** | 1,125 | 15 | 1.00 : 1 | 2,048 | 0.1611 | 66 pairs |
| **Flavia Leaf** | 1,907 | 33 | 77.00 : 1 | 2,048 | 0.1489 | 148 pairs |
| **Philippine Leaf** | 4,971 | 40 | 1.71 : 1 | 2,048 | 0.1253 | 21 pairs |

---

## 🚀 Quick Start Guide

### 1. Launch Java Spring Boot Backend (Port 8080)
Ensure you have [Maven](https://maven.apache.org/) installed and added to your system PATH, then run:
```bash
cd backend
mvn compile exec:java -Dexec.mainClass="WrapperCuckooSearchForFS.org.SpringBootApp"
```

### 2. Launch React Frontend Web Application (Port 3000)
```bash
cd frontend
npm install
npm run dev
```
Open **`http://localhost:3000`** in your browser to view the application.

---

## 🛠️ Technology Stack

*   **Backend**: Java 24, Spring Boot 3.2, Oracle Tribuo 4.3.1, Apache Commons Math 3.6
*   **Feature Extractor**: Python 3.14, PyTorch / Torchvision (Inception-V2 CNN), OpenCV, NumPy, Pillow
*   **Frontend**: React 18, Vite 8, TypeScript, Tailwind CSS v4, Recharts, Lucide React, Canvas Confetti
