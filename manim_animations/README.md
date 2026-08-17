# Manim "Math Engine" Animation Scenes for Baseline Paper

This directory contains standalone, production-ready [Manim](https://www.manim.community/) (Community Edition) animation scenes specifically designed to explain the baseline paper for your thesis:
**"A Binary Cuckoo Search and its Application for Feature Selection" (Pereira et al., 2014 / Yang & Deb, 2010)** applied to **Plant Leaf Feature Selection & Classification**.

---

## 🎬 Overview of Scenes

| File | Scene Class | Description |
| :--- | :--- | :--- |
| [`scene1_latex_morphing.py`](file:///c:/Users/janch/Thesis%20Project%20with%20Antigravity%20IDE/Binary-Cuckoo-Search-For-Plant-Leaf-Prediction-Fork/manim_animations/scene1_latex_morphing.py) | `LatexFormulaScene` | **LaTeX Formula Morphing**: Binary mask $\vec{X} \in \{0, 1\}^D$, multi-objective fitness morphing $f(\vec{X}) = \overline{\text{Acc}} + 0.001 \cdot (1 - \frac{\|S\|}{D})$, Lévy step equation, and nest abandonment probability $p_a$. |
| [`scene2_transfer_function.py`](file:///c:/Users/janch/Thesis%20Project%20with%20Antigravity%20IDE/Binary-Cuckoo-Search-For-Plant-Leaf-Prediction-Fork/manim_animations/scene2_transfer_function.py) | `TransferFunctionScene` | **V-Shaped Transfer Function**: Plot of $V_2(x) = \|\tan(x)\|$, threshold boundary at $\tau = 0.5$, dynamic probe tracing continuous velocity $x \to V_2(x) \to$ discrete bit $\{0, 1\}$. |
| [`scene3_levy_flight_walk.py`](file:///c:/Users/janch/Thesis%20Project%20with%20Antigravity%20IDE/Binary-Cuckoo-Search-For-Plant-Leaf-Prediction-Fork/manim_animations/scene3_levy_flight_walk.py) | `LevyFlightScene` | **2D Lévy Flight Simulation**: Exact Mantegna's algorithm random walk ($\lambda = 1.5$) simulated on a 2D coordinate search grid side-by-side with standard Brownian motion, showcasing how heavy-tailed leaps escape local optima. |
| [`master_baseline_explainer.py`](file:///c:/Users/janch/Thesis%20Project%20with%20Antigravity%20IDE/Binary-Cuckoo-Search-For-Plant-Leaf-Prediction-Fork/manim_animations/master_baseline_explainer.py) | `MasterBaselineExplainerScene` | **Complete Full-Length Master Video**: Seamlessly stitches Title $\to$ Math Formulation $\to$ Lévy Simulation $\to$ Transfer Function $\to$ Thesis Results into a unified presentation. |

---

## 🚀 How to Render the Animations

You can render any scene directly from your terminal in the `manim_animations` folder.

### 1. Fast Low-Resolution Preview (480p, 15 fps) — *Recommended for fast checking*
```bash
# Scene 1: LaTeX Formulas
manim -pql scene1_latex_morphing.py LatexFormulaScene

# Scene 2: V-Shaped Transfer Function V2
manim -pql scene2_transfer_function.py TransferFunctionScene

# Scene 3: Lévy Flight Simulation
manim -pql scene3_levy_flight_walk.py LevyFlightScene

# Master Full Video
manim -pql master_baseline_explainer.py MasterBaselineExplainerScene
```

### 2. High Definition (1080p, 60 fps) — *For your final presentation/video edit*
```bash
manim -pqh master_baseline_explainer.py MasterBaselineExplainerScene
```

### 3. Export as Transparent PNG Sequence or GIF
```bash
# Render animated GIF
manim -pqm -i scene2_transfer_function.py TransferFunctionScene
```

---

## 🎙️ Suggested Voiceover / Script Guide

### For Scene 1 (LaTeX Morphing):
> *"In the baseline Binary Cuckoo Search algorithm, each candidate solution $\vec{X}$ is represented as a binary bit vector of length $D$, corresponding to the total extracted leaf features. Our optimization objective is multi-objective: we seek to maximize classification accuracy using 10-Fold cross-validation while penalizing redundant features with a small regularization weight of $0.001$. The search explores the space via continuous Lévy flight steps and nest abandonment."*

### For Scene 2 (V2 Transfer Function):
> *"Because Lévy flight operations produce continuous coordinates, we apply the V-shaped transfer function $V_2(x) = |\tan(x)|$ to discretize the state. If the step velocity exceeds the threshold of $0.5$, the feature bit is activated ($1$). This symmetric V-shape prevents saturation and preserves strong exploratory power."*

### For Scene 3 (Lévy Flight Walk):
> *"Unlike standard Brownian motion, which performs Gaussian diffusion and easily gets trapped in local optima, Lévy flights follow a heavy-tailed power-law distribution generated via Mantegna's algorithm. This yields dense local exploration punctuated by sudden, long-range exploratory jumps that allow cuckoos to discover globally optimal feature subsets."*
