# BCI2a Motor Imagery Classification

A comprehensive benchmark of classical and deep-learning pipelines for 4-class EEG motor imagery classification on the BCI Competition IV Dataset 2a.

---

## Table of Contents

1. [Dataset](#dataset)
2. [Project Structure](#project-structure)
3. [Pipeline Overview](#pipeline-overview)
4. [Methods](#methods)
   - [CSP + LDA](#1-csp--lda)
   - [CSP + SVM (RBF)](#2-csp--svm-rbf)
   - [Voting Ensemble](#3-voting-ensemble-lda--svm--rf)
   - [Riemannian MDM](#4-riemannian-mdm)
   - [Riemannian Tangent Space + LR](#5-riemannian-tangent-space--lr)
   - [FBCSP + LR](#6-filter-bank-csp--lr-fbcsp)
   - [XGBoost on CSP Features](#7-xgboost-on-csp-features)
   - [EEGNet (Deep Learning)](#8-eegnet-deep-learning)
5. [Evaluation Protocol](#evaluation-protocol)
6. [Results Summary](#results-summary)
7. [Visualizations](#visualizations)
8. [Setup & Requirements](#setup--requirements)
9. [How to Run](#how-to-run)
10. [Key Findings & Conclusion](#key-findings--conclusion)

---

## Dataset

**BCI Competition IV Dataset 2a** — 9 subjects performing four motor imagery (MI) tasks:

| Class   | Imagined Movement |
|---------|-------------------|
| `left`  | Left hand         |
| `right` | Right hand        |
| `foot`  | Feet              |
| `tongue`| Tongue            |

- **Sampling rate**: 250 Hz  
- **Channels**: 22 EEG channels (standard 10-20 system)  
- **Trials per subject**: 288 (72 per class)  
- **Epoch window**: −0.1 to +0.7 s relative to cue onset (201 time samples)  
- **Location**: `bci2a/patients/patients/BCICIV_2a_{1..9}.csv`

> **Window limitation**: The provided epochs cover only 0.8 s post-cue. Motor imagery ERD/ERS typically peaks at 0.5–3 s, so realistic per-subject accuracy ceilings are 40–65 % for classical methods (vs. 60–80 % in literature using 2–4 s windows).

---

## Project Structure

```
bci2a_classification/
├── main.ipynb               # Full analysis notebook (sections 1–9)
├── eegnet_generalized.pt    # Saved EEGNet model (all 9 patients)
├── bci2a/
│   └── patients/patients/
│       ├── BCICIV_2a_1.csv
│       ├── ...
│       └── BCICIV_2a_9.csv
└── README.md
```

---

## Pipeline Overview

```
Raw EEG (N, 22, 201)
       │
       ▼
  Preprocessing
  ├── Baseline correction  (subtract pre-cue mean, −0.1 to 0.0 s)
  ├── Bandpass filter      (8–30 Hz for CSP/Riemannian; per-band for FBCSP)
  └── Artifact rejection   (drop trials where peak-to-peak > 100 µV)
       │
       ├──► CSP features  ──► LDA / SVM / XGBoost / Ensemble
       ├──► Covariance     ──► Riemannian MDM / Tangent Space + LR
       ├──► FBCSP features ──► Logistic Regression
       └──► Raw (normalised) ─► EEGNet (CNN)
```

---

## Methods

### 1. CSP + LDA

**Common Spatial Patterns** (OAS-regularised covariance) with 6 spatial filters per one-vs-rest decomposition. Log-variance features passed to **Linear Discriminant Analysis**.

- Gold-standard BCI pipeline
- Fast and interpretable
- Sensitivity: one fixed frequency band (8–30 Hz)

### 2. CSP + SVM (RBF)

Same CSP features as above, classified with a **Support Vector Machine** (RBF kernel, C=10, gamma='scale'). The RBF kernel handles non-linear boundaries in the log-variance feature space.

### 3. Voting Ensemble (LDA + SVM + RF)

Soft-vote over LDA, SVM, and a **Random Forest** (100 trees), all trained on the same CSP features. Diversity comes from the classifiers, not the features.

### 4. Riemannian MDM

**Minimum Distance to Mean** — computes per-trial covariance matrices (OAS) and assigns each trial to the class whose Riemannian geometric mean covariance is closest under the affine-invariant metric.

- No spatial filter selection required
- Robust to electrode perturbations
- Preserves full spectral structure of the covariance

### 5. Riemannian Tangent Space + LR

Projects covariance matrices to the **Euclidean tangent space** at the geometric mean of all training matrices, then applies Logistic Regression. Linearising the SPD manifold makes the full 22×22 covariance information accessible to linear classifiers.

### 6. Filter Bank CSP + LR (FBCSP)

Applies **independent CSP decompositions** across four frequency bands, then concatenates the log-variance features:

| Band       | Range    | Neural Correlate |
|------------|----------|-----------------|
| Theta      | 4–8 Hz   | Attentional modulation |
| Alpha / Mu | 8–13 Hz  | Sensorimotor idle rhythms |
| Beta low   | 13–22 Hz | Motor preparation/execution |
| Beta high  | 22–30 Hz | Beta rebound |

Features are standardised (z-score) before multinomial Logistic Regression. Uses baseline-corrected but **unfiltered** input so each band is cleanly extracted.

### 7. XGBoost on CSP Features

**XGBoost** (200 trees, max_depth=4, lr=0.1) applied on the same OAS-CSP log-variance features as LDA/SVM. Enables a direct comparison of non-linear tree ensembles vs. parametric classifiers on identical features.

### 8. EEGNet (Deep Learning)

**EEGNet** (Lawhern et al., 2018) — compact CNN with depthwise and separable convolutions operating on raw (bandpass-filtered, z-score-normalised) EEG:

```
Block 1: Temporal Conv (F1=8, kernel=64)
       → Depthwise Spatial Conv (D=2)
       → ELU → AvgPool(1×4) → Dropout(0.5)
Block 2: Depthwise Temporal Conv (kernel=16)
       → Pointwise Conv (F2=16)
       → ELU → AvgPool(1×8) → Dropout(0.5)
Classifier: Linear (F2 × T_out → 4)
```

- ~2,300 trainable parameters
- Trained with Adam (lr=1e-3, weight decay=1e-4)
- Early stopping on validation loss (patience=40)
- Evaluated under **LOSO** — a cross-subject protocol

---

## Evaluation Protocol

| Model Group | Protocol | Description |
|-------------|----------|-------------|
| Classical & Riemannian | **5-fold stratified CV** (per-subject) | Each subject's data split into 5 folds; model trained on 4, tested on 1. Repeated for all subjects. |
| EEGNet | **LOSO** (Leave-One-Subject-Out) | Trained on 8 subjects, tested on held-out 9th. 10 % of training subjects used as validation for early stopping. |

**Metrics reported**: Mean Accuracy ± std across subjects, Cohen's κ (chance = 0, perfect = 1).

> LOSO results are not directly comparable to per-subject CV — LOSO is a harder cross-subject generalisation test.

---

## Results Summary

Expected performance ranges (actual values populated when notebook is run):

| Method | Protocol | Expected Mean Acc |
|--------|----------|-------------------|
| CSP + LDA | 5-fold per-subject | 0.40–0.52 |
| CSP + SVM | 5-fold per-subject | 0.42–0.55 |
| Voting Ensemble | 5-fold per-subject | 0.43–0.56 |
| Riemannian MDM | 5-fold per-subject | 0.40–0.53 |
| Riem TS + LR | 5-fold per-subject | 0.43–0.56 |
| FBCSP + LR | 5-fold per-subject | 0.42–0.56 |
| XGBoost | 5-fold per-subject | 0.38–0.52 |
| EEGNet | LOSO (cross-subject) | 0.28–0.40 |

Chance level: **0.25** (4 classes).

---

## Visualizations

The notebook (Section 8) produces:

1. **Bar chart** — mean accuracy and Cohen's κ for all 8 methods with error bars
2. **Per-patient heatmap** — accuracy per method × patient (RdYlGn colormap, 0.20–0.85)
3. **Confusion matrices** — normalized 4×4 matrices for all models showing class-level confusion

---

## Setup & Requirements

### Python Environment

The project uses the `tf_env` conda/virtualenv:

```bash
/home/fwahyudi/tf_env/bin/python
```

### Dependencies

```
numpy>=1.26
pandas>=3.0
scipy>=1.17
matplotlib
mne>=1.12
scikit-learn
torch>=2.6 (CUDA 12.4)
pyriemann>=0.11
xgboost>=3.3
```

### Install missing packages

```bash
/home/fwahyudi/tf_env/bin/pip install pyriemann xgboost
```

---

## How to Run

1. Ensure all CSV files are present in `bci2a/patients/patients/`
2. Open `main.ipynb` in JupyterLab/Notebook using the `tf_env` kernel
3. Run all cells sequentially (Kernel → Restart & Run All)

**Expected total runtime**: 30–90 minutes depending on hardware (Sections 3–7 are the bottleneck; GPU accelerates Section 4 EEGNet only).

### Section outline

| Section | Content |
|---------|---------|
| 1 | EDA — data loading, class distribution, PSD, channel maps |
| 2 | Preprocessing — bandpass, envelope, artifact overview |
| 3 | CSP + LDA / SVM / Ensemble (per-subject 5-fold CV + LOSO baseline) |
| 4 | EEGNet — LOSO training, final model, inference helper |
| 5 | Riemannian MDM and Tangent Space + LR |
| 6 | Filter Bank CSP + Logistic Regression |
| 7 | XGBoost on CSP features |
| 8 | Comparison: evaluation matrix, bar charts, heatmap, confusion matrices |
| 9 | Conclusion and recommendations |

---

## Key Findings & Conclusion

### Accuracy ceiling from short window
The 0.8 s epoch window is the dominant constraint on all models. Extending to 2–4 s post-cue would be the single most impactful improvement.

### Riemannian geometry is the strongest classical approach
Working directly on covariance matrices preserves full spectral structure and avoids committing to a small number of spatial filters. Tangent-space projection makes the full 22-channel covariance accessible to linear classifiers.

### FBCSP adds multi-band information
Single-band CSP (8–30 Hz) discards discriminative information in theta and sub-band beta. FBCSP recovers this by concatenating features from four independent band-CSP decompositions.

### Inter-subject variability dominates
All models show ≥0.10 std across subjects. Some subjects are near-chance regardless of method; others exceed 0.70. Subject-specific tuning and longer calibration sessions are more impactful than method choice.

### EEGNet under LOSO is the hardest setting
Cross-subject generalisation is limited by between-subject variability in EEG topography and signal quality. Transfer learning or domain adaptation would be the natural next step.

### Recommended improvements

| Direction | Expected Gain |
|-----------|--------------|
| Extend epoch window to 2–4 s | +10–20 % |
| ICA artifact rejection | +3–5 % |
| FBCSP + mutual-information feature selection | +2–4 % |
| Deep Riemannian networks (SPDNet) | +5–10 % |
| Transfer learning / domain adaptation for EEGNet | Improved LOSO |

---

## References

- Brunner, C. et al. (2008). *BCI Competition 2008 – Graz Data Set A*. Graz University of Technology.
- Lawhern, V. J. et al. (2018). *EEGNet: A compact convolutional neural network for EEG-based brain–computer interfaces*. Journal of Neural Engineering, 15(5).
- Ang, K. K. et al. (2008). *Filter Bank Common Spatial Pattern (FBCSP) in Brain-Computer Interface*. IEEE IJCNN.
- Barachant, A. et al. (2012). *Multiclass Brain–Computer Interface Classification by Riemannian Geometry*. IEEE TNSRE, 20(3).
- Lotte, F. et al. (2018). *A review of classification algorithms for EEG-based brain–computer interfaces: a 10 year update*. Journal of Neural Engineering, 15(3).
