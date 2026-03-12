# Diagnostic Pneumotachometer Simulator with AI-Based Classification

## Executive Summary

This project presents a comprehensive, physics-based software simulation of a Fleisch diagnostic pneumotachometer (electronic spirometer) with an integrated artificial intelligence module for automated respiratory disease classification. Developed for the Medical Equipment II (SBE 3220) course, the system demonstrates the core electro-pneumatic operating principles of spirometry—from fluid dynamics and transducer physics to digital signal processing and clinical feature extraction—without the need for physical hardware.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Installation](#installation)
5. [Usage Instructions](#usage-instructions)
6. [AI Model Details](#ai-model-details)
7. [Technical Specifications](#technical-specifications)
8. [File Descriptions](#file-descriptions)
9. [Testing](#testing)

---

## Project Overview

### Background

Spirometry is the most widely performed respiratory diagnostic test globally, used to screen, diagnose, and monitor lung pathologies. Diagnostic pneumotachometers are precision medical instruments that measure instantaneous respiratory airflow by transducing the differential pressure drop across a resistive barrier into an electrical signal. 

### Problem Statement

Demonstrating the internal workings of a pneumotachometer requires visualizing microscopic physical interactions (Hagen-Poiseuille laminar flow, Wheatstone bridge piezoresistive effects) and high-speed digital signal processing (anti-aliasing, numerical integration). Physical hardware obscures these mathematical processes, making it difficult to analyze how specific hardware failures (e.g., thermal drift) or physiological changes (e.g., airway obstruction) impact the final clinical diagnosis.

### Solution

This project implements a complete "digital twin" of the measurement pipeline. It explicitly models the physics of a forced expiratory maneuver and the subsequent electronic processing, combining:
1. **First-Principles Physics Engine** - Simulates physiological pressure, adds transducer noise, and integrates the signal.
2. **AI-Powered Diagnostics** - A Machine Learning classifier that evaluates the output metrics against ATS/ERS clinical standards to detect Obstructive and Restrictive diseases.

---

## Key Features

### Realistic Signal Synthesis & Transduction
- **Physiological Modeling**: Generates forced expiratory pressure waveforms using dynamic time constants ($\tau$) to simulate lung elasticity and airway resistance.
- **Wheatstone Bridge Simulation**: Injects Additive White Gaussian Noise (AWGN) and simulates thermal DC baseline drift to represent raw, unfiltered hardware outputs.
- **Fluid Dynamics**: Strictly applies the Hagen-Poiseuille Law ($Q = K \cdot \Delta P$) under laminar flow constraints ($Re < 2300$).

### Advanced Digital Signal Processing
- **Zero-Phase Digital Filtering**: Utilizes a 2nd-order Butterworth low-pass filter (50 Hz) applied via `filtfilt` to prevent non-linear phase distortion that would skew clinical timings.
- **Real-Time Numerical Integration**: Converts discrete flow samples ($L/s$) into cumulative volume ($L$) using the second-order accurate ($O(\Delta t^2)$) Trapezoidal Rule.
- **Algorithmic Feature Extraction**: Automatically detects the end-of-test threshold (< 0.025 L/s) and latches the exact volume at $t=1.000s$ to extract FVC and $FEV_1$.

### Professional Clinical Dashboard (GUI)
- **4-Panel Diagnostic Display**: Replicates high-end medical software with real-time graphs for Pressure, Flow, Volume (Spirogram), and the Flow-Volume Loop.
- **Clinical Profiles**: Instantly switch between "Normal", "Obstructive (COPD)", "Restrictive", and "Sensor Zero-Drift" patients to see the mathematical variations in real-time.
- **Medical Equipment Aesthetic**: Dark theme with professional color coding and clean metric displays.

### AI Diagnostic Classification
- **Machine Learning Algorithm**: Random Forest Classifier trained on synthetic ATS/ERS-compliant data.
- **Diagnostic Output**: Instantly categorizes the maneuver as Normal, Obstructive, or Restrictive with an output confidence score.

---

## System Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                  Graphical User Interface (GUI)                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │   Patient    │  │   Clinical   │  │   Diagnostic Grids   │   │
│  │   Profiles   │  │   Metrics    │  │   (4-Panel Display)  │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────┴────────────────────────────────────┐
│              Pneumotachometer Physics Engine                    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Waveform Synthesis & Noise                             │    │
│  │  - Phase 1: Sinusoidal Rise (Muscular effort, 0-0.35s)  │    │
│  │  - Phase 2: Exponential Decay (Elastic recoil, tau)     │    │
│  │  - Noise: Additive White Gaussian Noise (AWGN)          │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Transduction & Signal Processing                       │    │
│  │  - LPF: Zero-phase 2nd-order Butterworth (50Hz)         │    │
│  │  - Physics: Hagen-Poiseuille Conversion (Q = K * dP)    │    │
│  │  - Math: Trapezoidal Numerical Integration              │    │
│  └─────────────────────────────────────────────────────────┘    │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────┴────────────────────────────────────┐
│            AI Diagnostic Module (Random Forest)                 │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Training Phase (Offline)                               │    │
│  │  - Generate 2000 ATS/ERS compliant clinical samples     │    │
│  │  - Feature extraction: [FVC, FEV1, Ratio]               │    │
│  │  - Train Random Forest Classifier                       │    │
│  └─────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Prediction Phase (Real-time)                           │    │
│  │  - Receive extracted metrics from Physics Engine        │    │
│  │  - Predict: Normal, Obstructive, or Restrictive         │    │
│  │  - Output diagnostic confidence percentage              │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘

```

---

## Installation

### Prerequisites

* Python 3.7 or higher
* pip package manager

### Required Libraries

```bash
pip install numpy>=1.19.0
pip install scipy>=1.7.0
pip install scikit-learn>=0.24.0
pip install matplotlib>=3.3.0

```

*(Note: `tkinter` is included with standard Python distributions).*

### Installation Steps

1. Clone or download the project files to your local machine.
2. Navigate to the project directory in your terminal.
3. Install the required dependencies.
4. Verify the core logic by running the test suite:
```bash
python test_system.py

```



---

## Usage Instructions

### Starting the Application

Launch the main graphical interface:

```bash
python spirometry_gui.py

```

### Running a Simulated Spirometry Test

1. **Select Patient Profile**: Use the dropdown menu in the upper left to select a clinical condition:
* *Normal*: Healthy adult baseline.
* *Obstructive (COPD)*: Simulates bronchospasm (low flow, high time constant).
* *Restrictive*: Simulates fibrosis (low volume, normal flow ratio).
* *Sensor Zero-Drift*: Simulates uncalibrated electronic drift to demonstrate numerical integration failure.


2. **Start Maneuver**: Click the green "START FVC MANEUVER" button.
3. **Monitor Real-Time Processing**: Watch the 4-panel graphs dynamically render the 6-second exhalation.
4. **Review Diagnosis**: Once the maneuver concludes, the system will automatically extract FVC, $FEV_1$, and the Tiffeneau Index, passing them to the AI for a final diagnosis.

### Understanding the Clinical Display

* **Panel 1 (Top Left)**: Shows the raw, noisy differential pressure signal overlayed with the cleanly filtered array.
* **Panel 2 (Top Right)**: Shows instantaneous Volumetric Flow $Q(t)$. The red dashed line signifies $t=1.0s$ (the $FEV_1$ boundary).
* **Panel 3 (Bottom Left)**: The Spirogram. Visualizes the continuous discrete-time integration of flow into volume $V(t)$.
* **Panel 4 (Bottom Right)**: The classic Flow-Volume loop. Clinicians use the "shape" of this curve (e.g., scooped descents) to verify obstructive patterns.

---

## AI Model Details

### Algorithm Selection: Random Forest Classifier

A Random Forest Classifier was chosen over standard logical thresholds to handle edge-cases and boundary conditions organically, making the system more robust to varying patient demographics.

### Training Data Characteristics

The AI is trained on 2,000 synthetically generated data points strictly following ATS/ERS clinical guidelines:

* **Normal**: $FVC \ge 4.0L$, $FEV_1/FVC \ge 70\%$
* **Obstructive**: $FEV_1/FVC < 70\%$ (hallmark of Asthma/COPD)
* **Restrictive**: $FVC < 80\%$ of predicted normal, but $FEV_1/FVC \ge 70\%$

### Real-Time Prediction

The model scales the newly generated features (`StandardScaler`), queries the forest, and outputs the highest probability class alongside a strict confidence metric.

---

## Technical Specifications

### Physics & Hardware Parameters (Fleisch Device)

| Parameter | Symbol | Simulated Value |
| --- | --- | --- |
| Capillary Count | $N$ | 500 |
| Capillary Radius | $R$ | 0.5 mm |
| Device Length | $L$ | 15 mm |
| Conductance | $K$ | $4.52 \times 10^{-5}$ $m^3/s/Pa$ |
| Peak Pressure | $\Delta P$ | ~177 Pa (1.33 mmHg) |
| Max Reynolds No. | $Re$ | 1294 (Strictly Laminar) |

### Integration Mathematics

The core integration uses the Trapezoidal Rule to minimize discrete-time approximation error at a 1 kHz sampling frequency ($dt = 1ms$):

```python
V[i] = V[i-1] + 0.5 * (Q[i] + Q[i-1]) * dt

```

This guarantees an integration error bounded by $O(\Delta t^2)$, satisfying the ATS requirement for volume accuracy ($\pm 50 mL$).

---

## File Descriptions

* **`pneumotach_engine.py`**: The core simulation engine. Contains the mathematical models for waveform synthesis, noise generation, `SciPy` digital filtering, and Trapezoidal integration.
* **`diagnostic_classifier.py`**: The AI module. Generates synthetic spirometry data, trains the Random Forest model, and provides the prediction API.
* **`spirometry_gui.py`**: The `Tkinter` application. Handles the layout, manages threading for the real-time playback, and renders the Matplotlib 4-panel diagnostic grid.
* **`test_system.py`**: Automated engineering validation suite. Tests the fluid dynamics boundaries, confirms integration accuracy, and validates AI classification logic.

---

## Testing

### Running the Test Suite

Execute the automated validation script:

```bash
python test_system.py

```

### Test Coverage

1. **Hagen-Poiseuille & Integration Engine**: Validates that generated waveforms correctly translate $Pa$ to $L/s$ and integrate strictly within physiological normal bounds (4.0 - 6.0L).
2. **Pathological Simulations**: Ensures that modifying the mathematical time constant ($\tau = 1.5s$) accurately forces the $FEV_1/FVC$ ratio below the 70% threshold.
3. **AI Diagnostic Validation**: Tests edge-case data points to ensure the Random Forest correctly identifies Obstruction vs. Restriction.

---

## Important Disclaimers

**EDUCATIONAL PURPOSE ONLY**

This is a software simulation project developed strictly for academic engineering purposes (Course SBE 3220). It is:

* NOT intended for actual medical use.
* NOT FDA approved or medically certified.
* NOT a substitute for actual medical equipment or spirometry hardware.
* NOT validated with real human patient data.

The system demonstrates engineering concepts, physical laws, and digital signal processing techniques but should never be used in any clinical setting or for diagnostic purposes on actual patients.

---

## Acknowledgments

**Institution**: Cairo University, Faculty of Engineering

**Department**: Systems & Biomedical Engineering

**Course**: Medical Equipment II (SBE 3220)

**Supervisor**: Dr. Sherif H. El-Gohary

**Team 7 Members**:

* Ahmed Salah Geoshy Elshenawy
* Ahmed Ahmed Mokhtar
* Osama Magdy Ali Khalifa
* Mohamed Hamdy Abdelhamed
* Mennat Allah Khalifa

---

**Document Last Updated**: March 2026