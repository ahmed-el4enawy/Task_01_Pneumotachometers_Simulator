# Diagnostic Pneumotachometer Simulator - Comprehensive Code & Science Breakdown

This document provides a detailed, line-by-line and section-by-section breakdown of the entire Diagnostic Pneumotachometer Simulator project. It covers the scientific background, the architectural code flow, and the specific Python implementations for all core modules.

---

## 1. Scientific & Clinical Background

### What is a Pneumotachometer?
A **pneumotachometer** is a medical device used in spirometry to measure the volume of air an individual can inhale or exhale as a function of time. Specifically, a **Fleisch pneumotachometer** measures flow by placing a slight resistance (a bundle of capillary tubes) in the airstream and measuring the pressure drop ($\Delta P$) across this resistance.

### Core Physics Equations
1. **Hagen-Poiseuille Law ($Q = K \cdot \Delta P$)**:
   Under strictly laminar flow conditions (Reynolds number $Re < 2300$), the volumetric flow rate ($Q$) of a fluid is directly proportional to the pressure drop ($\Delta P$) across a resistive element. The constant $K$ is the conductance of the device.

2. **Numerical Integration (Trapezoidal Rule)**:
   The physical sensor measures pressure, which is converted to flow ($L/s$). However, clinical spirometry requires Volume ($L$). Volume is the integral of flow over time: $V(t) = \int Q(t) dt$. In discrete digital systems, this is calculated using the Trapezoidal Rule:    
   `V[i] = V[i-1] + 0.5 * (Q[i] + Q[i-1]) * dt`
   This method limits the integration error to $O(\Delta t^2)$, making it highly accurate for medical use.

3. **Body Temperature and Pressure Saturated (BTPS)**:
   Air inside the lungs is at body temperature ($37^\circ C$) and 100% humidity. When exhaled into a room at standard temperature, the air cools and contracts. A standard BTPS correction factor (typically ~1.11) is mathematically applied to convert the measured room-temperature flow back to the actual lung volume.

### AI Diagnostics (Obstructive vs Restrictive)
Spirometry categorizes lung diseases based on two main metrics:
- **FVC (Forced Vital Capacity)**: Total volume exhaled forcefully.
- **FEV1 (Forced Expiratory Volume in 1 second)**: Volume exhaled in the first second.
- **Ratio (FEV1/FVC)**: Normal is $\ge 70\%$. 
  - **Obstructive** (e.g., Asthma, COPD): Ratio $< 70\%$, meaning air is exiting the lungs too slowly due to airway constriction.
  - **Restrictive** (e.g., Pulmonary Fibrosis): Ratio $\ge 70\%$ but FVC is severely reduced (Total lung capacity is small).

---

## 2. Project Architecture & Code Flow

The application follows a **Model-View-Controller (MVC) logic combined with a pipeline architecture**:
1. **User Input / View (`spirometry_gui.py`)**: The user selects a demographic profile and disease state, then clicks "Start".
2. **Physics Engine (`pneumotach_engine.py`)**: Generates a completely synthetic pressure waveform based on human physiology equations. It adds noise imitating an electronic sensor, applies digital filters, runs the Hagen-Poiseuille conversion, applies the BTPS factor, and integrates for volume.
3. **Real-time Loop (`spirometry_gui.py`)**: The GUI fetches data from the engine in chunks of 50ms to simulate real-time streaming to the charts.
4. **Clinical Metrics Output**: Once the engine reaches 6.0 seconds, it extracts the final FVC, FEV1, and Ratio to display them cleanly on the screen.

---

## 3. Line-by-Line / Section-by-Section Code Breakdown

### A. `pneumotach_engine.py`
This is the core mathematical engine.

* **Lines 1-8 (Imports)**: Imports `numpy` for fast array mathematics and `scipy.signal` for digital filtering (`butter`, `filtfilt`).
* **Lines 10-26 (`__init__`)**: Sets up system constants. 
  - `self.fs = 1000.0` means the system samples at 1 kHz (1000 times a second), giving a time step `dt` of 1 millisecond.
  - `self.t_end = 6.0` simulates a standard 6-second exhalation.
  - `self.K` calculates the Fleisch resistance constant based on a 177 Pa drop at 8.0 L/s flow.
* **Lines 28-37 (`_calculate_predicted`)**: Uses NHANES III empirical formulas to calculate what the patient's lung capacity *should* be based on their height, age, and sex. 
* **Lines 39-48 (`start_maneuver` & `stop_maneuver`)**: Control flags. Calling `start_maneuver` triggers the waveform generation.
* **Lines 49-67 (`_generate_waveform` setup)**: 
  Sets up arrays and determines physical constants based on disease:
  - `tau` ($\tau$) is the time constant of the lung. A high $\tau$ (e.g., 1.5s in Obstructive) means the lungs empty very slowly. A low $\tau$ (0.3s in Restrictive) means they empty fast but volume is low.
  - `dc_offset` creates a sensor calibration error used in the "Zero-Drift" profile.
* **Lines 69-85 (Waveform Synthesis)**:
  - **Phase 1 (Rise)**: Simulates the muscular effort of pushing air out during the first 0.35 seconds using a Sine wave: `peak_dP * np.sin(...)`.
  - **Phase 2 (Decay)**: The passive elastic recoil of the lungs pushes the rest of the air out as an exponential decay curve: `peak_dP * np.exp(...)`.
* **Lines 87-88 (Sensor Noise)**: Adds Additive White Gaussian Noise (AWGN) to make the signal look like it's coming from real silicon sensors.
* **Lines 90-93 (Digital Filter)**: Creates a 2nd-order Butterworth Low-Pass Filter at 50Hz. Uses `filtfilt` for zero-phase filtering so the signal isn't delayed in time, preserving clinical timings.
* **Lines 95-99 (Pneumatic Conversion)**: $Q = K \times \Delta P_{filtered} \times BTPS$. Sets any negative flow back to 0 (since patients aren't inhaling).
* **Lines 101-104 (Integration)**: The crucial loop that implements the Trapezoidal Rule to find Volume ($V$) from Flow ($Q$).
* **Lines 106-115 (Feature Extraction)**:
  - FEV1 is pulled exactly at the 1.0-second array index (`int(1.0 / self.dt)`).
  - FVC detects the end-of-test when flow drops below 25 mL/s (`Q < 0.025`).
* **Lines 129-155 (`get_current_state`)**: Acts as a data streamer. Instead of returning the whole array at once, it slices the arrays up to `current_index` and advances by `advance_by_ms` to trick the GUI into rendering an animation over time.

---

### B. `spirometry_gui.py`
The desktop frontend using `tkinter` and `matplotlib`.

* **Lines 1-16 (Imports)**: Standard GUI libraries, plotting utilities, CSV writer, and the core scripts built earlier.
* **Lines 18-39 (`__init__`)**: Window setup and color palette. A custom dictionary of hex codes is used to build a "dark-mode" medical aesthetic.
* **Lines 49-91 (`setup_styles`)**: Tkinter requires deep configuration of `ttk.Style` to override the native Windows white dropdown menus. This section forcibly maps the background and foreground rules for Comboboxes and Spinboxes to match the dark theme.
* **Lines 93-191 (`create_ui` Layout)**: Builds the main layout:
  - **Header**: Title and export button.
  - **Left Panel (Controls)**: Takes user demographic inputs (Age, Height, Sex). Binds these to Tkinter `Var` objects. Uses a Combobox for selecting the Disease Profile. Also tracks labels for FVC/FEV1 rendering.
* **Lines 191-206 (`create_ui` Graphics Setup)**: 
  - Initializes a Matplotlib `Figure`.
  - Creates 4 distinct subplots (`add_subplot(2,2,1)` through `(2,2,4)`).
  - Uses `FigureCanvasTkAgg` to stick the Matplotlib figure directly into the Tkinter window.
* **Lines 212-232 (`reset_graphs`)**: Cleans out the graphs and reformats them. Applies dark mode styling to the Matplotlib grids, axes, and spine borders before drawing.
* **Lines 234-260 (`start_test`)**: Event handler. Validates that human ages/heights are positive numbers. Grays out buttons during testing, then tells the `engine.start_maneuver()` to build the mathematical arrays.
* **Lines 262-322 (`update_loop`)**: The central animation logic.
  - Called constantly every 50 milliseconds via `self.root.after(50, self.update_loop)`.
  - Asks the engine, "Give me the state arrays up to the current time."
  - **Panel 1**: Plots noisy vs filtered pressure.
  - **Panel 2**: Plots flow against time (using `fill_between` for solid colors).
  - **Panel 3**: Plots the Spirogram (Volume vs Time).
  - **Panel 4**: Plots Flow vs Volume (The FV Loop).
  - **When finished**: Analyzes the final data, calculates predicted % using the demographics, updates the clinic cards, and restores the UI buttons.
* **Lines 324-355 (`export_csv`)**: Extracts data from `self.last_state`, opens an OS filepath dialog, and cleanly writes patient demographics and all 6000 lines of time-series arrays to a CSV file.

---

### C. `test_system.py`
An automated validation script to ensure physics math remains accurate during refactors.

* **Lines 1-20 (Setup)**: Safely attempts imports with error catching.
* **Lines 22-39 (Test 1: Calibration Physics Engine)**:
  - Generates a specific `3L Syringe Calibration` profile.
  - Clinically, spirometers are calibrated daily by pumping a 3.00 Liter cylindrical syringe into them.
  - The script checks if the Trapezoidal integration correctly converts this theoretical sine wave back into exactly 3.0 Liters. If the math drifts heavily outside 2.95–3.05L, the test aggressively fails.
* **Lines 41-45**: Final success prints.

---
**Summary:** This full-stack simulation perfectly chains low-level medical physics equations for fluid dynamics and DSP into an accessible UI.
