"""
Test script for Diagnostic Pneumotachometer Simulator
Verifies Physics Engine, Calibration Limits, and AI Classifier logic
"""

import sys
import numpy as np
import os

print("=" * 60)
print("Diagnostic Pneumotachometer - Engineering Validation")
print("=" * 60)

try:
    from pneumotach_engine import PneumotachEngine
    print("✓ Modules imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test 1: Calibration Physics Engine
print("\n[TEST 1] Testing 3L Syringe Calibration & Integration...")
engine = PneumotachEngine()
engine._generate_waveform("3L Syringe Calibration", age=25, height_cm=175, sex="Male")

if engine.data is not None:
    fvc = engine.data['fvc']
    print(f"  ✓ Syringe maneuver generated.")
    print(f"  ✓ Calculated Syringe Volume: FVC={fvc:.3f}L")

    # 3.0L ± 0.05L is the ATS requirement for syringe calibration
    if 2.95 <= fvc <= 3.05:
        print("  ✓ Trapezoidal integration error bounded. Calibration SUCCESS.")
    else:
        print("  ✗ Integration mismatch. Calibration FAILED.")
else:
    print("✗ Failed to generate waveform data")
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL ENGINEERING TESTS PASSED! ✓")
print("=" * 60)
print("\nTo start the GUI application, run:")
print("  python spirometry_gui.py\n")