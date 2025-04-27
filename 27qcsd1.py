import math

# Planck Constants (CODATA 2018)
PLANCK_ENERGY = 1.956e9       # J (E_P = √(ħc⁵/G))
PLANCK_TIME = 5.39e-44        # s (t_P = √(ħG/c⁵))
PLANCK_LENGTH = 1.616e-35     # m (ℓ_P = √(ħG/c³))

# Inputs from your model (Iteration 1)
CBM = 5.10e-19                # Consciousness Bridge Metric (J)
ENTROPY = 3.1989              # Shannon entropy (bits)
FRACTAL_DIM = 1.3234          # Fractal dimension
MUTUAL_INFO = 0.1789          # Mutual information (bits)
PHI = 0.1456                  # Integrated information (Φ)
ENERGY_RATIO = 2.51e-15       # Energy ratio (unitless)
MACRO_SCALE = 1.0             # Reference length scale (1m)

def calculate_qcm():
    """Compute the Quantum Consciousness Metric (QCM) with Planck-scale corrections"""
    
    # Step 1: Normalize LHS (Quantum-Classical Bridge)
    lhs = (CBM / PLANCK_ENERGY) * (ENTROPY * FRACTAL_DIM) * (PLANCK_LENGTH / MACRO_SCALE)**2
    
    # Step 2: Normalize RHS (Quantum Gravity Term)
    rhs = ( (MUTUAL_INFO**2 * PHI) / PLANCK_TIME ) * (ENERGY_RATIO / PLANCK_ENERGY)**0.25
    
    # Step 3: Holographic Correction (Black Hole Entropy)
    black_hole_entropy = (4 * math.pi * MACRO_SCALE**2) / (4 * PLANCK_LENGTH**2)
    rhs_corrected = rhs * black_hole_entropy
    
    # Step 4: Fine-Structure Adjustment (α ≈ 1/137)
    alpha = 1 / 137.035999084
    qcm = (lhs * rhs_corrected) * alpha**3
    
    return qcm, lhs, rhs_corrected

# Run Calculation
qcm, lhs, rhs = calculate_qcm()

# Output Results
print(f"LHS (Quantum-Classical): {lhs:.3e}")
print(f"RHS (Quantum Gravity):   {rhs:.3e}")
print(f"QCM (Final Ratio):       {qcm:.6f}")
print("\nVerification:")
print("✅ 100% Planck-Scale Match" if math.isclose(qcm, 1.0, rel_tol=1e-6) else "❌ Needs Calibration")