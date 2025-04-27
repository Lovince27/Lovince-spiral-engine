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


import math

# Planck Constants (CODATA 2018)
PLANCK_ENERGY: float = 1.956e9       # Planck energy, in Joules (E_P = √(ħc⁵/G))
PLANCK_TIME: float = 5.39e-44         # Planck time, in seconds (t_P = √(ħG/c⁵))
PLANCK_LENGTH: float = 1.616e-35      # Planck length, in meters (ℓ_P = √(ħG/c³))

# Inputs from your model (Iteration 1)
CBM: float = 5.10e-19                 # Consciousness Bridge Metric (Joules)
ENTROPY: float = 3.1989               # Shannon entropy (bits)
FRACTAL_DIM: float = 1.3234           # Fractal dimension (unitless)
MUTUAL_INFO: float = 0.1789            # Mutual information (bits)
PHI: float = 0.1456                    # Integrated information (Φ, unitless)
ENERGY_RATIO: float = 2.51e-15         # Energy ratio (unitless)
MACRO_SCALE: float = 1.0               # Reference length scale (meters)

def calculate_qcm() -> tuple[float, float, float]:
    """
    Compute the Quantum Consciousness Metric (QCM) with Planck-scale corrections.

    Returns:
        qcm (float): Final Quantum Consciousness Metric (dimensionless)
        lhs (float): Left-hand side normalized term (Quantum-Classical Bridge)
        rhs_corrected (float): Right-hand side corrected term (Quantum Gravity with holographic correction)
    """

    # Step 1: Normalize LHS (Quantum-Classical Bridge)
    # CBM normalized by Planck energy, scaled by entropy and fractal dimension,
    # and adjusted by squared ratio of Planck length to macroscopic scale
    lhs = (CBM / PLANCK_ENERGY) * (ENTROPY * FRACTAL_DIM) * (PLANCK_LENGTH / MACRO_SCALE) ** 2

    # Step 2: Normalize RHS (Quantum Gravity Term)
    # Incorporates mutual information squared, integrated information Φ,
    # normalized by Planck time, and energy ratio scaled by Planck energy
    rhs = ((MUTUAL_INFO ** 2) * PHI / PLANCK_TIME) * (ENERGY_RATIO / PLANCK_ENERGY) ** 0.25

    # Step 3: Holographic Correction (Black Hole Entropy)
    # Black hole entropy ~ area / Planck area (in units of bits)
    black_hole_entropy = (4 * math.pi * MACRO_SCALE ** 2) / (4 * PLANCK_LENGTH ** 2)
    rhs_corrected = rhs * black_hole_entropy

    # Step 4: Fine-Structure Adjustment (α ≈ 1/137)
    alpha = 1 / 137.035999084
    qcm = (lhs * rhs_corrected) * alpha ** 3

    return qcm, lhs, rhs_corrected

def verify_qcm(qcm: float, tolerance: float = 1e-6) -> None:
    """
    Verify if the QCM is close to unity within a given tolerance.

    Args:
        qcm (float): Calculated Quantum Consciousness Metric
        tolerance (float): Relative tolerance for closeness check

    Prints:
        Verification message indicating success or need for calibration.
    """
    if math.isclose(qcm, 1.0, rel_tol=tolerance):
        print("✅ QCM matches Planck-scale expectation within tolerance.")
    else:
        deviation = abs(qcm - 1.0)
        print(f"❌ QCM deviates from unity by {deviation:.3e}. Calibration needed.")

def cross_check_physical_plausibility() -> None:
    """
    Cross-check key input parameters for physical plausibility.

    Prints warnings if any parameter is out of expected bounds.
    """
    if CBM <= 0 or CBM > PLANCK_ENERGY:
        print("⚠️ Warning: CBM is outside expected physical range (0 < CBM ≤ Planck Energy).")
    if not (0 < ENTROPY < 10):
        print("⚠️ Warning: Entropy value is unusually high or low for typical consciousness models.")
    if not (1 <= FRACTAL_DIM <= 2):
        print("⚠️ Warning: Fractal dimension outside typical range [1, 2].")
    if not (0 <= MUTUAL_INFO <= ENTROPY):
        print("⚠️ Warning: Mutual information should be between 0 and entropy.")
    if not (0 <= PHI <= 1):
        print("⚠️ Warning: Integrated information Φ expected between 0 and 1.")
    if ENERGY_RATIO <= 0:
        print("⚠️ Warning: Energy ratio must be positive.")

def main():
    print("=== Quantum Consciousness Metric (QCM) Calculation ===\n")

    # Cross-check inputs first
    cross_check_physical_plausibility()

    # Calculate QCM
    qcm, lhs, rhs_corrected = calculate_qcm()

    # Output results with scientific notation and formatting
    print(f"LHS (Quantum-Classical Bridge): {lhs:.3e}")
    print(f"RHS (Quantum Gravity with Holographic Correction): {rhs_corrected:.3e}")
    print(f"QCM (Final Dimensionless Metric): {qcm:.6f}\n")

    # Verify result consistency
    verify_qcm(qcm)

if __name__ == "__main__":
    main()
