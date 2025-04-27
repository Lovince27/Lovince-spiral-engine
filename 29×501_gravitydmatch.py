import math
import unittest

# Constants and inputs from your model (same as in your main code)
PLANCK_ENERGY = 1.956e9
PLANCK_TIME = 5.39e-44
PLANCK_LENGTH = 1.616e-35

CBM = 5.10e-19
ENTROPY = 3.1989
FRACTAL_DIM = 1.3234
MUTUAL_INFO = 0.1789
PHI = 0.1456
ENERGY_RATIO = 2.51e-15
MACRO_SCALE = 1.0

alpha = 1 / 137.035999084

def calculate_lhs(cbm, entropy, fractal_dim, planck_energy, planck_length, macro_scale):
    cbm_norm = cbm / planck_energy
    scale_ratio = (planck_length / macro_scale) ** 2
    return cbm_norm * entropy * fractal_dim * scale_ratio

def calculate_rhs(mutual_info, phi, planck_time, energy_ratio, planck_energy):
    mutual_info_term = mutual_info ** 2 * phi
    time_norm = 1 / planck_time
    energy_norm = (energy_ratio / planck_energy) ** 0.25
    return mutual_info_term * time_norm * energy_norm

def holographic_correction(rhs, macro_scale, planck_length):
    bekenstein_entropy = (4 * math.pi * macro_scale ** 2) / (4 * planck_length ** 2)
    return rhs * bekenstein_entropy

def fine_structure_adjustment(lhs, rhs_corrected, alpha):
    return lhs * rhs_corrected * alpha ** 3

class TestQCMCalculation(unittest.TestCase):

    def test_lhs_calculation(self):
        lhs = calculate_lhs(CBM, ENTROPY, FRACTAL_DIM, PLANCK_ENERGY, PLANCK_LENGTH, MACRO_SCALE)
        # Expected value from your logs or manual calculation
        expected_lhs = 8.177e-135
        self.assertAlmostEqual(lhs, expected_lhs, delta=expected_lhs*1e-3)

    def test_rhs_calculation(self):
        rhs = calculate_rhs(MUTUAL_INFO, PHI, PLANCK_TIME, ENERGY_RATIO, PLANCK_ENERGY)
        # Expected value before holographic correction (approximate)
        expected_rhs = 9.09e54  # Rough estimate from your logs (rhs before correction)
        self.assertAlmostEqual(rhs, expected_rhs, delta=expected_rhs*1e-2)

    def test_holographic_correction(self):
        rhs = calculate_rhs(MUTUAL_INFO, PHI, PLANCK_TIME, ENERGY_RATIO, PLANCK_ENERGY)
        rhs_corr = holographic_correction(rhs, MACRO_SCALE, PLANCK_LENGTH)
        expected_rhs_corr = 9.195e56
        self.assertAlmostEqual(rhs_corr, expected_rhs_corr, delta=expected_rhs_corr*1e-3)

    def test_fine_structure_adjustment(self):
        lhs = calculate_lhs(CBM, ENTROPY, FRACTAL_DIM, PLANCK_ENERGY, PLANCK_LENGTH, MACRO_SCALE)
        rhs = calculate_rhs(MUTUAL_INFO, PHI, PLANCK_TIME, ENERGY_RATIO, PLANCK_ENERGY)
        rhs_corr = holographic_correction(rhs, MACRO_SCALE, PLANCK_LENGTH)
        qcm = fine_structure_adjustment(lhs, rhs_corr, alpha)
        expected_qcm = 1.000240
        self.assertAlmostEqual(qcm, expected_qcm, delta=1e-5)

    def test_qcm_close_to_unity(self):
        lhs = calculate_lhs(CBM, ENTROPY, FRACTAL_DIM, PLANCK_ENERGY, PLANCK_LENGTH, MACRO_SCALE)
        rhs = calculate_rhs(MUTUAL_INFO, PHI, PLANCK_TIME, ENERGY_RATIO, PLANCK_ENERGY)
        rhs_corr = holographic_correction(rhs, MACRO_SCALE, PLANCK_LENGTH)
        qcm = fine_structure_adjustment(lhs, rhs_corr, alpha)
        self.assertTrue(math.isclose(qcm, 1.0, rel_tol=1e-4))

if __name__ == '__main__':
    unittest.main()
