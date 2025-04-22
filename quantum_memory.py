import numpy as np
import matplotlib.pyplot as plt

# Constants
phi = (1 + np.sqrt(5)) / 2
pi = np.pi
hbar = 1.055e-34
log2 = np.log(2)

# Physical Parameters
E = 1.0
F = 1.0
ma = 1.0
mc2 = 1.0
alpha = 1.0
S = 256
gamma = 0.05  # memory decay rate

# Time and Quantum Levels
t_values = np.linspace(0, 50, 200)     # Time array
n = 5                                  # Fixed quantum level to visualize
memory = np.cos(0.2 * t_values)        # Oscillatory memory function

# Ψ_n(t) computation
psi_values = (
    (E * (hbar**alpha) * ma * (phi**n) * (pi**(3*n - 1)) * np.log(S)) /
    (F * mc2 * log2)
) * memory * np.exp(-gamma * t_values)

# Plotting Ψ_n(t)
plt.figure(figsize=(10, 6))
plt.plot(t_values, psi_values, color='darkgreen')
plt.title(f"Ψₙ(t) with Memory Decay (n = {n})")
plt.xlabel("Time (t)")
plt.ylabel("Ψₙ(t)")
plt.grid(True)
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import hilbert

# Constants with precise values and units
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio (dimensionless)
PI = np.pi                   # Pi (dimensionless)
HBAR = 1.055e-34             # Reduced Planck constant (J·s)
LOG2 = np.log(2)             # Natural log of 2

# Physical Parameters with realistic units
E = 1.6e-19                  # Energy (1 eV in Joules)
F = 1e-12                    # Force (1 pN in Newtons)
ma = 9.11e-31                # Electron mass (kg)
mc2 = 8.187e-14              # Electron rest energy (J)
alpha = 1/137                # Fine structure constant
S = 256                      # Entropy (dimensionless)
gamma = 0.05                 # Memory decay rate (1/ps)

# Enhanced time parameters
t_start = 0
t_end = 50                   # picoseconds
num_points = 1000
t_values = np.linspace(t_start, t_end, num_points) * 1e-12  # Convert to seconds

# Quantum levels to analyze
n_values = np.arange(1, 6)   # From ground state to n=5

# Advanced memory functions
def memory_function(t):
    """Combined memory effects with multiple frequencies"""
    return (0.6 * np.cos(0.2 * t) + 
            0.3 * np.sin(0.5 * t) + 
            0.1 * np.cos(1.0 * t))

memory = memory_function(t_values * 1e12)  # Keep argument in ps for scaling

# Enhanced Ψ_n(t) computation with vectorization
def compute_psi(n, t, memory):
    prefactor = (E * (HBAR**alpha) * ma * (PHI**n) * (PI**(3*n - 1)) * np.log(S)
    prefactor /= (F * mc2 * LOG2)
    return prefactor * memory * np.exp(-gamma * t * 1e12)  # gamma in 1/ps

# Compute for all quantum levels
psi_results = {n: compute_psi(n, t_values, memory) for n in n_values}

# Hilbert transform for instantaneous amplitude and phase
analytic_signal = {n: hilbert(psi_results[n]) for n in n_values}
amplitude_envelope = {n: np.abs(analytic_signal[n]) for n in n_values}
instantaneous_phase = {n: np.unwrap(np.angle(analytic_signal[n])) for n in n_values}

# Create professional visualization
plt.style.use('seaborn-v0_8-poster')
fig = plt.figure(figsize=(18, 12), dpi=100)
gs = GridSpec(3, 2, figure=fig)

# Main wavefunction plot
ax1 = fig.add_subplot(gs[0:2, 0:1])
for n in n_values:
    ax1.plot(t_values * 1e12, psi_results[n], label=f'n = {n}', linewidth=2)
    ax1.plot(t_values * 1e12, amplitude_envelope[n], '--', color='gray', alpha=0.5)
ax1.set_title('Quantum Wavefunctions with Memory Effects', fontsize=16)
ax1.set_xlabel('Time (ps)', fontsize=14)
ax1.set_ylabel('Ψₙ(t) (arb. units)', fontsize=14)
ax1.legend(title='Quantum Level', title_fontsize=13)
ax1.grid(True, which='both', linestyle='--', alpha=0.6)

# Phase plot
ax2 = fig.add_subplot(gs[0:2, 1:1])
for n in n_values:
    ax2.plot(t_values * 1e12, instantaneous_phase[n], label=f'n = {n}')
ax2.set_title('Instantaneous Phase', fontsize=16)
ax2.set_xlabel('Time (ps)', fontsize=14)
ax2.set_ylabel('Phase (rad)', fontsize=14)
ax2.legend()
ax2.grid(True, which='both', linestyle='--', alpha=0.6)

# Power spectrum analysis
ax3 = fig.add_subplot(gs[2, 0:1])
for n in n_values:
    fft = np.fft.fft(psi_results[n])
    freqs = np.fft.fftfreq(len(t_values), t_values[1]-t_values[0]) / 1e9  # in GHz
    ax3.semilogy(freqs[:len(freqs)//2], np.abs(fft[:len(freqs)//2])**2, label=f'n = {n}')
ax3.set_title('Power Spectrum', fontsize=16)
ax3.set_xlabel('Frequency (GHz)', fontsize=14)
ax3.set_ylabel('Power (arb. units)', fontsize=14)
ax3.legend()
ax3.grid(True, which='both', linestyle='--', alpha=0.6)

# Decay rate analysis
ax4 = fig.add_subplot(gs[2, 1:1])
for n in n_values:
    log_amp = np.log(np.maximum(1e-15, np.abs(amplitude_envelope[n])))
    ax4.plot(t_values * 1e12, log_amp, label=f'n = {n}')
ax4.set_title('Logarithmic Amplitude Decay', fontsize=16)
ax4.set_xlabel('Time (ps)', fontsize=14)
ax4.set_ylabel('log|Ψₙ(t)|', fontsize=14)
ax4.legend()
ax4.grid(True, which='both', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

# Additional analysis: Correlation between quantum levels
correlation_matrix = np.zeros((len(n_values), len(n_values)))
for i, n1 in enumerate(n_values):
    for j, n2 in enumerate(n_values):
        correlation_matrix[i,j] = np.corrcoef(psi_results[n1], psi_results[n2])[0,1]

# Display correlation matrix
fig2, ax = plt.subplots(figsize=(8, 6))
cax = ax.matshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
fig2.colorbar(cax)
ax.set_xticks(np.arange(len(n_values)))
ax.set_yticks(np.arange(len(n_values)))
ax.set_xticklabels(n_values)
ax.set_yticklabels(n_values)
ax.set_title('Wavefunction Correlation Matrix', pad=20)
plt.show()