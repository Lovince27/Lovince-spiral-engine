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


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import hilbert, spectrogram
from scipy.integrate import solve_ivp
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from tqdm import tqdm
from numba import jit
import qutip as qt

# Configure scientific plotting
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("rocket")
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.facecolor': 'white',
    'figure.dpi': 300,
    'figure.autolayout': True,
    'lines.linewidth': 1.2,
    'lines.markersize': 4,
    'mathtext.fontset': 'stix'
})

# ========== QUANTUM SYSTEM DEFINITION ==========
class NonMarkovianQuantumSystem:
    def __init__(self):
        # Fundamental constants (CODATA 2018)
        self.hbar = 1.054571817e-34      # Reduced Planck constant [J·s]
        self.kB = 1.380649e-23           # Boltzmann constant [J/K]
        self.e_charge = 1.602176634e-19  # Elementary charge [C]
        
        # System parameters
        self.T = 0.1                     # Temperature [K]
        self.omega0 = 2*np.pi*1e12       # Natural frequency [rad/s]
        self.gamma = 0.05e12             # Damping rate [rad/s]
        self.lambda_sb = 0.1             # System-bath coupling
        
        # Memory kernel parameters
        self.memory_timescales = np.array([0.1, 0.5, 2.0]) * 1e-12  # [s]
        self.memory_weights = np.array([0.6, 0.3, 0.1])
        
        # Numerical parameters
        self.t_start = 0                 # [ps]
        self.t_end = 50                  # [ps]
        self.num_points = 5000
        self.t_span = np.linspace(self.t_start, self.t_end, self.num_points) * 1e-12
        
        # Quantum state space
        self.n_levels = 5
        self.dim = self.n_levels + 1     # Hilbert space dimension
        self.psi0 = qt.basis(self.dim, 0) # Initial state |0⟩
        
    @jit(nopython=True)
    def memory_kernel(self, t):
        """Non-Markovian memory kernel with multiple exponential decays"""
        t_sec = t  # Time in seconds
        kernel = 0.0
        for tau, weight in zip(self.memory_timescales, self.memory_weights):
            kernel += weight * np.exp(-t_sec/tau)
        return kernel
    
    def spectral_density(self, omega):
        """Ohmic spectral density with Drude cutoff"""
        cutoff = 2*np.pi*1e13  # [rad/s]
        return 2 * self.lambda_sb * (omega/cutoff) / (1 + (omega/cutoff)**2)
    
    def non_linear_hamiltonian(self, t, psi):
        """Time-dependent nonlinear Hamiltonian"""
        # Convert state vector to Qobj
        psi_qt = qt.Qobj(psi.reshape(self.dim, 1))
        
        # Basic operators
        a = qt.destroy(self.dim)
        n = qt.num(self.dim)
        
        # Time-dependent components
        H0 = self.hbar * self.omega0 * (n + 0.5)
        H1 = 0.5 * self.hbar * self.gamma * (a + a.dag()) * np.sin(2*np.pi*1e12*t)
        
        # Nonlinear term (Kerr nonlinearity)
        xi = 0.1 * self.omega0  # Nonlinearity strength
        H_kerr = xi * (a.dag() * a)**2
        
        # Memory effects
        mem = self.memory_kernel(t)
        H_mem = mem * (a + a.dag())
        
        # Total Hamiltonian
        H = H0 + H1 + H_kerr + H_mem
        
        # Convert back to state vector
        return (-1j/self.hbar * H * psi_qt).full().flatten()
    
    def solve_dynamics(self):
        """Solve the non-Markovian quantum dynamics"""
        print("\nSolving non-Markovian quantum dynamics...")
        
        # Convert initial state to array
        psi0_array = self.psi0.full().flatten()
        
        # Solve using adaptive RK45 method
        sol = solve_ivp(
            fun=self.non_linear_hamiltonian,
            t_span=(self.t_span[0], self.t_span[-1]),
            y0=psi0_array,
            t_eval=self.t_span,
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        
        # Store results
        self.time = sol.t * 1e12  # Convert to ps
        self.states = sol.y
        
        # Calculate expectation values
        self.calculate_observables()
        
    def calculate_observables(self):
        """Calculate quantum observables from states"""
        print("Calculating quantum observables...")
        
        # Initialize operators
        a = qt.destroy(self.dim)
        n = qt.num(self.dim)
        x = (a + a.dag()) / np.sqrt(2)
        p = 1j*(a.dag() - a) / np.sqrt(2)
        
        # Initialize storage
        self.exp_n = np.zeros(len(self.time))
        self.exp_x = np.zeros(len(self.time))
        self.exp_p = np.zeros(len(self.time))
        self.wigner = []
        
        for i in tqdm(range(len(self.time))):
            # Convert to Qobj
            psi = qt.Qobj(self.states[:,i].reshape(self.dim,1))
            
            # Calculate expectations
            self.exp_n[i] = qt.expect(n, psi)
            self.exp_x[i] = qt.expect(x, psi)
            self.exp_p[i] = qt.expect(p, psi)
            
            # Calculate Wigner function at selected times
            if i % 500 == 0:
                xvec = np.linspace(-5, 5, 100)
                w = qt.wigner(psi, xvec, xvec)
                self.wigner.append((self.time[i], xvec, w))
    
    def thermal_correlation(self, t):
        """Thermal bath correlation function"""
        omega_c = 2*np.pi*1e13  # Cutoff frequency
        beta = 1/(self.kB * self.T)
        
        if t == 0:
            return self.lambda_sb * omega_c * (1/np.tan(0.5*beta*omega_c)) + 1j*self.lambda_sb*omega_c**2
        else:
            return (self.lambda_sb * omega_c / np.tan(0.5*beta*omega_c) * 
                   np.exp(-omega_c*np.abs(t)) + 
                   1j*self.lambda_sb*omega_c**2*np.exp(-omega_c*np.abs(t)))
    
    def visualize_results(self):
        """Create comprehensive visualization"""
        print("Generating visualizations...")
        
        # Create figure with constrained layout
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 3, figure=fig)
        
        # Plot expectation values
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.time, self.exp_n, label='⟨n⟩')
        ax1.plot(self.time, self.exp_x, label='⟨x⟩')
        ax1.plot(self.time, self.exp_p, label='⟨p⟩')
        ax1.set_xlabel('Time (ps)')
        ax1.set_ylabel('Expectation values')
        ax1.legend()
        ax1.set_title('Quantum Expectation Values')
        
        # Plot phase space
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.exp_x, self.exp_p)
        ax2.set_xlabel('⟨x⟩')
        ax2.set_ylabel('⟨p⟩')
        ax2.set_title('Quantum Phase Space')
        
        # Plot power spectrum
        ax3 = fig.add_subplot(gs[0, 2])
        fft = np.fft.fft(self.exp_x)
        freqs = np.fft.fftfreq(len(self.time), self.time[1]-self.time[0])
        ax3.semilogy(freqs[:len(freqs)//2], np.abs(fft[:len(freqs)//2])**2)
        ax3.set_xlabel('Frequency (THz)')
        ax3.set_ylabel('Power')
        ax3.set_title('Spectral Density')
        
        # Plot Wigner functions
        for i, (t, xvec, w) in enumerate(self.wigner):
            ax = fig.add_subplot(gs[1, i])
            cont = ax.contourf(xvec, xvec, w, 100, cmap='RdBu')
            ax.set_title(f'Wigner Function at t={t:.1f}ps')
            ax.set_xlabel('x')
            ax.set_ylabel('p')
            plt.colorbar(cont, ax=ax)
        
        # Plot density matrix evolution
        ax6 = fig.add_subplot(gs[2, 0])
        rho = qt.Qobj(self.states[:,-1].reshape(self.dim,1)) * qt.Qobj(self.states[:,-1].reshape(self.dim,1)).dag()
        mat = ax6.matshow(qt.expect(qt.qeye(self.dim), rho), cmap='viridis')
        plt.colorbar(mat, ax=ax6)
        ax6.set_title('Final Density Matrix')
        
        # Plot memory kernel
        ax7 = fig.add_subplot(gs[2, 1])
        memory = np.array([self.memory_kernel(t*1e-12) for t in self.time])
        ax7.plot(self.time, memory)
        ax7.set_xlabel('Time (ps)')
        ax7.set_ylabel('Memory Kernel')
        ax7.set_title('Non-Markovian Memory')
        
        # Plot thermal correlation
        ax8 = fig.add_subplot(gs[2, 2])
        corr_t = np.linspace(0, 5e-12, 100)
        corr = np.array([self.thermal_correlation(t) for t in corr_t])
        ax8.plot(corr_t*1e12, np.real(corr), label='Real')
        ax8.plot(corr_t*1e12, np.imag(corr), label='Imag')
        ax8.set_xlabel('Time (ps)')
        ax8.set_ylabel('Correlation')
        ax8.set_title('Bath Correlation Function')
        ax8.legend()
        
        plt.suptitle('Non-Markovian Quantum Dynamics', y=1.02)
        plt.show()

# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    print("🚀 Starting Advanced Quantum Memory Simulation")
    print("--------------------------------------------