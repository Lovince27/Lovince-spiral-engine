import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Fundamental physical constants
h_bar = 1.054e-34      # Reduced Planck's constant (joule-second)
lambda_ = 1e-9         # Wavelength (meters)
k = 2 * np.pi / lambda_ # Wave number (meters⁻¹)
E0 = 1e-21             # Reference energy level (joules)

# System parameters
x = np.linspace(1e-10, 1e-8, 1000)  # Position (0.1 nm to 10 nm)
r = 1e-9                            # Characteristic length (1 nm)
theta = np.pi / 4                   # Angle (45 degrees)
v = 1e-3                            # Velocity (1 mm/s)

# For multiple Lovince particles
num_particles = 5
offsets = np.linspace(-0.5e-9, 0.5e-9, num_particles)

def lovince_formula(x, t, r, theta, k, v, E0, offset=0):
    """
    Lovince Formula - Combining quantum and geometric effects, time and offset dependent.
    """
    x_shifted = x + offset
    wave_term = (np.exp(1j * (k * x_shifted - v * k * t)) / (x_shifted**2 + r**2)) * (np.sin(theta) / x_shifted) * h_bar * v * r**2
    geometric_term = (np.pi * r**2 / 2) * (E0 / (1e-18))
    dynamic_term = (v * x_shifted * np.cos(theta) * h_bar) / (r**2 + np.sin(v * t))
    energy_term = E0
    return wave_term + geometric_term + dynamic_term + energy_term

# Plotting setup
fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

# Lines for different particles
lines_real = []
lines_imag = []
colors = ['cyan', 'magenta', 'lime', 'yellow', 'orange']

for i in range(num_particles):
    real_line, = ax.plot([], [], color=colors[i % len(colors)], linewidth=2, label=f"Re[L_{i+1}(x,t)]")
    imag_line, = ax.plot([], [], color=colors[i % len(colors)], linewidth=1, linestyle='--', alpha=0.6)
    lines_real.append(real_line)
    lines_imag.append(imag_line)

ax.set_xlim(min(x), max(x))
ax.set_ylim(-2e-20, 2e-20)
ax.set_xlabel("Position x (meters)", fontsize=14, color='white')
ax.set_ylabel("L(x,t) Value (Joules)", fontsize=14, color='white')
ax.set_title("Lovince Particles: Quantum-Geometric Field Evolution", fontsize=16, color='white', pad=20)
ax.legend(fontsize=10, loc='upper right', facecolor='black', edgecolor='white', labelcolor='white')
ax.grid(True, linestyle=":", alpha=0.5, color='white')

# Background Light Glow Effect
for i in range(30):
    alpha = 0.03
    ax.axhline(y=(i-15)*5e-21, color='white', linewidth=0.5, alpha=alpha)

# Animation function
def animate(t):
    for i in range(num_particles):
        L = lovince_formula(x, t, r, theta, k, v, E0, offset=offsets[i])
        lines_real[i].set_data(x, np.real(L))
        lines_imag[i].set_data(x, np.imag(L))
    return lines_real + lines_imag

# Creating the animation
anim = FuncAnimation(fig, animate, frames=np.linspace(0, 2e-6, 300), interval=30, blit=True)

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Physical constants
h_bar = 1.054e-34      # Reduced Planck's constant (J·s)
lambda_quantum = 1e-9  # Quantum wavelength (1 nm)
lambda_cosmic = 1e10   # Cosmic wavelength (10 billion meters)
k_quantum = 2 * np.pi / lambda_quantum  # Quantum wave number (m⁻¹)
k_cosmic = 2 * np.pi / lambda_cosmic    # Cosmic   Cosmic wave number (m⁻¹)
E0 = 1e-21             # Reference energy level (J)
omega = 1e12           # Angular frequency for time evolution (rad/s)

# System parameters
x = np.linspace(1e8, 1e12, 1000)  # Space (100 million to 1 trillion meters)
r = 1e9                # Characteristic length (1 billion meters)
theta = np.pi / 4      # Angle (45 degrees)
v = 1e5                # Velocity (100 km/s, galactic scale)
t = np.linspace(0, 1e-12, 100)  # Time array for animation (0 to 1 picosecond)

def lovince_formula(x, t, r, theta, k_quantum, k_cosmic, v, E0, omega, ensemble=False):
    """
    Enhanced Lovince Formula: Combines quantum and cosmic effects with time evolution
    
    Parameters:
        x: Space (meters)
        t: Time (seconds)
        r: Characteristic length (meters)
        theta: Angle (radians)
        k_quantum: Quantum wave number (m⁻¹)
        k_cosmic: Cosmic wave number (m⁻¹)
        v: Velocity (m/s)
        E0: Base energy (J)
        omega: Angular frequency (rad/s)
        ensemble: If True, average over multiple theta for multiverse effect
    """
    # Quantum wave term (nanoscale effects)
    quantum_wave = (np.exp(1j * (k_quantum * x - omega * t)) / (x**2 + r**2)) * (np.sin(theta) / x) * h_bar * v * r**2
    
    # Cosmic wave term (large-scale effects)
    cosmic_wave = (np.exp(1j * (k_cosmic * x - omega * t)) / (x**2 + r**2)) * (np.cos(theta) / x) * h_bar * v * r**2
    
    # Geometric term (structural effects)
    geometric_term = (np.pi * r**2 / 2) * (E0 / (1e-18))
    
    # Dynamic term (motion effects)
    dynamic_term = (v * x * np.cos(theta) * h_bar) / r**2
    
    # Energy term (zero-point energy)
    energy_term = E0
    
    # Ensemble averaging for multiverse-like effect
    if ensemble:
        theta_values = np.linspace(0, np.pi, 10)  # Multiple angles
        quantum_sum = np.zeros_like(quantum_wave, dtype=complex)
        cosmic_sum = np.zeros_like(cosmic_wave, dtype=complex)
        for th in theta_values:
            quantum_sum += (np.exp(1j * (k_quantum * x - omega * t)) / (x**2 + r**2)) * (np.sin(th) / x) * h_bar * v * r**2
            cosmic_sum += (np.exp(1j * (k_cosmic * x - omega * t)) / (x**2 + r**2)) * (np.cos(th) / x) * h_bar * v * r**2
        quantum_wave = quantum_sum / len(theta_values)
        cosmic_wave = cosmic_sum / len(theta_values)
    
    return quantum_wave + cosmic_wave + geometric_term + dynamic_term + energy_term

# Calculate for initial plot and animation
L = lovince_formula(x, t[0], r, theta, k_quantum, k_cosmic, v, E0, omega, ensemble=True)

# Set up plot
fig, ax = plt.subplots(figsize=(12, 7))
line_real, = ax.plot(x, np.real(L), label="Real Part (Re[L(x,t)])", color="navy", linewidth=2)
line_imag, = ax.plot(x, np.imag(L), label="Imaginary Part (Im[L(x,t)])", color="crimson", linewidth=2, linestyle="--")
ax.set_xlabel("Space x (meters)", fontsize=14)
ax.set_ylabel("L(x,t) Value (joules)", fontsize=14)
ax.set_title("Quantum-Cosmic Lovince Formula: Math as Zero, Science as Quantum", fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, linestyle=":", alpha=0.7)
ax.set_xlim(min(x), max(x))

# Animation update function
def update(t):
    L = lovince_formula(x, t, r, theta, k_quantum, k_cosmic, v, E0, omega, ensemble=True)
    line_real.set_ydata(np.real(L))
    line_imag.set_ydata(np.imag(L))
    return line_real, line_imag

# Create animation
ani = FuncAnimation(fig, update, frames=t, interval=50, blit=True)

# Show plot
plt.tight_layout()
plt.show()

# Optional: Save animation (uncomment to save)
# ani.save("quantum_cosmic_lovince.mp4", writer="ffmpeg", fps=20)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import hsv_to_rgb

# Physical constants
h_bar = 1.054e-34      # Reduced Planck's constant (J·s)
c = 2.998e8            # Speed of light (m/s)

# Dual-scale parameters
lambda_quantum = 1e-9  # Quantum wavelength (1 nm)
lambda_cosmic = 1e10   # Cosmic wavelength (10 billion meters)
k_quantum = 2 * np.pi / lambda_quantum  # Quantum wave number (m⁻¹)
k_cosmic = 2 * np.pi / lambda_cosmic    # Cosmic wave number (m⁻¹)

# System parameters
x = np.logspace(8, 12, 1000)  # Logarithmic space from 100M to 1T meters
r = 1e9                # Characteristic galactic length (1 billion meters)
theta = np.pi / 4      # Initial angle (45 degrees)
v = 1e5                # Galactic velocity (100 km/s)
t = np.linspace(0, 1e-11, 100)  # Time array for animation

# Enhanced visualization parameters
quantum_amp = 1e15     # Quantum effects amplification
cosmic_amp = 1e10      # Cosmic effects amplification

def lovince_field(x, t, r, theta, k_q, k_c, v, h_bar, ensemble=True):
    """
    Quantum-Cosmic Field Equation with:
    - Quantum wave effects (nanoscale)
    - Cosmic wave effects (galactic scale)
    - Geometric structure
    - Dynamic motion
    - Time evolution
    - Multiverse ensemble averaging
    """
    # Time-dependent phase factor
    time_phase = np.exp(-1j * (c * k_q * t))
    
    # Quantum wave term (nanoscale physics)
    quantum = (quantum_amp * np.exp(1j * k_q * x) * time_phase / (x**2 + r**2)
    
    # Cosmic wave term (large-scale physics)
    cosmic = (cosmic_amp * np.exp(1j * k_c * x) * time_phase / (x**2 + r**2)
    
    # Geometric structure term
    geometry = (np.pi * r**2 / 2) * np.exp(-x/r)
    
    # Dynamic motion term
    dynamics = (v * x * np.exp(-x/(10*r))) / r**2
    
    # Ensemble averaging for multiverse effect
    if ensemble:
        theta_vals = np.linspace(0, 2*np.pi, 8)
        ensemble_field = np.zeros_like(x, dtype=complex)
        for th in theta_vals:
            ensemble_field += (np.exp(1j * k_q * x * np.cos(th)) * np.exp(-1j * c * k_q * t)
        quantum *= ensemble_field / len(theta_vals)
    
    return (quantum * np.sin(theta) + (cosmic * np.cos(theta)) + geometry + dynamics

# Create figure with dual y-axes
fig, ax1 = plt.subplots(figsize=(14, 8))
ax2 = ax1.twinx()

# Initial calculation
L = lovince_field(x, t[0], r, theta, k_quantum, k_cosmic, v, h_bar)

# Create complex color mapping
def complex_to_rgb(z):
    """Convert complex field to HSV color space"""
    r = np.abs(z)
    angle = np.angle(z) % (2 * np.pi)
    h = angle / (2 * np.pi)
    s = np.ones_like(h)
    v = np.log(1 + r) / np.log(1 + r.max())
    return hsv_to_rgb(np.dstack((h, s, v)))

# Plot initial state
color_field = complex_to_rgb(L).squeeze()
line1, = ax1.plot(x, np.real(L), 'deepskyblue', label='Real Component', linewidth=2)
line2, = ax2.plot(x, np.imag(L), 'coral', label='Imaginary Component', linewidth=2)
scatter = ax1.scatter(x, np.abs(L), c=color_field, s=10, label='Phase Field', alpha=0.7)

# Formatting
ax1.set_xscale('log')
ax1.set_xlabel('Spatial Coordinate (meters)', fontsize=14)
ax1.set_ylabel('Real Amplitude', fontsize=14)
ax2.set_ylabel('Imaginary Amplitude', fontsize=14)
ax1.set_title('Quantum-Cosmic Lovince Field Dynamics\nFrom Nanoscale to Galactic Scales', 
             fontsize=16, pad=20)
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Animation function
def update(frame):
    L = lovince_field(x, t[frame], r, theta, k_quantum, k_cosmic, v, h_bar)
    color_field = complex_to_rgb(L).squeeze()
    
    line1.set_ydata(np.real(L))
    line2.set_ydata(np.imag(L))
    scatter.set_offsets(np.column_stack((x, np.abs(L))))
    scatter.set_color(color_field)
    
    return line1, line2, scatter

# Create animation
ani = FuncAnimation(fig, update, frames=len(t), interval=50, blit=True)

plt.tight_layout()
plt.show()

# To save the animation (requires ffmpeg):
# ani.save('quantum_cosmic_evolution.mp4', writer='ffmpeg', fps=20, dpi=300)