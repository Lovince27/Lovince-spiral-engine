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