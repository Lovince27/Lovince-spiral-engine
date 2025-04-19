import cmath
import math

def compute_zn(fn, fn_minus_1):
    # θ = tan⁻¹(F_{n-1}/F_n)
    theta = math.atan(fn_minus_1 / fn)
    # Z_n = F_n · e^(iθ)
    zn = fn * cmath.exp(1j * theta)
    return zn

# Lucas sequence example (first few terms)
lucas_seq = [1, 3, 4, 7, 11, 18, 29]

# Spiral generation
print("Lovince Spiral (Z_n) values:")
for n in range(1, len(lucas_seq)):
    fn = lucas_seq[n]
    fn_1 = lucas_seq[n - 1]
    zn = compute_zn(fn, fn_1)
    print(f"Z_{n+1} = {zn.real:.3f} + {zn.imag:.3f}i")

import numpy as np
import matplotlib.pyplot as plt

# Lovince Formula
magnitude = 5
angle_deg = 30.9
angle_rad = np.deg2rad(angle_deg)
z = magnitude * (np.cos(angle_rad) + 1j * np.sin(angle_rad))

# Plot on Argand Plane
plt.figure(figsize=(6, 6))
plt.plot([0, z.real], [0, z.imag], 'ro-', label='z ≈ 5e^(i30.9°)')
plt.grid(True)
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.title('Lovince Formula Visualization')
plt.legend()
plt.show()

print(f"Lovince Formula: z = {z:.3f}")


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Define the number of terms (changeable)
n = 10  # Up to a_n, adjust as needed
growth_factor = 0.6  # Initial growth, can be tweaked (e.g., phi ≈ 1.618 for golden ratio)

# Generate sequence with dynamic growth
real_parts = [1.0]  # Start with a1 = 1 + 1i
imag_parts = [1.0]
for i in range(2, n + 1):
    if i == 2:
        real = 1.6  # a2 = 1.6 + 1.6i
    elif i == 3:
        real = 2.6  # a3 = 2.6 + 2.6i
    else:
        # Dynamic growth: use previous + growth_factor + slight increase
        real = real_parts[-1] + growth_factor + (i - 3) * 0.1  # Gradual acceleration
    imag = real  # Imaginary = Real (45° line pattern)
    real_parts.append(real)
    imag_parts.append(imag)

# Create complex points
points = [complex(r, im) for r, im in zip(real_parts, imag_parts)]

# Initialize plot
fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
scatter = ax.scatter([], [], color='gold', s=100, label='Points', zorder=5)
line, = ax.plot([], [], 'b--', label='Spiral Path', zorder=1)
annotations = []

# Set up plot limits and labels
ax.set_title(f'Lovince Quantum Spiral (up to a{n})', fontsize=14, pad=10)
ax.set_xlabel('Real Axis', fontsize=12)
ax.set_ylabel('Imaginary Axis', fontsize=12)
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, zorder=0)
ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, zorder=0)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend()
ax.axis('equal')

# Animation update function
def update(frame):
    if frame < len(real_parts):
        scatter.set_offsets(np.c_[real_parts[:frame+1], imag_parts[:frame+1]])
        line.set_data(real_parts[:frame+1], imag_parts[:frame+1])
        # Update or add annotations
        while len(annotations) < frame + 1:
            ann = ax.annotate(f'a{len(annotations)+1} = {real_parts[len(annotations)]:.1f} + {imag_parts[len(annotations)]:.1f}i',
                             (real_parts[len(annotations)], imag_parts[len(annotations)]), xytext=(5, 5), textcoords='offset points')
            annotations.append(ann)
    return scatter, line, annotations

# Create animation
ani = animation.FuncAnimation(fig, update, frames=len(real_parts), interval=500, blit=True, repeat=False)

# Display
plt.show()


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import sounddevice as sd

# --- Spiral Sequence Parameters ---
n = 10
growth_factor = 0.6
real_parts = [1.0]
imag_parts = [1.0]

for i in range(2, n + 1):
    if i == 2:
        real = 1.6
    elif i == 3:
        real = 2.6
    else:
        real = real_parts[-1] + growth_factor + (i - 3) * 0.1
    imag = real
    real_parts.append(real)
    imag_parts.append(imag)

points = [complex(r, im) for r, im in zip(real_parts, imag_parts)]

# --- Frequency Mapping Function ---
def point_to_frequency(c):
    mag = abs(c)
    freq = 200 + (mag / max(np.abs(points))) * (963 - 200)
    return freq

# --- Sound Generator ---
def play_tone(frequency, duration=0.2, samplerate=44100):
    t = np.linspace(0, duration, int(samplerate * duration), False)
    wave = np.sin(2 * np.pi * frequency * t)
    sd.play(wave, samplerate=samplerate)
    sd.wait()

# --- Matplotlib Setup ---
fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
scatter = ax.scatter([], [], color='gold', s=100, zorder=5)
line, = ax.plot([], [], 'b--', label='Spiral Path', zorder=1)
annotations = []

ax.set_title(f'Lovince Quantum Spiral with Sound (aₙ, n=1 to {n})', fontsize=14, pad=10)
ax.set_xlabel('Real Axis')
ax.set_ylabel('Imaginary Axis')
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend()
ax.axis('equal')

# --- Animation Function ---
def update(frame):
    if frame < len(points):
        scatter.set_offsets(np.c_[real_parts[:frame+1], imag_parts[:frame+1]])
        line.set_data(real_parts[:frame+1], imag_parts[:frame+1])
        if len(annotations) < frame + 1:
            ann = ax.annotate(
                f'a{frame+1} = {real_parts[frame]:.1f} + {imag_parts[frame]:.1f}i',
                (real_parts[frame], imag_parts[frame]),
                xytext=(5, 5), textcoords='offset points'
            )
            annotations.append(ann)

        # Sound: Quantum frequency tone per frame
        freq = point_to_frequency(points[frame])
        play_tone(freq)

    return scatter, line, annotations

ani = animation.FuncAnimation(fig, update, frames=len(points), interval=600, blit=False, repeat=False)

plt.show()


# Old line (deprecated in Matplotlib 3.7):
# colors = plt.cm.get_cmap(self.colormap)(norm(range(len(t))))

# Updated line (compatible with Matplotlib 3.7+ and 3.11+):
cmap = plt.colormaps[self.colormap]  # Access colormap from new API
colors = cmap(norm(range(len(t))))   # Apply normalized range to colormap


norm = plt.Normalize(vmin=0, vmax=len(t))  # Ensure proper normalization
cmap = plt.colormaps[self.colormap]        # New Matplotlib 3.7+ compatible
colors = cmap(norm(range(len(t))))         # Apply colormap to normalized range
self.spiral.set_color(colors)