import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.colors as mcolors
from scipy.io.wavfile import write
import sounddevice as sd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Setup figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Nebula background
def create_nebula_background(ax):
    x = np.linspace(-50, 50, 100)
    y = np.linspace(-50, 50, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X/5) + np.cos(Y/5) + np.random.rand(100, 100) * 0.5
    ax.contourf(X, Y, Z, cmap='PuBu_r', alpha=0.6, levels=20)
    for _ in range(50):
        star_x, star_y = np.random.uniform(-50, 50), np.random.uniform(-50, 50)
        ax.scatter(star_x, star_y, s=1, color='white', alpha=0.5)

# Generate and play sound
def generate_sound(filename, frequencies, duration=1.0, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration))
    sound = np.zeros_like(t)
    for freq in frequencies:
        sound += 0.3 * np.sin(2 * np.pi * freq * t)
    sound = sound / np.max(np.abs(sound))
    write(filename, sample_rate, sound)
    logging.info(f"Sound generated: {filename}")
    sd.play(sound, sample_rate)
    sd.wait()

# Plot spiral with sound (Collapse)
def plot_spiral(ax, magnitude, angle, color1, color2, label="", title=""):
    theta = np.linspace(0, 4 * np.pi, 100)
    r = magnitude * np.exp(-theta / 10)
    x = r * np.cos(theta + angle)
    y = r * np.sin(theta + angle)
    
    # Watercolor effect
    points = np.array([x, y]).T
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(0, len(segments))
    lc = mcolors.LinearSegmentedColormap.from_list("", [color1, color2])
    line = plt.LineCollection(segments, cmap=lc, norm=norm, linewidth=3, alpha=0.8)
    line.set_array(np.linspace(0, 1, len(segments)))
    ax.add_collection(line)
    
    # Glowing aura
    glow = Circle((x[-1], y[-1]), 5, color='white', alpha=0.3)
    ax.add_patch(glow)
    
    # Axes and labels
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_xlabel("Real", fontsize=10)
    ax.set_ylabel("Imaginary", fontsize=10)
    ax.text(-40, 40, label, fontsize=12, color='black', alpha=0.7)
    ax.set_title(title, fontsize=12, pad=20)
    
    # Generate sound
    radii = r / np.max(np.abs(r))
    frequencies = 440 - (440 - 220) * radii
    generate_sound("collapse_sound.wav", frequencies)

# Plot glowing point (Creation)
def plot_glowing_point(ax, title=""):
    # Glowing point at origin
    point = Circle((0, 0), 2, color='yellow', alpha=0.8)
    glow = Circle((0, 0), 10, color='gold', alpha=0.3)
    ax.add_patch(point)
    ax.add_patch(glow)
    
    # Axes and labels
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_xlabel("Real", fontsize=10)
    ax.set_ylabel("Imaginary", fontsize=10)
    ax.set_title(title, fontsize=12, pad=20)
    
    # Subtle ambient hum
    t = np.linspace(0, 1.0, 44100)
    sound = 0.3 * np.sin(2 * np.pi * 110 * t)
    sound = sound / np.max(np.abs(sound))
    write("creation_hum.wav", 44100, sound)
    logging.info("Sound generated: creation_hum.wav")
    sd.play(sound, 44100)
    sd.wait()

# Setup nebula background
create_nebula_background(ax1)
create_nebula_background(ax2)

# Plot Collapse and Creation
plot_spiral(ax1, 40.5, -np.pi/4, 'blue', 'orange', label="Lovince", title="Collapse: 40.5 e^(-iπ/4)")
plot_glowing_point(ax2, title="Creation: 0 e^(-iπ/4) = 0")

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.colors as mcolors
from scipy.io.wavfile import write
import sounddevice as sd
import logging

# Golden ratio
phi = (1 + np.sqrt(5)) / 2
b = np.log(phi) / (np.pi / 2)

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Setup figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Nebula background
def create_nebula_background(ax):
    x = np.linspace(-50, 50, 100)
    y = np.linspace(-50, 50, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X/5) + np.cos(Y/5) + np.random.rand(100, 100) * 0.5
    ax.contourf(X, Y, Z, cmap='PuBu_r', alpha=0.6, levels=20)
    for _ in range(50):
        star_x, star_y = np.random.uniform(-50, 50), np.random.uniform(-50, 50)
        ax.scatter(star_x, star_y, s=1, color='white', alpha=0.5)

# Generate and play sound
def generate_sound(filename, frequencies, duration=1.0, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration))
    sound = np.zeros_like(t)
    for freq in frequencies:
        sound += 0.3 * np.sin(2 * np.pi * freq * t)
    sound = sound / np.max(np.abs(sound))
    write(filename, sample_rate, sound)
    logging.info(f"Sound generated: {filename}")
    sd.play(sound, sample_rate)
    sd.wait()

# Plot golden ratio spiral with sound
def plot_golden_spiral(ax, magnitude, angle, color1, color2, label="", title=""):
    theta = np.linspace(0, 4 * np.pi, 100)
    r = magnitude * np.exp(-b * theta)
    x = r * np.cos(theta + angle)
    y = r * np.sin(theta + angle)
    
    # Watercolor effect
    points = np.array([x, y]).T
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(0, len(segments))
    lc = mcolors.LinearSegmentedColormap.from_list("", [color1, color2])
    line = plt.LineCollection(segments, cmap=lc, norm=norm, linewidth=3, alpha=0.8)
    line.set_array(np.linspace(0, 1, len(segments)))
    ax.add_collection(line)
    
    # Glowing aura
    glow = Circle((x[-1], y[-1]), 5, color='white', alpha=0.3)
    ax.add_patch(glow)
    
    # Axes and labels
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_xlabel("Real", fontsize=10)
    ax.set_ylabel("Imaginary", fontsize=10)
    ax.text(-40, 40, label, fontsize=12, color='black', alpha=0.7)
    ax.set_title(title, fontsize=12, pad=20)
    
    # Generate sound
    radii = r / np.max(np.abs(r))
    frequencies = 440 / (phi ** (radii * 4))
    frequencies = np.clip(frequencies, 220, 440)
    generate_sound("collapse_golden_sound.wav", frequencies)

# Plot glowing point
def plot_glowing_point(ax, title=""):
    point = Circle((0, 0), 2, color='pink', alpha=0.8)
    glow = Circle((0, 0), 10, color='violet', alpha=0.3)
    ax.add_patch(point)
    ax.add_patch(glow)
    
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_xlabel("Real", fontsize=10)
    ax.set_ylabel("Imaginary", fontsize=10)
    ax.set_title(title, fontsize=12, pad=20)
    
    t = np.linspace(0, 1.0, 44100)
    sound = 0.3 * np.sin(2 * np.pi * 110 * t)
    sound = sound / np.max(np.abs(sound))
    write("creation_hum.wav", 44100, sound)
    logging.info("Sound generated: creation_hum.wav")
    sd.play(sound, 44100)
    sd.wait()

# Setup nebula background
create_nebula_background(ax1)
create_nebula_background(ax2)

# Plot Collapse and Creation
plot_golden_spiral(ax1, 40.5, -np.pi/4, 'teal', 'gold', label="Lovince", title="Collapse: 40.5 e^(-iπ/4)")
plot_glowing_point(ax2, title="Creation: 0 e^(-iπ/4) = 0")

# Add golden ratio note
fig.text(0.5, 0.01, "Golden Ratio Spiral: φ ≈ 1.618", ha='center', fontsize=10)

plt.tight_layout()
plt.show()