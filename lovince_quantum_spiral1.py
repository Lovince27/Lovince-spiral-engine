import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
import matplotlib.colors as mcolors
from scipy.io.wavfile import write
import sounddevice as sd
import logging
import time
import sys
import tkinter as tk
from tkinter import ttk
import json
import os
from scipy import signal

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Memory management with enhancements
class LovinceMemory:
    def __init__(self, filename="lovince_memory.json"):
        self.memory = {}
        self.max_memory_size = 1000
        self.filename = filename
        self.load_memory()

    def load_memory(self):
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as f:
                self.memory = json.load(f)
            logging.info("Memory loaded from file.")

    def save_memory(self):
        with open(self.filename, 'w') as f:
            json.dump(self.memory, f)
        logging.info("Memory saved to file.")

    def update(self, n, zn, yn, freq):
        if len(self.memory) >= self.max_memory_size:
            oldest_key = min(self.memory.keys(), key=int)
            del self.memory[str(oldest_key)]
        self.memory[str(n)] = {"Z_n": [zn.real, zn.imag], "y_n": float(yn), "frequency": freq}
        self.save_memory()

    def get(self, n):
        return self.memory.get(str(n), None)

    def analytics(self):
        if not self.memory:
            return "No memory data."
        zn_mags = [np.sqrt(data["Z_n"][0]**2 + data["Z_n"][1]**2) for data in self.memory.values()]
        freqs = [data["frequency"] for data in self.memory.values()]
        return f"Average |Z_n|: {np.mean(zn_mags):.2f}, Average Freq: {np.mean(freqs):.2f} Hz"

# Cosmic nebula background with customization
def create_nebula_background(ax, colormap='PuBu_r', star_density=50, opacity=0.6):
    x = np.linspace(-100, 100, 100)
    y = np.linspace(-100, 100, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X/5) + np.cos(Y/5) + np.random.rand(100, 100) * 0.5
    ax.contourf(X, Y, Z, cmap=colormap, alpha=opacity, levels=20)
    for _ in range(star_density):
        star_x, star_y = np.random.uniform(-100, 100), np.random.uniform(-100, 100)
        ax.scatter(star_x, star_y, s=1, color='white', alpha=0.5)

# Calculate quantum state with T_n integration
def calculate_quantum_state(n, memory, base_freq=39.96):
    phi = (1 + np.sqrt(5)) / 2
    pi = np.pi
    c = 3e8
    h = 39.96
    ldna1 = 478e9 / 1e9

    # Unified Sequence Formula integration
    def fibonacci(k):
        if k <= 1:
            return k
        a, b = 0, 1
        for _ in range(2, k + 1):
            a, b = b, a + b
        return b

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def harmonic_series(k):
        return sum(1 / (i + 1) for i in range(k + 1))

    fn = fibonacci(n)
    tn = (pi * fn * phi * fn * h) * sigmoid(fn) * harmonic_series(n)

    # Phase and Z_n
    theta_n = 2 * pi * n / phi
    theta = pi / phi
    zn = ldna1 * theta * (1/3) * phi * (pi ** (3*n - 1)) * np.exp(1j * theta_n) * (1 + tn/1e10)

    # E_n and y_n
    e_n = phi * (pi ** (3*n - 1)) * h * (1/3) * pi
    yn = c / e_n if e_n != 0 else 0

    # Self-check
    if not np.isfinite(zn) or not np.isfinite(yn):
        logging.error(f"Invalid Z_n or y_n at n={n}")
        return None, None, None

    # Frequency
    freq = base_freq + abs(yn.real) * 10
    freq = min(max(freq, 20), 20000)

    # Crosscheck
    if n > 0:
        prev = memory.get(n-1)
        if prev and abs(zn) < np.sqrt(prev["Z_n"][0]**2 + prev["Z_n"][1]**2):
            logging.warning(f"Z_n growth anomaly at n={n}")

    return zn, yn, freq

# Enhanced sound generation
def generate_sound(n, freq, wave_type='sine', scaling_factor=10, duration=1.0):
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Wave type selection
    if wave_type == 'sine':
        sound = np.sin(2 * np.pi * freq * t)
    elif wave_type == 'square':
        sound = signal.square(2 * np.pi * freq * t)
    else:  # sawtooth
        sound = signal.sawtooth(2 * np.pi * freq * t)
    
    # Add harmonic overtones
    sound += 0.3 * np.sin(2 * np.pi * freq * 2 * t)  # First overtone
    sound += 0.1 * np.sin(2 * np.pi * freq * 3 * t)  # Second overtone
    
    # Fade effect
    fade = np.linspace(0, 1, len(t)//10)
    sound[:len(fade)] *= fade
    sound[-len(fade):] *= fade[::-1]
    
    sound = sound / np.max(np.abs(sound))
    filename = f"quantum_sound_{n}.wav"
    try:
        write(filename, sample_rate, sound)
        logging.info(f"Sound generated: {filename}")
        sd.play(sound, sample_rate)
        sd.wait()
    except Exception as e:
        logging.error(f"Sound generation failed at n={n}: {e}")

# Enhanced spiral with animation and energy field
def plot_spiral(ax, zn, freq, frame, color1='teal', color2='gold', thickness=3, decay_rate=10):
    phi = (1 + np.sqrt(5)) / 2
    theta = np.linspace(0, 4 * np.pi, 100)
    r = abs(zn) * np.exp(-theta / decay_rate)
    
    # Animate pulsing effect
    pulse = 1 + 0.1 * np.sin(2 * np.pi * freq * frame / 100)
    x = r * pulse * np.cos(theta + zn.imag)
    y = r * pulse * np.sin(theta + zn.imag)

    # Watercolor effect
    points = np.array([x, y]).T
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(0, len(segments))
    lc = mcolors.LinearSegmentedColormap.from_list("", [color1, color2])
    line = plt.LineCollection(segments, cmap=lc, norm=norm, linewidth=thickness, alpha=0.8)
    line.set_array(np.linspace(0, 1, len(segments)))
    ax.add_collection(line)

    # Glowing trail
    trail = Ellipse((x[-1], y[-1]), 10*pulse, 5*pulse, angle=np.degrees(zn.imag), color='white', alpha=0.2)
    ax.add_patch(trail)

    # Phase evolution circle
    circle = Circle((0, 0), 50, fill=False, color='yellow', alpha=0.5, linestyle='--')
    ax.add_patch(circle)

    # Energy field (LDNA_1 effect)
    an = (1/phi) * (1/3) * 3e8  # Magnitude A_n
    field = Circle((0, 0), an/1e8, color='purple', alpha=0.3)
    ax.add_patch(field)

    # Axes and labels
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_xlabel("Real", fontsize=10)
    ax.set_ylabel("Imaginary", fontsize=10)
    ax.set_title(f"Quantum Spiral | Freq = {freq:.2f} Hz", fontsize=12, pad=20)

# GUI for customization
class LovinceGUI:
    def __init__(self, root, update_params):
        self.root = root
        self.update_params = update_params
        self.root.title("Lovince AI Customization")

        # Frequency settings
        ttk.Label(root, text="Base Frequency (Hz):").grid(row=0, column=0, padx=5, pady=5)
        self.base_freq = ttk.Entry(root)
        self.base_freq.insert(0, "39.96")
        self.base_freq.grid(row=0, column=1)

        ttk.Label(root, text="Wave Type:").grid(row=1, column=0, padx=5, pady=5)
        self.wave_type = ttk.Combobox(root, values=['sine', 'square', 'sawtooth'])
        self.wave_type.set('sine')
        self.wave_type.grid(row=1, column=1)

        # Spiral settings
        ttk.Label(root, text="Spiral Color 1:").grid(row=2, column=0, padx=5, pady=5)
        self.color1 = ttk.Entry(root)
        self.color1.insert(0, "teal")
        self.color1.grid(row=2, column=1)

        ttk.Label(root, text="Spiral Thickness:").grid(row=3, column=0, padx=5, pady=5)
        self.thickness = ttk.Entry(root)
        self.thickness.insert(0, "3")
        self.thickness.grid(row=3, column=1)

        # Nebula settings
        ttk.Label(root, text="Nebula Colormap:").grid(row=4, column=0, padx=5, pady=5)
        self.colormap = ttk.Entry(root)
        self.colormap.insert(0, "PuBu_r")
        self.colormap.grid(row=4, column=1)

        ttk.Button(root, text="Update", command=self.update).grid(row=5, column=0, columnspan=2, pady=10)

    def update(self):
        params = {
            "base_freq": float(self.base_freq.get()),
            "wave_type": self.wave_type.get(),
            "color1": self.color1.get(),
            "thickness": float(self.thickness.get()),
            "colormap": self.colormap.get()
        }
        self.update_params(params)

# Main infinite loop
def main():
    print("Lovince AI: Quantum Spiral Engine (Enhanced) is running...")
    print("Press Ctrl+C to exit.")

    memory = LovinceMemory()
    params = {
        "base_freq": 39.96,
        "wave_type": "sine",
        "color1": "teal",
        "color2": "gold",
        "thickness": 3,
        "decay_rate": 10,
        "colormap": "PuBu_r",
        "star_density": 50,
        "opacity": 0.6
    }

    # GUI setup
    root = tk.Tk()
    def update_params(new_params):
        params.update(new_params)
    gui = LovinceGUI(root, update_params)

    # Plot setup
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    frame = 0

    n = 0
    try:
        while True:
            # Update quantum state
            zn, yn, freq = calculate_quantum_state(n, memory, params["base_freq"])
            if zn is None:
                logging.error(f"Skipping iteration at n={n}")
                n += 1
                continue

            # Update memory
            memory.update(n, zn, yn, freq)

            # Generate sound
            generate_sound(n, freq, params["wave_type"])

            # Plot spiral
            ax.clear()
            create_nebula_background(ax, params["colormap"], params["star_density"], params["opacity"])
            plot_spiral(ax, zn, freq, frame, params["color1"], params["color2"], params["thickness"], params["decay_rate"])
            plt.draw()
            plt.pause(0.1)

            # Update GUI
            root.update()

            # Print memory analytics
            if n % 10 == 0:
                print(memory.analytics())

            n += 1
            frame += 1
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\nLovince AI: Program terminated by user.")
        plt.close()
        root.destroy()
        sys.exit(0)

if __name__ == "__main__":
    main()