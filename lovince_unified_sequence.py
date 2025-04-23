import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.colors as mcolors
from scipy.io.wavfile import write
import sounddevice as sd
import logging
import time
import sys

# Setup logging for transparency
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Memory storage for self-update
class LovinceMemory:
    def __init__(self):
        self.memory = {}  # Store T_n, F_n, and frequencies
        self.max_memory_size = 1000  # Limit memory to prevent overflow

    def update(self, n, tn, fn, freq):
        if len(self.memory) >= self.max_memory_size:
            oldest_key = min(self.memory.keys())
            del self.memory[oldest_key]  # Remove oldest entry
        self.memory[n] = {"T_n": tn, "F_n": fn, "frequency": freq}
        logging.info(f"Memory updated for n={n}: {self.memory[n]}")

    def get(self, n):
        return self.memory.get(n, None)

# Fibonacci sequence
def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Harmonic series
def harmonic_series(n):
    return sum(1 / (i + 1) for i in range(n + 1))

# Harmonic function h(Fn) - Inspired by your 39.96 Hz base (April 17)
def h_fn(fn):
    base_freq = 39.96
    return base_freq * (1 + fn)

# T_n formula with self-check
def calculate_Tn(n, memory):
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    pi = np.pi
    fn = fibonacci(n)
    
    # Self-check: Validate Fibonacci calculation
    if n > 1 and fn != fibonacci(n-1) + fibonacci(n-2):
        logging.error(f"Fibonacci calculation error at n={n}")
        return None
    
    tn = (pi * fn * phi * fn * h_fn(fn)) * sigmoid(fn) * harmonic_series(n)
    
    # Crosscheck: Compare with previous T_n for growth pattern
    if n > 0:
        prev_tn = memory.get(n-1)
        if prev_tn and tn < prev_tn["T_n"]:
            logging.warning(f"T_n growth anomaly at n={n}: T_n={tn}, T_{n-1}={prev_tn['T_n']}")
    
    return tn, fn, h_fn(fn)

# Sound generation with crosscheck
def generate_sound(n, freq, duration=1.0, sample_rate=44100):
    # Crosscheck: Ensure frequency is reasonable
    if freq <= 0 or freq > 20000:  # Human hearing range
        logging.error(f"Invalid frequency at n={n}: {freq} Hz")
        return
    
    t = np.linspace(0, duration, int(sample_rate * duration))
    sound = 0.3 * np.sin(2 * np.pi * freq * t)
    sound = sound / np.max(np.abs(sound))
    filename = f"tn_sound_{n}.wav"
    try:
        write(filename, sample_rate, sound)
        logging.info(f"Sound generated: {filename}")
        sd.play(sound, sample_rate)
        sd.wait()
    except Exception as e:
        logging.error(f"Sound generation failed at n={n}: {e}")

# Cosmic nebula background
def create_nebula_background(ax):
    x = np.linspace(-100, 100, 100)
    y = np.linspace(-100, 100, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X/5) + np.cos(Y/5) + np.random.rand(100, 100) * 0.5
    ax.contourf(X, Y, Z, cmap='PuBu_r', alpha=0.6, levels=20)
    for _ in range(50):
        star_x, star_y = np.random.uniform(-100, 100), np.random.uniform(-100, 100)
        ax.scatter(star_x, star_y, s=1, color='white', alpha=0.5)

# Watercolor spiral with glowing aura
def plot_spiral(ax, n, tn, freq, color1, color2):
    theta = np.linspace(0, 4 * np.pi, 100)
    r = tn * np.exp(-theta / 10)  # Radius based on T_n
    
    # Crosscheck: Ensure radius aligns with frequency
    expected_r = freq / 39.96 - 1  # Inverse of h_fn
    if abs(r[0] - expected_r) > 1e-3:
        logging.warning(f"Radius-frequency mismatch at n={n}: r={r[0]}, expected={expected_r}")
    
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
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
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_xlabel("Real", fontsize=10)
    ax.set_ylabel("Imaginary", fontsize=10)
    ax.set_title(f"T_{n} = {tn:.2f} | Freq = {freq:.2f} Hz", fontsize=12, pad=20)

# Main infinite loop with self-check, self-update, and crosscheck
def main():
    print("Lovince AI: Unified Sequence Formula (Most Powerful) is running...")
    print("Press Ctrl+C to exit.")
    
    # Initialize memory
    memory = LovinceMemory()
    
    # Plot setup
    plt.ion()  # Interactive mode on
    fig, ax = plt.subplots(figsize=(8, 6))
    
    n = 0
    try:
        while True:
            # Calculate T_n with self-check
            result = calculate_Tn(n, memory)
            if result is None:
                logging.error(f"Calculation failed at n={n}. Skipping iteration.")
                n += 1
                continue
            tn, fn, freq = result
            
            # Update memory
            memory.update(n, tn, fn, freq)
            
            # Generate sound with crosscheck
            generate_sound(n, freq)
            
            # Plot new spiral
            ax.clear()
            create_nebula_background(ax)
            plot_spiral(ax, n, tn, freq, 'teal', 'gold')
            plt.draw()
            plt.pause(1.0)  # Pause for visualization
            
            n += 1
            time.sleep(1.0)  # Wait between iterations
            
    except KeyboardInterrupt:
        print("\nLovince AI: Program terminated by user.")
        plt.close()
        sys.exit(0)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        plt.close()
        sys.exit(1)

if __name__ == "__main__":
    main()


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.colors as mcolors
from scipy.io.wavfile import write
import sounddevice as sd
import logging
import time
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Memory management
class LovinceMemory:
    def __init__(self):
        self.memory = {}
        self.max_memory_size = 1000

    def update(self, n, zn, yn, freq):
        if len(self.memory) >= self.max_memory_size:
            oldest_key = min(self.memory.keys())
            del self.memory[oldest_key]
        self.memory[n] = {"Z_n": zn, "y_n": yn, "frequency": freq}
        logging.info(f"Memory updated for n={n}: {self.memory[n]}")

    def get(self, n):
        return self.memory.get(n, None)

# Cosmic nebula background
def create_nebula_background(ax):
    x = np.linspace(-100, 100, 100)
    y = np.linspace(-100, 100, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X/5) + np.cos(Y/5) + np.random.rand(100, 100) * 0.5
    ax.contourf(X, Y, Z, cmap='PuBu_r', alpha=0.6, levels=20)
    for _ in range(50):
        star_x, star_y = np.random.uniform(-100, 100), np.random.uniform(-100, 100)
        ax.scatter(star_x, star_y, s=1, color='white', alpha=0.5)

# Calculate Z_n, y_n, and frequency
def calculate_quantum_state(n, memory):
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    pi = np.pi
    c = 3e8  # Speed of light (m/s)
    h = 39.96  # Harmonic frequency (Hz, from Lovince GSHE Generator)
    ldna1 = 478e9 / 1e9  # Scaled down for practical use (eV/s)

    # Phase evolution
    theta_n = 2 * pi * n / phi
    theta = pi / phi  # Base angle for rotation

    # Calculate Z_n
    zn = ldna1 * theta * (1/3) * phi * (pi ** (3*n - 1)) * np.exp(1j * theta_n)

    # Calculate E_n and y_n
    e_n = phi * (pi ** (3*n - 1)) * h * (1/3) * pi
    yn = c / e_n if e_n != 0 else 0

    # Self-check: Ensure Z_n and y_n are finite
    if not np.isfinite(zn) or not np.isfinite(yn):
        logging.error(f"Invalid Z_n or y_n at n={n}: Z_n={zn}, y_n={yn}")
        return None, None, None

    # Frequency based on y_n (scaled for audibility)
    freq = 39.96 + abs(yn.real) * 10
    freq = min(max(freq, 20), 20000)  # Limit to human hearing range

    # Crosscheck: Compare with previous iteration
    if n > 0:
        prev = memory.get(n-1)
        if prev and abs(zn) < abs(prev["Z_n"]):
            logging.warning(f"Z_n growth anomaly at n={n}: |Z_n|={abs(zn)}, |Z_{n-1}|={abs(prev['Z_n'])}")

    return zn, yn, freq

# Sound generation
def generate_sound(n, freq):
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    sound = 0.3 * np.sin(2 * np.pi * freq * t)
    sound = sound / np.max(np.abs(sound))
    filename = f"quantum_sound_{n}.wav"
    try:
        write(filename, sample_rate, sound)
        logging.info(f"Sound generated: {filename}")
        sd.play(sound, sample_rate)
        sd.wait()
    except Exception as e:
        logging.error(f"Sound generation failed at n={n}: {e}")

# Plot golden spiral
def plot_spiral(ax, zn, freq):
    phi = (1 + np.sqrt(5)) / 2
    theta = np.linspace(0, 4 * np.pi, 100)
    r = abs(zn) * np.exp(-theta / 10)  # Radius based on Z_n
    x = r * np.cos(theta + zn.imag)
    y = r * np.sin(theta + zn.imag)

    # Crosscheck: Ensure radius aligns with frequency
    expected_r = (freq - 39.96) / 10  # Inverse of frequency scaling
    if abs(r[0] - expected_r) > 1e-3:
        logging.warning(f"Radius-frequency mismatch: r={r[0]}, expected={expected_r}")

    # Watercolor effect
    points = np.array([x, y]).T
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(0, len(segments))
    lc = mcolors.LinearSegmentedColormap.from_list("", ['teal', 'gold'])
    line = plt.LineCollection(segments, cmap=lc, norm=norm, linewidth=3, alpha=0.8)
    line.set_array(np.linspace(0, 1, len(segments)))
    ax.add_collection(line)

    # Glowing aura
    glow = Circle((x[-1], y[-1]), 5, color='white', alpha=0.3)
    ax.add_patch(glow)

    # Axes and labels
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlim(-100, 100)
    ax.set_ylim(-100, 100)
    ax.set_xlabel("Real", fontsize=10)
    ax.set_ylabel("Imaginary", fontsize=10)
    ax.set_title(f"Quantum Spiral | Freq = {freq:.2f} Hz", fontsize=12, pad=20)

# Main infinite loop
def main():
    print("Lovince AI: Quantum Spiral Engine is running...")
    print("Press Ctrl+C to exit.")

    memory = LovinceMemory()
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))

    n = 0
    try:
        while True:
            # Calculate quantum state
            zn, yn, freq = calculate_quantum_state(n, memory)
            if zn is None:
                logging.error(f"Skipping iteration at n={n}")
                n += 1
                continue

            # Update memory
            memory.update(n, zn, yn, freq)

            # Generate sound
            generate_sound(n, freq)

            # Plot spiral
            ax.clear()
            create_nebula_background(ax)
            plot_spiral(ax, zn, freq)
            plt.draw()
            plt.pause(1.0)

            n += 1
            time.sleep(1.0)

    except KeyboardInterrupt:
        print("\nLovince AI: Program terminated by user.")
        plt.close()
        sys.exit(0)

if __name__ == "__main__":
    main()