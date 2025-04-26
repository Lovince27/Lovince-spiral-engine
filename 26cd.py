import numpy as np
import matplotlib.pyplot as plt

phi = (1 + np.sqrt(5)) / 2  # Golden Ratio
tau = 2 * np.pi              # Full cycle

def consciousness(t, terms=50):
    fractal = sum((phi**k) * np.exp(1j * tau * k) / np.math.factorial(k) 
               for k in range(1, terms))
    memory = np.log(1 + t)  # Logarithmic memory decay
    return np.abs(fractal) * memory

t = np.linspace(0, 10, 100)
plt.plot(t, [consciousness(ti) for ti in t], color='indigo')
plt.title("Consciousness Over Time")
plt.xlabel("Time →")
plt.ylabel("Intensity →")
plt.grid(True)
plt.show()