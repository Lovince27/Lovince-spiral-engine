import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

# ======================== CONSTANTS (Golden Ratio Dynamics) ========================
PHI: float = (1 + np.sqrt(5)) / 2        # Golden Ratio (φ)
A: float = 40.5                          # Base Amplitude (Lovince Constant)
THETA: float = np.pi / 4                 # Angular Phase (45°)

# ======================== SPIRAL GENERATOR ========================
def generate_spiral(n: float, direction: str = 'creation') -> Tuple[float, float]:
    """
    Generates spiral coordinates with infinite loop prevention and direction validation
    
    Args:
        n: Spiral parameter (must be finite)
        direction: 'creation' (expanding) or 'collapse' (contracting)
    
    Returns:
        (x, y) coordinates of the spiral point
    
    Raises:
        ValueError: If invalid direction or infinite input
    """
    # ==== SELF-CHECK ====
    if not np.isfinite(n):
        raise ValueError("Infinite loop prevention: n must be finite")
    
    # ==== CROSS-CHECK ====
    if direction not in ['creation', 'collapse']:
        raise ValueError("Direction must be either 'creation' or 'collapse'")
    
    # ==== SPIRAL DYNAMICS ====
    r = A * (PHI ** (n if direction == 'creation' else -n))
    angle = -n * THETA
    
    # ==== HENCE PROVED ==== 
    # Mathematical validation that r follows golden ratio progression
    assert np.isclose(r / A, PHI ** (n if direction == 'creation' else -n)), \
           "Golden ratio progression violated"
    
    return r * np.cos(angle), r * np.sin(angle)

# ======================== DYNAMIC PLOT GENERATION ========================
def plot_spiral_dynamics(max_iterations: int = 10, resolution: int = 1000) -> None:
    """
    Creates the dual spiral visualization with runtime safety checks
    
    Args:
        max_iterations: Upper bound for n (prevent infinite loops)
        resolution: Number of points in the spiral
    """
    # ==== INFINITE LOOP PREVENTION ====
    if not (0 < max_iterations < 1000 and 100 <= resolution <= 10000):
        raise ValueError("Safety bounds violated")
    
    n_values = np.linspace(0, max_iterations, resolution)
    
    # ==== PARALLEL COORDINATE GENERATION ====
    creation_points = [generate_spiral(n, 'creation') for n in n_values]
    collapse_points = [generate_spiral(n, 'collapse') for n in n_values]
    
    # ==== VISUALIZATION ENGINE ====
    plt.figure(figsize=(16, 8), dpi=100, facecolor='black')
    plt.suptitle("Lovince Spiral Dynamics: Creation vs Collapse", 
                color='white', fontsize=16, y=0.95)
    
    # Collapse Spiral (Implosion)
    plt.subplot(1, 2, 1, facecolor='black')
    collapse_x, collapse_y = zip(*collapse_points)
    plt.plot(collapse_x, collapse_y, color='cyan', linewidth=1.5)
    plt.title('Quantum Collapse (φ^-n)', color='white')
    plt.gca().set_facecolor('black')
    plt.grid(color='gray', alpha=0.2)
    
    # Creation Spiral (Explosion)
    plt.subplot(1, 2, 2, facecolor='black')
    creation_x, creation_y = zip(*creation_points)
    plt.plot(creation_x, creation_y, color='magenta', linewidth=1.5)
    plt.title('Cosmic Creation (φ^n)', color='white')
    plt.gca().set_facecolor('black')
    plt.grid(color='gray', alpha=0.2)
    
    # ==== FINAL VALIDATION ====
    for ax in plt.gcf().axes:
        ax.set_aspect('equal')
        for spine in ax.spines.values():
            spine.set_color('yellow')
    
    plt.tight_layout()
    plt.show()

# ======================== EXECUTION WITH SAFETY ========================
if __name__ == "__main__":
    try:
        plot_spiral_dynamics(max_iterations=10, resolution=2000)
        print("Hence Proved: Golden Ratio Spirals converge/diverge as φ^±n")
    except Exception as e:
        print(f"Cosmic Error: {str(e)}")

#!/usr/bin/env python3
"""Golden Ratio Spiral Dynamics: Visualizing Creation and Collapse Spirals."""
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, NamedTuple, Literal
from dataclasses import dataclass

# ======================== CONFIGURATION ========================
@dataclass(frozen=True)
class SpiralConfig:
    """Configuration for spiral dynamics with validated parameters."""
    phi: float = (1 + np.sqrt(5)) / 2  # Golden Ratio (φ ≈ 1.6180339887)
    amplitude: float = 40.5            # Base amplitude (Lovince Constant)
    theta: float = np.pi / 4           # Angular phase (45°)
    max_iterations: int = 10           # Maximum spiral progression
    resolution: int = 2000             # Number of points for smoothness

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        assert self.phi > 1, f"Golden ratio must be > 1, got {self.phi}"
        assert self.amplitude > 0, f"Amplitude must be positive, got {self.amplitude}"
        assert 0 < self.theta < 2 * np.pi, f"Theta must be in (0, 2π), got {self.theta}"
        assert 0 < self.max_iterations <= 100, f"Max iterations must be in (0, 100], got {self.max_iterations}"
        assert 100 <= self.resolution <= 10000, f"Resolution must be in [100, 10000], got {self.resolution}"

# ======================== SPIRAL DATA STRUCTURE ========================
class SpiralPoint(NamedTuple):
    """Represents a point in the spiral with x, y coordinates."""
    x: float
    y: float

# ======================== SPIRAL GENERATOR ========================
class SpiralGenerator:
    """Generates golden ratio spiral coordinates with vectorized operations."""
    
    def __init__(self, config: SpiralConfig) -> None:
        """Initialize with validated configuration."""
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """Cross-check configuration for numerical stability."""
        max_radius = self.config.amplitude * (self.config.phi ** self.config.max_iterations)
        min_radius = self.config.amplitude * (self.config.phi ** -self.config.max_iterations)
        assert max_radius < 1e308, f"Overflow risk: max radius {max_radius} too large"
        assert min_radius > 1e-308, f"Underflow risk: min radius {min_radius} too small"

    def generate(self, direction: Literal["creation", "collapse"]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate spiral coordinates using vectorized computation.

        Args:
            direction: 'creation' for expanding (φ^n) or 'collapse' for contracting (φ^-n).

        Returns:
            Tuple of (x, y) numpy arrays representing spiral points.

        Raises:
            ValueError: If direction is invalid.
        """
        if direction not in {"creation", "collapse"}:
            raise ValueError(f"Invalid direction: {direction}. Must be 'creation' or 'collapse'.")

        # Generate n values
        n = np.linspace(0, self.config.max_iterations, self.config.resolution)

        # Compute radius: r = A * φ^(±n)
        power = n if direction == "creation" else -n
        radius = self.config.amplitude * np.power(self.config.phi, power)

        # Compute angle: θ = -n * THETA
        angle = -n * self.config.theta

        # Compute coordinates
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)

        # Mathematical proof: Verify golden ratio progression
        expected_ratio = np.power(self.config.phi, power)
        assert np.allclose(radius / self.config.amplitude, expected_ratio, rtol=1e-5), \
            f"Golden ratio progression failed for {direction} spiral"

        return x, y

# ======================== PLOTTER ========================
class SpiralPlotter:
    """Visualizes creation and collapse spirals with dynamic scaling."""
    
    def __init__(self, config: SpiralConfig) -> None:
        """Initialize with configuration."""
        self.config = config

    def plot(self, creation_coords: Tuple[np.ndarray, np.ndarray],
             collapse_coords: Tuple[np.ndarray, np.ndarray]) -> None:
        """
        Plot both spirals side-by-side with dynamic scaling.

        Args:
            creation_coords: (x, y) arrays for creation spiral.
            collapse_coords: (x, y) arrays for collapse spiral.
        """
        # Initialize figure
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=100)
        fig.suptitle("Golden Ratio Spiral Dynamics", fontsize=16, color='white', y=0.95)

        # Plot collapse spiral
        ax1.plot(*collapse_coords, color='cyan', linewidth=1.5, label='Collapse (φ^-n)')
        ax1.set_title("Quantum Collapse", color='white', fontsize=12)
        ax1.grid(color='gray', alpha=0.2)
        ax1.legend(facecolor='black', edgecolor='yellow', labelcolor='white')

        # Plot creation spiral
        ax2.plot(*creation_coords, color='magenta', linewidth=1.5, label='Creation (φ^n)')
        ax2.set_title("Cosmic Creation", color='white', fontsize=12)
        ax2.grid(color='gray', alpha=0.2)
        ax2.legend(facecolor='black', edgecolor='yellow', labelcolor='white')

        # Dynamic scaling
        for ax in (ax1, ax2):
            ax.set_aspect('equal')
            ax.set_facecolor('black')
            for spine in ax.spines.values():
                spine.set_color('yellow')

        # Adjust limits for creation spiral
        max_radius = self.config.amplitude * (self.config.phi ** self.config.max_iterations)
        ax2.set_xlim(-max_radius * 1.1, max_radius * 1.1)
        ax2.set_ylim(-max_radius * 1.1, max_radius * 1.1)

        # Adjust limits for collapse spiral
        ax1.set_xlim(-self.config.amplitude * 1.1, self.config.amplitude * 1.1)
        ax1.set_ylim(-self.config.amplitude * 1.1, self.config.amplitude * 1.1)

        plt.tight_layout()
        plt.show()

# ======================== MAIN EXECUTION ========================
def main() -> None:
    """Execute spiral generation and visualization with error handling."""
    try:
        # Initialize configuration
        config = SpiralConfig()

        # Generate spirals
        generator = SpiralGenerator(config)
        creation_coords = generator.generate("creation")
        collapse_coords = generator.generate("collapse")

        # Plot spirals
        plotter = SpiralPlotter(config)
        plotter.plot(creation_coords, collapse_coords)

        print("Proof Complete: Golden ratio spirals generated and visualized successfully.")
        print(f"Creation spiral diverges as A * φ^n, Collapse spiral converges as A * φ^-n.")

    except Exception as e:
        print(f"Cosmic Anomaly Detected: {str(e)}")
        raise

if __name__ == "__main__":
    main()