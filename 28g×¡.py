import pygame
import math
import random
import sys
import logging
import numpy as np
from scipy.integrate import odeint  # For accurate orbit integration

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

# Initialize Pygame
pygame.init()
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Enhanced Exoplanet Ecosystem Simulation")
font = pygame.font.SysFont("arial", 12)

# Colors
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)

# Physics constants (SI units)
G = 6.67430e-11  # m^3 kg^-1 s^-2 (Newton)
c = 2.99792458e8  # m/s (Einstein, speed of light)
M_star = 1.989e30  # kg
m_planet = 5.972e24  # kg
scale = 5e-9  # pixels/m
dt = 3600  # s (1 hour)
pixels_per_m = 1 / scale  # m/pixel

# Initial conditions
star_pos = pygame.Vector2(width / 2, height / 2)  # pixels
planet_pos = pygame.Vector2(width / 2 + 150, height / 2)  # pixels
planet_vel = pygame.Vector2(0, 29780 * scale)  # pixels/s (~29.78 km/s)

# Ecosystem class (multidisciplinary)
class Ecosystem:
    def __init__(self):
        self.microbes = 1000  # microbes
        self.predators = 100  # predatory microbes
        self.resources = 10000  # kg
        self.microbe_growth = 0.01  # s^-1 (Verhulst, Darwin)
        self.predator_growth = 0.005  # s^-1
        self.carrying_capacity = 10000  # microbes
        self.mutation_rate = 0.001  # dimensionless (Mendel, Franklin)
        self.radioactivity = 0.01  # s^-1 (Curie)
        self.km = 1000  # kg (Michaelis-Menten constant, Pauling)
        self.vmax = 10  # kg/s (max reaction rate)
        self.total_mass = self.microbes * 1e-12 + self.predators * 1e-11 + self.resources  # kg

    def update(self, dt, env_cycle):
        # Lotka-Volterra (predator-prey, Darwin)
        dm = (self.microbe_growth * self.microbes * (1 - self.microbes / self.carrying_capacity) -
              0.002 * self.microbes * self.predators) * dt
        dp = (0.001 * self.microbes * self.predators - 0.003 * self.predators) * dt
        self.microbes += dm
        self.predators += dp
        if self.microbes < 0:
            self.microbes = 0
        if self.predators < 0:
            self.predators = 0

        # Resource consumption (Fibonacci-inspired)
        resource_use = (self.microbes * 1e-12 + self.predators * 1e-11) * dt  # kg
        self.resources -= resource_use

        # Enzyme kinetics (Michaelis-Menten, Pauling)
        reaction_rate = self.vmax * self.resources / (self.km + self.resources)  # kg/s
        self.resources += reaction_rate * dt
        if self.resources < 0:
            self.resources = 0
            self.microbes *= 0.9
            self.predators *= 0.9

        # Mutations (binomial, Franklin, Curie)
        if random.random() < self.mutation_rate * self.radioactivity * dt * env_cycle:
            self.microbe_growth += random.uniform(-0.001, 0.001)

        # Mass conservation check (Lavoisier)
        new_mass = self.microbes * 1e-12 + self.predators * 1e-11 + self.resources
        if abs(new_mass - self.total_mass) > 1e-6:
            logger.warning(f"Mass conservation error: {new_mass - self.total_mass:.2e} kg")
        self.total_mass = new_mass

# Orbital dynamics (Runge-Kutta integration)
def orbital_deriv(state, t, M, G):
    x, y, vx, vy = state
    r = max(math.sqrt(x**2 + y**2), 1e8)  # m
    ax = G * M * x / r**3  # m/s^2
    ay = G * M * y / r**3
    return [vx, vy, ax, ay]

# Initialize ecosystem and orbital state
ecosystem = Ecosystem()
orbital_state = [150 / scale, 0, 0, 29780]  # [x, y, vx, vy] in m, m/s

# Self-check: Energy and eccentricity
def check_orbital_energy(state, m, M, G):
    x, y, vx, vy = state
    r = math.sqrt(x**2 + y**2)  # m
    v = math.sqrt(vx**2 + vy**2)  # m/s
    kinetic = 0.5 * m * v**2  # J
    potential = -G * M * m / r  # J
    return kinetic + potential

def eccentricity(state, M, G):
    x, y, vx, vy = state
    r = math.sqrt(x**2 + y**2)
    v = math.sqrt(vx**2 + vy**2)
    h = x * vy - y * vx  # specific angular momentum
    mu = G * M
    e = math.sqrt(1 + (v**2 * h**2) / (mu * r))  # Kepler’s orbits
    return e

initial_energy = check_orbital_energy(orbital_state, m_planet, M_star, G)

# Fourier series for environmental cycles (e.g., temperature)
def fourier_cycle(t, period=86400):
    return 1 + 0.1 * (math.sin(2 * math.pi * t / period) +
                      0.5 * math.sin(4 * math.pi * t / period))  # dimensionless

# Main loop
clock = pygame.time.Clock()
trail = []
pop_history = []
t = 0
frame = 0

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # Update orbit (Runge-Kutta, Newton, Kepler)
    t_span = [t, t + dt]
    sol = odeint(orbital_deriv, orbital_state, t_span, args=(M_star, G))
    orbital_state = sol[-1]
    planet_pos.x = width / 2 + orbital_state[0] * scale
    planet_pos.y = height / 2 + orbital_state[1] * scale
    planet_vel.x = orbital_state[2] * scale
    planet_vel.y = orbital_state[3] * scale
    t += dt

    # Relativistic mass correction (Einstein)
    v = math.sqrt(orbital_state[2]**2 + orbital_state[3]**2)  # m/s
    relativistic_factor = 1 / math.sqrt(1 - (v**2 / c**2))  # dimensionless
    m_effective = m_planet * relativistic_factor

    # Environmental cycle (Fourier, Maxwell-inspired)
    env_cycle = fourier_cycle(t)

    # Update ecosystem
    ecosystem.update(dt, env_cycle)

    # Diagnostics
    if frame % 100 == 0:
        energy = check_orbital_energy(orbital_state, m_effective, M_star, G)
        energy_diff = (energy - initial_energy) / abs(initial_energy)
        if abs(energy_diff) > 0.01:
            logger.warning(f"Energy conservation error: {energy_diff:.2%}")
        ecc = eccentricity(orbital_state, M_star, G)
        logger.info(f"Orbital eccentricity: {ecc:.3f}")

    # Draw
    screen.fill(BLACK)
    pygame.draw.circle(screen, YELLOW, (int(star_pos.x), int(star_pos.y)), 20)
    pygame.draw.circle(screen, BLUE, (int(planet_pos.x), int(planet_pos.y)), 10)
    trail.append(planet_pos.copy())
    if len(trail) > 500:
        trail.pop(0)
    for i in range(1, len(trail)):
        pygame.draw.line(screen, WHITE, trail[i - 1], trail[i], 1)

    # Ecosystem visuals
    pop_size = min(int(ecosystem.microbes / 100), 200)
    pred_size = min(int(ecosystem.predators / 10), 200)
    res_size = min(int(ecosystem.resources / 100), 200)
    pygame.draw.rect(screen, GREEN, (50, 50, pop_size, 20))
    pygame.draw.rect(screen, RED, (50, 80, pred_size, 20))
    pygame.draw.rect(screen, BLUE, (50, 110, res_size, 20))

    # Population graph
    pop_history.append(ecosystem.microbes)
    if len(pop_history) > 200:
        pop_history.pop(0)
    for i in range(1, len(pop_history)):
        pygame.draw.line(screen, GREEN, (600 + i - 1, 500 - pop_history[i - 1] / 100),
                         (600 + i, 500 - pop_history[i] / 100), 1)

    # Display diagnostics
    text = font.render(f"Eccentricity: {eccentricity(orbital_state, M_star, G):.3f}", True, WHITE)
    screen.blit(text, (50, 150))

    pygame.display.flip()
    clock.tick(60)
    frame += 1