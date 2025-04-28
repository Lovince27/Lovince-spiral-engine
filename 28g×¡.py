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


import pygame
import math
import random
import sys
import logging
import numpy as np
from scipy.integrate import odeint
import pandas as pd
import json
from datetime import datetime

# Setup logging for diagnostics
logging.basicConfig(level=logging.INFO, filename='exoplanet_simulation.log', format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Initialize Pygame
pygame.init()
width, height = 1200, 800
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Final Exoplanet Ecosystem Simulation")
font = pygame.font.SysFont("arial", 14, bold=True)

# Colors
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
PURPLE = (128, 0, 128)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)

# Physics constants (SI units)
G = 6.67430e-11  # m^3 kg^-1 s^-2 (Newton)
c = 2.99792458e8  # m/s (Einstein)
M_star = 1.989e30  # kg (Sun-like)
scale = 5e-9  # pixels/m
dt = 3600  # s (1 hour)
pixels_per_m = 1 / scale

# Planet class
class Planet:
    def __init__(self, pos, vel, mass, color, name):
        self.pos = pygame.Vector2(pos)  # pixels
        self.vel = pygame.Vector2(vel)  # pixels/s
        self.mass = mass  # kg
        self.color = color
        self.name = name
        self.trail = []
        self.state = [pos[0] / scale, pos[1] / scale, vel[0] / scale, vel[1] / scale]  # [x, y, vx, vy] in m, m/s

# Ecosystem class
class Ecosystem:
    def __init__(self, planet_name):
        self.microbes = 1000  # microbes
        self.predators = 100  # predatory microbes
        self.resources = 10000  # kg
        self.microbe_growth = 0.01  # s^-1
        self.predator_growth = 0.005  # s^-1
        self.carrying_capacity = 10000
        self.mutation_rate = 0.001  # dimensionless
        self.radioactivity = 0.01  # s^-1
        self.km = 1000  # kg (Michaelis-Menten)
        self.vmax = 10  # kg/s
        self.total_mass = self.microbes * 1e-12 + self.predators * 1e-11 + self.resources
        self.data = []
        self.planet_name = planet_name

    def update(self, dt, env_cycle):
        # Lotka-Volterra (Darwin, Lotka)
        dm = (self.microbe_growth * self.microbes * (1 - self.microbes / self.carrying_capacity) -
              0.002 * self.microbes * self.predators) * dt
        dp = (0.001 * self.microbes * self.predators - 0.003 * self.predators) * dt
        self.microbes = max(0, self.microbes + dm)
        self.predators = max(0, self.predators + dp)

        # Resource dynamics (Pauling, Lavoisier)
        resource_use = (self.microbes * 1e-12 + self.predators * 1e-11) * dt
        self.resources -= resource_use
        reaction_rate = self.vmax * self.resources / (self.km + self.resources)
        self.resources += reaction_rate * dt
        self.resources = max(0, self.resources)

        # Mutations (Mendel, Franklin, Curie)
        if random.random() < self.mutation_rate * self.radioactivity * dt * env_cycle:
            self.microbe_growth += random.uniform(-0.001, 0.001)

        # Mass conservation check
        new_mass = self.microbes * 1e-12 + self.predators * 1e-11 + self.resources
        if abs(new_mass - self.total_mass) > 1e-6:
            logger.warning(f"{self.planet_name} mass conservation error: {new_mass - self.total_mass:.2e} kg")
        self.total_mass = new_mass

        # Log data
        self.data.append({
            'time_days': t / 86400,
            'microbes': self.microbes,
            'predators': self.predators,
            'resources': self.resources
        })

# Orbital dynamics
def orbital_deriv(state, t, M, masses):
    n = len(state) // 4
    derivs = []
    for i in range(n):
        x, y = state[i*4], state[i*4+1]
        vx, vy = state[i*4+2], state[i*4+3]
        ax, ay = 0, 0
        for j in range(n):
            if i != j:
                dx = state[j*4] - x
                dy = state[j*4+1] - y
                r = max(math.sqrt(dx**2 + dy**2), 1e8)
                ax += G * masses[j] * dx / r**3
                ay += G * masses[j] * dy / r**3
        r_star = max(math.sqrt(x**2 + y**2), 1e8)
        ax += G * M * x / r_star**3
        ay += G * M * y / r_star**3
        derivs.extend([vx, vy, ax, ay])
    return derivs

# Initialize planets and ecosystems
planets = [
    Planet((width / 2 + 150, height / 2), (0, 29780 * scale), 5.972e24, BLUE, "ExoEarth"),
    Planet((width / 2 + 200, height / 2), (0, 24000 * scale), 3.285e23, PURPLE, "ExoMars")
]
ecosystems = [Ecosystem(p.name) for p in planets]
star_pos = pygame.Vector2(width / 2, height / 2)
orbital_state = [s for p in planets for s in p.state]
masses = [p.mass for p in planets]

# Diagnostics
def check_energy(state, masses, M):
    energy = 0
    n = len(state) // 4
    for i in range(n):
        x, y, vx, vy = state[i*4:i*4+4]
        r = math.sqrt(x**2 + y**2)
        v = math.sqrt(vx**2 + vy**2)
        kinetic = 0.5 * masses[i] * v**2
        potential = -G * M * masses[i] / r
        for j in range(i + 1, n):
            dx = x - state[j*4]
            dy = y - state[j*4+1]
            r_ij = max(math.sqrt(dx**2 + dy**2), 1e8)
            potential -= G * masses[i] * masses[j] / r_ij
        energy += kinetic + potential
    return energy

def eccentricity(state, mass, M):
    x, y, vx, vy = state
    r = math.sqrt(x**2 + y**2)
    v = math.sqrt(vx**2 + vy**2)
    h = x * vy - y * vx
    mu = G * (M + mass)
    e = math.sqrt(max(0, 1 + (v**2 * h**2) / (mu * r) - 2 * h**2 / (mu * r**2)))
    return e

initial_energy = check_energy(orbital_state, masses, M_star)

# Fourier cycle
def fourier_cycle(t, period=86400):
    return 1 + 0.1 * (math.sin(2 * math.pi * t / period) +
                      0.5 * math.sin(4 * math.pi * t / period))

# Interactive controls
class Slider:
    def __init__(self, x, y, w, min_val, max_val, initial, label):
        self.rect = pygame.Rect(x, y, w, 10)
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial
        self.label = label

    def draw(self, screen):
        pygame.draw.rect(screen, GRAY, self.rect)
        pos = self.rect.x + (self.value - self.min_val) / (self.max_val - self.min_val) * self.rect.w
        pygame.draw.circle(screen, RED, (int(pos), self.rect.y + 5), 7)
        text = font.render(f"{self.label}: {self.value:.3f}", True, WHITE)
        screen.blit(text, (self.rect.x, self.rect.y - 20))

    def update(self, mouse_pos, mouse_pressed):
        if mouse_pressed and self.rect.collidepoint(mouse_pos):
            x = max(self.rect.x, min(mouse_pos[0], self.rect.x + self.rect.w))
            self.value = self.min_val + (x - self.rect.x) / self.rect.w * (self.max_val - self.min_val)

class Button:
    def __init__(self, x, y, w, h, text, action):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = font.render(text, True, WHITE)
        self.action = action

    def draw(self, screen):
        pygame.draw.rect(screen, GRAY, self.rect)
        screen.blit(self.text, (self.rect.x + 10, self.rect.y + 5))

    def check_click(self, pos):
        if self.rect.collidepoint(pos):
            self.action()

def save_data():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for i, eco in enumerate(ecosystems):
        pd.DataFrame(eco.data).to_csv(f"{eco.planet_name}_data_{timestamp}.csv")
        with open(f"{eco.planet_name}_summary_{timestamp}.json", 'w') as f:
            json.dump({
                'planet': eco.planet_name,
                'final_microbes': eco.microbes,
                'final_predators': eco.predators,
                'final_resources': eco.resources
            }, f, indent=4)
    logger.info("Data saved successfully")

sliders = [
    Slider(50, 650, 200, 0.005, 0.015, 0.01, "Microbe Growth"),
    Slider(50, 700, 200, 0.001, 0.01, 0.005, "Predator Growth"),
    Slider(50, 750, 200, 0.005, 0.015, 0.01, "Radioactivity")
]
buttons = [Button(300, 650, 100, 30, "Save Data", save_data)]

# Main loop
clock = pygame.time.Clock()
t = 0
frame = 0
pop_histories = [[] for _ in ecosystems]
running = True

while running:
    mouse_pos = pygame.mouse.get_pos()
    mouse_pressed = pygame.mouse.get_pressed()[0]
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            for button in buttons:
                button.check_click(mouse_pos)

    # Update sliders
    for slider in sliders:
        slider.update(mouse_pos, mouse_pressed)
    ecosystems[0].microbe_growth = sliders[0].value
    ecosystems[0].predator_growth = sliders[1].value
    ecosystems[0].radioactivity = sliders[2].value

    # Update orbits
    t_span = [t, t + dt]
    sol = odeint(orbital_deriv, orbital_state, t_span, args=(M_star, masses))
    orbital_state = sol[-1]
    for i, planet in enumerate(planets):
        planet.state = orbital_state[i*4:i*4+4]
        planet.pos.x = width / 2 + planet.state[0] * scale
        planet.pos.y = height / 2 + planet.state[1] * scale
        planet.vel.x = planet.state[2] * scale
        planet.vel.y = planet.state[3] * scale
        planet.trail.append(planet.pos.copy())
        if len(planet.trail) > 500:
            planet.trail.pop(0)
    t += dt

    # Update ecosystems
    env_cycle = fourier_cycle(t)
    for eco in ecosystems:
        eco.update(dt, env_cycle)

    # Diagnostics
    if frame % 100 == 0:
        energy = check_energy(orbital_state, masses, M_star)
        energy_diff = (energy - initial_energy) / abs(initial_energy)
        if abs(energy_diff) > 0.01:
            logger.warning(f"Energy conservation error: {energy_diff:.2%}")
        for i, planet in enumerate(planets):
            ecc = eccentricity(planet.state, planet.mass, M_star)
            logger.info(f"{planet.name} eccentricity: {ecc:.3f}")

    # Draw
    screen.fill(BLACK)
    pygame.draw.circle(screen, YELLOW, (int(star_pos.x), int(star_pos.y)), 20)
    for planet in planets:
        pygame.draw.circle(screen, planet.color, (int(planet.pos.x), int(planet.pos.y)), 10)
        for i in range(1, len(planet.trail)):
            pygame.draw.line(screen, WHITE, planet.trail[i-1], planet.trail[i], 1)
        text = font.render(planet.name, True, WHITE)
        screen.blit(text, (planet.pos.x + 15, planet.pos.y))

    # Ecosystem visuals
    for i, eco in enumerate(ecosystems):
        y_offset = 50 + i * 120
        pop_size = min(int(eco.microbes / 100), 200)
        pred_size = min(int(eco.predators / 10), 200)
        res_size = min(int(eco.resources / 100), 200)
        pygame.draw.rect(screen, GREEN, (50, y_offset, pop_size, 20))
        pygame.draw.rect(screen, RED, (50, y_offset + 30, pred_size, 20))
        pygame.draw.rect(screen, BLUE, (50, y_offset + 60, res_size, 20))
        text = font.render(f"{eco.planet_name} Microbes: {eco.microbes:.0f} | Predators: {eco.predators:.0f}", True, WHITE)
        screen.blit(text, (50, y_offset - 20))

        # Population graph
        pop_histories[i].append(eco.microbes)
        if len(pop_histories[i]) > 200:
            pop_histories[i].pop(0)
        for j in range(1, len(pop_histories[i])):
            pygame.draw.line(screen, GREEN, (800 + j - 1, 300 + i * 150 - pop_histories[i][j-1] / 100),
                             (800 + j, 300 + i * 150 - pop_histories[i][j] / 100), 1)

    # Draw controls
    for slider in sliders:
        slider.draw(screen)
    for button in buttons:
        button.draw(screen)

    # Analytics
    text = font.render(f"Time: {t / 86400:.2f} days | Energy Diff: {energy_diff:.2%}", True, WHITE)
    screen.blit(text, (50, 20))

    pygame.display.flip()
    clock.tick(60)
    frame += 1

pygame.quit()
sys.exit()