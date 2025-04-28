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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(level=logging.INFO, filename='exoplanet_simulation.log', format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Initialize Pygame
pygame.init()
width, height = 1200, 800
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Super Lovince Exoplanet Simulation")
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
CYAN = (0, 255, 255)

# Physics constants
G = 6.67430e-11  # m^3 kg^-1 s^-2
c = 2.99792458e8  # m/s
M_star1 = 0.0898 * 1.989e30  # kg (TRAPPIST-1)
M_star2 = 0.05 * 1.989e30  # kg (Second star)
scale = 1e-7  # pixels/m
dt = 3600  # s
pixels_per_m = 1 / scale
dark_matter_density = 1e-22  # kg/m^3

# Planet class
class Planet:
    def __init__(self, pos, vel, mass, color, name, semi_major_axis, magnetic_field, star_id, chaos_factor):
        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(vel)
        self.mass = mass
        self.color = color
        self.name = name
        self.semi_major_axis = semi_major_axis
        self.magnetic_field = magnetic_field
        self.star_id = star_id  # 1 for star1, 2 for star2
        self.chaos_factor = chaos_factor
        self.trail = []
        self.state = [pos[0] / scale, pos[1] / scale, vel[0] / scale, vel[1] / scale]
        self.z = random.uniform(-0.1, 0.1)

# Ecosystem class
class Ecosystem:
    def __init__(self, planet_name, semi_major_axis, magnetic_field):
        self.microbes = 1000
        self.predators = 100
        self.resources = 10000
        self.co2 = 0.04
        self.o2 = 0.21
        self.temperature = 288
        self.microbe_growth = 0.01
        self.predator_growth = 0.005
        self.carrying_capacity = 10000
        self.mutation_rate = 0.001
        self.radioactivity = 0.01
        self.tunneling_prob = 0.0001
        self.km = 1000
        self.vmax = 10
        self.total_mass = self.microbes * 1e-12 + self.predators * 1e-11 + self.resources
        self.planet_name = planet_name
        self.semi_major_axis = semi_major_axis
        self.magnetic_field = magnetic_field
        self.mutation_flash = 0
        self.data = []
        self.ai_model = LinearRegression()
        self.scaler = StandardScaler()
        self.ai_predictions = []

    def update(self, dt, env_cycle):
        dm = (self.microbe_growth * self.microbes * (1 - self.microbes / self.carrying_capacity) -
              0.002 * self.microbes * self.predators) * dt
        dp = (0.001 * self.microbes * self.predators - 0.003 * self.predators) * dt
        self.microbes = max(0, self.microbes + dm)
        self.predators = max(0, self.predators + dp)

        resource_use = (self.microbes * 1e-12 + self.predators * 1e-11) * dt
        self.resources -= resource_use
        reaction_rate = self.vmax * self.resources / (self.km + self.resources)
        self.resources += reaction_rate * dt
        self.resources = max(0, self.resources)

        co2_consumption = 0.001 * self.microbes * self.co2 * dt
        o2_production = 0.0008 * self.microbes * self.co2 * dt
        self.co2 = max(0, self.co2 - co2_consumption)
        self.o2 = min(1, self.o2 + o2_production)

        temp_base = 300 / (self.semi_major_axis / 1.5e11)**2
        temp_increase = 10 * self.co2
        self.temperature = temp_base + temp_increase
        if self.temperature > 350 or self.temperature < 250:
            self.microbe_growth *= 0.99

        tunneling_factor = self.tunneling_prob * self.radioactivity / (1 + self.magnetic_field)
        if random.gauss(0, 1) > 2 and random.random() < tunneling_factor * dt * env_cycle:
            self.microbe_growth += random.uniform(-0.002, 0.002)
            self.mutation_flash = 10
        if random.random() < self.mutation_rate * self.radioactivity * dt * env_cycle / (1 + self.magnetic_field):
            self.microbe_growth += random.uniform(-0.001, 0.001)
        self.mutation_flash = max(0, self.mutation_flash - 1)

        biosignature = self.o2 * 0.5 + self.microbes / 10000

        if len(self.data) > 10:
            X = np.array([[d['time_days'], d['o2'], d['co2'], d['resources']] for d in self.data[-10:]])
            y = np.array([d['microbes'] for d in self.data[-10:]])
            X_scaled = self.scaler.fit_transform(X)
            self.ai_model.fit(X_scaled, y)
            next_X = self.scaler.transform([[t / 86400 + 1, self.o2, self.co2, self.resources]])
            self.ai_predictions.append(self.ai_model.predict(next_X)[0])

        new_mass = self.microbes * 1e-12 + self.predators * 1e-11 + self.resources
        if abs(new_mass - self.total_mass) > 1e-6:
            logger.warning(f"{self.planet_name} mass conservation error: {new_mass - self.total_mass:.2e} kg")
        self.total_mass = new_mass

        habitability = min(1, max(0, (self.o2 * 2 + self.resources / 10000 - abs(self.temperature - 288) / 50)))
        self.data.append({
            'time_days': t / 86400,
            'microbes': self.microbes,
            'predators': self.predators,
            'resources': self.resources,
            'co2': self.co2,
            'o2': self.o2,
            'temperature': self.temperature,
            'habitability': habitability,
            'biosignature': biosignature
        })

# Orbital dynamics
def orbital_deriv(state, t, M1, M2, masses, chaos_factors, star2_pos):
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
        r1 = max(math.sqrt(x**2 + y**2), 1e8)
        ax += G * M1 * x / r1**3
        ay += G * M1 * y / r1**3
        dx2, dy2 = x - star2_pos.x / scale, y - star2_pos.y / scale
        r2 = max(math.sqrt(dx2**2 + dy2**2), 1e8)
        ax += G * M2 * dx2 / r2**3
        ay += G * M2 * dy2 / r2**3
        ax += -G * dark_matter_density * x
        ay += -G * dark_matter_density * y
        chaos = chaos_factors[i] * (math.sin(t / 86400) * r1)
        ax += chaos * (vx - x / r1)
        ay += chaos * (vy - y / r1)
        derivs.extend([vx, vy, ax, ay])
    return derivs

# Initialize planets
planets = [
    Planet((width / 2 + 50, height / 2), (0, 35000 * scale), 5.972e24, BLUE, "TRAPPIST-1e", 4.05e9, 5e-5, 1, 1e-10),
    Planet((width / 2 + 80, height / 2), (0, 32000 * scale), 3.285e23, PURPLE, "TRAPPIST-1f", 6.1e9, 1e-5, 1, 1e-10)
]
ecosystems = [Ecosystem(p.name, p.semi_major_axis, p.magnetic_field) for p in planets]
star_pos = pygame.Vector2(width / 2, height / 2)
star2_pos = pygame.Vector2(width / 2 - 100, height / 2)
orbital_state = [s for p in planets for s in p.state]
masses = [p.mass for p in planets]
chaos_factors = [p.chaos_factor for p in planets]

# Diagnostics
def check_energy(state, masses, M1, M2, star2_pos):
    energy = 0
    n = len(state) // 4
    for i in range(n):
        x, y, vx, vy = state[i*4:i*4+4]
        r1 = math.sqrt(x**2 + y**2)
        dx2, dy2 = x - star2_pos.x / scale, y - star2_pos.y / scale
        r2 = math.sqrt(dx2**2 + dy2**2)
        v = math.sqrt(vx**2 + vy**2)
        kinetic = 0.5 * masses[i] * v**2
        potential = -G * M1 * masses[i] / r1 - G * M2 * masses[i] / r2
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

initial_energy = check_energy(orbital_state, masses, M_star1, M_star2, star2_pos)

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

# Scenarios and Cosmic Mode
scenarios = {
    "Earth-like": {'microbe_growth': 0.01, 'predator_growth': 0.005, 'radioactivity': 0.01, 'co2': 0.04, 'o2': 0.21},
    "Extreme Radiation": {'microbe_growth': 0.005, 'predator_growth': 0.002, 'radioactivity': 0.05, 'co2': 0.1, 'o2': 0.1}
}
cosmic_mode = False

def save_data():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for i, eco in enumerate(ecosystems):
        pd.DataFrame(eco.data).to_csv(f"{eco.planet_name}_data_{timestamp}.csv")
        with open(f"{eco.planet_name}_summary_{timestamp}.json", 'w') as f:
            json.dump({
                'planet': eco.planet_name,
                'final_microbes': eco.microbes,
                'final_predators': eco.predators,
                'final_resources': eco.resources,
                'final_co2': eco.co2,
                'final_o2': eco.o2,
                'final_temperature': eco.temperature,
                'final_habitability': eco.data[-1]['habitability'],
                'final_biosignature': eco.data[-1]['biosignature']
            }, f, indent=4)
    logger.info("Data saved successfully")

def apply_scenario(scenario_name):
    for eco in ecosystems:
        eco.microbe_growth = scenarios[scenario_name]['microbe_growth']
        eco.predator_growth = scenarios[scenario_name]['predator_growth']
        eco.radioactivity = scenarios[scenario_name]['radioactivity']
        eco.co2 = scenarios[scenario_name]['co2']
        eco.o2 = scenarios[scenario_name]['o2']
    logger.info(f"Applied scenario: {scenario_name}")

def toggle_cosmic_mode():
    global cosmic_mode
    cosmic_mode = not cosmic_mode
    logger.info(f"Cosmic Mode: {'ON' if cosmic_mode else 'OFF'}")

sliders = [
    Slider(50, 650, 200, 0.005, 0.015, 0.01, "Microbe Growth"),
    Slider(50, 690, 200, 0.001, 0.01, 0.005, "Predator Growth"),
    Slider(50, 730, 200, 0.005, 0.015, 0.01, "Radioactivity"),
    Slider(50, 770, 200, 0.5, 2.0, 1.0, "Zoom")
]
buttons = [
    Button(300, 650, 100, 30, "Save Data", save_data),
    Button(300, 690, 150, 30, "Earth-like", lambda: apply_scenario("Earth-like")),
    Button(300, 730, 150, 30, "Extreme Radiation", lambda: apply_scenario("Extreme Radiation")),
    Button(300, 770, 150, 30, "Cosmic Mode", toggle_cosmic_mode)
]

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

    for slider in sliders:
        slider.update(mouse_pos, mouse_pressed)
    ecosystems[0].microbe_growth = sliders[0].value
    ecosystems[0].predator_growth = sliders[1].value
    ecosystems[0].radioactivity = sliders[2].value
    global_scale = sliders[3].value

    t_span = [t, t + dt]
    sol = odeint(orbital_deriv, orbital_state, t_span, args=(M_star1, M_star2, masses, chaos_factors, star2_pos))
    orbital_state = sol[-1]
    for i, planet in enumerate(planets):
        planet.state = orbital_state[i*4:i*4+4]
        planet.pos.x = width / 2 + planet.state[0] * scale * global_scale * (1 + planet.z)
        planet.pos.y = height / 2 + planet.state[1] * scale * global_scale * (1 + planet.z)
        planet.vel.x = planet.state[2] * scale
        planet.vel.y = planet.state[3] * scale
        planet.trail.append(planet.pos.copy())
        if len(planet.trail) > 500:
            planet.trail.pop(0)
    t += dt

    env_cycle = fourier_cycle(t)
    for eco in ecosystems:
        eco.update(dt, env_cycle)

    if frame % 100 == 0:
        energy = check_energy(orbital_state, masses, M_star1, M_star2, star2_pos)
        energy_diff = (energy - initial_energy) / abs(initial_energy)
        if abs(energy_diff) > 0.01:
            logger.warning(f"Energy conservation error: {energy_diff:.2%}")
        for i, planet in enumerate(planets):
            ecc = eccentricity(planet.state, planet.mass, M_star1 if planet.star_id == 1 else M_star2)
            logger.info(f"{planet.name} eccentricity: {ecc:.3f}")

    screen.fill(BLACK)
    if cosmic_mode:
        for i in range(height):
            alpha = int(50 * (1 - i / height))
            pygame.draw.line(screen, (50, 50, 100, alpha), (0, i), (width, i))
    pygame.draw.circle(screen, YELLOW, (int(star_pos.x), int(star_pos.y)), 20)
    pygame.draw.circle(screen, YELLOW, (int(star2_pos.x), int(star2_pos.y)), 15)
    pygame.draw.circle(screen, (50, 50, 50, 50), (int(star_pos.x), int(star_pos.y)), 300, 0)

    for i, planet in enumerate(planets):
        scale_factor = 1 + planet.z * 0.5
        planet_radius = int(10 * scale_factor * global_scale)
        glow_radius = int(15 * scale_factor * global_scale)
        pygame.draw.circle(screen, (*planet.color, 50), (int(planet.pos.x), int(planet.pos.y)), glow_radius, 0)
        pygame.draw.circle(screen, planet.color, (int(planet.pos.x), int(planet.pos.y)), planet_radius)
        if ecosystems[i].mutation_flash > 0:
            pygame.draw.circle(screen, WHITE, (int(planet.pos.x), int(planet.pos.y)), glow_radius + 5, 2)
        for j in range(1, len(planet.trail)):
            alpha = int(255 * (j / len(planet.trail)) * (2 if cosmic_mode else 1))
            pygame.draw.line(screen, (*WHITE, alpha), planet.trail[j-1], planet.trail[j], 1)
        text = font.render(planet.name, True, WHITE)
        screen.blit(text, (planet.pos.x + 15, planet.pos.y))

    for i, eco in enumerate(ecosystems):
        y_offset = 50 + i * 150
        pop_size = min(int(eco.microbes / 100), 200)
        pred_size = min(int(eco.predators / 10), 200)
        res_size = min(int(eco.resources / 100), 200)
        pygame.draw.rect(screen, GREEN, (50, y_offset, pop_size, 20))
        pygame.draw.rect(screen, RED, (50, y_offset + 30, pred_size, 20))
        pygame.draw.rect(screen, BLUE, (50, y_offset + 60, res_size, 20))
        text = font.render(f"{eco.planet_name} | Microbes: {eco.microbes:.0f} | O2: {eco.o2:.2f}% | Temp: {eco.temperature:.0f}K", True, WHITE)
        screen.blit(text, (50, y_offset - 20))

        pop_histories[i].append(eco.microbes)
        if len(pop_histories[i]) > 200:
            pop_histories[i].pop(0)
        for j in range(1, len(pop_histories[i])):
            pygame.draw.line(screen, GREEN, (800 + j - 1, 300 + i * 150 - pop_histories[i][j-1] / 100),
                             (800 + j, 300 + i * 150 - pop_histories[i][j] / 100), 1)
        if eco.ai_predictions:
            pred_y = 300 + i * 150 - eco.ai_predictions[-1] / 100
            pygame.draw.circle(screen, CYAN, (800 + 200, int(pred_y)), 5)

    resonance_stability = min(1, max(0, 1 - abs(energy_diff) * 100))
    stability_color = (int(255 * (1 - resonance_stability)), int(255 * resonance_stability), 0)
    pygame.draw.rect(screen, stability_color, (1100, 20, 50, 20))
    text = font.render("Resonance", True, WHITE)
    screen.blit(text, (1100, 40))

    for slider in sliders:
        slider.draw(screen)
    for button in buttons:
        button.draw(screen)

    dashboard = [
        f"Time: {t / 86400:.2f} days",
        f"Energy Diff: {energy_diff:.2%}",
        f"TRAPPIST-1e Hab: {ecosystems[0].data[-1]['habitability']:.2f} | Bio: {ecosystems[0].data[-1]['biosignature']:.2f}",
        f"TRAPPIST-1f Hab: {ecosystems[1].data[-1]['habitability']:.2f} | Bio: {ecosystems[1].data[-1]['biosignature']:.2f}",
        f"Cosmic Mode: {'ON' if cosmic_mode else 'OFF'}"
    ]
    for i, line in enumerate(dashboard):
        text = font.render(line, True, WHITE)
        screen.blit(text, (50, 20 + i * 20))

    pygame.display.flip()
    clock.tick(60)
    frame += 1

pygame.quit()
sys.exit()