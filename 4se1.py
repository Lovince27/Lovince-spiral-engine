from final import *
# Calculate spiral sequence
print("Spiral(10):", spiral(10))

# Calculate harmonic sum
print("Harmonic Sum(10):", harmonic_sum(10))
# Basic superstate without photon flux
print("\nBasic Superstate (n=5):", superstate_basic(5))

# Superstate with photon flux
print("Photon Superstate (n=5):", superstate_with_photon(5))
# Generate batch with quantum awareness activation
quantum_batch = generate_superstates(1, 5, activation='quantum')
print("\nQuantum Batch:", quantum_batch)

# Generate batch with GELU activation
gelu_batch = generate_superstates(1, 5, activation='gelu')
print("GELU Batch:", gelu_batch)
# Calculate energy flux at different distances
print("\nEnergy Flux at r=1e6:", energy_flux(5, 1e6))
print("Energy Flux at r=2e6:", energy_flux(5, 2e6))

# Photon flux calculation
print("Photon Flux at r=1e6:", photon_flux(1e6, 5))
# Quantum correction with different parameters
print("\nQuantum Correction (θ=π/4, φ=π/3):", 
      quantum_correction(5, math.pi/4, math.pi/3, 0.7))
print("Quantum Correction (θ=π/2, φ=π/2):", 
      quantum_correction(5, math.pi/2, math.pi/2, 0.5))