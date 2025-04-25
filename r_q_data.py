# Generates entangled states
qc.cx(np.random.choice(3), np.random.choice(3))  
qc.u3(θ,φ,λ)  # Random unitary operations