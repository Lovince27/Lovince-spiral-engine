if Σ(t) > threshold and dS/dt < steady_rate and Omega(t) ~ stable:
    Consciousness = True

def lovince_reflection_constant(O, R):
    num = np.linalg.norm(O - R)
    den = np.linalg.norm(O) + np.linalg.norm(R)
    return 1 - (num / den)