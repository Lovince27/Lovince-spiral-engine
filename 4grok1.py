import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# 1. Binary Truth: Classical true/false representation
binary_truth = np.array([1, 0, 1, 1, 0], dtype=bool)  # True=1, False=0
print("Binary Truth (True/False):", binary_truth)
print("Numerical Form:", binary_truth.astype(int))  # Convert to 1/0 for clarity

# 2. Probabilistic Truth: Quantum-inspired probabilities
# Simulating qubit-like probabilities (e.g., superposition states)
probabilistic_truth = np.random.uniform(0, 1, 5)  # Random probabilities between 0 and 1
probabilistic_truth = probabilistic_truth / np.sum(probabilistic_truth)  # Normalize to sum to 1
print("\nProbabilistic Truth (Quantum-inspired):", probabilistic_truth)
print("Sum of Probabilities:", np.sum(probabilistic_truth))  # Should be ~1

# 3. Similarity to Ground Truth: Measuring reflection vs. shadow
# Ground truth vector (e.g., true data, like a DNA sequence or AI prediction)
ground_truth = np.array([1.0, 2.0, 3.0, 4.0])

# Observed data (reflection, close to truth)
observed_reflection = np.array([1.1, 1.9, 3.2, 3.8])

# Distorted data (shadow, further from truth)
observed_shadow = np.array([0.5, 1.0, 2.0, 5.0])

# Cosine similarity: Measures how closely data aligns with ground truth
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

reflection_similarity = cosine_similarity(ground_truth, observed_reflection)
shadow_similarity = cosine_similarity(ground_truth, observed_shadow)

print("\nGround Truth:", ground_truth)
print("Observed Reflection (Mirror):", observed_reflection)
print("Cosine Similarity (Reflection):", reflection_similarity)
print("Observed Shadow (Parchhai):", observed_shadow)
print("Cosine Similarity (Shadow):", shadow_similarity)

# Mean Squared Error (MSE) to quantify deviation
mse_reflection = np.mean((ground_truth - observed_reflection) ** 2)
mse_shadow = np.mean((ground_truth - observed_shadow) ** 2)
print("\nMSE (Reflection):", mse_reflection)
print("MSE (Shadow):", mse_shadow)