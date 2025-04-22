import jax
import jax.numpy as jnp
import haiku as hk
import optax
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from tqdm import tqdm

# Set random seed for reproducibility
rng_key = jax.random.PRNGKey(42)
np.random.seed(42)

# ----- Quantum-Inspired Energy Function -----
def quantum_energy(n, t, S=256, alpha=1.0, gamma=0.05, hbar=1.0):
    """
    Quantum-inspired energy with oscillatory and decay terms.
    Normalized for numerical stability.
    """
    phi = (1 + jnp.sqrt(5)) / 2  # Golden ratio
    pi = jnp.pi
    log2 = jnp.log(2)
    
    # Oscillatory and decay terms
    memory_oscillation = jnp.cos(0.5 * t)
    memory_decay = jnp.exp(-gamma * t)
    
    # Energy term
    energy = (hbar ** alpha) * (phi ** n) * (pi ** n) * jnp.log(S) / log2
    energy = energy / (1e6 + energy)  # Prevent overflow
    return energy * memory_oscillation * memory_decay

# ----- Transformer Model Definition -----
def transformer_fn(x, t, hidden_size, num_heads, output_size, n_levels=5, dropout_rate=0.3):
    """
    Transformer model with quantum energy modulation using Haiku.
    """
    # Initialize Haiku modules
    x = hk.Linear(hidden_size)(x)  # Input embedding
    x = hk.dropout(hk.next_rng_key(), dropout_rate, x)
    
    # Self-attention (simplified transformer layer)
    attn = hk.MultiHeadAttention(
        num_heads=num_heads,
        key_size=hidden_size // num_heads,
        model_size=hidden_size,
        w_init_scale=1.0
    )
    x = attn(x, x, x)  # Self-attention
    x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
    x = hk.dropout(hk.next_rng_key(), dropout_rate, x)
    
    # Feed-forward network
    ff = hk.Sequential([
        hk.Linear(hidden_size * 4), jax.nn.relu,
        hk.Linear(hidden_size)
    ])
    x = x + ff(x)  # Residual connection
    x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(x)
    
    # Quantum energy modulation
    energy = quantum_energy(n_levels, t)
    x = x * energy
    
    # Output layer
    x = hk.Linear(output_size)(x)
    x = jax.nn.sigmoid(x)
    return x

# ----- Initialize Haiku Model -----
def init_model(input_size, hidden_size, output_size, num_heads, n_levels=5, dropout_rate=0.3):
    transformer = hk.transform(partial(
        transformer_fn,
        hidden_size=hidden_size,
        num_heads=num_heads,
        output_size=output_size,
        n_levels=n_levels,
        dropout_rate=dropout_rate
    ))
    return transformer

# ----- Loss Function -----
def loss_fn(params, rng, x, y, t):
    logits = model.apply(params, rng, x, t)
    return jnp.mean(optax.sigmoid_binary_cross_entropy(logits, y))

# ----- Training Step -----
@partial(jax.jit, static_argnums=(4,))
def train_step(params, rng, opt_state, batch, optimizer):
    x, y, t = batch
    rng, subrng = jax.random.split(rng)
    loss, grads = jax.value_and_grad(loss_fn)(params, subrng, x, y, t)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, rng

# ----- Training Loop -----
def train_model(params, rng, X, y, epochs=1000, lr=0.001):
    optimizer = optax.adam(lr, b1=0.9, b2=0.999)
    opt_state = optimizer.init(params)
    
    losses = []
    for epoch in tqdm(range(epochs), desc="Training"):
        t = epoch / epochs * 10  # Time evolution
        batch = (X, y, t)
        rng, subrng = jax.random.split(rng)
        params, opt_state, loss, rng = train_step(params, subrng, opt_state, batch, optimizer)
        losses.append(loss)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    return params, losses

# ----- Visualization Function -----
def visualize_results(losses, X, y, params, rng, t=10.0):
    predict_fn = jax.jit(lambda x: model.apply(params, rng, x, t))
    predictions = predict_fn(X)
    
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    
    # Plot predictions vs. true values
    plt.subplot(1, 2, 2)
    plt.scatter(range(len(y)), y, label="True", marker='o')
    plt.scatter(range(len(y)), predictions, label="Predicted", marker='x')
    plt.xlabel("Sample")
    plt.ylabel("Output")
    plt.title("Predictions vs. True Values")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# ----- Generate Synthetic Quantum-Inspired Dataset -----
def generate_quantum_dataset(n_samples=100, input_size=4):
    X = np.random.randn(n_samples, input_size)
    t = np.linspace(0, 10, n_samples)
    y = np.zeros(n_samples)
    for i in range(n_samples):
        energy = quantum_energy(n=5, t=t[i])
        y[i] = 1 if energy > 0.5 else 0  # Binary classification
    return jnp.array(X), jnp.array(y).reshape(-1, 1)

# ----- Main Execution -----
if __name__ == "__main__":
    # Model parameters
    input_size = 4
    hidden_size = 16
    output_size = 1
    num_heads = 2
    n_levels = 5
    dropout_rate = 0.3
    epochs = 2000
    lr = 0.001
    
    # Generate dataset
    X, y = generate_quantum_dataset(n_samples=100, input_size=input_size)
    
    # Initialize model
    model = init_model(input_size, hidden_size, output_size, num_heads, n_levels, dropout_rate)
    rng, init_rng = jax.random.split(rng_key)
    params = model.init(init_rng, X[:1], t=0.0)
    
    # Train model
    params, losses = train_model(params, rng, X, y, epochs=epochs, lr=lr)
    
    # Visualize results
    visualize_results(losses, X, y, params, rng, t=10.0)
    
    # Print final predictions
    predict_fn = jax.jit(lambda x: model.apply(params, rng, x, t=10.0))
    final_predictions = predict_fn(X)
    print("Final Predictions (Top 5):\n", final_predictions[:5])