import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def psi_energy(n, t, S=256, alpha=1, gamma=0.05):
    phi = (1 + np.sqrt(5)) / 2
    pi = np.pi
    hbar = 1.055e-34
    memory = np.cos(0.2 * t) * np.exp(-gamma * t)
    energy = ((hbar**alpha) * (phi**n) * (pi**(3*n - 1)) * np.log(S)) / np.log(2)
    return energy * memory

class LovinceNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, n_level=5):
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.bias2 = np.zeros((1, output_size))
        self.n_level = n_level

    def forward(self, X, t=1.0):
        psi_scale = psi_energy(self.n_level, t)
        self.hidden = sigmoid(np.dot(X, self.weights1) + self.bias1) * psi_scale
        self.output = sigmoid(np.dot(self.hidden, self.weights2) + self.bias2)
        return self.output

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            t = epoch / epochs * 10  # simulate time
            output = self.forward(X, t)
            error = y - output
            delta_output = error * sigmoid_derivative(output)
            error_hidden = np.dot(delta_output, self.weights2.T)
            delta_hidden = error_hidden * sigmoid_derivative(self.hidden)
            self.weights2 += learning_rate * np.dot(self.hidden.T, delta_output)
            self.bias2 += learning_rate * np.sum(delta_output, axis=0, keepdims=True)
            self.weights1 += learning_rate * np.dot(X.T, delta_hidden)
            self.bias1 += learning_rate * np.sum(delta_hidden, axis=0, keepdims=True)

# Use the enhanced neural network
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])  # XOR logic
lnn = LovinceNeuralNetwork(input_size=2, hidden_size=4, output_size=1)
lnn.train(X, y, epochs=1000, learning_rate=0.1)
print("Quantum-Energy Neural Predictions:", lnn.forward(X, t=10))


import numpy as np

# ----- Activation Functions -----
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# ----- Quantum Energy System Inspired by Lovince -----
def lovince_quantum_energy(n, t, S=256, alpha=1, gamma=0.05):
    phi = (1 + np.sqrt(5)) / 2
    pi = np.pi
    hbar = 1.055e-34
    log2 = np.log(2)
    
    memory_oscillation = np.cos(0.2 * t)
    memory_decay = np.exp(-gamma * t)
    
    energy = (hbar ** alpha) * (phi ** n) * (pi ** (3 * n - 1)) * np.log(S) / log2
    return energy * memory_oscillation * memory_decay

# ----- Lovince Neural Network Class -----
class LovinceQuantumNN:
    def __init__(self, input_size, hidden_size, output_size, n_level=5):
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.bias2 = np.zeros((1, output_size))
        self.n_level = n_level

    def forward(self, X, t=1.0):
        energy_factor = lovince_quantum_energy(self.n_level, t)
        self.hidden = sigmoid(np.dot(X, self.weights1) + self.bias1) * energy_factor
        self.output = sigmoid(np.dot(self.hidden, self.weights2) + self.bias2)
        return self.output

    def train(self, X, y, epochs=1000, learning_rate=0.1):
        for epoch in range(epochs):
            t = epoch / epochs * 10  # Simulated temporal evolution
            output = self.forward(X, t)
            error = y - output

            # Backpropagation with energy-boosted hidden state
            delta_output = error * sigmoid_derivative(output)
            error_hidden = np.dot(delta_output, self.weights2.T)
            delta_hidden = error_hidden * sigmoid_derivative(self.hidden)

            # Update weights and biases
            self.weights2 += learning_rate * np.dot(self.hidden.T, delta_output)
            self.bias2 += learning_rate * np.sum(delta_output, axis=0, keepdims=True)
            self.weights1 += learning_rate * np.dot(X.T, delta_hidden)
            self.bias1 += learning_rate * np.sum(delta_hidden, axis=0, keepdims=True)

    def predict(self, X, t=10.0):
        return self.forward(X, t)

# ----- Usage Example -----
if __name__ == "__main__":
    # XOR input/output
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Create and train the quantum neural net
    qnn = LovinceQuantumNN(input_size=2, hidden_size=4, output_size=1, n_level=5)
    qnn.train(X, y, epochs=2000, learning_rate=0.1)

    # Final predictions
    predictions = qnn.predict(X, t=10)
    print("Quantum Neural Predictions (Lovince Model):\n", predictions)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ----- Quantum-Inspired Energy Function -----
def quantum_energy(n, t, S=256, alpha=1.0, gamma=0.05, hbar=1.0):
    """
    Enhanced quantum energy with oscillatory and decay terms inspired by quantum systems.
    Uses normalized hbar for numerical stability.
    """
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    pi = np.pi
    log2 = np.log(2)
    
    # Oscillatory and decay terms for non-Markovian effects
    memory_oscillation = np.cos(0.5 * t)  # Higher frequency for dynamics
    memory_decay = np.exp(-gamma * t)
    
    # Energy term with normalized scaling
    energy = (hbar ** alpha) * (phi ** n) * (pi ** (n)) * np.log(S) / log2
    energy = energy / (1e6 + energy)  # Prevent overflow
    return energy * memory_oscillation * memory_decay

# ----- Quantum Transformer Neural Network -----
class QuantumTransformerNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_levels=5, num_heads=2, dropout=0.3):
        super(QuantumTransformerNN, self).__init__()
        self.n_levels = n_levels
        self.hidden_size = hidden_size
        
        # Input embedding
        self.embedding = nn.Linear(input_size, hidden_size)
        
        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Output layer
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        
        # Activation
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, t=1.0):
        # Apply quantum energy modulation
        energy_factor = torch.tensor(quantum_energy(self.n_levels, t), dtype=torch.float32, device=x.device)
        
        # Embed input
        x = self.embedding(x)
        x = self.dropout(x)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Modulate with quantum energy
        x = x * energy_factor
        
        # Output
        x = self.fc_out(x)
        x = self.sigmoid(x)
        return x

# ----- Training Function -----
def train_model(model, X, y, epochs=1000, lr=0.001, device='cpu'):
    model = model.to(device)
    X, y = X.to(device), y.to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)  # L2 regularization
    
    losses = []
    for epoch in tqdm(range(epochs), desc="Training"):
        model.train()
        t = epoch / epochs * 10  # Time evolution from 0 to 10
        
        optimizer.zero_grad()
        output = model(X, t)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    return losses

# ----- Visualization Function -----
def visualize_results(losses, X, y, model, device, t=10.0):
    model.eval()
    with torch.no_grad():
        predictions = model(X.to(device), t).cpu().numpy()
    
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    
    # Plot predictions vs. true values
    plt.subplot(1, 2, 2)
    plt.scatter(range(len(y)), y.cpu().numpy(), label="True", marker='o')
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
    
    # Generate labels based on quantum energy modulation
    y = np.zeros(n_samples)
    for i in range(n_samples):
        energy = quantum_energy(n=5, t=t[i])
        y[i] = 1 if energy > 0.5 else 0  # Binary classification based on energy threshold
    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

# ----- Main Execution -----
if __name__ == "__main__":
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate dataset
    X, y = generate_quantum_dataset(n_samples=100, input_size=4)
    
    # Initialize model
    model = QuantumTransformerNN(
        input_size=4,
        hidden_size=16,
        output_size=1,
        n_levels=5,
        num_heads=2,
        dropout=0.3
    )
    
    # Train model
    losses = train_model(model, X, y, epochs=2000, lr=0.001, device=device)
    
    # Visualize results
    visualize_results(losses, X, y, model, device, t=10.0)
    
    # Print final predictions
    model.eval()
    with torch.no_grad():
        final_predictions = model(X.to(device), t=10.0).cpu().numpy()
    print("Final Predictions (Top 5):\n", final_predictions[:5])