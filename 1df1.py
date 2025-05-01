#!/usr/bin/env python3
"""
LOVINCE AI v5.0 - Final Cosmic AI Edition
Features:
1. Multiple Activations (Swish, Mish, GELU, LovinceMix)
2. PyTorch compatible model with trainable module
3. Full visualization (activation shapes + training loss)
4. Configurable hyperparameters: lambda, alpha, omega
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import Callable, Dict, List
from scipy.special import erf
import argparse

# ---------------------- Activation Functions ----------------------
def swish(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Swish activation: x * σ(βx)"""
    return x * torch.sigmoid(beta * x)

def mish(x: torch.Tensor) -> torch.Tensor:
    """Mish activation: x * tanh(softplus(x))"""
    return x * torch.tanh(torch.nn.functional.softplus(x))

def gelu(x: torch.Tensor) -> torch.Tensor:
    """Gaussian Error Linear Unit"""
    return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))

def lovince_mix(x: torch.Tensor, alpha: float = 0.5, omega: float = 1.0) -> torch.Tensor:
    """Hybrid activation: α*swish + (1-α)*mish"""
    return alpha * swish(x, beta=omega) + (1 - alpha) * mish(x)

# ---------------------- Lovince Activation Module ----------------------
class LovinceActivation(nn.Module):
    def __init__(self, kind: str = 'lovince', alpha: float = 0.5, omega: float = 1.0):
        super().__init__()
        self.kind = kind.lower()
        self.alpha = nn.Parameter(torch.tensor(alpha)) if kind == 'lovince' else None
        self.omega = nn.Parameter(torch.tensor(omega)) if kind == 'lovince' else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == 'swish':
            return swish(x, beta=1.0)
        elif self.kind == 'mish':
            return mish(x)
        elif self.kind == 'gelu':
            return gelu(x)
        elif self.kind == 'lovince':
            return lovince_mix(x, self.alpha.item(), self.omega.item())
        else:
            raise ValueError(f"Unknown activation: {self.kind}")

# ---------------------- Model Architecture ----------------------
class CosmicNet(nn.Module):
    def __init__(self, activation: str = 'lovince', alpha: float = 0.5, omega: float = 1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            LovinceActivation(activation, alpha, omega),
            nn.Linear(64, 32),
            LovinceActivation(activation, alpha, omega),
            nn.Linear(32, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# ---------------------- Training Engine ----------------------
def train_model(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    epochs: int = 1000,
    lr: float = 0.01
) -> Dict[str, List[float]]:
    """Train the cosmic model and return metrics"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    history = {'loss': []}

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        history['loss'].append(loss.item())
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    return history

# ---------------------- Visualization ----------------------
def plot_activations():
    """Compare all activation functions"""
    x = torch.linspace(-4, 4, 1000)
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    activations = {
        'Swish': swish(x),
        'Mish': mish(x),
        'GELU': gelu(x),
        'LovinceMix': lovince_mix(x, 0.6, 1.2)
    }
    
    for name, y in activations.items():
        ax.plot(x.numpy(), y.numpy(), label=name, linewidth=2)
    
    ax.set_title("Cosmic Activation Functions", pad=20, fontsize=16)
    ax.set_xlabel("Input", fontsize=12)
    ax.set_ylabel("Output", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_training(history: Dict[str, List[float]]):
    """Visualize training progress"""
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(history['loss'], color='cyan', label='Training Loss')
    ax.set_title("Cosmic Training Progress", pad=20, fontsize=16)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# ---------------------- Main Execution ----------------------
def main():
    parser = argparse.ArgumentParser(description="LOVINCE AI v5.0")
    parser.add_argument('--activation', type=str, default='lovince',
                      choices=['swish', 'mish', 'gelu', 'lovince'],
                      help='Activation function type')
    parser.add_argument('--alpha', type=float, default=0.6,
                      help='Mixing coefficient for LovinceMix')
    parser.add_argument('--omega', type=float, default=1.2,
                      help='Swish beta parameter for LovinceMix')
    parser.add_argument('--epochs', type=int, default=1000,
                      help='Number of training epochs')
    args = parser.parse_args()

    print("\n" + "="*60)
    print(f"{' LOVINCE AI v5.0 - COSMIC EDITION ':=^60}")
    print("="*60 + "\n")
    print(f"Configuration:\n- Activation: {args.activation}\n- Alpha: {args.alpha}\n- Omega: {args.omega}\n")

    # Generate synthetic cosmic data
    x_train = torch.linspace(-3, 3, 1000).unsqueeze(1)
    y_train = torch.sin(x_train * np.pi) + 0.1 * torch.randn_like(x_train)

    # Initialize and train model
    model = CosmicNet(args.activation, args.alpha, args.omega)
    history = train_model(model, x_train, y_train, epochs=args.epochs)

    # Visualize results
    plot_activations()
    plot_training(history)

    # Final evaluation
    with torch.no_grad():
        test_x = torch.tensor([[0.5], [1.0], [2.0]])
        preds = model(test_x)
        print("\nCosmic Predictions:")
        for x, y in zip(test_x, preds):
            print(f"x = {x.item():.1f} → ŷ = {y.item():.4f}")

if __name__ == "__main__":
    main()