import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

class QxBinEdge:
    """QxBin Logic on Edge: Personal Cubit Simulator - Rupesh Malpani Framework"""
    def __init__(self, grid_size: int = 4):
        self.grid_size = grid_size
        self.state = np.random.rand(grid_size, grid_size)
        self.state /= self.state.sum()

    def apply_superposition(self, bias: float = 0.7, power_n: int = 2, power_m: int = 1):
        frac_heads = bias ** power_n
        frac_tails = (1 - bias) ** power_m
        prob_matrix = np.outer(np.linspace(frac_heads, frac_tails, self.grid_size),
                               np.linspace(frac_heads, frac_tails, self.grid_size))
        self.state = (self.state + prob_matrix) / 2
        self.state /= self.state.sum()
        return self.state

    def measure(self) -> np.ndarray:
        flat = self.state.flatten()
        outcome_idx = np.random.choice(len(flat), p=flat)
        collapsed = np.zeros_like(flat)
        collapsed[outcome_idx] = 1.0
        return collapsed.reshape(self.state.shape)

    def visualize(self, title: str = "QxBin Probability Grid"):
        plt.figure(figsize=(8, 6))
        plt.imshow(self.state, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Probability Amplitude')
        plt.title(title)
        y, x = np.indices(self.state.shape)
        plt.scatter(x.flatten(), y.flatten(), s=self.state.flatten()*500, 
                    c='red', alpha=0.6, edgecolors='white')
        plt.grid(True, alpha=0.3)
        plt.show()

if __name__ == "__main__":
    qx = QxBinEdge(grid_size=5)
    print("Initial QxBin State:\n", qx.state)
    qx.apply_superposition(bias=0.75, power_n=3, power_m=1)
    print("After Superposition:\n", qx.state)
    qx.visualize("Edge QxBin: Room-Temp Cubit Simulation")
    measured = qx.measure()
    print("Measured (Collapsed):\n", measured)