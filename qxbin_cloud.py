import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt

@njit(parallel=True)
def qxbin_chain_parallel(states: np.ndarray, biases: np.ndarray, powers: np.ndarray) -> np.ndarray:
    n = len(states)
    for i in prange(n):
        b, pn, pm = biases[i], powers[i,0], powers[i,1]
        frac = b ** pn
        states[i] = (states[i] * frac + (1 - states[i]) ** pm) / 2
        states[i] /= states[i].sum()
    return states

class QxBinCloud:
    """QxBin Logic on Cloud: Scalable Probability Chains - Rupesh Malpani Framework"""
    def __init__(self, num_cubits: int = 10, grid_size: int = 6):
        self.num_cubits = num_cubits
        self.grid_size = grid_size
        self.states = np.random.rand(num_cubits, grid_size, grid_size)
        self.states /= self.states.sum(axis=(1,2))[:, None, None]

    def evolve_chains(self, biases: np.ndarray = None):
        if biases is None:
            biases = np.random.uniform(0.4, 0.8, self.num_cubits)
        powers = np.random.randint(1, 5, (self.num_cubits, 2))
        flat_states = self.states.reshape(self.num_cubits, -1)
        flat_states = qxbin_chain_parallel(flat_states, biases, powers)
        self.states = flat_states.reshape(self.num_cubits, self.grid_size, self.grid_size)
        return self.states.mean(axis=0)

    def visualize_aggregate(self):
        agg = self.states.mean(axis=0)
        plt.figure(figsize=(8, 6))
        plt.imshow(agg, cmap='plasma')
        plt.colorbar()
        plt.title(f"Cloud QxBin Aggregate ( {self.num_cubits} Cubits )")
        plt.show()

if __name__ == "__main__":
    qx_cloud = QxBinCloud(num_cubits=20, grid_size=8)
    agg = qx_cloud.evolve_chains()
    qx_cloud.visualize_aggregate()
    print("Optimized aggregate shape:", agg.shape)