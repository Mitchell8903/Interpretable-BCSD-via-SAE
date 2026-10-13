"""The SAE training config and the reported grid.

Every field is a train_sae.py flag of the same name, and every run saves its
resolved config in run.json.
"""

from dataclasses import dataclass
from pathlib import Path

SWEEP_LAYERS = (6,)
SWEEP_SEEDS = (0, 1, 2)
SWEEP_K = (16, 32, 64)
SWEEP_D_SAE = (4096, 8192, 16384)


@dataclass
class SAEConfig:

    layer: int = 6
    activations_dir: str = "data/activations"
    heldout_frac: float = 0.2

    split_seed: int = 0

    d_sae: int = 16384

    # Active features per token: the TopK sparsity, and so the L0
    k: int = 32

    # AuxK parameters
    k_aux: int = 512
    aux_coef: float = 1 / 32
    dead_tokens: int = 10_000_000

    # Training parameters
    lr: float = 1e-4
    batch_size: int = 4096
    epochs: int = 60

    seed: int = 0

    # Empty means results/layer<layer>/sweep.
    out_dir: str = ""

    def sweep_dir(self):
        """The sweep this run belongs to."""
        return Path(self.out_dir or f"results/layer{self.layer}/sweep")

    def name(self):
        """Directory name identifying this run within its sweep."""
        return f"dsae{self.d_sae}_k{self.k}_seed{self.seed}"

    def run_dir(self):
        """Where this run's checkpoint and metrics are written."""
        return self.sweep_dir() / self.name()
