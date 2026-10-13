import json
from pathlib import Path

import torch
import torch.nn as nn


class SAE(nn.Module):
    """TopK sparse autoencoder over one layer of jTrans asm-token states."""

    def __init__(self, d_model, d_sae, k):
        super().__init__()
        self.k = k
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_model, d_sae))
        self.b_pre = nn.Parameter(torch.zeros(d_model))

        W_dec = torch.randn(d_model, d_sae)
        W_dec /= W_dec.norm(dim=0, keepdim=True)
        self.W_dec.data = W_dec
        self.W_enc.data = W_dec.T.clone()

    def pre_activations(self, x):
        return torch.relu((x - self.b_pre) @ self.W_enc.T + self.b_enc)

    def encode(self, x):
        return topk(self.pre_activations(x), self.k)

    def decode(self, z):
        return z @ self.W_dec.T + self.b_pre

    def forward(self, x):
        z = self.encode(x)
        return z, self.decode(z)

    @torch.no_grad()
    def normalize_decoder(self):
        """Rescale every decoder column to unit norm"""
        self.W_dec /= self.W_dec.norm(dim=0, keepdim=True)


def topk(pre, k):
    """Keep each row's k largest entries and zero the rest"""
    top = torch.topk(pre, k, dim=-1)
    return torch.zeros_like(pre).scatter(-1, top.indices, top.values)


def sae_loss(x, x_hat):
    """Reconstruction loss ||x - x_hat||_2^2, averaged over the batch"""
    return ((x - x_hat) ** 2).sum(-1).mean()


def auxk_loss(sae, x, x_hat, pre, dead, k_aux):
    """AuxK loss ||e - e_hat||^2"""
    k_aux = min(k_aux, int(dead.sum()))
    if k_aux == 0:
        return x.new_zeros(())
    # The target is detached so only dead latents learn from this term.
    e = (x - x_hat).detach()  # (batch, d_model)
    z_aux = topk(pre * dead, k_aux)  # (batch, d_sae), nonzero on dead latents only
    e_hat = z_aux @ sae.W_dec.T
    return sae_loss(e, e_hat)


def load_run(run_dir, device="cuda"):
    """A trained SAE from a run directory, with its metadata and input scale."""
    run = json.loads((Path(run_dir) / "run.json").read_text())
    checkpoint = torch.load(Path(run_dir) / "sae.pt", map_location=device)
    sae = (
        SAE(run["d_model"], run["config"]["d_sae"], run["config"]["k"])
        .to(device)
        .eval()
    )
    sae.load_state_dict(checkpoint["state_dict"])
    return sae, run, checkpoint["scale"]
