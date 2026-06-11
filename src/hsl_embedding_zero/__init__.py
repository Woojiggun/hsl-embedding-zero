# -*- coding: utf-8 -*-
"""hsl-embedding-zero — feed bytes to a transformer with ZERO learned input parameters.

Raw bytes fed directly to a transformer are known to fail — that is why learned embeddings
exist. This package tests (and ships) the alternative the HSL substrate makes possible: the
27-D byte-signal features (change-rate / Gray-code Δ, Δ², boundary, exact 8-point Fourier,
phase) carry enough structure that the learned front door becomes unnecessary:

    bytes → HSL features (frozen, lossless, 4.6 KB LUT) → fixed zero-pad → transformer

No tokenizer. No embedding table. No learned input projection. Channels enter UNMIXED —
each feature keeps a fixed address; the first learned combination happens inside attention,
where it is trainable and inspectable.

Measured (59M-class lean decoder, 3-modality mix, fixed 3000-step budget, seed 0 — see README):

    input front door                         text bpb   caption bpb   learned input params
    zero (this package)                        2.483       1.503                  0
    learned projection on HSL features         2.457       1.329           ~125,000
    plain learned byte embedding (standard)    2.848       2.532           ~132,000

Same quality as a learned front door (≤1%), and clearly better than a standard learned byte
embedding at equal budget — with nothing to train at the door. Halving the slot count
(K=16) keeps text/caption bpb identical (2.4815 / 1.4965).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
import hsl_embedding as hsl

__version__ = "0.1.0"
FEAT_DIM = int(hsl.FEAT_DIM)                  # 27


class ZeroInput(torch.nn.Module):
    """Byte ids → transformer-ready slots with zero learned parameters.

    forward(ids): [B, L] longs in 0..255 → [B, L//K, dim]   (K bytes per attention slot)
    stream(ids):  [B, L] → [B, L, dim]                       (per-byte path, e.g. AR output stream)

    The HSL features fill the first K*FEAT_DIM (resp. FEAT_DIM) channels at fixed addresses;
    remaining channels are zero. dim must satisfy K*FEAT_DIM <= dim (K<=18 at dim=512).
    """

    def __init__(self, K: int = 8, dim: int = 512):
        super().__init__()
        if K * FEAT_DIM > dim:
            raise ValueError(f"K*{FEAT_DIM}={K*FEAT_DIM} exceeds dim={dim} (max K={dim//FEAT_DIM})")
        self.K, self.dim = K, dim
        self._emb = hsl.Embedding()           # exact tensor path (bit-identical to hsl.embed)
        for p in self._emb.parameters():
            p.requires_grad_(False)           # the substrate is frozen — that is the point

    @property
    def learned_parameters(self) -> int:
        return 0

    def features(self, ids: torch.Tensor) -> torch.Tensor:
        """[..., L] byte ids → [..., L, 27] exact HSL features (frozen)."""
        with torch.no_grad():
            return self._emb(ids)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, L = ids.shape
        n = (L // self.K) * self.K
        f = self.features(ids[:, :n]).reshape(B, n // self.K, self.K * FEAT_DIM)
        return F.pad(f, (0, self.dim - f.shape[-1]))

    def stream(self, ids: torch.Tensor) -> torch.Tensor:
        f = self.features(ids)
        return F.pad(f, (0, self.dim - FEAT_DIM))


def zero_input(data: bytes, K: int = 8, dim: int = 512, device="cpu") -> torch.Tensor:
    """One-call convenience: raw bytes → [1, len//K, dim] slots (zero learned params)."""
    ids = torch.tensor(list(data), dtype=torch.long, device=device).unsqueeze(0)
    return ZeroInput(K, dim).to(device)(ids)
