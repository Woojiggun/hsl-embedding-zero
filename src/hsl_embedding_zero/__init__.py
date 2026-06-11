# -*- coding: utf-8 -*-
"""hsl-embedding-zero -- feed bytes to a transformer with ZERO learned input parameters.

Raw bytes fed directly to a transformer are known to fail -- that is why learned embeddings
exist. This package ships the alternative the HSL substrate makes possible: the 27-D
byte-signal features (change-rate / Gray-code delta, delta^2, boundary, exact 8-point
Fourier, phase) carry enough structure that the learned front door becomes unnecessary:

    bytes -> HSL features (frozen, lossless, 4.6 KB LUT) -> fixed zero-pad -> transformer

No tokenizer. No embedding table. No learned input projection. Channels enter UNMIXED --
each feature keeps a fixed address; the first learned combination happens inside attention,
where it is trainable and inspectable.

Measured (lean decoder, 3-modality byte mix, fixed 3000-step budget, seed 0 -- see README):

    input front door                         text bpb   caption bpb   learned input params
    zero (this package)                        2.483       1.503                  0
    learned projection on HSL features         2.457       1.329           ~125,000
    plain learned byte embedding (standard)    2.848       2.532           ~132,000

It works: within 1% of a learned front door with nothing to train at the door, and the
standard learned-byte-embedding arm does not reach this at equal budget. Read as a
POSSIBILITY, not a victory -- capability per FLOP and per watt is the direction. Halving
the slot count (K=16) keeps text/caption bpb identical (2.4815 / 1.4965).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
import hsl_embedding as hsl

__version__ = "0.1.2"
FEAT_DIM = int(hsl.FEAT_DIM)                  # 27


class ZeroInput(torch.nn.Module):
    """Byte ids -> transformer-ready slots with zero learned parameters.

    forward(ids): [B, L] longs in 0..255 -> [B, L//K, dim]   (K bytes per attention slot)
    stream(ids):  [B, L] -> [B, L, dim]                       (per-byte path, e.g. AR output stream)

    The HSL features fill the first K*FEAT_DIM (resp. FEAT_DIM) channels at fixed addresses;
    remaining channels are zero. dim must satisfy K*FEAT_DIM <= dim (K<=18 at dim=512).
    """

    def __init__(self, K: int = 8, dim: int = 512, tail: str = "pad"):
        super().__init__()
        if K * FEAT_DIM > dim:
            raise ValueError(f"K*{FEAT_DIM}={K*FEAT_DIM} exceeds dim={dim} (max K={dim//FEAT_DIM})")
        if tail not in ("pad", "drop"):
            raise ValueError("tail must be 'pad' (0x00-pad to slot boundary) or 'drop'")
        self.K, self.dim, self.tail = K, dim, tail
        self._emb = hsl.Embedding()           # exact tensor path, zero learned parameters (LUT buffers)

    @property
    def learned_parameters(self) -> int:
        return 0

    def features(self, ids: torch.Tensor) -> torch.Tensor:
        """[..., L] byte ids -> [..., L, 27] exact HSL features (frozen)."""
        with torch.no_grad():
            return self._emb(ids)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """[B, L] -> [B, ceil(L/K), dim]. tail='pad' (default) 0x00-pads the last partial slot
        so NO bytes are silently lost; tail='drop' discards the L%K remainder (explicit opt-in)."""
        B, L = ids.shape
        r = L % self.K
        if r and self.tail == "pad":
            ids = F.pad(ids, (0, self.K - r))                      # origin-byte pad, documented
        n = (ids.shape[1] // self.K) * self.K
        f = self.features(ids[:, :n]).reshape(B, n // self.K, self.K * FEAT_DIM)
        return F.pad(f, (0, self.dim - f.shape[-1]))

    def stream(self, ids: torch.Tensor) -> torch.Tensor:
        f = self.features(ids)
        return F.pad(f, (0, self.dim - FEAT_DIM))


def zero_input(data: bytes, K: int = 8, dim: int = 512, device="cpu") -> torch.Tensor:
    """One-call convenience: raw bytes -> [1, len//K, dim] slots (zero learned params)."""
    ids = torch.tensor(list(data), dtype=torch.long, device=device).unsqueeze(0)
    return ZeroInput(K, dim).to(device)(ids)
