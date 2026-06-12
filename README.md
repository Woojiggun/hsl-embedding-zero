# hsl-embedding-zero

[![PyPI](https://img.shields.io/pypi/v/hsl-embedding-zero.svg)](https://pypi.org/project/hsl-embedding-zero/)
[![DOI](https://zenodo.org/badge/1266175250.svg)](https://doi.org/10.5281/zenodo.20643551)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Downloads](https://static.pepy.tech/badge/hsl-embedding-zero)](https://pepy.tech/project/hsl-embedding-zero)

**Feed bytes to a transformer with ZERO learned input parameters.**

```bash
pip install hsl-embedding-zero
```

> To our knowledge, HoLo-ZeRo is the first **public** Transformer input substrate with **zero learned
> input parameters** that nevertheless runs on sequences **shorter than raw byte-position models** —
> reversing the sequence-length explosion of prior embedding-free byte models via K-byte packing over
> deterministic signal-density features.

**Prior-art boundary (the claim is the combination, not any single property):**
- *Not* the first embedding-free byte model — fixed one-hot byte inputs were done in
  [Shaham & Levy, NAACL 2021](https://aclanthology.org/2021.naacl-main.17/) (but at raw byte-position length).
- *Not* the first tokenizer-free / byte-patching model — [ByT5](https://arxiv.org/abs/2105.13626),
  [Charformer](https://arxiv.org/abs/2106.12672), [MEGABYTE](https://arxiv.org/abs/2305.07185),
  [BLT](https://arxiv.org/abs/2412.09871) all shorten byte sequences (but with *learned* embeddings /
  downsamplers / patch encoders).
- **The novelty is the combination:** zero learned input parameters **+** K-byte-packed signal substrate
  **+** shorter-than-byte-position input. No prior work occupies both halves at once. This is a
  **deployable proof-of-concept** (`pip install` above), not a production or superiority claim.

```python
import torch
from hsl_embedding_zero import ZeroInput

door = ZeroInput(K=8, dim=512)        # 0 learned parameters, no tokenizer, no vocab
slots = door(byte_ids)                # [B, L] bytes -> [B, L//8, 512] attention slots
stream = door.stream(byte_ids)        # per-byte path for AR output streams
```

## The idea

Raw bytes fed directly into a transformer are known to fail — that is **why** learned
embeddings exist: something has to lift discrete symbols into a usable geometry.

This package tests the alternative: let a **frozen signal representation** do that lifting.
The [HSL substrate](https://github.com/Woojiggun/hsl-embedding) (MIT, `pip install
hsl-embedding`) maps every byte to 27 exact channels — change-rate (Gray-code Δ), Δ²,
boundary, an exact 8-point Fourier transform, and phase — grounded in a lossless codec.
If that representation already does the embedding's job, the learned front door should be
removable:

```
bytes → HSL features (frozen 4.6 KB LUT) → fixed zero-pad → transformer
```

Channels enter **unmixed** — every feature keeps a fixed address (dim 0–7 is always Δ,
17–24 always Fourier, …). The first learned combination happens inside attention, where it
is trainable and inspectable, not at the door where it would be blind.

## Measured (the table is the claim)

Same lean decoder body (dim 512 / 8 layers-class), same 3-modality byte mix
(text / video windows / audio-caption windows), same fixed 3000-step budget, same seed.
Capacity-matched arms via `hsl_embedding.ablation`. Lower bits/byte = better.

| input front door | text bpb | caption bpb | audio→caption binding gap | learned input params |
|---|---|---|---|---|
| **zero (this package)** | **2.483** | **1.503** | **+0.063** | **0** |
| learned projection on HSL features | 2.457 | 1.329 | +0.057 | ~125k |
| plain learned byte embedding (standard) | 2.848 | 2.532 | +0.080 | ~132k |

- **It works.** With nothing to train at the door, the model trains normally and lands
  within 1% of a learned input projection — the signal already carries what the learned
  door would otherwise have to learn.
- At equal budget, the standard learned-byte-embedding arm measured 2.848 text bpb; the
  substrate path reaches 2.483 with zero trained input parameters. We read this not as a
  victory over embeddings but as a **possibility**: the lifting that embeddings are trained
  to do can come from an exact, frozen signal description instead.
- **Binding gap** = extra caption bits/byte when the in-window audio is swapped for a wrong
  one (cross-modal grounding measure). The zero door preserves it.

**Sequence halving holds quality.** With K=16 (16 bytes per attention slot — half the
prefix positions, attention cost /4 on the input side):

| K=16 front door | text bpb | caption bpb | binding gap |
|---|---|---|---|
| **zero** | 2.4815 | **1.4965** | **+0.042** |
| learned projection | 2.4650 | 1.5398 | +0.031 |

At K=16 the two doors are interchangeable on every metric (text within 0.7%) — and the
zero door takes K up to 18 at dim 512 **without adding a single parameter**, while a
learned projection grows with K. *(Trade-off, honestly: binding softens for both doors at
K=16 vs K=8 — fine-grained cross-modal alignment prefers smaller slots.)*

## The point

Didn't we all want this direction — more capability per FLOP and per watt, not less?
A byte front end with **nothing to train, nothing to store beyond a 4.6 KB table, no
tokenizer pass, and sequence density as a free knob** is one concrete step that way.
This is not a claim that embeddings are beaten; it is a measured demonstration that
**a possibility now exists**, small enough for anyone to verify on one consumer GPU.

### Honest limits

Fixed small budget (3000 steps), lean ~25M body, one consumer GPU; seed-0 table (multi-seed
run in progress — numbers will be appended, not replaced). A learned embedding may close the
gap with a longer schedule. The claim is **not** "embeddings are obsolete"; it is: *on this
substrate, the learned front door is measurably unnecessary, and a standard learned byte
embedding does not reach the substrate's quality at equal budget.* Reproduce or refute:
the ablation kit ships in `hsl_embedding.ablation` (hsl / learned / random / permuted,
capacity-matched).

## Why this matters

- **0 learned input parameters** — vs ~38M for a GPT-2-class token embedding table.
- **No tokenizer** — any modality that is bytes (text, audio, raster, video windows) goes
  through the same door; this is the input layer of the byte-native multimodal
  [HoLo](https://github.com/Woojiggun/holo-hsl) line of work (59M, 3-stage curriculum,
  [weights public](https://huggingface.co/ggunio/HoLo-6.5.1)).
- **Deterministic & inspectable** — the representation cannot drift, leak, or overfit;
  what enters the model is an exact, invertible signal description.
- **K is free** — packing density (sequence length vs slot width) becomes a pure
  architecture knob, not a new parameter budget.

## API

| call | shape | learned params |
|---|---|---|
| `ZeroInput(K, dim)(ids)` | `[B, L] → [B, L//K, dim]` | 0 |
| `ZeroInput(K, dim).stream(ids)` | `[B, L] → [B, L, dim]` | 0 |
| `ZeroInput(K, dim).features(ids)` | `[B, L] → [B, L, 27]` (raw substrate) | 0 |
| `zero_input(b"raw bytes")` | one-call convenience | 0 |

## Cite

Software (this package, all versions): DOI [10.5281/zenodo.20643551](https://doi.org/10.5281/zenodo.20643551).
Companion paper: Jinhyun Woo, *A Feasibility Study of Change-Rate-Based Multimodal Unification* —
DOI [10.5281/zenodo.20581805](https://doi.org/10.5281/zenodo.20581805). The substrate itself is
citable separately: [10.5281/zenodo.20628599](https://doi.org/10.5281/zenodo.20628599)
(`hsl-embedding`).

MIT © 2026 Jinhyun Woo — independent research, released on personal time.
