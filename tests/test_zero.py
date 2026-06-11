import sys, torch
sys.path.insert(0, r"G:\내 드라이브\hsl-embedding-zero\src")
from hsl_embedding_zero import ZeroInput, zero_input, FEAT_DIM
import hsl_embedding as hsl

door = ZeroInput(K=8, dim=512)
ids = torch.tensor([list(b"hello world, byte signal" * 4)], dtype=torch.long)
out = door(ids)
assert out.shape == (1, ids.shape[1] // 8, 512), out.shape
st = door.stream(ids)
assert st.shape == (1, ids.shape[1], 512)
f = door.features(ids)
ref, _ = hsl.embed(bytes(ids[0].tolist()))
assert torch.equal(f[0], ref), "zero door must be BIT-IDENTICAL to hsl.embed"
assert out[0, 0, 8*FEAT_DIM:].abs().sum() == 0, "pad region must be zero"
assert sum(p.numel() for p in door.parameters() if p.requires_grad) == 0, "must have 0 learned params"
z = zero_input(b"0123456789abcdef")
assert z.shape == (1, 2, 512)
try:
    ZeroInput(K=19, dim=512); raise SystemExit("FAIL: K=19 should raise")
except ValueError: pass
print("ALL GREEN: shapes, bit-identity to hsl.embed, zero-pad, 0 learned params, K guard")
