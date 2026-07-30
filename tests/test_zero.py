import pytest
import torch

from hsl_embedding_zero import FEAT_DIM, ZeroInput, zero_input
import hsl_embedding as hsl


def test_shapes_and_bit_identity():
    door = ZeroInput(K=8, dim=512)
    ids = torch.tensor([list(b"hello world, byte signal" * 4)], dtype=torch.long)

    out = door(ids)
    stream = door.stream(ids)
    features = door.features(ids)
    reference, _ = hsl.embed(bytes(ids[0].tolist()))

    assert out.shape == (1, ids.shape[1] // 8, 512)
    assert stream.shape == (1, ids.shape[1], 512)
    assert torch.equal(features[0], reference)
    assert out[0, 0, 8 * FEAT_DIM :].abs().sum() == 0


def test_zero_learned_parameters_and_convenience_api():
    door = ZeroInput(K=8, dim=512)

    assert sum(p.numel() for p in door.parameters() if p.requires_grad) == 0
    assert zero_input(b"0123456789abcdef").shape == (1, 2, 512)


def test_invalid_k_raises():
    with pytest.raises(ValueError):
        ZeroInput(K=19, dim=512)


def test_tail_handling_keeps_bytes_by_default():
    ids = torch.arange(100, dtype=torch.long).unsqueeze(0)

    assert ZeroInput(K=8, dim=512)(ids).shape == (1, 13, 512)
    assert ZeroInput(K=8, dim=512, tail="drop")(ids).shape == (1, 12, 512)


def test_invalid_tail_raises():
    with pytest.raises(ValueError):
        ZeroInput(tail="oops")
