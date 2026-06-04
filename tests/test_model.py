import torch

from superres.model import build_model, count_params
from superres.losses import charbonnier


def test_forward_shape_preserved():
    m = build_model({"base_width": 8, "levels": 2, "blocks_per_level": 1})
    x = torch.randn(2, 1, 32, 32, 32)
    y = m(x)
    assert y.shape == x.shape  # x1 restoration: same grid


def test_residual_identity_at_init():
    # zero-init head + residual => exact identity before any training
    m = build_model({"base_width": 8, "levels": 2, "residual": True})
    x = torch.randn(1, 1, 24, 24, 24)
    with torch.no_grad():
        y = m(x)
    assert torch.allclose(y, x, atol=1e-6)


def test_non_residual_not_identity():
    m = build_model({"base_width": 8, "levels": 2, "residual": False})
    x = torch.randn(1, 1, 24, 24, 24)
    with torch.no_grad():
        y = m(x)
    assert not torch.allclose(y, x)


def test_backward_step_runs():
    m = build_model({"base_width": 8, "levels": 2, "blocks_per_level": 1})
    x = torch.randn(2, 1, 16, 16, 16)
    target = torch.randn(2, 1, 16, 16, 16)
    loss = charbonnier(m(x), target)
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)


def test_param_count_reasonable():
    m = build_model({"base_width": 32, "levels": 3, "blocks_per_level": 2})
    n = count_params(m)
    assert 5_000_000 < n < 40_000_000  # ~ design target order of magnitude


def test_odd_size_handled():
    # Up-path padding should handle non-power-of-two patches
    m = build_model({"base_width": 8, "levels": 2})
    x = torch.randn(1, 1, 20, 20, 20)
    y = m(x)
    assert y.shape == x.shape
