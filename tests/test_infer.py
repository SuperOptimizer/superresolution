import numpy as np
import torch

from superres.infer import infer_volume, _cosine_window, _starts


class Identity(torch.nn.Module):
    def forward(self, x):
        return x


def test_starts_cover_volume():
    s = _starts(100, 32, 8)
    assert s[0] == 0
    assert s[-1] == 100 - 32  # last window flush to the end
    # consecutive windows overlap by at least `overlap`
    for a, b in zip(s, s[1:]):
        assert b - a <= 32 - 8 + 1


def test_cosine_window_partition_of_unity_uniform_overlap():
    # On a uniformly-tiled axis (no flush-end snapping), complementary cosine
    # ramps offset by `overlap` should sum to ~constant in the interior.
    win, ov = 32, 8
    step = win - ov
    n = win + 3 * step  # exact multiple => uniform overlap everywhere
    blend = _cosine_window((win, 1, 1), ov)[:, 0, 0]
    acc = np.zeros(n)
    for z0 in range(0, n - win + 1, step):
        acc[z0 : z0 + win] += blend
    interior = acc[ov : n - ov]
    assert np.allclose(interior, interior.mean(), rtol=0.05)


def test_cosine_window_strictly_positive():
    # floor keeps weights > 0 so acc/wacc never divides by zero at volume edges
    blend = _cosine_window((16, 16, 16), 4)
    assert blend.min() > 0.0


def test_identity_model_reconstructs_volume_no_seams():
    rng = np.random.default_rng(0)
    vol = rng.random((40, 40, 40)).astype(np.float32)
    out = infer_volume(Identity(), vol, window=(24, 24, 24), overlap=8,
                       occupancy_skip=False, amp_bf16=False)
    assert out.shape == vol.shape
    # identity through the tiler must reconstruct to high precision (no seams)
    assert np.max(np.abs(out - vol)) < 1e-4


def test_occupancy_skip_passes_air_through():
    vol = np.zeros((32, 32, 32), dtype=np.float32)
    vol[:8, :8, :8] = 0.5  # one occupied corner

    class AddOne(torch.nn.Module):
        def forward(self, x):
            return x + 1.0

    out = infer_volume(AddOne(), vol, window=(16, 16, 16), overlap=4,
                       occupancy_skip=True, amp_bf16=False)
    # all-air tiles skipped -> remain 0; occupied region changed
    assert out[24:, 24:, 24:].max() == 0.0
    assert out[:8, :8, :8].max() > 0.0
