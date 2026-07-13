from types import SimpleNamespace

import pytest
import torch

from maxent_irl_maps.utils import get_state_visitations


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
    ],
)
def test_state_visitations_masks_points_outside_rounded_grid(device):
    metadata = SimpleNamespace(
        origin=torch.tensor([[0.0, 0.0]], device=device),
        length=torch.tensor([[2.1, 2.1]], device=device),
        resolution=torch.tensor([[1.0, 1.0]], device=device),
        N=torch.tensor([[2, 2]], device=device),
    )
    trajs = torch.tensor([[[0.5, 0.5], [2.05, 0.5]]], device=device)

    visitations = get_state_visitations(trajs, metadata)
    if device == "cuda":
        torch.cuda.synchronize()

    assert visitations.shape == (1, 2, 2)
    assert torch.isfinite(visitations).all()
    assert visitations[0, 0, 0] == 1
    assert visitations.sum() == 1
