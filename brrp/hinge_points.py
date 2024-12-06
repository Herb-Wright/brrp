import torch
from torch import Tensor
import numpy as np



class GaussianHingePointTransform:
    """Hinge point transform with Gaussian kernel"""
    
    def __init__(self, n_side: int, scale: float, centers: Tensor | None = None):
        """creates a GaussianHingePointTransform

        Creates a cube of evenly spaced hinge points and a corresponding
        transform. The kernel is Gaussian and bandwith is decided based
        on distance between hinge points. A bias term is included

        Args:
            n_side (int): the number of hinge points along each side. There will 
                be a total of n_side^3 hinge points
            scale (float): the bounds of the grid (from [-scale, scale])
            center (Tensor): shape of (N, 3) the centers/offset of hinge point grid
        """
        if centers is None:
            centers = torch.zeros(1, 3)

        assert n_side > 1, "must have at least 2 hinge points on each side"
        assert scale > 0.0
        assert centers.shape[1] == 3

        self.n_side = n_side
        self.scale = scale
        self.center = centers
        self.grid_len = (2 * scale) / (n_side - 1)
        self.gamma = 1 / (self.grid_len ** 2)
        self.dim = (n_side ** 3) + 1
        hinge_points_pre = generate_hingepoint_grid(
            torch.zeros(3) - scale,
            torch.zeros(3) + scale + 0.5 * self.grid_len,
            resolution=self.grid_len,
            dtype=torch.float32
        )  # (H, 3)
        self.n = centers.shape[0]
        self.hinge_points = centers.unsqueeze(1) + hinge_points_pre.to(centers.device)

    def transform(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (N, P, 3)

        Returns:
            features: (N, P, H)
        """
        assert x.shape[0] == self.n or x.shape[0] == 1 
        if x.shape[0] == 1:
            x = x.expand((self.n, *x.shape[1:]))
        feats = get_gaussian_kernel(self.gamma, x, self.hinge_points)
        feats = feats * (feats >= 1e-7)  # IMPORTANT FOR NUMERICAL REASONS
        return torch.concatenate([feats, torch.ones_like(x[..., 0:1])], dim=-1)

    def move_to_device(self, device: torch.device) -> None:
        self.hinge_points = self.hinge_points.to(device)
        self.center = self.center.to(device)


def generate_hingepoint_grid(
    min: list[float], 
    max: list[float], 
    resolution: float,
    *,
    dtype: torch.dtype | None = None
) -> Tensor:
    """generates hingepoints in a grid
    
    Args:
        min (list[float]): a list of D floats
        max (list[float]): a list of D floats
        resolution (float): the grid length
        dtype (optional): dtype for hinge points

    Returns:
        (H, D) the hinge points as a torch.Tensor
    """
    assert len(max) == len(min)
    D = len(max)
    ranges = [np.arange(min[i], max[i], step=resolution) for i in range(D)]
    grid: Tensor = torch.tensor(np.stack(np.meshgrid(*ranges), axis=-1))
    if dtype is not None:
        grid = grid.to(dtype)
    return grid.reshape((-1, D))

def _square_dists(x: Tensor, y: Tensor) -> Tensor:
    square_dists = torch.sum(x ** 2, dim=-1, keepdim=True) + torch.sum(y ** 2, dim=-1).unsqueeze(-2) - 2 * (x @ y.transpose(-1, -2))
    return torch.maximum(square_dists, torch.zeros(1, dtype=x.dtype, device=x.device))  # for numerical stability w sqrt

def get_gaussian_kernel(gamma: float, x: Tensor, y: Tensor) -> Tensor:
    """computes k(x, y) for a Gaussian kernel with desired gamma"""
    return torch.exp(- gamma * _square_dists(x, y))


