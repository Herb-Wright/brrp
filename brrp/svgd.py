import math
from typing import Protocol

import torch
from torch import Tensor
from torch.nn import Module



class Kernel(Protocol):
    """Protocol for kernels"""
    def __call__(self, x: Tensor, y: Tensor) -> Tensor:
        """(N, D), (M, D) -> (N, M)"""
        pass


class GaussianMedianKernel(Module):
    """Gaussian kernel with median scaling as described in original SVGD paper (Liu and Wang 2016)"""
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        m = y.shape[0]
        dists = torch.cdist(x, y, p=2)  # (N, M)
        median = torch.median(dists)
        h_inv =  math.log(m + 1e-8) / (median ** 2)
        out = torch.exp(- h_inv * (dists ** 2))
        return out


def get_svgd_phi(particles: Tensor, ln_probs_particles: Tensor, kernel: Kernel) -> Tensor:
    """Calculates $\hat{\phi}^*(x)$ as described in SVGD paper (Liu and Wang 2016)
    
    Adapted from https://github.com/activatedgeek/svgd/blob/master/svgd.py
    
    Args:
        particles: (N, D)
        ln_probs_particles: (N,)
        kernel: (Kernel)
    
    Returns:
        phi_star: (N, D)
        ln_prob: (N,)
    """
    num_particles = particles.shape[0]
    ln_prob_grad = torch.autograd.grad(torch.sum(ln_probs_particles), particles)[0]  # grad of ln prob w.r.t. particles
    particles = particles.detach().requires_grad_(True)  # (N, D)
    print("ln_prob_grad", ln_prob_grad.shape, ln_prob_grad.mean())
    print("particles", particles.shape, particles.mean())
    gram_mat = kernel(
        particles.reshape((num_particles, -1)), 
        particles.reshape((num_particles, -1)).detach()
    )  # (N, N)
    print("gram_mat", gram_mat.shape, gram_mat.mean())
    kernel_grad = - torch.autograd.grad(torch.sum(gram_mat), particles)[0]
    phi_star = (gram_mat @ ln_prob_grad.reshape((num_particles, -1))).reshape(ln_prob_grad.shape) + kernel_grad  # (N, D)
    return phi_star, ln_probs_particles.detach()


