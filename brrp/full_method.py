import os
import logging
import json

import torch
import numpy as np

from .hinge_points import GaussianHingePointTransform
from .svgd import GaussianMedianKernel, get_svgd_phi
from .utils import abspath
from .clip import crop_and_resize, run_clip
from .registration import ransac_registration
from .negative_sampling import brrp_negative_sample


def full_brrp_method(
    rgb: torch.Tensor,
    xyz: torch.Tensor,
    mask: torch.Tensor,
    prior_path: str,
    *,
    device_str: str = "cpu",
    global_scale: float = 0.12,  # 12cm
    n_hinge_points_each_side: int = 12,
    camera_pos: torch.Tensor = None
) -> tuple[torch.Tensor, GaussianHingePointTransform]:

    device = torch.device(device_str)
    if camera_pos is None:
        camera_pos = torch.zeros(3, device=device)
    xyz[torch.isnan(xyz[:, :, 0])] = camera_pos
    n_objects = torch.amax(mask)

    logging.debug("loading prior classes")
    with open(abspath(os.path.join(prior_path, "classes.json")), "r") as f:
        classes = json.load(f)

    logging.debug("running CLIP")
    imgs_for_clip = crop_and_resize(rgb.permute(2, 0, 1), mask)
    probs = run_clip(imgs_for_clip, classes=classes, device=device_str)  # (4, 50)
    top_k = torch.topk(probs, k=3)

    logging.debug("retreiving prior point clouds")
    point_clouds = torch.from_numpy(np.load(abspath(os.path.join(prior_path, "point_clouds.npy")))).to(torch.float32)
    point_clouds = point_clouds * global_scale

    logging.debug("running registrations")
    object_centers = []
    best_clouds = []
    transforms = []
    scales = []
    for (i, class_idxs) in enumerate(top_k.indices):
        object_point_cloud = xyz[mask == i+1]
        object_center = 0.5 * (torch.amax(object_point_cloud, dim=0) + torch.amin(object_point_cloud, dim=0))
        object_centers.append(object_center)
        object_point_cloud = (object_point_cloud - object_center)
        best_clouds_i = []
        transforms_i = []
        scales_i = []
        for class_idx in class_idxs:
            best_scale = None
            best_cloud = None
            best_inliers = 0.0
            best_trans = None
            for scale in [0.4, 0.5, 0.6, 0.7, 0.8, 0.90, 1.00, 1.10, 1.22, 1.34]:
                trans, new_point_cloud, n_inliers = ransac_registration(
                    point_clouds[class_idx] * scale, 
                    object_point_cloud, 
                    with_scale=False
                )
                if n_inliers > best_inliers:
                    best_inliers = n_inliers
                    best_scale = scale
                    best_cloud = new_point_cloud
                    best_trans = trans
            transforms_i.append(best_trans)
            scales_i.append(best_scale)
            best_clouds_i.append(best_cloud)
        best_clouds.append(best_clouds_i)
        transforms.append(torch.stack(transforms_i))
        scales.append(torch.tensor(scales_i))
    scales_qp = torch.stack(scales).to(torch.float32).to(device)
    trans_qp = torch.stack(transforms).to(torch.float32).to(device)
    object_centers_qp = torch.stack(object_centers).to(device)  # (N, 3)

    logging.debug("retrieving query samples")
    pts_list = []
    labels_list = []
    for (i, class_idxs) in enumerate(top_k.indices):
        pts_i = []
        labels_i = []
        for idx in class_idxs:
            pts_i.append(torch.from_numpy(
                np.load(abspath(os.path.join(prior_path, f"query_points/{idx:08d}.npy")))
            ).to(torch.float32).to(device))
            labels_i.append(torch.from_numpy(
                np.load(abspath(os.path.join(prior_path, f"sdf_values/{idx:08d}.npy")))
            ).to(torch.float32).to(device))
        pts_list.append(torch.stack(pts_i))
        labels_list.append(torch.stack(labels_i))
    prior_samples = torch.stack(pts_list)
    labels = (torch.stack(labels_list) <= 0).to(torch.float32)

    logging.debug("transforming query samples")
    prior_samples = prior_samples * global_scale * scales_qp.unsqueeze(2).unsqueeze(3)
    prior_samples = prior_samples @ trans_qp[:, :, :3, :3].permute(0, 1, 3, 2) + trans_qp[:, :, :3, 3].unsqueeze(2)
    prior_samples = prior_samples + object_centers_qp.unsqueeze(1).unsqueeze(2)  # (N, K, QP, 3)

    print("xyz shape: " + str(xyz.shape))
    print("# of background: " + str(torch.sum(mask == 0)) )
    import trimesh
    pc1 = trimesh.PointCloud(xyz[mask > 0].cpu(), colors=[255, 0, 0])
    pc2 = trimesh.PointCloud(xyz[mask == 0].cpu())
    trimesh.Scene([pc1, pc2]).show()
    # trimesh.PointCloud([[0, 0, 0], [1, 0, 0]]).show()

    logging.debug("negative sampling")
    observed_samples, observed_labels = brrp_negative_sample(
        xyz.reshape(-1, 3).to(device),
        mask.reshape(-1).to(device),
        camera_pos=camera_pos,
    )
    logging.debug(f"found samples of shape {observed_samples.shape}")

    logging.debug("doing reconstruction")
    hp_trans = GaussianHingePointTransform(
        n_hinge_points_each_side, 
        1.25 * global_scale, 
        centers=object_centers_qp.to(device)
    )
    hp_trans.move_to_device(device)
    weights = svgd_reconstruction(
        observed_samples, 
        observed_labels, 
        prior_samples, 
        labels, 
        top_k.values, 
        n_objects, 
        hp_trans
    )
    return weights, hp_trans


def svgd_reconstruction(
    observed_samples: torch.Tensor, 
    observed_labels: torch.Tensor,
    prior_samples: torch.Tensor,
    prior_labels: torch.Tensor,
    top_k_probs: torch.Tensor,
    n_objects: int,
    hp_trans: GaussianHingePointTransform,
    *,
    observed_batch_size: int = 2048,
    prior_batch_size: int = 256,
    lambda_obs: float = 50.0,
    lambda_prior: float = 1.0,
    lambda_reg: float = 0.000001,
) -> torch.Tensor:
    """performs SVGD reconstruction.

    Args:
        observed_samples: (S, 3)
        observed_labels: (S,)
        prior_samples: (N, K, Q, 3)
        prior_labels: (N, K, Q)
        top_k_probs: (N, K)
        n_objects: number of objects
        hp_trans: the hinge point feature transform
    
    Returns:
        weights_final: final weights after SVGD optimization
    """
    device = observed_samples.device
    
    n_epochs = 10
    n_particles = 8

    med_kernel = GaussianMedianKernel()

    weights = torch.randn(n_particles, n_objects, hp_trans.dim, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([weights], lr=0.1)

    for i in range(n_epochs):
        running_loss = 0.0
        n_iter = 0
        for X_b, y_b in batchify_X_y(observed_samples, observed_labels, batch_size=observed_batch_size):
            optimizer.zero_grad()
            weights_detached = weights.detach().requires_grad_(True)  # (P, N, H) Needed for SVGD
            X_b = X_b.unsqueeze(0)  # (1, BS, 3)
            y_b = (y_b == (
                1 + torch.arange(n_objects, device=device)).unsqueeze(1)
            ).to(torch.float32)  # (N, BS)
            feats = hp_trans.transform(X_b)  # (N, BS, H)
            dot_prods = torch.sum(weights_detached.unsqueeze(2) * feats, dim=-1)  # (P, N, BS)
            obs_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                dot_prods, 
                y_b.expand(dot_prods.shape), 
                reduction="none",
            )
            obs_loss = torch.sum(torch.mean(obs_loss, dim=-1), dim=-1)  # (P,)
            query_idx = torch.randperm(prior_samples.shape[2], device=device)[:prior_batch_size]
            X_q = prior_samples[:, :, query_idx]  # (N, K, BQ, 3)
            y_q = prior_labels[:, :, query_idx]  # (N, K, BQ)
            feats_q = hp_trans.transform(
                X_q.reshape(n_objects, -1, 3)
            ).reshape(n_objects, X_q.shape[1], X_q.shape[2], -1)  # (N, K, BQ, H)
            dot_prods = torch.sum(
                weights_detached.unsqueeze(2).unsqueeze(3) * feats_q, 
                dim=-1
            )  # (P, N, K, BQ)
            query_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                dot_prods, 
                y_q.expand(dot_prods.shape), 
                reduction="none",
            )  # (P, N, K, BQ)
            query_loss = torch.mean(query_loss, dim=-1)  # (P, N, K)
            query_loss = -torch.log(torch.sum(
                torch.exp(-query_loss) * top_k_probs, 
                dim=(1, 2)
            ))  # (P,) Mixture
            norms = torch.mean(torch.norm(weights_detached, dim=-1), dim=-1)  # (P,)
            ln_probs = lambda_obs * obs_loss + lambda_prior * query_loss + lambda_reg * norms

            phi_star, ln_probs = get_svgd_phi(weights_detached, -ln_probs, kernel=med_kernel)
            weights.grad = -phi_star
            optimizer.step()
            running_loss += torch.mean(ln_probs).item()
            n_iter += 1
        logging.debug(f"epoch {i} had avg loss {running_loss / n_iter} ({n_iter} iter)")
    return weights.detach()



def batchify_X_y(X: torch.Tensor, y: torch.Tensor, batch_size: int):
    """Generator that loops through data for specified batch size.
    
    Args:
        X: (N, ...)
        y: (N, ...)
        batch_size: the batch size, denoted B

    Yields:
        (X, y): tuple of two torch.Tensor objects with shapes (B, ...) and (B, ...)
    """
    N = X.shape[0]
    assert y.shape[0] == N, "X, y need to have the same number of points"
    idxs = torch.randperm(N, device=X.device)
    curr = 0
    while curr < N:
        idxs_i = idxs[curr:curr+batch_size]
        yield X[idxs_i], y[idxs_i]
        curr += batch_size


