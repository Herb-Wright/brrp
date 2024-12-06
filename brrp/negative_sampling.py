# this file is adapted from https://github.com/Herb-Wright/v-prism/blob/main/src/v_prism/data_loading/negative_sampling.py 

import torch
import open3d as o3d
import numpy as np

from .subsample import grid_subsample_different_res


def brrp_negative_sample(
    points: torch.Tensor,
    seg_mask: torch.Tensor,
    object_sphere_radius: float = 0.2,
    *,
    ray_step_size: float = 0.2,
    camera_pos: torch.Tensor | None = None,
    scene_sphere_radius: float = 0.6,
    subsample_grid_size_unocc: float = 0.015,
    subsample_grid_size_occ: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """performs negative sampling for the B2RP method"""
    scene_center = 0.5 * (
        torch.amax(points[seg_mask > 0], dim=0) 
        + torch.amin(points[seg_mask > 0], dim=0)
    )
    X, y = negative_sample_rays_in_sphere_stratified_sample_multiclass(
        points=points, 
        mask=seg_mask, 
        step_size=ray_step_size, 
        radius=object_sphere_radius, 
        camera_pos=camera_pos,
    )
    plane = run_RANSAC_for_plane(points, seg_mask, scene_sphere_radius, 0.01, 1000)
    X, y = add_negative_points_below_plane_multiple_objects(
        X, y, plane=plane, center=scene_center, radius=object_sphere_radius, k=50_000
    )
    X, y = grid_subsample_different_res(
        X, 
        y, 
        subsample_grid_size_unocc=subsample_grid_size_unocc, 
        subsample_grid_size_occ=subsample_grid_size_occ
    )
    return X, y


def negative_sample_rays_in_sphere_stratified_sample_multiclass(
    points: torch.Tensor,
    mask: torch.Tensor,
    step_size: float,
    radius: float,
    *,
    camera_pos: torch.Tensor | None = None,
    max_points: int = 1_000_000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """negatively samples points along rays within a sphere around the object

    Args:
        points: (P', 3) the pointcloud
        mask: (P',) the integer mask of the objects where 0 is background and >0 is object id 
        step_size: (float) the step size to step forward
        radius: (float) radius of the ball around the centroid of the points within the mask
        camera_pos (optional): (3,) position of the camera. If None, defaults to [0,0,0]

    Returns:
        (X, y): (P, 3) points and (P,) labels
    """
    assert mask.device == points.device, "mask and points must be on the same device"
    dtype = points.dtype
    device = points.device
    if camera_pos is None:
        camera_pos = torch.zeros(3, device=device, dtype=dtype)
    if torch.sum(mask > 0) <= 0:
        raise Exception("must have more than one point in the mask to be part of an object")
    obj_points = points[mask > 0]
    K = len(torch.unique(mask[mask > 0]))
    centers = []
    for i in torch.unique(mask[mask > 0]):
        centers.append(0.5 * (torch.amin(points[mask == i], dim=0) + torch.amax(points[mask == i], dim=0)))
    centroids = torch.stack(centers) # (K, 3)
    centroid_dist = torch.linalg.norm(centroids - camera_pos, dim=1)  # (K,)
    min_dist = torch.amin(centroid_dist) - radius
    max_dist = torch.amax(centroid_dist) + radius
    ray_vects =  points - camera_pos
    depth: torch.Tensor = torch.linalg.norm(ray_vects, axis=1)
    rays: torch.Tensor = ray_vects / depth.reshape((-1, 1))
    P_poss = rays.shape[0]
    dists = torch.arange(min_dist, max_dist, step=step_size, device=device, dtype=dtype)  # (R,)
    R = dists.shape[0]
    dists_noise = dists.reshape((-1, 1)) + torch.rand((R, P_poss), device=device, dtype=dtype) * step_size
    possible_neg_points = camera_pos + dists_noise.reshape((R, P_poss, 1)) * rays  # (R, P~, 3)
    depth_mask = dists_noise < depth
    dist_mask = torch.amin(torch.linalg.norm(possible_neg_points - centroids.reshape((K, 1, 1, 3)), dim=3), dim=0) <= radius
    mask2 = torch.logical_and(depth_mask, dist_mask)
    negative_points = possible_neg_points[mask2]
    P_neg = negative_points.shape[0]
    if P_neg > max_points:
        idxs = torch.randperm(P_neg, device=device)[:max_points]
        negative_points = negative_points[idxs]
    X = torch.concatenate([obj_points, negative_points], axis=0)
    y = torch.concatenate([
        mask[mask > 0],  # object id
        torch.zeros_like(negative_points[:, 0]),  # 0
    ])
    return X, y

def run_RANSAC_for_plane(points, seg_mask, radius, dist_thresh, n_iters):
    pts_fg = points[seg_mask > 0]
    scene_center = 0.5 * (torch.amax(pts_fg, dim=0) + torch.amin(pts_fg, dim=0))
    dists_from_center = torch.norm(points - scene_center, dim=1)
    filtered_points = points[torch.logical_and(dists_from_center <= radius, seg_mask == 0)]
    f_points_np = filtered_points.cpu().numpy()
    o3d_point_cloud = o3d.geometry.PointCloud()
    o3d_point_cloud.points = o3d.utility.Vector3dVector(f_points_np)
    plane_model, inliers = o3d_point_cloud.segment_plane(
        distance_threshold=dist_thresh,
        ransac_n=3,
        num_iterations=n_iters
    )
    [a, b, c, d] = plane_model  # bias is d; normal is [a, b, c]
    center_dist_from_plane_sgn = scene_center[0]*a + scene_center[1]*b + scene_center[2]*c + d
    if center_dist_from_plane_sgn < 0:
        plane_model = [-a, -b, -c, -d]
    return plane_model

def add_negative_points_below_plane_multiple_objects(
    X: torch.Tensor, 
    y: torch.Tensor, 
    plane: list[float],
    center: torch.Tensor, 
    radius: float,
    k: int = 100_000
) -> tuple[torch.Tensor, torch.Tensor]:
    """adds points below plane in sphere around each object

    Args:
        - X: (P', 3) point cloud
        - y: (P',) segmentation labels for point cloud
        - plane: (list[float]) specifies the plane plane[3] = plane[:3]^T x
        - center: (3,) scene center
        - radius: (float) radius from object center to keep points from
        - k: (int) number of potential samples

    Returns: (tuple)
        - X_new: (P, 3)
        - y_new: (P,)
    """
    centers = []
    for i in torch.unique(y[y > 0]):
        centers.append(0.5 * (torch.amin(X[y == i], dim=0) + torch.amax(X[y == i], dim=0)))
    centroids = torch.stack(centers) # (N, 3)
    N, D = centroids.shape
    unit_samples = 2 * torch.rand(size=(k, len(center)), dtype=X.dtype, device=X.device) - 1
    mask1 = torch.sum(unit_samples ** 2, axis=1) <= 1
    sphere_samples = (centroids.reshape((N, 1, D)) + radius * unit_samples[mask1]).reshape((-1, D))
    normal = torch.tensor([plane[0], plane[1], plane[2]], dtype=X.dtype, device=X.device)
    mask_plane = (sphere_samples @ normal) + plane[3] < 0
    if torch.sum(mask_plane) == 0:
        return X, y  # early exit if no negative points found
    samples = sphere_samples[mask_plane]
    X_new = torch.concatenate([X, samples], axis=0)
    y_new = torch.concatenate([y, torch.zeros_like(samples[:, 0])], axis=0)
    return X_new, y_new
