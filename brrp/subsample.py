# This file is adapted from https://github.com/Herb-Wright/v-prism/blob/main/src/v_prism/utils/subsample.py
import torch
from torch_geometric.nn import voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import pool_pos, pool_batch



def grid_subsample(
    points: torch.Tensor, 
    batch: torch.Tensor, 
    subsample_grid_size: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    performs grid subsampling on points in packed format with a batch index tensor.
    inspired by various methods from the torch_points3d package
    
    Args:
        points (P, 3): points in packed format
        batch (P,): batch indices
        subsample_grid_size (float): grid size used
    
    Returns:
        sampled_points (P', 3)
        sampled_batch (P'): batch indices
    """
    cluster_ids = voxel_grid(points, subsample_grid_size, batch=batch)  # (P,)
    cluster_idxs, unique_pos_indices = consecutive_cluster(cluster_ids)
    sampled_points = pool_pos(cluster_idxs, points)
    sampled_batch = pool_batch(unique_pos_indices, batch)
    return sampled_points, sampled_batch


def grid_subsample_different_res(
    points: torch.Tensor, 
    batch: torch.Tensor, 
    subsample_grid_size_unocc: float,
    subsample_grid_size_occ: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """grid subsampling used in our method, where different resolution is used for different labels

    Args:
        points: (P', 3) point cloud
        batch: (P',) labels corresponding to points
        subsample_grid_size_unocc: (float) grid size for unoccupied points
        subsample_grid_size_occ: (float) grid size for surface/occupied points
    
    Returns:
        (new_points, new_batch): (P, 3) and (P,)
    """
    empty_points = points[batch == 0]
    empty_batch = batch[batch == 0]
    surface_points = points[batch > 0]
    surface_points_batch = batch[batch > 0]
    new_empty_points, new_empty_batch = grid_subsample(empty_points, empty_batch, subsample_grid_size_unocc)
    new_surface_points, new_surface_batch = grid_subsample(surface_points, surface_points_batch, subsample_grid_size_occ)
    new_points = torch.concat([new_empty_points, new_surface_points], dim=0)
    new_batch = torch.concat([new_empty_batch, new_surface_batch], dim=0)
    return new_points, new_batch

