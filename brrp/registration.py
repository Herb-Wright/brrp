import torch
import open3d as o3d
import numpy as np


def ransac_registration(
    source: torch.Tensor, 
    target: torch.Tensor, 
    *, 
    voxel_size: float = 0.01,
    distance_threshold: float = 0.0075,
    with_scale: bool = False
) -> tuple[torch.Tensor, torch.Tensor, float]:
    dst = convert_to_open3d_pointcloud(target)
    dst_down, dst_fpfh = preprocess_point_cloud(dst, voxel_size)
    src = convert_to_open3d_pointcloud(source)
    src_down, src_fpfh = preprocess_point_cloud(src, voxel_size)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down,
        dst_down,
        src_fpfh,
        dst_fpfh,
        mutual_filter=False,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.
        TransformationEstimationPointToPoint(with_scale),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.
            CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            100000, 0.99),
    )
    new_src = src.transform(result.transformation)
    src_tensor = torch.from_numpy(np.asarray(new_src.points))
    return torch.tensor(result.transformation), src_tensor, len(result.correspondence_set)


def preprocess_point_cloud(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0,
                                             max_nn=30))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0,
                                             max_nn=100))
    return (pcd_down, pcd_fpfh)


def convert_to_open3d_pointcloud(tensor: torch.Tensor) -> o3d.geometry.PointCloud:
    """Converts a PyTorch tensor to an Open3D PointCloud object."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(tensor.cpu().numpy())
    return pcd



