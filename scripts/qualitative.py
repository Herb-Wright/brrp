"""
need to pip install v_prism for this one.

```sh
pip install https://github.com/Herb-Wright/v-prism.git
```
"""
import logging
import os

import numpy as np
import torch
from PIL import Image
import trimesh
from v_prism.utils.pointsdf import PointSDF, scale_and_center_object_points
from v_prism.utils.pointsdf import scale_and_center_queries, index_points
from v_prism.utils.pointsdf import  farthest_point_sample
from v_prism import full_VPRISM_method

from brrp.utils import abspath, mkdir_if_not_exists, setup_logger
from brrp.visualization import gen_image_of_trimesh_scene, gen_mesh_for_sdf_batch_3d, some_colors
from brrp.subsample import grid_subsample
from brrp.negative_sampling import run_RANSAC_for_plane
from brrp.full_method import full_brrp_method


setup_logger()

device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)


scenes = [
    abspath("~/data/brrp_real_world_scenes/0000"),
    abspath("~/data/brrp_real_world_scenes/0001"),
    abspath("~/data/brrp_real_world_scenes/0002"),
    abspath("~/data/brrp_real_world_scenes/0003"),
    abspath("~/data/brrp_real_world_scenes/0004"),
    abspath("~/data/brrp_real_world_scenes/0005"),
]


def main():
    logging.info(f"in main of qualitative1.py")

    out_dir = mkdir_if_not_exists(abspath(os.path.join(os.path.dirname(__file__), "../out/qual_figs")))

    for i, scene_dir in enumerate(scenes):
        logging.info(f"doing scene ({i}) {scene_dir}")

        colors_pre = some_colors * 4  # make list repeated
        colors = torch.tensor([[0, 0, 0]] + colors_pre[3*i:]) / 255

        rgb_img = Image.open(os.path.join(scene_dir, "rgb.jpg"))
        rgb_np = np.array(rgb_img)
        xyz_np = np.load(os.path.join(scene_dir, "xyz.npy"))
        rgb = torch.from_numpy(rgb_np) / 255
        xyz = torch.from_numpy(xyz_np)
        mask_np = np.load(os.path.join(scene_dir, "segmentation.npy"))
        mask = torch.from_numpy(mask_np).to(torch.int64)
        n_objects = torch.amax(mask)
        scene_center = 0.5 * (torch.amin(xyz[mask > 0], dim=0) + torch.amax(xyz[mask > 0], dim=0))

        # save rgb
        rgb_img.save(os.path.join(out_dir, f"rgb_{i}.jpg"))

        # save point cloud
        scene_center = 0.5 * (torch.amin(xyz[mask > 0], dim=0) + torch.amax(xyz[mask > 0], dim=0))
        point_cloud = xyz[torch.norm(xyz - scene_center, dim=-1) < 0.6]
        seg = mask[torch.norm(xyz - scene_center, dim=-1) < 0.6]
        x, y = grid_subsample(point_cloud, seg, subsample_grid_size=0.01)
        seg_colors = colors[y]
        tm_pc = trimesh.PointCloud(x, colors=seg_colors)
        img = gen_image_of_trimesh_scene(
            trimesh.Scene([tm_pc]), 
            theta=0.3,
            phi=0.6,
            pi=3.3, 
            lookat_position=scene_center, 
            rotate=False, 
            rho=0.8,
            resolution=(800, 600),
            line_settings={'point_size': 10}
        )
        img.save(os.path.join(out_dir, f"point_cloud_{i}.png"))

        # save PointSDF?
        psdf_img = reconstruct_scene_pointsdf(scene_dir, colors)
        psdf_img.save(os.path.join(out_dir, f"pointsdf_{i}.png"))

        # save V-PRISM?
        logging.info("V-PRISM reconstruct")
        vprism_img = reconstruct_vprism(scene_dir, colors)
        vprism_img.save(os.path.join(out_dir, f"vprism_{i}.png"))

        # save BRRP
        logging.info("BRRP reconstruction")
        brrp_img = reconstruct_brrp(scene_dir, colors)
        brrp_img.save(os.path.join(out_dir, f"brrp_{i}.png"))




scene_sphere_radius = 0.6
radius_for_obj = 0.3

npoint = 256
model = PointSDF()
model.load_state_dict(torch.load(abspath("./out/models/pointsdf.pt")))  # <-- model path here
tau = 0.3

view_params={
    "theta": 0.3,
    "phi": 0.6,
    "pi": 3.3, 
    "rotate": False, 
    "rho": 0.8,
    "resolution": (800, 600),
    "line_settings": {'point_size': 10},
}

def reconstruct_scene_pointsdf(scene_dir: str, cmap) -> Image.Image:
    rgb_path = os.path.join(scene_dir, "rgb.jpg")
    xyz_path = os.path.join(scene_dir, "xyz.npy")
    seg_path = os.path.join(scene_dir, "segmentation.npy")
    rgb = np.asarray(Image.open(rgb_path))
    rgb = rgb / 255
    xyz = np.load(xyz_path)
    # xyz[:, :, 1] = -xyz[:, :, 1]
    # xyz[:, :, 2] = -xyz[:, :, 2]
    seg_map = np.load(seg_path)[:, :]
    points = np.reshape(xyz, (-1, 3))
    seg_mask = np.reshape(seg_map, -1)
    # pre process points
    depth = np.linalg.norm(points, axis=1)
    points_valid_mask = depth > 0.1
    points = points[points_valid_mask]
    seg_mask = seg_mask[points_valid_mask]
    for id in np.unique(seg_mask[seg_mask > 0]):
        mean_pts = np.mean(points[seg_mask == id], axis=0)  # (3,)
        dist_to_mean_all_pts = np.linalg.norm(points - mean_pts, axis=1)
        neg_mask = np.logical_and(dist_to_mean_all_pts > radius_for_obj, seg_mask == id)
        seg_mask[neg_mask] = 0

    points = torch.from_numpy(points).to(torch.float32)
    seg_mask = torch.from_numpy(seg_mask)
    print(f"points: {points.shape}")
    print(f"seg_mask: {seg_mask.shape}")
    num_classes = int(torch.amax(seg_mask).item()) + 1
    obj_points_list = []
    for i in range(1, num_classes):
        point_cloud = points[seg_mask == i]
        sampled_points_idx = farthest_point_sample(point_cloud.unsqueeze(0), npoint=npoint)
        sampled_points = index_points(point_cloud.unsqueeze(0), sampled_points_idx)
        obj_points_list.append(sampled_points.reshape(npoint, 3))
    batched_points_uncentered = torch.stack(obj_points_list)  # (N, p, 3)
    obj_points, centers = scale_and_center_object_points(batched_points_uncentered)
    N = centers.shape[0]
    model.to(device)
    model.eval()
    with torch.no_grad():
        obj_feats = model.get_latent_features(obj_points.to(device))
    def occ_func(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            query_pts = scale_and_center_queries(centers.to(device), x.to(torch.float).to(device).unsqueeze(0).repeat((N, 1, 1)))
            preds = model.get_preds(obj_feats, query_pts)  # (N, P)
        return preds.cpu()
    meshes = []    
    for i in range(N):
        occ_func_i = lambda x: occ_func(x)[i]
        mins = torch.amin(batched_points_uncentered[i], dim=0)
        maxs = torch.amax(batched_points_uncentered[i], dim=0)
        cntr = 0.5 * (mins + maxs)
        bound = 0.1
        mesh = gen_mesh_for_sdf_batch_3d(
            occ_func_i,
            xlim=[cntr[0] - 0.4, cntr[0] + 0.4], 
            ylim=[cntr[1] - 0.4, cntr[1] + 0.4], 
            zlim=[cntr[2] - 0.4, cntr[2] + 0.4], 
            resolution=0.01,
            confidence=tau
        )
        if mesh is None:
            print("empty mesh :-(")
            continue
        color = np.array([0, 0, 0, 0.9])
        color[:3] = cmap[i+1]
        mesh.visual.vertex_colors = color
        meshes.append(mesh)
        
    [a, b, c, d] = run_RANSAC_for_plane(points, seg_mask, n_iters=10000, dist_thresh=0.01, radius=scene_sphere_radius)
    # plane.normal_vect = plane.normal_vect.to(points.dtype)
    scene_center = 0.5 * (
        torch.amax(points[seg_mask > 0], dim=0) 
        + torch.amin(points[seg_mask > 0], dim=0)
    )
    normal_vect = torch.tensor([a, b, c])
    bias = d
    plane_mesh = gen_mesh_for_sdf_batch_3d(
        lambda x: (x @ normal_vect.to(x.dtype) + bias < 0).to(x.dtype), 
        xlim=[scene_center[0] - 0.5, scene_center[0] + 0.5], 
        ylim=[scene_center[1] - 0.5, scene_center[1] + 0.5], 
        zlim=[scene_center[2] - 0.5, scene_center[2] + 0.5], 
        resolution=0.0075,
    )

    trimesh.smoothing.filter_laplacian(plane_mesh, iterations=40)

    recon_mesh_img = gen_image_of_trimesh_scene(
        trimesh.Scene([plane_mesh, *meshes,]),
        **view_params,
        lookat_position=scene_center
    )

    return recon_mesh_img


def reconstruct_vprism(scene_dir: str, cmap):
    rgb_path = os.path.join(scene_dir, "rgb.jpg")
    xyz_path = os.path.join(scene_dir, "xyz.npy")
    seg_path = os.path.join(scene_dir, "segmentation.npy")
    rgb = np.asarray(Image.open(rgb_path))
    rgb = rgb / 255
    xyz = np.load(xyz_path)
    # xyz[:, :, 1] = -xyz[:, :, 1]
    # xyz[:, :, 2] = -xyz[:, :, 2]
    seg_map = np.load(seg_path)[:, :]
    points = np.reshape(xyz, (-1, 3))
    seg_mask = np.reshape(seg_map, -1)
    # pre process points
    depth = np.linalg.norm(points, axis=1)
    points_valid_mask = depth > 0.1
    points = torch.from_numpy(points[points_valid_mask]).to(torch.float32)
    seg_mask = torch.from_numpy(seg_mask[points_valid_mask])
    num_classes = int(torch.amax(seg_mask).item()) + 1

    map = full_VPRISM_method(points, seg_mask, num_classes, torch.zeros(3), device=device, kernel_param=500)

    def occ_func(x: torch.Tensor) -> torch.Tensor:
        out = map.predict(x.to(device).to(map.hinge_points.dtype))
        return out.transpose(0, 1).cpu()
    obj_points_list = []
    for i in range(1, num_classes):
        point_cloud = points[seg_mask == i]
        sampled_points_idx = farthest_point_sample(point_cloud.unsqueeze(0), npoint=npoint)
        sampled_points = index_points(point_cloud.unsqueeze(0), sampled_points_idx)
        obj_points_list.append(sampled_points.reshape(npoint, 3))
    batched_points_uncentered = torch.stack(obj_points_list)  # (N, p, 3)
    batched_points_uncentered = torch.stack(obj_points_list)  # (N, p, 3)
    meshes = []    
    for i in range(1, num_classes):
        occ_func_i = lambda x: occ_func(x)[i]
        mins = torch.amin(batched_points_uncentered[i-1], dim=0)
        maxs = torch.amax(batched_points_uncentered[i-1], dim=0)
        cntr = 0.5 * (mins + maxs)
        bound = 0.1
        mesh = gen_mesh_for_sdf_batch_3d(
            occ_func_i,
            xlim=[cntr[0] - 0.4, cntr[0] + 0.4], 
            ylim=[cntr[1] - 0.4, cntr[1] + 0.4], 
            zlim=[cntr[2] - 0.4, cntr[2] + 0.4], 
            resolution=0.01,
            confidence=0.5,
        )
        if mesh is None:
            print("empty mesh :-(")
            continue
        color = np.array([0, 0, 0, 0.9])
        color[:3] = cmap[i]
        mesh.visual.vertex_colors = color
        meshes.append(mesh)
        
    [a, b, c, d] = run_RANSAC_for_plane(points, seg_mask, n_iters=10000, dist_thresh=0.01, radius=scene_sphere_radius)
    # plane.normal_vect = plane.normal_vect.to(points.dtype)
    scene_center = 0.5 * (
        torch.amax(points[seg_mask > 0], dim=0) 
        + torch.amin(points[seg_mask > 0], dim=0)
    )
    normal_vect = torch.tensor([a, b, c])
    bias = d
    plane_mesh = gen_mesh_for_sdf_batch_3d(
        lambda x: (x @ normal_vect.to(x.dtype) + bias < 0).to(x.dtype), 
        xlim=[scene_center[0] - 0.5, scene_center[0] + 0.5], 
        ylim=[scene_center[1] - 0.5, scene_center[1] + 0.5], 
        zlim=[scene_center[2] - 0.5, scene_center[2] + 0.5], 
        resolution=0.0075,
    )

    trimesh.smoothing.filter_laplacian(plane_mesh, iterations=40)

    recon_mesh_img = gen_image_of_trimesh_scene(
        trimesh.Scene([plane_mesh, *meshes,]),
        **view_params,
        lookat_position=scene_center
    )

    return recon_mesh_img
    

def reconstruct_brrp(scene_dir: str, cmap):
    rgb_path = os.path.join(scene_dir, "rgb.jpg")
    xyz_path = os.path.join(scene_dir, "xyz.npy")
    seg_path = os.path.join(scene_dir, "segmentation.npy")
    rgb = torch.from_numpy(np.asarray(Image.open(rgb_path))).to(device)
    rgb = rgb / 255
    xyz = torch.from_numpy(np.load(xyz_path)).to(device).to(torch.float32)
    seg_map = torch.from_numpy(np.load(seg_path)).to(device)
    num_classes = int(torch.amax(seg_map).item()) + 1

    weights, hp_trans = full_brrp_method(
        rgb=rgb, 
        xyz=xyz, 
        mask=seg_map, 
        prior_path=abspath("~/data/ycb_prior"), 
        device_str=device_str
    )
    
    def occ_func(x: torch.Tensor) -> torch.Tensor:
        out = x.to(device).to(torch.float32)
        out = torch.mean(torch.sigmoid(torch.sum(hp_trans.transform(out.unsqueeze(0)) * weights.unsqueeze(2), dim=-1)), dim=0)
        return out.cpu()
    obj_points_list = []
    for i in range(1, num_classes):
        point_cloud = xyz[seg_map == i]
        depth = torch.norm(point_cloud, dim=1)
        points_valid_mask = depth > 0.1
        point_cloud = point_cloud[points_valid_mask] 
        sampled_points_idx = farthest_point_sample(point_cloud.unsqueeze(0), npoint=npoint)
        sampled_points = index_points(point_cloud.unsqueeze(0), sampled_points_idx)
        obj_points_list.append(sampled_points.reshape(npoint, 3))
    batched_points_uncentered = torch.stack(obj_points_list)  # (N, p, 3)
    batched_points_uncentered = torch.stack(obj_points_list)  # (N, p, 3)
    meshes = []    
    for i in range(0, num_classes-1):
        occ_func_i = lambda x: occ_func(x)[i]
        mins = torch.amin(batched_points_uncentered[i-1], dim=0)
        maxs = torch.amax(batched_points_uncentered[i-1], dim=0)
        cntr = 0.5 * (mins + maxs).cpu()
        bound = 0.1
        mesh = gen_mesh_for_sdf_batch_3d(
            occ_func_i,
            xlim=[cntr[0] - 0.4, cntr[0] + 0.4], 
            ylim=[cntr[1] - 0.4, cntr[1] + 0.4], 
            zlim=[cntr[2] - 0.4, cntr[2] + 0.4], 
            resolution=0.01,
            confidence=0.5,
        )
        if mesh is None:
            print("empty mesh :-(")
            continue
        color = np.array([0, 0, 0, 0.9])
        color[:3] = cmap[i+1]
        mesh.visual.vertex_colors = color
        meshes.append(mesh)
    
    points = xyz.reshape(-1, 3).cpu()
    depth = torch.norm(points, dim=1)
    points_valid_mask = depth > 0.1
    points = points[points_valid_mask]
    seg_mask = seg_map.reshape(-1)[points_valid_mask].cpu()

    [a, b, c, d] = run_RANSAC_for_plane(points, seg_mask, n_iters=10000, dist_thresh=0.01, radius=scene_sphere_radius)
    # plane.normal_vect = plane.normal_vect.to(points.dtype)
    scene_center = 0.5 * (
        torch.amax(points[seg_mask > 0], dim=0) 
        + torch.amin(points[seg_mask > 0], dim=0)
    ).cpu()
    normal_vect = torch.tensor([a, b, c])
    bias = d
    plane_mesh = gen_mesh_for_sdf_batch_3d(
        lambda x: (x @ normal_vect.to(x.dtype) + bias < 0).to(x.dtype), 
        xlim=[scene_center[0] - 0.5, scene_center[0] + 0.5], 
        ylim=[scene_center[1] - 0.5, scene_center[1] + 0.5], 
        zlim=[scene_center[2] - 0.5, scene_center[2] + 0.5], 
        resolution=0.0075,
    )

    trimesh.smoothing.filter_laplacian(plane_mesh, iterations=40)

    recon_mesh_img = gen_image_of_trimesh_scene(
        trimesh.Scene([plane_mesh, *meshes,]),
        **view_params,
        lookat_position=scene_center
    )

    return recon_mesh_img
    



if __name__ == "__main__":
    main()






