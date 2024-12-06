import logging
import os

import numpy as np
import torch
from PIL import Image
import trimesh

from brrp.utils import abspath, setup_logger
from brrp.visualization import gen_mesh_for_sdf_batch_3d, some_colors
from brrp.subsample import grid_subsample
from brrp.full_method import full_brrp_method


scene_dir = abspath("~/data/brrp_real_world_scenes/0003")  # CHANGE THIS


setup_logger()
device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)
logging.info(f"beginning example.py with device {device}")

# load stuff in
rgb_path = os.path.join(scene_dir, "rgb.jpg")
xyz_path = os.path.join(scene_dir, "xyz.npy")
seg_path = os.path.join(scene_dir, "segmentation.npy")
rgb = torch.from_numpy(np.asarray(Image.open(rgb_path))).to(device)
rgb = rgb / 255
xyz = torch.from_numpy(np.load(xyz_path)).to(device).to(torch.float32)
seg_map = torch.from_numpy(np.load(seg_path)).to(device)
num_classes = int(torch.amax(seg_map).item()) + 1
scene_center = 0.5 * (torch.amin(xyz[seg_map > 0], dim=0) + torch.amax(xyz[seg_map > 0], dim=0))

logging.info("beginning BRRP method")
weights, hp_trans = full_brrp_method(
    rgb=rgb, 
    xyz=xyz, 
    mask=seg_map, 
    prior_path=abspath("~/data/ycb_prior"), 
    device_str=device_str
)
logging.info("BRRP finished.")

obj_points_list = []
for i in range(1, num_classes):
    point_cloud = xyz[seg_map == i]
    depth = torch.norm(point_cloud, dim=1)
    points_valid_mask = depth > 0.1
    point_cloud = point_cloud[points_valid_mask] 
    pc, batch = grid_subsample(point_cloud, torch.zeros_like(point_cloud[:, 0], dtype=torch.int64), 0.01)
    obj_points_list.append(pc)

logging.info("marching cubes...")
def occ_func(x: torch.Tensor) -> torch.Tensor:
    out = x.to(device).to(torch.float32)
    out = torch.mean(torch.sigmoid(torch.sum(hp_trans.transform(out.unsqueeze(0)) * weights.unsqueeze(2), dim=-1)), dim=0)
    return out.cpu()
meshes = []
for i in range(0, num_classes-1):
    occ_func_i = lambda x: occ_func(x)[i]
    mins = torch.amin(obj_points_list[i-1], dim=0)
    maxs = torch.amax(obj_points_list[i-1], dim=0)
    cntr = 0.5 * (mins + maxs).cpu()
    bound = 0.1
    mesh = gen_mesh_for_sdf_batch_3d(
        occ_func_i,
        xlim=[cntr[0] - 0.4, cntr[0] + 0.4], 
        ylim=[cntr[1] - 0.4, cntr[1] + 0.4], 
        zlim=[cntr[2] - 0.4, cntr[2] + 0.4], 
        resolution=0.015,
        confidence=0.5,
    )
    if mesh is None:
        logging.warning("empty mesh :-(")
        continue
    mesh.visual.vertex_colors = some_colors[i] + [215]
    meshes.append(mesh)

logging.info("displaying...")
colors_tensor = torch.tensor([[0, 0, 0]] + some_colors)
filter_mask = torch.norm(xyz - scene_center, dim=-1) <= 0.6
points_filtered = xyz[filter_mask]
seg_filtered = seg_map[filter_mask]
points_filtered, batch = grid_subsample(points_filtered, seg_filtered, 0.02)

# trimesh.Scene([trimesh.PointCloud(points_filtered.cpu(), colors=colors_tensor[batch.cpu()])]).show()
trimesh.Scene(meshes + [trimesh.PointCloud(points_filtered.cpu(), colors=colors_tensor[batch.cpu()])]).show()