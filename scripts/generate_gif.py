"""
do `pip install imageio[ffmpeg]` for this
"""
import logging
import os
import io

import numpy as np
import torch
from PIL import Image
import trimesh
import imageio

from brrp.utils import abspath, setup_logger
from brrp.visualization import gen_mesh_for_sdf_batch_3d, some_colors
from brrp.subsample import grid_subsample
from brrp.full_method import full_brrp_method


scene_dir = abspath("~/data/brrp_real_world_scenes/0003")  # CHANGE THIS


setup_logger(level=logging.INFO)
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
# seg_map = (seg_map == 2).to(seg_map.dtype)
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
        resolution=0.01,
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
points_filtered, batch = grid_subsample(points_filtered, seg_filtered, 0.01)


def create_rotation_video(
    scene,
    output_path="reconstruction.mp4",
    num_frames=60,
    phi=0,
    pi_angle=3.2,
    rho=1.3,
    lookat_position=None,
    resolution=(800, 600),
    line_settings=None,
    fps=30
):
    """
    Create a video of a trimesh scene with rotating camera.
    
    Parameters:
    -----------
    scene : trimesh.Scene
        The trimesh scene to render
    output_path : str
        Path to save the output video (default: "reconstruction.mp4")
    num_frames : int
        Number of frames in the rotation (default: 60)
    phi : float
        Vertical angle in radians (default: 0)
    theta : float
        Initial horizontal angle in radians (default: 0)
    rho : float
        Distance from the center (default: 2.0)
    lookat_position : array-like or None
        Center position to look at. If None, uses scene centroid
    resolution : tuple
        Image resolution as (width, height) (default: (800, 600))
    line_settings : dict or None
        Line settings for rendering (default: None)
    fps : int
        Frames per second for the output video (default: 30)
    """
    frames = []
    
    # Use scene centroid if lookat_position not provided
    if lookat_position is None:
        lookat_position = scene.centroid
    
        # Generate frames by rotating pi angle from 0 to 2*pi
    for i in range(num_frames):
        # Rotate camera horizontally (full 360 degree rotation)
        phi = -0.9*np.pi * (i / num_frames - 0.5)
        
        # Set camera position
        scene.set_camera(
            np.array([pi_angle, phi, 0.0]),
            distance=rho,
            center=lookat_position,
        )
        
        # Render the scene
        img_bytes = scene.save_image(resolution=resolution, line_settings=line_settings)
        img = Image.open(io.BytesIO(img_bytes))
        
        # Convert to numpy array for imageio
        frames.append(np.array(img))
    
    # Save as MP4 video
    imageio.mimsave(output_path, frames, fps=fps, format='FFMPEG', codec='libx264')
    logging.info(f"Video saved to {output_path}")

# trimesh.Scene([trimesh.PointCloud(points_filtered.cpu(), colors=colors_tensor[batch.cpu()])]).show()
scene = trimesh.Scene(meshes + [trimesh.PointCloud(points_filtered.cpu(), colors=colors_tensor[batch.cpu()])])
# scene = trimesh.Scene([trimesh.PointCloud(points_filtered.cpu(), colors=colors_tensor[batch.cpu()])])

create_rotation_video(scene)