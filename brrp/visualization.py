from typing import Callable
import io
import logging

import torch
import numpy as np
import trimesh
from tqdm import trange
from skimage import measure
from PIL import Image
from PIL.Image import Image as PILImage


def gen_mesh_for_sdf_batch_3d(
    occ_func: Callable[[np.ndarray], np.ndarray], 
    xlim: list[int] = [0, 1], 
    ylim: list[int] = [0, 1], 
    zlim: list[int] = [0, 1],
    resolution: float = 0.01,
    *,
    confidence: float = 0.5,
    device: torch.device = torch.device('cpu'),
    only_batch_one_dim: bool = False,
) -> trimesh.Trimesh:
    """Performs marching cubes to generate a mesh for a given sdf 
    
    (done in batch so it is faster)

    Args:
        - occ_func: (Callable: (N, 3) -> (N,))
        - xlim: (list[int]) list of two values corresponding to max and min x
        - ylim: (list[int]) list of two values corresponding to max and min y
        - zlim: (list[int]) list of two values corresponding to max and min z
        - resolution: (float) the resolution for the mesh
        - (optional) confidence: (float) the level set value. Defaults to 0.5
        - (optional) device: (torch.device) device for calculation. Defaults to cpu
        - (optional) only_batch_one_dim: (bool) flag for performance. Don't touch, defaults to false.
    
    Returns:
        - mesh: (Trimesh)
    """
    xyz = torch.tensor(np.stack(np.meshgrid(
        np.arange(xlim[0], xlim[1], step=resolution),
        np.arange(ylim[0], ylim[1], step=resolution),
        np.arange(zlim[0], zlim[1], step=resolution),
    ), axis=-1), device=device)
    X, Y, Z, _ = xyz.shape
    if only_batch_one_dim:
        tsdf = torch.zeros((X, Y, Z), device=device)
        for i in trange(X):
            for j in range(Y):
                # print('here', xyz[i, j].shape)
                tsdf[i, j] = occ_func(xyz[i, j])
    else:
        xyz = xyz.reshape((xyz.shape[0], -1, 3))
        tsdf = torch.zeros((X, Y * Z), device=device)
        for i in trange(X):
            tsdf[i] = occ_func(xyz[i]) - confidence
        tsdf = tsdf.reshape(X, Y, Z)
    logging.debug(f'reconstructing mesh; max={tsdf.amax()}, min={tsdf.amin()}')
    try:
        verts, faces, normals, _ = measure.marching_cubes(
            tsdf.cpu().numpy(), 
            0, 
            spacing=[resolution for i in range(3)],
        )
        idx = np.array([1, 0, 2], dtype=np.int64)
        verts = verts[:, idx]
        normals = normals[:, idx]
        verts = verts + np.array([xlim[0], ylim[0], zlim[0]])
        mesh = trimesh.Trimesh(verts, faces, normals)
        mesh.visual.vertex_colors = np.array([100, 100, 200, 215])
    except Exception as err:
        logging.warning(err)
        logging.warning('error while running marching cubes; returning empty mesh')
        mesh = None
    return mesh


def gen_image_of_trimesh_scene(
    scene: trimesh.Scene, 
    theta: float,
    *,
    phi: float = 0.35 * np.pi,
    rho: float = 0.35,
    lookat_position: np.ndarray | None = None,
    rotate: bool = True,
    pi: float = 0.0,
    resolution: list[float] = (512, 512),
    line_settings: dict | None = None
) -> PILImage:
    """returns a color image of the trimesh scene

    Args:
        - scene: (trimesh.Scene) trimesh scene to take img of
        - theta: (float) angle around ground plane in radians
        - phi: (float) angle from up direction in radians. default to 0.35*pi.
        - rho: (float) distance from center of scene. default to 0.35.
        - lookat_position: (3,) position in middle of screen
    
    Returns:
        - img: (H, W, 3) image of scene
    """
    if lookat_position is None:
        lookat_position = np.mean(scene.bounds, axis=0)
    scene.set_camera(
        np.array([pi, phi, theta]),
        distance=rho,
        center=lookat_position,
    )
    img = Image.open(io.BytesIO(scene.save_image(resolution=resolution, line_settings=line_settings)))
    if rotate:
        img = img.rotate(-90)
    return img


some_colors = [
    [0, 100, 0],  # dark green (gray 59)
    [240, 20, 20],  # red (gray 86)
    [106, 90, 205],  # slate blue (gray 108)
    [0, 230, 230],  # aqua (gray 161)
    [0, 240, 0],  # lime (gray 141)
    [148, 0, 0],  # maroon (gray 44)
    [47, 79, 79],  # slate grey (gray 69)
    [255, 0, 255],  # fuchsia (gray 105)
    [255, 215, 0],  # gold (gray 152)
    [25, 25, 112],  # midnight blue (gray 35)
    [240, 175, 183],  # pink (gray 195)
]
