"""This is the script to run BRRP qualitative experiments"""
import os
import logging
from typing import Protocol, Callable
from argparse import ArgumentParser
import time
import json

import point_cloud_utils as pcu
import torch
import numpy as np
import trimesh
from scipy.spatial.transform.rotation import Rotation
from PIL import Image

from brrp.utils import setup_logger, abspath
from brrp.hinge_points import generate_hingepoint_grid
from brrp.visualization import gen_mesh_for_sdf_batch_3d
from brrp.full_method import full_brrp_method


# constants
RESOLUTION = 0.015
EPSILON = 1e-7

# argument parser
parser = ArgumentParser()
parser.add_argument("-m", "--method", type=str)
parser.add_argument("-d", "--dataset", type=str)
parser.add_argument("-o", "--log_file", type=str, default=None)
parser.add_argument("--camera_at_origin", action="store_true")
parser.add_argument("-v", "--verbose", action="store_true")
args = parser.parse_args()

# setup the logger
log_file = None
if args.log_file:
    log_file = os.path.join(os.path.join(os.path.dirname(__file__), "../out", args.log_file))
setup_logger(file=log_file, level=logging.DEBUG if args.verbose else logging.INFO)


# torch device
device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)
logging.info(f"using device {device}")

# main method - do the stuff
def main():
    logging.info("in main func of run_sim_experiment.py")
    data_dir = abspath("~/data")
    
    logging.info(f"getting method: {args.method}")
    method = get_method(args.method)

    ious = []
    chamfers = []

    logging.info(f"dataset: {args.dataset}")
    dataset_path = os.path.join(data_dir, args.dataset)
    scene_dirs = sorted([(
        os.path.join(dataset_path, f)
    ) for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))])
    
    fit_times = []

    for i, scene_dir in enumerate(scene_dirs):  # only first view of each scene
        
        view_dir = os.path.join(scene_dir, "0000")
        rgb = torch.from_numpy(np.array(Image.open(os.path.join(view_dir, "rgb.jpg")))).to(device)
        if torch.amax(rgb) > 1.0:
            rgb = rgb / 255
        depth_np = np.load(os.path.join(view_dir, "depth.npy"))
        camera_data = np.load(os.path.join(view_dir, "camera.npz"))
        xyz = torch.from_numpy(_depth_to_xyz(
            depth_np, 
            camera_data["projection_matrix"], 
            camera_data["view_matrix"] if not args.camera_at_origin else np.eye(4),
        )).to(torch.float32).to(device)
        seg = torch.from_numpy(np.load(os.path.join(view_dir, "seg_mask.npy"))).to(torch.int64)

        with open(os.path.join(scene_dir, "objects.json"), "r") as f:
            objects_data = json.load(f)
        ids = []
        idxs = []
        metadata = []
        unique_ids = torch.unique(seg)
        seg_new = torch.zeros_like(seg)
        for i, obj in enumerate(objects_data):
            if obj["id"] in unique_ids and torch.sum(seg == obj["id"]) >= 16:  # we want there to be at least 16 points
                idxs.append(i)
                ids.append(len(idxs))
                seg_new[seg == obj["id"]] = len(idxs)
                metadata.append(objects_data[i])
        seg = seg_new
        camera_pos = torch.from_numpy(
            np.linalg.inv(camera_data["view_matrix"])[:3, 3] if not args.camera_at_origin else np.zeros(3)
        ).to(torch.float32).to(device)
        start = time.time()
        method.fit(rgb, xyz, seg.to(device), camera_pos=camera_pos)
        end_fit = time.time()
        fit_times.append(end_fit - start)
        logging.debug(f"scene {i} fit in {end_fit - start} seconds.")
        if args.camera_at_origin:
            view_dir = os.path.join(data_dir, args.dataset, f"{i:08d}", "0000")
            camera_data = np.load(os.path.join(view_dir, "camera.npz"))
            view_mat = torch.from_numpy(camera_data["view_matrix"])
            func_wrapper = get_func_wrapper(view_mat)
        for i, obj_data in enumerate(metadata):
            mesh_path = os.path.join(data_dir, obj_data["mesh_path"])
            func = lambda x: method.predict(x)[:, i]
            if args.camera_at_origin:
                func = func_wrapper(func)
            iou = calc_iou(
                mesh_path,
                pred_func=func,
                resolution=RESOLUTION,
                mesh_position=np.array(obj_data["position"]),
                mesh_orientation=np.array(obj_data["orientation"]),
                mesh_scale=np.array(obj_data["scale"]),
                conf=method.conf,
            )
            chamfer = chamfer_dist_for_mesh(
                mesh_path,
                pred_func=func,
                resolution=RESOLUTION,
                mesh_position=np.array(obj_data["position"]),
                mesh_orientation=np.array(obj_data["orientation"]),
                mesh_scale=np.array(obj_data["scale"]),
                conf=method.conf,
                method=method
            )
            ious.append(iou)
            if chamfer is not torch.nan:
                chamfers.append(chamfer)
            logging.debug(f"metrics: {iou} {chamfer}")
        end_calcs = time.time()
        logging.debug(f"metrics calculated in {end_calcs - end_fit} seconds")
        if len(chamfers) == 0:
            continue
        logging.info(f"{i} running avg metrics: {sum(ious) / len(ious)} {sum(chamfers) / len(chamfers)} in avg fit time {sum(fit_times) / len(fit_times)}")
    logging.info("=================================")
    logging.info(f"IoU: {sum(ious) / len(ious)}")
    logging.info(f"Chamfer: {sum(chamfers) / len(chamfers)}")
    logging.info(f"Speed: {sum(fit_times) / len(fit_times)}")
    logging.info("=================================")


def get_func_wrapper(view_mat):
    def wrapper(func):
        def new_func(x):
            x_aug = torch.concat([x, torch.ones_like(x[:, :1])], dim=1)
            out = x_aug @ view_mat.T
            return func(out[:, :3])
        return new_func
    return wrapper



# ======== METHODS ========


class Method(Protocol):
    def fit(self, rgb: torch.Tensor, xyz: torch.Tensor, seg: torch.Tensor, *, camera_pos: torch.Tensor):
        pass

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        pass


class BRRPMethod:
    def __init__(self):
        self.conf = 0.5

    def fit(self, rgb: torch.Tensor, xyz: torch.Tensor, seg: torch.Tensor, *, camera_pos: torch.Tensor):
        w, hp = full_brrp_method(rgb, xyz, seg, abspath("~/data/ycb_prior"), device_str=device_str, camera_pos=camera_pos)
        self.hp = hp
        self.w = w  # (P, N, H)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        out = self.hp.transform(x.to(self.w.dtype).to(self.w.device).unsqueeze(0))  # (N, Pts, H)
        out = torch.sum(out * self.w.unsqueeze(2), dim=-1)  # (P, N, Pts)
        out = torch.mean(torch.sigmoid(out), dim=0)  # (N, Pts)
        return out.transpose(0, 1).cpu()  # (Pts, N)
        





def get_method(method_str: str) -> Method:
    if method_str == "BRRP":
        return BRRPMethod()
    if method_str == "V-PRISM":
        return None
    if method_str == "PointSDF":
        return None
    raise Exception(f"there is no such method '{method_str}'")





# ======== EVALUATION METHODS ========



def calc_iou_trimesh(
    mesh_path: str, 
    pred_func: Callable[[torch.Tensor], torch.Tensor], 
    resolution: float,
    *,
    conf: float = 0.5,
    mesh_position: list[float],
    mesh_orientation: list[float],
    mesh_scale: float,
    max_elem_in_func_call: int = 10_000
) -> float:
    mesh = _as_mesh(trimesh.load(mesh_path))
    mesh.vertices = _transform_array(mesh.vertices, mesh_position, mesh_orientation, mesh_scale)

    mins = torch.amin(torch.from_numpy(mesh.vertices), dim=0)
    maxs = torch.amax(torch.from_numpy(mesh.vertices), dim=0)
    centroid = 0.5 * (maxs + mins)
    
    # 1. create input points
    grid = generate_hingepoint_grid(centroid - BUFFER, centroid + BUFFER, resolution=resolution)
    fn1_preds = torch.zeros_like(grid[:, 0], dtype=torch.bool)
    fn2_preds = torch.zeros_like(grid[:, 0], dtype=torch.bool)
    # 2. evaluate fn1 and fn2
    perm = torch.randperm(grid.shape[0])
    k = 0
    while k < len(perm):
        idxs = perm[k:k + max_elem_in_func_call]
        fn1_preds[idxs] = pred_func(grid[idxs]) > conf
        fn2_preds[idxs] = torch.from_numpy(mesh.contains(grid[idxs])) > conf
        k += max_elem_in_func_call
    
    # 3. calc iou
    intersection = torch.sum(torch.logical_and(fn1_preds, fn2_preds))
    union = torch.sum(torch.logical_or(fn1_preds, fn2_preds))
    if union == 0:
        return 0.0
    return intersection / union

def calc_iou(
    mesh_path: str, 
    pred_func: Callable[[torch.Tensor], torch.Tensor], 
    resolution: float,
    *,
    conf: float = 0.5,
    mesh_position: list[float],
    mesh_orientation: list[float],
    mesh_scale: float,
    max_elem_in_func_call: int = 4_000
) -> float:
    v, f = pcu.load_mesh_vf(mesh_path)
    v, f = pcu.make_mesh_watertight(v, f, 4_000)
    v = _transform_array(v, mesh_position, mesh_orientation, mesh_scale)

    mins = torch.amin(torch.from_numpy(v), dim=0)
    maxs = torch.amax(torch.from_numpy(v), dim=0)
    centroid = 0.5 * (maxs + mins)
    
    # 1. create input points
    grid = generate_hingepoint_grid(centroid - BUFFER, centroid + BUFFER, resolution=resolution)
    fn1_preds = torch.zeros_like(grid[:, 0], dtype=torch.bool)
    fn2_preds = torch.zeros_like(grid[:, 0], dtype=torch.bool)
    # 2. evaluate fn1 and fn2
    perm = torch.randperm(grid.shape[0])
    k = 0
    while k < len(perm):
        idxs = perm[k:k + max_elem_in_func_call]
        fn1_preds[idxs] = pred_func(grid[idxs]) > conf
        gt_sdf_vals, _, _  = pcu.signed_distance_to_mesh(grid[idxs].numpy(), v, f)
        fn2_preds[idxs] = torch.from_numpy(gt_sdf_vals) < 0
        k += max_elem_in_func_call
    
    # 3. calc iou
    intersection = torch.sum(torch.logical_and(fn1_preds, fn2_preds))
    union = torch.sum(torch.logical_or(fn1_preds, fn2_preds))
    if union == 0:
        return 0.0
    return intersection / union


BUFFER = 0.2


def chamfer_dist_for_mesh(
    mesh_path: str, 
    pred_func: Callable[[torch.Tensor], torch.Tensor], 
    resolution: float,
    *,
    conf: float = 0.5,
    mesh_position: list[float],
    mesh_orientation: list[float],
    mesh_scale: float,
    sample_count: int = 8_000,
    device: torch.device = torch.device("cpu"),
    method: Method | None = None
) -> float:
    # (1) load mesh
    v, f = pcu.load_mesh_vf(mesh_path)
    v = _transform_array(v, np.array(mesh_position), np.array(mesh_orientation), mesh_scale)
    vm, fm = pcu.make_mesh_watertight(v, f, 10_000)
    # (2) calc counts
    mins = torch.amin(torch.from_numpy(v), dim=0)
    maxs = torch.amax(torch.from_numpy(v), dim=0)
    centroid = 0.5 * (maxs + mins)
    # construct mesh
    recon_mesh = gen_mesh_for_sdf_batch_3d(
        pred_func, 
        xlim=[centroid[0] - BUFFER, centroid[0] + BUFFER], 
        ylim=[centroid[1] - BUFFER, centroid[1] + BUFFER], 
        zlim=[centroid[2] - BUFFER, centroid[2] + BUFFER], 
        resolution=resolution, 
        confidence=conf, 
        device=device
    )

    if recon_mesh is None:
        return torch.nan  # what to do here?
    recon_samples = recon_mesh.sample(sample_count)
    fid, bc = pcu.sample_mesh_random(vm, fm, sample_count)
    gt_samples = pcu.interpolate_barycentric_coords(fm, fid, bc, vm)
    dist = pcu.chamfer_distance(recon_samples, gt_samples)
    return dist

def _transform_array(
    points: np.ndarray, 
    translation: np.ndarray, 
    quat: np.ndarray,
    scale: float = 1,
) -> np.ndarray:
    my_rotation: Rotation = Rotation.from_quat(quat)
    new_points = points * scale
    new_points: np.ndarray = my_rotation.apply(new_points)
    new_points = new_points + translation
    return new_points


def _as_mesh(scene_or_mesh) -> trimesh.Trimesh:
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        mesh = scene_or_mesh
        assert(isinstance(mesh, trimesh.Trimesh))
    return mesh


# ======== TO HELP WITH CONVERTING DEPTH ========

def _depth_to_xyz(
    depth: np.ndarray, 
    proj_matrix: np.ndarray, 
    view_matrix: np.ndarray = np.eye(4)
) -> np.ndarray:
    H, W = depth.shape
    tran_pix_world = np.linalg.inv(proj_matrix @ view_matrix)
    y, x = np.mgrid[-1:1:2 / H, -1:1:2 / W]
    y *= -1.
    x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
    h = np.ones_like(z)
    pixels = np.stack([x, y, z, h], axis=1)
    pixels[:, 2] = 2 * pixels[:, 2] - 1
    points = (tran_pix_world @ pixels.T).T
    points = points / points[:, 3: 4]
    points: np.ndarray = points[:, :3]
    return points.reshape((H, W, 3))
    


# ======== MAIN METHOD CALL (THE END) ========

if __name__ == "__main__":
    main()

