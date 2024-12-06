"""
This file will train PointSDF in Pytorch. It requires v_prism, which can be downloaded with:
```
pip install git+https://github.com/Herb-Wright/v-prism.git
```
Then this file can be ran without any arguments:
```
python scripts/train_pointsdf.py
```

**Question:** *Should I make a SDF version or just do the occupancy version*
"""
import os
import math
import logging

import torch
from torch.optim import Adam
from torch import Tensor
from torch.nn import BCELoss
from tqdm import tqdm
from v_prism.utils.pointsdf import PointSDF, scale_and_center_object_points
from v_prism.utils.pointsdf import scale_and_center_queries, index_points
from v_prism.utils.pointsdf import  farthest_point_sample
from v_prism.utils.ui import abspath, mkdir_if_not_exists
from v_prism.data_loading.bullet_reconstruction_dataset import ReconBulletDataset

from brrp.utils import setup_logger


out_dir = abspath(os.path.join(os.path.dirname(__file__), "../out"))

setup_logger(file=os.path.join(out_dir, "point_sdf_train2.log"))
logging.info("beginning train_pointsdf.py")


dataset_name = "psdf_shapenet_train_2000"
num_epochs = 50
learning_rate = 1e-3
batch_size = 2   # in scenes, not objects
model_name = "pointsdf_occ.pt"
num_query_pts = 512
pointcloud_size = 256
save_every_n_steps = 200   # could be None

logging.info(f"hyperparams: {dataset_name} {num_epochs} {learning_rate} {batch_size} {num_query_pts} {pointcloud_size} {save_every_n_steps}")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
logging.info(f"using device {device}!")
data_dir = abspath("~/data")
out_dir = abspath(os.path.join(os.path.dirname(__file__), "../out"))
mkdir_if_not_exists(os.path.join(out_dir, "models"))
model_path = abspath(os.path.join(out_dir, "models", model_name))
model = PointSDF(0.2, "sigmoid")

if os.path.exists(model_path):
    logging.info(f"loading model from path {model_path}")
    model.load_state_dict(torch.load(model_path))


class CustomDataLoader:
    def __init__(self, dataset: ReconBulletDataset, batch_size: int):
        self.dataset = dataset
        self.batch_size = batch_size

    def __len__(self) -> int:
        return math.ceil(len(dataset) / self.batch_size)

    def __iter__(self):
        self.curr_idx = 0
        self.perm = torch.randperm(len(self.dataset))
        return self

    def __next__(self) -> tuple[Tensor, Tensor, Tensor]:
        if self.curr_idx >= len(self.perm):
            raise StopIteration
        next_idx = self.curr_idx + self.batch_size
        ds_idxs = self.perm[self.curr_idx:next_idx]
        self.curr_idx = next_idx
        elems = [self.dataset[i] for i in ds_idxs]
        object_points_list = []
        query_pts_list = []
        sdf_vals_list = []
        for scene in elems:
            Q = scene.sdf_vals.shape[1]
            qp_for_scene_list = []
            sdf_for_scene_list = []
            scene_points = scene.points.to(device)  # for speed up! 
            for i, metadata in enumerate(scene.object_metadata):
                point_cloud = scene_points[scene.seg_mask == i+1]
                sampled_points_idx = farthest_point_sample(point_cloud.unsqueeze(0), npoint=pointcloud_size)
                sampled_points = index_points(point_cloud.unsqueeze(0), sampled_points_idx)
                object_points_list.append(sampled_points.reshape(pointcloud_size, 3))
                query_sdf_idx = i
                random_query_pts_idx = torch.randperm(Q)[:num_query_pts]
                qp_for_scene_list.append(scene.query_pts[query_sdf_idx][random_query_pts_idx])
                sdf_for_scene_list.append(scene.sdf_vals[query_sdf_idx][random_query_pts_idx])
            query_pts_list.append(torch.stack(qp_for_scene_list))
            sdf_vals_list.append(torch.stack(sdf_for_scene_list))
        obj_points = torch.stack(object_points_list).to(torch.float).to(torch.device("cpu"))  # (N, P, 3)
        query_pts = torch.concat(query_pts_list, dim=0).to(torch.float)
        sdf_vals = torch.concat(sdf_vals_list, dim=0).to(torch.float)
        obj_points, centers = scale_and_center_object_points(obj_points)
        query_pts = scale_and_center_queries(centers, query_pts)
        occ_vals = sdf_vals <= 0
        return obj_points, query_pts, occ_vals.to(torch.float)



dataset = ReconBulletDataset(dataset_name, data_dir, keep_camera_at_origin=True)
dataloader = CustomDataLoader(dataset, batch_size)
optimizer = Adam(model.parameters(), learning_rate)
loss_func = BCELoss()

model.to(device)

for epoch in range(num_epochs):
    # Train loop
    model.train()
    train_loss = 0
    n_batch = len(dataloader) 
    for i, batch in (bar := tqdm(
        enumerate(dataloader),
        desc=f"epoch {epoch}...",
        total=n_batch
    )):
        optimizer.zero_grad()
        obj_points, query_pts, occ_vals = batch
        obj_points = obj_points.to(device)
        obj_points = obj_points + torch.randn_like(obj_points) * 0.002  # data augmentation baby!
        query_pts = query_pts.to(device)
        occ_vals = occ_vals.to(device)
        preds = model(obj_points, query_pts)
        loss = loss_func(preds, occ_vals)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()
        bar.set_description(f"epoch {epoch} loss: {train_loss / (i+1):0.4f}")
        if save_every_n_steps is not None and (i + 1) % save_every_n_steps == 0:
            logging.debug("saving the model.")
            torch.save(model.state_dict(), model_path)
    logging.info(f"finished epoch {epoch} with loss {train_loss / n_batch}.")
    logging.debug("saving the model.")
    torch.save(model.state_dict(), model_path)

logging.info("Finished training.")