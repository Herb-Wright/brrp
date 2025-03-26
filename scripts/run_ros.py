import logging

import torch
from sensor_msgs.msg import PointCloud2, Image as msg_image
import ros_numpy
import rospy
import numpy as np
from shape_msgs.msg import Mesh
import trimesh

from custom_msgs.msg import MeshList
from brrp.full_method import full_brrp_method
from brrp.segmenter import GroundedSamSegmenter
from brrp.visualization import gen_mesh_for_sdf_batch_3d, some_colors

rate = 0.2  # Hz; once ever 5 seconds
path_to_prior = "./ycb_prior"

def main():
    r = rospy.Rate(rate) # run at rate Hz
    node = BRRPNode(
        image_topic="/rgb/image_raw",
        point_cloud_topic="/points2",
        mesh_list_topic="/perception/mesh_list"
    )
    while not rospy.is_shutdown():
        node.publish()
        r.sleep()

class BRRPNode:
    def __init__(
        self: str, 
        image_topic: str, 
        point_cloud_topic: str,
        mesh_list_topic: str,
    ) -> None:
        self.xyz = None
        self.rgb = None
        rospy.Subscriber(image_topic, msg_image, self.callback_image, queue_size=1)
        rospy.Subscriber(point_cloud_topic, PointCloud2, self.callback_point_cloud, queue_size=1)
        self.pub = rospy.Publisher(mesh_list_topic, MeshList, queue_size=1)
        self.convert_depth_to_res = True  # Change to false if realsense
        self.res = None
        self.segmenter = GroundedSamSegmenter("cuda")

    def callback_point_cloud(self, msg: PointCloud2):
        if msg.height == 1:
            if not self.convert_depth_to_res:
                rospy.logwarn("PointCloud2 has a height of 1. Cannot segment.")
            if self.res is not None:
                msg.height = self.res[0]
                msg.width = self.res[1]

        array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg, remove_nans=False)
        self.xyz = array.reshape((msg.height, msg.width, -1))

    def callback_image(self, msg: msg_image):
        if self.convert_depth_to_res:
            self.res = (msg.height, msg.width)  # (720, 1280)?
            # rospy.loginfo(f"image had shape ({msg.height}, {msg.width})")
        rgb_raw = ros_numpy.image.image_to_numpy(msg) / 255
        if msg.encoding == "bgra8":
            idxs = np.array([2, 1, 0])
            rgb_raw = rgb_raw[: , :, idxs]
        self.rgb = rgb_raw

    def publish(self):
        if self.xyz is None or self.rgb is None or self.xyz.shape != self.rgb.shape:
            rospy.loginfo("Can't publish yet: either not enough data or wrong shape")
            return
        rgb = torch.from_numpy(self.rgb)
        xyz = torch.from_numpy(self.xyz)
        seg_mask = self.segmenter.segment(self.rgb, self.xyz)

        weights, hp_transform = full_brrp_method(rgb, xyz, seg_mask, path_to_prior)

        num_classes = int(seg_mask.amax().item() + 1)
        def occ_func(x: torch.Tensor) -> torch.Tensor:
            out = x.to(torch.device("cuda")).to(torch.float32)
            out = torch.mean(torch.sigmoid(torch.sum(hp_transform.transform(out.unsqueeze(0)) * weights.unsqueeze(2), dim=-1)), dim=0)
            return out.cpu()
        meshes: list[trimesh.Trimesh] = []
        for i in range(0, num_classes-1):
            occ_func_i = lambda x: occ_func(x)[i]
            mins = torch.amin(xyz[seg_mask == i+1], dim=0)
            maxs = torch.amax(xyz[seg_mask == i+1], dim=0)
            cntr = 0.5 * (mins + maxs).cpu()
            mesh = gen_mesh_for_sdf_batch_3d(
                occ_func_i,
                xlim=[cntr[0] - 0.4, cntr[0] + 0.4], 
                ylim=[cntr[1] - 0.4, cntr[1] + 0.4], 
                zlim=[cntr[2] - 0.4, cntr[2] + 0.4], 
                resolution=0.015,  # change this for better resolution
                confidence=0.5,
            )
            if mesh is None:
                logging.warning("empty mesh :-(")
                continue
            mesh.visual.vertex_colors = some_colors[i] + [215]
            meshes.append(mesh)

        msg = MeshList()
        msg_meshes = []
        for mesh in meshes:
            msg_mesh = Mesh()
            msg_mesh.triangles = mesh.faces
            msg_mesh.vertices = mesh.vertices
            msg_meshes.append(msg_mesh)
        msg.meshes = msg_meshes
        self.pub.publish(msg)



if __name__ == "__main__":
    main()