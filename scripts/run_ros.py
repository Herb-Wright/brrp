import logging

import torch
from sensor_msgs.msg import PointCloud2, Image as msg_image
import ros_numpy
import rospy
import numpy as np
from shape_msgs.msg import Mesh, MeshTriangle
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Vector3
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import ColorRGBA
import trimesh

from custom_msgs.msg import MeshList
from brrp.full_method import full_brrp_method
from brrp.segmenter import GroundedSamSegmenter
from brrp.visualization import gen_mesh_for_sdf_batch_3d, some_colors


rate = 0.2  # Hz; once ever 5 seconds
path_to_prior = "./ycb_prior"

def main():
    rospy.init_node("BRRP", log_level=rospy.DEBUG, anonymous=True)
    r = rospy.Rate(rate) # run at rate Hz
    node = BRRPNode(
        image_topic="/rgb/image_raw",
        point_cloud_topic="/points2",
        mesh_list_topic="/perception/mesh_list",
        visualization_topic="/perception/visualization/mesh_marker_list"
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
        visualization_topic: str | None = None,
    ) -> None:
        self.xyz = None
        self.rgb = None
        rospy.Subscriber(image_topic, msg_image, self.callback_image, queue_size=1)
        rospy.Subscriber(point_cloud_topic, PointCloud2, self.callback_point_cloud, queue_size=1)
        self.pub = rospy.Publisher(mesh_list_topic, MeshList, queue_size=1)
        self.seg_pub = rospy.Publisher("/perception/segmented_pointcloud", PointCloud2)
        self.convert_depth_to_res = True  # Change to false if realsense
        self.res = None
        self.segmenter = GroundedSamSegmenter(
            "cuda",
            prompt="an object that can be picked up with one hand and is resting on a supporting surface",
            max_depth=2.0,
            min_depth=0.5,
            threshold=0.10,
            quantile_max=0.25
        )
        if visualization_topic:
            self.vis_pub = rospy.Publisher(visualization_topic, MarkerArray, queue_size=1)
        else:
            self.vis_pub = None

    def callback_point_cloud(self, msg: PointCloud2):
        if msg.height == 1:
            if not self.convert_depth_to_res:
                rospy.logwarn("PointCloud2 has a height of 1. Cannot segment.")
            if self.res is not None:
                msg.height = self.res[0]
                msg.width = self.res[1]

        array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg, remove_nans=False)
        self.xyz = array.reshape((msg.height, msg.width, -1))
        self.frame_id = msg.header.frame_id

    def callback_image(self, msg: msg_image):
        if self.convert_depth_to_res:
            self.res = (msg.height, msg.width)  # (720, 1280)?
        rgb_raw = ros_numpy.image.image_to_numpy(msg) / 255
        if msg.encoding == "bgra8":
            idxs = np.array([2, 1, 0])
            rgb_raw = rgb_raw[: , :, idxs]
        self.rgb = rgb_raw

    def publish(self):
        if self.xyz is None or self.rgb is None or self.xyz.shape != self.rgb.shape:
            rospy.loginfo("Can't publish yet: either not enough data or wrong shape")
            return
        rgb = torch.from_numpy(self.rgb).to(torch.device("cuda")) # (H, W, 3)
        xyz = torch.from_numpy(self.xyz).to(torch.device("cuda")) # (H, W, 3)
        seg_mask = self.segmenter.segment(self.rgb, self.xyz).to(torch.device("cuda")) # (H, W)
        seg_xyz = xyz[seg_mask > 0].cpu().numpy()
        seg_mask_flat = seg_mask[seg_mask > 0].cpu().numpy()
        P = seg_xyz.shape[0]
        dtype_for_array = [("x", np.float32), ("y", np.float32), ("z", np.float32), ("seg", np.int32)]
        out_array = np.recarray((P,), dtype=dtype_for_array)
        out_array["x"] = seg_xyz[:, 0]
        out_array["y"] = seg_xyz[:, 1]
        out_array["z"] = seg_xyz[:, 2]
        out_array["seg"] = seg_mask_flat
        msg_out = ros_numpy.point_cloud2.array_to_pointcloud2(out_array)
        msg_out.header.frame_id = "rgb_camera_link"
        self.seg_pub.publish(msg_out)
        rospy.loginfo("running BRRP")
        weights, hp_transform = full_brrp_method(
            rgb.to(torch.float32),
            xyz.to(torch.float32),
            seg_mask, path_to_prior,
            device_str="cuda",
            lambda_prior=5.0
        )

        num_classes = int(seg_mask.amax().item() + 1)
        rospy.loginfo(f"running marching cubes num_classes={num_classes}")
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
            mesh.visual.vertex_colors = some_colors[i % len(some_colors)] + [215]
            meshes.append(mesh)

        rospy.loginfo(f"publishing meshes n_meshes={len(meshes)}")
        msg = MeshList()
        msg_meshes = []
        for mesh in meshes:
            msg_mesh = create_mesh_message(mesh.vertices, mesh.faces)
            msg_meshes.append(msg_mesh)
        msg.meshes = msg_meshes
        self.pub.publish(msg)

        if self.vis_pub:
            rospy.loginfo(f"publishing to visualization topic frame_id={self.frame_id}")
            marker_array = []
            for i, mesh in enumerate(meshes):
                marker = create_triangle_list_marker(mesh.vertices, mesh.faces, i, self.frame_id)
                marker_array.append(marker)
            self.vis_pub.publish(marker_array)


def create_triangle_list_marker(v, f, mesh_id, frame_id):
    """
    Convert vertices and faces to TRIANGLE_LIST marker

    Expected mesh format:
    - mesh.vertices: array of vertices, each with x, y, z
    - mesh.faces: array of faces, each with 3 vertex indices
    """
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = "triangle_meshes"
    marker.id = mesh_id
    marker.type = Marker.TRIANGLE_LIST
    marker.action = Marker.ADD
    marker.lifetime = rospy.Duration(70)

    # Set pose (usually identity for vertex-based meshes)
    marker.pose.orientation.w = 1.0

    # Set scale (usually 1.0 for vertex coordinates)
    marker.scale = Vector3(1.0, 1.0, 1.0)

    # Set color
    color = some_colors[mesh_id % len(some_colors)]
    marker.color = ColorRGBA(color[0], color[1], color[2], 0.8)

    # Convert faces to triangle points
    try:
        triangle_points = faces_to_triangle_points(v, f)
        marker.points = triangle_points

        # Optional: Set colors per vertex (same length as points)
        marker.colors = [marker.color] * len(triangle_points)

        return marker
    
    except Exception as e:
        rospy.logerr(f"Error creating triangle marker for mesh {mesh_id}: {e}")
        return None
    
def faces_to_triangle_points(vertices, faces):
    """
    Convert vertex array and face indices to triangle points for RViz

    Args:
        vertices: List/array of vertices, each with x, y, z attributes or [x,y,z] array
        faces: List/array of faces, each with 3 vertex indices

    Returns:
        List of geometry_msgs/Point for TRIANGLE_LIST marker
    """
    triangle_points = []

    for face in faces:
        # Each face should have 3 vertex indices
        if len(face) != 3:
            rospy.logwarn(f"Face has {len(face)} vertices, expected 3. Skipping.")
            continue

        # Get the 3 vertices for this triangle
        for vertex_idx in face:
            if vertex_idx >= len(vertices):
                rospy.logwarn(f"Vertex index {vertex_idx} out of range. Skipping face.")
                break

            vertex = vertices[vertex_idx]
            point = Point()

            # Handle different vertex formats
            if hasattr(vertex, 'x'):  # ROS Point/Vector3 message
                point.x = vertex.x
                point.y = vertex.y
                point.z = vertex.z
            elif isinstance(vertex, (list, tuple, np.ndarray)):  # Array-like
                point.x = float(vertex[0])
                point.y = float(vertex[1])
                point.z = float(vertex[2])
            else:
                rospy.logwarn(f"Unknown vertex format: {type(vertex)}")
                continue

            triangle_points.append(point)

    return triangle_points


def create_mesh_message(vertices, faces):
    """
    Create a ROS Mesh message from vertices and faces arrays.
    
    Args:
        vertices: np.array of shape (N, 3) containing vertex coordinates
        faces: np.array of shape (M, 3) containing triangle indices
    
    Returns:
        shape_msgs.msg.Mesh: The constructed mesh message
    """
    mesh_msg = Mesh()
    
    # Convert vertices to Point messages
    for vertex in vertices:
        point = Point()
        point.x = float(vertex[0])
        point.y = float(vertex[1])
        point.z = float(vertex[2])
        mesh_msg.vertices.append(point)
    
    # Convert faces to MeshTriangle messages
    for face in faces:
        triangle = MeshTriangle()
        triangle.vertex_indices = [int(face[0]), int(face[1]), int(face[2])]
        mesh_msg.triangles.append(triangle)
    
    return mesh_msg

if __name__ == "__main__":
    main()

