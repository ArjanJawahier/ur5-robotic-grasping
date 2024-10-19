import os
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import List, Optional

import einops
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import torch.utils.data
import tqdm
from pybullet_utils.bullet_client import BulletClient
from pyquaternion import Quaternion
from sklearn.neighbors import NearestNeighbors

from environment.utilities import Camera
from logger import get_logger
from network.hardware.device import get_device
from network.inference.post_process import post_process_output
from network.utils.data.camera_data import CameraData
from network.utils.dataset_processing.grasp import detect_grasps
from network.utils.visualisation.plot import plot_results, save_results

LOGGER = get_logger(__file__)

GRIPPER_LENGTH = 0.23


class GraspGenerator:
    IMG_WIDTH = 224
    IMG_ROTATION = -np.pi * 0.5
    CAM_ROTATION = 0
    PIX_CONVERSION = 277
    DIST_BACKGROUND = 1.115
    MAX_GRASP = 0.085

    def __init__(self, net_path, camera, depth_radius):
        self.device = get_device(force_cpu=False)
        self.net = torch.load(net_path, map_location="cpu")
        self.net = self.net.to(self.device)

        self.near = camera.near
        self.far = camera.far
        self.depth_r = depth_radius

        # Get rotation matrix
        img_center = self.IMG_WIDTH / 2 - 0.5
        self.img_to_cam = self.get_transform_matrix(
            -img_center / self.PIX_CONVERSION,
            img_center / self.PIX_CONVERSION,
            0,
            self.IMG_ROTATION,
        )
        self.cam_to_robot_base = self.get_transform_matrix(
            camera.x, camera.y, camera.z, self.CAM_ROTATION
        )

    def get_transform_matrix(self, x, y, z, rot):
        return np.array(
            [
                [np.cos(rot), -np.sin(rot), 0, x],
                [np.sin(rot), np.cos(rot), 0, y],
                [0, 0, 1, z],
                [0, 0, 0, 1],
            ]
        )

    def grasp_to_robot_frame(self, grasp, depth_img):
        """
        return: x, y, z, roll, opening length gripper, object height
        """
        # Get x, y, z of center pixel
        x_p, y_p = grasp.center[0], grasp.center[1]

        # Get area of depth values around center pixel
        x_min = np.clip(x_p - self.depth_r, 0, self.IMG_WIDTH)
        x_max = np.clip(x_p + self.depth_r, 0, self.IMG_WIDTH)
        y_min = np.clip(y_p - self.depth_r, 0, self.IMG_WIDTH)
        y_max = np.clip(y_p + self.depth_r, 0, self.IMG_WIDTH)
        depth_values = depth_img[x_min:x_max, y_min:y_max]

        # Get minimum depth value from selected area
        z_p = np.amin(depth_values)

        # Convert pixels to meters
        x_p /= self.PIX_CONVERSION
        y_p /= self.PIX_CONVERSION
        z_p = self.far * self.near / (self.far - (self.far - self.near) * z_p)

        # Convert image space to camera's 3D space
        img_xyz = np.array([x_p, y_p, -z_p, 1])
        cam_space = np.matmul(self.img_to_cam, img_xyz)

        # Convert camera's 3D space to robot frame of reference
        robot_frame_ref = np.matmul(self.cam_to_robot_base, cam_space)

        # Change direction of the angle and rotate by alpha rad
        roll = grasp.angle * -1 + (self.IMG_ROTATION)
        if roll < -np.pi / 2:
            roll += np.pi

        # Covert pixel width to gripper width
        opening_length = (
            grasp.length / int(self.MAX_GRASP * self.PIX_CONVERSION)
        ) * self.MAX_GRASP

        obj_height = self.DIST_BACKGROUND - z_p

        # return x, y, z, roll, opening length gripper
        return (
            robot_frame_ref[0],
            robot_frame_ref[1],
            robot_frame_ref[2],
            roll,
            opening_length,
            obj_height,
        )

    def predict(self, rgb, depth, n_grasps=1, show_output=False):
        depth = np.expand_dims(np.array(depth), axis=2)
        img_data = CameraData(width=self.IMG_WIDTH, height=self.IMG_WIDTH)
        x, depth_img, rgb_img = img_data.get_data(rgb=rgb, depth=depth)

        with torch.no_grad():
            xc = x.to(self.device)
            pred = self.net.predict(xc)
            pixels_max_grasp = int(self.MAX_GRASP * self.PIX_CONVERSION)
            q_img, ang_img, width_img = post_process_output(
                pred["pos"], pred["cos"], pred["sin"], pred["width"], pixels_max_grasp
            )
            save_name = None
            if show_output:
                fig = plt.figure(figsize=(10, 10))
                plot_results(
                    fig=fig,
                    rgb_img=img_data.get_rgb(rgb, False),
                    grasp_q_img=q_img,
                    grasp_angle_img=ang_img,
                    no_grasps=3,
                    grasp_width_img=width_img,
                )

                if not os.path.exists("network_output"):
                    os.mkdir("network_output")
                time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                save_name = "network_output/{}".format(time)
                fig.savefig(save_name + ".png")

            grasps = detect_grasps(
                q_img, ang_img, width_img=width_img, no_grasps=n_grasps
            )
            return grasps, save_name

    def predict_grasp(self, rgb, depth, n_grasps=1, show_output=False, **kwargs):
        predictions, save_name = self.predict(
            rgb, depth, n_grasps=n_grasps, show_output=show_output
        )
        grasps = []
        for grasp in predictions:
            x, y, z, roll, opening_len, obj_height = self.grasp_to_robot_frame(
                grasp, depth
            )
            grasps.append((x, y, z, roll, opening_len, obj_height))

        return grasps, save_name


def pcd_from_pybullet_depth(
    depth: np.ndarray,
    cam,
    filter_far: bool = True,
    seg_ids: Optional[np.ndarray] = None,
) -> np.ndarray:
    """ "Computes the pointcloud from a linear depth map made with PyBullet."""
    view_mat = np.asarray(cam.view_matrix).reshape([4, 4], order="F")
    proj_mat = np.asarray(cam.projection_matrix).reshape([4, 4], order="F")
    cam_to_world = np.linalg.inv(np.matmul(proj_mat, view_mat))

    height, width = depth.shape[:2]
    y, x = np.mgrid[-1 : 1 : 2 / height, -1 : 1 : 2 / width]
    y *= -1.0
    x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
    h = np.ones_like(z)
    pixels = np.stack([x, y, z, h], axis=1)
    if filter_far:
        pixels = pixels[z < 0.999]
        if seg_ids is not None:
            seg_ids = seg_ids.reshape(-1)
            seg_ids = seg_ids[z < 0.999]
    pixels[:, 2] = 2 * pixels[:, 2] - 1

    # turn pixels to world coordinates
    points = np.matmul(cam_to_world, pixels.T).T
    points /= points[:, 3:4]
    points = points[:, :3]
    if seg_ids is not None:
        # Return the seg ids of the points that survived the filter.
        return points.astype(np.float16), seg_ids
    else:
        return points.astype(np.float16)


def sample_n_points(
    pcd: np.ndarray, num_points: int, seg_ids: np.ndarray | None
) -> np.ndarray:
    if pcd.shape[0] < num_points:
        # Need extra points, sample from pcd and copy
        indices = np.random.choice(
            pcd.shape[0], size=num_points - pcd.shape[0], replace=True
        )
        pcd = np.concatenate([pcd, pcd[indices]], axis=0)
        if seg_ids is not None:
            seg_ids = np.concatenate((seg_ids, seg_ids[indices]), axis=0)
    else:
        # Need fewer points, sample from pcd and delete
        indices = np.random.choice(pcd.shape[0], size=num_points, replace=False)
        pcd = pcd[indices]
        if seg_ids is not None:
            seg_ids = seg_ids[indices]
    return pcd, seg_ids


def sample_grasp_poses(
    pcd: np.ndarray,
    num_poses: int,
    min_dist: float = 0.15,
    camera_lookat: np.ndarray | None = None,
    num_upvecs_per_pos: int = 1,
    seg_ids: np.ndarray | None = None,
):
    """
    Before, we sampled naively.
    Here, we sample poses with positions near the points of the pcd.
    Then we sample a random point on the pcd (A).
    Then we sample a random point B at least `min_dist`,
    and at most GRIPPER_LENGTH away from A.
    We sample point B by adding dist * (a random vector) to A. The random vector is -LOOKAT.
    We sample a random up vector orthogonal to the lookat vector.
    If the `camera_lookat` is given, we only sample lookat vectors that have a positive dot product with the `camera_lookat`
    """
    if len(pcd) == 0:
        print("WARNING: pcd has no points, cannot sample poses.")
        raise StopIteration

    # For each point in the pcd, find out how many other points are in a small neighbourhood around it
    # The more points, the higher the likelihood this point is sampled by the random choice
    # First, find out the average distance between points by sampling some points
    pcd = pcd.squeeze()

    if seg_ids is not None:
        # table has seg_ids
        pcd = pcd[seg_ids > 4]  # obj ids are higher than 4 in our case

    indices = np.arange(len(pcd))
    sample_points = pcd[
        np.random.choice(indices, size=min(len(pcd), 100), replace=False)
    ]
    pairs = combinations(sample_points, r=2)
    distances = [np.linalg.norm(pair[0] - pair[1]) for pair in pairs]
    average_distance = np.mean(distances)
    densities = np.array(
        [
            sum(
                [1 for other in pcd if np.linalg.norm(point - other) < average_distance]
            )
            for point in sample_points
        ],
        dtype=np.float32,
    )
    densities /= np.sum(densities)

    nbrs = NearestNeighbors(n_neighbors=4).fit(pcd)

    sample_indices = np.arange(len(sample_points))
    index_A = np.random.choice(
        sample_indices, size=num_poses, replace=True, p=densities
    )
    A = sample_points[index_A]
    normals = []

    # Find the normals of the points in A
    for query_point in A:
        distances, indices = nbrs.kneighbors([query_point])
        neighbors = pcd[indices[0, 1:]]  # Take the 3 points next to the point itself.
        centered_neighbors = neighbors - query_point
        cov_matrix = np.dot(centered_neighbors.T, centered_neighbors).astype(np.float32)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        normal = eigenvectors[:, np.argmin(eigenvalues)]
        normals.append(normal)

    R = np.array(normals)

    d = np.random.uniform(min_dist, GRIPPER_LENGTH, size=num_poses)
    d = np.repeat(d[:, None], 3, axis=-1)
    L = -R
    if camera_lookat is not None:
        L_camera_lookat_dots = np.dot(L, camera_lookat)
        negative_dots = L_camera_lookat_dots < 0
        L[negative_dots, 0] *= -1
    B = A + d * -L

    for pose_idx in range(num_poses):
        S = np.random.uniform(-1, 1, size=(num_upvecs_per_pos, 3))
        S /= np.repeat(np.linalg.norm(S, axis=1)[:, None], 3, axis=-1)
        U = np.cross(L[pose_idx], S)
        U /= np.repeat(np.linalg.norm(U, axis=1)[:, None], 3, axis=-1)
        for upvec_idx in range(num_upvecs_per_pos):
            yield B[pose_idx], L[pose_idx], U[upvec_idx]


def quat_from_lookat_and_upvec(
    gripper_lookat: np.ndarray, gripper_upvec: np.ndarray, pb_format: bool = True
) -> tuple[float, float, float, float] | Quaternion:
    """
    Returns a quaternion in pybullet format: a tuple of floats: (x, y, z, w) if pb_format is True.
    Returna a pyquaternion.Quaternion if pb_format is False.
    :param np.ndarray gripper_lookat: (3,) array representing the look-at vector.
    :param np.ndarray gripper_upvec: (3,) array representing the up-vector.
    :param bool pb_format: Whether to return a PyBullet quaternion (x, y, z, w).
    """

    lookat_init = np.array([1.0, 0.0, 0.0])
    upvec_init = np.array([0.0, 1.0, 0.0])

    lookat_rot_axis = np.cross(lookat_init, gripper_lookat)
    magnitude = np.linalg.norm(lookat_rot_axis)
    if magnitude == 0:
        raise ZeroDivisionError("magnitude of lookat_rot_axis is 0")
    lookat_rot_axis /= magnitude
    lookat_dot = np.clip(np.dot(lookat_init, gripper_lookat), -1.0, 1.0)
    lookat_rot_angle = np.arccos(lookat_dot)
    first_rot_quat = Quaternion(axis=lookat_rot_axis, angle=lookat_rot_angle)

    rotated_upvec = first_rot_quat.rotate(upvec_init)
    upvec_rot_axis = np.cross(rotated_upvec, gripper_upvec)
    upvec_rot_axis /= np.linalg.norm(upvec_rot_axis)
    upvec_dot = np.dot(rotated_upvec, gripper_upvec)
    upvec_dot = np.clip(upvec_dot, -1.0, 1.0)
    upvec_rot_angle = np.arccos(upvec_dot)
    second_rot_quat = Quaternion(axis=upvec_rot_axis, angle=upvec_rot_angle)
    gripper_quat = second_rot_quat * first_rot_quat  # ORDER IS IMPORTANT

    if pb_format:
        gripper_quat = (*gripper_quat.vector, gripper_quat.scalar)

    return gripper_quat


def t_mat_from_grasp_pose(
    position: np.ndarray, pybullet_quat: tuple[float, float, float, float]
):
    qx, qy, qz, qw = pybullet_quat
    convention_quat = Quaternion(w=qw, x=qx, y=qy, z=qz)
    grasp_rot_matrix = convention_quat.rotation_matrix
    grasp_T_mat = np.eye(4)
    grasp_T_mat[:3, :3] = grasp_rot_matrix
    grasp_T_mat[:3, 3] = position
    return grasp_T_mat


def determine_physical_graspability(
    grasp_pos: np.ndarray,
    grasp_orn: np.ndarray,
    robot_id: int,
    object_ids: List[int],
    client: BulletClient,
    ee_link_idx: int = 7,
) -> bool:
    """
    Determine whether the grasp pose can be reached using the pybullet IK solver.
    If it's not reachable (according to the joint constraints), will return False
    Determine whether the grasp pose would lead to collisions.
    """
    num_joints = client.getNumJoints(robot_id)
    
    joint_lower_limits = [p.getJointInfo(robot_id, i)[8] for i in range(num_joints)]
    joint_upper_limits = [p.getJointInfo(robot_id, i)[9] for i in range(num_joints)]
    joint_angles = client.calculateInverseKinematics(
        bodyUniqueId=robot_id,
        endEffectorLinkIndex=ee_link_idx,
        targetPosition=grasp_pos,
        targetOrientation=grasp_orn,
        lowerLimits=joint_lower_limits,
        upperLimits=joint_upper_limits
    )

    # Loop through each joint and get its constraints
    movable_joint_idx = 0
    for joint_index in range(num_joints):
        joint_info = p.getJointInfo(robot_id, joint_index)
        # Extract the joint name
        joint_name = joint_info[1].decode("utf-8")

        # Extract joint type (revolute, prismatic, fixed, etc.)
        joint_type = joint_info[2]

        if joint_type == p.JOINT_FIXED:
            continue

        # Extract joint limits (for revolute and prismatic joints)
        joint_lower_limit = joint_info[8]
        joint_upper_limit = joint_info[9]

        # Extract joint velocity and torque limits (if specified)
        joint_max_velocity = joint_info[11]
        joint_max_torque = joint_info[10]

        # print(f"Joint {joint_index} ({joint_name}):")
        # print(f"  - Type: {joint_type}")
        # print(f"  - Lower limit: {joint_lower_limit}")
        # print(f"  - Upper limit: {joint_upper_limit}")
        # print(f"  - Max velocity: {joint_max_velocity}")
        # print(f"  - Max torque: {joint_max_torque}")
        # print(joint_angles[movable_joint_idx])
        if joint_angles[movable_joint_idx] < joint_lower_limit or joint_angles[movable_joint_idx] > joint_upper_limit:
            # Grasp is not physically attainable
            return False
        movable_joint_idx += 1

    # Check whether the joints would lead to collisions
    current_joint_positions = [
        client.getJointState(robot_id, i)[0] for i in range(num_joints)
    ]

    i = 0
    for joint_index in range(num_joints):
        joint_info = p.getJointInfo(robot_id, joint_index)
        if joint_info[2] == p.JOINT_FIXED:
            continue
        client.resetJointState(robot_id, i, joint_angles[i])
        i += 1

    is_colliding = False
    for obj_id in object_ids:
        collision_points = p.getClosestPoints(bodyA=robot_id, bodyB=obj_id, distance=0)
        if len(collision_points) > 0:
            is_colliding = True
            break

    # Reset back the joints
    for i in range(num_joints):
        client.resetJointState(robot_id, i, current_joint_positions[i])

    return is_colliding


class GraspGenerator6DOF(GraspGenerator):
    def __init__(
        self,
        net_path: Path,
        camera: Camera,
        depth_radius: int,
        robot_id: int,
        pb_client: BulletClient,
        num_points_per_pcd: int = 2048,
        num_sampled_grasps: int = 10,
        ee_link_idx: int = 7
    ):
        super().__init__(net_path, camera, depth_radius)
        self.camera = camera
        self.num_points_per_pcd = num_points_per_pcd
        self.num_sampled_grasps = num_sampled_grasps
        self.robot_id = robot_id
        self.pb_client = pb_client
        self.ee_link_idx = ee_link_idx

    def predict(
        self, rgb: np.ndarray, depth: np.ndarray, seg_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # For the point net, we have to convert the depth array to a pointcloud
        pcd, seg_ids = pcd_from_pybullet_depth(depth, self.camera, seg_ids=seg_ids)

        if pcd.shape[0] == 0:
            # No points
            raise ValueError("No points in the point cloud, cannot predict")

        pcd, seg_ids = sample_n_points(
            pcd, num_points=self.num_points_per_pcd, seg_ids=seg_ids
        )
        pcd = pcd[None, ...]
        sampled_grasps = sample_grasp_poses(
            pcd,
            num_poses=self.num_sampled_grasps,
            min_dist=0.15,
            camera_lookat=self.camera.cam_lookat,
            seg_ids=seg_ids,
        )
        pcd = torch.concat(
            (torch.FloatTensor(pcd), torch.ones((1, pcd.shape[1], 1))), axis=2
        )
        pcd = pcd.to(self.device)
        preds = []
        grasp_positions = []
        grasp_orns = []
        grasp_lookats = []
        for gripper_pos, gripper_lookat, gripper_upvec in tqdm.tqdm(sampled_grasps):
            try:
                gripper_quat = quat_from_lookat_and_upvec(
                    gripper_lookat, gripper_upvec, pb_format=True
                )
            except ZeroDivisionError:
                continue

            physical_graspability = determine_physical_graspability(
                gripper_pos,
                gripper_quat,
                self.robot_id,
                np.unique(seg_ids),
                self.pb_client,
                self.ee_link_idx,
            )
            if not physical_graspability:
                continue

            t_mat = t_mat_from_grasp_pose(gripper_pos, gripper_quat)
            inv_t_mat = np.linalg.inv(t_mat)
            inv_t_mat = torch.FloatTensor(inv_t_mat[None, ...]).to(self.device)
            inv_t_mat = einops.rearrange(inv_t_mat, "b h w -> b (h w)")
            pred = self.net(pcd=pcd, grasp_pose=inv_t_mat)
            preds.append(pred.cpu().detach().numpy())
            grasp_positions.append(gripper_pos)
            grasp_orns.append(gripper_quat)
            grasp_lookats.append(gripper_lookat)

        LOGGER.info(f"{len(preds)} grasps deemed attainable")

        preds = np.array(preds).squeeze()
        rank_indices = np.flip(np.argsort(preds))
        ranked_positions = np.array(grasp_positions)[rank_indices]
        ranked_orns = np.array(grasp_orns)[rank_indices]
        ranked_lookats = np.array(grasp_lookats)[rank_indices]
        return ranked_positions, ranked_orns, ranked_lookats

    def predict_grasp(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        seg_ids: np.ndarray,
        n_grasps: int = 1,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        positions, orientations, lookats = self.predict(rgb, depth, seg_ids)
        # represent grasps as 10-dim (*pos, *quat, *lookat)
        grasps = np.hstack((positions, orientations, lookats))
        return grasps[:n_grasps], None
