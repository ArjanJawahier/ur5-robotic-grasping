import torch
import einops
import torch.nn as nn
import torch.nn.functional as F
from point_net.transformation_net import TransformationNet
import matplotlib.pyplot as plt


class PointNet(nn.Module):
    # output: float = grasp success probability
    def __init__(self, width_scaling_factor: float = 1.0, num_global_features: int = 1024):
        self.saved_args = locals()
        super().__init__()
        self.input_transform_net = TransformationNet(
            4, 4, width_scaling_factor=width_scaling_factor, num_global_features=num_global_features
        )

        # self.bn0 = nn.BatchNorm1d(4)
        num_features_1 = int(64 * width_scaling_factor)
        self.conv1 = nn.Conv1d(4, num_features_1, 1)
        self.bn1 = nn.BatchNorm1d(num_features_1)
        self.feature_transform_net = TransformationNet(
            num_features_1,
            num_features_1,
            width_scaling_factor=width_scaling_factor,
            num_global_features=num_global_features,
        )

        num_features_2 = int(128 * width_scaling_factor)
        # self.bn2 = nn.BatchNorm1d(num_features_1)
        self.conv3 = nn.Conv1d(num_features_1, num_features_2, 1)
        self.bn3 = nn.BatchNorm1d(num_features_2)

        self.num_global_features = num_global_features
        self.conv4 = nn.Conv1d(num_features_2, self.num_global_features, 1)
        self.bn4 = nn.BatchNorm1d(self.num_global_features)

        num_features_4 = int(512 * width_scaling_factor)
        self.fc1 = nn.Linear(self.num_global_features, num_features_4)
        self.bn5 = nn.BatchNorm1d(num_features_4)

        num_features_5 = int(256 * width_scaling_factor)
        self.fc2 = nn.Linear(num_features_4, num_features_5)
        self.bn6 = nn.BatchNorm1d(num_features_5)
        self.fc3 = nn.Linear(num_features_5, 1)

    def forward(self, **kwargs):
        visualizing = kwargs.get("visualizing")
        vis_index = kwargs.get("index")
        x, grasp_pose = kwargs.pop("pcd"), kwargs.pop("grasp_pose")

        # if visualizing:
        #     visualize_x_and_grasp_pose(x, grasp_pose, vis_index, "input", **kwargs)

        grasp_pose = einops.rearrange(grasp_pose, "b (v d) -> b v d", d=4)
        # x = torch.concat(
        #     (x, grasp_pose), axis=1
        # )  # concat the pcd and the grasp pose in the points dimension
        # x = torch.concat(
        #     (x, torch.ones(x.shape[0], x.shape[1], 1).to(x.device)), axis=2
        # )  # Concat ones, so we can do 4x4 transforms
        # input_t_matrix = self.input_transform_net(x)
        # input_t_matrix = torch.eye(4).to(x.device)
        # tr_x = torch.matmul(x, grasp_pose) # Grasp pose is a 4x4 T mat
        # x = x.transpose(2, 1)
        tr_x = torch.matmul(x, grasp_pose.transpose(2, 1)) # Grasp pose is a 4x4 T mat
        num_points = tr_x.shape[1]
        tr_x = tr_x.transpose(2, 1)

        # features = self.bn0(tr_x)
        features = tr_x
        features = self.conv1(features)
        features = self.bn1(features)
        features = F.relu(features)

        features = features.transpose(2, 1)
        # print(features.shape)
        feature_t_matrix = self.feature_transform_net(features)
        # print(features.shape)
        # tr_features = torch.matmul(features, feature_t_matrix)
        # print(tr_features.shape)
        # tr_features = tr_features.transpose(2, 1)
        tr_features = features.transpose(2, 1)
        # print(tr_features.shape)
        # tr_features = self.bn2(tr_features) #MAYBE THIS ONE IS NOT NEEDED
        # tr_features = features  # DEBUG
        tr_features = F.relu(self.bn3(self.conv3(tr_features)))
        # print(tr_features.shape)
        tr_features = F.relu(self.bn4(self.conv4(tr_features)))
        # print(tr_features.shape)

        critpointset_indices = None
        global_features, critpointset_indices = nn.MaxPool1d(
            kernel_size=num_points,
            stride=num_points,
            ceil_mode=False,
            return_indices=True,
        )(tr_features)

        # print(global_features.shape)
        global_features = global_features.view(-1, self.num_global_features)
        # print(global_features.shape)
        mlp_features = F.relu(self.bn5(self.fc1(global_features)))
        # print(mlp_features.shape)
        mlp_features = F.relu(self.bn6(self.fc2(mlp_features)))
        # print(mlp_features.shape)
        outputs = self.fc3(mlp_features)
        # print(outputs.shape)

        # Sigmoid turns the output into a grasp success probability
        outputs = torch.sigmoid(outputs)
        outputs = outputs.squeeze()
        # print(outputs.shape)

        if kwargs.get("return_critical_point_set"):
            return outputs, critpointset_indices
        return outputs


def visualize_x_and_grasp_pose(
    x: torch.Tensor, grasp_pose: torch.Tensor, vis_index: int, suffix: str, **kwargs
):
    save_dir = kwargs.get("save_dir") or "visualization/point_net"

    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection="3d")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    v_x = x.cpu().detach()
    v_grasp_pose = grasp_pose.cpu().detach()
    s = kwargs.get("successful")
    ax.scatter3D(
        v_x[0, ..., 0],
        v_x[0, ..., 1],
        v_x[0, ..., 2],
        color="blue" if (s is not None and s.cpu().detach().numpy()[0] == 1.0) else "yellow",
    )
    ax.scatter3D(v_grasp_pose[0, ..., 0], v_grasp_pose[0, ..., 1], v_grasp_pose[0, ..., 2])

    v_pos = v_grasp_pose[0, ..., :3]
    v_lookat = v_grasp_pose[0, ..., 3:6]
    v_upvec = v_grasp_pose[0, ..., 6:9]
    ax.quiver(
        v_pos[..., 0],
        v_pos[..., 1],
        v_pos[..., 2],
        v_lookat[..., 0] / 3,
        v_lookat[..., 1] / 3,
        v_lookat[..., 2] / 3,
        color="green",
    )
    ax.quiver(
        v_pos[..., 0],
        v_pos[..., 1],
        v_pos[..., 2],
        v_upvec[..., 0] / 3,
        v_upvec[..., 1] / 3,
        v_upvec[..., 2] / 3,
        color="red",
    )
    ax.set_xlim3d(0, 1)
    ax.set_ylim3d(0, 1)
    ax.set_zlim3d(0, 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.savefig(f"{save_dir}/input{vis_index:05}_{suffix}.png")
    plt.close(fig)
