import torch
import einops
import torch.nn as nn
import torch.nn.functional as F
from point_net.transformation_net import TransformationNet
import matplotlib.pyplot as plt


class MLP(nn.Module):
    # output: float = grasp success probability
    def __init__(self, width_scaling_factor: float = 1.0, num_points_per_pcd:int=1024):
        self.saved_args = locals()
        super().__init__()
        # num_features_1 = int(64 * width_scaling_factor)
        # self.conv1 = nn.Conv1d(4, num_features_1, 1)
        # self.conv1 = nn.Linear(4*1024 + 16, num_features_1)
        self.conv1 = nn.Linear(4*num_points_per_pcd+9, 1024)
        self.output_layer = nn.Linear(1024, 1)

        # self.bn1 = nn.BatchNorm1d(num_features_1)
        # num_features_2 = int(128 * width_scaling_factor)
        # self.conv3 = nn.Conv1d(num_features_1, num_features_2, 1)
        # self.conv3 = nn.Linear(num_features_1, num_features_2)
        # self.bn3 = nn.BatchNorm1d(num_features_2)

        # num_features_3 = int(1024 * width_scaling_factor)
        # self.conv4 = nn.Conv1d(num_features_2, num_features_3, 1)
        # self.conv4 = nn.Linear(num_features_2, num_features_3)
        # self.bn4 = nn.BatchNorm1d(self.num_global_features)

        # num_features_4 = int(512 * width_scaling_factor)
        # num_features_4 = int(512 * width_scaling_factor)
        # num_features_5 = int(256 * width_scaling_factor)

        # self.fc1 = nn.Linear(num_features_3, num_features_4)
        # self.bn5 = nn.BatchNorm1d(num_features_4)
        # self.bn6 = nn.BatchNorm1d(num_features_5)
        # self.fc3 = nn.Linear(num_features_4, 1)

    def forward(self, **kwargs):
        x, grasp_pose = kwargs.pop("pcd"), kwargs.pop("grasp_pose")
        tr_x = einops.rearrange(x, "b h c -> b (h c)")
        batch_size = tr_x.shape[0]
        gp = grasp_pose.view(batch_size, -1)
        tr_x = torch.concat((tr_x, gp), dim=1)
        outputs = self.conv1(tr_x)
        outputs = F.relu(outputs)
        outputs = self.output_layer(outputs)
        outputs = torch.sigmoid(outputs)
        return outputs.squeeze()

        # print(tr_x.shape)
        # features = self.bn0(tr_x)
        features = self.conv1(tr_x)
        # print(features.shape)
        # features = self.bn1(features)
        features = F.relu(features)

        tr_features = features
        tr_features = self.conv3(tr_features)
        # print(tr_features.shape)
        # tr_features = self.bn3(tr_features)
        tr_features = F.relu(tr_features)

        tr_features = self.conv4(tr_features)
        # print(tr_features.shape)
        # tr_features = self.bn4(tr_features)
        tr_features = F.relu(tr_features)#.transpose(2, 1)
        # global_features, critpointset_indices = nn.MaxPool1d(
        #     kernel_size=num_points,
        #     # stride=num_points,
        #     ceil_mode=False,
        #     return_indices=True,
        # )(tr_features)
        # mlp_features = self.fc1(global_features.squeeze())

        # print(tr_features.shape, gp.shape)
        # print(mlp_features.shape)
        mlp_features = self.fc1(mlp_features)
        mlp_features = F.relu(mlp_features)
        # mlp_features = self.fc2(mlp_features)
        # print(mlp_features.shape)
        # mlp_features = F.relu(mlp_features)
        outputs = self.fc3(mlp_features)

        # Sigmoid turns the output into a grasp success probability
        outputs = torch.sigmoid(outputs)
        outputs = outputs.squeeze()
        return outputs

