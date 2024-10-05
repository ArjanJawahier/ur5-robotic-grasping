import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformationNet(nn.Module):

    def __init__(
        self,
        input_dim,
        output_dim,
        width_scaling_factor: float = 1.0,
        num_global_features: int = 1024,
    ):
        super(TransformationNet, self).__init__()
        self.num_global_features = num_global_features
        self.output_dim = output_dim
        self.conv_1 = nn.Conv1d(input_dim, int(64 * width_scaling_factor), 1)
        self.bn_1 = nn.BatchNorm1d(int(64 * width_scaling_factor))
        self.conv_2 = nn.Conv1d(int(64 * width_scaling_factor), int(128 * width_scaling_factor), 1)
        self.bn_2 = nn.BatchNorm1d(int(128 * width_scaling_factor))
        self.conv_3 = nn.Conv1d(int(128 * width_scaling_factor), self.num_global_features, 1)
        self.bn_3 = nn.BatchNorm1d(self.num_global_features)

        self.fc_1 = nn.Linear(self.num_global_features, int(512 * width_scaling_factor))
        self.bn_4 = nn.BatchNorm1d(int(512 * width_scaling_factor))
        self.fc_2 = nn.Linear(int(512 * width_scaling_factor), int(256 * width_scaling_factor))
        self.bn_5 = nn.BatchNorm1d(int(256 * width_scaling_factor))
        self.fc_3 = nn.Linear(int(256 * width_scaling_factor), self.output_dim * self.output_dim)

    def forward(self, x):
        num_points = x.shape[1]
        x = x.transpose(2, 1)
        x = F.relu(self.bn_1(self.conv_1(x)))
        x = F.relu(self.bn_2(self.conv_2(x)))
        x = F.relu(self.bn_3(self.conv_3(x)))
        x = nn.MaxPool1d(num_points)(x)
        x = x.view(-1, self.num_global_features)
        x = F.relu(self.bn_4(self.fc_1(x)))
        x = F.relu(self.bn_5(self.fc_2(x)))
        x = self.fc_3(x)
        identity_matrix = torch.eye(self.output_dim).to(x.device)
        x = x.view(-1, self.output_dim, self.output_dim) + identity_matrix
        return x


if __name__ == "__main__":
    # Test the network
    net = TransformationNet(3, 3)
    point_cloud = torch.randn(16, 200, 3)
    transform_matrix = net(point_cloud)
    print(transform_matrix.shape)
    net = TransformationNet(64, 64)
    pcd = torch.randn(16, 200, 64)
    transform_matrix = net(pcd)
    print(transform_matrix.shape)
