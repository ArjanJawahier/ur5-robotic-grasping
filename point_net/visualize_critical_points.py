import argparse
import torch
from pathlib import Path
import numpy as np
np.random.seed(42)

import matplotlib.pyplot as plt

# It would be interesting to see the critical point sets, that contributed to the max pool, while considering different grasps.
# This would mean the network is looking at different points the gripper is gripping from different orientations

flat_objects = ["Banana", "Hammer", "Scissors", "MediumClamp", "PowerDrill"]
containers = [
    "ChipsCan",
    "FoamBrick",
    "CrackerBox",
    "GelatinBox",
    "PottedMeatCan",
    "TomatoSoupCan",
    "MustardBottle",
]
round_objects = ["Pear", "Strawberry", "TennisBall"]
all_objects = flat_objects + containers + round_objects

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Provide a path to a network and data",
    )
    parser.add_argument("netpath", type=Path, help="The path to the network")
    parser.add_argument("pcdpath", type=Path, help="The path to the pcd data to display")
    parser.add_argument("-n", "--nsamples", type=int, default=1, help="The number of samples to plot")
    parser.add_argument("--no-grasp", action="store_true", help="If provided, won't plot quivers for the grasp.")
    args = parser.parse_args()
    gripvars_path = args.pcdpath.parents[1] / "gripper_vars_pose" / f"{args.pcdpath.stem}.txt"
    pcd = np.load(args.pcdpath)
    labels = np.loadtxt(gripvars_path)
    fig = draw_pcds_and_grasps_before_and_after_net(pcd, labels, args.nsamples, args.netpath, args.no_grasp)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


# REUSED FROM datasets.grasp_6dof_sanity_check, TODO: REFACTOR
def draw_pcds_and_grasps_before_and_after_net(pcd, labels, num_samples, netpath: Path, no_grasps: bool):
    # Each line is structured:
    # grasp_success (1) grasp_position (3) grasp_lookat (3) grasp_up_vec (3)
    # for a total of 10 values per line
    succeeded_grasps = labels[labels[:, 0] == 1]
    sampled_indices = np.random.choice(
        range(len(succeeded_grasps)), replace=False, size=num_samples
    )
    sampled_grasps = succeeded_grasps[sampled_indices]
    ncols = 2
    nrows = num_samples

    # We have all the original samples on the left, and show the critical point set on the right
    fig, ax = plt.subplots(
        figsize=(12, 12), nrows=nrows, ncols=ncols, subplot_kw={"projection": "3d"}
    )

    if len(ax.shape) == 1:
        ax = ax[None, :]

    xlim = [np.min(pcd[:, 0]), np.max(pcd[:, 0])]
    ylim = [np.min(pcd[:, 1]), np.max(pcd[:, 1])]
    zlim = [np.min(pcd[:, 2]), np.max(pcd[:, 2])]
    for sample_idx in range(num_samples):
        grasp_pose = sampled_grasps[sample_idx, 1:10]
        xlim[0] = min(xlim[0], grasp_pose[0])
        xlim[1] = max(xlim[1], grasp_pose[0])
        ylim[0] = min(ylim[0], grasp_pose[1])
        ylim[1] = max(ylim[1], grasp_pose[1])
        zlim[0] = min(zlim[0], grasp_pose[2])
        zlim[1] = max(zlim[1], grasp_pose[2])

    for sample_idx in range(num_samples):
        for i in range(0, 2):
            ax[sample_idx, i].set_xlim(xlim)
            ax[sample_idx, i].set_ylim(ylim)
            ax[sample_idx, i].set_zlim(zlim)
            ax[sample_idx, i].set_axis_off()

    for sample_idx in range(num_samples):
        ax[sample_idx, 0].scatter3D(pcd[:, 0], pcd[:, 1], pcd[:, 2], s=1)
        if not no_grasps:
            ax[sample_idx, 0].quiver(
                sampled_grasps[sample_idx, 1],
                sampled_grasps[sample_idx, 2],
                sampled_grasps[sample_idx, 3],
                sampled_grasps[sample_idx, 4] * 0.23,
                sampled_grasps[sample_idx, 5] * 0.23,
                sampled_grasps[sample_idx, 6] * 0.23,
                color="green",
            )

    netpath = Path(netpath)
    if not "tensors" in str(netpath):
        netpath = netpath.parent / f"{netpath.stem}_tensors.pt"
        print(f"WARNING: loading {netpath}")
    net = torch.load(netpath, map_location="cuda:0")
    net.eval()

    pcd_tensor = torch.tensor(pcd, dtype=torch.float32).view(1, *pcd.shape)
    pcd_tensor = pcd_tensor.to("cuda:0")
    num_points = len(pcd)

    for sample_idx in range(num_samples):
        grasp_pose = sampled_grasps[sample_idx, 1:10]
        grasp_pose = grasp_pose.reshape(1, *grasp_pose.shape)
        grasp_pose_tensor = torch.tensor(grasp_pose, dtype=torch.float32)
        grasp_pose_tensor = grasp_pose_tensor.to("cuda:0")

        output, crit_indices = net(
            pcd=pcd_tensor, grasp_pose=grasp_pose_tensor, return_critical_point_set=True
        )

        crit_grasp_pose = crit_indices[crit_indices >= num_points]
        is_grasp_pose_critical = torch.any(crit_grasp_pose)
        is_grasp_pos_critical = num_points in crit_grasp_pose
        is_grasp_lookat_critical = (num_points + 1) in crit_grasp_pose
        is_grasp_upvec_critical = (num_points + 2) in crit_grasp_pose

        print(
            f"""
            {is_grasp_pose_critical=},
            {is_grasp_pos_critical=},
            {is_grasp_lookat_critical=},
            {is_grasp_upvec_critical=}
            """
        )

        crit_indices = crit_indices[crit_indices < num_points]
        crit = pcd[torch.unique(crit_indices).cpu()]
        if len(crit.shape) == 1:
            crit = crit[None, :]

        print(f"There were {len(pcd)} points in the pcd.")
        print(f"There are {len(crit_indices) + len(crit_grasp_pose)} global features.")
        print(f"For {len(crit_indices)} features in the global features, something in the point cloud was the most important.")
        print(f"For {len(crit_grasp_pose)} features in the global features, something in the grasp pose was the most important.")
        print(f"There are {len(crit)} points in the critical point set.")

        ax[sample_idx, 1].scatter3D(crit[:, 0], crit[:, 1], crit[:, 2], s=10)
        ax[sample_idx, 1].set_xlabel("x")
        ax[sample_idx, 1].set_ylabel("y")
        ax[sample_idx, 1].set_zlabel("z")
        if not no_grasps:
            ax[sample_idx, 1].quiver(
                sampled_grasps[sample_idx, 1],
                sampled_grasps[sample_idx, 2],
                sampled_grasps[sample_idx, 3],
                sampled_grasps[sample_idx, 4] * 0.23,
                sampled_grasps[sample_idx, 5] * 0.23,
                sampled_grasps[sample_idx, 6] * 0.23,
                color="green",
            )

    return fig


if __name__ == "__main__":
    main()
