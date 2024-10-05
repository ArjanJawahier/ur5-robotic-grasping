import argparse
from pathlib import Path

import matplotlib
import numpy as np

from termcolor import colored

matplotlib.use("agg")
import torch
from tqdm import tqdm

from point_net.pointnet import PointNet

np.random.seed(42)
torch.random.seed()


def rank_and_detect(
    net: PointNet, pcdspath: Path, device: str,num_points: int = 2048
):
    """
    Given a PointNet and a path to point clouds,
    this function ranks all the grasps and outputs the precision of the top-k ranks
    """

    net.eval()
    tp, tn, fp, fn = 0, 0, 0, 0
    with torch.no_grad():
        top_1_correct = []
        top_2_correct = []
        top_3_correct = []
        top_5_correct = []
        top_10_correct = []
        for pcd_path in tqdm(sorted(list(pcdspath.rglob("*.npy"))), colour="red"):
            pcd = np.load(pcd_path)
            pcd = np.concatenate((pcd, np.ones((pcd.shape[0], 1))), axis=1)
            if pcd.shape[0] > num_points:

                indices = np.random.choice(np.arange(pcd.shape[0]), size=num_points, replace=False)
                pcd = pcd[indices]

            gripper_var_path = pcdspath.parent / "gripper_vars_pose_inverted_t_mat" / f"{pcd_path.stem}.txt"
            try:
                gripper_vars = np.loadtxt(gripper_var_path)
            except FileNotFoundError:
                print(f"Skipping {pcd_path}, because {gripper_var_path} does not exist")
                continue
            grasp_successes = gripper_vars[:, 0]
            grasp_poses = gripper_vars[:, 1:17]

            preds = []

            pcd = torch.FloatTensor(pcd[None, ...])
            pcd = pcd.to(device)

            for gp_index, gp in enumerate(grasp_poses):
                gp = torch.FloatTensor(gp[None, ...])
                gp = gp.to(device)
                y_pred = net(pcd=pcd, grasp_pose=gp)
                preds.append(y_pred.cpu().numpy().squeeze())

            preds = np.array(preds).squeeze()

            for gs, pred in zip(grasp_successes, preds):
                if pred >= 0.5:
                    if gs == 1:
                        tp += 1
                    else:
                        fp += 1
                elif pred < 0.5:
                    if gs == 1:
                        fn += 1
                    else:
                        tn += 1

            rank_indices = np.flip(np.argsort(preds))

            ranked_grasp_successes = grasp_successes[rank_indices]
            ranked_preds = preds[rank_indices]
            for s, p in zip(ranked_grasp_successes[:8], ranked_preds[:8]):
                print(f"{pcd_path} Succ: {s}, pred: {p}")

            for k, array in zip(
                [1, 2, 3, 5, 10],
                [top_1_correct, top_2_correct, top_3_correct, top_5_correct, top_10_correct],
            ):
                for i in range(k):
                    if ranked_grasp_successes[i] == 1:
                        array.append(True)
                        break
                else:
                    array.append(False)

    for name, array in zip(
        ["Top1", "Top2", "Top3", "Top5", "Top10"],
        [top_1_correct, top_2_correct, top_3_correct, top_5_correct, top_10_correct],
    ):
        print(colored(f"{name}-accuracy: {round(np.sum(array) / len(array), 4)}", "green"))
    

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    print("Precision: ", round(precision, 5), " Recall: ", round(recall, 5))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("netpath", type=Path, help="The path to the network weights")
    parser.add_argument("pcdspath", type=Path, help="The path to the point clouds.")
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="The device to move the tensors to."
    )
    parser.add_argument("--num-points", type=int, default=2048, help="Number of points in the pcds")
    args = parser.parse_args()
    net = torch.load(args.netpath, map_location=args.device)
    rank_and_detect(
        net,
        pcdspath=args.pcdspath,
        device=args.device,
        num_points=args.num_points,
    )
