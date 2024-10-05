import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

from termcolor import colored

matplotlib.use("agg")
import torch
from torch.utils.data import DataLoader
import pandas as pd
import torch.nn as nn

from sklearn import metrics
from tqdm import tqdm

from datasets.dataset import get_dataloaders, Dataset6DOF
from point_net.pointnet import PointNet, visualize_x_and_grasp_pose

np.random.seed(42)
torch.random.seed()


def test(
    net: PointNet, test_dataloader: DataLoader, device: str, save_dir: Path, vis_probability: float
):
    save_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.BCELoss()
    vis_dir = save_dir / "visualizations " / "test_vis"
    if vis_probability > 0:
        vis_dir.mkdir(parents=True, exist_ok=False)

    net.eval()
    test_losses = []
    test_acc = []
    test_prec = []
    test_rec = []
    test_f_score = []
    
    pred_objects = []
    ground_truth = []
    predictions = []
    with torch.no_grad():
        pbar = tqdm(test_dataloader)
        for idx, batch in enumerate(pbar):
            pcds, grasp_poses, grasp_successes, object_names = batch
            pcds = pcds.to(device)
            grasp_poses = grasp_poses.to(device)
            grasp_successes = grasp_successes.to(device)

            if idx < vis_probability * len(test_dataloader):
                visualize_x_and_grasp_pose(
                    pcds,
                    grasp_poses,
                    idx,
                    "test",
                    save_dir=str(vis_dir),
                    successful=grasp_successes,
                )

            preds = net(
                pcd=pcds,
                grasp_pose=grasp_poses,
                index=idx,
                visualizing=idx < vis_probability * len(test_dataloader),
                save_dir=str(vis_dir),
            )

            if torch.any(torch.isnan(preds)):
                print("NAN DETECTED IN PREDS")
                print(preds, grasp_successes)
                exit()

            loss = criterion(preds, grasp_successes)

            y_true = grasp_successes.cpu().view(-1, 1)
            y_pred = preds.cpu().view(-1, 1) > 0.5

            pred_objects.extend(object_names)
            ground_truth.extend(y_true.numpy().squeeze().tolist())
            predictions.extend(y_pred.numpy().squeeze().tolist())

            acc = metrics.accuracy_score(y_true, y_pred)
            prec, rec, fscore, support = metrics.precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0.0
            )

            test_losses.append(loss.cpu().numpy())
            test_acc.append(acc)
            test_prec.append(prec)
            test_rec.append(rec)
            test_f_score.append(fscore)

    mean_test_loss = np.mean(test_losses)
    mean_test_acc = np.mean(test_acc)
    mean_test_prec = np.mean(test_prec)
    mean_test_rec = np.mean(test_rec)
    mean_test_f_score = np.mean(test_f_score)

    print(colored(f"Mean test loss: {mean_test_loss}", "cyan"))
    print(colored(f"Mean test acc: {mean_test_acc}", "cyan"))
    print(colored(f"Mean test prec: {mean_test_prec}", "cyan"))
    print(colored(f"Mean test rec: {mean_test_rec}", "cyan"))
    print(colored(f"Mean test f1-score: {mean_test_f_score}", "cyan"))
    results = {
        "mean_test_loss": float(mean_test_loss),
        "mean_test_acc": float(mean_test_acc),
        "mean_test_prec": float(mean_test_prec),
        "mean_test_rec": float(mean_test_rec),
        "mean_test_f1_score": float(mean_test_f_score),
    }

    data_dict = {"object": pred_objects, "y_true": ground_truth, "y_pred": predictions}
    df = pd.DataFrame.from_dict(data_dict)
    pred_save_file = save_dir / "predictions.csv"
    df.to_csv(pred_save_file)

    for obj in pd.unique(df["object"]):
        obj_df = df[df["object"] == obj]
        acc_on_obj = metrics.accuracy_score(obj_df["y_true"], obj_df["y_pred"] > 0.5)
        prec_on_obj, rec_on_obj, fscore_on_obj, _ = metrics.precision_recall_fscore_support(obj_df["y_true"], obj_df["y_pred"] > 0.5, average="binary")
        results[str(obj)] = {
            "acc": acc_on_obj,
            "prec": prec_on_obj,
            "rec": rec_on_obj,
            "f1_score": fscore_on_obj
        }

    with (save_dir / "results.json").open("w") as f:
        json.dump(results, f, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("netpath", type=Path, help="The path to the network weights")
    parser.add_argument("pcdspath", type=Path, help="The path to the point clouds.")
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="The device to move the tensors to."
    )
    parser.add_argument("--num-workers", type=int, default=16, help="The number of dataworkers.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--vis-prob", type=float, default=0, help="Vis probability")
    parser.add_argument("--num-points", type=int, default=2048, help="The number of points per pcd.")

    args = parser.parse_args()

    _, test_dataloader = get_dataloaders(
        pcd_path=args.pcdspath, num_points=args.num_points, num_workers=args.num_workers, batch_size=args.batch_size
    )

    net = torch.load(args.netpath, map_location=args.device)

    test(
        net,
        test_dataloader,
        args.device,
        save_dir=args.netpath.parents[1] / f"test_{args.pcdspath.parent.name}_{args.pcdspath.name}",
        vis_probability=args.vis_prob,
    )
