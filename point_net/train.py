import sys
from pathlib import Path

import matplotlib
from termcolor import colored

matplotlib.use("agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset import Dataset6DOF, get_dataloaders
from point_net.pointnet import PointNet, visualize_x_and_grasp_pose
from point_net.mlp import MLP


np.random.seed(42)
torch.random.seed()


def train(
    net: PointNet,
    num_epochs: int,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    device: str,
    save_dir: Path,
    vis_probability: float,
    learning_rate: float,
):
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate)
    save_net_info(pnet, optimizer, save_dir)
    (save_dir / "weights").mkdir(parents=True, exist_ok=True)

    criterion = nn.BCELoss()
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_train_accuracies = []
    epoch_val_accuracies = []
    epoch_train_precisions = []
    epoch_val_precisions = []
    epoch_train_recalls = []
    epoch_val_recalls = []
    epoch_train_f1_scores = []
    epoch_val_f1_scores = []

    best_val_loss, best_epoch = np.inf, 0

    print(
        f"{'Epoch':>7}{'Loss':>12}{'Accuracy':>12}{'Precision':>12}{'Recall':>12}{'F1-Score':>12}"
    )
    for epoch_idx in range(num_epochs):
        vis_dir = save_dir / "visualizations" / f"epoch{epoch_idx:03}"
        if vis_probability > 0:
            vis_dir.mkdir(parents=True, exist_ok=False)
        # Training part of the epoch
        train_losses = np.full((500,), np.nan)
        train_accuracies = np.full((500,), np.nan)
        train_precisions = np.full((500,), np.nan)
        train_recalls = np.full((500,), np.nan)
        train_f1_scores = np.full((500,), np.nan)
        epoch_preds = np.full((500 * args.batch_size,), np.nan)

        net.train()
        pbar = tqdm(train_dataloader, colour="green")

        for idx, batch in enumerate(pbar):
            net.zero_grad()
            optimizer.zero_grad()
            pcds, grasp_poses, grasp_successes, object_names = batch
            pcds = pcds.to(device)
            grasp_poses = grasp_poses.to(device)
            grasp_successes = grasp_successes.to(device)

            preds = net(pcd=pcds, grasp_pose=grasp_poses)

            if torch.any(torch.isnan(preds)):
                print("NAN DETECTED IN PREDS")
                print(preds, grasp_successes)
                sys.exit(-1)

            loss = criterion(preds, grasp_successes)
            loss.backward()
            optimizer.step()
            y_true, y_pred = (
                grasp_successes.cpu().view(-1, 1),
                preds.cpu().view(-1, 1) > 0.5,
            )

            prec, rec, f1_score, _ = metrics.precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0.0
            )

            # This buffer index keeps track of the latest addition to the lists
            buffer_index = idx % 500
            epoch_preds[
                args.batch_size * buffer_index : args.batch_size * (buffer_index + 1)
            ] = (preds.detach().cpu().numpy().flatten())
            train_losses[buffer_index] = float(loss.detach().cpu().numpy())
            train_accuracies[buffer_index] = metrics.accuracy_score(y_true, y_pred)
            train_precisions[buffer_index] = prec
            train_recalls[buffer_index] = rec
            train_f1_scores[buffer_index] = f1_score

            running_train_loss = np.nanmean(train_losses)
            running_train_acc = np.nanmean(train_accuracies)
            running_train_prec = np.nanmean(train_precisions)
            running_train_rec = np.nanmean(train_recalls)
            running_train_f1_score = np.nanmean(train_f1_scores)

            pbar_description = f"{epoch_idx:3}/{num_epochs-1:3}{running_train_loss:12.3}{running_train_acc:12.3}{running_train_prec:12.3}{running_train_rec:12.3}{running_train_f1_score:12.3} av pred: {round(float(np.nanmean(epoch_preds)), 5):1.5}, std: {round(float(np.nanstd(epoch_preds)), 5):1.5}"
            pbar.set_description(pbar_description)


        epoch_train_losses.append(np.nanmean(train_losses))
        epoch_train_accuracies.append(np.nanmean(train_accuracies))
        epoch_train_precisions.append(np.nanmean(train_precisions))
        epoch_train_recalls.append(np.nanmean(train_recalls))
        epoch_train_f1_scores.append(np.nanmean(train_f1_scores))

        # Validation part of the epoch
        net.eval()
        with torch.no_grad():
            val_losses = []
            val_accuracies = []
            val_precisions = []
            val_recalls = []
            val_f1_scores = []
            pbar = tqdm(val_dataloader, colour="yellow")
            for batch in pbar:
                pcds, grasp_poses, grasp_successes, object_names = batch
                pcds = pcds.to(device)
                grasp_poses = grasp_poses.to(device)
                grasp_successes = grasp_successes.to(device)
                preds = net(pcd=pcds, grasp_pose=grasp_poses)
                loss = criterion(preds, grasp_successes)
                val_losses.append(float(loss.detach().cpu().numpy()))
                y_true, y_pred = (
                    grasp_successes.cpu().view(-1, 1),
                    preds.cpu().view(-1, 1) > 0.5,
                )
                val_accuracies.append(metrics.accuracy_score(y_true, y_pred))
                val_prec, val_rec, val_f1_score, _ = (
                    metrics.precision_recall_fscore_support(
                        y_true, y_pred, average="binary", zero_division=0.0
                    )
                )
                val_precisions.append(val_prec)
                val_recalls.append(val_rec)
                val_f1_scores.append(val_f1_score)
                pbar_description = f"{epoch_idx:3}/{num_epochs-1:3}{np.nanmean(val_losses):12.03}{np.nanmean(val_accuracies):12.03}{np.nanmean(val_precisions):12.03}{np.nanmean(val_recalls):12.03}{np.nanmean(val_f1_scores):12.03}"
                pbar.set_description(pbar_description)

        mean_val_loss = np.nanmean(val_losses)
        epoch_val_losses.append(mean_val_loss)
        mean_val_accuracy = np.nanmean(val_accuracies)
        epoch_val_accuracies.append(mean_val_accuracy)

        mean_val_prec = np.nanmean(val_precisions)
        epoch_val_precisions.append(mean_val_prec)
        mean_val_rec = np.nanmean(val_recalls)
        epoch_val_recalls.append(mean_val_rec)
        mean_val_f1_score = np.nanmean(val_f1_scores)
        epoch_val_f1_scores.append(mean_val_f1_score)

        checkpoint_info = {
            "epoch": epoch_idx,
            "loss": mean_val_loss,
            "acc": mean_val_accuracy,
            "prec": mean_val_prec,
            "rec": mean_val_rec,
            "f1_score": mean_val_f1_score,
            "model_state_dict": net.state_dict(),
            "model_T_net_state_dict": net.feature_transform_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": net.saved_args,
        }
        # Save the checkpoint after each epoch
        torch.save(checkpoint_info, save_dir / "weights" / f"epoch_{epoch_idx}.pt")
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_epoch = epoch_idx
            # Save the best checkpoint
            torch.save(checkpoint_info, save_dir / "weights" / f"best.pt")
            torch.save(net, save_dir / "weights" / f"best_tensors.pt")
        torch.save(net, save_dir / "weights" / f"epoch_{epoch_idx}_tensors.pt")

        plot_losses(
            train_losses=epoch_train_losses,
            val_losses=epoch_val_losses,
            save_dir=save_dir,
            best_epoch=best_epoch,
        )

        plot_scores(
            train_scores=epoch_train_accuracies,
            val_scores=epoch_val_accuracies,
            save_dir=save_dir,
            best_epoch=best_epoch,
            metric="Accuracy",
        )

        plot_scores(
            train_scores=epoch_train_precisions,
            val_scores=epoch_val_precisions,
            save_dir=save_dir,
            best_epoch=best_epoch,
            metric="Precision",
        )

        plot_scores(
            train_scores=epoch_train_recalls,
            val_scores=epoch_val_recalls,
            save_dir=save_dir,
            best_epoch=best_epoch,
            metric="Recall",
        )

        plot_scores(
            train_scores=epoch_train_f1_scores,
            val_scores=epoch_val_f1_scores,
            save_dir=save_dir,
            best_epoch=best_epoch,
            metric="F1-Score",
        )

    return epoch_train_losses, epoch_val_losses


def save_net_info(net: nn.Module, optimizer: torch.optim.Optimizer, save_dir: Path):
    save_path = save_dir / "net_info.txt"
    with save_path.open("w") as f:

        # Print model's state_dict
        f.write("Model's state_dict:\n")
        for param_tensor in net.state_dict():
            f.write(f"{param_tensor} \t {net.state_dict()[param_tensor].size()} \n")

        # Print optimizer's state_dict
        f.write("\nOptimizer's state_dict:\n")
        for var_name in optimizer.state_dict():
            f.write(f"{var_name} \t {optimizer.state_dict()[var_name]}\n")


def plot_losses(
    train_losses: list[float],
    val_losses: list[float],
    save_dir: Path,
    best_epoch: int,
    fig_name: str = "loss_plot",
):
    assert (
        type(train_losses) == list and type(val_losses) == list
    ), "Only lists are supported within train.py :: plot_losses"
    all_losses = train_losses + val_losses
    epoch_types = ["train" for _ in train_losses] + ["val" for _ in val_losses]
    epochs = list(range(len(train_losses))) + list(range(len(val_losses)))
    data = {"loss": all_losses, "epoch": epochs, "epoch_type": epoch_types}
    fig = plt.figure()
    sns.lineplot(data=data, x="epoch", y="loss", hue="epoch_type")
    sns.scatterplot(
        x=[best_epoch],
        y=[val_losses[best_epoch]],
        edgecolor="black",
        color="green",
        marker="*",
        s=400,
    )
    plt.title(f"BCE Loss against epoch. Best epoch: {best_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.savefig(save_dir / f"{fig_name}.png")
    plt.close(fig)


def plot_scores(
    train_scores: list[float],
    val_scores: list[float],
    save_dir: Path,
    best_epoch: int,
    metric: str = "Accuracy",
):
    assert (
        type(train_scores) == list and type(val_scores) == list
    ), "Only lists are supported within train.py :: plot_scores"
    all_scores = train_scores + val_scores
    epoch_types = ["train" for _ in train_scores] + ["val" for _ in val_scores]
    epochs = list(range(len(train_scores))) + list(range(len(val_scores)))
    data = {"score": all_scores, "epoch": epochs, "epoch_type": epoch_types}
    fig = plt.figure()
    sns.lineplot(data=data, x="epoch", y="score", hue="epoch_type")
    sns.scatterplot(
        x=[best_epoch],
        y=[val_scores[best_epoch]],
        edgecolor="black",
        color="green",
        marker="*",
        s=400,
    )
    plt.title(f"{metric} against epochs. Best epoch: {best_epoch}")
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.savefig(save_dir / f"{metric}_plot.png")
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--pcdpath", type=Path, required=True, help="The path to the dataset"
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--num-points", type=int, default=2048, help="Num points per pcd"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=30,
        help="The number of epochs trained. One epoch is an iteration over the whole dataset.",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda:0",
        choices=["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cpu"],
        help="The device on which to train the network.",
    )
    parser.add_argument(
        "--name", default="NAME", help="Save dir name (within runs/train/)"
    )
    parser.add_argument(
        "--width-scaling-factor",
        type=float,
        default=1.0,
        help="The factor that gets multiplied to the number of features in the PointNet.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=16,
        help="The number of workers with which to fetch data.",
    )
    parser.add_argument(
        "--vis-prob",
        type=float,
        default=0.0,
        help="The probability visualizations are made for each training iteration. Recommended to keep close to 0.",
    )
    parser.add_argument(
        "-lr", "--learning-rate", type=float, default=1e-5, help="The learning rate."
    )
    parser.add_argument(
        "-K",
        "--num-global-features",
        type=int,
        default=1024,
        help="The number of global features taken from the point clouds.",
    )
    args = parser.parse_args()

    save_dir: Path = Path(__file__).parent / "runs" / "train" / args.name
    save_dir_counter = 0
    while save_dir.exists():
        save_dir_counter += 1
        save_dir = save_dir.parent / f"{args.name}_{save_dir_counter:02}"
    save_dir.mkdir(parents=True, exist_ok=False)

    pnet = PointNet(
        width_scaling_factor=args.width_scaling_factor,
        num_global_features=args.num_global_features,
    ).to(
        args.device
    )  # 1 output grasp success probability

    train_dataloader, test_dataloader = get_dataloaders(
        pcd_path=args.pcdpath,
        num_points=args.num_points,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )

    train(
        pnet,
        args.num_epochs,
        train_dataloader=train_dataloader,
        val_dataloader=test_dataloader,
        device=args.device,
        save_dir=save_dir,
        vis_probability=args.vis_prob,
        learning_rate=args.learning_rate,
    )
