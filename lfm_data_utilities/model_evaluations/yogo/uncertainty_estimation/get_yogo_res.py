import sys
import os
import subprocess
import csv
import torch
import numpy as np

from pathlib import Path

from yogo.model import YOGO
from yogo.utils import format_preds_and_labels
from lfm_data_utilities.model_evaluations.yogo.rank_yogo_loss import get_dataloader


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = Path(
        "/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/misc/yogo-statistics/"
    )
    folder = Path(sys.argv[1])
    data_file = base_dir / folder / "res.csv"

    if not os.path.exists(base_dir / folder):
        os.makedirs(base_dir / folder)

    model_file = "/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/yogo_models/twilight-aardvark-1094/best.pth"
    dataset_def_file = "dataset_defs/all-labelled-data.yml"

    git_branch = (
        subprocess.check_output(["git", "symbolic-ref", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )
    git_commit = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    )

    metadata = {
        "yogo_model": model_file,
        "dataset_def": dataset_def_file,
        "git_branch": git_branch,
        "git_commit": git_commit,
    }

    with open(base_dir / folder / "metadata.csv", "w") as csv_file:
        writer = csv.writer(csv_file)
        for key, value in metadata.items():
            writer.writerow([key, value])

    net, cfg = YOGO.from_pth(model_file, inference=True)
    net.to(device)

    dataloaders = get_dataloader(
        dataset_def_file,
        1,
        Sx=129,
        Sy=97,
        normalize_images=cfg["normalize_images"],
    )

    with torch.no_grad():
        for img_count, (img, label, img_path, label_path) in enumerate(dataloaders):
            preds = net(img.to(device)).cpu()

            formatted_preds, formatted_labels = format_preds_and_labels(preds, label)

            num_preds = formatted_labels[0].shape[0]

            res = np.empty((num_preds, 3))

            for pred_count, (fpred, flabel) in enumerate(
                zip(formatted_preds[0], formatted_labels[0])
            ):
                model_confidence = fpred[5:]
                model_class = torch.argmax(model_confidence)
                actual_class = int(flabel[5])
                correct = model_class == actual_class

                res[pred_count] = [
                    model_confidence[actual_class],
                    correct,
                    actual_class,
                ]

                # print(f"{correct} - ACTUAL {actual_class}, MODEL {model_class} ({model_confidence[model_class]})")

            if img_count == 0:
                all_res = res
            else:
                all_res = np.concatenate((all_res, res), axis=0)

            print(f"{img_count} - {img_path[0]}")

    np.savetxt(base_dir / folder / "res.csv", all_res, delimiter=",")
    print(f"Saved data to {base_dir / folder}")
