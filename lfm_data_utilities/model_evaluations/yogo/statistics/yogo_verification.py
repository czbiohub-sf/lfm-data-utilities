import sys
import os
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

    net, cfg = YOGO.from_pth(
        "/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/yogo_models/honest-sweep-51/best.pth",
        inference=True,
    )
    net.to(device)

    dataloaders = get_dataloader(
        "/hpc/projects/flexo/MicroscopyData/Bioengineering/LFM_scope/biohub-labels/dataset_defs/4verification-all-labelled-data-test.yml",
        1,
        Sx=129,
        Sy=97,
        normalize_images=cfg["normalize_images"],
    )

    with torch.no_grad():
        for img_count, (img, label, image_path, label_path) in enumerate(dataloaders):
            preds = net(img.to(device)).cpu()

            formatted_preds, formatted_labels = format_preds_and_labels(
                preds, label
            )

            num_preds = formatted_labels[0].shape[0]

            res = np.empty((num_preds, 2))

            for pred_count, (fpred, flabel) in enumerate(
                zip(formatted_preds[0], formatted_labels[0])
            ):
                model_confidence = fpred[5:]
                model_class = torch.argmax(model_confidence)
                actual_class = int(flabel[5])
                correct = model_class == actual_class

                res[pred_count] = [model_confidence[actual_class], correct]

                # print(f"{correct} - ACTUAL {actual_class}, MODEL {model_class} ({model_confidence[model_class]})")

            if img_count == 0:
                all_res = res
            else:
                all_res = np.concatenate((all_res, res), axis=0)

            print(img_count)

    np.savetxt(data_file, all_res, delimiter=",")
    print(f"Saved data to {data_file}")
