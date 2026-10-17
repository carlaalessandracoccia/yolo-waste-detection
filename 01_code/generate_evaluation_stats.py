# -*- coding: utf-8 -*-

"""
Evaluate the final finetune_splitov4_960 model.

A) Predict MICOLI validation images
   -> saves images with predicted bounding boxes
   -> saves bounding box coordinates as txt files

B) Print false positives per image at confidence = 0.5
"""

import os
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

import shutil
from multiprocessing import freeze_support


def main():

    """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                    Script setup
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""

    # Choose current user
    who = "Carla_windows"

    USERS = {
        "Carla_windows": Path(r"C:\Users\caco"),
        "Carla_mac": Path("/Users/carlacoccia"),
        "Carla_linux": Path("/mnt/c/Users/caco"),
    }

    if who not in USERS:
        raise ValueError(f"Unknown user profile: {who}")

    home = USERS[who]

    path_to_folder = Path(__file__).resolve().parents[1]
    data_dir = path_to_folder / "03_assets" / "02_data" / "data"
    results_dir = path_to_folder / "02_results"

    os.chdir(path_to_folder)


    # Choose device
    if torch.cuda.is_available():
        device = 0
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("Using device:", device)




    """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                    Functions
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""

    def make_runtime_yaml(yaml_file):

        source_yaml = data_dir / "yaml_files" / yaml_file
        runtime_yaml = data_dir / "yaml_files" / "_runtime" / yaml_file

        runtime_yaml.parent.mkdir(parents=True, exist_ok=True)

        with open(source_yaml, "r") as f:
            config = yaml.safe_load(f)

        # Absolute path valid for current user / operating system
        config["path"] = str(data_dir.resolve())

        with open(runtime_yaml, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

        return runtime_yaml



    """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         Predict MICOLI validation images and save bounding boxes for paper
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""

    # Predict MICOLI validation images
    micoli_test_folder = data_dir / "images" / "valid" / "micoli"
    prediction_dir = path_to_folder / "03_assets" / "04_predictions"

    # Use finetuned model
    model_path = (path_to_folder / "03_assets" / "03_saved_runs" / "runs_yolo26_finetune"
        / "finetune_splitov4_960" / "weights" / "best.pt")

    model = YOLO(str(model_path))

    runtime_yaml = make_runtime_yaml("micoli.yaml")
    
    # Calculate false positives per validation image:
    metrics = model.val(
    data=runtime_yaml,
    conf=0.5,
    imgsz=960,
    device=device,
    workers=0)

    fp = metrics.confusion_matrix.tp_fp()[1].sum()

    n_images = len(list(micoli_test_folder.glob("*.*")))

    print("FP per image:", fp / n_images)

    # Do predictions
    model.predict(
        source=str(micoli_test_folder),
        imgsz=960,
        conf=0.5,
        device=device,
        save=True,       # Save images with bounding boxes
        save_txt=True,   # Save bounding box coordinates
        save_conf=True,  # Include confidence in txt files
        project=str(prediction_dir),
        exist_ok=True
    )


    # Copy selected prediction images to the paper folder
    #-------------------------------------------------------------------------
    paper_dir = results_dir / "possible_prediction_photos_for_paper"

    selected_photos = [
        40, 90, 95, 148, 165, 168, 172, 188, 191, 199
    ]

    # Predictions created above
    source_dir = prediction_dir / "predict"

    for number in selected_photos:

        matches = list(source_dir.glob(f"{number}_*.jpg"))

        if len(matches) == 0:
            print(f"Could not find photo {number}")

        for source_file in matches:
            destination_file = paper_dir / source_file.name
            shutil.copy2(source_file, destination_file)

            print("Copied:", source_file.name)



if __name__ == "__main__":
    freeze_support()
    main()