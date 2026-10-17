# -*- coding: utf-8 -*-
"""
This file trains a YOLO26 model using different combinations of datasets.
It saves important metrics to the 02_results/results_used_in_paper.xlsx and copies
the YOLO output run folder (that includes the weights) to 03_assets/03_saved_runs.

In order keep the correct format for the training folder, it moves photos
and labels back and forth between the 03_assets/02_data/99_temp and the main
03_assets/02_data/data training folder, according to which datasets should be used
for the training.
"""

import os
from ultralytics import YOLO
from datetime import date
import shutil
import torch
import stat
import pandas as pd
import glob
import re
from pathlib import Path
import yaml


"""%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                  Script setup
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
def main():
    # change working directory
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
    data_dir = Path("03_assets") / "02_data" / "data"
    results_file = path_to_folder / "02_results" / "results_used_in_paper.xlsx"

    if results_file.exists():
        results = pd.read_excel(results_file)
    else:
        results = pd.DataFrame()


    os.chdir(path_to_folder)

    # Ensure to use the GPU runtime
    gpu_info = os.system("nvidia-smi")
    torch.cuda.is_available()


    """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                      Functions
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""

    def remove_readonly(func, path, _):
        """
        Clear the readonly bit and reattempt the removal
        """
        os.chmod(path, stat.S_IWRITE)
        func(path)

    def copy_to_data_folder(dictionary_entry):

        folder_name = dictionary_entry["folder_name"]
        pattern_regex = dictionary_entry["pattern_regex"]

        source_dir = temp_dir / folder_name

        move = False
        for item in (source_dir / "images" / "train").iterdir():
            if re.match(pattern_regex, item.name):
                move = True
                break

        if move:
            for item in source_dir.rglob("*"):
                if item.is_file() and item.suffix.lower() in [".jpg", ".jpeg", ".txt"]:

                    relative_path = item.relative_to(source_dir)

                    dst = data_dir / relative_path

                    dst.parent.mkdir(parents=True, exist_ok=True)

                    shutil.move(str(item), str(dst))

    def remove_from_data_folder(dictionary_entry):

        folder_name = dictionary_entry["folder_name"]
        pattern_regex = dictionary_entry["pattern_regex"]
        pattern_bash = dictionary_entry["pattern_bash"]

        move = False

        for item in train_dir.iterdir():
            if re.match(pattern_regex, item.name):
                move = True
                break

        if move:

            for folder_type in ["images", "labels"]:

                source_dir = data_dir / folder_type / "train"
                target_dir = temp_dir / folder_name / folder_type / "train"

                for item in source_dir.glob(pattern_bash):

                    target_dir.mkdir(parents=True, exist_ok=True)

                    dst = target_dir / item.name

                    shutil.move(str(item), str(dst))

    def make_runtime_yaml(yaml_file):
        source_yaml = data_dir / "yaml_files" / yaml_file
        runtime_yaml = data_dir / "yaml_files" / "_runtime" / yaml_file

        runtime_yaml.parent.mkdir(parents=True, exist_ok=True)

        with open(source_yaml, "r") as f:
            config = yaml.safe_load(f)

        # Absolute path valid for the current user / operating system
        config["path"] = str(data_dir.resolve())

        with open(runtime_yaml, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

        return runtime_yaml

    """%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                            Training and saving results
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"""
    """This part loops through several dataset combinations and trains with each combination"""

    #
    combinations = [
        ["micoli"],
        ["taco"],
        ["taco", "taco_extended"],
        ["micoli", "taco", "taco_extended"],
        ["micoli", "taco", "taco_extended", "plastopol"]
    ]

    # train_dir = os.path.join(data_dir,"images/train")
    # temp_dir= "03_assets/02_data/99_temp"
    train_dir = data_dir / "images" / "train"
    temp_dir = Path("03_assets") / "02_data" / "99_temp"

    # dictionary to identify photos and folders from each dataset and be able to
    # move them to the main folder and back
    data_dict = {
        "micoli": {"pattern_bash": '*-[ru][ep].[jJt][pPx][gGt]',
                   "pattern_regex": ".*(-re|-up)\.(txt|jpg|JPG)$",
                   "folder_name": "03_MICOLI"},
        "micoli_split": {"pattern_bash": '*-[ru][ep]-split-*.[jJt][pPx][gGt]',
                   "pattern_regex": ".*(-re|-up)-split-.\.(txt|jpg|JPG)$",
                   "folder_name": "06_MICOLI_split"},
        "micoli_split_overlapping": {
                    "pattern_bash": '*-splitov-*.[jJt][pPx][gGt]',
                    "pattern_regex": r".*(-re|-up)-splitov-.*\.(txt|jpg|JPG)$",
                    "folder_name": "07_MICOLI_split_overlapping"},
        "taco": {"pattern_bash": '*taco-*.[jJt][pPx][gGt]',
                   "pattern_regex": ".*taco-.*\.(txt|jpg|JPG)$",
                   "folder_name": "01_TACO"},
        "taco_extended": {"pattern_bash": '*taco_extended-*.[jJt][pPx][gGt]',
                   "pattern_regex": ".*taco_extended-.*\.(txt|jpg|JPG)$",
                   "folder_name": "02_TACO_extended"},
        "plastopol": {"pattern_bash": '*plastopol-*.[jJt][pPx][gGt]',
                   "pattern_regex": ".*plastopol-.*\.(txt|jpg|JPG)$",
                   "folder_name": "04_Plastopol"},
    }


    # Prepare main folder for training,
    for combination in combinations:

        # Add all relevant and remove all irrelevant photos
        datasets = ["micoli", "micoli_split", "micoli_split_overlapping", "taco", "taco_extended", "plastopol"]
        for dataset in datasets:
            if dataset in combination:
                copy_to_data_folder(data_dict[dataset])
            else:
                remove_from_data_folder(data_dict[dataset])

        model = YOLO(str(path_to_folder / "03_assets" / "yolo26m.pt"))  # build a new model
        yaml_file = "micoli.yaml"

        today = str( date.today())

        runtime_yaml = make_runtime_yaml(yaml_file)

        # Train the model
        model.train(data=str(runtime_yaml),
                    imgsz=640,  # higher resolution might be better here bc of small pieces of trash
                    epochs=200,
                    batch=8,
                    device=0,
                    workers=4,
                    name="v26_m",
                    save_period=50)  # train the model


        # Evaluate best model
        best_model_path = Path(model.trainer.best)
        best_model = YOLO(str(best_model_path))

        model_results = {
            "model": "_".join(combination)
        }

        for dataset in ["micoli", "taco", "taco_extended", "plastopol"]:

            runtime_yaml = make_runtime_yaml(f"{dataset}.yaml")

            metrics = best_model.val(
                data=str(runtime_yaml),
                imgsz=640,
                device=0,
            )

            precision = float(metrics.box.mp)
            recall = float(metrics.box.mr)
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0 else 0
            )

            metrics_conf50 = best_model.val(
                data=str(runtime_yaml),
                imgsz=640,
                conf=0.5,
                device=0,
            )

            tp, fp = metrics_conf50.confusion_matrix.tp_fp()

            model_results[f"{dataset}_map50"] = float(metrics.box.map50)
            model_results[f"{dataset}_map50_95"] = float(metrics.box.map)
            model_results[f"{dataset}_map75"] = float(metrics.box.map75)
            model_results[f"{dataset}_precision"] = precision
            model_results[f"{dataset}_recall"] = recall
            model_results[f"{dataset}_f1"] = f1
            model_results[f"{dataset}_true_positives_conf50"] = int(tp.sum())
            model_results[f"{dataset}_false_positives_conf50"] = int(fp.sum())

        results = results[results["model"] != model_results["model"]] if "model" in results.columns else results
        results = pd.concat(
            [results, pd.DataFrame([model_results])],
            ignore_index=True
        )
        results.to_excel(results_file, index=False)


         # Save/copy the YOLO26 run folder (including weights and other results)
        name = '_'.join(combination)
        name = today+"-"+name
        results_path = os.path.join("03_assets/03_saved_runs/runs_yolo26", name)
        shutil.copytree("runs", results_path)

        # Delete original run folder for next training and delete cache (try this ten times due to some Windows bug)
        for i in range(10):
            if os.path.isdir("runs"):
                try:
                    shutil.rmtree("runs", onerror=remove_readonly)
                except PermissionError:
                    pass
        caches = glob.glob(os.path.join(data_dir, '**', '*.cache'),recursive=True)
        for cache in caches:
            os.remove(cache)

        torch.cuda.empty_cache()



if __name__ == "__main__":
    main()