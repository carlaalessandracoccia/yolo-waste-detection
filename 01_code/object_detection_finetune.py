# -*- coding: utf-8 -*-

"""
Fine-tune the best YOLO26 model for MICOLI.

Starting model:
    MICOLI + TACO + TACO_extended + Plastopol

Completed experiments 1-19 are archived below with their complete
historical training configurations so they can be reproduced later.

Most experiments start independently from the original best.pt.
Staged experiments explicitly state when they start from a previous best.pt.

For every experiment:
    - training run and weights are saved
    - best.pt is evaluated on MICOLI and TACO
    - AP50, mAP50-95, AP75, precision, recall and F1 are saved

The script saves:
    03_assets/03_saved_runs/runs_yolo26_finetune
and:
    02_results/results_used_in_paper_finetune.xlsx

"""

import os
import glob
import re
import shutil
from pathlib import Path
from datetime import date

import pandas as pd
import torch
import yaml
from ultralytics import YOLO


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#                                  SCRIPT SETUP
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def main():

    # -------------------------------------------------------------------------
    # Choose current user
    # -------------------------------------------------------------------------

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
    temp_dir = path_to_folder / "03_assets" / "02_data" / "99_temp"
    train_dir = data_dir / "images" / "train"

    os.chdir(path_to_folder)

    # -------------------------------------------------------------------------
    # Choose device
    # -------------------------------------------------------------------------

    if torch.cuda.is_available():
        device = 0
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print("Using device:", device)

    if torch.cuda.is_available():
        os.system("nvidia-smi")


    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #                       IMPORTANT PATHS
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # This is the BEST MODEL from:
    # MICOLI + TACO + TACO_extended + Plastopol
    #
    # Historical baseline used to initialize the fine-tuning experiments.

    base_model_path = (
        path_to_folder
        / "03_assets"
        / "03_saved_runs"
        / "runs_yolo26"
        / "2026-09-09-micoli_taco_taco_extended_plastopol"
        / "detect"
        / "v26_m"
        / "weights"
        / "best.pt"
    )



    # Folder where ALL new fine-tuning runs will be saved
    finetune_runs_dir = (
        path_to_folder
        / "03_assets"
        / "03_saved_runs"
        / "runs_yolo26_finetune"
    )

    finetune_runs_dir.mkdir(parents=True, exist_ok=True)

    # Results spreadsheet
    results_dir = path_to_folder / "02_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / "results_used_in_paper_finetune.xlsx"


    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #                           HELPER FUNCTIONS
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def make_runtime_yaml(yaml_file):
        """
        Make a runtime YAML with an absolute dataset path.
        """

        source_yaml = data_dir / "yaml_files" / yaml_file
        runtime_yaml = data_dir / "yaml_files" / "_runtime" / yaml_file

        runtime_yaml.parent.mkdir(parents=True, exist_ok=True)

        with open(source_yaml, "r") as f:
            config = yaml.safe_load(f)

        config["path"] = str(data_dir.resolve())

        with open(runtime_yaml, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)

        return runtime_yaml


    # -------------------------------------------------------------------------
    # Dataset moving functions
    #
    # -------------------------------------------------------------------------

    data_dict = {

        "micoli": {
            "pattern_bash": '*-[ru][ep].[jJt][pPx][gGt]',
            "pattern_regex": r".*(-re|-up)\.(txt|jpg|JPG)$",
            "folder_name": "03_MICOLI"
        },

        "micoli_split": {
            "pattern_bash": '*-split-*.[jJt][pPx][gGt]',
            "pattern_regex": r".*(-re|-up)-split-.*\.(txt|jpg|JPG)$",
            "folder_name": "06_MICOLI_split"
        },

        "micoli_split_overlapping": {
            "pattern_bash": '*-splitov-*.[jJt][pPx][gGt]',
            "pattern_regex": r".*(-re|-up)-splitov-.*\.(txt|jpg|JPG)$",
            "folder_name": "07_MICOLI_split_overlapping"
        },

        "taco": {
            "pattern_bash": '*taco-*.[jJt][pPx][gGt]',
            "pattern_regex": r".*taco-.*\.(txt|jpg|JPG)$",
            "folder_name": "01_TACO"
        },

        "taco_extended": {
            "pattern_bash": '*taco_extended-*.[jJt][pPx][gGt]',
            "pattern_regex": r".*taco_extended-.*\.(txt|jpg|JPG)$",
            "folder_name": "02_TACO_extended"
        },

        "plastopol": {
            "pattern_bash": '*plastopol-*.[jJt][pPx][gGt]',
            "pattern_regex": r".*plastopol-.*\.(txt|jpg|JPG)$",
            "folder_name": "04_Plastopol"
        },
    }


    def copy_to_data_folder(dictionary_entry):
        """
        Move one dataset from 99_temp into the main YOLO training folder.
        """

        folder_name = dictionary_entry["folder_name"]
        pattern_regex = dictionary_entry["pattern_regex"]

        source_dir = temp_dir / folder_name

        # Dataset may already be in the main data directory
        source_train_images = source_dir / "images" / "train"

        if not source_train_images.exists():
            return

        move = False

        for item in source_train_images.iterdir():
            if re.match(pattern_regex, item.name):
                move = True
                break

        if move:

            for item in source_dir.rglob("*"):

                if (
                    item.is_file()
                    and item.suffix.lower() in [".jpg", ".jpeg", ".txt"]
                ):

                    relative_path = item.relative_to(source_dir)

                    dst = data_dir / relative_path

                    dst.parent.mkdir(parents=True, exist_ok=True)

                    shutil.move(str(item), str(dst))


    def remove_from_data_folder(dictionary_entry):
        """
        Move one dataset from the main YOLO training folder back to 99_temp.
        """

        folder_name = dictionary_entry["folder_name"]
        pattern_regex = dictionary_entry["pattern_regex"]
        pattern_bash = dictionary_entry["pattern_bash"]

        if not train_dir.exists():
            return

        move = False

        for item in train_dir.iterdir():
            if re.match(pattern_regex, item.name):
                move = True
                break

        if move:

            for folder_type in ["images", "labels"]:

                source_dir = data_dir / folder_type / "train"
                target_dir = temp_dir / folder_name / folder_type / "train"

                if not source_dir.exists():
                    continue

                for item in source_dir.glob(pattern_bash):

                    target_dir.mkdir(parents=True, exist_ok=True)

                    dst = target_dir / item.name

                    shutil.move(str(item), str(dst))


    def set_training_datasets(datasets_to_use):
        """
        Put only the requested datasets into images/train + labels/train.

        Dataset combinations are specified in the experiment dictionaries.
        """

        all_datasets = [
            "micoli",
            "micoli_split",
            "micoli_split_overlapping",
            "taco",
            "taco_extended",
            "plastopol"
        ]

        for dataset in all_datasets:

            if dataset in datasets_to_use:
                copy_to_data_folder(data_dict[dataset])

            else:
                remove_from_data_folder(data_dict[dataset])


    def clear_cache():
        """
        Remove YOLO dataset cache files so that changed training contents
        are detected correctly.
        """

        caches = glob.glob(
            str(data_dir / "**" / "*.cache"),
            recursive=True
        )

        for cache in caches:
            try:
                os.remove(cache)
            except FileNotFoundError:
                pass


    def clear_micoli_oversample():
        """
        Remove temporary MICOLI oversampling copies created by this script.

        Oversampled files are deliberately named with "__micoli_os" near the
        end of the stem so they do NOT match the normal MICOLI move regex.
        """
        for folder_type in ["images", "labels"]:
            folder = data_dir / folder_type / "train"
            if not folder.exists():
                continue

            for item in folder.glob("*__micoli_os*"):
                try:
                    item.unlink()
                except FileNotFoundError:
                    pass


    def create_micoli_oversample(multiplier):
        """
        Make MICOLI appear approximately `multiplier` times in the shared
        training folder, while leaving TACO/TACO_extended/Plastopol unchanged.

        multiplier=1 -> no duplication
        multiplier=2 -> one temporary copy of every MICOLI training image/label
        multiplier=3 -> two temporary copies of every MICOLI training image/label

        Copies are removed after each experiment. Any left by an interrupted
        run are cleared before the next experiment.
        """
        clear_micoli_oversample()

        if multiplier <= 1:
            print("MICOLI oversampling: x1 (no temporary copies)")
            return

        image_dir = data_dir / "images" / "train"
        label_dir = data_dir / "labels" / "train"

        if not image_dir.exists():
            raise FileNotFoundError(f"Training image folder not found: {image_dir}")

        micoli_regex = re.compile(data_dict["micoli"]["pattern_regex"])

        originals = [
            p for p in image_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in [".jpg", ".jpeg"]
            and micoli_regex.match(p.name)
        ]

        copies_created = 0
        missing_labels = []

        for image_path in originals:
            label_path = label_dir / f"{image_path.stem}.txt"

            if not label_path.exists():
                missing_labels.append(str(label_path))
                continue

            # Original counts as x1. Add multiplier-1 paired copies.
            for copy_number in range(2, multiplier + 1):
                new_stem = f"{image_path.stem}__micoli_os{copy_number}"

                image_copy = image_dir / f"{new_stem}{image_path.suffix}"
                label_copy = label_dir / f"{new_stem}.txt"

                shutil.copy2(image_path, image_copy)
                shutil.copy2(label_path, label_copy)

                copies_created += 1

        if missing_labels:
            print(
                f"WARNING: {len(missing_labels)} MICOLI images had no matching "
                "label file and were not oversampled."
            )



    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #                       PREPARE MICOLI TRAINING SET
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # Runtime YAMLs for evaluation.
    # Training data will be switched per experiment below.
    evaluation_yamls = {
        "micoli": make_runtime_yaml("micoli.yaml"),
        "taco": make_runtime_yaml("taco.yaml"),
    }



    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #                           EXPERIMENTS
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    """
    COMPLETE EXPERIMENT ARCHIVE

    To select an experiment, copy its dictionary from
    `previous_experiments_DO_NOT_RUN` into `experiments`.
    Existing best.pt files are evaluated without retraining.
    """

    # -------------------------------------------------------------------------
    # Checkpoint used only by experiment 7.
    #
    # Experiment 7 was explicitly staged from the already-trained all4_960
    # best.pt rather than from the original 640 checkpoint.
    # -------------------------------------------------------------------------

    all4_960_best_path = (
        finetune_runs_dir
        / "finetune_all4_960"
        / "weights"
        / "best.pt"
    )

    split4_960_best_path = (
        finetune_runs_dir
        / "finetune_split4_960"
        / "weights"
        / "best.pt"
    )


    # =========================================================================
    # FULL ARCHIVE OF EXPERIMENTS 1-19 -- DOES NOT RUN
    # =========================================================================

    previous_experiments_DO_NOT_RUN = [

        # =====================================================================
        # 1) MICOLI ONLY @ 640
        # Historical source: initial MICOLI fine-tuning script
        # =====================================================================
        {
            "type": "standard",
            "name": "finetune_micoli_640",
            "datasets": ["micoli"],
            "start_model": base_model_path,
            "start_model_description": "original all4 640 best.pt",
            "imgsz": 640,
            "batch": 8,
            "micoli_oversample": 1,
            "train_args": {
                "optimizer": "AdamW",
                "lr0": 0.0005,
                "time": 6.0,
                "patience": 15,
                "save_period": 10,
                "close_mosaic": 10,
            },
        },

        # =====================================================================
        # 2) MICOLI ONLY @ 960
        # =====================================================================
        {
            "type": "standard",
            "name": "finetune_micoli_960",
            "datasets": ["micoli"],
            "start_model": base_model_path,
            "start_model_description": "original all4 640 best.pt",
            "imgsz": 960,
            "batch": 4,
            "micoli_oversample": 1,
            "train_args": {
                "optimizer": "AdamW",
                "lr0": 0.0005,
                "time": 6.0,
                "patience": 15,
                "save_period": 10,
                "close_mosaic": 10,
            },
        },

        # =====================================================================
        # 3) MICOLI ONLY @ 960 + MODEST AUGMENTATION
        # =====================================================================
        {
            "type": "standard",
            "name": "finetune_micoli_960_aug",
            "datasets": ["micoli"],
            "start_model": base_model_path,
            "start_model_description": "original all4 640 best.pt",
            "imgsz": 960,
            "batch": 4,
            "micoli_oversample": 1,
            "train_args": {
                "optimizer": "AdamW",
                "lr0": 0.0005,
                "time": 6.0,
                "patience": 15,
                "save_period": 10,
                "degrees": 10.0,
                "translate": 0.10,
                "scale": 0.50,
                "fliplr": 0.50,
                "mosaic": 1.0,
                "close_mosaic": 15,
            },
        },

        # =====================================================================
        # 4) MICOLI + TACO @ 960
        # =====================================================================
        {
            "type": "standard",
            "name": "finetune_micoli_taco_960",
            "datasets": ["micoli", "taco"],
            "start_model": base_model_path,
            "start_model_description": "original all4 640 best.pt",
            "imgsz": 960,
            "batch": 4,
            "micoli_oversample": 1,
            "train_args": {
                "optimizer": "AdamW",
                "lr0": 0.0002,
                "time": 6.0,
                "patience": 20,
                "save_period": 10,
                "close_mosaic": 10,
            },
        },

        # =====================================================================
        # 5) MICOLI + TACO @ 960 + MODEST AUGMENTATION
        # =====================================================================
        {
            "type": "standard",
            "name": "finetune_micoli_taco_960_aug",
            "datasets": ["micoli", "taco"],
            "start_model": base_model_path,
            "start_model_description": "original all4 640 best.pt",
            "imgsz": 960,
            "batch": 4,
            "micoli_oversample": 1,
            "train_args": {
                "optimizer": "AdamW",
                "lr0": 0.0002,
                "time": 6.0,
                "patience": 20,
                "save_period": 10,
                "degrees": 7.0,
                "translate": 0.08,
                "scale": 0.40,
                "fliplr": 0.50,
                "mosaic": 1.0,
                "close_mosaic": 15,
            },
        },

        # =====================================================================
        # 6) ALL FOUR @ 960
        # =====================================================================
        {
            "type": "standard",
            "name": "finetune_all4_960",
            "datasets": ["micoli", "taco", "taco_extended", "plastopol"],
            "start_model": base_model_path,
            "start_model_description": "original all4 640 best.pt",
            "imgsz": 960,
            "batch": 4,
            "micoli_oversample": 1,
            "train_args": {
                "optimizer": "AdamW",
                "lr0": 0.00015,
                "time": 6.0,
                "patience": 20,
                "save_period": 10,
                "close_mosaic": 10,
            },
        },

        # =====================================================================
        # 7) ALL FOUR @ 1280, STAGED FROM EXPERIMENT 6 BEST.PT
        # =====================================================================
        {
            "type": "standard",
            "name": "finetune_all4_1280_staged",
            "datasets": ["micoli", "taco", "taco_extended", "plastopol"],
            "start_model": all4_960_best_path,
            "start_model_description": "finetune_all4_960 best.pt",
            "imgsz": 1280,
            "batch": 2,
            "micoli_oversample": 1,
            "train_args": {
                "optimizer": "AdamW",
                "lr0": 0.00008,
                "time": 6.0,
                "patience": 20,
                "save_period": 10,
                "close_mosaic": 10,
            },
        },

        # =====================================================================
        # 8) ALL FOUR @ 960, PROGRESSIVE FREEZE -> UNFREEZE
        #
        # Stage 1:
        #     original all4 640 best.pt
        #     freeze first 10 layers
        #     lr0 = 0.00015
        #     max 2 h
        #
        # Stage 2:
        #     starts from Stage-1 best.pt
        #     all layers unfrozen
        #     lr0 = 0.00005
        #     max 4 h
        # =====================================================================
        {
            "type": "progressive",
            "name": "finetune_all4_960_progressive",
            "datasets": ["micoli", "taco", "taco_extended", "plastopol"],
            "start_model": base_model_path,
            "start_model_description": "original all4 640 best.pt",
            "imgsz": 960,
            "batch": 4,
            "micoli_oversample": 1,

            "stage1": {
                "name": "finetune_all4_960_progressive_stage1_frozen",
                "optimizer": "AdamW",
                "lr0": 0.00015,
                "time": 2.0,
                "patience": 10,
                "save_period": 10,
                "freeze": 10,
                "close_mosaic": 10,
            },

            "stage2": {
                "name": "finetune_all4_960_progressive",
                "optimizer": "AdamW",
                "lr0": 0.00005,
                "time": 4.0,
                "patience": 20,
                "save_period": 10,
                "freeze": 0,
                "close_mosaic": 10,
            },
        },

        # =====================================================================
        # 9) ALL FOUR @ 960 + MODEST AUGMENTATION
        # =====================================================================
        {
            "type": "standard",
            "name": "finetune_all4_960_aug",
            "datasets": ["micoli", "taco", "taco_extended", "plastopol"],
            "start_model": base_model_path,
            "start_model_description": "original all4 640 best.pt",
            "imgsz": 960,
            "batch": 4,
            "micoli_oversample": 1,
            "train_args": {
                "optimizer": "AdamW",
                "lr0": 0.00015,
                "time": 6.0,
                "patience": 20,
                "save_period": 10,
                "degrees": 7.0,
                "translate": 0.08,
                "scale": 0.40,
                "fliplr": 0.50,
                "mosaic": 1.0,
                "close_mosaic": 15,
            },
        },

        # =====================================================================
        # 10) ALL FOUR @ 960, MICOLI x2
        # =====================================================================
        {
            "type": "standard",
            "name": "finetune_all4_960_micoli_x2",
            "datasets": ["micoli", "taco", "taco_extended", "plastopol"],
            "start_model": base_model_path,
            "start_model_description": "original all4 640 best.pt",
            "imgsz": 960,
            "batch": 4,
            "micoli_oversample": 2,
            "train_args": {
                "optimizer": "AdamW",
                "lr0": 0.00015,
                "time": 6.0,
                "patience": 20,
                "save_period": 10,
                "close_mosaic": 10,
            },
        },

        # =====================================================================
        # 11) ALL FOUR @ 960, MICOLI x3
        # =====================================================================
        {
            "type": "standard",
            "name": "finetune_all4_960_micoli_x3",
            "datasets": ["micoli", "taco", "taco_extended", "plastopol"],
            "start_model": base_model_path,
            "start_model_description": "original all4 640 best.pt",
            "imgsz": 960,
            "batch": 4,
            "micoli_oversample": 3,
            "train_args": {
                "optimizer": "AdamW",
                "lr0": 0.00015,
                "time": 6.0,
                "patience": 20,
                "save_period": 10,
                "close_mosaic": 10,
            },
        },

        # =====================================================================
        # 12) ALL FOUR @ 960, LOWER LR
        # =====================================================================
        {
            "type": "standard",
            "name": "finetune_all4_960_lr000008",
            "datasets": ["micoli", "taco", "taco_extended", "plastopol"],
            "start_model": base_model_path,
            "start_model_description": "original all4 640 best.pt",
            "imgsz": 960,
            "batch": 4,
            "micoli_oversample": 1,
            "train_args": {
                "optimizer": "AdamW",
                "lr0": 0.00008,
                "time": 6.0,
                "patience": 20,
                "save_period": 10,
                "close_mosaic": 10,
            },
        },

        # =====================================================================
        # 13) ALL FOUR @ 960, HIGHER LR
        # =====================================================================
        {
            "type": "standard",
            "name": "finetune_all4_960_lr00003",
            "datasets": ["micoli", "taco", "taco_extended", "plastopol"],
            "start_model": base_model_path,
            "start_model_description": "original all4 640 best.pt",
            "imgsz": 960,
            "batch": 4,
            "micoli_oversample": 1,
            "train_args": {
                "optimizer": "AdamW",
                "lr0": 0.0003,
                "time": 6.0,
                "patience": 20,
                "save_period": 10,
                "close_mosaic": 10,
            },
        },

        # =====================================================================
        # 14) MICOLI_SPLIT + TACO + TACO_EXTENDED + PLASTOPOL @ 960
        # =====================================================================
        {
            "type": "standard",
            "name": "finetune_split4_960",
            "datasets": [
                "micoli_split",
                "taco",
                "taco_extended",
                "plastopol",
            ],
            "start_model": base_model_path,
            "start_model_description": "original all4 640 best.pt",
            "imgsz": 960,
            "batch": 4,
            "micoli_oversample": 1,
            "train_args": {
                "optimizer": "AdamW",
                "lr0": 0.00015,
                "time": 6.0,
                "patience": 20,
                "save_period": 10,
                "close_mosaic": 10,
            },
        },

        # =====================================================================
        # 15) FULL MICOLI + MICOLI_SPLIT + TACO + TACO_EXTENDED + PLASTOPOL
        # =====================================================================
        {
            "type": "standard",
            "name": "finetune_all4_plus_split_960",
            "datasets": [
                "micoli",
                "micoli_split",
                "taco",
                "taco_extended",
                "plastopol",
            ],
            "start_model": base_model_path,
            "start_model_description": "original all4 640 best.pt",
            "imgsz": 960,
            "batch": 4,
            "micoli_oversample": 1,
            "train_args": {
                "optimizer": "AdamW",
                "lr0": 0.00015,
                "time": 6.0,
                "patience": 20,
                "save_period": 10,
                "close_mosaic": 10,
            },
        },

        # =====================================================================
        # 16) NON-OVERLAPPING SPLIT4 @ 960, LOWER LR
        # =====================================================================
        {
            "type": "standard",
            "name": "finetune_split4_960_lr000008",
            "datasets": [
                "micoli_split",
                "taco",
                "taco_extended",
                "plastopol",
            ],
            "start_model": base_model_path,
            "start_model_description": "original all4 640 best.pt",
            "imgsz": 960,
            "batch": 4,
            "micoli_oversample": 1,
            "train_args": {
                "optimizer": "AdamW",
                "lr0": 0.00008,
                "time": 6.0,
                "patience": 20,
                "save_period": 10,
                "close_mosaic": 10,
            },
        },

        # =====================================================================
        # 17) OVERLAPPING MICOLI_SPLIT + TACO + TACO_EXTENDED + PLASTOPOL
        #
        # Uses the splitov images produced by prepare_data:
        #     *-splitov-0.jpg ... *-splitov-3.jpg
        # =====================================================================
        {
            "type": "standard",
            "name": "finetune_splitov4_960",
            "datasets": [
                "micoli_split_overlapping",
                "taco",
                "taco_extended",
                "plastopol",
            ],
            "start_model": base_model_path,
            "start_model_description": "original all4 640 best.pt",
            "imgsz": 960,
            "batch": 4,
            "micoli_oversample": 1,
            "train_args": {
                "optimizer": "AdamW",
                "lr0": 0.00015,
                "time": 6.0,
                "patience": 20,
                "save_period": 10,
                "close_mosaic": 10,
            },
        },

        # =====================================================================
        # 18) OVERLAPPING MICOLI SPLIT + P2 SMALL-OBJECT HEAD @ 960
        # =====================================================================
        {
            "type": "standard",
            "name": "finetune_splitov4_960_p2",
            "datasets": [
                "micoli_split_overlapping",
                "taco",
                "taco_extended",
                "plastopol",
            ],
            "start_model": base_model_path,
            "start_model_description": "original all4 640 best.pt -> YOLO26m P2",
            "model_architecture": "yolo26m-p2.yaml",
            "imgsz": 960,
            "batch": 4,
            "micoli_oversample": 1,
            "train_args": {
                "optimizer": "AdamW",
                "lr0": 0.00015,
                "time": 6.0,
                "patience": 20,
                "save_period": 10,
                "close_mosaic": 10,
            },
        },

        # =====================================================================
        # 19) SPLIT4_960 -> SHORT OVERLAP FINE-TUNE @ 960, LR 0.00005
        # =====================================================================
        {
            "type": "standard",
            "name": "finetune_split4_to_splitov4_960_lr000005",
            "datasets": [
                "micoli_split_overlapping",
                "taco",
                "taco_extended",
                "plastopol",
            ],
            "start_model": split4_960_best_path,
            "start_model_description": "finetune_split4_960 best.pt",
            "imgsz": 960,
            "batch": 4,
            "micoli_oversample": 1,
            "train_args": {
                "optimizer": "AdamW",
                "lr0": 0.00005,
                "time": 3.0,
                "patience": 15,
                "save_period": 10,
                "close_mosaic": 10,
            },
        },
    ]


    # =========================================================================
    # ACTIVE EXPERIMENTS
    # =========================================================================

    experiments = [

    ]

    if results_file.exists():
        old_results = pd.read_excel(results_file)
        evaluation_rows = old_results.to_dict("records")
        print("\nExisting fine-tuning results loaded:")
        print(results_file)

    else:
        evaluation_rows = []

    today = str(date.today())

    def save_or_replace_row(row):
        nonlocal evaluation_rows
        evaluation_rows = [
            r for r in evaluation_rows
            if r.get("experiment") != row.get("experiment")
        ]
        evaluation_rows.append(row)
        pd.DataFrame(evaluation_rows).to_excel(results_file, index=False)

    def evaluate_and_save(
        experiment_name,
        best_model_path,
        imgsz,
        batch,
        datasets_to_use,
        starting_model,
        start_model_description,
        optimizer,
        lr0,
        max_training_hours,
        micoli_oversample=1,
        extra_info=None,
    ):
        best_model = YOLO(str(best_model_path))

        row = {
            "date": today,
            "experiment": experiment_name,
            "status": "TRAINED_NOT_YET_EVALUATED",
            "error": "",
            "starting_model": str(starting_model),
            "starting_model_description": start_model_description,
            "best_model": str(best_model_path),
            "training_dataset": " + ".join(datasets_to_use),
            "imgsz": imgsz,
            "batch": batch,
            "lr0": lr0,
            "optimizer": optimizer,
            "max_training_hours": max_training_hours,
            "micoli_oversample_multiplier": micoli_oversample,
        }

        if extra_info:
            row.update(extra_info)

        save_or_replace_row(row)

        for dataset_name, dataset_yaml in evaluation_yamls.items():
            print("\nEvaluating:")
            print("Model:", experiment_name)
            print("Dataset:", dataset_name)


            metrics = best_model.val(
                data=str(dataset_yaml),
                imgsz=imgsz,
                device=device,
                workers=0,
            )

            precision = float(metrics.box.mp)
            recall = float(metrics.box.mr)
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall > 0
                else 0.0
            )

            row[f"{dataset_name}_ap50"] = float(metrics.box.map50)
            row[f"{dataset_name}_map50_95"] = float(metrics.box.map)
            row[f"{dataset_name}_ap75"] = float(metrics.box.map75)
            row[f"{dataset_name}_precision"] = precision
            row[f"{dataset_name}_recall"] = recall
            row[f"{dataset_name}_f1"] = f1


            save_or_replace_row(row)

        row["status"] = "FINISHED"
        save_or_replace_row(row)

        del best_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    for experiment_number, experiment in enumerate(experiments, start=1):

        experiment_type = experiment["type"]
        experiment_name = experiment["name"]
        datasets_to_use = experiment["datasets"]
        imgsz = experiment["imgsz"]
        batch = experiment["batch"]
        micoli_oversample = experiment.get("micoli_oversample", 1)
        start_model = Path(experiment["start_model"])
        start_model_description = experiment["start_model_description"]
        model_architecture = experiment.get("model_architecture")


        # Remove old temporary oversampling copies (from an interrupted run)
        clear_micoli_oversample()
        set_training_datasets(datasets_to_use)
        create_micoli_oversample(micoli_oversample)

        clear_cache()
        training_yaml = make_runtime_yaml("micoli.yaml")

        if experiment_type == "standard":

            train_args = experiment["train_args"]

            best_model_path = (
                finetune_runs_dir
                / experiment_name
                / "weights"
                / "best.pt"
            )

            if best_model_path.exists():
                print("\nExisting best.pt found -- skipping training:")
                print(best_model_path)
            else:
                if model_architecture:
                    model = YOLO(model_architecture)
                    model.load(str(start_model))
                else:
                    model = YOLO(str(start_model))


                model.train(
                    data=str(training_yaml),
                    imgsz=imgsz,
                    batch=batch,
                    device=device,
                    workers=4,
                    project=str(finetune_runs_dir),
                    name=experiment_name,
                    exist_ok=True,
                    save=True,
                    **train_args,
                )


            evaluate_and_save(
                experiment_name=experiment_name,
                best_model_path=best_model_path,
                imgsz=imgsz,
                batch=batch,
                datasets_to_use=datasets_to_use,
                starting_model=start_model,
                start_model_description=start_model_description,
                optimizer=train_args["optimizer"],
                lr0=train_args["lr0"],
                max_training_hours=train_args["time"],
                micoli_oversample=micoli_oversample,
                extra_info={
                    "model_architecture":
                        model_architecture if model_architecture else "standard",
                },
            )


        elif experiment_type == "progressive":

            stage1 = experiment["stage1"]
            stage2 = experiment["stage2"]

            stage1_best_path = (
                finetune_runs_dir / stage1["name"] / "weights" / "best.pt"
            )
            final_best_path = (
                finetune_runs_dir / stage2["name"] / "weights" / "best.pt"
            )

            if not final_best_path.exists():

                if not stage1_best_path.exists():
                    print("\nProgressive Stage 1:", stage1["name"])

                    model = YOLO(str(start_model))
                    model.train(
                        data=str(training_yaml),
                        imgsz=imgsz,
                        batch=batch,
                        device=device,
                        workers=4,
                        project=str(finetune_runs_dir),
                        name=stage1["name"],
                        exist_ok=True,
                        save=True,
                        optimizer=stage1["optimizer"],
                        lr0=stage1["lr0"],
                        time=stage1["time"],
                        patience=stage1["patience"],
                        save_period=stage1["save_period"],
                        freeze=stage1["freeze"],
                        close_mosaic=stage1["close_mosaic"],
                    )

                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                print("\nProgressive Stage 2:", stage2["name"])

                model = YOLO(str(stage1_best_path))
                model.train(
                    data=str(training_yaml),
                    imgsz=imgsz,
                    batch=batch,
                    device=device,
                    workers=4,
                    project=str(finetune_runs_dir),
                    name=stage2["name"],
                    exist_ok=True,
                    save=True,
                    optimizer=stage2["optimizer"],
                    lr0=stage2["lr0"],
                    time=stage2["time"],
                    patience=stage2["patience"],
                    save_period=stage2["save_period"],
                    freeze=stage2["freeze"],
                    close_mosaic=stage2["close_mosaic"],
                )

                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            evaluate_and_save(
                experiment_name=experiment_name,
                best_model_path=final_best_path,
                imgsz=imgsz,
                batch=batch,
                datasets_to_use=datasets_to_use,
                starting_model=start_model,
                start_model_description=start_model_description,
                optimizer=stage2["optimizer"],
                lr0=stage2["lr0"],
                max_training_hours=stage1["time"] + stage2["time"],
                extra_info={
                    "progressive_stage1_best": str(stage1_best_path),
                    "progressive_stage1_freeze": stage1["freeze"],
                    "progressive_stage1_lr0": stage1["lr0"],
                    "progressive_stage1_max_hours": stage1["time"],
                    "progressive_stage2_freeze": stage2["freeze"],
                    "progressive_stage2_lr0": stage2["lr0"],
                    "progressive_stage2_max_hours": stage2["time"],
                },
            )

        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")

        # Avoid temporary oversampling copies leaking into the next run
        clear_micoli_oversample()
        clear_cache()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    clear_micoli_oversample()

    set_training_datasets([
        "micoli",
        "taco",
        "taco_extended",
        "plastopol"
    ])

    clear_cache()

    print("Done.")


if __name__ == "__main__":
    main()
