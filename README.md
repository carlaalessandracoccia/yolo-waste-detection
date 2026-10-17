# YOLO Waste Detection

Code and experiment records for training a YOLO26 waste detector on MICOLI, TACO, extended TACO and Plastopol images. The main output is the trained `best.pt` model; predictions for the wider research project are handled separately.

## Setup

1. Download or clone this repository.
2. Download the **`03_assets` folder from https://www.dropbox.com/scl/fo/u06ai4b4y9uzvy7jz8ajv/ACGrYqXW0KVLKoVwVNd9Jzk?rlkey=gu58rpc2hwfzriutbdnokuw3p&st=6grl7cw9&dl=0 **.
3. Unzip it and place `03_assets` next to `01_code` and `02_results`:

```text
project/
├── README.md
├── 01_code/                    # Preparation, training and evaluation scripts
├── 02_results/                 # Result tables and selected prediction photos
└── 03_assets/
    ├── 01_raw_data/
    ├── 02_data/
    ├── 03_saved_runs/
    ├── 04_predictions/
    └── yolo26m.pt
```

The scripts find the project folder automatically.

**Dependencies:** Python with Ultralytics, PyTorch, torchvision, pandas, NumPy, scikit-learn, Pillow, PyYAML and openpyxl. The saved models record Ultralytics version **8.4.116**. Baseline training requires a CUDA GPU with the current settings.

## Scripts

| Script in `01_code` | Purpose |
| --- | --- |
| `prepare_data.py` | Prepare images and labels, create MICOLI crops and assign train/validation filenames. |
| `object_detection.py` | Train the five baseline dataset combinations. |
| `object_detection_finetune.py` | Record and run the 19 fine-tuning experiments. |
| `generate_evaluation_stats.py` | Evaluate the final model and save validation predictions. |

The scripts also serve as a research log. Original preparation steps are retained in a disabled block. To select a fine-tuning experiment, copy its configuration from `previous_experiments_DO_NOT_RUN` into `experiments`. Existing checkpoints are evaluated without retraining.

Use a working copy when rerunning scripts: preparation changes data files, training scripts move datasets, and evaluation can overwrite result tables. The empty fine-tuning experiment list prevents training but still runs dataset cleanup.

## Final model

The selected model is **experiment 17: `finetune_splitov4_960`**, trained at image size 960 using overlapping MICOLI crops alongside the three external datasets.

```text
03_assets/03_saved_runs/runs_yolo26_finetune/finetune_splitov4_960/weights/best.pt
```

Evaluation tables are in `02_results`. Individual training settings and metrics are stored with each run in `03_assets/03_saved_runs`.
