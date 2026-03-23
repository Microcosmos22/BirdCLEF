# Cell 2 — Imports and run config
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import gc
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import tensorflow as tf

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
tf.experimental.numpy.experimental_enable_numpy_behavior()

BASE = Path("..\\data")
MODEL_DIR = Path("/kaggle/input/models/google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1")

SR = 32000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
FILE_SAMPLES = 60 * SR
N_WINDOWS = 12

# Cell 1 — Mode switch
MODE = "train"

assert MODE in {"train", "submit"}

print("MODE =", MODE)

CFG = {
    "mode": MODE,
    "verbose": MODE == "train",

    # expensive research blocks
    "run_oof_baseline": MODE == "train",
    "run_probe_check": MODE == "train",
    "run_probe_grid": MODE == "train",

    # inference
    "batch_files": 16,
    "dryrun_n_files": 50 if MODE == "train" else 20,

    # cache behavior
    "require_full_cache_in_submit": True,
    "full_cache_input_dir": Path("/kaggle/input/datasets/jaejohn/perch-meta"),  # attach this dataset
    "full_cache_work_dir": Path("/kaggle/working/perch_cache"),

    # frozen baseline fusion params
    "best_fusion": {
        "lambda_event": 0.4,
        "lambda_texture": 1.0,
        "lambda_proxy_texture": 0.8,
        "smooth_texture": 0.35,
    },

    # frozen probe params from OOF tuning
    "frozen_best_probe": {
        "pca_dim": 64,
        "min_pos": 8,
        "C": 0.50,
        "alpha": 0.40,
    },
}

CFG["full_cache_work_dir"].mkdir(parents=True, exist_ok=True)

print("TensorFlow:", tf.__version__)
print("Competition dir exists:", BASE.exists())
print("Model dir exists:", MODEL_DIR.exists())
print(json.dumps(
    {k: (str(v) if isinstance(v, Path) else v) for k, v in CFG.items()},
    indent=2
))
