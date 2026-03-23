import gc
from priors import *
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

print("#############################################")
BASE = Path("..\\data")
test_paths = sorted((BASE / "test_soundscapes").glob("*.ogg"))

Y_SC = np.zeros((len(sc_clean), N_CLASSES), dtype=np.uint8)

for i, labels in enumerate(sc_clean["label_list"]):
    idxs = [label_to_idx[lbl] for lbl in labels if lbl in label_to_idx]
    if idxs:
        Y_SC[i, idxs] = 1


meta_test, scores_test, emb_test = infer_perch_with_embeddings(
    test_paths,
    batch_files=CFG["batch_files"],
    verbose=CFG["verbose"],
    proxy_reduce="max",
)

print(scores_test)


final_prior_tables = fit_prior_tables(sc_clean.copy(), Y_SC)

test_base_scores, test_prior_scores = fuse_scores_with_tables(
    scores_test,
    sites=meta_test["site"].to_numpy(),
    hours=meta_test["hour_utc"].to_numpy(),
    tables=final_prior_tables,
)

print(meta_test.head())

print()
