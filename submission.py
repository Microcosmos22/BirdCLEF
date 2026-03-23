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

# Cell 16 — Infer Perch on hidden test with embeddings
test_paths = sorted((BASE / "test_soundscapes").glob("*.ogg"))

if len(test_paths) == 0:
    print(f"Hidden test not mounted. Dry-run on first {CFG['dryrun_n_files']} train soundscapes.")
    test_paths = sorted((BASE / "train_soundscapes").glob("*.ogg"))[:CFG["dryrun_n_files"]]
else:
    print(f"Hidden test files: {len(test_paths)}")

meta_test, scores_test_raw, emb_test = infer_perch_with_embeddings(
    test_paths,
    batch_files=CFG["batch_files"],
    verbose=CFG["verbose"],
    proxy_reduce="max",
)

print("meta_test:", meta_test.shape)
print("scores_test_raw:", scores_test_raw.shape)
print("emb_test:", emb_test.shape)

# Fuse raw test scores with priors
test_base_scores, test_prior_scores = fuse_scores_with_tables(
    scores_test_raw,
    sites=meta_test["site"].to_numpy(),
    hours=meta_test["hour_utc"].to_numpy(),
    tables=final_prior_tables,
)

# Use base + prior as final test features
final_test_scores = test_base_scores.copy()

submission = pd.DataFrame(sigmoid(final_test_scores), columns=PRIMARY_LABELS)
submission.insert(0, "row_id", meta_test["row_id"].values)
submission[PRIMARY_LABELS] = submission[PRIMARY_LABELS].astype(np.float32)

expected_rows = len(test_paths) * N_WINDOWS
assert len(submission) == expected_rows, f"Expected {expected_rows}, got {len(submission)}"
assert submission.columns.tolist() == ["row_id"] + PRIMARY_LABELS
assert not submission.isna().any().any()

submission.to_csv("submission.csv", index=False)

print("Saved submission.csv")
print("Submission shape:", submission.shape)
print(submission.iloc[:3, :8])

# Cell 19 — Diagnostics
if MODE == "train":
    modeled_labels = [PRIMARY_LABELS[i] for i in sorted(probe_models.keys())]

    print("Modeled class count:", len(modeled_labels))
    print("First modeled labels:", modeled_labels[:25])

    print("\nSubmission probability stats:")
    print(submission.iloc[:, 1:].stack().describe())

    print("\nAny NaNs:", submission.isna().any().any())
    print("Any probs < 0:", bool((submission.iloc[:, 1:] < 0).any().any()))
    print("Any probs > 1:", bool((submission.iloc[:, 1:] > 1).any().any()))
else:
    print("Submit mode completed.")
