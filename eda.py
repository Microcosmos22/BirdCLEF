from config import *
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


taxonomy = pd.read_csv(BASE / "taxonomy.csv")
sample_sub = pd.read_csv(BASE / "sample_submission.csv")
soundscape_labels = pd.read_csv(BASE / "train_soundscapes_labels.csv")

#print(taxonomy.head())
"""
  primary_label  inat_taxon_id         scientific_name                 common_name class_name
0       1161364        1161364            Guyalna cuta                Guyalna cuta    Insecta
1        116570         116570           Caiman yacare  Southern Spectacled Caiman   Reptilia
"""
#print(sample_sub.head())
"""
                                    row_id   1161364    116570  ...    yecpar   yehcar1   yeofly1
0   BC2026_Test_0001_S05_20250227_010002_5  0.004274  0.004274  ...  0.004274  0.004274  0.004274
1  BC2026_Test_0001_S05_20250227_010002_10  0.004274  0.004274  ...  0.004274  0.004274  0.004274
"""
#print(soundscape_labels.head())
"""
[3 rows x 235 columns]
                                    filename     start       end                   primary_label
0  BC2026_Train_0039_S22_20211231_201500.ogg  00:00:00  00:00:05  22961;23158;24321;517063;65380
1  BC2026_Train_0039_S22_20211231_201500.ogg  00:00:05  00:00:10  22961;23158;24321;517063;65380
"""

PRIMARY_LABELS = sample_sub.columns[1:].tolist() # id of every possible species
N_CLASSES = len(PRIMARY_LABELS)

taxonomy["primary_label"] = taxonomy["primary_label"].astype(str)
soundscape_labels["primary_label"] = soundscape_labels["primary_label"].astype(str)

def parse_soundscape_labels(x):
    if pd.isna(x):
        return []
    return [t.strip() for t in str(x).split(";") if t.strip()]

FNAME_RE = re.compile(r"BC2026_(?:Train|Test)_(\d+)_(S\d+)_(\d{8})_(\d{6})\.ogg")

def parse_soundscape_filename(name):
    """
    From the the filename, it gets the properties below and returns a dictionary with them
    """
    m = FNAME_RE.match(name)
    if not m:
        return {
            "file_id": None,
            "site": None,
            "date": pd.NaT,
            "time_utc": None,
            "hour_utc": -1,
            "month": -1,
        }
    file_id, site, ymd, hms = m.groups()
    dt = pd.to_datetime(ymd, format="%Y%m%d", errors="coerce")
    return {
        "file_id": file_id,
        "site": site,
        "date": dt,
        "time_utc": hms,
        "hour_utc": int(hms[:2]),
        "month": int(dt.month) if pd.notna(dt) else -1,
    }

def union_labels(series):
    """
    Takes last series primary_label: 22961;23158;24321;517063;65380 etc
    and returns each of the labels in a list(set)
    """
    return sorted(set(lbl for x in series for lbl in parse_soundscape_labels(x)))

# Deduplicate duplicated rows and aggregate labels per 5s window
sc_clean = (
    soundscape_labels
    .groupby(["filename", "start", "end"])["primary_label"]
    .apply(union_labels)
    .reset_index(name="label_list")
)


sc_clean["start_sec"] = pd.to_timedelta(sc_clean["start"]).dt.total_seconds().astype(int)
sc_clean["end_sec"] = pd.to_timedelta(sc_clean["end"]).dt.total_seconds().astype(int)
sc_clean["row_id"] = sc_clean["filename"].str.replace(".ogg", "", regex=False) + "_" + sc_clean["end_sec"].astype(str)

"""
This concatenates/appends some new columns, the ones from the dictionary from the parse function
"""
meta = sc_clean["filename"].apply(parse_soundscape_filename).apply(pd.Series)
sc_clean = pd.concat([sc_clean, meta], axis=1)

windows_per_file = sc_clean.groupby("filename").size()

full_files = sorted(windows_per_file[windows_per_file == N_WINDOWS].index.tolist())
sc_clean["file_fully_labeled"] = sc_clean["filename"].isin(full_files)

# Multi-hot label matrix aligned with sc_clean row order
label_to_idx = {c: i for i, c in enumerate(PRIMARY_LABELS)}
Y_SC = np.zeros((len(sc_clean), N_CLASSES), dtype=np.uint8)


for i, labels in enumerate(sc_clean["label_list"]):
    idxs = [label_to_idx[lbl] for lbl in labels if lbl in label_to_idx]
    if idxs:
        Y_SC[i, idxs] = 1

full_truth = (
    sc_clean[sc_clean["file_fully_labeled"]]
    .sort_values(["filename", "end_sec"])
    .reset_index(drop=False)
)

Y_FULL_TRUTH = Y_SC[full_truth["index"].to_numpy()]


if __name__ == "__main__":
    """
    In BirdCLEF soundscape labels, it's common that:
    The same 5-second window appears multiple times
    Each row may list different species
    """
    # Fully-labeled files
    print(f"SOUNDSCAPE_LABELS LEN: {len(soundscape_labels)}")
    print(soundscape_labels.head())
    print(soundscape_labels.columns)
    print(f"SC_CLEAN LEN: {len(sc_clean)}")
    print(sc_clean.head())
    print(sc_clean.columns)

    soundscape_labels.to_csv("soundscape_labels.csv")
    sc_clean.to_csv("sc_clean.csv")

    #print([len(i) for i in sc_clean["label_list"]])

    print(windows_per_file)

    print("sc_clean:", sc_clean.shape)
    print("Y_SC:", Y_SC.shape, Y_SC.dtype)
    print("Full files:", len(full_files))
    print("Trusted full windows:", len(full_truth))
    print("Active classes in full windows:", int((Y_FULL_TRUTH.sum(axis=0) > 0).sum()))
