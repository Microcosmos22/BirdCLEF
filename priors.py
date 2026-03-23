import gc
from model import *
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

# Cell 18 — Build submission
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# Cell 7 — Fold-safe metadata prior tables
def fit_prior_tables(prior_df, Y_prior):
    """
    P(species)
    P(species | site)
    P(species | hour)
    P(species | site, hour)

    These are called priors.
    Later they will be used to improve predictions like:

    If bird X is common at site S1 at 14:00
    → increase probability of bird X

    This is very common in BirdCLEF pipelines.

    Audio
      ↓
    Perch embeddings
      ↓
    Classifier predictions
      ↓
    Metadata priors  ← THIS FUNCTION
      ↓
    Final predictions
    """
    prior_df = prior_df.reset_index(drop=True)

    global_p = Y_prior.mean(axis=0).astype(np.float32)

    # Site
    site_keys = sorted(prior_df["site"].dropna().astype(str).unique().tolist())
    site_to_i = {k: i for i, k in enumerate(site_keys)}
    site_n = np.zeros(len(site_keys), dtype=np.float32)
    site_p = np.zeros((len(site_keys), Y_prior.shape[1]), dtype=np.float32)

    for s in site_keys:
        i = site_to_i[s]
        mask = prior_df["site"].astype(str).values == s
        site_n[i] = mask.sum()
        site_p[i] = Y_prior[mask].mean(axis=0)

    # Hour
    hour_keys = sorted(prior_df["hour_utc"].dropna().astype(int).unique().tolist())
    hour_to_i = {h: i for i, h in enumerate(hour_keys)}
    hour_n = np.zeros(len(hour_keys), dtype=np.float32)
    hour_p = np.zeros((len(hour_keys), Y_prior.shape[1]), dtype=np.float32)

    for h in hour_keys:
        i = hour_to_i[h]
        mask = prior_df["hour_utc"].astype(int).values == h
        hour_n[i] = mask.sum()
        hour_p[i] = Y_prior[mask].mean(axis=0)

    # Site-hour
    sh_to_i = {}
    sh_n_list = []
    sh_p_list = []

    for (s, h), idx in prior_df.groupby(["site", "hour_utc"]).groups.items():
        sh_to_i[(str(s), int(h))] = len(sh_n_list)
        idx = np.array(list(idx))
        sh_n_list.append(len(idx))
        sh_p_list.append(Y_prior[idx].mean(axis=0))

    sh_n = np.array(sh_n_list, dtype=np.float32)
    sh_p = np.stack(sh_p_list).astype(np.float32) if len(sh_p_list) else np.zeros((0, Y_prior.shape[1]), dtype=np.float32)

    return {
        "global_p": global_p,
        "site_to_i": site_to_i,
        "site_n": site_n,
        "site_p": site_p,
        "hour_to_i": hour_to_i,
        "hour_n": hour_n,
        "hour_p": hour_p,
        "sh_to_i": sh_to_i,
        "sh_n": sh_n,
        "sh_p": sh_p,
    }

def prior_logits_from_tables(sites, hours, tables, eps=1e-4):
    n = len(sites)
    p = np.repeat(tables["global_p"][None, :], n, axis=0).astype(np.float32, copy=True)

    site_idx = np.fromiter(
        (tables["site_to_i"].get(str(s), -1) for s in sites),
        dtype=np.int32,
        count=n
    )
    hour_idx = np.fromiter(
        (tables["hour_to_i"].get(int(h), -1) if int(h) >= 0 else -1 for h in hours),
        dtype=np.int32,
        count=n
    )
    sh_idx = np.fromiter(
        (tables["sh_to_i"].get((str(s), int(h)), -1) if int(h) >= 0 else -1 for s, h in zip(sites, hours)),
        dtype=np.int32,
        count=n
    )

    valid = hour_idx >= 0
    if valid.any():
        nh = tables["hour_n"][hour_idx[valid]][:, None]
        wh = nh / (nh + 8.0)
        p[valid] = wh * tables["hour_p"][hour_idx[valid]] + (1.0 - wh) * p[valid]

    valid = site_idx >= 0
    if valid.any():
        ns = tables["site_n"][site_idx[valid]][:, None]
        ws = ns / (ns + 8.0)
        p[valid] = ws * tables["site_p"][site_idx[valid]] + (1.0 - ws) * p[valid]

    valid = sh_idx >= 0
    if valid.any():
        nsh = tables["sh_n"][sh_idx[valid]][:, None]
        wsh = nsh / (nsh + 4.0)
        p[valid] = wsh * tables["sh_p"][sh_idx[valid]] + (1.0 - wsh) * p[valid]

    np.clip(p, eps, 1.0 - eps, out=p)
    return (np.log(p) - np.log1p(-p)).astype(np.float32, copy=False)


def fuse_scores_with_tables(base_scores, sites, hours, tables,
                            lambda_event=BEST["lambda_event"],
                            lambda_texture=BEST["lambda_texture"],
                            lambda_proxy_texture=BEST["lambda_proxy_texture"],
                            smooth_texture=BEST["smooth_texture"]):
    scores = base_scores.copy()
    prior = prior_logits_from_tables(sites, hours, tables)

    # mapped active: species with both base model prediction and prior.
    if len(idx_mapped_active_event):
        scores[:, idx_mapped_active_event] += lambda_event * prior[:, idx_mapped_active_event]

    if len(idx_mapped_active_texture):
        scores[:, idx_mapped_active_texture] += lambda_texture * prior[:, idx_mapped_active_texture]

    # selected frog proxies
    if len(idx_selected_proxy_active_texture):
        scores[:, idx_selected_proxy_active_texture] += lambda_proxy_texture * prior[:, idx_selected_proxy_active_texture]

    # prior-only active unmapped: species with no base prediction but a prior exists.
    if len(idx_selected_prioronly_active_event):
        scores[:, idx_selected_prioronly_active_event] = lambda_event * prior[:, idx_selected_prioronly_active_event]

    if len(idx_selected_prioronly_active_texture):
        scores[:, idx_selected_prioronly_active_texture] = lambda_texture * prior[:, idx_selected_prioronly_active_texture]

    # inactive unmapped
    if len(idx_unmapped_inactive):
        scores[:, idx_unmapped_inactive] = -8.0

    scores = smooth_cols_fixed12(scores, idx_active_texture, alpha=smooth_texture)
    return scores.astype(np.float32, copy=False), prior

# Cell 8 — Honest OOF base/prior meta-features (required for final stacker fit)
def build_oof_base_prior(scores_full_raw, meta_full, sc_clean, Y_SC, n_splits=5, verbose=True):
    """
    2. Stacker = learned combination model

    Input features:
        base_scores (perch)
        prior_scores
        embeddings
        metadata

    Output:
        final predictions
    """
    groups_full = meta_full["filename"].to_numpy()
    gkf = GroupKFold(n_splits=n_splits)

    oof_base = np.zeros_like(scores_full_raw, dtype=np.float32)
    oof_prior = np.zeros_like(scores_full_raw, dtype=np.float32)
    fold_id = np.full(len(meta_full), -1, dtype=np.int16)

    splits = list(gkf.split(scores_full_raw, groups=groups_full))
    iterator = tqdm(splits, desc="OOF base/prior folds", disable=not verbose)

    for fold, (tr_idx, va_idx) in enumerate(iterator, 1):
        """ Loop over folds """
        tr_idx = np.sort(tr_idx)
        va_idx = np.sort(va_idx)

        val_files = set(meta_full.iloc[va_idx]["filename"].tolist())

        # Fold-safe prior tables: exclude all validation files
        prior_mask = ~sc_clean["filename"].isin(val_files).values
        prior_df_fold = sc_clean.loc[prior_mask].reset_index(drop=True)
        Y_prior_fold = Y_SC[prior_mask]

        """ Establish priors from training subset"""
        tables = fit_prior_tables(prior_df_fold, Y_prior_fold)

        """ Fuse perch output with the priors """
        va_base, va_prior = fuse_scores_with_tables(
            scores_full_raw[va_idx],
            sites=meta_full.iloc[va_idx]["site"].to_numpy(),
            hours=meta_full.iloc[va_idx]["hour_utc"].to_numpy(),
            tables=tables,
        )

        oof_base[va_idx] = va_base
        oof_prior[va_idx] = va_prior
        fold_id[va_idx] = fold

    assert (fold_id >= 0).all()
    return oof_base, oof_prior, fold_id

OOF_META_CACHE = CFG["full_cache_work_dir"] / "full_oof_meta_features.npz"

if __name__ == "__main__":
    print("Building OOF meta-features...")
    oof_base, oof_prior, oof_fold_id = build_oof_base_prior(
        scores_full_raw=scores_full_raw,
        meta_full=meta_full,
        sc_clean=sc_clean,
        Y_SC=Y_SC,
        n_splits=5,
        verbose=CFG["verbose"],
    )

    baseline_oof_auc = macro_auc_skip_empty(Y_FULL, oof_base)

    if MODE == "train":
        raw_local_auc = macro_auc_skip_empty(Y_FULL, scores_full_raw)
        print(f"Raw local AUC (not OOF-dependent): {raw_local_auc:.6f}")
        print(f"Honest OOF baseline AUC: {baseline_oof_auc:.6f}")
