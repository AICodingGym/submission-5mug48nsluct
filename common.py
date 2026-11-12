"""Shared pieces for every experiment: data loading, the fixed validation split,
the competition metric, and the results log."""
import csv
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(ROOT, "cache")
SPLIT_FILE = os.path.join(ROOT, "splits", "val_breath_ids.csv")
RESULTS_FILE = os.path.join(ROOT, "results.csv")
SUBMISSIONS = os.path.join(ROOT, "submissions")

DTYPES = {"id": "int32", "breath_id": "int32", "R": "int8", "C": "int8",
          "time_step": "float32", "u_in": "float32", "u_out": "int8",
          "pressure": "float32"}


def load(name):
    """Load train/test, cached as parquet after the first read. Rows are sorted
    by (breath_id, id) so every breath is 80 consecutive rows."""
    os.makedirs(CACHE, exist_ok=True)
    pq = os.path.join(CACHE, f"{name}.parquet")
    if os.path.exists(pq):
        return pd.read_parquet(pq)
    df = pd.read_csv(os.path.join(DATA, f"{name}.csv"), dtype=DTYPES)
    df = df.sort_values(["breath_id", "id"]).reset_index(drop=True)
    df.to_parquet(pq, index=False)
    return df


def make_split(train, frac=0.2, seed=42):
    """Hold out `frac` of breaths, stratified by (R, C). Written once and then
    reused so every experiment is scored on the same breaths."""
    if os.path.exists(SPLIT_FILE):
        return set(pd.read_csv(SPLIT_FILE)["breath_id"])
    rng = np.random.default_rng(seed)
    breaths = train.groupby("breath_id")[["R", "C"]].first().reset_index()
    val = []
    for _, g in breaths.groupby(["R", "C"]):
        ids = g["breath_id"].to_numpy()
        val.extend(rng.choice(ids, size=round(len(ids) * frac), replace=False))
    os.makedirs(os.path.dirname(SPLIT_FILE), exist_ok=True)
    pd.DataFrame({"breath_id": sorted(val)}).to_csv(SPLIT_FILE, index=False)
    return set(val)


def split(train):
    val_ids = make_split(train)
    is_val = train["breath_id"].isin(val_ids).to_numpy()
    return is_val


def insp_mae(y_true, y_pred, u_out):
    """Competition metric: MAE over inspiratory rows (u_out == 0) only."""
    m = np.asarray(u_out) == 0
    return float(np.mean(np.abs(np.asarray(y_true)[m] - np.asarray(y_pred)[m])))


def log_result(name, val_mae, runtime_s, device, notes=""):
    new = not os.path.exists(RESULTS_FILE)
    with open(RESULTS_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "experiment", "val_insp_mae", "runtime_s", "device", "notes"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), name,
                    f"{val_mae:.5f}", f"{runtime_s:.1f}", device, notes])
    print(f"[{name}] val inspiratory MAE = {val_mae:.5f}  ({runtime_s:.1f}s on {device})")


def write_submission(test, pred, name):
    """Write id,pressure in sample_submission order; checks shape and finiteness."""
    sample = pd.read_csv(os.path.join(DATA, "sample_submission.csv"))
    sub = pd.DataFrame({"id": test["id"].to_numpy(), "pressure": np.asarray(pred, dtype=np.float64)})
    sub = sample[["id"]].merge(sub, on="id", how="left")
    assert len(sub) == len(sample) and np.isfinite(sub["pressure"]).all()
    os.makedirs(SUBMISSIONS, exist_ok=True)
    path = os.path.join(SUBMISSIONS, f"{name}.csv")
    sub.to_csv(path, index=False)
    print(f"wrote {path}")
    return path


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *a):
        self.s = time.time() - self.t0
