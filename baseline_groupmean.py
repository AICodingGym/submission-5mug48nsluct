"""Baseline 0: predict the training-set median pressure for each
(R, C, step-within-breath) cell. A sanity floor, no learning of u_in at all."""
import numpy as np

from common import Timer, insp_mae, load, log_result, split, write_submission

train, test = load("train"), load("test")
for df in (train, test):
    df["step"] = df.groupby("breath_id").cumcount()

# Data check: R and C should be constant within each breath.
assert (train.groupby("breath_id")[["R", "C"]].nunique() == 1).all().all()
assert (test.groupby("breath_id")[["R", "C"]].nunique() == 1).all().all()

is_val = split(train)
tr, va = train[~is_val], train[is_val]
keys = ["R", "C", "step"]

with Timer() as t:
    table = tr.groupby(keys)["pressure"].median().rename("pred").reset_index()
    pred = va.merge(table, on=keys, how="left")["pred"].to_numpy()
mae = insp_mae(va["pressure"], pred, va["u_out"])
log_result("groupmedian_RC_step", mae, t.s, "cpu", "median pressure per (R,C,step)")

full = train.groupby(keys)["pressure"].median().rename("pred").reset_index()
write_submission(test, test.merge(full, on=keys, how="left")["pred"].to_numpy(), "groupmedian_RC_step")
print("train breaths:", tr.breath_id.nunique(), "val breaths:", va.breath_id.nunique())
