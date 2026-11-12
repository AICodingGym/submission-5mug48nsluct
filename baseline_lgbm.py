"""Baseline 1: LightGBM on per-row features plus simple within-breath context
(cumulative u_in, lags/leads, differences). Trained on inspiratory rows only,
since those are the only rows scored; features are computed over the whole breath."""
import lightgbm as lgb
import numpy as np

from common import Timer, insp_mae, load, log_result, split, write_submission


def features(df):
    g = df.groupby("breath_id")
    df["step"] = g.cumcount().astype("int16")
    df["RC"] = (df["R"].astype("int16") * 100 + df["C"]).astype("category")
    # Volume delivered so far ~ integral of inflow (u_in is a valve %, so a proxy).
    dt = g["time_step"].diff().fillna(0)
    df["u_in_cumsum"] = g["u_in"].cumsum()
    df["u_in_area"] = (df["u_in"] * dt).groupby(df["breath_id"]).cumsum()
    for k in (1, 2, 3):
        df[f"u_in_lag{k}"] = g["u_in"].shift(k).fillna(0)
        df[f"u_in_lead{k}"] = g["u_in"].shift(-k).fillna(0)
    df["u_in_diff1"] = df["u_in"] - df["u_in_lag1"]
    df["u_out_lead1"] = g["u_out"].shift(-1).fillna(1)
    df["u_in_breath_mean"] = g["u_in"].transform("mean")
    df["u_in_breath_max"] = g["u_in"].transform("max")
    return df


FEATS = ["R", "C", "RC", "time_step", "step", "u_in", "u_out", "u_in_cumsum", "u_in_area",
         "u_in_lag1", "u_in_lag2", "u_in_lag3", "u_in_lead1", "u_in_lead2", "u_in_lead3",
         "u_in_diff1", "u_out_lead1", "u_in_breath_mean", "u_in_breath_max"]
PARAMS = dict(objective="regression_l1", learning_rate=0.1, num_leaves=255,
              min_child_samples=50, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, n_jobs=-1, verbose=-1, seed=0)

train, test = features(load("train")), features(load("test"))
is_val = split(train)
insp = (train["u_out"] == 0).to_numpy()
tr, va = train[~is_val & insp], train[is_val]

with Timer() as t:
    model = lgb.train(PARAMS, lgb.Dataset(tr[FEATS], tr["pressure"]), num_boost_round=3000,
                      valid_sets=[lgb.Dataset(va.loc[va.u_out == 0, FEATS], va.loc[va.u_out == 0, "pressure"])],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(250)])
    pred = model.predict(va[FEATS], num_iteration=model.best_iteration)
mae = insp_mae(va["pressure"], pred, va["u_out"])
log_result("lgbm_v1", mae, t.s, "cpu (M4, 10 threads)",
           f"L1 obj, insp-only train, {len(FEATS)} feats, best_iter={model.best_iteration}")

# Error breakdown by lung setting, for deciding what to try next.
va = va.assign(err=np.abs(va["pressure"] - pred))[va.u_out == 0]
print(va.groupby(["R", "C"])["err"].mean().round(3).unstack())

imp = sorted(zip(model.feature_importance("gain"), FEATS), reverse=True)
print("top features (gain):", [f for _, f in imp[:8]])

# Test predictions: refit on all training breaths with the chosen number of rounds.
full = train[insp]
final = lgb.train(PARAMS, lgb.Dataset(full[FEATS], full["pressure"]),
                  num_boost_round=int(model.best_iteration * 1.1))
write_submission(test, final.predict(test[FEATS]), "lgbm_v1")
