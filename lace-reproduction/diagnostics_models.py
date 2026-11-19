import argparse
import time
from pathlib import Path

import numpy as np

from common import SEED, atomic, sha256, write_json

MODEL_NAMES = ("LACE", "MoE", "MLP", "LinearSVR", "XGBoost", "Linear", "LightGBM", "GBM")
VARIANTS = ("full", "wo_cls", "wo_static", "wo_statistical", "wo_edcs", "wo_deltas")


def evaluation(out):
    source = out / "representative.npz"
    with np.load(source) as trace:
        y = trace["target_s"]
        eligible = np.flatnonzero(np.isfinite(y))
        indexes = np.sort(np.random.default_rng(SEED).choice(eligible, min(300000, len(eligible)), replace=False))
        x = trace["features"][indexes]
    def save(path):
        with path.open("wb") as stream:
            np.savez_compressed(stream, x=x, y=y[indexes], event_index=indexes)
    atomic(out / "diagnostic_test.npz", save)
    write_json(out / "diagnostic_test.json", dict(test_examples=len(indexes), seed=SEED,
               replay_sha256=sha256(source), sample_sha256=sha256(out / "diagnostic_test.npz")))


def metrics(pred, target):
    error = pred - target
    mse = np.mean(error**2)
    nonzero = target > 0
    return dict(examples=len(target), mape_examples=int(nonzero.sum()),
                mae_s=float(np.mean(np.abs(error))), rmse_s=float(np.sqrt(mse)),
                mape_pct=float(100*np.mean(np.abs(error[nonzero]/target[nonzero]))),
                r2=float(1-mse/target.var()), mse_s2=float(mse))


def baseline(name, out):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVR

    if name == "MLP":
        estimator = MLPRegressor(hidden_layer_sizes=(96, 96, 96), max_iter=24, batch_size=512,
                                 random_state=SEED, early_stopping=False)
    elif name == "LinearSVR":
        estimator = LinearSVR(C=.1, dual=False, loss="squared_epsilon_insensitive", max_iter=3000, random_state=SEED)
    elif name == "Linear":
        estimator = LinearRegression()
    elif name == "GBM":
        estimator = GradientBoostingRegressor(n_estimators=80, max_depth=3, random_state=SEED)
    elif name == "XGBoost":
        from xgboost import XGBRegressor
        estimator = XGBRegressor(n_estimators=120, max_depth=6, tree_method="hist", n_jobs=4, random_state=SEED)
    elif name == "LightGBM":
        from lightgbm import LGBMRegressor
        estimator = LGBMRegressor(n_estimators=120, n_jobs=4, random_state=SEED, verbosity=-1)
    else:
        raise ValueError(name)
    learner = make_pipeline(StandardScaler(), estimator)
    with np.load(out / "fit.npz") as train, np.load(out / "diagnostic_test.npz") as test:
        started = time.monotonic()
        learner.fit(train["train_x"], train["train_y"]/600)
        fit_s = time.monotonic() - started
        pred = learner.predict(test["x"])*600
        result = metrics(pred, test["y"])
    write_json(out / "method_scores" / f"{name}.json",
               dict(model=name, status="complete", fit_elapsed_s=fit_s,
                    config=repr(estimator), test_sha256=sha256(out / "diagnostic_test.npz"), **result))


def learned(out, variant):
    from model import Predictor, configure
    configure(1)
    checkpoint = out / variant / "model.pt"
    predictor = Predictor(checkpoint)
    with np.load(out / "diagnostic_test.npz") as test:
        pred, _ = predictor.predict(test["x"])
        result = metrics(pred, test["y"])
    write_json(out / "method_scores" / f"{variant}.json",
               dict(model="LACE" if variant == "full" else "MoE" if variant == "wo_cls" else variant,
                    status="complete", model_sha256=sha256(checkpoint),
                    test_sha256=sha256(out / "diagnostic_test.npz"), **result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("evaluation", "baseline", "learned", "train"))
    parser.add_argument("out", type=Path)
    parser.add_argument("variant", nargs="?")
    opts = parser.parse_args()
    if opts.stage == "evaluation":
        evaluation(opts.out)
    elif opts.stage == "baseline":
        baseline(opts.variant, opts.out)
    elif opts.stage == "learned":
        learned(opts.out, opts.variant)
    else:
        from model import train
        train(opts.out / "fit.npz", opts.out / opts.variant, 24, SEED, 4, opts.variant, 96)
