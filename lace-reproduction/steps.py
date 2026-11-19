import argparse
from pathlib import Path

from common import SEED, WORKLOADS, write_csv


def predict(out, checkpoint):
    import numpy as np

    from model import Predictor, configure
    from workloads import save_arrays

    configure(1)
    model = Predictor(checkpoint)
    rows = []
    for workload in WORKLOADS:
        stem = workload.lower()
        with np.load(out / f"{stem}.npz") as trace:
            pred, gate = model.predict(trace["features"])
        counts = (gate > 0).sum(0)
        weights = gate.astype(np.float64).sum(0)
        means = np.divide(weights, counts, out=np.zeros_like(weights), where=counts > 0)
        assert (counts.sum() == 3*len(pred))
        assert np.isfinite(pred).all() and (pred >= 0).all()
        save_arrays(out / f"{stem}_predictions.npz", pred_s=pred)
        for i in range(len(counts)):
            rows.append(dict(workload=workload, expert=i+1, top_k_assignments=int(counts[i]),
                             summed_weight=float(weights[i]), mean_selected_weight=float(means[i]),
                             normalized_count=float(counts[i]/counts.sum()),
                             normalized_summed_weight=float(weights[i]/weights.sum()),
                             normalized_mean_weight=float(means[i]/means.sum()), invocations=len(pred)))
    write_csv(out / "routing.csv", rows)


def table(out):
    import numpy as np

    from model import Predictor, configure

    configure(1)
    with np.load(out / "representative.npz") as data:
        proposed = data["target_s"]
        valid = np.isfinite(proposed)
        x = data["features"][valid]
        y = proposed[valid]
    rows = []
    for variant in ("full", "wo_cls"):
        path = out / variant / "model.pt"
        if not path.exists():
            continue
        pred, _ = Predictor(path).predict(x)
        err = pred - y
        nonzero = y > 0
        mse = (err ** 2).mean()
        variance = y.var()
        rows.append(dict(predictor=f"LACE_{variant}", workload="Representative", target_unit="s",
                         examples=len(y), mape_examples=int(nonzero.sum()), zero_targets_excluded=int((~nonzero).sum()),
                         mae_s=np.abs(err).mean(), mse_s2=mse, rmse_s=np.sqrt(mse),
                         mape_pct=100*np.abs(err[nonzero]/y[nonzero]).mean(), r2=1-mse/variance))
    write_csv(out / "table_03_iat_metrics.csv", rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("source", "prepare", "train", "predict", "table"))
    parser.add_argument("arguments", nargs="+")
    options = parser.parse_args()
    a = options.arguments
    if options.stage == "source":
        from data import acquire
        acquire(Path(a[0]), Path(a[1]) if len(a) > 1 else None)
    elif options.stage == "prepare":
        from workloads import build
        build(Path(a[0]), Path(a[1]), int(a[2]), int(a[3]), factor=float(a[4]),
              historical_activity_limit=int(a[5]) if len(a) > 5 else None)
    elif options.stage == "train":
        from model import train
        train(Path(a[0]) / "fit.npz", Path(a[1]), int(a[2]), SEED, 4, a[3])
    elif options.stage == "predict":
        predict(Path(a[0]), Path(a[1]))
    else:
        table(Path(a[0]))
