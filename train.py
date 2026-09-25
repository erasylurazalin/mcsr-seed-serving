"""Train the MLP, log it to MLflow, export it to ONNX and check the export.

Same target and split as model.py in mcsr-seed-difficulty: log(minutes), 80/20
grouped by seed_id with random_state 0, so the numbers are comparable to the
table in that README. HistGradientBoosting is refitted here on the exact same
encoded features, so the comparison is on identical inputs.

    .venv/bin/python train.py
    .venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import argparse
import time
from pathlib import Path

import mlflow
import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn

from serving import features

SEED = 0
ROOT = Path(__file__).parent
DATA = ROOT / "data" / "runs.parquet"
MODEL_DIR = ROOT / "model"


class RunTimeMLP(nn.Module):
    """Raw feature vector in, minutes out.

    Input scaling and the exp() on the output live inside the network, so they
    end up inside the ONNX graph too. The API then passes encode() output
    straight in and reads minutes straight out, with no statistics to keep in
    sync on the serving side.
    """

    def __init__(self, x_mean, x_std, y_mean, y_std, hidden=(128, 64), dropout=0.1):
        super().__init__()
        self.register_buffer("x_mean", torch.as_tensor(x_mean))
        self.register_buffer("x_std", torch.as_tensor(x_std))
        self.register_buffer("y_mean", torch.tensor(y_mean))
        self.register_buffer("y_std", torch.tensor(y_std))

        layers, width = [], len(x_mean)
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
            width = h
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def standardized(self, x):
        """Prediction on the standardized log scale, which is what training fits."""
        return self.net((x - self.x_mean) / self.x_std).squeeze(-1)

    def forward(self, x):
        return torch.exp(self.standardized(x) * self.y_std + self.y_mean)


def records(df):
    return [{"elo": (r.elo_min, r.elo_max), "overworld": r.overworld,
             "nether": r.nether, "end_towers": list(r.end_towers),
             "variations": list(r.variations)}
            for r in df.itertuples()]


def grouped_split(df, test_size, seed=SEED):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    a, b = next(splitter.split(df, groups=df.seed_id))
    return df.iloc[a], df.iloc[b]


def score(pred_minutes, true_minutes):
    return {
        "mae_min": float(mean_absolute_error(true_minutes, pred_minutes)),
        "r2_log": float(r2_score(np.log(true_minutes), np.log(pred_minutes))),
    }


def fit_mlp(Xtr, ytr, Xval, yval, args):
    torch.manual_seed(SEED)
    x_std = Xtr.std(axis=0)
    x_std[x_std == 0] = 1
    model = RunTimeMLP(Xtr.mean(axis=0), x_std, float(ytr.mean()), float(ytr.std()),
                       hidden=args.hidden, dropout=args.dropout)

    Xtr_t, Xval_t = torch.from_numpy(Xtr), torch.from_numpy(Xval)
    ytr_t = torch.from_numpy(((ytr - ytr.mean()) / ytr.std()).astype(np.float32))
    yval_t = torch.from_numpy(((yval - ytr.mean()) / ytr.std()).astype(np.float32))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best, best_state, stale = float("inf"), None, 0

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad()
            loss_fn(model.standardized(Xtr_t[idx]), ytr_t[idx]).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val = loss_fn(model.standardized(Xval_t), yval_t).item()
        mlflow.log_metric("val_loss", val, step=epoch)
        print(f"  epoch {epoch:>2}  val loss {val:.4f}")

        # early stopping on a validation slice of train, never on test
        if val < best - 1e-4:
            best, stale = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


def export_onnx(model, n_features, path):
    program = torch.onnx.export(
        model, (torch.zeros(2, n_features),),
        input_names=["features"], output_names=["minutes"],
        dynamic_shapes={"x": {0: torch.export.Dim("batch")}},
        dynamo=True,
    )
    # one self-contained file, no separate .data file for the weights
    program.save(str(path), external_data=False)


def main(args):
    df = pd.read_parquet(DATA)
    train, test = grouped_split(df, 0.2)
    fit_part, val_part = grouped_split(train, 0.1, seed=SEED + 1)
    print(f"{len(fit_part):,} fit, {len(val_part):,} val, {len(test):,} test")

    spec = features.fit(records(train))
    Xfit, Xval, Xtest = (features.encode(spec, records(d)) for d in (fit_part, val_part, test))
    yfit, yval = np.log(fit_part.minutes.to_numpy()), np.log(val_part.minutes.to_numpy())
    print(f"{Xfit.shape[1]} features")

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("mcsr-run-time")

    with mlflow.start_run():
        mlflow.log_params({**vars(args), "n_features": Xfit.shape[1],
                           "n_train": len(fit_part), "n_test": len(test)})

        t = time.time()
        mlp = fit_mlp(Xfit, yfit, Xval, yval, args)
        mlflow.log_metric("train_seconds", time.time() - t)
        with torch.no_grad():
            torch_pred = mlp(torch.from_numpy(Xtest)).numpy()
        mlp_scores = score(torch_pred, test.minutes)
        mlflow.log_metrics({f"test_{k}": v for k, v in mlp_scores.items()})

        # the reference: same features, same rows, the model from the analysis
        hgb = HistGradientBoostingRegressor(random_state=SEED, max_iter=300)
        hgb.fit(np.vstack([Xfit, Xval]), np.concatenate([yfit, yval]))
        hgb_scores = score(np.exp(hgb.predict(Xtest)), test.minutes)
        mlflow.log_metrics({f"hgb_test_{k}": v for k, v in hgb_scores.items()})

        MODEL_DIR.mkdir(exist_ok=True)
        onnx_path, spec_path = MODEL_DIR / "model.onnx", MODEL_DIR / "features.json"
        export_onnx(mlp, Xfit.shape[1], onnx_path)
        spec.save(spec_path)

        # the ONNX file is what gets served, so check it says what PyTorch says
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        onnx_pred = sess.run(None, {"features": Xtest})[0]
        diff = float(np.abs(onnx_pred - torch_pred).max())
        mlflow.log_metric("onnx_max_abs_diff_min", diff)
        if diff > 1e-3:
            raise SystemExit(f"ONNX output differs from PyTorch by {diff:.2e} minutes")

        mlflow.log_artifact(str(onnx_path))
        mlflow.log_artifact(str(spec_path))

    print(f"\n{'':<22}{'mean abs error':>16}{'r2 (log scale)':>16}")
    for name, s in [("MLP", mlp_scores), ("HistGradientBoosting", hgb_scores)]:
        print(f"{name:<22}{s['mae_min']:>12.2f} min{s['r2_log']:>16.3f}")
    print(f"\nONNX vs PyTorch, largest difference on {len(test):,} test runs: {diff:.2e} min")
    print(f"wrote {onnx_path} ({onnx_path.stat().st_size / 1024:.0f} KB) and {spec_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    main(p.parse_args())
