# MCSR run time, served

A small PyTorch model that predicts how long an MCSR Ranked match will take,
exported to ONNX and served over HTTP from a Docker image.

The analysis lives in [mcsr-seed-difficulty](https://github.com/erasylurazalin/mcsr-seed-difficulty).
That repo asks how much a seed costs. This one takes the same data and features
and does the deployment side: training runs tracked in MLflow, an ONNX export
checked against PyTorch, a FastAPI service, and CI that builds the image and
calls it.

## Results

Same target (log minutes), same features, same 80/20 split grouped by seed as
the analysis repo, on 244k finished ranked runs:

```
                        mean abs error  r2 (log scale)
MLP                           2.95 min           0.652
HistGradientBoosting          2.93 min           0.656
```

The MLP ties gradient boosting and does not beat it. That's expected on 93
mostly binary tabular features, and the analysis already showed elo explains
nearly all of it. The MLP is here because it exports to ONNX cleanly, not
because it's the better model.

The ONNX file and PyTorch agree to within 1e-5 minutes on all 48,856 test runs.
`train.py` checks this on every run and fails if they disagree by more than
1e-3.

## How it fits together

```
mcsr-seed-difficulty/data/matches.db  (2 GB, not here)
        |  scripts/export_data.py
        v
data/runs.parquet  ->  train.py  ->  model/model.onnx + model/features.json
                          |                  |
                       MLflow         Docker image, FastAPI + onnxruntime
```

A few choices worth knowing:

- `serving/features.py` is the only code that turns a match into numbers.
  Training and the API both call `encode()`, so the API can't drift away from
  what was evaluated. It's plain numpy so the image doesn't need pandas.
- Input scaling and the `exp()` back to minutes are inside the network, so they
  end up in the ONNX graph. The API has no statistics to keep in sync.
- The tag list and tower medians are fitted on train only and frozen in
  `features.json`.
- The model is 92 KB, so it's committed. CI never needs the data.
- The image has no torch in it, only onnxruntime.

## Running it

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements-train.txt

.venv/bin/python scripts/export_data.py ~/projects/mcsr-seed-difficulty
.venv/bin/python train.py
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Serving:

```sh
docker build -t mcsr-serving .
docker run --rm -p 8000:8000 mcsr-serving

curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d @examples/match.json
# {"minutes":11.66,"model_version":"760e47c09d69"}
```

`model_version` is the first 12 characters of the ONNX file's sha256, so any
response can be traced back to the exact model that produced it.

## Known limitations

- Forfeits are dropped, same as in the analysis. Bad seeds get quit on, so the
  model is optimistic about bad seeds.
- The image is about 400 MB. Most of it is onnxruntime and its dependencies. I
  haven't tried to shrink it.
- No quantization yet. At 92 KB there isn't much to save on size, so it would
  mostly be a latency experiment.
