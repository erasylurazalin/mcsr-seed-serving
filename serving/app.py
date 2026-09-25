"""HTTP API: send a match, get back the predicted run time.

    uvicorn serving.app:app --port 8000

    curl -s localhost:8000/predict -H 'content-type: application/json' \\
      -d @examples/match.json
"""

import hashlib
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel, Field

from serving import features

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).parent.parent / "model"))
ONNX_PATH = MODEL_DIR / "model.onnx"

spec = features.FeatureSpec.load(MODEL_DIR / "features.json")
session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
# short hash of the served file, so a response can be traced to the exact model
MODEL_VERSION = hashlib.sha256(ONNX_PATH.read_bytes()).hexdigest()[:12]

if session.get_inputs()[0].shape[1] != len(spec.columns):
    raise RuntimeError(f"{ONNX_PATH} expects {session.get_inputs()[0].shape[1]} "
                       f"features but features.json describes {len(spec.columns)}")

app = FastAPI(title="MCSR run time")


class Match(BaseModel):
    elo: list[float] = Field(min_length=2, max_length=2, description="both players")
    overworld: str = Field(examples=["RUINED_PORTAL"])
    nether: str | None = Field(default=None, examples=["BRIDGE"])
    end_towers: list[int] = Field(default=[], examples=[[88, 82, 103, 94]])
    variations: list[str] = Field(default=[], examples=[["bastion:triple:1"]])


class Prediction(BaseModel):
    minutes: float
    model_version: str


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict")
def predict(match: Match) -> Prediction:
    X = features.encode(spec, [match.model_dump()])
    minutes = session.run(None, {"features": X})[0]
    return Prediction(minutes=float(np.asarray(minutes).ravel()[0]),
                      model_version=MODEL_VERSION)
