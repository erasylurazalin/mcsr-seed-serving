import json
from pathlib import Path

from fastapi.testclient import TestClient

from serving import features
from serving.app import app, spec

client = TestClient(app)
EXAMPLE = json.loads((Path(__file__).parent.parent / "examples" / "match.json").read_text())


def test_predict_returns_plausible_minutes():
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200
    # ranked runs finish somewhere between a few minutes and about an hour
    assert 3 < r.json()["minutes"] < 60


def test_higher_elo_predicts_faster():
    low = client.post("/predict", json={**EXAMPLE, "elo": [800, 820]}).json()
    high = client.post("/predict", json={**EXAMPLE, "elo": [1800, 1820]}).json()
    assert high["minutes"] < low["minutes"]


def test_player_order_does_not_matter():
    a = client.post("/predict", json={**EXAMPLE, "elo": [1500, 1400]}).json()
    b = client.post("/predict", json={**EXAMPLE, "elo": [1400, 1500]}).json()
    assert a["minutes"] == b["minutes"]


def test_rejects_one_player():
    assert client.post("/predict", json={**EXAMPLE, "elo": [1500]}).status_code == 422


def test_unknown_values_set_no_columns():
    X = features.encode(spec, [{"elo": [1000, 1000], "overworld": "NOT_A_SEED",
                                "nether": None, "end_towers": [],
                                "variations": ["never:seen"]}])
    # elo mean, and the two tower medians standing in for missing towers
    assert (X != 0).sum() == 3
