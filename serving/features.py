"""Turn one match into the model's input vector.

Training and the API both go through `encode`, so there is exactly one place
where a match becomes numbers. If the two had their own copies, they would
drift apart and the API would quietly serve a different model than the one
that was evaluated.

The encoding matches model.py in mcsr-seed-difficulty: elo mean and gap per
100 elo, one hot overworld and nether, end tower min and mean, and a multi-hot
of every variation tag seen at least 200 times. One difference: the tag list
and the tower medians are fitted on the training set only and then frozen.

No pandas here on purpose. The serving image only needs numpy.
"""

import json
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np

MIN_TAG_COUNT = 200


@dataclass
class FeatureSpec:
    overworld: list[str]
    nether: list[str]
    variations: list[str]
    tower_min_median: float
    tower_mean_median: float

    @property
    def columns(self):
        return (["elo_mean", "elo_gap"]
                + [f"ow_{v}" for v in self.overworld]
                + [f"nether_{v}" for v in self.nether]
                + ["tower_min", "tower_mean"]
                + self.variations)

    def save(self, path):
        with open(path, "w") as f:
            json.dump({**asdict(self), "columns": self.columns}, f, indent=1)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        d.pop("columns")
        return cls(**d)


def fit(matches, min_tag_count=MIN_TAG_COUNT):
    """Build the spec from training matches only."""
    tags = Counter(t for m in matches for t in m["variations"])
    towers = [m["end_towers"] for m in matches if m["end_towers"]]
    return FeatureSpec(
        overworld=sorted({m["overworld"] for m in matches}),
        nether=sorted({m["nether"] for m in matches if m["nether"]}),
        variations=sorted(t for t, n in tags.items() if n >= min_tag_count),
        tower_min_median=float(np.median([min(t) for t in towers])),
        tower_mean_median=float(np.median([np.mean(t) for t in towers])),
    )


def encode(spec, matches):
    """Encode a list of matches into a float32 matrix, one row per match.

    A match is a dict with `elo` (both players), `overworld`, `nether`,
    `end_towers` and `variations`. Anything the spec has never seen, an unknown
    seed type or a rare tag, simply sets no column, the same as it did in
    training.
    """
    index = {c: i for i, c in enumerate(spec.columns)}
    X = np.zeros((len(matches), len(index)), dtype=np.float32)

    for row, m in enumerate(matches):
        lo, hi = min(m["elo"]), max(m["elo"])
        X[row, 0] = (lo + hi) / 2 / 100
        X[row, 1] = (hi - lo) / 100

        for key in (f"ow_{m['overworld']}", f"nether_{m['nether']}", *m["variations"]):
            col = index.get(key)
            if col is not None:
                X[row, col] = 1

        towers = m["end_towers"]
        X[row, index["tower_min"]] = min(towers) if towers else spec.tower_min_median
        X[row, index["tower_mean"]] = np.mean(towers) if towers else spec.tower_mean_median

    return X
