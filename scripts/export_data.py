"""Export finished runs from the analysis repo into one parquet file.

The match database is 2 GB and lives in mcsr-seed-difficulty. This repo only
needs the columns the model reads, so it takes those through that repo's own
dataset.load, which is where forfeits and non-ranked matches get filtered out.

    .venv/bin/python scripts/export_data.py ~/projects/mcsr-seed-difficulty
"""

import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "runs.parquet"


def main(analysis_repo):
    sys.path.insert(0, str(Path(analysis_repo).expanduser().resolve()))
    import dataset

    df = dataset.load(include_censored=False)
    df = df[["id", "seed_id", "minutes", "elo_min", "elo_max",
             "overworld", "nether", "end_towers", "variations"]]

    OUT.parent.mkdir(exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"{len(df):,} finished runs -> {OUT}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
