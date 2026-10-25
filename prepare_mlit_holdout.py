"""Build an actual latest-quarter MLIT holdout for chronological model comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from legacy_engine import normalize
from research_adaptive import prepare_site_training


def build(root: Path, output: Path, metadata: Path) -> dict:
    parts = []
    sources = []
    for path in sorted((root / "data").glob("mlit_*.csv")):
        if path.name == output.name:
            continue
        raw = pd.read_csv(path, low_memory=False)
        rows = normalize(raw)
        if not rows.empty:
            parts.append(rows)
            sources.append(path.name)
    if not parts:
        raise RuntimeError("No usable MLIT transaction CSVs were found")
    all_rows = pd.concat(parts, ignore_index=True, sort=False).drop_duplicates()
    periods = all_rows[["Year", "Quarter"]].dropna().astype(int).drop_duplicates()
    if periods.empty:
        raise RuntimeError("MLIT data has no transaction year/quarter")
    latest = tuple(periods.sort_values(["Year", "Quarter"]).iloc[-1].tolist())
    selected = all_rows.loc[all_rows.Year.eq(latest[0]) & all_rows.Quarter.eq(latest[1])]
    holdout = prepare_site_training(selected)
    if len(holdout) < 200:
        raise RuntimeError(f"Latest MLIT quarter has too few used-house rows: {len(holdout)}")
    holdout["price_yen"] = holdout["price_man"] * 10000
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".tmp")
    holdout.to_csv(temp, index=False, encoding="utf-8")
    temp.replace(output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    report = {
        "source": "MLIT REINFOLIB XIT001",
        "year": int(latest[0]),
        "quarter": int(latest[1]),
        "rows": int(len(holdout)),
        "holdout_sha256": digest,
        "source_files": sources,
    }
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=Path("data/unused_holdout.csv"))
    parser.add_argument("--metadata", type=Path, default=Path("data/mlit_latest_holdout.json"))
    args = parser.parse_args()
    if not args.output.is_absolute():
        args.output = args.root / args.output
    if not args.metadata.is_absolute():
        args.metadata = args.root / args.metadata
    report = build(args.root, args.output, args.metadata)
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
            stream.write(f"sha256={report['holdout_sha256']}\n")
            stream.write(f"year={report['year']}\n")
            stream.write(f"quarter={report['quarter']}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
