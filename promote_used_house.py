"""Fail-closed promotion of a used-house model after independent evaluation.

Usage:
 python promote_used_house.py --incumbent models/candidates/v7_used_house_candidate.joblib --challenger models/candidates/adaptive_used_house_candidate.joblib --holdout data/unused_holdout.csv --holdout-year 2026 --site .
The holdout must contain actual price in yen (price_yen) and all model features.
Do not use a holdout that has previously been used for model selection.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
from evaluation_policy import independence, sha
import joblib
import numpy as np
import pandas as pd


def predict(bundle, data):
    if not isinstance(bundle, dict) or bundle.get("model_key") != "used_house" or bundle.get("target_unit") != "yen":
        raise ValueError("Unsupported model type or target unit")
    if bundle.get("ensemble_components") or any(bundle.get(k) is not None for k in ("post_calibration", "calibration", "verified_post_calibration")):
        raise ValueError("Complex incumbent calibration cannot be reproduced safely")
    cols = bundle.get("feature_columns")
    pipe = bundle.get("pipeline")
    if not isinstance(cols, (list, tuple)) or not cols or not hasattr(pipe, "predict"):
        raise ValueError("Missing model pipeline or feature columns")
    missing = set(cols).difference(data.columns)
    if missing:
        raise ValueError("Missing evaluation features: " + ", ".join(sorted(missing)))
    # Evaluate both contenders with the same fields the Site can supply.
    frame = data[list(cols)].copy()
    if 'StationMinutes' in frame and 'StationMinutes' not in bundle.get('site_available_features', []): frame['StationMinutes'] = np.nan
    if 'PriceCategory' in frame: frame['PriceCategory'] = np.nan
    prediction = np.asarray(pipe.predict(frame), dtype=float)
    if prediction.shape != (len(data),) or not np.all(np.isfinite(prediction)) or np.any(prediction <= 0):
        raise ValueError("Invalid prediction")
    return prediction


def score(y, pred):
    errors = np.abs(y - pred)
    return {"mae_yen": float(errors.mean()),
            "median_ape_pct": float(np.median(errors / y) * 100),
            "within20_pct": float(np.mean(errors / y <= 0.20) * 100)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--incumbent", type=Path, required=True)
    p.add_argument("--challenger", type=Path, required=True)
    p.add_argument("--holdout", type=Path, required=True)
    p.add_argument("--holdout-year", type=int, required=True)
    p.add_argument("--site", type=Path, default=Path("."))
    p.add_argument("--min-rows", type=int, default=200)
    args = p.parse_args()
    data = pd.read_csv(args.holdout)
    if len(data) < args.min_rows or "price_yen" not in data:
        p.error("Independent holdout is missing price_yen or has too few records")
    y = pd.to_numeric(data["price_yen"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(y)) or np.any(y <= 0):
        p.error("Invalid holdout prices")
    if "TransactionYear" not in data.columns:
        p.error("Holdout requires TransactionYear for chronological validation")
    years = pd.to_numeric(data["TransactionYear"], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(years)) or np.any(years < args.holdout_year):
        p.error("Holdout contains records older than the required independent evaluation year")
    target_dir = args.site / "models" / "candidates"
    manifest_path = target_dir / "approved_used_house.json"
    holdout_sha256 = hashlib.sha256(args.holdout.read_bytes()).hexdigest()
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("holdout_sha256") == holdout_sha256:
            p.error("This holdout was already used for model approval; provide fresh unseen data")
    incumbent = joblib.load(args.incumbent)
    challenger = joblib.load(args.challenger)
    try:
        evidence = independence(data, args.holdout, {
            sha(args.incumbent): incumbent.get('training_end_year'),
            sha(args.challenger): challenger.get('training_end_year'),
        })
    except ValueError as exc:
        p.error(str(exc))
    before = score(y, predict(incumbent, data))
    after = score(y, predict(challenger, data))
    report = {"incumbent": before, "challenger": after, "holdout_year": args.holdout_year,
              "holdout_rows": len(data), "holdout_sha256": holdout_sha256,
              "input_policy": "site-residential-used-house-v2", "status": "rejected",
              "independence": evidence, "candidate_sha256": sha(args.challenger)}
    # Require material accuracy improvement and no significant regression.
    approved = (after["median_ape_pct"] <= before["median_ape_pct"] - 0.5
                and after["mae_yen"] <= before["mae_yen"] * 1.01
                and after["within20_pct"] >= before["within20_pct"] - 1)
    if approved:
        target_dir = args.site / "models" / "candidates"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / "approved_used_house.joblib"
        temp = target.with_suffix(".tmp")
        try:
            bundle = dict(challenger)
            bundle["candidate_only"] = False
            bundle["approval_status"] = "approved"
            joblib.dump(bundle, temp)
            checked = joblib.load(temp)
            predict(checked, data.iloc[:min(len(data), 10)])
            digest = hashlib.sha256(temp.read_bytes()).hexdigest()
            os.replace(temp, target)
            report["status"] = "approved"
            manifest = {"approved": True, "filename": target.name, "sha256": digest,
                        "holdout_sha256": holdout_sha256,
                        "evaluation": report, "model_key": "used_house"}
            temp_manifest = target_dir / "approved_used_house.json.tmp"
            temp_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(temp_manifest, target_dir / "approved_used_house.json")
        finally:
            temp.unlink(missing_ok=True)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not approved:
        raise SystemExit("No safe improvement; incumbent retained")


if __name__ == "__main__":
    main()
