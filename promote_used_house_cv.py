"""Provisionally promote a better model using rolling-origin CV on existing data.

This path is deliberately labeled provisional: it does not claim a fresh,
independent market holdout. A later MLIT quarter still runs the stronger
independent comparison and can replace or reject the provisional model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def score(y, pred):
    errors = np.abs(y - pred)
    return {
        'n': int(len(y)),
        'mae_yen': float(errors.mean()),
        'median_ape_pct': float(np.median(errors / y) * 100),
        'within20_pct': float(np.mean(errors / y <= .20) * 100),
    }


def paired_median_ape_delta_ci(y, baseline, challenger, seed=42,
                               repetitions=2000, max_sample=10000):
    """Paired bootstrap CI for challenger minus baseline median APE, in points."""
    y = np.asarray(y, dtype=float)
    base = np.abs(y - np.asarray(baseline, dtype=float)) / y
    cand = np.abs(y - np.asarray(challenger, dtype=float)) / y
    n = len(y)
    sample_size = min(n, max_sample)
    rng = np.random.default_rng(seed)
    deltas = np.empty(repetitions, dtype=float)
    for start in range(0, repetitions, 100):
        count = min(100, repetitions - start)
        indexes = rng.integers(0, n, size=(count, sample_size), dtype=np.int32)
        deltas[start:start + count] = (
            np.median(cand[indexes], axis=1) - np.median(base[indexes], axis=1)
        ) * 100
    low, high = np.percentile(deltas, [2.5, 97.5])
    return [float(low), float(high)]


def evaluate(report, prediction_path, min_rows=1000):
    if report.get('status') != 'completed':
        raise ValueError('Adaptive research report is not marked completed')
    cv = report.get('internal_cv_confirmation') or {}
    if cv.get('method') != 'rolling-origin-confirmation-v1':
        raise ValueError('No compatible rolling-origin confirmation in the research report')
    signature = report.get('data_signature')
    if not isinstance(signature, str) or len(signature) != 64:
        raise ValueError('Missing training-data fingerprint')
    if cv.get('data_signature') != signature:
        raise ValueError('Research and confirmation data fingerprints differ')
    frame = pd.read_csv(prediction_path)
    required = {'TransactionYear', 'price_yen', 'baseline_prediction_yen',
                'challenger_prediction_yen'}
    if not required.issubset(frame.columns) or len(frame) < min_rows:
        raise ValueError('Rolling-origin confirmation file is incomplete or too small')
    years = pd.to_numeric(frame.TransactionYear, errors='coerce').to_numpy(float)
    reserved_years = cv.get('years') or [cv.get('year')]
    reserved_years = [int(year) for year in reserved_years if year is not None]
    if (not reserved_years or not np.all(np.isfinite(years))
            or not set(years.astype(int)).issubset(set(reserved_years))
            or not set(reserved_years).issubset(set(years.astype(int)))):
        raise ValueError('Confirmation rows do not match all reserved years')
    y = pd.to_numeric(frame.price_yen, errors='coerce').to_numpy(float)
    baseline = pd.to_numeric(frame.baseline_prediction_yen, errors='coerce').to_numpy(float)
    challenger = pd.to_numeric(frame.challenger_prediction_yen, errors='coerce').to_numpy(float)
    if any(not np.all(np.isfinite(a)) for a in (y, baseline, challenger)) or any(np.any(a <= 0) for a in (y, baseline, challenger)):
        raise ValueError('Confirmation contains invalid prices or predictions')
    before, after = score(y, baseline), score(y, challenger)
    ci = paired_median_ape_delta_ci(y, baseline, challenger)
    approved = (ci[1] < 0
                and after['mae_yen'] <= before['mae_yen'] * 1.01
                and after['within20_pct'] >= before['within20_pct'] - 1)
    return {
        'status': 'provisional_cv_approved' if approved else 'rejected',
        'validation_type': 'rolling_origin_cv_existing_transactions',
        'independent_holdout': False,
        'confirmation_years': reserved_years,
        'rows': int(len(frame)),
        'data_signature': signature,
        'baseline': before,
        'challenger': after,
        'paired_median_ape_delta_95ci_percentage_points': ci,
        'gate': {
            'ci_upper_below_zero': bool(ci[1] < 0),
            'mae_no_more_than_1pct_worse': bool(after['mae_yen'] <= before['mae_yen'] * 1.01),
            'within20_no_more_than_1pp_worse': bool(after['within20_pct'] >= before['within20_pct'] - 1),
        },
    }


def promote(candidate_path, report_path, prediction_path, site, min_rows=1000):
    report = json.loads(report_path.read_text(encoding='utf-8'))
    evaluation = evaluate(report, prediction_path, min_rows=min_rows)
    candidates = site / 'models' / 'candidates'
    candidates.mkdir(parents=True, exist_ok=True)
    if evaluation['status'] != 'provisional_cv_approved':
        return evaluation
    candidate = joblib.load(candidate_path)
    if (not isinstance(candidate, dict) or candidate.get('model_key') != 'used_house'
            or candidate.get('target_unit') != 'yen' or not hasattr(candidate.get('pipeline'), 'predict')):
        raise ValueError('Unsupported challenger model bundle')
    candidate['candidate_only'] = False
    candidate['approval_status'] = 'approved'
    candidate['approval_type'] = 'provisional_rolling_origin_cv'
    candidate['validation_report'] = report
    target = candidates / 'approved_used_house.joblib'
    temp = target.with_suffix('.tmp')
    try:
        joblib.dump(candidate, temp)
        loaded = joblib.load(temp)
        sample_columns = loaded.get('feature_columns') or []
        if not sample_columns:
            raise ValueError('Missing model feature columns')
        sample = pd.DataFrame([{column: np.nan for column in sample_columns}])
        prediction = np.asarray(loaded['pipeline'].predict(sample), dtype=float)
        if prediction.shape != (1,) or not np.all(np.isfinite(prediction)) or prediction[0] <= 0:
            raise ValueError('Promoted model reload check failed')
        digest = hashlib.sha256(temp.read_bytes()).hexdigest()
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)
    manifest = {
        'approved': True,
        'filename': target.name,
        'sha256': digest,
        'model_key': 'used_house',
        'validation_type': evaluation['validation_type'],
        'independent_holdout': False,
        'evaluation': evaluation,
    }
    manifest_path = candidates / 'approved_used_house.json'
    temp_manifest = manifest_path.with_suffix('.tmp')
    temp_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp_manifest, manifest_path)
    evaluation['approved_model_sha256'] = digest
    return evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--site', type=Path, default=Path('.'))
    parser.add_argument('--min-rows', type=int, default=1000)
    args = parser.parse_args()
    result = promote(args.candidate, args.report, args.predictions, args.site, args.min_rows)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as output:
            output.write(f"promoted={str(result['status'] == 'provisional_cv_approved').lower()}\n")


if __name__ == '__main__':
    main()
