"""V7 continuous, bounded property-model research. Does not modify production models.
Requires research_v6.py and legacy_engine.py beside this file.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time, traceback
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from research_v6 import prepare, make_model, FEATURES, REGIONS, V6RegionalPipeline
from legacy_engine import discover, load_history, load_external

# Each experiment is an explicit hypothesis, not arbitrary repeated retraining.
EXPERIMENTS = [
    ('national_63', 63, 300, 0.0),
    ('national_127', 127, 450, 0.0),
    ('national_255', 255, 450, 0.0),
    ('regional_25', 127, 450, 0.25),
    ('regional_50', 127, 450, 0.50),
    ('regional_75', 127, 450, 0.75),
    ('regional_255_50', 255, 450, 0.50),
]

def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    os.replace(tmp, path)

def signature(paths):
    h = hashlib.sha256()
    for p in paths:
        if p and p.exists():
            s = p.stat()
            h.update(f'{p.resolve()}|{s.st_size}|{s.st_mtime_ns}'.encode())
    return h.hexdigest()

def stats(df, pred):
    y = df.price_man.to_numpy(dtype=float)
    err = np.abs(pred-y)
    ape = err / np.maximum(y, 1)
    return {'n': len(y), 'mae_man_yen': round(float(err.mean()), 2),
            'median_ape_pct': round(float(np.median(ape)*100), 2),
            'bias_pct': round(float(np.median((pred-y)/y)*100), 2)}

def diagnostics(df, pred):
    result = {}
    for field in ['Prefecture', 'TransactionYear']:
        group = {}
        for key, sub in df.groupby(field):
            if len(sub) < 100: continue
            idx = df.index.get_indexer(sub.index)
            group[str(key)] = stats(sub, pred[idx])
        result[field] = dict(sorted(group.items(), key=lambda kv: kv[1]['median_ape_pct'], reverse=True)[:12])
    for field in ['BuildingAge', 'StationMinutes', 'TotalFloorArea']:
        result[field+'_missing_pct'] = round(float(df[field].isna().mean()*100), 1)
    return result

def fit(df, leaves, iters):
    m = make_model(leaves, iters)
    m.fit(df[FEATURES], np.log(df.price_man.to_numpy()))
    return m

def predict(model, df):
    return np.clip(np.exp(np.clip(model.predict(df[FEATURES]), -10, 15)), 50, 100000)

def fit_regions(df, iters=280):
    out = {}
    for region in REGIONS:
        subset = df[df.Prefecture.eq(region)]
        if len(subset) >= 4000:
            out[region] = fit(subset, 63, iters)
    return out

def blended(global_model, locals_, df, weight):
    p = predict(global_model, df)
    if weight:
        for region, model in locals_.items():
            mask = df.Prefecture.eq(region).to_numpy()
            if mask.any():
                p[mask] = (1-weight)*p[mask] + weight*predict(model, df.loc[mask])
    return p

# Unlike the V6 class, this class can be imported by name for joblib loading.
class V7Pipeline:
    def __init__(self, global_model, regional_models, regional_weight):
        self.global_model = global_model
        self.regional_models = regional_models
        self.regional_weight = regional_weight
    def predict(self, frame):
        df = frame.copy()
        for c in FEATURES:
            if c not in df:
                df[c] = np.nan if c not in ['Prefecture','MunicipalityCode','DistrictName','Structure','CityPlanning','Use','PriceCategory'] else '不明'
        for c in ['Prefecture','MunicipalityCode','DistrictName','Structure','CityPlanning','Use','PriceCategory']:
            df[c] = df[c].fillna('不明').astype(str)
        for c in set(FEATURES)-{'Prefecture','MunicipalityCode','DistrictName','Structure','CityPlanning','Use','PriceCategory'}:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return blended(self.global_model, self.regional_models, df, self.regional_weight)*10000

# A stable import path is essential: models saved by a script otherwise pickle
# V7Pipeline as __main__.V7Pipeline and fail when loaded from Streamlit.
if __name__ == '__main__':
    sys.modules.setdefault('research_v7', sys.modules[__name__])
V7Pipeline.__module__ = 'research_v7'


def compare_production_on_external(site, external, candidate_bundle):
    """Diagnostic on identical 2026 records. NOT an unbiased promotion gate.

    The production model's training dates and feature provenance are unknown;
    some site-only features (e.g. land-price context) are unavailable here.
    """
    result = {'status': 'not_comparable', 'production_model_unchanged': True,
              'warning': ('同じ2026年の物件で参考比較するだけです。既存モデルの学習期間が不明で、'
                          'サイト側の地価等の特徴量も不足するため、優劣の確定や自動採用はしません。')}
    if not external or not external.is_file():
        result['reason'] = '2026年の検証データがありません'
        return result
    model_path = site / 'models' / 'used_house_model.joblib'
    if not model_path.is_file():
        result['reason'] = '本番の中古戸建てモデルがありません'
        return result
    try:
        df = prepare(load_external(external)).reset_index(drop=True)
        if df.empty:
            result['reason'] = '2026年の対象物件がありません'
            return result
        y = df.price_man.to_numpy(dtype=float)
        challenger_yen = np.asarray(candidate_bundle['pipeline'].predict(df), dtype=float)
        incumbent = joblib.load(model_path)
        if not isinstance(incumbent, dict):
            raise TypeError('本番モデルの保存形式が想定外です')
        # The site prediction path supports either a single pipeline or a
        # weighted geometric ensemble. Mirror that calculation, without
        # fabricating unavailable land-price features.
        components = incumbent.get('ensemble_components') or []
        if components:
            logs, weights = [], []
            for comp in components:
                pipe = comp.get('pipeline')
                w = float(comp.get('weight', 0) or 0)
                if pipe is None or w <= 0:
                    continue
                cols = comp.get('feature_columns') or []
                frame = pd.DataFrame({c: df[c] if c in df.columns else np.nan for c in cols})
                pred = np.asarray(pipe.predict(frame), dtype=float)
                logs.append(w * np.log(np.maximum(pred, 1)))
                weights.append(w)
            if not weights:
                raise ValueError('本番モデルの有効な構成要素がありません')
            incumbent_yen = np.exp(sum(logs) / sum(weights))
        else:
            pipe = incumbent['pipeline']
            cols = incumbent.get('feature_columns') or list(df.columns)
            frame = pd.DataFrame({c: df[c] if c in df.columns else np.nan for c in cols})
            incumbent_yen = np.asarray(pipe.predict(frame), dtype=float)
        # Production post-calibration is deliberately not approximated: it may
        # depend on prefecture or prediction bands and needs the site path.
        valid = (np.isfinite(y) & (y > 0) & np.isfinite(incumbent_yen)
                 & (incumbent_yen > 0) & np.isfinite(challenger_yen)
                 & (challenger_yen > 0))
        if valid.sum() < 100:
            raise ValueError('両モデルで評価可能な物件が100件未満です')
        subset = df.loc[valid].reset_index(drop=True)
        result.update({
            'status': 'diagnostic_only',
            'same_records': int(valid.sum()),
            'production_uncalibrated': stats(subset, incumbent_yen[valid] / 10000),
            'v7_candidate': stats(subset, challenger_yen[valid] / 10000),
            'reason': '本番モデルの補正・学習期間・特徴量の整合性が未検証',
        })
    except Exception as exc:
        result['reason'] = f'{type(exc).__name__}: {exc}'
    return result


def study(history, external, site, quick=False):
    raw = load_history(history, 300 if quick else None)
    data = prepare(raw)
    train = data[data.TransactionYear <= 2023].reset_index(drop=True)
    select = data[data.TransactionYear == 2024].reset_index(drop=True)
    diagnostic = data[data.TransactionYear == 2025].reset_index(drop=True)
    if min(len(train), len(select)) < 100: raise RuntimeError('2023年以前または2024年のデータが不足')
    print(f'学習 {len(train):,} / 選定 {len(select):,}', flush=True)
    experiments = EXPERIMENTS[:2] if quick else EXPERIMENTS
    model_cache = {}
    regional = None
    results = []
    for name, leaves, iters, weight in experiments:
        key = (leaves, iters)
        if key not in model_cache: model_cache[key] = fit(train, leaves, iters)
        if weight and regional is None: regional = fit_regions(train)
        pred = blended(model_cache[key], regional or {}, select, weight)
        m = stats(select, pred)
        results.append({'name': name, 'leaves': leaves, 'iterations': iters, 'regional_weight': weight, 'selection_2024': m})
        print(name, m, flush=True)
    # Selection on 2024 only. 2025/2026 diagnostics never used for selection.
    winner = min(results, key=lambda x: (x['selection_2024']['median_ape_pct'], x['selection_2024']['mae_man_yen']))
    w = winner['regional_weight']; leaves = winner['leaves']; iters = winner['iterations']
    print('選定:', winner['name'], flush=True)
    prior = data[data.TransactionYear <= 2024].reset_index(drop=True)
    diag_model = fit(prior, leaves, iters)
    diag_regions = fit_regions(prior) if w else {}
    report = {'version': 7, 'selected_on': '2024', 'winner': winner, 'experiments': results,
              'warning': '既存サイトモデルとの同一検証比較は未実施。自動本番切替は行わない。2025/2026は過去研究で閲覧済み。',
              'scope': '宅地(土地と建物)のみ', 'updated_at': datetime.now().isoformat()}
    if len(diagnostic):
        p = blended(diag_model, diag_regions, diagnostic, w)
        report['diagnostic_2025'] = stats(diagnostic, p)
        report['error_analysis_2025'] = diagnostics(diagnostic, p)
    if external and external.is_file():
        d26 = prepare(load_external(external, 300 if quick else None)).reset_index(drop=True)
        if len(d26):
            p26 = blended(diag_model, diag_regions, d26, w)
            report['diagnostic_2026'] = stats(d26, p26)
            report['error_analysis_2026'] = diagnostics(d26, p26)
    if quick: return report, None
    final = data[data.TransactionYear <= 2025].reset_index(drop=True)
    final_global = fit(final, leaves, iters)
    final_regions = fit_regions(final) if w else {}
    bundle = {'pipeline': V7Pipeline(final_global, final_regions, w),
              'feature_columns': FEATURES, 'model_key': 'used_house', 'target_unit': 'yen',
              'candidate_only': True, 'display_name': 'V7 自動研究・戸建て候補',
              'validation_report': report}
    return report, bundle

def run_once(args):
    site = args.site.resolve()
    if not (site/'app.py').is_file() or not (site/'models').is_dir():
        raise FileNotFoundError('サイトフォルダーが不正: '+str(site))
    root = Path(__file__).resolve().parent
    h,e = discover(root)
    h = args.history or h; e = args.external or e
    if not h or not h.is_file(): raise FileNotFoundError('All_20214_20254.zip が見つかりません')
    dest = site/'models'/'candidates'
    dest.mkdir(parents=True, exist_ok=True)
    state_path = dest/'v7_state.json'
    state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
    sig = signature([h,e])
    if not args.force and not args.quick and state.get('completed_signature') == sig:
        print('データに変更なし。前回の研究結果を維持します。', flush=True)
        return
    report, bundle = study(h,e,site,args.quick)
    report['data_signature'] = sig
    if bundle is not None:
        report['production_comparison_2026'] = compare_production_on_external(site, e, bundle)
        bundle['validation_report'] = report
    if args.quick:
        atomic_json(dest/'v7_quick_report.json', report)
        print('動作確認完了。候補モデルは保存していません。', flush=True)
        return
    # Conservative champion/challenger gate. If validation populations changed,
    # the two scores are not comparable, so retain the prior candidate for review.
    candidate = dest/'v7_used_house_candidate.joblib'
    previous_report = dest/'v7_report.json'
    if candidate.exists() and previous_report.exists():
        old = json.loads(previous_report.read_text(encoding='utf-8'))
        old_sig = old.get('data_signature')
        old_score = old.get('winner', {}).get('selection_2024', {}).get('median_ape_pct')
        new_score = report['winner']['selection_2024']['median_ape_pct']
        if old_sig != sig:
            # No claim of superiority if historical data may have changed.
            print('データ変更を検出。旧候補との公平な比較が未確認のため旧候補を維持します。', flush=True)
            atomic_json(dest/'v7_challenger_report.json', report)
            atomic_json(state_path, {'completed_signature': sig, 'updated_at': datetime.now().isoformat(),
                                     'candidate': str(candidate), 'status': 'needs_comparable_validation'})
            return
        if not args.force and old_score is not None and new_score >= old_score:
            print('旧候補の精度を超えなかったため旧候補を維持します。', flush=True)
            atomic_json(state_path, {'completed_signature': sig, 'updated_at': datetime.now().isoformat(),
                                     'candidate': str(candidate), 'status': 'incumbent_retained'})
            return
    # Retain only one candidate: replace atomically AFTER full successful training.
    candidate = dest/'v7_used_house_candidate.joblib'
    temp = dest/'v7_used_house_candidate.tmp.joblib'
    try:
        joblib.dump(bundle, temp)
        # Verify before replacing the previous candidate.
        loaded = joblib.load(temp)
        if not hasattr(loaded['pipeline'], 'predict'): raise RuntimeError('モデル読込テスト失敗')
        os.replace(temp, candidate)
    finally:
        if temp.exists(): temp.unlink()
    atomic_json(dest/'v7_report.json', report)
    atomic_json(state_path, {'completed_signature': sig, 'updated_at': datetime.now().isoformat(),
                             'candidate': str(candidate), 'selected': report['winner']['name']})
    # Only V7's own obsolete candidate files are cleaned up. Never delete older V6 or production files automatically.
    for p in dest.glob('v7_used_house_*.joblib'):
        if p != candidate: p.unlink()
    print('候補更新:', candidate, flush=True)
    print('既存の本番モデルは変更していません。', flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--site', type=Path, required=True)
    ap.add_argument('--history', type=Path)
    ap.add_argument('--external', type=Path)
    ap.add_argument('--interval-hours', type=float, default=6)
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--force', action='store_true')
    args=ap.parse_args()
    if args.interval_hours < 1: ap.error('--interval-hours は1以上にしてください')
    print('V7継続研究を開始。終了するには Ctrl+C', flush=True)
    while True:
        try:
            run_once(args)
        except KeyboardInterrupt: raise
        except Exception:
            traceback.print_exc()
            print('今回の処理は失敗。既存モデル・前回候補は維持。', flush=True)
        if args.once or args.quick: break
        print(f'次回確認まで {args.interval_hours:g} 時間待機（新データがなければ再学習しません）', flush=True)
        time.sleep(args.interval_hours*3600)

if __name__=='__main__':
    main()
