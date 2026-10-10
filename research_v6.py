"""V6: V4-based regional LightGBM experiments and safe candidate export.
No changes to production models or app. Keep this module available when loading V6 joblib.
"""
from __future__ import annotations
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from legacy_engine import discover, load_history, load_external, metrics

CATS = ['Prefecture', 'MunicipalityCode', 'DistrictName', 'Structure', 'CityPlanning', 'Use', 'PriceCategory']
NUMS = ['Area', 'TotalFloorArea', 'BuildingAge', 'TransactionYear', 'Quarter',
        'StationMinutes', 'LogArea', 'LogTotalFloorArea', 'LandBuildingRatio']
FEATURES = NUMS + CATS
REGIONS = ['東京都', '大阪府', '神奈川県', '愛知県', '埼玉県', '千葉県', '兵庫県', '北海道', '福岡県']


def prepare(raw):
    """Convert V4 historical fields to the feature schema expected by app.py."""
    x = pd.DataFrame(index=raw.index)
    for name in CATS:
        x[name] = raw[name] if name in raw else '不明'
        x[name] = x[name].fillna('不明').astype(str).replace({'': '不明', 'nan': '不明'})
    for dest, src in [('Area', 'Area'), ('TotalFloorArea', 'TotalFloorArea'),
                      ('BuildingAge', 'Age'), ('TransactionYear', 'Year'),
                      ('Quarter', 'Quarter'), ('StationMinutes', 'StationMinutes')]:
        x[dest] = pd.to_numeric(raw[src], errors='coerce') if src in raw else np.nan
    a = x['Area']; f = x['TotalFloorArea']
    x['LogArea'] = np.log1p(a.where(a > 0))
    x['LogTotalFloorArea'] = np.log1p(f.where(f > 0))
    x['LandBuildingRatio'] = a / f.where(f > 0)
    x['price_man'] = pd.to_numeric(raw['price'], errors='coerce')
    x = x.loc[x['price_man'].between(100, 30000) & x['Area'].between(20, 2000)].copy()
    return x.reset_index(drop=True)


def make_model(leaves, iterations):
    from lightgbm import LGBMRegressor
    num_pipe = Pipeline([('fill', SimpleImputer(strategy='constant', fill_value=-1))])
    cat_pipe = Pipeline([
        ('fill', SimpleImputer(strategy='constant', fill_value='不明')),
        ('encode', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
    ])
    pre = ColumnTransformer([('n', num_pipe, NUMS), ('c', cat_pipe, CATS)])
    return Pipeline([('pre', pre), ('lgb', LGBMRegressor(
        n_estimators=iterations, num_leaves=leaves, learning_rate=0.045,
        min_child_samples=60, reg_lambda=8, verbosity=-1, n_jobs=4, random_state=42))])


def fit_one(df, leaves=255, iterations=450):
    model = make_model(leaves, iterations)
    model.fit(df[FEATURES], np.log(df['price_man'].to_numpy()))
    return model


def predict_man(model, df):
    return np.clip(np.exp(np.clip(model.predict(df[FEATURES]), -10, 15)), 50, 100000)


def fit_regional(df, iterations=350):
    models = {}
    for region in REGIONS:
        rows = df[df['Prefecture'].eq(region)]
        if len(rows) < 4000:
            continue
        print(f'  地域別学習: {region} {len(rows):,}件', flush=True)
        models[region] = fit_one(rows, leaves=63, iterations=iterations)
    return models


def predict_blend(global_model, regional_models, df):
    result = predict_man(global_model, df)
    for region, model in regional_models.items():
        mask = df['Prefecture'].eq(region).to_numpy()
        if mask.any():
            local = predict_man(model, df.loc[mask])
            result[mask] = 0.5 * result[mask] + 0.5 * local
    return result


class V6RegionalPipeline:
    """Joblib-compatible .predict(DataFrame), returning yen for app.py."""
    def __init__(self, global_model, regional_models, blend_enabled):
        self.global_model = global_model
        self.regional_models = regional_models
        self.blend_enabled = blend_enabled

    def predict(self, frame):
        df = frame.copy()
        for name in CATS:
            if name not in df:
                df[name] = '不明'
            df[name] = df[name].fillna('不明').astype(str)
        for name in NUMS:
            if name not in df:
                df[name] = np.nan
            df[name] = pd.to_numeric(df[name], errors='coerce')
        result = (predict_blend(self.global_model, self.regional_models, df)
                  if self.blend_enabled else predict_man(self.global_model, df))
        return result * 10000.0


def measure(df, pred):
    return metrics(df['price_man'].to_numpy(), pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--history', type=Path, help='Path to All_20214_20254.zip')
    ap.add_argument('--external', type=Path, help='Path to mlit_2026_Q1.csv')
    ap.add_argument('--site', type=Path, required=True, help='Current Streamlit site folder')
    ap.add_argument('--quick', action='store_true', help='Small test run, never exports a model')
    args = ap.parse_args()
    start = time.time()
    root = Path(__file__).resolve().parent
    history, external = discover(root)
    history = args.history or history
    external = args.external or external
    if not history or not history.is_file():
        raise FileNotFoundError('過去データZIPが見つかりません。--history で指定してください。')
    site = args.site.resolve()
    if not (site / 'app.py').is_file() or not (site / 'models').is_dir():
        raise FileNotFoundError(f'サイトフォルダーが見つかりません: {site}')
    print('V6研究開始（既存モデルは変更しません）', flush=True)
    print('過去データ:', history, flush=True)
    raw = load_history(history, 300 if args.quick else None)
    data = prepare(raw)
    train = data[data.TransactionYear <= 2023].copy()
    select = data[data.TransactionYear == 2024].copy()
    if min(len(train), len(select)) < 100:
        raise ValueError('2023年以前の学習データまたは2024年の選定データが不足しています')
    print(f'学習 {len(train):,}件 / 2024年選定 {len(select):,}件', flush=True)
    global_model = fit_one(train, iterations=100 if args.quick else 450)
    global_pred = predict_man(global_model, select)
    regions = fit_regional(train, iterations=100 if args.quick else 350)
    blend_pred = predict_blend(global_model, regions, select)
    g = measure(select, global_pred)
    b = measure(select, blend_pred)
    # Select using 2024 only; do not use later years to select a method.
    use_blend = (b['median_ape_pct'], b['mae_man_yen']) < (g['median_ape_pct'], g['mae_man_yen'])
    selected = 'regional_blend' if use_blend else 'national_only'
    print('2024 全国:', g, flush=True)
    print('2024 地域併用:', b, flush=True)
    print('採用候補:', selected, flush=True)
    diag_train = data[data.TransactionYear <= 2024].copy()
    diag_global = fit_one(diag_train, iterations=100 if args.quick else 450)
    diag_regions = fit_regional(diag_train, iterations=100 if args.quick else 350) if use_blend else {}
    report = {'status': 'completed', 'candidate_only': True, 'quick': args.quick,
              'scope': '宅地(土地と建物)のみ。中古マンション・賃貸には適用不可',
              'selection_2024': {'national': g, 'regional_blend': b, 'selected': selected},
              'regions': list(regions), 'training_rows_through_2023': len(train),
              'selection_rows_2024': len(select), 'limitations': [
                  '2025/2026は過去研究でも確認済みで完全な未使用検証データではない',
                  '既存サイトAIとの同一データ比較は未実施',
                  '物件状態・改装・階数・地価の学習値はデータにないため未使用',
                  '既存モデルは自動上書きしない']}
    d25 = data[data.TransactionYear == 2025].copy()
    if len(d25):
        p25 = predict_blend(diag_global, diag_regions, d25) if use_blend else predict_man(diag_global, d25)
        report['diagnostic_2025'] = measure(d25, p25)
        print('2025検証:', report['diagnostic_2025'], flush=True)
    if external and external.is_file():
        d26 = prepare(load_external(external, 300 if args.quick else None))
        if len(d26):
            p26 = predict_blend(diag_global, diag_regions, d26) if use_blend else predict_man(diag_global, d26)
            report['diagnostic_2026'] = measure(d26, p26)
            print('2026検証:', report['diagnostic_2026'], flush=True)
    dest = site / 'models' / 'candidates'
    dest.mkdir(parents=True, exist_ok=True)
    if not args.quick:
        # 2025 may be used for final fit only after diagnostics; no claimed pristine holdout.
        final_train = data[data.TransactionYear <= 2025].copy()
        print('候補モデルの最終学習:', len(final_train), '件', flush=True)
        final_global = fit_one(final_train)
        final_regions = fit_regional(final_train) if use_blend else {}
        bundle = {'pipeline': V6RegionalPipeline(final_global, final_regions, use_blend),
                  'feature_columns': FEATURES, 'model_key': 'used_house',
                  'display_name': 'V6 地域別戸建て研究候補', 'target': 'TradePrice',
                  'target_unit': 'yen', 'trained_at': datetime.now().isoformat(),
                  'training_rows': len(final_train), 'candidate_only': True,
                  'comparable_weight_max': 0.0, 'algorithm': 'V6 LightGBM regional blend',
                  'validation_report': report}
        out = dest / ('v6_used_house_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.joblib')
        joblib.dump(bundle, out)
        print('候補保存:', out, flush=True)
        print('重要: このモデルの読み込みには research_v6.py がサイト側にも必要です。', flush=True)
        report['candidate_path'] = str(out)
    report['elapsed_seconds'] = round(time.time() - start, 1)
    report_path = dest / ('v6_quick_report.json' if args.quick else 'v6_report.json')
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('結果保存:', report_path, flush=True)
    print('終了。既存モデルは変更していません。', flush=True)


if __name__ == '__main__':
    main()
