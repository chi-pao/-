"""V8: sequential feature/hyperparameter research, no six-hour idle loop.
Keeps production models and V7 candidate untouched. 2024 is development validation;
2025/2026 were previously inspected, so neither is an untouched test set.
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback, hashlib
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from research_v6 import prepare, FEATURES, CATS, NUMS, REGIONS
from legacy_engine import discover, load_history, load_external
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMRegressor

# Hypotheses are chosen in advance; no unbounded search on the validation set.
TESTS = [
    ('baseline', [], 127, 450, 60, 8.0, 0.045),
    ('without_station', ['StationMinutes'], 127, 450, 60, 8.0, 0.045),
    ('without_district', ['DistrictName'], 127, 450, 60, 8.0, 0.045),
    ('without_city', ['MunicipalityCode'], 127, 450, 60, 8.0, 0.045),
    ('without_age', ['BuildingAge'], 127, 450, 60, 8.0, 0.045),
    ('without_structure', ['Structure'], 127, 450, 60, 8.0, 0.045),
    ('without_planning', ['CityPlanning'], 127, 450, 60, 8.0, 0.045),
    ('without_floor', ['TotalFloorArea','LogTotalFloorArea','LandBuildingRatio'], 127, 450, 60, 8.0, 0.045),
    ('without_land', ['Area','LogArea','LandBuildingRatio'], 127, 450, 60, 8.0, 0.045),
    ('regularized', [], 63, 550, 110, 16.0, 0.035),
    ('fine_leaves', [], 255, 650, 80, 14.0, 0.035),
    ('strong_regularization', [], 127, 650, 140, 30.0, 0.035),
]

def metric(frame, p):
    y = frame.price_man.to_numpy(dtype=float)
    e = np.abs(y-p)
    return {'n':int(len(y)), 'mae_man_yen':round(float(np.mean(e)),2),
            'median_ape_pct':round(float(np.median(e/np.maximum(y,1))*100),2),
            'bias_pct':round(float(np.median((p-y)/np.maximum(y,1))*100),2)}

def input_frame(frame, excluded):
    x = frame.copy()
    for c in excluded:
        x[c] = '不明' if c in CATS else np.nan
    return x

def model_factory(spec):
    _, excluded, leaves, iters, min_child, regularization, lr = spec
    cats = [x for x in CATS if x not in excluded]
    nums = [x for x in NUMS if x not in excluded]
    pre = ColumnTransformer([
        ('num', Pipeline([('fill', SimpleImputer(strategy='constant',fill_value=-1))]), nums),
        ('cat', Pipeline([('fill',SimpleImputer(strategy='constant',fill_value='不明')),
                          ('encode',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1))]), cats)
    ])
    return Pipeline([('pre',pre),('lgb',LGBMRegressor(n_estimators=iters,num_leaves=leaves,
        min_child_samples=min_child,reg_lambda=regularization,learning_rate=lr,
        verbosity=-1,n_jobs=4,random_state=42))])

def fit(frame, spec):
    m = model_factory(spec)
    m.fit(input_frame(frame,spec[1])[FEATURES], np.log(frame.price_man.to_numpy(dtype=float)))
    return m

def pred(model, frame, excluded):
    v = model.predict(input_frame(frame,excluded)[FEATURES])
    return np.clip(np.exp(np.clip(v,-10,15)),50,100000)

class V8Pipeline:
    """Site-compatible predictor returning yen. Loaded via importable research_v8 module."""
    def __init__(self, model, excluded, regional=None, regional_weight=0.0):
        self.model=model; self.excluded=excluded
        self.regional=regional or {}; self.regional_weight=regional_weight
    def predict(self, frame):
        x=frame.copy()
        for c in FEATURES:
            if c not in x: x[c] = '不明' if c in CATS else np.nan
        for c in CATS: x[c]=x[c].fillna('不明').astype(str)
        for c in NUMS: x[c]=pd.to_numeric(x[c],errors='coerce')
        p=pred(self.model,x,self.excluded)
        for region,m in self.regional.items():
            mask=x.Prefecture.eq(region).to_numpy()
            if mask.any():
                p[mask]=(1-self.regional_weight)*p[mask]+self.regional_weight*pred(m,x.loc[mask],self.excluded)
        return p*10000

if __name__=='__main__':
    sys.modules.setdefault('research_v8',sys.modules[__name__])
V8Pipeline.__module__='research_v8'

def write_json(path,obj):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    os.replace(tmp,path)

def run(args):
    site=args.site.resolve(); root=Path(__file__).resolve().parent
    if not (site/'app.py').is_file(): raise FileNotFoundError(f'サイトがありません: {site}')
    h,e=discover(root); h=args.history or h; e=args.external or e
    if not h or not h.is_file(): raise FileNotFoundError('過去データZIPがありません')
    out=site/'models'/'candidates';out.mkdir(parents=True,exist_ok=True)
    report_path=out/'v8_research_report.json'
    signature=hashlib.sha256('|'.join(f'{p.resolve()}:{p.stat().st_size}:{p.stat().st_mtime_ns}' for p in (h,e) if p and p.is_file()).encode()).hexdigest()
    if not args.quick and not args.force and report_path.exists():
        prior=json.loads(report_path.read_text(encoding='utf-8'))
        if prior.get('status')=='completed' and prior.get('data_signature')==signature:
            print('入力データに変更なし。前回の実験結果を保持（無駄な再学習なし）。',flush=True)
            return prior
    data=prepare(load_history(h,300 if args.quick else None))
    train=data.loc[data.TransactionYear<=2023].reset_index(drop=True)
    val=data.loc[data.TransactionYear==2024].reset_index(drop=True)
    diag=data.loc[data.TransactionYear==2025].reset_index(drop=True)
    if len(train)<100 or len(val)<100:raise RuntimeError('2023年以前と2024年のデータが不足')
    external=prepare(load_external(e,300 if args.quick else None)) if e and e.is_file() else pd.DataFrame()
    print(f'V8開始: 学習{len(train):,}件 / 2024年検証{len(val):,}件',flush=True)
    report={'version':8,'status':'running','updated_at':datetime.now().isoformat(),
            'train_n':len(train),'development_validation_2024_n':len(val),'data_signature':signature,
            'note':'2024年は繰り返し利用する開発検証データ。2025/2026も過去研究で閲覧済み。未知データでの本番精度は未検証。',
            'experiments':[]}
    winner=None; winner_model=None
    tests=TESTS[:3] if args.quick else TESTS
    for i,spec in enumerate(tests,1):
        name,excluded,*_=spec
        try:
            model=fit(train,spec)
            v=metric(val,pred(model,val,excluded))
            entry={'name':name,'excluded':excluded,'2024':v,'status':'ok'}
            # These diagnostics must NOT be used to select the experiment.
            if len(diag):entry['2025_diagnostic']=metric(diag,pred(model,diag,excluded))
            if len(external):entry['2026_diagnostic']=metric(external,pred(model,external,excluded))
            if winner is None or (v['median_ape_pct'],v['mae_man_yen']) < (winner['2024']['median_ape_pct'],winner['2024']['mae_man_yen']):
                winner=entry; winner_model=model; winner_spec=spec
            print(f'[{i}/{len(tests)}] {name}: 2024誤差中央値 {v["median_ape_pct"]:.2f}% / MAE {v["mae_man_yen"]:.2f}万円',flush=True)
        except Exception as exc:
            entry={'name':name,'excluded':excluded,'status':'failed','reason':f'{type(exc).__name__}: {exc}'}
            print(f'[{i}/{len(tests)}] {name}: 失敗 {exc}',flush=True)
        report['experiments'].append(entry)
        report['best_so_far']=winner
        write_json(report_path,report)
    if winner is None:raise RuntimeError('成功した実験がありません')
    # Compare V7 only on its documented 2024 development score, not an independent proof.
    v7_path=out/'v7_report.json'
    v7=json.loads(v7_path.read_text(encoding='utf-8')) if v7_path.is_file() else {}
    old=v7.get('winner',{}).get('selection_2024',{}).get('median_ape_pct')
    # No automatic replacement of the V7 site candidate: comparison is exploratory.
    report['winner']=winner
    report['v7_2024_reference_median_ape_pct']=old
    report['better_than_v7_development_score']=bool(old is not None and winner['2024']['median_ape_pct']<float(old))
    report['status']='completed'
    report['completed_at']=datetime.now().isoformat()
    if not args.quick:
        # Store challenger only; the site keeps serving V7 until manually approved.
        full=data.loc[data.TransactionYear<=2025].reset_index(drop=True)
        final=fit(full,winner_spec)
        bundle={'pipeline':V8Pipeline(final,winner_spec[1]),'feature_columns':FEATURES,
                'model_key':'used_house','target_unit':'yen','candidate_only':True,
                'display_name':'V8研究候補（未承認）','validation_report':report}
        target=out/'v8_used_house_candidate.joblib';tmp=out/'v8_used_house_candidate.tmp.joblib'
        try:
            joblib.dump(bundle,tmp)
            if not hasattr(joblib.load(tmp)['pipeline'],'predict'):raise RuntimeError('読み込みテスト失敗')
            os.replace(tmp,target)
        finally:
            if tmp.exists():tmp.unlink()
        print('V8候補保存（サイトへの自動切替なし）:',target,flush=True)
    write_json(report_path,report)
    print('実験完了。研究レポート:',report_path,flush=True)
    return report

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--site',required=True,type=Path)
    ap.add_argument('--history',type=Path)
    ap.add_argument('--external',type=Path)
    ap.add_argument('--quick',action='store_true')
    ap.add_argument('--force',action='store_true')
    args=ap.parse_args()
    run(args)

if __name__=='__main__':main()
