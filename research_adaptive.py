"""V9: sequential feature/hyperparameter research, no six-hour idle loop.
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
    sys.modules.setdefault('research_adaptive',sys.modules[__name__])
V8Pipeline.__module__='research_adaptive'

def write_json(path,obj):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    os.replace(tmp,path)


# Adaptive research: successive candidates inherit the current best configuration.
# This is deterministic hypothesis generation from prior results, not LLM reasoning.
import random

def diagnose(frame, predictions):
    """Find high-error segments to determine the next experiments."""
    actual = frame.price_man.to_numpy(dtype=float)
    ape = np.abs(predictions - actual) / np.maximum(actual, 1)
    diagnostics = []
    dimensions = ['Prefecture', 'BuildingAge', 'Area', 'StationMinutes', 'Structure']
    for col in dimensions:
        if col not in frame: continue
        series = frame[col]
        if col in ('BuildingAge', 'Area', 'StationMinutes'):
            try: labels = pd.qcut(pd.to_numeric(series, errors='coerce'), q=5, duplicates='drop').astype(str)
            except ValueError: continue
        else:
            labels = series.fillna('unknown').astype(str)
        tmp = pd.DataFrame({'segment': labels.to_numpy(), 'ape': ape})
        grouped = tmp.groupby('segment', observed=True).ape.agg(['size', 'median'])
        grouped = grouped.loc[grouped['size'] >= max(100, len(frame)//500)]
        for label, row in grouped.sort_values('median', ascending=False).head(3).iterrows():
            diagnostics.append({'feature': col, 'segment': str(label),
                                'count': int(row['size']), 'median_ape_pct': round(float(row['median'])*100, 3)})
    return sorted(diagnostics, key=lambda d: d['median_ape_pct'], reverse=True)[:12]

FEATURE_GROUPS = {
    'station': ['StationMinutes'], 'district': ['DistrictName'],
    'city': ['MunicipalityCode'], 'age': ['BuildingAge'],
    'structure': ['Structure'], 'planning': ['CityPlanning'],
    'floor': ['TotalFloorArea', 'LogTotalFloorArea', 'LandBuildingRatio'],
    'land': ['Area', 'LogArea', 'LandBuildingRatio'],
}
FEATURE_HINTS = {'StationMinutes': 'station', 'DistrictName': 'district',
                 'MunicipalityCode': 'city', 'BuildingAge': 'age',
                 'Structure': 'structure', 'Area': 'land'}

def propose(champion, diagnostics, experiments, round_no, seed=42):
    """Feedback-directed, reproducible hypothesis generator, not a fixed experiment list.

    Mutations use previous successes/failures and the worst error segments.
    """
    rng = random.Random(seed + round_no)
    _, excluded, leaves, trees, child, reg, lr = champion
    seen = {json.dumps(e.get('spec', [])[1:], sort_keys=True) for e in experiments if e.get('spec')}
    ideas = []
    priority = []
    for d in diagnostics:
        group = FEATURE_HINTS.get(d['feature'])
        if group and group not in priority: priority.append(group)
    priority += [g for g in FEATURE_GROUPS if g not in priority]
    # Exploit successful directions, while also trying an opposing direction.
    wins = [e for e in experiments[-30:] if e.get('new_champion')]
    if wins:
        successful = wins[-1].get('hypothesis', '')
        if 'regularization' in successful: priority.insert(0, 'planning')
    def add(reason, ex, lv, tr, ch, rg, rate):
        lv = int(max(15, min(511, lv)))
        tr = int(max(120, min(1400, tr)))
        ch = int(max(15, min(300, ch)))
        rg = round(float(max(.01, min(150, rg))), 5)
        rate = round(float(max(.008, min(.15, rate))), 5)
        spec = (f'round{round_no}_{len(ideas)+1}', sorted(set(ex)), lv, tr, ch, rg, rate)
        key = json.dumps(spec[1:], sort_keys=True)
        if key not in seen and all(json.dumps(x[1][1:],sort_keys=True)!=key for x in ideas):
            ideas.append((reason, spec))
    for group in priority[:3]:
        columns = FEATURE_GROUPS[group]
        if all(x in excluded for x in columns):
            ex = [x for x in excluded if x not in columns]
            add('restore feature group: '+group, ex, leaves, trees, child, reg, lr)
        else:
            add('remove feature group: '+group, list(excluded)+columns, leaves, trees, child, reg, lr)
    # Explore both more expressive and more conservative models.
    factors = [rng.uniform(1.15, 1.7), rng.uniform(.55, .9)]
    rng.shuffle(factors)
    for f in factors:
        add('model capacity mutation', excluded, leaves*f, trees*(1.08 if f>1 else .93), child/(f**.5), reg, lr)
    for f in [rng.uniform(1.35,2.5), rng.uniform(.35,.75)]:
        add('regularization mutation', excluded, leaves, trees, child, reg*f, lr)
    f = rng.choice([rng.uniform(.6,.9), rng.uniform(1.1,1.5)])
    add('learning-rate and trees mutation', excluded, leaves, trees/f, child, reg, lr*f)
    # A novel combined mutation based on a random subset of the champion's genes.
    for _ in range(5):
        add('combined exploratory mutation', excluded, leaves*rng.uniform(.7,1.4),
            trees*rng.uniform(.8,1.25), child*rng.uniform(.75,1.35),
            reg*rng.uniform(.7,1.5), lr*rng.uniform(.8,1.2))
        if len(ideas)>=8: break
    return ideas[:8]

def score(v):
    return (v['median_ape_pct'],v['mae_man_yen'])

def run(args):
    site=args.site.resolve(); root=Path(__file__).resolve().parent
    if not (site/'app.py').is_file(): raise FileNotFoundError('Site app.py not found')
    h,e=discover(root); h=args.history or h; e=args.external or e
    if not h or not h.is_file(): raise FileNotFoundError('Historical ZIP not found')
    out=site/'models'/'candidates'; out.mkdir(parents=True,exist_ok=True)
    report_path=out/'adaptive_research_report.json'
    signature=hashlib.sha256('|'.join(f'{p.resolve()}:{p.stat().st_size}:{p.stat().st_mtime_ns}' for p in (h,e) if p and p.is_file()).encode()).hexdigest()
    data=prepare(load_history(h,300 if args.quick else None))
    train=data.loc[data.TransactionYear<=2023].reset_index(drop=True)
    val=data.loc[data.TransactionYear==2024].reset_index(drop=True)
    diagnostic=data.loc[data.TransactionYear==2025].reset_index(drop=True)
    if len(train)<100 or len(val)<100: raise RuntimeError('Insufficient train/validation data')
    # The 2024 set is repeatedly optimized against, NOT an independent test.
    # No automatic production promotion. Existing V7/V8 and 3 production models remain intact.
    champion=('baseline',[],127,450,60,8.0,.045)
    # Seed the search from the V8 winner when its report is available.
    v8report=out/'v8_research_report.json'
    if v8report.is_file() and not args.restart:
        try:
            v8info=json.loads(v8report.read_text(encoding='utf-8'))
            winner_name=(v8info.get('winner') or {}).get('name')
            matching=[spec for spec in TESTS if spec[0]==winner_name]
            if matching: champion=matching[0]
        except (OSError,ValueError): pass
    prior=report_path
    completed=[]
    if prior.is_file() and not args.restart:
        try:
            old=json.loads(prior.read_text(encoding='utf-8'))
            if old.get('data_signature')==signature:
                completed=old.get('experiments',[])
                c=old.get('champion_spec')
                if c and len(c)==7: champion=tuple(c)
            else: print('Data changed; beginning a fresh search',flush=True)
        except (ValueError,OSError): pass
    report={'version':'adaptive-1','status':'running','data_signature':signature,'started_at':datetime.now().isoformat(),
        'train_n':len(train),'validation_2024_n':len(val),'diagnostic_2025_n':len(diagnostic),
        'warning':'2024 is reused for model selection; 2025/2026 have been inspected in earlier research. Neither is a pristine holdout. No automatic production deployment.',
        'experiments':completed,'champion_spec':list(champion)}
    # Recheck champion score on same data, independent of persisted rounded metrics.
    champion_model=fit(train,champion)
    current_predictions=pred(champion_model,val,champion[1])
    best=metric(val,current_predictions)
    print(f'Adaptive research: train={len(train):,} validation={len(val):,}; baseline median APE={best["median_ape_pct"]}%',flush=True)
    round_no=1+max([int(x.get('round',0)) for x in completed] or [0])
    rounds=1 if args.quick else args.rounds
    for r in range(round_no,round_no+rounds):
        improvement=False
        diagnostics=diagnose(val,current_predictions)
        report['diagnostics']=diagnostics
        hypotheses=propose(champion,diagnostics,report['experiments'],r)
        for hypothesis, spec in hypotheses:
            entry={'round':r,'name':spec[0],'hypothesis':hypothesis,'spec':list(spec),'started_at':datetime.now().isoformat()}
            try:
                model=fit(train,spec)
                candidate_predictions=pred(model,val,spec[1])
                m=metric(val,candidate_predictions)
                entry['validation_2024']=m;entry['status']='ok'
                if score(m)<score(best):
                    champion=spec;champion_model=model;best=m;current_predictions=candidate_predictions;improvement=True
                    entry['new_champion']=True
            except Exception as ex:
                entry.update(status='failed',error=f'{type(ex).__name__}: {ex}')
            report['experiments'].append(entry)
            report['champion_spec']=list(champion)
            report['best_2024']=best
            report['updated_at']=datetime.now().isoformat()
            write_json(report_path,report)
            print(f'[{r}] {spec[0]}: {entry.get("validation_2024",{}).get("median_ape_pct","ERROR")}% best={best["median_ape_pct"]}%',flush=True)
        print(f'Round {r} finished; improved={improvement}; worst segment={diagnostics[0] if diagnostics else None}',flush=True)
    # Diagnostic only, never used for selecting the champion.
    if len(diagnostic): report['diagnostic_2025']=metric(diagnostic,pred(champion_model,diagnostic,champion[1]))
    report['status']='completed';report['completed_at']=datetime.now().isoformat()
    report['candidate_only']=True
    if not args.quick:
        full=data.loc[data.TransactionYear<=2025].reset_index(drop=True)
        final=fit(full,champion)
        bundle={'pipeline':V8Pipeline(final,champion[1]),'feature_columns':FEATURES,
                'model_key':'used_house','target_unit':'yen','candidate_only':True,
                'display_name':'Feedback-driven adaptive research candidate (not approved)', 'validation_report':report}
        target=out/'adaptive_used_house_candidate.joblib';tmp=out/'adaptive_used_house_candidate.tmp.joblib'
        try:
            joblib.dump(bundle,tmp)
            if not hasattr(joblib.load(tmp)['pipeline'],'predict'): raise RuntimeError('Reload check failed')
            os.replace(tmp,target)
        finally:
            if tmp.exists():tmp.unlink()
        print('Saved candidate (NOT production):',target,flush=True)
    write_json(report_path,report)
    print('Research complete:',report_path,flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--site',required=True,type=Path)
    ap.add_argument('--history',type=Path)
    ap.add_argument('--external',type=Path)
    ap.add_argument('--rounds',type=int,default=4)
    ap.add_argument('--quick',action='store_true')
    ap.add_argument('--restart',action='store_true')
    args=ap.parse_args()
    if args.rounds<1 or args.rounds>100: ap.error('--rounds must be 1..100')
    run(args)

if __name__=='__main__': main()
