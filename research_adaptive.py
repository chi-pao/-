"""Adaptive price research with categorical, robust-loss and recency experiments.
Model selection uses older rolling folds; the newest one or two years are reserved
for a separate provisional confirmation and never guide the experiment search.
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback, hashlib
from datetime import datetime
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from research_v6 import prepare, FEATURES, CATS, NUMS, REGIONS
from legacy_engine import discover, load_history, load_external, normalize
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMRegressor

# Match the fields and property scope actually available in the Sites checker.
DATA_POLICY = 'site-residential-used-house-v2'
SITE_UNAVAILABLE = {'PriceCategory'}
EXPERIMENTAL_NUMS = ['Frontage', 'Breadth', 'CoverageRatio', 'FloorAreaRatio']
EXPERIMENTAL_CATS = ['LandShape', 'Direction', 'Classification', 'Renovation']
EXPERIMENTAL_FEATURES = EXPERIMENTAL_NUMS + EXPERIMENTAL_CATS
SITE_AVAILABLE = set(FEATURES) - SITE_UNAVAILABLE

def normalize_city_codes(values):
    text = values.fillna('').astype(str).str.normalize('NFKC').str.strip()
    codes = pd.to_numeric(text, errors='coerce')
    valid = codes.notna() & codes.ge(1000) & codes.le(99999) & codes.eq(codes.round())
    return codes.where(valid).map(lambda code: f'{int(code):05d}' if pd.notna(code) else '不明')

def prepare_site_training(raw):
    frame = prepare(raw)
    # Shared houses, shops, mixed-use and unknown-use records are not detached
    # residential examples. Age zero and unknown age cannot establish used status.
    frame = frame.loc[frame.Use.eq('住宅') & frame.BuildingAge.ge(1)].copy()
    frame['MunicipalityCode'] = normalize_city_codes(frame['MunicipalityCode'])
    return frame.reset_index(drop=True)

def load_training_rows(history_path, root, quick=False):
    historical = load_history(history_path, 300 if quick else None)
    if quick:
        return historical
    metadata_path = root / 'data' / 'mlit_latest_holdout.json'
    latest = None
    if metadata_path.is_file():
        try:
            meta = json.loads(metadata_path.read_text(encoding='utf-8'))
            latest = (int(meta['year']), int(meta['quarter']))
        except (OSError, ValueError, KeyError, TypeError):
            latest = None
    recent_parts = []
    for path in sorted((root / 'data').glob('mlit_*.csv')):
        recent = normalize(pd.read_csv(path, low_memory=False))
        if not recent.empty:
            recent_parts.append(recent)
    if latest is None and recent_parts:
        observed = pd.concat(recent_parts, ignore_index=True, sort=False)[['Year','Quarter']].dropna()
        if not observed.empty:
            period = observed.astype(int).drop_duplicates().sort_values(['Year','Quarter']).iloc[-1]
            latest = (int(period.Year), int(period.Quarter))
    parts = [historical]
    for recent in recent_parts:
        if latest:
            is_latest = recent.Year.eq(latest[0]) & recent.Quarter.eq(latest[1])
            recent = recent.loc[~is_latest]
        if not recent.empty:
            parts.append(recent)
    return pd.concat(parts, ignore_index=True, sort=False).drop_duplicates().reset_index(drop=True)

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
            'within20_pct':round(float(np.mean(e/np.maximum(y,1)<=.2)*100),2),
            'p90_ape_pct':round(float(np.percentile(e/np.maximum(y,1),90)*100),2),
            'bias_pct':round(float(np.median((p-y)/np.maximum(y,1))*100),2)}

def input_frame(frame, excluded):
    x = frame.copy()
    for c in set(excluded) | SITE_UNAVAILABLE:
        x[c] = '不明' if c in CATS or c in EXPERIMENTAL_CATS else np.nan
    for c in EXPERIMENTAL_CATS:
        if c not in x: x[c] = '不明'
        x[c] = x[c].fillna('不明').astype(str)
    for c in EXPERIMENTAL_NUMS:
        if c not in x: x[c] = np.nan
        x[c] = pd.to_numeric(x[c], errors='coerce')
    return x

def temporal_cv(data, spec, years=(2022, 2023, 2024), min_rows=1000):
    """Score with out-of-time predictions; each row is predicted by a model
    trained only on earlier transaction years."""
    held_frames = []
    held_predictions = []
    folds = []
    for year in years:
        train = data.loc[pd.to_numeric(data.TransactionYear, errors='coerce') < year].reset_index(drop=True)
        valid = data.loc[pd.to_numeric(data.TransactionYear, errors='coerce') == year].reset_index(drop=True)
        if len(train) < min_rows or len(valid) < min_rows:
            continue
        model = fit(train, spec)
        predictions = pred(model, valid, spec[1])
        folds.append({'year': int(year), **metric(valid, predictions)})
        held_frames.append(valid)
        held_predictions.append(predictions)
    if len(folds) < 2:
        raise ValueError('At least two sufficiently large chronological validation folds are required')
    pooled = pd.concat(held_frames, ignore_index=True)
    pooled_predictions = np.concatenate(held_predictions)
    return metric(pooled, pooled_predictions), folds


def temporal_confirmation(data, baseline_spec, candidate_spec, years, output_path, min_rows=1000):
    """Compare the selected challenger with its starting model on a later year.

    These confirmation years are excluded from every experiment that selects the
    challenger. Predictions are out-of-time; the compact row-level file is only
    a workflow-local input to the paired promotion gate and is never committed.
    """
    transaction_years = pd.to_numeric(data.TransactionYear, errors='coerce')
    parts = []
    fold_results = []
    for year in years:
        train = data.loc[transaction_years < year].reset_index(drop=True)
        valid = data.loc[transaction_years == year].reset_index(drop=True)
        if len(train) < min_rows or len(valid) < min_rows:
            raise ValueError(f'Insufficient rolling-origin confirmation rows for {year}')
        baseline_model = fit(train, baseline_spec)
        candidate_model = fit(train, candidate_spec)
        baseline_predictions = pred(baseline_model, valid, baseline_spec[1]) * 10000
        candidate_predictions = pred(candidate_model, valid, candidate_spec[1]) * 10000
        truth = valid.price_man.to_numpy(dtype=float) * 10000
        parts.append(pd.DataFrame({
            'TransactionYear': valid.TransactionYear.to_numpy(),
            'price_yen': truth,
            'baseline_prediction_yen': baseline_predictions,
            'challenger_prediction_yen': candidate_predictions,
        }))
        fold_results.append({
            'year': int(year), 'rows': int(len(valid)),
            'baseline': metric(valid, baseline_predictions / 10000),
            'challenger': metric(valid, candidate_predictions / 10000),
        })
    compact = pd.concat(parts, ignore_index=True)
    compact.to_csv(output_path, index=False)
    return {
        'method': 'rolling-origin-confirmation-v1', 'years': [int(year) for year in years],
        'validation_rows': int(len(compact)), 'folds': fold_results,
        'prediction_file': str(output_path),
    }



def model_options(spec):
    """Old seven-field experiments keep exactly their original semantics."""
    return dict(spec[7]) if len(spec) == 8 else {}

def training_weights(frame, half_life=0):
    """Recency is based only on training timestamps, never validation prices."""
    if not half_life:
        return np.ones(len(frame))
    dates = pd.to_numeric(frame.TransactionYear, errors='coerce') + (pd.to_numeric(frame.Quarter, errors='coerce') - 1) / 4
    if not dates.notna().all():
        raise ValueError('Recency weights require valid training dates')
    return np.exp2((dates.to_numpy() - dates.max()) / half_life)

def model_factory(spec):
    _, excluded, leaves, iters, min_child, regularization, lr = spec[:7]
    options = model_options(spec)
    excluded = set(excluded) | SITE_UNAVAILABLE
    requested = [x for x in options.get('extra_features', []) if x in EXPERIMENTAL_FEATURES]
    cats = [x for x in CATS if x not in excluded] + [x for x in requested if x in EXPERIMENTAL_CATS]
    nums = [x for x in NUMS if x not in excluded] + [x for x in requested if x in EXPERIMENTAL_NUMS]
    pre = ColumnTransformer([
        ('num', Pipeline([('fill', SimpleImputer(strategy='constant',fill_value=-1))]), nums),
        ('cat', Pipeline([('fill',SimpleImputer(strategy='constant',fill_value='不明')),
                          ('encode',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1))]), cats)
    ])
    return Pipeline([('pre',pre),('lgb',LGBMRegressor(n_estimators=iters,num_leaves=leaves,
        min_child_samples=min_child,reg_lambda=regularization,learning_rate=lr,
        objective=options.get('objective','regression'),
        cat_smooth=options.get('cat_smooth',10.0),
        verbosity=-1,n_jobs=4,random_state=42))])

def fit(frame, spec):
    m = model_factory(spec)
    options = model_options(spec)
    fit_args = {'lgb__sample_weight': training_weights(frame, options.get('half_life',0))}
    if options.get('native_categories'):
        transformers = m.named_steps['pre'].transformers
        n_numeric = len(transformers[0][2])
        fit_args['lgb__categorical_feature'] = list(range(n_numeric, n_numeric + len(transformers[1][2])))
    extra = [x for x in model_options(spec).get('extra_features', []) if x in EXPERIMENTAL_FEATURES]
    columns = FEATURES + [x for x in extra if x not in FEATURES]
    m.fit(input_frame(frame,spec[1])[columns], np.log(frame.price_man.to_numpy(dtype=float)), **fit_args)
    return m

def pred(model, frame, excluded):
    x = input_frame(frame,excluded)
    for c in getattr(model, 'feature_names_in_', FEATURES):
        if c not in x: x[c] = '不明' if c in CATS or c in EXPERIMENTAL_CATS else np.nan
    v = model.predict(x[list(model.feature_names_in_)])
    return np.clip(np.exp(np.clip(v,-10,15)),50,100000)

class V8Pipeline:
    """Site-compatible predictor returning yen. Loaded via importable research_v8 module."""
    def __init__(self, model, excluded, regional=None, regional_weight=0.0):
        self.model=model; self.excluded=excluded; self.city_codes_padded=True
        self.regional=regional or {}; self.regional_weight=regional_weight
    def predict(self, frame):
        x=frame.copy()
        for c in getattr(self.model, 'feature_names_in_', FEATURES):
            if c not in x: x[c] = '不明' if c in CATS or c in EXPERIMENTAL_CATS else np.nan
        for c in CATS: x[c]=x[c].fillna('不明').astype(str)
        if getattr(self,'city_codes_padded',False):
            x['MunicipalityCode'] = normalize_city_codes(x['MunicipalityCode'])
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
    'road_access': ['Breadth', 'Direction', 'Classification'],
    'lot_shape': ['LandShape', 'Frontage'],
    'zoning_limits': ['CoverageRatio', 'FloorAreaRatio'],
    'renovation_proxy': ['Renovation'],
}
FEATURE_HINTS = {'StationMinutes': 'station', 'DistrictName': 'district',
                 'MunicipalityCode': 'city', 'BuildingAge': 'age',
                 'Structure': 'structure', 'Area': 'land',
                 'Breadth': 'road_access', 'LandShape': 'lot_shape',
                 'Renovation': 'renovation_proxy'}

def propose(champion, diagnostics, experiments, round_no, seed=42):
    """Feedback-directed, reproducible hypothesis generator, not a fixed experiment list.

    Mutations use previous successes/failures and the worst error segments.
    """
    rng = random.Random(seed + round_no)
    _, excluded, leaves, trees, child, reg, lr = champion[:7]
    options = model_options(champion)
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
    def add(reason, ex, lv, tr, ch, rg, rate, settings=None):
        lv = int(max(15, min(511, lv)))
        tr = int(max(120, min(1400, tr)))
        ch = int(max(15, min(300, ch)))
        rg = round(float(max(.01, min(150, rg))), 5)
        rate = round(float(max(.008, min(.15, rate))), 5)
        spec = (f'round{round_no}_{len(ideas)+1}', sorted(set(ex) | SITE_UNAVAILABLE), lv, tr, ch, rg, rate)
        settings = options if settings is None else settings
        if settings: spec = (*spec, dict(settings))
        key = json.dumps(spec[1:], sort_keys=True)
        if key not in seen and all(json.dumps(x[1][1:],sort_keys=True)!=key for x in ideas):
            ideas.append((reason, spec))
    # Test methods as well as tree counts. All choices are made on development
    # data only; production promotion still requires the independent gate.
    method_tests = [
        ('native category splits', {'native_categories': True}),
        ('robust absolute log-price loss', {'objective': 'regression_l1'}),
        ('recent training transactions, two-year half-life', {'half_life': 2.0}),
        ('native categories with robust loss and recency',
         {'native_categories': True, 'objective': 'regression_l1', 'half_life': 2.0}),
    ]
    for reason, changes in method_tests:
        settings = {**options, **changes}
        if settings != options:
            add(reason, excluded, leaves, trees, child, reg, lr, settings)
    # Station minutes are now supplied by the Site; always compare this hypothesis,
    # including when other error segments occupy the first diagnostic positions.
    if 'StationMinutes' in excluded:
        add('use available station walking minutes', [x for x in excluded if x != 'StationMinutes'], leaves, trees, child, reg, lr)
    else:
        add('compare without station walking minutes', list(excluded)+['StationMinutes'], leaves, trees, child, reg, lr)
    experimental_hypotheses = [
        ('experiment road access fields', ['Breadth', 'Direction', 'Classification']),
        ('experiment lot shape and frontage', ['LandShape', 'Frontage']),
        ('experiment zoning limits', ['CoverageRatio', 'FloorAreaRatio']),
        ('experiment renovation proxy', ['Renovation']),
    ]
    reason, fields = experimental_hypotheses[(round_no - 1) % len(experimental_hypotheses)]
    settings = {**options, 'extra_features': fields}
    if settings != options:
        add(reason, excluded, leaves, trees, child, reg, lr, settings)
    for group in priority[:1]:
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

def better_research(candidate, incumbent):
    # Do not improve the middle case by making expensive errors much worse.
    return (candidate['median_ape_pct'] < incumbent['median_ape_pct']
            and candidate['mae_man_yen'] <= incumbent['mae_man_yen'] * 1.01
            and candidate['within20_pct'] >= incumbent['within20_pct'] - 1)

def run(args):
    site=args.site.resolve(); root=Path(__file__).resolve().parent
    if not (site/'app.py').is_file(): raise FileNotFoundError('Site app.py not found')
    h,e=discover(root); h=args.history or h; e=args.external or e
    if not h or not h.is_file(): raise FileNotFoundError('Historical ZIP not found')
    out=site/'models'/'candidates'; out.mkdir(parents=True,exist_ok=True)
    report_path=out/'adaptive_research_report.json'
    # Re-key the search whenever the archived or downloaded MLIT data changes.
    input_paths=[h]
    if not args.quick:
        input_paths.extend(sorted((root/'data').glob('mlit_*.csv')))
    metadata_path=root/'data'/'mlit_latest_holdout.json'
    if not args.quick and metadata_path.is_file(): input_paths.append(metadata_path)
    fingerprints=[]
    for p in input_paths:
        if p and p.is_file():
            digest=hashlib.sha256()
            with p.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024*1024), b''):
                    digest.update(chunk)
            fingerprints.append(digest.hexdigest())
    validation_method='rolling-origin-selection-reserved-confirmation-years-excluding-latest-mlit-quarter-v2'
    signature=hashlib.sha256((DATA_POLICY+'|'+validation_method+'|'+'|'.join(fingerprints)).encode()).hexdigest()
    raw=load_training_rows(h,root,args.quick)
    data=prepare_site_training(raw)
    if len(data)<2000: raise RuntimeError('Insufficient research data')
    all_years=tuple(int(y) for y in sorted(pd.to_numeric(data.TransactionYear,errors='coerce').dropna().unique()) if int(y)>=2022)
    if len(all_years)<3 and not args.quick:
        raise RuntimeError('At least three chronological years are required for selection plus confirmation')
    confirmation_years = all_years[-2:] if len(all_years) >= 4 else all_years[-1:]
    validation_years = all_years[-2:] if args.quick else all_years[:-len(confirmation_years)]
    if len(validation_years)<2: raise RuntimeError('At least two chronological selection folds are required')
    # The newest downloaded quarter is reserved for incumbent comparison; older
    # downloaded quarters join training and rolling out-of-time validation.
    champion=('baseline',sorted(SITE_UNAVAILABLE),127,450,60,8.0,.045)
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
                if c and len(c) in (7,8): champion=tuple(c)
            else:
                # Keep the last successful search configuration as the starting
                # incumbent, but never re-use its experiment history on new data.
                c=old.get('champion_spec')
                if c and len(c) in (7,8): champion=tuple(c)
                print('Data or validation policy changed; restarting experiments from the prior champion',flush=True)
        except (ValueError,OSError): pass
    champion=(champion[0],sorted(set(champion[1]) | SITE_UNAVAILABLE),*champion[2:])
    coverage = {c: round(float(data[c].notna().mean() * 100), 2) if c in data else 0.0 for c in EXPERIMENTAL_FEATURES}
    report={'version':'adaptive-6','scope':DATA_POLICY,'validation_method':validation_method,'site_unavailable_features':sorted(SITE_UNAVAILABLE),'experimental_feature_coverage_pct':coverage,'status':'running','data_signature':signature,'started_at':datetime.now().isoformat(),
        'training_n':len(data),'validation_years':list(validation_years),
        'mlit_holdout_metadata':json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.is_file() else None,
        'warning':'The newest fetched MLIT quarter is excluded from fitting and reserved for model comparison; earlier quarters join training and rolling validation.',
        'experiments':completed,'champion_spec':list(champion)}
    # Error segments use the most recent validation year; model selection pools chronological folds.
    diagnostic_year=validation_years[-1]
    diagnostic_train=data.loc[data.TransactionYear<diagnostic_year].reset_index(drop=True)
    val=data.loc[data.TransactionYear==diagnostic_year].reset_index(drop=True)
    if len(diagnostic_train)<1000 or len(val)<1000: raise RuntimeError('Insufficient latest-year validation rows')
    diagnostic_model=fit(diagnostic_train,champion)
    current_predictions=pred(diagnostic_model,val,champion[1])
    starting_spec=tuple(champion)
    best, best_folds=temporal_cv(data,champion,years=validation_years,min_rows=(100 if args.quick else 1000))
    report['starting_cv']=dict(best);report['starting_cv_folds']=best_folds
    report['starting_spec']=list(champion)
    print(f'Adaptive research: rolling folds={len(best_folds)} rows={best["n"]:,}; baseline median APE={best["median_ape_pct"]}%',flush=True)
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
                m, fold_metrics=temporal_cv(data,spec,years=validation_years,min_rows=(100 if args.quick else 1000))
                entry['temporal_cv']=m;entry['folds']=fold_metrics;entry['status']='ok'
                if better_research(m,best):
                    champion=spec;best=m;best_folds=fold_metrics;improvement=True
                    diagnostic_model=fit(diagnostic_train,champion)
                    current_predictions=pred(diagnostic_model,val,champion[1])
                    entry['new_champion']=True
            except Exception as ex:
                entry.update(status='failed',error=f'{type(ex).__name__}: {ex}')
            report['experiments'].append(entry)
            report['champion_spec']=list(champion)
            report['best_cv']=best;report['best_cv_folds']=best_folds
            report['updated_at']=datetime.now().isoformat()
            write_json(report_path,report)
            print(f'[{r}] {spec[0]}: {entry.get("temporal_cv",{}).get("median_ape_pct","ERROR")}% best={best["median_ape_pct"]}%',flush=True)
        print(f'Round {r} finished; improved={improvement}; worst segment={diagnostics[0] if diagnostics else None}',flush=True)
    report['latest_validation_fold']=best_folds[-1] if best_folds else None
    report['status']='completed';report['completed_at']=datetime.now().isoformat()
    report['candidate_only']=True
    if not args.quick:
        confirmation_path=root/'data'/'rolling_origin_confirmation.csv'
        report['internal_cv_confirmation']=temporal_confirmation(
            data, starting_spec, champion, confirmation_years, confirmation_path)
        report['internal_cv_confirmation']['data_signature']=signature
        full=data.reset_index(drop=True)
        final=fit(full,champion)
        end_year=int(pd.to_numeric(full.TransactionYear,errors='coerce').max())
        end_quarter=int(pd.to_numeric(full.loc[full.TransactionYear.eq(end_year),'Quarter'],errors='coerce').max())
        experimental = [x for x in model_options(champion).get('extra_features', []) if x in EXPERIMENTAL_FEATURES]
        bundle={'pipeline':V8Pipeline(final,champion[1]),'feature_columns':list(final.feature_names_in_),
                'required_site_features':experimental,
                'model_key':'used_house','target_unit':'yen','training_end_year':end_year,
                'training_end_quarter':end_quarter,'candidate_only':True,
                'categorical_features':CATS,'data_policy':DATA_POLICY,
                'site_available_features':sorted(SITE_AVAILABLE),
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
