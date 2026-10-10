"""全国不動産価格・自動比較研究エンジン。既存アプリを書き換えない。"""
from __future__ import annotations
import argparse, json, re, time, zipfile, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor

JCOL={'種類':'Type','価格情報区分':'PriceCategory','都道府県名':'Prefecture','市区町村名':'Municipality','市区町村コード':'MunicipalityCode','地区名':'DistrictName','最寄駅：名称':'NearestStation','最寄駅：距離（分）':'StationMinutes','取引価格（総額）':'TradePrice','面積（㎡）':'Area','延床面積（㎡）':'TotalFloorArea','建築年':'BuildingYear','建物の構造':'Structure','都市計画':'CityPlanning','取引時期':'Period','用途':'Use'}
NUM=['Area','TotalFloorArea','Age','StationMinutes','Year','Quarter']
CAT=['Prefecture','MunicipalityCode','DistrictName','Structure','CityPlanning','Use']

def discover(root:Path):
    paths=[root,root/'data',Path.home()/'Downloads',Path.home()/'OneDrive'/'Desktop'/'data',Path.home()/'OneDrive'/'Desktop']
    def find(name):
        for p in paths:
            if (p/name).is_file(): return p/name
        return None
    return find('All_20214_20254.zip'),find('mlit_2026_Q1.csv')

def normalize(df):
    df=df.rename(columns=JCOL)
    if 'Type' not in df: return pd.DataFrame()
    df=df[df.Type.eq('宅地(土地と建物)')].copy()
    for c in ['TradePrice','Area','TotalFloorArea','StationMinutes']:
        if c not in df: df[c]=np.nan
        df[c]=pd.to_numeric(df[c],errors='coerce')
    df['Year']=pd.to_numeric(df['Period'].astype(str).str.extract(r'(20\d{2})')[0],errors='coerce')
    df['Quarter']=pd.to_numeric(df['Period'].astype(str).str.extract(r'第\s*([1-4１-４])')[0].replace({'１':'1','２':'2','３':'3','４':'4'}),errors='coerce')
    # Missing construction year is genuinely missing, not a zero-year-old building.
    built=pd.to_numeric(df.get('BuildingYear',pd.Series(index=df.index,dtype=object)).astype(str).str.extract(r'(19\d{2}|20\d{2})')[0],errors='coerce')
    df['Age']=(df['Year']-built).where(built.le(df['Year']))
    for c in CAT:
        if c not in df: df[c]='不明'
        df[c]=df[c].fillna('不明').astype(str).str.strip().replace({'':'不明','nan':'不明'})
    if 'PriceCategory' not in df: df['PriceCategory']='不明'
    df['PriceCategory']=df['PriceCategory'].fillna('不明').astype(str)
    df['price']=df.TradePrice/10000
    return df[df.price.between(100,30000)&df.Area.between(20,2000)&df.TotalFloorArea.between(20,1000)&df.Year.between(2021,2026)].copy()

def load_history(path, limit=None):
    parts=[]
    with zipfile.ZipFile(path) as z:
        names=[n for n in z.namelist() if n.lower().endswith('.csv')]
        if not names: raise ValueError('過去データZIPにCSVがありません')
        for name in names:
            with z.open(name) as f:
                df=pd.read_csv(f,encoding='cp932',low_memory=False)
            parts.append(normalize(df))
    out=pd.concat(parts,ignore_index=True).drop_duplicates()
    if limit: out=out.loc[np.concatenate([g.sample(min(len(g),limit),random_state=41).index.to_numpy() for _,g in out.groupby('Year')])].reset_index(drop=True)
    return out

def load_external(path, limit=None):
    df=normalize(pd.read_csv(path,low_memory=False))
    df=df[df.Year.eq(2026)].drop_duplicates()
    if limit: df=df.loc[np.concatenate([g.sample(min(len(g),limit),random_state=41).index.to_numpy() for _,g in df.groupby('PriceCategory')])].reset_index(drop=True)
    return df

def featurize(train,other,mode):
    cat=CAT if mode=='all' else ['Prefecture','MunicipalityCode','DistrictName','Structure']
    maps={c:{v:i for i,v in enumerate(train[c].unique())} for c in cat}
    def transform(d):
        x=d[NUM].copy()
        for c in cat:x[c]=d[c].map(maps[c]).fillna(-1).astype('float32')
        if mode=='robust':
            x=x.drop(columns=['StationMinutes','Age'])
        return x.replace([np.inf,-np.inf],np.nan).astype('float32')
    return transform(train),transform(other)

def metrics(y,p):
    y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float)
    mask=np.isfinite(y)&np.isfinite(p)&(y>0);y=y[mask];p=p[mask]
    if not len(y):return {'n':0}
    ape=np.abs(y-p)/y
    return {'n':int(len(y)),'median_ape_pct':round(float(np.median(ape)*100),2),'within20_pct':round(float(np.mean(ape<=.2)*100),2),'mae_man_yen':round(float(mean_absolute_error(y,p)),1),'bias_pct':round(float(np.median((p-y)/y)*100),2)}

def model_factory(spec):
    kind=spec['kind']
    if kind=='baseline':return DummyRegressor(strategy='median')
    if kind=='hgb':return HistGradientBoostingRegressor(max_iter=spec['iters'],max_leaf_nodes=spec['leaves'],learning_rate=.05,l2_regularization=10,random_state=42)
    if kind=='lgb':
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=spec['iters'],num_leaves=spec['leaves'],learning_rate=.045,min_child_samples=60,reg_lambda=8,verbosity=-1,n_jobs=4,random_state=42)
    if kind=='cat':
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=spec['iters'],depth=spec['depth'],learning_rate=.05,loss_function='RMSE',verbose=False,thread_count=4,random_seed=42,l2_leaf_reg=8)
    raise ValueError(kind)

def evaluate(spec,train,valid):
    xt,xv=featurize(train,valid,spec['mode'])
    xt=xt.fillna(-1);xv=xv.fillna(-1)
    model=model_factory(spec)
    model.fit(xt,np.log(train.price.to_numpy()))
    return np.clip(np.exp(model.predict(xv)),50,100000)

def groups(df,p):
    out={}
    for col in ['PriceCategory','Prefecture']:
        out[col]={}
        for name,ix in df.groupby(col).groups.items():
            if len(ix)<30:continue
            loc=df.index.get_indexer(ix)
            out[col][str(name)]=metrics(df.iloc[loc].price,p[loc])
    return out

def candidates(quick=False,catboost=False):
    specs=[{'kind':'baseline','mode':'robust','name':'全国中央値'},
           {'kind':'lgb','mode':'robust','leaves':31,'iters':220,'name':'LightGBM欠損耐性31'},
           {'kind':'lgb','mode':'all','leaves':63,'iters':250,'name':'LightGBM全特徴63'},
           {'kind':'lgb','mode':'robust','leaves':127,'iters':300,'name':'LightGBM欠損耐性127'},
           {'kind':'hgb','mode':'robust','leaves':31,'iters':170,'name':'HistGB欠損耐性'}]
    if not quick:
        specs += [{'kind':'lgb','mode':'all','leaves':255,'iters':450,'name':'LightGBM全特徴255'},
                  {'kind':'lgb','mode':'robust','leaves':255,'iters':450,'name':'LightGBM欠損耐性255'}]
    if catboost and not quick:
        specs += [{'kind':'cat','mode':'robust','depth':6,'iters':400,'name':'CatBoost欠損耐性6'}, {'kind':'cat','mode':'all','depth':8,'iters':500,'name':'CatBoost全特徴8'}]
    return specs

def main():
    parser=argparse.ArgumentParser(description='全国取引価格AI自動研究。既存アプリは変更しません')
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--history',type=Path)
    parser.add_argument('--external',type=Path)
    parser.add_argument('--final-only',action='store_true',help='2024年比較の暫定首位255葉を2025/2026で検証')
    parser.add_argument('--catboost',action='store_true',help='追加でCatBoostも比較（時間がかかります）')
    parser.add_argument('--quick',action='store_true',help='動作確認: 年ごとに最大3000件で短時間比較')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args(); t=time.time()
    hist,ext=discover(args.root);hist=args.history or hist;ext=args.external or ext
    if not hist or not hist.exists():raise SystemExit('過去の全国ZIPがありません: All_20214_20254.zip をこのファイルと同じフォルダーへ配置してください')
    if not ext or not ext.exists():raise SystemExit('2026全国CSVがありません: mlit_2026_Q1.csv を同じフォルダーか data フォルダーへ配置してください')
    out=args.out or args.root/'research_output';out.mkdir(parents=True,exist_ok=True)
    print('入力:',hist,ext,flush=True)
    d=load_history(hist,3000 if args.quick else None)
    # 2025 never influences selection. 2026 never influences model selection.
    tr=d[d.Year<=2023].reset_index(drop=True);va=d[d.Year.eq(2024)].reset_index(drop=True);test=d[d.Year.eq(2025)].reset_index(drop=True)
    if min(len(tr),len(va),len(test))<100:raise SystemExit('学習・選定・最終検証のいずれかのデータが不足')
    results=[];rank=[]
    for spec in ([] if args.final_only else candidates(args.quick,args.catboost)):
        try:
            p=evaluate(spec,tr,va);m=metrics(va.price,p)
            results.append({'model':spec,'validation_2024':m});rank.append((m['median_ape_pct'],m['mae_man_yen'],spec))
            print('2024',spec['name'],m,flush=True)
        except Exception as e:
            results.append({'model':spec,'error':repr(e)})
            print('SKIP',spec['name'],str(e),flush=True)
    if args.final_only:
        best={'kind':'lgb','mode':'all','leaves':255,'iters':450,'name':'LightGBM全特徴255'}
    else:
        if not rank:raise SystemExit('すべての候補が失敗しました')
        best=min(rank,key=lambda r:(r[0],r[1]))[2]
    # Train through 2024, evaluate 2025 once. Do not use 2025 to choose among candidates.
    train=d[d.Year<=2024].reset_index(drop=True)
    p25=evaluate(best,train,test)
    # 2026 external evaluation with same fixed winner, no 2026 retraining.
    extd=load_external(ext,3000 if args.quick else None).reset_index(drop=True)
    p26=evaluate(best,train,extd)
    report={'status':'completed','quick_sample_only':args.quick,'input':{'history':str(hist),'external':str(ext)},'split':{'train_2021_2023':len(tr),'selection_2024':len(va),'final_2025':len(test),'external_2026':len(extd)},'candidates':results,'selected_on_2024_only':best,'selection_note':'final-only uses previously observed 2024 leader; candidate table not repeated' if args.final_only else 'selected automatically from candidates','final_2025':metrics(test.price,p25),'external_2026':metrics(extd.price,p26),'external_2026_by_group':groups(extd,p26),'limitations':['2026 is external holdout, not training/selection','historical transaction data is not exclusively broker-confirmed closed sales','missing station distance and different city planning classifications in 2026 can reduce accuracy','property condition and precise address may be absent','No production model is promoted automatically'],'elapsed_seconds':round(time.time()-t,1)}
    dest=out/('quick_report.json' if args.quick else ('final_only_report.json' if args.final_only else 'full_report.json'))
    dest.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DONE',dest,flush=True)
    print('2025',report['final_2025'],'2026',report['external_2026'],flush=True)

if __name__=='__main__':
    warnings.filterwarnings('ignore',category=FutureWarning)
    main()
