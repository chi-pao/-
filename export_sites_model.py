"""Publish only independently approved, baseline-tested, inference-verified models.

Never publishes evaluation rows. An immutable binary and aggregate approval
metadata are the only public outputs. Failure leaves the current manifest intact.
"""
import argparse, gzip, hashlib, json, os, struct, subprocess, tempfile
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

BASELINE_SHA='a446effe7ec69ad07e635739ca2a2ccfd2f20a6d19a487a632b289994834ec51'
REFERENCE={'year':2026,'quarter':4,'label':'2026年10月'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def scores(y,p):
    e=np.abs(y-p)
    return {'median_ape_pct':float(np.median(e/y)*100),'mae_yen':float(e.mean()),'within20_pct':float((e/y<=.2).mean()*100)}
def improves(before,after):
    return (after['median_ape_pct']<=before['median_ape_pct']-.5 and after['mae_yen']<=before['mae_yen']*1.01 and after['within20_pct']>=before['within20_pct']-1)
def numeric_json(v):
    if isinstance(v,np.generic):v=v.item()
    return None if isinstance(v,float) and not np.isfinite(v) else v

def export_pipeline(pipe,path,public_path):
    pre=pipe.named_steps['pre'];model=pipe.named_steps['lgb']
    nums=list(pre.transformers_[0][2]);cats=list(pre.transformers_[1][2])
    num=pre.named_transformers_['num'];cat=pre.named_transformers_['cat']
    defaults=[float(v) for v in num.steps[0][1].statistics_]
    cat_defaults=[str(v) for v in cat.steps[0][1].statistics_]
    categories=[[str(v) for v in c] for c in cat.steps[-1][1].categories_]
    dump=model.booster_.dump_model()
    if dump.get('num_tree_per_iteration',1)!=1:raise ValueError('Only scalar regression is supported')
    data=bytearray();roots=[]
    def emit(n):
        index=len(data)//16
        if 'leaf_value' in n:data.extend(struct.pack('<dihh',float(n['leaf_value']),-1,-1,0));return index
        if n['decision_type']!='<=' or n.get('missing_type')=='Zero':raise ValueError('Unsupported LightGBM split')
        data.extend(struct.pack('<dihh',float(n['threshold']),-1,int(n['split_feature']),0))
        assert emit(n['left_child'])==index+1
        right=emit(n['right_child']);struct.pack_into('<i',data,index*16+8,right);return index
    for tree in dump['tree_info']:roots.append(emit(tree['tree_structure']))
    packed=gzip.compress(bytes(data),compresslevel=9,mtime=0)
    if len(packed)>=25000000:raise ValueError('Export exceeds browser model limit')
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(packed)
    return {'path':public_path,'sha256':hashlib.sha256(packed).hexdigest(),'bytes':len(packed),'nodes':len(data)//16,
            'algorithm':'lightgbm','postprocess':'v8_log_man','baseline':0,'log':False,'roots':roots,'weight':1,
            'numeric':nums,'numericDefaults':defaults,'categorical':cats,'categoricalDefaults':cat_defaults,'categories':categories}

def prediction(bundle,frame):
    out=np.asarray(bundle['pipeline'].predict(frame[bundle['feature_columns']]),dtype=float)
    if out.shape!=(len(frame),) or not np.all(np.isfinite(out)) or np.any(out<=0):raise ValueError('Invalid model predictions')
    return out

def publish(args):
    approval=json.loads(args.approval.read_text());digest=sha(args.approved);holdout_sha=sha(args.holdout)
    if approval.get('approved') is not True or approval.get('sha256')!=digest:raise ValueError('Model approval fingerprint mismatch')
    if approval.get('holdout_sha256')!=holdout_sha:raise ValueError('Approval and export must use the same unseen holdout')
    expected=os.environ.get('INDEPENDENT_HOLDOUT_SHA256','').lower()
    if expected!=holdout_sha:raise ValueError('Independent holdout fingerprint is not configured or does not match')
    if sha(args.baseline)!=BASELINE_SHA:raise ValueError('Baseline differs from the Site incumbent')
    candidate=joblib.load(args.approved);baseline=joblib.load(args.baseline)
    if candidate.get('approval_status')!='approved' or candidate.get('model_key')!='used_house' or candidate.get('target_unit')!='yen':raise ValueError('Unsupported or unapproved bundle')
    data=pd.read_csv(args.holdout);y=pd.to_numeric(data['price_yen'],errors='coerce').to_numpy(float)
    if len(data)<200 or not np.all(np.isfinite(y)) or np.any(y<=0):raise ValueError('Insufficient or invalid holdout prices')
    years=pd.to_numeric(data['TransactionYear'],errors='coerce').to_numpy(float)
    # The saved timestamp provides a conservative upper bound, not a claimed
    # exact training cutoff. Same-year data are not assumed to be unseen.
    baseline_end=int(baseline.get('training_end_year') or str(baseline.get('trained_at',''))[:4])
    training_end=int(candidate['training_end_year'])
    if not np.all(np.isfinite(years)) or np.any(years<=max(baseline_end,training_end)):raise ValueError('Holdout must postdate both models; the bundled Site baseline was saved in '+str(baseline_end))
    before=scores(y,prediction(baseline,data));after=scores(y,prediction(candidate,data));prior=approval['evaluation']['incumbent']
    if not improves(before,after) or not improves(prior,after):raise ValueError('Candidate does not improve both incumbent and Site baseline')
    wrapper=candidate['pipeline'];components=[];out=args.output;public_root='site-models/versions/'+digest
    pipelines=[('global',None,wrapper.model)]+[(f'region-{i}',name,pipe) for i,(name,pipe) in enumerate(wrapper.regional.items())]
    for tag,region,pipe in pipelines:
        c=export_pipeline(pipe,out/'versions'/digest/(tag+'.dat'),public_root+'/'+tag+'.dat');c['region']=region;components.append(c)
    model={'name':'自動学習・精度確認済みモデル（中古戸建て）','metrics':after,'components':components,'regionalWeight':float(wrapper.regional_weight)}
    # Check the exact JavaScript runtime used by the Site against fitted Python.
    sample=data.sample(n=min(32,len(data)),random_state=42)
    expected_predictions=prediction(candidate,sample)
    fixtures=[{'row':{k:numeric_json(v) for k,v in row.items()},'expected':float(expected_predictions[i])} for i,row in enumerate(sample.to_dict('records'))]
    with tempfile.TemporaryDirectory() as temp:
        file=Path(temp)/'private-parity.json';file.write_text(json.dumps({'model':model,'root':str(out.parent.resolve()),'fixtures':fixtures},ensure_ascii=False,allow_nan=False))
        subprocess.run(['node',str(args.verifier.resolve()),str(file)],check=True)
    manifest={'schema':1,'status':'approved','model_key':'used_house','model_sha256':digest,'baseline_sha256':BASELINE_SHA,
              'reference':REFERENCE,'training_end_year':training_end,'baseline_training_end_year':baseline_end,
              'approved_at':datetime.now(timezone.utc).isoformat(),'model':model,
              'evaluation':{'baseline':before,'incumbent':prior,'challenger':after,'holdout_sha256':holdout_sha,'holdout_rows':len(data),'holdout_year':int(years.min())}}
    path=out/'manifest.json'
    if path.exists():
        previous=json.loads(path.read_text())
        if previous.get('status')=='approved':previous.pop('previous',None);manifest['previous']=previous
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(manifest,ensure_ascii=False,separators=(',',':')));os.replace(temp,path)
    return True

def main():
    p=argparse.ArgumentParser();p.add_argument('--approved',type=Path,required=True);p.add_argument('--approval',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--holdout',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('site-models'));p.add_argument('--verifier',type=Path,default=Path('verify_sites_export.mjs'))
    args=p.parse_args()
    try:exported=publish(args)
    except Exception as exc:print('Site model retained:',str(exc),flush=True);exported=False
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'],'a') as f:f.write('exported='+str(exported).lower()+'\n')
    print(json.dumps({'exported':exported}));return 0
if __name__=='__main__':raise SystemExit(main())
