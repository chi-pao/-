export function transform(row,c) {
  const v=c.numeric.map((n,i)=>row[n]!=null&&Number.isFinite(Number(row[n]))?Number(row[n]):c.numericDefaults[i]);
  c.categorical.forEach((n,i)=>{let value=row[n]==null||row[n]===''?c.categoricalDefaults[i]:row[n];if(c.algorithm==='lightgbm'){value=String(value);if(n==='MunicipalityCode'&&/^\d+$/.test(value))value=value.padStart(5,'0');}let index=c.categories[i].indexOf(value);v.push(index)});
  return c.algorithm==='forest'?v.map(Math.fround):v;
}
export function componentPrediction(row,c,buffer) {
  const x=transform(row,c),d=new DataView(buffer);let sum=c.baseline;
  for(const root of c.roots){let j=root;for(let depth=0;depth<200;depth++){
    const offset=j*16,f=d.getInt16(offset+12,true),t=d.getFloat64(offset,true);
    if(f<0){sum+=t;break}
    const flags=d.getInt16(offset+14,true);
    let left;
    if(c.algorithm==='lightgbm'&&(flags&1)){
      const members=c.categoricalSplits?.[t];
      if(!Array.isArray(members))throw Error('分類モデルの読み込みに失敗しました');
      left=x[f]<0||!Number.isFinite(x[f])?Boolean(flags&2):members.includes(Math.trunc(x[f]));
    }else left=x[f]<=t;
    j=left?j+1:d.getInt32(offset+8,true);
    if(depth===199)throw Error('モデルの探索に失敗しました');
  }}
  if(c.algorithm==='forest')sum/=c.roots.length;
  if(c.postprocess==='v8_log_man')return Math.max(50,Math.min(100000,Math.exp(Math.max(-10,Math.min(15,sum)))))*10000;
  return c.log?Math.expm1(sum):sum;
}
export function combine(predictions,components) {
  if(predictions.length===1)return predictions[0];
  const weights=components.reduce((s,c)=>s+c.weight,0);
  return Math.exp(predictions.reduce((s,v,i)=>s+components[i].weight*Math.log(Math.max(1,v)),0)/weights);
}
