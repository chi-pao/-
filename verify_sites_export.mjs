import {readFileSync} from 'node:fs';
import {gunzipSync} from 'node:zlib';
import {createHash} from 'node:crypto';
import {componentPrediction} from './sites_model_runtime.mjs';
const data=JSON.parse(readFileSync(process.argv[2]));
const components=data.model.components.map(c=>{const bytes=readFileSync(data.root+'/'+c.path);if(createHash('sha256').update(bytes).digest('hex')!==c.sha256)throw Error('Export hash mismatch');const b=gunzipSync(bytes);return [c,b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength)];});
const predictions=[];
for(const f of data.fixtures){const global=components.find(([c])=>c.region==null),regional=components.find(([c])=>c.region===f.row.Prefecture);let got=componentPrediction(f.row,...global);if(regional){const w=data.model.regionalWeight;got=(1-w)*got+w*componentPrediction(f.row,...regional);}predictions.push(got);if(!data.predictionOnly&&Math.abs(got-f.expected)/f.expected>1e-10)throw Error('Python/JavaScript prediction mismatch');}
if(data.predictionOnly)console.log(JSON.stringify(predictions));else console.log('Portable learned-model parity passed:',data.fixtures.length);
