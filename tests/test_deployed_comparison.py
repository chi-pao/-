"""Synthetic portable models exercise comparison with the actual Site incumbent."""
import gzip,hashlib,io,json,struct,unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from export_sites_model import deployed_scores,BASELINE_SHA

class DeployedComparisonTests(unittest.TestCase):
    def fixture(self,digest='b'*64):
        blob=gzip.compress(struct.pack('<dihh',6.907755278982137,-1,-1,0))
        component={'path':'models/current.dat','bytes':len(blob),'sha256':hashlib.sha256(blob).hexdigest(),
            'algorithm':'lightgbm','postprocess':'v8_log_man','numeric':[],'numericDefaults':[],
            'categorical':[],'categoricalDefaults':[],'categories':[],'categoricalSplits':[],
            'roots':[0],'baseline':0,'log':False,'weight':1}
        catalog={'models':{'used_house':{'sourceSha256':digest,'components':[component]}}}
        return catalog,blob
    def read(self,catalog,blob):
        return lambda url,**kw:io.BytesIO(json.dumps(catalog).encode() if url.endswith('catalog.json') else blob)
    def test_compares_portable_current_model(self):
        catalog,blob=self.fixture()
        with patch('export_sites_model.urlopen',self.read(catalog,blob)):
            metrics,digest=deployed_scores(pd.DataFrame({'price_yen':[10000000,20000000]}),Path(__file__).resolve().parents[1]/'verify_sites_export.mjs')
        self.assertEqual(digest,'b'*64)
        self.assertAlmostEqual(metrics['median_ape_pct'],25)
        self.assertAlmostEqual(metrics['within20_pct'],50)
    def test_original_baseline_needs_no_second_inference(self):
        catalog,blob=self.fixture(BASELINE_SHA)
        with patch('export_sites_model.urlopen',self.read(catalog,blob)):
            self.assertEqual(deployed_scores(pd.DataFrame(),Path('unused')),(None,BASELINE_SHA))
    def test_modified_model_bytes_rejected(self):
        catalog,blob=self.fixture()
        with patch('export_sites_model.urlopen',self.read(catalog,blob+b'x')):
            with self.assertRaisesRegex(ValueError,'integrity'):
                deployed_scores(pd.DataFrame({'price_yen':[1000]}),Path('unused'))
    def test_model_path_cannot_escape_site(self):
        catalog,blob=self.fixture();catalog['models']['used_house']['components'][0]['path']='models/../../secrets.dat'
        with patch('export_sites_model.urlopen',self.read(catalog,blob)):
            with self.assertRaisesRegex(ValueError,'path'):
                deployed_scores(pd.DataFrame({'price_yen':[1000]}),Path('unused'))
