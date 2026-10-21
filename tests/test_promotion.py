"""Synthetic-only safety checks; these tests never authorize a production model."""
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

ROOT = Path(__file__).resolve().parents[1]


def bundle(value):
    model = DummyRegressor(strategy="constant", constant=value)
    model.fit(pd.DataFrame({"feature": [0, 1]}), [value, value])
    return {
        "model_key": "used_house",
        "target_unit": "yen",
        "training_end_year": 2025,
        "feature_columns": ["feature"],
        "pipeline": model,
    }


class PromotionSafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "data").mkdir()
        (self.root / "models" / "candidates").mkdir(parents=True)
        self.holdout = self.root / "data" / "unused_holdout.csv"
        pd.DataFrame({
            "feature": np.arange(220),
            "price_yen": [10000000] * 220,
            "TransactionYear": [2027] * 220,
        }).to_csv(self.holdout, index=False)
        self.incumbent = self.root / "incumbent.joblib"
        self.challenger = self.root / "challenger.joblib"
        joblib.dump(bundle(12000000), self.incumbent)
        joblib.dump(bundle(10000000), self.challenger)

    def run_promotion(self):
        return subprocess.run([
            sys.executable, str(ROOT / "promote_used_house.py"),
            "--incumbent", str(self.incumbent),
            "--challenger", str(self.challenger),
            "--holdout", str(self.holdout),
            "--holdout-year", "2027",
            "--site", str(self.root),
        ], capture_output=True, text=True)

    def test_improvement_approved_and_same_holdout_cannot_repeat(self):
        first = self.run_promotion()
        self.assertEqual(first.returncode, 0, first.stderr)
        manifest = self.root / "models" / "candidates" / "approved_used_house.json"
        self.assertTrue(json.loads(manifest.read_text())["approved"])
        second = self.run_promotion()
        self.assertNotEqual(second.returncode, 0)
        self.assertIn("already used", second.stderr)

    def test_worse_model_never_replaces_incumbent(self):
        joblib.dump(bundle(14000000), self.challenger)
        result = self.run_promotion()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "models" / "candidates" / "approved_used_house.joblib").exists())

    def test_old_holdout_rejected(self):
        frame = pd.read_csv(self.holdout)
        frame["TransactionYear"] = 2025
        frame.to_csv(self.holdout, index=False)
        result = self.run_promotion()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "models" / "candidates" / "approved_used_house.joblib").exists())

    def test_missing_price_rejected(self):
        frame = pd.read_csv(self.holdout).drop(columns=["price_yen"])
        frame.to_csv(self.holdout, index=False)
        self.assertNotEqual(self.run_promotion().returncode, 0)


class SiteInputPolicyTests(unittest.TestCase):
    def test_residential_used_scope_and_city_codes(self):
        from legacy_engine import normalize
        from research_adaptive import prepare_site_training
        raw = pd.DataFrame({
            'Type': ['宅地(土地と建物)'] * 5,
            'TradePrice': [20000000] * 5, 'Area': [150] * 5,
            'TotalFloorArea': [100] * 5, 'Period': ['2024年第1四半期'] * 5,
            'BuildingYear': ['2000年', '2000年', '2000年', '2024年', None],
            'MunicipalityCode': [1101] * 5, 'Prefecture': ['北海道'] * 5,
            'Use': ['住宅', '共同住宅', '店舗', '住宅', '住宅'],
        })
        result = prepare_site_training(normalize(raw))
        self.assertEqual(len(result), 1)
        self.assertEqual(result.MunicipalityCode.tolist(), ['01101'])
        self.assertEqual(result.BuildingAge.tolist(), [24])

    def test_numeric_and_fullwidth_city_codes(self):
        from research_adaptive import normalize_city_codes
        result = normalize_city_codes(pd.Series([1101, 1101.0, '０１１０１', '13108', None, 'bad', 1101.5]))
        self.assertEqual(result.tolist(), ['01101', '01101', '01101', '13108', '不明', '不明', '不明'])

    def test_search_cannot_restore_unavailable_site_features(self):
        from research_adaptive import SITE_UNAVAILABLE, model_factory, propose
        spec = ('test', [], 31, 120, 15, 8, .05)
        model = model_factory(spec)
        columns = [c for _, _, cols in model.named_steps['pre'].transformers for c in cols]
        self.assertTrue(SITE_UNAVAILABLE.isdisjoint(columns))
        for _, candidate in propose(spec, [], [], 1):
            self.assertTrue(SITE_UNAVAILABLE.issubset(candidate[1]))

    def test_new_predictor_normalizes_codes_without_reinterpreting_old_models(self):
        import research_adaptive as research
        frame = pd.DataFrame({'MunicipalityCode': [1101]})
        model = research.V8Pipeline(None, [])
        def check_code(expected):
            def predict(_, data, excluded):
                self.assertEqual(data.MunicipalityCode.tolist(), [expected])
                return np.array([1000.0])
            return predict
        with patch.object(research, 'pred', check_code('01101')):
            self.assertEqual(model.predict(frame).tolist(), [10000000])
        del model.city_codes_padded
        with patch.object(research, 'pred', check_code('1101')):
            model.predict(frame)

    def test_promotion_and_export_use_the_same_site_inputs(self):
        import promote_used_house
        import export_sites_model
        class Capture:
            def predict(self, frame):
                self.frame = frame.copy()
                return np.full(len(frame), 10000000.0)
        pipe = Capture()
        data = pd.DataFrame({'StationMinutes': [3], 'PriceCategory': ['成約価格'], 'MunicipalityCode': [1101]})
        item = {'model_key': 'used_house', 'target_unit': 'yen',
                'feature_columns': list(data.columns), 'pipeline': pipe,
                'categorical_features': ['PriceCategory', 'MunicipalityCode']}
        for predict in (promote_used_house.predict, export_sites_model.prediction):
            predict(item, data)
            self.assertTrue(pipe.frame.StationMinutes.isna().all())
            self.assertTrue(pipe.frame.PriceCategory.isna().all())
        self.assertEqual(pipe.frame.MunicipalityCode.tolist(), ['01101'])
        self.assertEqual(data.StationMinutes.tolist(), [3])


class PrecisionResearchTests(unittest.TestCase):
    def test_recency_weights_use_training_dates_only(self):
        from research_adaptive import training_weights
        frame = pd.DataFrame({'TransactionYear':[2021,2023,2023], 'Quarter':[1,1,1],
                              'price_man':[500,3000,9999]})
        np.testing.assert_allclose(training_weights(frame,2), [.5,1,1])
        np.testing.assert_allclose(training_weights(frame,0), [1,1,1])
        frame['price_man'] *= 100
        np.testing.assert_allclose(training_weights(frame,2), [.5,1,1])

    def test_method_search_preserves_settings_and_budget(self):
        from research_adaptive import propose, model_options, SITE_UNAVAILABLE
        old = ('old', [], 31,120,15,8,.05)
        self.assertEqual(model_options(old),{})
        ideas = propose(old,[],[],1)
        self.assertLessEqual(len(ideas),8)
        self.assertTrue(any(model_options(s).get('native_categories') for _,s in ideas))
        self.assertTrue(any(model_options(s).get('objective')=='regression_l1' for _,s in ideas))
        self.assertTrue(any(model_options(s).get('half_life')==2 for _,s in ideas))
        champion = (*old, {'native_categories':True,'objective':'regression_l1','half_life':2})
        for _,spec in propose(champion,[],[],2):
            self.assertEqual(model_options(spec),model_options(champion))
            self.assertTrue(SITE_UNAVAILABLE.issubset(spec[1]))

    def test_middle_error_improvement_cannot_hide_average_or_coverage_regression(self):
        from research_adaptive import better_research
        before={'median_ape_pct':30,'mae_man_yen':700,'within20_pct':40}
        self.assertFalse(better_research(dict(before,median_ape_pct=29,mae_man_yen=800),before))
        self.assertFalse(better_research(dict(before,median_ape_pct=29,within20_pct=38),before))
        self.assertTrue(better_research(dict(before,median_ape_pct=29),before))

    def test_real_lightgbm_category_and_numeric_exports_match_javascript(self):
        from research_adaptive import fit, V8Pipeline, FEATURES, CATS, NUMS
        from export_sites_model import export_pipeline
        # Artificial prices exercise the implementation, not model accuracy.
        rng=np.random.default_rng(14)
        town=rng.integers(0,12,960)
        frame=pd.DataFrame({c:np.full(960,100.) for c in NUMS})
        for c in CATS:frame[c]='住宅' if c=='Use' else '不明'
        frame['MunicipalityCode']='01101'
        frame['DistrictName']=['町'+str(i) for i in town]
        frame['TransactionYear']=2023
        frame['Quarter']=1
        effects=np.array([.1,1.4,.3,1.8,.6,1.1,.2,1.6,.7,1.9,.4,1.2])
        frame['price_man']=np.exp(7+effects[town]+rng.normal(0,.01,960))
        cases=frame.iloc[:24].copy()
        cases.loc[cases.index[0],'DistrictName']='未学習の町'
        cases.loc[cases.index[1],'DistrictName']=None
        cases.loc[cases.index[2],'Area']=np.nan
        spec=('test',[],31,120,15,.01,.05)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for native in (False,True):
                fitted=fit(frame,(*spec,{'native_categories':True}) if native else spec)
                component=export_pipeline(fitted,root/'model.dat','model.dat')
                if native:self.assertGreater(len(component['categoricalSplits']),0)
                else:self.assertEqual(component['categoricalSplits'],[])
                expected=V8Pipeline(fitted,[]).predict(cases)
                rows=cases[FEATURES].astype(object).where(cases[FEATURES].notna(),None).to_dict('records')
                fixtures=[{'row':row,'expected':float(p)} for row,p in zip(rows,expected)]
                config={'root':str(root),'model':{'components':[component],'regionalWeight':0},'fixtures':fixtures}
                file=root/'fixtures.json';file.write_text(json.dumps(config,ensure_ascii=False,allow_nan=False))
                result=subprocess.run(['node',str(ROOT/'verify_sites_export.mjs'),str(file)],capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)


if __name__ == "__main__":
    unittest.main()
