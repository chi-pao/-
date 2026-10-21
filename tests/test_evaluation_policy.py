"""Synthetic fixtures validate the policy; they are never production evidence."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from evaluation_policy import independence, sha

class EvaluationPolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.holdout=self.root/'unused_holdout.csv'
        self.data=pd.DataFrame({'TransactionYear':[2024,2024], 'record_id':['a','b']})
        self.data.to_csv(self.holdout,index=False)
        self.models={'a'*64:2025,'b'*64:2026}
        self.audit={'schema':1,'holdout_sha256':sha(self.holdout),
            'identity_policy':'stable-source-transaction-id-v1',
            'source_evidence':'Synthetic test inventory, never used for production',
            'reserved_before_selection':True,
            'models':{m:{'complete_for':['training','selection','diagnostics','evaluation'],
                'exposed_record_ids':['old-row']} for m in self.models}}
        self.path=self.root/'evaluation_audit.json'
    def verify(self):
        self.path.write_text(json.dumps(self.audit))
        with patch.dict(os.environ,{'EVALUATION_AUDIT_SHA256':sha(self.path)}):
            return independence(self.data,self.holdout,self.models)
    def test_historical_unused_rows_pass_with_pinned_complete_audit(self):
        self.assertEqual(self.verify()['method'],'audited_unused_rows')
    def test_year_does_not_block_audited_rows(self):
        self.data['TransactionYear']=2020
        self.data.to_csv(self.holdout,index=False);self.audit['holdout_sha256']=sha(self.holdout)
        self.assertEqual(self.verify()['method'],'audited_unused_rows')
    def test_old_rows_without_evidence_fail(self):
        with patch.dict(os.environ,{'EVALUATION_AUDIT_SHA256':''}),self.assertRaisesRegex(ValueError,'pinned'):
            independence(self.data,self.holdout,self.models)
    def test_overlap_fails_for_either_model(self):
        for m in self.models:
            self.audit['models'][m]['exposed_record_ids']=['a']
            with self.assertRaisesRegex(ValueError,'overlaps'):self.verify()
            self.audit['models'][m]['exposed_record_ids']=['old-row']
    def test_incomplete_inventory_fails(self):
        self.audit['models']['a'*64]['complete_for']=['training']
        with self.assertRaisesRegex(ValueError,'Incomplete'):self.verify()
    def test_missing_model_inventory_fails(self):
        del self.audit['models']['b'*64]
        with self.assertRaises(ValueError):self.verify()
    def test_tampered_data_fails(self):
        self.holdout.write_text('modified')
        with self.assertRaisesRegex(ValueError,'match'):self.verify()
    def test_unpinned_or_tampered_audit_fails(self):
        with patch.dict(os.environ,{'EVALUATION_AUDIT_SHA256':'0'*64}),self.assertRaisesRegex(ValueError,'fingerprint'):
            independence(self.data,self.holdout,self.models)
    def test_duplicate_or_empty_identity_fails(self):
        for values in (['a','a'],['a',None],['a',' ']):
            self.data['record_id']=values
            with self.assertRaises(ValueError):self.verify()
    def test_reservation_after_selection_is_rejected(self):
        self.audit['reserved_before_selection']=False
        with self.assertRaisesRegex(ValueError,'reserved'):self.verify()
    def test_future_chronological_path_still_works(self):
        self.data['TransactionYear']=2027
        with patch.dict(os.environ,{'EVALUATION_AUDIT_SHA256':''}):
            self.assertEqual(independence(self.data,self.holdout,self.models)['method'],'chronological')
    def test_invalid_year_rejected_even_with_audit(self):
        for year in (2024.5,0,float('nan')):
            self.data['TransactionYear']=year
            with self.assertRaises(ValueError):self.verify()
