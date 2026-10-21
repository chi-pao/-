"""Accept chronological holdouts or independently audited, unused historical rows.

Historical evidence is private and pinned by EVALUATION_AUDIT_SHA256. Its model
inventories must cover *all* training, tuning, diagnostics and prior evaluation,
not just the final training set. Configure EVALUATION_AUDIT_URL and its SHA256
in the existing workflow only after reviewing the complete source inventories.
No audit is fabricated from a random split of already-used rows. The public
manifest contains aggregate verification metadata, never the private IDs.
Source record identities must be stable across
all inventories; assigning new IDs to already-seen transactions is not an audit.
"""
import hashlib
import json
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independence(data, holdout, models, audit_path=None):
    """models maps exact artifact SHA256 to conservative training end year.

    audit_path defaults to the private data/evaluation_audit.json beside holdout.
    Pinned audits may supersede year cutoffs, never sample or accuracy checks.
    """
    years = pd.to_numeric(data['TransactionYear'], errors='coerce').to_numpy(float)
    if not len(years) or not np.all(np.isfinite(years)) or np.any(years != np.floor(years)):
        raise ValueError('Invalid transaction years')
    if np.any(years < 1900) or np.any(years > 2200):
        raise ValueError('Invalid transaction years')
    path = Path(audit_path) if audit_path else Path(holdout).with_name('evaluation_audit.json')
    expected = os.environ.get('EVALUATION_AUDIT_SHA256', '').lower()
    if not expected:
        if int(years.min()) > 2025 and all(end is not None and int(years.min()) > int(end) for end in models.values()):
            return {'method': 'chronological', 'holdout_sha256': sha(holdout)}
        raise ValueError('Historical/same-year holdout requires a pinned unused-row audit; year alone cannot prove independence')
    if not re.fullmatch('[a-f0-9]{64}', expected) or not path.is_file() or sha(path) != expected:
        raise ValueError('Evaluation audit missing or fingerprint mismatch')
    audit = json.loads(path.read_text())
    if audit.get('schema') != 1 or audit.get('holdout_sha256') != sha(holdout):
        raise ValueError('Audit does not match evaluation data')
    if audit.get('identity_policy') != 'stable-source-transaction-id-v1' or not str(audit.get('source_evidence', '')).strip():
        raise ValueError('Stable source identity evidence is required')
    if audit.get('reserved_before_selection') is not True:
        raise ValueError('Evaluation must be reserved before model selection')
    if 'record_id' not in data or data['record_id'].isna().any():
        raise ValueError('Audited holdout requires stable record_id')
    ids = data['record_id'].astype(str).str.strip()
    if ids.eq('').any() or ids.duplicated().any():
        raise ValueError('Empty or duplicate evaluation identities')
    records = set(ids)
    inventories = audit.get('models', {})
    for digest in models:
        inventory = inventories.get(digest, {})
        if inventory.get('complete_for') != ['training', 'selection', 'diagnostics', 'evaluation']:
            raise ValueError('Incomplete model exposure inventory')
        exposed = inventory.get('exposed_record_ids')
        if not isinstance(exposed, list) or not exposed or not all(isinstance(i, str) and i.strip() == i and i for i in exposed):
            raise ValueError('Missing model exposure identities')
        if records.intersection(exposed):
            raise ValueError('Evaluation overlaps previously exposed transactions')
    return {'method': 'audited_unused_rows', 'audit_sha256': expected,
            'holdout_sha256': sha(holdout), 'verified_model_sha256': sorted(models),
            'training_disjoint': True, 'selection_disjoint': True,
            'source_identity_verified': True}
