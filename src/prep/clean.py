import ast
from collections import Counter
import numpy as np
import pandas as pd
from . import constants as C

def normalize_risk(series: pd.Series) -> pd.Series:
    raw_variants = series.nunique()
    assert raw_variants == C.EXPECTED['risk_raw_variants'], raw_variants
    norm = series.str.strip().str.title()
    counts = norm.value_counts().to_dict()
    assert counts == C.EXPECTED['risk_counts'], counts
    return norm

def parse_evidence(cell) -> list[str]:
    if pd.isna(cell):
        return []
    s = C.LEAK_RE.sub('', str(cell))
    if s.strip().lower() == 'none':
        return []
    return [frag.strip() for frag in s.split(';') if frag.strip()]

def evidence_columns(df: pd.DataFrame) -> pd.DataFrame:
    col = df['evidence for suicide risk level']
    assert int(col.isna().sum()) == C.EXPECTED['evidence_nan_rows']
    leak_hit = col.fillna('').astype(str).str.contains('phrases that lead to this assessment')
    assert set(df.loc[leak_hit, 'row_id']) == set(C.LEAKAGE_ROWS), 'leakage rows drifted'
    raw_none = col.fillna('').astype(str).str.strip().str.lower().eq('none')
    assert int(raw_none.sum()) == C.EXPECTED['evidence_none_cells']
    spans = col.map(parse_evidence)
    df = df.copy()
    df['gold_eval_spans'] = spans
    df['has_evidence'] = spans.map(len) > 0
    return df

def parse_factors_cell(cell: str) -> list[str]:
    lst = ast.literal_eval(cell)
    assert isinstance(lst, list)
    for f in lst:
        assert f in C.FACTOR_INDEX, f'unknown factor string: {f!r}'
    return lst

def factors_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mentions = df['factors'].map(parse_factors_cell)
    df['factors_mentions'] = mentions
    df['factors_set'] = mentions.map(lambda lst: sorted(set(lst), key=C.FACTOR_INDEX.__getitem__))
    mat = np.zeros((len(df), len(C.FACTORS_24)), dtype=np.uint8)
    for i, lst in enumerate(df['factors_set']):
        for f in lst:
            mat[i, C.FACTOR_INDEX[f]] = 1
    for j in range(len(C.FACTORS_24)):
        df[f'f_{j:02d}'] = mat[:, j]
    supports = {f: int(mat[:, j].sum()) for f, j in C.FACTOR_INDEX.items()}
    assert supports == C.FACTOR_SUPPORTS, {k: (supports[k], C.FACTOR_SUPPORTS[k]) for k in supports if supports[k] != C.FACTOR_SUPPORTS[k]}
    assert int((df['factors_set'].map(len) == 0).sum()) == C.EXPECTED['empty_factor_lists']
    max_repeat = max((max(Counter(m).values()) for m in mentions if m))
    assert max_repeat == C.EXPECTED['max_factor_repeat'], max_repeat
    assert df['factors_set'].map(len).max() == C.EXPECTED['max_distinct_factors']
    return df
