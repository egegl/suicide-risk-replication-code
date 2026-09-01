import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from . import constants as C

def assign_folds(df: pd.DataFrame) -> pd.DataFrame:
    d = df.sort_values('row_id').reset_index(drop=True)
    sgkf = StratifiedGroupKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.FOLD_SEED)
    fold = pd.Series(-1, index=d.index, dtype='int8')
    for k, (_, val_idx) in enumerate(sgkf.split(d, y=d['risk_norm'], groups=d['cv_component'])):
        fold.iloc[val_idx] = k
    assert (fold >= 0).all()
    d['fold'] = fold
    assert (d.groupby('cv_component')['fold'].nunique() == 1).all()
    per_fold_attempt = d[d['risk_norm'] == 'Attempt'].groupby('fold').size()
    assert len(per_fold_attempt) == C.N_FOLDS
    assert (per_fold_attempt >= C.MIN_ATTEMPT_PER_FOLD).all(), per_fold_attempt.to_dict()
    return d[['row_id', 'cv_component', 'fold']]
