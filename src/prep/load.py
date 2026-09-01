import hashlib
import re
from pathlib import Path
import pandas as pd
from . import constants as C

def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _read(path: str, expected_cols: list[str]) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=C.SHEET, engine='openpyxl')
    assert list(df.columns) == expected_cols, f'columns drifted: {list(df.columns)}'
    for col in ('row_id', 'anon_user_id', 'post'):
        df[col] = df[col].astype(str)
    df['post_id'] = df['post_id'].astype('int64')
    assert df['post'].notna().all() and (df['post'].str.len() > 0).all()
    assert df['row_id'].str.fullmatch('P\\d{5}').all()
    assert df['row_id'].is_unique
    return df

def _assert_post_id_consecutive(df: pd.DataFrame) -> None:
    for _, g in df.groupby('anon_user_id', sort=False):
        ids = g['post_id'].tolist()
        assert ids == list(range(len(ids))), f"post_id not consecutive for {g['anon_user_id'].iloc[0]}"

def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    for name, path in (('train.xlsx', C.RAW_TRAIN), ('leaderboard.xlsx', C.RAW_LEADERBOARD)):
        actual = sha256_file(path)
        assert actual == C.RAW_SHA256[name], f'{name} sha256 drifted: {actual}'
    train = _read(C.RAW_TRAIN, C.TRAIN_COLUMNS)
    lb = _read(C.RAW_LEADERBOARD, C.LB_COLUMNS)
    assert train.shape == C.EXPECTED['train_shape'], train.shape
    assert lb.shape == C.EXPECTED['lb_shape'], lb.shape
    assert not set(train['row_id']) & set(lb['row_id'])
    assert not set(train['anon_user_id']) & set(lb['anon_user_id']), 'user overlap!'
    assert train['anon_user_id'].nunique() == C.EXPECTED['n_train_users']
    assert lb['anon_user_id'].nunique() == C.EXPECTED['n_lb_users']
    _assert_post_id_consecutive(train)
    _assert_post_id_consecutive(lb)
    return (train, lb)
