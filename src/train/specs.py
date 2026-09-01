import json
from pathlib import Path
from synth import state
from . import constants as C

def _train_file(arm: str, fold) -> str:
    if arm == 'real_only':
        name = 'full_train.jsonl' if fold == 'full' else f'train_fold{fold}.jsonl'
        return f'{C.SFT_DIR}/{name}'
    name = 'train_full.jsonl' if fold == 'full' else f'train_fold{fold}.jsonl'
    return f'{C.ARMS_DIR}/{arm}/{name}'

def _mean_weight(path: Path) -> float:
    ws = [float(r['weight']) for r in state.read_jsonl(path)]
    return sum(ws) / len(ws)

def build_specs(repo_root: Path) -> list[dict]:
    repo_root = Path(repo_root)
    seeds = sorted(C.TRAIN_SEEDS)
    file_info: dict = {}

    def info(arm: str, train_rel: str):
        if train_rel not in file_info:
            path = repo_root / train_rel
            if path.exists():
                file_info[train_rel] = (state.sha256_file(path), _mean_weight(path))
            elif arm in C.SYNTH_ARMS:
                file_info[train_rel] = ('pending', None)
            else:
                raise SystemExit(f'specs: real_only train file missing: {train_rel}')
        return file_info[train_rel]
    specs: list[dict] = []

    def add(arm: str, fold, seed: int) -> None:
        train_rel = _train_file(arm, fold)
        sha, mw = info(arm, train_rel)
        run_id = f'{arm}-full-s{seed}' if fold == 'full' else f'{arm}-f{fold}-s{seed}'
        specs.append({'run_id': run_id, 'index': len(specs), 'arm': arm, 'fold': fold, 'seed': seed, 'train_file': train_rel, 'train_file_sha': sha, 'mean_weight': mw, 'val_file': None if fold == 'full' else f'{C.SFT_DIR}/val_fold{fold}.jsonl', 'base_model': C.BASE_MODEL, 'total_steps': C.TOTAL_STEPS, 'checkpoint_steps': list(C.CHECKPOINT_STEPS)})
    for arm in C.ARMS:
        for fold in range(C.N_FOLDS):
            for seed in seeds:
                add(arm, fold, seed)
    for arm in C.ARMS:
        for seed in seeds:
            add(arm, 'full', seed)
    return specs

def write_specs(repo_root: Path, out_path: Path) -> list[dict]:
    specs = build_specs(repo_root)
    state.atomic_write_json(Path(out_path), specs)
    return specs

def load_specs(path: Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding='utf-8'))

def resolve(specs: list[dict], index: int | None=None, run_id: str | None=None) -> dict:
    if (index is None) == (run_id is None):
        raise SystemExit('resolve: pass exactly one of index / run_id')
    if index is not None:
        if not 0 <= index < len(specs):
            raise SystemExit(f'resolve: index {index} out of range 0..{len(specs) - 1}')
        spec = specs[index]
        assert spec['index'] == index, (spec['index'], index)
        return spec
    for spec in specs:
        if spec['run_id'] == run_id:
            return spec
    raise SystemExit(f'resolve: unknown run_id {run_id!r}')
