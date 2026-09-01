import json
import random
from pathlib import Path
from prep import constants as PC
from prep import target as prep_target
from . import constants as SC
from . import state
FOLDS = [(str(k), f'train_fold{k}.jsonl', f'train_fold{k}.jsonl') for k in range(PC.N_FOLDS)]
FOLDS.append(('full', 'full_train.jsonl', 'train_full.jsonl'))

def _load(path: Path) -> list[dict]:
    return state.read_jsonl(path) if path.exists() else []

def _timelines(pool_records: list[dict]) -> dict:
    tls: dict = {}
    for r in sorted(pool_records, key=lambda r: (r['synth']['bundle_id'], r['row_id'])):
        tls.setdefault(r['synth']['su_id'], []).append(r)
    return tls

def _rare_positives(records: list[dict]) -> dict:
    counts = {f: 0 for f in SC.RARE_FACTORS}
    for r in records:
        for f in prep_target.true_factors(json.loads(r['target'])):
            if f in counts:
                counts[f] += 1
    return counts

def _select_synthetic(pool_records: list[dict], n_target: int) -> list[dict]:
    tls = _timelines(pool_records)
    order = list(tls)
    by_factor: dict = {f: [] for f in SC.RARE_FACTORS}
    for su in order:
        by_factor[tls[su][0]['synth']['target_factor']].append(su)
    ptr = {f: 0 for f in SC.RARE_FACTORS}
    used: set = set()
    selected: list[dict] = []
    counts = {f: 0 for f in SC.RARE_FACTORS}
    while True:
        deficit = [(counts[f], PC.FACTOR_INDEX[f], f) for f in SC.RARE_FACTORS if counts[f] < n_target]
        if not deficit:
            break
        deficit.sort()
        _, _, f = deficit[0]
        while ptr[f] < len(by_factor[f]) and by_factor[f][ptr[f]] in used:
            ptr[f] += 1
        if ptr[f] >= len(by_factor[f]):
            raise SystemExit(f'pool exhausted for {f!r}: have {counts[f]} positives, need {n_target}. Generate more (raise OVERSAMPLE) or lower the arm target.')
        su = by_factor[f][ptr[f]]
        used.add(su)
        recs = tls[su]
        selected.extend(recs)
        for k, v in _rare_positives(recs).items():
            counts[k] += v
    return selected

def _interleave(real: list[dict], synth: list[dict], seed_key: str) -> list[dict]:
    merged = real + synth
    random.Random(seed_key).shuffle(merged)
    return merged

def _assert_fold_safe(synth: list[dict], fold_key: str, fold_by_row: dict) -> None:
    if fold_key == 'full':
        return
    k = int(fold_key)
    for r in synth:
        for ex in r['synth']['exemplar_row_ids']:
            assert fold_by_row.get(ex) != k, (r['row_id'], ex, k)

def assemble_arm(arm: str, pools_dir: Path, sft_dir: Path, out_dir: Path, fold_by_row: dict, match_sizes: dict | None=None) -> dict:
    arm_dir = out_dir / arm
    manifest = {'arm': arm, 'seed': SC.SYNTH_SEED, 'prompt_version': None, 'folds': {}}
    prompt_versions: set = set()
    sizes = {}
    for fold_key, real_name, out_name in FOLDS:
        real = _load(sft_dir / real_name)
        pool = _load(pools_dir / f'pool_fold{fold_key}.jsonl')
        if arm == 'synthetic':
            synth = _select_synthetic(pool, SC.SYNTHETIC_POSITIVES_PER_FACTOR)
        elif arm == 'repeated_real':
            target = (match_sizes or {}).get(fold_key, len(real))
            synth = []
            real = _upsample(real, target, f'{SC.SYNTH_SEED}|repeated_real|{fold_key}')
        else:
            raise SystemExit(f'unknown arm {arm!r}')
        _assert_fold_safe(synth, fold_key, fold_by_row)
        fold_versions = sorted({r['synth']['prompt_version'] for r in synth})
        prompt_versions.update(fold_versions)
        if len(prompt_versions) > 1:
            raise SystemExit(f'arm {arm!r} fold {fold_key}: selected records span mixed prompt versions {sorted(prompt_versions)}; an arm must come from a single generation run. Regenerate the pools before assembling.')
        merged = _interleave(real, synth, f'{SC.SYNTH_SEED}|assemble|{arm}|{fold_key}')
        out_path = arm_dir / out_name
        state.atomic_write_jsonl(out_path, merged)
        sizes[fold_key] = len(merged)
        manifest['folds'][fold_key] = {'out': out_name, 'sha256_out': state.sha256_file(out_path), 'sha256_real_in': state.sha256_file(sft_dir / real_name) if (sft_dir / real_name).exists() else None, 'sha256_pool_in': state.sha256_file(pools_dir / f'pool_fold{fold_key}.jsonl') if (pools_dir / f'pool_fold{fold_key}.jsonl').exists() else None, 'n_real': len(real), 'n_synth': len(synth), 'synth_positives_per_rare_class': _rare_positives(synth), 'batch_ids': sorted({r['synth']['batch_id'] for r in synth}), 'models': sorted({r['synth']['model'] for r in synth}), 'target_schemas': sorted({r['synth'].get('target_schema', 'v1') for r in synth}), 'prompt_versions': fold_versions}
    if prompt_versions:
        manifest['prompt_version'] = prompt_versions.pop()
    state.atomic_write_json(arm_dir / 'manifest.json', manifest)
    manifest['_sizes'] = sizes
    return manifest

def _upsample(real: list[dict], target: int, seed_key: str) -> list[dict]:
    if target <= len(real):
        return list(real)
    out = list(real) * (target // len(real))
    rem = target - len(out)
    extra = list(real)
    random.Random(seed_key).shuffle(extra)
    return out + extra[:rem]

def _match_arm_sizes(pools_dir: Path, sft_dir: Path) -> dict:
    sizes = {}
    for fold_key, real_name, _ in FOLDS:
        real = _load(sft_dir / real_name)
        pool = _load(pools_dir / f'pool_fold{fold_key}.jsonl')
        sizes[fold_key] = len(real) + len(_select_synthetic(pool, SC.SYNTHETIC_POSITIVES_PER_FACTOR))
    return sizes

def assemble(arms: list[str], pools_dir: Path, sft_dir: Path, out_dir: Path, fold_by_row: dict) -> dict:
    requested = set(arms)
    ordered = [a for a in SC.ARMS if a in requested]
    match_sizes = None
    if 'repeated_real' in requested and SC.MATCH_ARM not in requested:
        match_sizes = _match_arm_sizes(pools_dir, sft_dir)
    manifests = {}
    for arm in ordered:
        m = assemble_arm(arm, pools_dir, sft_dir, out_dir, fold_by_row, match_sizes)
        if arm == SC.MATCH_ARM:
            match_sizes = m['_sizes']
        manifests[arm] = m
    return manifests
