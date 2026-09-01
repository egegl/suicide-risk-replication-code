import random
from pathlib import Path
import pandas as pd
from synth import state
from . import constants as C

def load_arm_records(repo_root: Path, spec: dict) -> list[dict]:
    path = Path(repo_root) / spec['train_file']
    if spec['train_file_sha'] == 'pending' or not path.exists():
        raise SystemExit(f"{spec['run_id']}: train file {spec['train_file']!r} is not available (sha={spec['train_file_sha']!r}, exists={path.exists()}). Run `generate_synthetic.py assemble` (realism-gated) to build data/synthetic/arms/, then rebuild specs.")
    return state.read_jsonl(path)

def assert_fold_safe(records: list[dict], fold, folds_parquet_path: Path) -> None:
    if fold == 'full':
        return
    df = pd.read_parquet(folds_parquet_path)
    fold_by_row = dict(zip(df.row_id, df.fold.astype(int)))
    for r in records:
        if 'synth' in r:
            for ex in r['synth']['exemplar_row_ids']:
                assert fold_by_row.get(ex) != fold, (r['row_id'], ex, fold)
        else:
            assert r['fold'] != fold, (r['row_id'], r['fold'], fold)
_SENT_A_USER = 'sentinel user turn for suffix derivation'
_SENT_A_TARGET = '{"evidence": [], "risk": "Indicator"}'
_SENT_B_USER = 'a different sentinel user turn, for template verification'
_SENT_B_TARGET = '{"evidence": ["some phrase"], "factors": {"hopelessness": true}, "risk": "Ideation"}'
_SUFFIX_CACHE: dict[int, tuple[object, list[int], list[int]]] = {}

def _ids(rendered) -> list[int]:
    if isinstance(rendered, list):
        return rendered
    return list(rendered['input_ids'])

def build_prompt_ids(tokenizer, input_text: str) -> list[int]:
    return _ids(tokenizer.apply_chat_template([{'role': 'user', 'content': input_text}], add_generation_prompt=True, tokenize=True))

def _template_full(tokenizer, user_text: str, target_text: str) -> list[int]:
    return _ids(tokenizer.apply_chat_template([{'role': 'user', 'content': user_text}, {'role': 'assistant', 'content': target_text}], tokenize=True))

def _dump(tag: str, ids: list[int], tokenizer) -> str:
    try:
        decoded = repr(tokenizer.decode(ids))
    except Exception:
        decoded = '<decode unavailable>'
    return f'{tag}: n={len(ids)} ids={ids} decoded={decoded}'

def _lcp_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i

def _derive_tails(tokenizer) -> tuple[list[int], list[int]]:
    key = id(tokenizer)
    if key not in _SUFFIX_CACHE:
        full = _template_full(tokenizer, _SENT_A_USER, _SENT_A_TARGET)
        prefix = build_prompt_ids(tokenizer, _SENT_A_USER)
        target = tokenizer.encode(_SENT_A_TARGET, add_special_tokens=False)
        core = _lcp_len(full, prefix)
        if full[core:core + len(target)] != target:
            raise AssertionError('suffix_ids: chat template does not decompose as prefix + target + suffix (full render != common-core + target + suffix)\n' + _dump('full', full, tokenizer) + '\n' + _dump('prefix', prefix, tokenizer) + '\n' + _dump('target', target, tokenizer) + '\n' + _dump('core(common)', full[:core], tokenizer))
        _SUFFIX_CACHE[key] = (tokenizer, full[core + len(target):], prefix[core:])
    _, suffix, gen_tail = _SUFFIX_CACHE[key]
    return (list(suffix), list(gen_tail))

def suffix_ids(tokenizer) -> list[int]:
    return _derive_tails(tokenizer)[0]

def assert_template_prefix(tokenizer) -> None:
    full = _template_full(tokenizer, _SENT_B_USER, _SENT_B_TARGET)
    prefix = build_prompt_ids(tokenizer, _SENT_B_USER)
    target = tokenizer.encode(_SENT_B_TARGET, add_special_tokens=False)
    suffix, gen_tail = _derive_tails(tokenizer)
    problems = []
    core = prefix[:len(prefix) - len(gen_tail)] if gen_tail else prefix
    if gen_tail and prefix[len(prefix) - len(gen_tail):] != gen_tail:
        problems.append('build_prompt_ids(user) does not end with the derived generation tail — gen_tail is content-DEPENDENT, refusing to mask it')
    if full[:len(core)] != core:
        problems.append('prompt-core is NOT a token-prefix of the conversation')
    if full[len(core):] != target + suffix:
        problems.append('remainder after prompt-core != target_ids + suffix_ids')
    if problems:
        raise AssertionError('assert_template_prefix FAILED — concatenation masking would be wrong for this tokenizer:\n- ' + '\n- '.join(problems) + '\n' + _dump('full', full, tokenizer) + '\n' + _dump('prefix', prefix, tokenizer) + '\n' + _dump('target', target, tokenizer) + '\n' + _dump('suffix', suffix, tokenizer) + '\n' + _dump('gen_tail', gen_tail, tokenizer))

def encode_example(tokenizer, rec: dict) -> dict:
    prefix = build_prompt_ids(tokenizer, rec['input'])
    target = tokenizer.encode(rec['target'], add_special_tokens=False) + suffix_ids(tokenizer)
    ids = prefix + target
    assert len(ids) <= C.MAX_SEQ_LEN, f"{rec['row_id']}: {len(ids)} tokens > MAX_SEQ_LEN={C.MAX_SEQ_LEN}; never truncate — re-pin MAX_SEQ_LEN at preflight"
    return {'input_ids': ids, 'labels': [-100] * len(prefix) + target, 'weight': float(rec['weight']), 'row_id': rec['row_id']}

def assert_prompt_budget(prompt_ids: list[list[int]], row_ids: list[str], max_model_len: int, max_new_tokens: int | None=None) -> None:
    if max_new_tokens is None:
        max_new_tokens = C.MAX_NEW_TOKENS
    budget = max_model_len - max_new_tokens
    bad = [(rid, len(ids)) for rid, ids in zip(row_ids, prompt_ids) if len(ids) > budget]
    if bad:
        raise AssertionError(f"{len(bad)} prompts exceed max_model_len({max_model_len}) - MAX_NEW_TOKENS({max_new_tokens}) = {budget} tokens; never truncate — re-pin VLLM_MAX_MODEL_LEN at preflight. Offenders (row_id, n_tokens): {bad[:10]}{(' ...' if len(bad) > 10 else '')}")

def epochs(n: int, seed_effective: int, run_id: str):
    e = 0
    while True:
        idx = list(range(n))
        random.Random(f'{seed_effective}|{run_id}|epoch{e}').shuffle(idx)
        yield idx
        e += 1

def shuffled_order(n: int, seed_effective: int, run_id: str):
    for epoch in epochs(n, seed_effective, run_id):
        yield from epoch

def collate(batch: list[dict], pad_id: int) -> dict:
    import torch
    n, width = (len(batch), max((len(b['input_ids']) for b in batch)))
    input_ids = torch.full((n, width), pad_id, dtype=torch.long)
    labels = torch.full((n, width), -100, dtype=torch.long)
    attention_mask = torch.zeros((n, width), dtype=torch.long)
    for i, b in enumerate(batch):
        m = len(b['input_ids'])
        input_ids[i, :m] = torch.tensor(b['input_ids'], dtype=torch.long)
        labels[i, :m] = torch.tensor(b['labels'], dtype=torch.long)
        attention_mask[i, :m] = 1
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels, 'weight': torch.tensor([b['weight'] for b in batch], dtype=torch.float32), 'row_ids': [b['row_id'] for b in batch]}

def weighted_reduce(per_ex, weights, mean_weight: float):
    return (per_ex * weights).sum() / (len(per_ex) * mean_weight)
