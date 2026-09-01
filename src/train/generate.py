import hashlib
import re
import time
from pathlib import Path

def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
from prep import constants as PC
from scorer import load_gold
from synth import constants as SC
from synth import state
from . import constants as C
from . import data as data_mod
from . import oof as oof_mod
from . import parse as parse_mod
from . import probe as probe_mod

def build_v2_regex() -> str:
    jstr = '"(?:[^"\\\\\\x00-\\x1f]|\\\\.)*"'
    evidence = '\\[(?:{s}(?:, {s})*)?\\]'.format(s=jstr)
    factors = '\\{' + ', '.join((f'"{re.escape(k)}": (?:true|false)' for k in PC.FACTORS_24)) + '\\}'
    risk = '"(?:' + '|'.join((re.escape(c) for c in PC.RISK_CLASSES)) + ')"'
    return '\\{"evidence": ' + evidence + ', "factors": ' + factors + ', "risk": ' + risk + '\\}'

def _structured_kwargs(regex: str) -> dict:
    try:
        from vllm.sampling_params import StructuredOutputsParams
        return {'structured_outputs': StructuredOutputsParams(regex=regex)}
    except ImportError:
        from vllm.sampling_params import GuidedDecodingParams
        return {'guided_decoding': GuidedDecodingParams(regex=regex)}

def _plain_output(req_out) -> dict:
    comp = req_out.outputs[0]
    steps = []
    for step in comp.logprobs or []:
        steps.append({int(tid): float(getattr(lp, 'logprob', lp)) for tid, lp in step.items()})
    return {'text': comp.text, 'token_ids': [int(t) for t in comp.token_ids], 'logprobs': steps}

class Engine:

    def __init__(self, base_model: str, max_model_len: int=C.VLLM_MAX_MODEL_LEN, gpu_mem_util: float=C.VLLM_GPU_MEM_UTIL, max_loras: int=C.VLLM_MAX_LORAS, enable_guided: bool=False, quantization: str | None=C.VLLM_QUANTIZATION, tensor_parallel: int=C.VLLM_TENSOR_PARALLEL):
        from vllm import LLM
        self.base_model = base_model
        self.max_model_len = max_model_len
        self.enable_guided = enable_guided
        self.quantization = quantization
        self._lora_ids: dict[str, int] = {}
        quant_kwargs = {'quantization': quantization} if quantization else {}
        self.llm = LLM(model=base_model, dtype='bfloat16', enable_lora=True, max_lora_rank=C.LORA_R, max_loras=max_loras, max_model_len=max_model_len, tensor_parallel_size=tensor_parallel, gpu_memory_utilization=gpu_mem_util, **quant_kwargs)

    def get_tokenizer(self):
        return self.llm.get_tokenizer()

    def _lora_id(self, adapter_path: Path) -> int:
        key = str(adapter_path)
        if key not in self._lora_ids:
            cand = int(hashlib.sha256(key.encode('utf-8')).hexdigest(), 16) % (2 ** 31 - 2) + 1
            taken = set(self._lora_ids.values())
            while cand in taken:
                cand = cand % (2 ** 31 - 2) + 1
            self._lora_ids[key] = cand
        return self._lora_ids[key]

    def generate(self, prompt_token_ids: list[list[int]], adapter_path: Path | None, seeds: list[int] | None=None, temperature: float=C.GEN_TEMPERATURE, guided: bool=False) -> list[dict]:
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest
        if guided and (not self.enable_guided):
            raise RuntimeError('guided=True on an Engine built with enable_guided=False')
        sp_kwargs: dict = {'temperature': temperature, 'max_tokens': C.MAX_NEW_TOKENS, 'logprobs': C.LOGPROBS_TOPK}
        if guided:
            sp_kwargs.update(_structured_kwargs(build_v2_regex()))
        if seeds is None:
            params = SamplingParams(**sp_kwargs)
        else:
            assert len(seeds) == len(prompt_token_ids), (len(seeds), len(prompt_token_ids))
            params = [SamplingParams(seed=int(s), **sp_kwargs) for s in seeds]
        lora_req = None
        if adapter_path is not None:
            p = Path(adapter_path)
            name = f'{p.parent.parent.name}-{p.name}'
            lora_req = LoRARequest(name, self._lora_id(p), str(p))
        prompts = [{'prompt_token_ids': list(ids)} for ids in prompt_token_ids]
        outs = self.llm.generate(prompts, params, lora_request=lora_req, use_tqdm=False)
        return [_plain_output(o) for o in outs]

def present_steps(adapters_dir: Path) -> list[int]:
    if not adapters_dir.is_dir():
        return []
    out = []
    for p in adapters_dir.iterdir():
        m = re.fullmatch('step(\\d+)', p.name)
        if m and p.is_dir():
            out.append(int(m.group(1)))
    return sorted(out)

def expected_steps(manifest, spec: dict) -> list[int] | None:
    rec = manifest.data.get('train')
    if rec is None:
        return None
    cfg = rec['config']
    ck = (cfg.get('overrides') or {}).get('checkpoint_steps') or cfg['spec']['checkpoint_steps']
    return sorted((int(s) for s in ck))

def adapter_problem(run_dir: Path, spec: dict) -> str | None:
    run_dir = Path(run_dir)
    present = present_steps(run_dir / 'adapters')
    expect = expected_steps(state.Manifest(run_dir), spec)
    if expect is None:
        return f"no recorded training run; adapters on disk: {present or 'none'}"
    if present != expect:
        return f'adapters on disk {present} do not match checkpoints {expect}; retrain with --force'
    return None

def eval_config(spec: dict, steps: list[int], model_override: str | None) -> dict:
    return {'model': model_override or spec['base_model'], 'model_override': model_override, 'quantization': C.VLLM_QUANTIZATION, 'tensor_parallel': C.VLLM_TENSOR_PARALLEL, 'guided': C.GUIDED_DECODING, 'temperature': C.GEN_TEMPERATURE, 'retry_temps': list(C.RETRY_TEMPS), 'max_new_tokens': C.MAX_NEW_TOKENS, 'logprobs_topk': C.LOGPROBS_TOPK, 'ee_primary': C.EE_PRIMARY, 'steps': list(steps), 'spec': {k: spec[k] for k in ('run_id', 'arm', 'fold', 'seed', 'train_file', 'train_file_sha')}}

def eval_inputs(repo_root: Path, spec: dict, adapters_dir: Path, steps: list[int]) -> list[Path]:
    repo_root = Path(repo_root)
    paths = [repo_root / spec['val_file'], repo_root / 'data/processed/gold.jsonl', repo_root / 'data/processed/posts.parquet']
    for step in steps:
        paths.extend(sorted((p for p in (adapters_dir / f'step{step}').iterdir() if p.is_file())))
    return paths

def _raw_post_map(repo_root: Path) -> dict[str, str]:
    import pandas as pd
    df = pd.read_parquet(repo_root / 'data/processed/posts.parquet', columns=['row_id', 'post_raw'])
    return dict(zip(df.row_id, df.post_raw))

def retry_seed(run_id: str, row_id: str, attempt: int) -> int:
    h = hashlib.sha256(f'{run_id}|{row_id}|{attempt}'.encode('utf-8')).hexdigest()
    return int(h[:8], 16)

def _process_output(out: dict, raw_post: str, tokenizer, probe: dict) -> dict | None:
    pred = parse_mod.parse_generation(out['text'], raw_post)
    if pred is None:
        return None
    dropped = pred.pop('align_dropped', 0)
    pred, flags = parse_mod.repair_and_flag(pred)
    if dropped:
        flags['align_dropped'] = dropped
    p_true, pflags = probe_mod.factor_probs(out['token_ids'], out['logprobs'], tokenizer, probe)
    flags.update(pflags)
    rl = probe_mod.risk_logprobs(out['token_ids'], out['logprobs'], tokenizer)
    return {'risk': pred['risk'], 'spans': pred['spans'], 'factors': pred['factors'], 'p_true': p_true, 'risk_logprobs': rl, 'flags': flags, 'text': out['text']}

def _fallback_record(last_text: str) -> dict:
    return {'risk': C.FALLBACK_PRED['risk'], 'spans': list(C.FALLBACK_PRED['spans']), 'factors': list(C.FALLBACK_PRED['factors']), 'p_true': {f: 0.0 for f in PC.FACTORS_24}, 'risk_logprobs': [], 'flags': {'parse_failure': True}, 'text': last_text}

def _rare_class_f1(preds: dict, gold: dict) -> dict[str, float]:
    out = {}
    ids = [rid for rid in gold if rid in preds]
    for f in SC.RARE_FACTORS:
        tp = fp = fn = 0
        for rid in ids:
            t = f in gold[rid]['factors']
            p = f in preds[rid]['factors']
            tp += t and p
            fp += p and (not t)
            fn += t and (not p)
        denom = 2 * tp + fp + fn
        out[f] = 2 * tp / denom if denom else 0.0
    return out

def _score_fields(result: dict, preds: dict, gold: dict) -> dict:
    if 'composite_ee1' in result:
        fields = {k: result[k] for k in ('composite_ee1', 'composite_skip', 'risk_wf1', 'phrase_f1_ee1', 'phrase_f1_skip', 'macro_f1', 'n_posts')}
        rare = result.get('per_rare_class_f1')
    else:
        ee1, skip = (result['ee1'], result['skip'])
        fields = {'composite_ee1': ee1['composite'], 'composite_skip': skip['composite'], 'risk_wf1': ee1['risk_wf1'], 'phrase_f1_ee1': ee1['phrase_f1'], 'phrase_f1_skip': skip['phrase_f1'], 'macro_f1': ee1['macro_f1'], 'n_posts': ee1['n_posts']}
        rare = result.get('per_rare_class_f1')
    fields['per_rare_class_f1'] = rare if rare is not None else _rare_class_f1(preds, gold)
    return fields

def _argmax_row(metrics_rows: list[dict], key: str) -> dict:
    best = metrics_rows[0]
    for r in metrics_rows[1:]:
        if r[key] > best[key]:
            best = r
    return best

def eval_run(repo_root: Path, spec: dict, runs_dir: Path, engine, model_override: str | None=None) -> dict:
    repo_root, runs_dir = (Path(repo_root), Path(runs_dir))
    run_id = spec['run_id']
    assert spec.get('val_file'), f'{run_id}: no val_file — full-train runs are not evaluated'
    fold = spec['fold']
    run_dir = runs_dir / run_id
    adapters_dir = run_dir / 'adapters'
    problem = adapter_problem(run_dir, spec)
    if problem:
        raise FileNotFoundError(f'{run_id}: {problem}')
    steps = present_steps(adapters_dir)
    val_rows = state.read_jsonl(repo_root / spec['val_file'])
    row_ids = [r['row_id'] for r in val_rows]
    gold = load_gold(repo_root / 'data/processed/gold.jsonl', fold=fold)
    raw_by_row = _raw_post_map(repo_root)
    tokenizer = engine.get_tokenizer()
    probe = probe_mod.build_probe(tokenizer)
    probe_mod.probe_self_test(tokenizer)
    prompt_ids = [data_mod.build_prompt_ids(tokenizer, r['input']) for r in val_rows]
    data_mod.assert_prompt_budget(prompt_ids, row_ids, engine.max_model_len)
    n = len(val_rows)
    metrics_rows: list[dict] = []
    oof_rows_by_step: dict[int, list[dict]] = {}
    _log(f'{run_id}: evaluating {len(steps)} checkpoints {steps} on fold {fold} ({n} val rows)')
    for step in steps:
        adapter = adapters_dir / f'step{step}'
        t0 = time.time()
        outs = engine.generate(prompt_ids, adapter, seeds=None, temperature=C.GEN_TEMPERATURE, guided=C.GUIDED_DECODING)
        records: list[dict | None] = []
        last_text: list[str] = []
        for i, out in enumerate(outs):
            records.append(_process_output(out, raw_by_row[row_ids[i]], tokenizer, probe))
            last_text.append(out['text'])
        n_retries = 0
        for attempt, temp in enumerate(C.RETRY_TEMPS):
            pending = [i for i in range(n) if records[i] is None or records[i]['flags'].get('coupling_violation')]
            if not pending:
                break
            n_retries += len(pending)
            seeds = [retry_seed(run_id, row_ids[i], attempt) for i in pending]
            outs2 = engine.generate([prompt_ids[i] for i in pending], adapter, seeds=seeds, temperature=temp, guided=C.GUIDED_DECODING)
            for i, out in zip(pending, outs2):
                last_text[i] = out['text']
                rec = _process_output(out, raw_by_row[row_ids[i]], tokenizer, probe)
                if rec is None:
                    continue
                if records[i] is None or not rec['flags'].get('coupling_violation'):
                    rec['flags']['retried'] = attempt + 1
                    records[i] = rec
        parse_failures = 0
        for i in range(n):
            if records[i] is None:
                records[i] = _fallback_record(last_text[i])
                parse_failures += 1
        gen_seconds = time.time() - t0
        align_dropped = sum((r['flags'].get('align_dropped', 0) for r in records))
        coupling_violations = sum((bool(r['flags'].get('coupling_violation')) for r in records))
        probe_low_conf = sum((r['flags'].get('probe_low_conf', 0) for r in records))
        probe_key_missing = sum((r['flags'].get('probe_key_missing', 0) for r in records))
        preds = {row_ids[i]: {'risk': records[i]['risk'], 'spans': list(records[i]['spans']), 'factors': list(records[i]['factors'])} for i in range(n)}
        fields = _score_fields(oof_mod.score_fold(preds, fold, repo_root), preds, gold)
        rare = fields.pop('per_rare_class_f1')
        metrics_rows.append({'run_id': run_id, 'step': step, 'composite_ee1': fields['composite_ee1'], 'composite_skip': fields['composite_skip'], 'risk_wf1': fields['risk_wf1'], 'phrase_f1_ee1': fields['phrase_f1_ee1'], 'phrase_f1_skip': fields['phrase_f1_skip'], 'macro_f1': fields['macro_f1'], 'n_posts': fields['n_posts'], 'parse_failures': parse_failures, 'retries': n_retries, 'align_dropped': align_dropped, 'coupling_violations': coupling_violations, 'probe_low_conf': probe_low_conf, 'probe_key_missing': probe_key_missing, 'per_rare_class_f1': rare, 'gen_seconds': round(gen_seconds, 2)})
        oof_rows_by_step[step] = [{'row_id': row_ids[i], 'fold': fold, 'step': step, 'risk': records[i]['risk'], 'spans': list(records[i]['spans']), 'factors': list(records[i]['factors']), 'p_true': records[i]['p_true'], 'risk_logprobs': records[i]['risk_logprobs'], 'flags': records[i]['flags']} for i in range(n)]
        state.atomic_write_jsonl(run_dir / 'eval' / f'step{step}' / 'preds.jsonl', [{**oof_rows_by_step[step][i], 'text': records[i]['text']} for i in range(n)])
        m = metrics_rows[-1]
        _log(f"{run_id}: step{step} composite_ee1 {m['composite_ee1']:.4f} skip {m['composite_skip']:.4f} parse_failures {parse_failures} retries {n_retries} probe_low_conf {probe_low_conf} probe_key_missing {probe_key_missing} ({gen_seconds:.0f}s)")
    metrics_path = run_dir / 'eval' / 'metrics.jsonl'
    state.atomic_write_jsonl(metrics_path, metrics_rows)
    best = _argmax_row(metrics_rows, 'composite_ee1')
    best_skip = _argmax_row(metrics_rows, 'composite_skip')
    skip_step = best_skip['step'] if best_skip['step'] != best['step'] else None
    selected = {'run_id': run_id, 'step': best['step'], 'composite_ee1': best['composite_ee1'], 'composite_skip': best['composite_skip'], 'skip_step': skip_step, 'parse_fail_rate': best['parse_failures'] / n, 'coupling_violation_rate': best['coupling_violations'] / n}
    selected_path = run_dir / 'selected.json'
    state.atomic_write_json(selected_path, selected)
    oof_path = run_dir / 'oof.jsonl'
    state.atomic_write_jsonl(oof_path, oof_rows_by_step[best['step']])
    outputs = [metrics_path, selected_path, oof_path]
    if skip_step is not None:
        skip_path = run_dir / 'oof_skip.jsonl'
        state.atomic_write_jsonl(skip_path, oof_rows_by_step[skip_step])
        outputs.append(skip_path)
    manifest = state.Manifest(run_dir)
    manifest.record('eval', eval_config(spec, steps, model_override), eval_inputs(repo_root, spec, adapters_dir, steps), outputs)
    return {**selected, 'steps_evaluated': steps, 'gen_seconds': {r['step']: r['gen_seconds'] for r in metrics_rows}}
