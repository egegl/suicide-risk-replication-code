import json
import time
from pathlib import Path
from synth import state
from . import constants as C
from . import data

def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
_CONFIG_KEYS = ('BASE_MODEL', 'SEED_BASE', 'MAX_SEQ_LEN', 'MICRO_BATCH', 'GRAD_ACCUM', 'TOTAL_STEPS', 'CHECKPOINT_STEPS', 'LR', 'LR_SCHEDULE', 'MIN_LR_RATIO', 'WARMUP_STEPS', 'WEIGHT_DECAY', 'MAX_GRAD_NORM', 'ADAM_BETAS', 'PRECISION', 'ATTN_IMPL', 'ATTN_IMPL_FALLBACK', 'GRAD_CKPT', 'LOAD_IN_4BIT', 'BNB_4BIT_QUANT_TYPE', 'BNB_4BIT_DOUBLE_QUANT', 'LORA_R', 'LORA_ALPHA', 'LORA_DROPOUT', 'LORA_TARGETS')

def constants_snapshot() -> dict:
    return json.loads(json.dumps({k: getattr(C, k) for k in _CONFIG_KEYS}))

def train_config(spec: dict, model_override: str | None=None, steps_override: int | None=None, checkpoint_steps_override: list[int] | None=None) -> dict:
    return json.loads(json.dumps({'spec': spec, 'constants': constants_snapshot(), 'overrides': {'model': model_override, 'steps': steps_override, 'checkpoint_steps': checkpoint_steps_override}}))
_LOGITS_KW: dict = {'kw': None}
_FALLBACK_MAX_PARAMS = 2000000000

def _forward_logits(model, input_ids, attention_mask, keep: int):
    if _LOGITS_KW['kw'] is None:
        width = input_ids.shape[1]
        for kw in ('logits_to_keep', 'num_logits_to_keep'):
            try:
                logits = model(input_ids=input_ids, attention_mask=attention_mask, **{kw: keep}).logits
            except TypeError:
                continue
            if logits.shape[1] == keep and keep < width:
                _LOGITS_KW['kw'] = kw
                return logits
        _LOGITS_KW['kw'] = ''
        if sum((p.numel() for p in model.parameters())) > _FALLBACK_MAX_PARAMS:
            raise RuntimeError('neither logits_to_keep nor num_logits_to_keep is honored by this transformers/model combo; refusing the full-sequence-logits fallback on a large model (review fix A-4) — pin a transformers version that supports logits windowing for this architecture')
    if _LOGITS_KW['kw']:
        return model(input_ids=input_ids, attention_mask=attention_mask, **{_LOGITS_KW['kw']: keep}).logits
    return model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -keep:]

def _micro_loss(model, batch: dict, mean_weight: float, device: str):
    import torch.nn.functional as F
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    weights = batch['weight'].to(device)
    lab_mask = labels != -100
    first = int(lab_mask.int().argmax(dim=1).min().item())
    assert first >= 1, 'example with no prompt prefix — labels start at position 0'
    width = labels.shape[1]
    keep = width - first + 1
    logits = _forward_logits(model, input_ids, attention_mask, keep)
    pred = logits[:, :-1].float()
    lab = labels[:, first:]
    ce = F.cross_entropy(pred.transpose(1, 2), lab, ignore_index=-100, reduction='none')
    n_tok = lab_mask[:, first:].sum(-1).clamp(min=1)
    per_ex = ce.sum(-1) / n_tok
    return data.weighted_reduce(per_ex, weights, mean_weight)
_NON_TEXT_MARKERS = ('vision', 'multi_modal', 'multimodal', 'audio')

def resolve_lora_targets(model) -> list[str]:
    import torch.nn as nn
    targets: list[str] = []
    seen_leaves: set[str] = set()
    for name, mod in model.named_modules():
        leaf = name.rsplit('.', 1)[-1]
        if leaf not in C.LORA_TARGETS:
            continue
        if any((m in name.lower() for m in _NON_TEXT_MARKERS)):
            continue
        if isinstance(mod, nn.Linear):
            targets.append(name)
        elif isinstance(getattr(mod, 'linear', None), nn.Linear):
            targets.append(f'{name}.linear')
        else:
            raise RuntimeError(f'LoRA target {name} is {type(mod).__name__}, neither nn.Linear nor a container with an inner nn.Linear named `linear` — extend resolve_lora_targets for this architecture')
        seen_leaves.add(leaf)
    missing = set(C.LORA_TARGETS) - seen_leaves
    if missing:
        raise RuntimeError(f'LoRA targets {sorted(missing)} matched no text-stack module — constants.LORA_TARGETS is stale for this architecture')
    return targets

def _normalize_adapter_dir(save_dir: Path) -> None:
    import json as _json
    from safetensors.torch import load_file, save_file
    save_dir = Path(save_dir)
    cfg_path = save_dir / 'adapter_config.json'
    cfg = _json.loads(cfg_path.read_text(encoding='utf-8'))
    cfg['target_modules'] = sorted((t[:-len('.linear')] if t.endswith('.linear') else t for t in cfg['target_modules']))
    cfg_path.write_text(_json.dumps(cfg, indent=2), encoding='utf-8')
    st_path = save_dir / 'adapter_model.safetensors'
    tensors = load_file(str(st_path))
    save_file({k.replace('.linear.lora_', '.lora_'): v for k, v in tensors.items()}, str(st_path))

def _load_model(base_model: str, quantize: bool):
    import torch
    from transformers import AutoModelForCausalLM
    kwargs: dict = {}
    if quantize:
        from transformers import BitsAndBytesConfig
        kwargs['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type=C.BNB_4BIT_QUANT_TYPE, bnb_4bit_use_double_quant=C.BNB_4BIT_DOUBLE_QUANT, bnb_4bit_compute_dtype=torch.bfloat16)
        kwargs['device_map'] = {'': 0}
    last_err = None
    for impl in (C.ATTN_IMPL, C.ATTN_IMPL_FALLBACK):
        try:
            model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, attn_implementation=impl, **kwargs)
            return (model, impl)
        except (ValueError, ImportError) as err:
            last_err = err
    raise last_err

def train_run(repo_root: Path, spec: dict, runs_dir: Path, model_override: str | None=None, steps_override: int | None=None, checkpoint_steps_override: list[int] | None=None) -> dict:
    t0 = time.time()
    repo_root, run_id = (Path(repo_root), spec['run_id'])
    run_dir = Path(runs_dir) / run_id
    records = data.load_arm_records(repo_root, spec)
    train_path = repo_root / spec['train_file']
    actual_sha = state.sha256_file(train_path)
    if actual_sha != spec['train_file_sha']:
        raise SystemExit(f"{run_id}: {spec['train_file']} sha mismatch (spec {spec['train_file_sha'][:12]}, disk {actual_sha[:12]}) — the file changed since specs were built; rebuild specs")
    data.assert_fold_safe(records, spec['fold'], repo_root / 'data/processed/folds.parquet')
    base_model = model_override or spec['base_model']
    total_steps = steps_override or spec['total_steps']
    ckpt_steps = sorted(checkpoint_steps_override or spec['checkpoint_steps'])
    assert ckpt_steps and ckpt_steps[-1] <= total_steps, (ckpt_steps, total_steps)
    adapters_dir = run_dir / 'adapters'
    if adapters_dir.exists():
        import shutil
        _log(f'{run_id}: clearing stale {adapters_dir} from a previous attempt')
        shutil.rmtree(adapters_dir)
    mean_weight = spec['mean_weight'] if spec['mean_weight'] is not None else 1.0
    seed_effective = C.SEED_BASE + spec['seed']
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, get_scheduler
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    data.assert_template_prefix(tokenizer)
    encoded = [data.encode_example(tokenizer, r) for r in records]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    torch.manual_seed(seed_effective)
    quantize = C.LOAD_IN_4BIT and torch.cuda.is_available()
    _log(f"{run_id}: loading {base_model} ({('nf4 4-bit, bf16 compute' if quantize else 'bf16')}; attn {C.ATTN_IMPL} -> {C.ATTN_IMPL_FALLBACK} fallback) — a few minutes on the 31B")
    model, attn_impl = _load_model(base_model, quantize)
    if quantize:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=C.GRAD_CKPT, gradient_checkpointing_kwargs={'use_reentrant': False})
    lora_targets = resolve_lora_targets(model)
    n_wrapped = sum((t.endswith('.linear') for t in lora_targets))
    _log(f'{run_id}: LoRA targets: {len(lora_targets)} text-stack modules resolved from {len(C.LORA_TARGETS)} leaf names ({n_wrapped} via wrapped inner .linear)')
    model = get_peft_model(model, LoraConfig(r=C.LORA_R, lora_alpha=C.LORA_ALPHA, lora_dropout=C.LORA_DROPOUT, target_modules=lora_targets, bias='none', task_type='CAUSAL_LM'))
    if C.GRAD_CKPT and (not quantize):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not quantize:
        model.to(device)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=C.LR, betas=tuple(C.ADAM_BETAS), weight_decay=C.WEIGHT_DECAY)
    scheduler = get_scheduler(C.LR_SCHEDULE, optimizer, num_warmup_steps=C.WARMUP_STEPS, num_training_steps=total_steps, scheduler_specific_kwargs={'min_lr_rate': C.MIN_LR_RATIO})
    order = data.shuffled_order(len(encoded), seed_effective, run_id)
    log_path = run_dir / 'train_log.jsonl'
    log_rows: list[dict] = []
    saved_steps: list[int] = []
    _log(f'{run_id}: training on {device} (attn_impl={attn_impl}, {sum((p.numel() for p in params)):,} trainable params, {len(records)} records, {total_steps} steps x {C.GRAD_ACCUM} accum, ckpts {ckpt_steps})')
    t_loop = time.time()
    for step in range(1, total_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(C.GRAD_ACCUM):
            batch = data.collate([encoded[next(order)] for _ in range(C.MICRO_BATCH)], pad_id)
            loss = _micro_loss(model, batch, mean_weight, device) / C.GRAD_ACCUM
            loss.backward()
            step_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(params, C.MAX_GRAD_NORM)
        lr_now = optimizer.param_groups[0]['lr']
        optimizer.step()
        scheduler.step()
        log_rows.append({'step': step, 'loss': step_loss, 'lr': lr_now})
        rate = (time.time() - t_loop) / step
        mem = f' mem {torch.cuda.max_memory_allocated() / 2 ** 30:.1f}GiB' if device == 'cuda' else ''
        _log(f'{run_id}: step {step}/{total_steps} loss {step_loss:.4f} lr {lr_now:.2e} {rate:.1f}s/step eta {rate * (total_steps - step) / 60:.0f}m{mem}')
        if step in ckpt_steps:
            ckpt_dir = run_dir / 'adapters' / f'step{step}'
            model.save_pretrained(str(ckpt_dir))
            if n_wrapped:
                _normalize_adapter_dir(ckpt_dir)
            saved_steps.append(step)
            state.atomic_write_jsonl(log_path, log_rows)
            _log(f'{run_id}: saved adapter checkpoint step{step}')
    state.atomic_write_jsonl(log_path, log_rows)
    manifest = state.Manifest(run_dir)
    manifest.record('train', train_config(spec, model_override, steps_override, checkpoint_steps_override), inputs=[train_path], outputs=[log_path])
    return {'run_id': run_id, 'arm': spec['arm'], 'fold': spec['fold'], 'seed': spec['seed'], 'base_model': base_model, 'attn_impl': attn_impl, 'device': device, 'quantized_4bit': quantize, 'n_records': len(records), 'n_trainable_params': sum((p.numel() for p in params)), 'mean_weight': mean_weight, 'steps': total_steps, 'checkpoints_saved': saved_steps, 'final_loss': log_rows[-1]['loss'], 'train_seconds': round(time.time() - t0, 1)}
