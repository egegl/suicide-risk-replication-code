"""Local writers and the fold labeler (GPU jobs only).

generate: local writers on the same prompts.jsonl as the Claude arms, thinking
off, one seeded sample per draft. Branches: Qwen3.8-27B vs Hemmingway-1 (its
writing fine-tune; Qwen non-thinking sampling for both) and Gemma-4-12B-it vs
Gemma-4-31B-it (Gemma's recommended sampling for both). Qwen3.8-Flash-Next
(FP8) joins the Qwen branch with Qwen sampling.

label: blind labels for every arm's posts from the real-data fold-0 model
(real-only run baseline-f0-s1, Gemma-4-31B LoRA at step 160, trained on folds 1-4 only;
full dose: baseline-f<k>-s1 at its selected checkpoint for held-out fold k).
Each post is placed in its host timeline and read exactly as the model reads a
real post, greedy, with the eval retry temperatures. Gemma-4-31B-it as a
writer is the labeler's base model; that arm is not blind in that sense.

    python scripts/pkgc_local.py [--version pkgc-v1] generate --arm qwen [--shard 0 --shards 4]
    python scripts/pkgc_local.py [--version pkgc-v1] label --arms sonnet qwen
    python scripts/pkgc_local.py --version pkgc-v2 --fold 3 generate --arm gemma31   # full dose
    python scripts/pkgc_local.py --version pkgc-v2 --fold 3 first-draft --arm gemma31
    python scripts/pkgc_local.py --version pkgc-v2 --fold 3 label --arms sonnet gemma31

Writers are read from $MODELS_DIR/<name> (default models/). E2 keeps one draft per
case for both writers: Gemma generates the v2 pilot's two drafts per case, and
`first-draft` rebuilds outputs/<arm>.jsonl from draft 0 alone, through the same
draft selection as the Sonnet batch (the two-draft selection is kept as
outputs/<arm>_2draft.jsonl and is unused).

Generation is sharded for array jobs: each shard writes parts/<arm>.<i>of<n>.jsonl
and skips itself if that part exists (overflow preemption requeues from the
top); whichever shard completes the set merges the parts, writes drafts/ for
v2, and keeps one draft per case with pkgc.select_draft_rows. Tensor
parallelism follows the GPUs the job was given.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from synth import pkgc, state

_LABS = os.environ.get("MODELS_DIR", "models")
MODELS = {
    "qwen": f"{_LABS}/Qwen3.8-27B",
    "hemmingway": f"{_LABS}/Hemmingway-1",
    "gemma12": f"{_LABS}/gemma-4-12B-it",
    "gemma31": f"{_LABS}/gemma-4-31B-it",
    # 180B MoE (512 experts, 10 active), architecture qwen4_exp: needs vLLM >= 0.30
    # (.venv-qwen4exp) and expert parallelism for the FP8 checkpoint (vLLM recipe).
    "qwenflash": f"{_LABS}/Qwen3.8-Flash-Next-FP8",
}
_QWEN_SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}    # Qwen non-thinking
_GEMMA_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}  # Gemma model card
SAMPLING = {"qwen": _QWEN_SAMPLING, "hemmingway": _QWEN_SAMPLING, "qwenflash": _QWEN_SAMPLING,
            "gemma12": _GEMMA_SAMPLING, "gemma31": _GEMMA_SAMPLING}
# Per-arm engine settings beyond the shared ones (recipes.vllm.ai, H100 FP8 entry).
ENGINE_EXTRA = {"qwenflash": {"enable_expert_parallel": True, "gpu_memory_utilization": 0.85,
                              "kernel_config": {"moe_backend": "triton"},
                              "limit_mm_per_prompt": {"image": 0, "video": 0}}}
MAX_TOKENS = 1024
TEMPLATE_KWARGS = {"enable_thinking": False}
LABELER_RUN = Path("runs31/baseline-f0-s1")   # pilot; full dose uses baseline-f<k>-s1
LABELER_STEP = 160
_THINK = re.compile(r"<think>.*?</think>", re.S)


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def _gpus() -> int:
    return len([d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if d])


def _jobs(prompts: list[dict], drafts: int) -> list[tuple[str, dict, int]]:
    """(row id, prompt, draft k). v1 keeps its single seed per case; v2 drafts
    get ids "C0001-d1" and a seed per draft."""
    if drafts == 1:
        return [(p["case_id"], p, -1) for p in prompts]
    return [(f"{p['case_id']}-d{k}", p, k) for p in prompts for k in range(drafts)]


def _case_seed(arm: str, p: dict, k: int) -> int:
    return _seed(pkgc.SEED, arm, p["case_id"]) if k < 0 else _seed(pkgc.SEED, arm, p["case_id"], k)


def _chat(tok, p: dict) -> str:
    return tok.apply_chat_template(
        [{"role": "system", "content": p["system"]}, {"role": "user", "content": p["user"]}],
        tokenize=False, add_generation_prompt=True, **TEMPLATE_KWARGS)


def _generate_hf(arm: str, jobs: list) -> list[dict]:
    """transformers path for Hemmingway-1: it ships the text-only
    Qwen3_5ForCausalLM, which vLLM 0.26 does not register (only the multimodal
    Qwen3_5ForConditionalGeneration). Same chat template and sampling as the
    vLLM arm; seeds are per batch instead of per draft."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path, samp = MODELS[arm], SAMPLING[arm]
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="auto")
    model.eval()
    eos_cfg = model.generation_config.eos_token_id
    eos = set(eos_cfg if isinstance(eos_cfg, list) else [eos_cfg])
    rows, bs = [], 16
    for start in range(0, len(jobs), bs):
        chunk = jobs[start:start + bs]
        enc = tok([_chat(tok, p) for _, p, _ in chunk], return_tensors="pt",
                  padding=True).to(model.device)
        torch.manual_seed(_seed(pkgc.SEED, arm, "batch", chunk[0][0]))
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=True, max_new_tokens=MAX_TOKENS,
                                 pad_token_id=tok.pad_token_id, **samp)
        for (rid, _, _), ids, mask in zip(chunk, gen[:, enc["input_ids"].shape[1]:],
                                          enc["attention_mask"]):
            ids = ids.tolist()
            ended = next((i for i, t in enumerate(ids) if t in eos), None)
            out_ids = ids if ended is None else ids[:ended]
            rows.append({"case_id": rid, "arm": arm, "model": path, "engine": "hf",
                         "text": _THINK.sub("", tok.decode(out_ids, skip_special_tokens=True)).strip(),
                         "status": "ok" if ended is not None else "length",
                         "usage": {"input_tokens": int(mask.sum()), "output_tokens": len(out_ids)}})
        print(f"{start + len(chunk)}/{len(jobs)}", flush=True)
    return rows


def _generate_vllm(arm: str, jobs: list) -> list[dict]:
    from vllm import LLM, SamplingParams
    kwargs = {"dtype": "bfloat16", "max_model_len": 8192, "gpu_memory_utilization": 0.90,
              "enforce_eager": True, "disable_custom_all_reduce": True}   # serving pins
    kwargs.update(ENGINE_EXTRA.get(arm, {}))
    llm = LLM(model=MODELS[arm], tensor_parallel_size=_gpus(), **kwargs)
    tok = llm.get_tokenizer()
    params = [SamplingParams(seed=_case_seed(arm, p, k), max_tokens=MAX_TOKENS, **SAMPLING[arm])
              for _, p, k in jobs]
    outs = llm.generate([_chat(tok, p) for _, p, _ in jobs], params)
    rows = []
    for (rid, _, _), o in zip(jobs, outs):
        c = o.outputs[0]
        rows.append({"case_id": rid, "arm": arm, "model": MODELS[arm],
                     "text": _THINK.sub("", c.text).strip(),
                     "status": "ok" if c.finish_reason == "stop" else c.finish_reason,
                     "usage": {"input_tokens": len(o.prompt_token_ids),
                               "output_tokens": len(c.token_ids)}})
    return rows


def _merge(arm: str, root: Path, shards: int, drafts: int) -> None:
    parts = [root / "parts" / f"{arm}.{i}of{shards}.jsonl" for i in range(shards)]
    if not all(p.exists() for p in parts):
        print(f"{arm}: {sum(p.exists() for p in parts)}/{shards} parts done; merge later")
        return
    rows = sorted((r for p in parts for r in state.read_jsonl(p)), key=lambda r: r["case_id"])
    if drafts > 1:
        state.atomic_write_jsonl(root / "drafts" / f"{arm}.jsonl", rows)
        rows = pkgc.select_draft_rows(rows, root)
    state.atomic_write_jsonl(root / "outputs" / f"{arm}.jsonl", rows)
    print(arm, "merged:", {s: sum(r["status"] == s for r in rows) for s in {r["status"] for r in rows}})


def _root(args) -> Path:
    return pkgc.ROOTS[args.version] if args.fold is None else pkgc.full_root(args.version, args.fold)


def _labeler(args) -> tuple[Path, int]:
    """Fold k's posts are read by the real-only model that never saw fold k:
    baseline-f<k>-s1 at its selected checkpoint (the pilot keeps f0 step 160)."""
    if args.fold is None:
        return LABELER_RUN, LABELER_STEP
    run = Path(f"runs31/baseline-f{args.fold}-s1")
    return run, int(json.loads((run / "selected.json").read_text())["step"])


def cmd_generate(args) -> None:
    root = _root(args)
    drafts = 1 if args.version == "pkgc-v1" else pkgc.DRAFTS
    part = root / "parts" / f"{args.arm}.{args.shard}of{args.shards}.jsonl"
    if part.exists():
        print(f"{part} exists; skipping generation")
    else:
        jobs = _jobs(state.read_jsonl(root / "prompts.jsonl"), drafts)[args.shard::args.shards]
        gen = _generate_hf if args.arm == "hemmingway" else _generate_vllm
        state.atomic_write_jsonl(part, gen(args.arm, jobs))
    _merge(args.arm, root, args.shards, drafts)


def cmd_label(args) -> None:
    from train import constants as C, data as data_mod, generate, probe as probe_mod
    processed = Path("data/processed")
    root = _root(args)
    labeler_run, labeler_step = _labeler(args)
    cases = {c["case_id"]: c for c in state.read_jsonl(root / "cases.jsonl")}
    timelines = pkgc._timelines(processed, pkgc.POOL if args.fold is None else args.fold)
    jobs = []   # (arm, case_id, post, input_text)
    for arm in args.arms:
        path = root / "outputs" / f"{arm}.jsonl"
        if (root / "labels" / f"{arm}.jsonl").exists():
            print(f"skip {arm}: already labeled")
            continue
        if not path.exists():
            print(f"skip {arm}: no outputs yet")
            continue
        for o in state.read_jsonl(path):
            post = pkgc.clean_post(o.get("text"))
            if o.get("status") == "ok" and post:
                case = cases[o["case_id"]]
                jobs.append((arm, o["case_id"], post, pkgc.labeler_input(case, post, timelines)))
    if not jobs:
        return
    engine = generate.Engine(C.BASE_MODEL, max_model_len=C.VLLM_MAX_MODEL_LEN,
                             enable_guided=C.GUIDED_DECODING, tensor_parallel=_gpus(),
                             # custom all-reduce failed in CUDA-graph capture on 2x
                             # RTX 6000 Ada; these serving pins avoid that path
                             enforce_eager=True, disable_custom_all_reduce=True)
    tok = engine.get_tokenizer()
    probe = probe_mod.build_probe(tok)
    probe_mod.probe_self_test(tok)
    # Keep the model's own truncation; long synthetic posts are cut like real ones.
    prompt_ids = [data_mod.build_prompt_ids(tok, j[3]) for j in jobs]
    data_mod.assert_prompt_budget(prompt_ids, [f"{j[0]}:{j[1]}" for j in jobs],
                                  engine.max_model_len, C.MAX_NEW_TOKENS)
    adapter = labeler_run / "adapters" / f"step{labeler_step}"
    outs = engine.generate(prompt_ids, adapter, temperature=C.GEN_TEMPERATURE)
    recs = [generate._process_output(o, j[2], tok, probe) for o, j in zip(outs, jobs)]
    for attempt, temp in enumerate(C.RETRY_TEMPS):
        pending = [i for i, r in enumerate(recs) if r is None]
        if not pending:
            break
        seeds = [generate.retry_seed("pkgc-label", f"{jobs[i][0]}:{jobs[i][1]}", attempt)
                 for i in pending]
        outs2 = engine.generate([prompt_ids[i] for i in pending], adapter,
                                seeds=seeds, temperature=temp)
        for i, o in zip(pending, outs2):
            recs[i] = generate._process_output(o, jobs[i][2], tok, probe)
    by_arm: dict[str, list[dict]] = {}
    for (arm, case_id, _, _), r in zip(jobs, recs):
        row = {"case_id": case_id, "labeler": f"{labeler_run.name}@step{labeler_step}"}
        # spans: the labeler's evidence phrases, the full-dose training targets' evidence
        row.update({"risk": None, "factors": [], "spans": [], "parse_failure": True} if r is None else
                   {"risk": r["risk"], "factors": r["factors"], "spans": r["spans"],
                    "p_true": r["p_true"]})
        by_arm.setdefault(arm, []).append(row)
    for arm, rows in by_arm.items():
        state.atomic_write_jsonl(root / "labels" / f"{arm}.jsonl", rows)
        print(arm, len(rows), "labeled,", sum(r.get("parse_failure", False) for r in rows), "failed")


def cmd_first_draft(args) -> None:
    """outputs/<arm>.jsonl from draft 0 alone (E2: one draft per case for both writers)."""
    root = _root(args)
    rows = [r for r in state.read_jsonl(root / "drafts" / f"{args.arm}.jsonl")
            if r["case_id"].endswith("-d0")]
    out = root / "outputs" / f"{args.arm}.jsonl"
    if out.exists() and not (root / "outputs" / f"{args.arm}_2draft.jsonl").exists():
        out.rename(root / "outputs" / f"{args.arm}_2draft.jsonl")
    kept = pkgc.select_draft_rows(rows, root)
    state.atomic_write_jsonl(out, kept)
    print(args.arm, len(kept), "rows from draft 0")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=sorted(pkgc.ROOTS), default=pkgc.PROMPT_VERSION)
    ap.add_argument("--fold", type=int, choices=range(5), default=None,
                    help="full-dose fold k (data/synthetic_pkgc/full/<version>/fold<k>); omit for the pilot")
    sub = ap.add_subparsers(required=True)
    g = sub.add_parser("generate")
    g.add_argument("--arm", choices=sorted(MODELS), required=True)
    g.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    g.add_argument("--shards", type=int, default=int(os.environ.get("N_SHARDS", 1)))
    g.set_defaults(fn=cmd_generate)
    fd = sub.add_parser("first-draft")
    fd.add_argument("--arm", choices=sorted(MODELS), required=True)
    fd.set_defaults(fn=cmd_first_draft)
    lab = sub.add_parser("label")
    lab.add_argument("--arms", nargs="+", choices=pkgc.ARMS, default=list(pkgc.ARMS))
    lab.set_defaults(fn=cmd_label)
    args = ap.parse_args()
    args.fn(args)
