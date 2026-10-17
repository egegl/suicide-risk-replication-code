"""Claude writers on the Message Batches API: the pilot (Haiku 5.5, Sonnet 5.5) and E2 (Sonnet 5.5).

Same prompts.jsonl as the local writers. Both models run with thinking off and
effort pinned to medium, so the only difference is the model:
Haiku takes {"type": "disabled"}; Sonnet 5.5 rejects that and takes
{"type": "between_tools"} (no extended thinking). Neither accepts sampling
parameters, so none are sent.

    PYTHONPATH=src .venv/bin/python scripts/pkgc_claude.py submit            # v2: Sonnet, 2 drafts
    PYTHONPATH=src .venv/bin/python scripts/pkgc_claude.py collect           # until ended
    PYTHONPATH=src .venv/bin/python scripts/pkgc_claude.py --version pkgc-v1 collect   # v1 record
    PYTHONPATH=src .venv/bin/python scripts/pkgc_claude.py submit-full       # E2: all folds, 1 draft
    PYTHONPATH=src .venv/bin/python scripts/pkgc_claude.py collect-full

Run `scripts/e2_data.py prepare` first. Reads ANTHROPIC_API_KEY or CLAUDE_API_KEY from the
environment or .env (never printed). The batch costs real money (about $15 for E2).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import anthropic

from synth import pkgc, state

ARMS = {
    "haiku": {"model": "claude-haiku-5-5", "thinking": {"type": "disabled"},
              "batch_usd_per_mtok": (0.05, 0.25)},
    "sonnet": {"model": "claude-sonnet-5-5", "thinking": {"type": "between_tools"},
               "batch_usd_per_mtok": (1.00, 5.00)},
}
EFFORT = "medium"
MAX_TOKENS = 1024
# Fail-closed worst case (every request at MAX_TOKENS). v1 both arms: ~$1.9;
# v2 Sonnet with 2 drafts per case: ~$3.6 worst case, ~$1.3 expected.
SPEND_CEILING_USD = {"pkgc-v1": 2.50, "pkgc-v2": 4.00}


def _state(version: str) -> Path:
    return pkgc.ROOTS[version] / "claude_batches.json"


def _client() -> anthropic.Anthropic:
    if not os.environ.get("ANTHROPIC_API_KEY") and Path(".env").exists():
        for line in Path(".env").read_text().splitlines():
            if line.startswith(("ANTHROPIC_API_KEY=", "CLAUDE_API_KEY=")):
                os.environ["ANTHROPIC_API_KEY"] = line.split("=", 1)[1].strip().strip("\"'")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set (environment or .env)")
    # The direct API, not any proxy base URL inherited from the shell.
    return anthropic.Anthropic(base_url="https://api.anthropic.com")


def _params(arm: str, prompt: dict) -> dict:
    cfg = ARMS[arm]
    return {"model": cfg["model"], "max_tokens": MAX_TOKENS, "system": prompt["system"],
            "messages": [{"role": "user", "content": prompt["user"]}],
            "thinking": cfg["thinking"], "output_config": {"effort": EFFORT}}


def _worst_case_usd(arm: str, prompts: list[dict], client, drafts: int) -> float:
    p_in, p_out = ARMS[arm]["batch_usd_per_mtok"]
    sample = prompts[:: max(1, len(prompts) // 20)]
    toks = [client.messages.count_tokens(
        model=ARMS[arm]["model"], system=p["system"],
        messages=[{"role": "user", "content": p["user"]}]).input_tokens for p in sample]
    mean_in = sum(toks) / len(toks)
    return drafts * len(prompts) * (mean_in * p_in + MAX_TOKENS * p_out) / 1e6


def cmd_submit(args) -> None:
    root, STATE = pkgc.ROOTS[args.version], _state(args.version)
    if STATE.exists():
        sys.exit(f"{STATE} exists: batches already submitted; run collect")
    manifest = json.loads((root / "manifest.json").read_text())
    prompts = state.read_jsonl(root / "prompts.jsonl")
    assert state.sha256_file(root / "prompts.jsonl") == manifest["prompts_sha256"]
    client = _client()
    ceiling = {arm: round(_worst_case_usd(arm, prompts, client, args.drafts), 4)
               for arm in args.arms}
    print("worst-case spend:", ceiling)
    if sum(ceiling.values()) > SPEND_CEILING_USD[args.version]:
        sys.exit(f"worst case ${sum(ceiling.values()):.2f} exceeds ceiling "
                 f"${SPEND_CEILING_USD[args.version]}")
    record = {"prompts_sha256": manifest["prompts_sha256"], "effort": EFFORT,
              "max_tokens": MAX_TOKENS, "drafts": args.drafts,
              "worst_case_usd": ceiling, "batches": {}}
    for arm in args.arms:
        ids = ([(p["case_id"], p) for p in prompts] if args.drafts == 1 else
               [(f"{p['case_id']}-d{k}", p) for p in prompts for k in range(args.drafts)])
        batch = client.messages.batches.create(requests=[
            {"custom_id": cid, "params": _params(arm, p)} for cid, p in ids])
        record["batches"][arm] = {"id": batch.id, "model": ARMS[arm]["model"],
                                  "thinking": ARMS[arm]["thinking"]}
        print(arm, batch.id, batch.processing_status)
        state.atomic_write_json(STATE, record)   # persist after each create


def _result_row(res, arm: str, info: dict) -> dict:
    row = {"case_id": res.custom_id, "arm": arm, "model": info["model"]}
    if res.result.type != "succeeded":
        row.update(status=f"batch_{res.result.type}", text="")
        return row
    msg = res.result.message
    row["text"] = "".join(b.text for b in msg.content if b.type == "text")
    row["stop_reason"] = msg.stop_reason
    row["status"] = {"end_turn": "ok", "refusal": "refusal",
                     "max_tokens": "max_tokens"}.get(msg.stop_reason, msg.stop_reason)
    if msg.stop_reason == "refusal" and msg.stop_details:
        row["refusal_category"] = msg.stop_details.category
    row["model_echo"] = msg.model
    row["usage"] = {"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens}
    return row


def _usd(arm: str, row: dict) -> float:
    u = row.get("usage") or {}
    p_in, p_out = ARMS[arm]["batch_usd_per_mtok"]
    return (u.get("input_tokens", 0) * p_in + u.get("output_tokens", 0) * p_out) / 1e6


def cmd_collect(args) -> None:
    root, STATE = pkgc.ROOTS[args.version], _state(args.version)
    record = json.loads(STATE.read_text())
    client = _client()
    spend = 0.0
    for arm, info in record["batches"].items():
        batch = client.messages.batches.retrieve(info["id"])
        print(arm, batch.processing_status, dict(batch.request_counts))
        if batch.processing_status != "ended":
            continue
        rows = [_result_row(res, arm, info) for res in client.messages.batches.results(info["id"])]
        spend += sum(_usd(arm, r) for r in rows)
        rows.sort(key=lambda r: r["case_id"])
        if record.get("drafts", 1) > 1:
            state.atomic_write_jsonl(root / "drafts" / f"{arm}.jsonl", rows)
            rows = pkgc.select_draft_rows(rows, root)
        state.atomic_write_jsonl(root / "outputs" / f"{arm}.jsonl", rows)
        print(f"  wrote {len(rows)} rows")
    record["billed_usd_estimate"] = round(spend, 4)
    state.atomic_write_json(STATE, record)
    print(f"estimated spend so far: ${spend:.4f}")


# ---------------------------------------------------------------- full dose (E2)
# One batch over every fold's prompts, one draft per case (custom id "K3C00042-d0"),
# kept through the same pkgc.select_draft_rows path as the Gemma arm's draft 0.
FULL_MAX_TOKENS = 800            # pilot v2 Sonnet outputs: p99 627, max 644 tokens
# 247 tokens measured on the 600 v2 pilot drafts (1.78 tokens per target word),
# scaled to the full-dose cases' mean target length (142.2 vs 138.8 words).
PROJECTED_MEAN_OUT_TOKENS = 253
OUT_MARGIN = 1.1                 # projected $15.33; x1.1 on output gives $16.22
FULL_SPEND_CEILING_USD = 16.50


def _full_state(version: str) -> Path:
    return pkgc.FULL_ROOT / version / "claude_batches.json"


def cmd_submit_full(args) -> None:
    STATE = _full_state(args.version)
    if STATE.exists():
        sys.exit(f"{STATE} exists: batch already submitted; run collect-full")
    prompts, shas = [], {}
    for k in args.folds:
        root = pkgc.full_root(args.version, k)
        manifest = json.loads((root / "manifest.json").read_text())
        assert state.sha256_file(root / "prompts.jsonl") == manifest["prompts_sha256"]
        shas[k] = manifest["prompts_sha256"]
        prompts += state.read_jsonl(root / "prompts.jsonl")
    client = _client()
    p_in, p_out = ARMS[args.arm]["batch_usd_per_mtok"]
    sample = prompts[:: max(1, len(prompts) // 25)]
    mean_in = sum(client.messages.count_tokens(
        model=ARMS[args.arm]["model"], system=p["system"],
        messages=[{"role": "user", "content": p["user"]}]).input_tokens for p in sample) / len(sample)
    expected = len(prompts) * (mean_in * p_in + OUT_MARGIN * PROJECTED_MEAN_OUT_TOKENS * p_out) / 1e6
    worst = len(prompts) * (mean_in * p_in + FULL_MAX_TOKENS * p_out) / 1e6
    print(f"{len(prompts)} requests; mean input {mean_in:.0f} tokens; "
          f"expected (output x{OUT_MARGIN}) ${expected:.2f}; worst case ${worst:.2f}")
    if expected > FULL_SPEND_CEILING_USD:
        sys.exit(f"expected ${expected:.2f} exceeds ceiling ${FULL_SPEND_CEILING_USD}")
    global MAX_TOKENS
    MAX_TOKENS = FULL_MAX_TOKENS
    batch = client.messages.batches.create(requests=[
        {"custom_id": f"{p['case_id']}-d0", "params": _params(args.arm, p)} for p in prompts])
    state.atomic_write_json(STATE, {
        "arm": args.arm, "model": ARMS[args.arm]["model"], "thinking": ARMS[args.arm]["thinking"],
        "effort": EFFORT, "max_tokens": FULL_MAX_TOKENS, "folds": args.folds,
        "prompts_sha256": shas, "n_requests": len(prompts),
        "expected_usd": round(expected, 2), "worst_case_usd": round(worst, 2), "batch_id": batch.id})
    print(args.arm, batch.id, batch.processing_status)


def cmd_collect_full(args) -> None:
    STATE = _full_state(args.version)
    record = json.loads(STATE.read_text())
    client = _client()
    batch = client.messages.batches.retrieve(record["batch_id"])
    print(batch.processing_status, dict(batch.request_counts))
    if batch.processing_status != "ended":
        return
    arm, info = record["arm"], {"model": record["model"]}
    rows = [_result_row(res, arm, info) for res in client.messages.batches.results(record["batch_id"])]
    by_fold: dict[int, list[dict]] = {}
    for r in rows:
        by_fold.setdefault(int(r["case_id"][1:r["case_id"].index("C")]), []).append(r)
    for k, fold_rows in sorted(by_fold.items()):
        root = pkgc.full_root(args.version, k)
        fold_rows.sort(key=lambda r: r["case_id"])
        state.atomic_write_jsonl(root / "drafts" / f"{arm}.jsonl", fold_rows)
        kept = pkgc.select_draft_rows(fold_rows, root)
        state.atomic_write_jsonl(root / "outputs" / f"{arm}.jsonl", kept)
        print(f"fold{k}: {len(kept)} rows", {s: sum(r["status"] == s for r in kept)
                                             for s in {r["status"] for r in kept}})
    record["billed_usd_estimate"] = round(sum(_usd(arm, r) for r in rows), 4)
    state.atomic_write_json(STATE, record)
    print(f"estimated spend: ${record['billed_usd_estimate']:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", choices=sorted(pkgc.ROOTS), default=pkgc.PROMPT_VERSION)
    sub = ap.add_subparsers(required=True)
    s1 = sub.add_parser("submit")
    s1.add_argument("--arms", nargs="+", choices=sorted(ARMS), default=["sonnet"])
    s1.add_argument("--drafts", type=int, default=pkgc.DRAFTS)
    s1.set_defaults(fn=cmd_submit)
    sub.add_parser("collect").set_defaults(fn=cmd_collect)
    s2 = sub.add_parser("submit-full")
    s2.add_argument("--arm", choices=sorted(ARMS), default="sonnet")
    s2.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    s2.set_defaults(fn=cmd_submit_full)
    sub.add_parser("collect-full").set_defaults(fn=cmd_collect_full)
    args = ap.parse_args()
    args.fn(args)
