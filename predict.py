#!/usr/bin/env python3
"""Run the adopted two-seed R9 system on unseen organizer examples."""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from prep import constants as PC
from prep import context
from train import constants as C
from train import data as data_mod
from train import ensemble
from train import generate
from train import probe as probe_mod
import submission

INPUT_COLUMNS = ["row_id", "anon_user_id", "post_id", "post"]
DEFAULT_ADAPTER_1 = ROOT / "adapters" / "seed1-step160"
DEFAULT_ADAPTER_2 = ROOT / "adapters" / "seed2-step120"
DEFAULT_THRESHOLDS = ROOT / "artifacts" / "r9_thresholds.json"


def load_examples(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        frame = pd.read_excel(path, sheet_name="Sheet1", engine="openpyxl")
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError("input must be an .xlsx or .csv file")
    if list(frame.columns) != INPUT_COLUMNS:
        raise ValueError(
            f"expected columns {INPUT_COLUMNS}, found {list(frame.columns)}")
    frame = frame.copy()
    for column in ("row_id", "anon_user_id", "post"):
        if frame[column].isna().any():
            raise ValueError(f"{column} contains missing values")
        frame[column] = frame[column].astype(str)
    if frame["row_id"].duplicated().any():
        raise ValueError("row_id values must be unique")
    if (frame["post"].str.len() == 0).any():
        raise ValueError("post values must be non-empty")
    try:
        frame["post_id"] = frame["post_id"].astype("int64")
    except (TypeError, ValueError) as error:
        raise ValueError("post_id values must be integers") from error
    for user_id, group in frame.groupby("anon_user_id", sort=False):
        ids = sorted(group["post_id"].tolist())
        if ids != list(range(len(ids))):
            raise ValueError(
                f"post_id must be consecutive from 0 for user {user_id!r}")
    return frame


def build_inference_rows(examples: pd.DataFrame) -> list[dict]:
    records = []
    for _, user_frame in examples.groupby("anon_user_id", sort=False):
        user_frame = user_frame.sort_values("post_id")
        posts = user_frame["post"].tolist()
        for position, row in enumerate(user_frame.itertuples()):
            input_text, _, is_truncated = context.build_input(posts, position)
            records.append({"row_id": row.row_id, "input": input_text,
                            "is_truncated": is_truncated})
    return records


def load_thresholds(path: str | Path) -> tuple[dict, dict[str, float]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, dict) or set(thresholds) != set(PC.FACTORS_24):
        raise ValueError("threshold artifact must contain exactly the 24 factors")
    values = {factor: float(thresholds[factor]) for factor in PC.FACTORS_24}
    if any(not 0.0 <= value <= 1.0 for value in values.values()):
        raise ValueError("all thresholds must lie in [0, 1]")
    members = payload.get("members")
    expected = [("baseline-full-s1", 160), ("baseline-full-s2", 120)]
    actual = [(member.get("run_id"), member.get("step")) for member in members or []]
    if actual != expected:
        raise ValueError(f"threshold artifact has unexpected R9 members: {actual}")
    return payload, values


def validate_adapter(path: str | Path, label: str) -> Path:
    path = Path(path).resolve()
    required = [path / "adapter_config.json", path / "adapter_model.safetensors"]
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError(f"{label} adapter is incomplete; missing {missing}")
    return path


def _member_rows(engine, adapter: Path, run_id: str, rows: list[dict],
                 raw_posts: dict[str, str], tokenizer, factor_probe) -> tuple[list[dict], dict]:
    row_ids = [row["row_id"] for row in rows]
    prompt_ids = [data_mod.build_prompt_ids(tokenizer, row["input"]) for row in rows]
    data_mod.assert_prompt_budget(prompt_ids, row_ids, engine.max_model_len)
    outputs = engine.generate(prompt_ids, adapter, guided=C.GUIDED_DECODING)
    records = [generate._process_output(output, raw_posts[row_id], tokenizer,
                                        factor_probe)
               for output, row_id in zip(outputs, row_ids)]
    retry_count = 0
    for attempt, temperature in enumerate(C.RETRY_TEMPS):
        pending = [index for index, record in enumerate(records)
                   if record is None or record["flags"].get("coupling_violation")]
        if not pending:
            break
        retry_count += len(pending)
        retry_outputs = engine.generate(
            [prompt_ids[index] for index in pending], adapter,
            seeds=[generate.retry_seed(run_id, row_ids[index], attempt)
                   for index in pending], temperature=temperature,
            guided=C.GUIDED_DECODING)
        for index, output in zip(pending, retry_outputs):
            candidate = generate._process_output(
                output, raw_posts[row_ids[index]], tokenizer, factor_probe)
            if candidate is not None and (
                    records[index] is None
                    or not candidate["flags"].get("coupling_violation")):
                records[index] = candidate
    fallback_count = 0
    member_rows = []
    for row_id, record in zip(row_ids, records):
        if record is None or record["flags"].get("coupling_violation"):
            fallback_count += 1
            member_rows.append({"row_id": row_id, "fold": -1,
                                "risk": C.FALLBACK_PRED["risk"], "spans": [],
                                "factors": [], "p_true": {},
                                "risk_logprobs": []})
        else:
            member_rows.append({"row_id": row_id, "fold": -1,
                                "risk": record["risk"],
                                "spans": list(record["spans"]), "factors": [],
                                "p_true": record["p_true"],
                                "risk_logprobs": record["risk_logprobs"]})
    return member_rows, {"retries": retry_count, "fallbacks": fallback_count}


def run(args) -> dict:
    model = args.model or os.environ.get("GEMMA_MODEL", "")
    if not model:
        raise ValueError("set GEMMA_MODEL or pass --model")
    examples = load_examples(args.input)
    rows = build_inference_rows(examples)
    _, thresholds = load_thresholds(args.thresholds)
    adapter1 = validate_adapter(args.adapter_seed1, "seed 1 step 160")
    adapter2 = validate_adapter(args.adapter_seed2, "seed 2 step 120")
    engine = generate.Engine(
        model, max_model_len=C.VLLM_MAX_MODEL_LEN,
        gpu_mem_util=args.gpu_memory_utilization,
        tensor_parallel=args.tensor_parallel_size,
        enable_guided=C.GUIDED_DECODING)
    tokenizer = engine.get_tokenizer()
    data_mod.assert_template_prefix(tokenizer)
    factor_probe = probe_mod.build_probe(tokenizer)
    probe_mod.probe_self_test(tokenizer)
    raw_posts = dict(zip(examples["row_id"], examples["post"]))
    member1, stats1 = _member_rows(
        engine, adapter1, "baseline-full-s1", rows, raw_posts, tokenizer,
        factor_probe)
    member2, stats2 = _member_rows(
        engine, adapter2, "baseline-full-s2", rows, raw_posts, tokenizer,
        factor_probe)
    merged = ensemble.merge_rows(member1, member2)
    for record in merged:
        record["factors"] = (
            [factor for factor in PC.FACTORS_24
             if record["p_true"][factor] >= thresholds[factor]]
            if record.get("p_true") else [])
    predictions = {
        record["row_id"]: {"risk": record["risk"],
                           "spans": list(record["spans"]),
                           "factors": list(record["factors"])}
        for record in merged
    }
    output = submission.write_submission(predictions, examples, args.output)
    return {"system": "R9", "rows": len(examples), "output": str(output),
            "members": {"seed1_step160": stats1, "seed2_step120": stats2}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                        help="organizer .xlsx or .csv with row_id, anon_user_id, post_id, post")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", help="local Gemma snapshot; defaults to GEMMA_MODEL")
    parser.add_argument("--adapter-seed1", default=str(DEFAULT_ADAPTER_1))
    parser.add_argument("--adapter-seed2", default=str(DEFAULT_ADAPTER_2))
    parser.add_argument("--thresholds", default=str(DEFAULT_THRESHOLDS))
    parser.add_argument("--tensor-parallel-size", type=int,
                        default=C.VLLM_TENSOR_PARALLEL)
    parser.add_argument("--gpu-memory-utilization", type=float,
                        default=C.VLLM_GPU_MEM_UTIL)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
