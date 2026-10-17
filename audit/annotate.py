#!/usr/bin/env python3
"""Blinded annotation tool for the synthetic-data label audits.

The interface intentionally hides generator identity, target factor, and assigned
labels during annotation.  ``report`` is the only command that opens the assigned
label key, and it requires an explicit ``--unlock-key`` acknowledgement.

``serve`` is the annotation interface: a local, dependency-free NER-style web UI
where the reader highlights risk-evidence spans directly in the focal post
instead of retyping them.  It writes exactly the same annotation records as the
terminal ``annotate`` fallback.

The E2 audit packet is written by ``scripts/e2_data.py audit-packet``; each
annotator then runs, from the repository root:

  python audit/annotate.py serve --audit-dir data/synthetic_pkgc/audit/e2_v1 --annotator annotator1

and the 2-of-3 consensus and report come from ``scripts/e2_data.py
audit-consensus`` and ``audit-report``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import threading
import webbrowser
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import audit_constants as PC  # noqa: E402  (standalone export: no project repo needed)
SC = PC
prep_target = None  # only `prepare` uses it, and the frozen packet is already prepared

PROTOCOL = "synthetic-semantic-audit-v1"
UI_INTERFACE = "ner-ui-v1"
DEFAULT_AUDIT_DIR = ROOT / "audit"
DEFAULT_SYNTH_DIR = ROOT / "data" / "synthetic"
DEFAULT_ANNOTATORS = ("annotator1", "annotator2")
DEFAULT_SEED = 20260727
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
MAX_NOTES_CHARS = 4000
MAX_BODY_BYTES = 1 << 20

RISK_SHORTCUTS = {
    "i": "Indicator",
    "d": "Ideation",
    "b": "Behavior",
    "a": "Attempt",
    "?": "Unclear",
}
RISK_DEFINITIONS = {
    "Indicator": "No explicit mention of suicide.",
    "Ideation": "Explicit suicidal expression without a suicidal plan.",
    "Behavior": "Explicit suicidal expression with a suicidal plan or self-harm (a past or ongoing self-harm act counts).",
    "Attempt": "Explicit mention of the writer's own recent or past suicide attempt. Self-harm without a suicide attempt is not Attempt.",
}
ISSUE_TAGS = (
    "negation_or_hypothetical",
    "other_person_or_wrong_experiencer",
    "temporal_status_mismatch",
    "risk_protective_polarity_mismatch",
    "label_not_expressed",
    "extra_unassigned_content",
    "stereotype_or_identity_concern",
    "internally_inconsistent",
    "unnatural_or_checklist_like",
    "other",
)


class StopSession(Exception):
    """Raised for the interactive :quit command."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _stable_rank(seed: int, *parts: object) -> str:
    payload = "|".join([str(seed), *map(str, parts)]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not slug:
        raise ValueError("annotator name must contain a letter or digit")
    return slug


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_no}: invalid JSON") from error
    return out


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _pool_paths(synth_dir: Path) -> list[Path]:
    paths = sorted((synth_dir / "pools").glob("pool_fold*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"no pool_fold*.jsonl files under {synth_dir / 'pools'}")
    return paths


def load_emitted_records(synth_dir: Path) -> tuple[list[dict], list[Path]]:
    """Load every unique emitted record and recover its focal text."""
    provenance_path = synth_dir / "provenance.jsonl"
    if not provenance_path.exists():
        raise FileNotFoundError(provenance_path)
    provenance_rows = _read_jsonl(provenance_path)
    provenance = {row["row_id"]: row for row in provenance_rows}
    if len(provenance) != len(provenance_rows):
        raise ValueError("duplicate row_id in synthetic provenance")

    source_paths = [provenance_path, *_pool_paths(synth_dir)]
    records: list[dict] = []
    seen: set[str] = set()
    for path in source_paths[1:]:
        for row in _read_jsonl(path):
            row_id = row["row_id"]
            if row_id in seen:
                raise ValueError(f"duplicate emitted row_id across pools: {row_id}")
            seen.add(row_id)
            if row_id not in provenance:
                raise ValueError(f"{row_id} missing from {provenance_path}")
            prov = provenance[row_id]
            focal_pos = int(prov["focal_pos"])
            timeline = prov["timeline_texts"]
            if not 0 <= focal_pos < len(timeline):
                raise ValueError(f"{row_id}: invalid focal_pos {focal_pos}")
            target = json.loads(row["target"])
            evidence = list(target["evidence"])
            focal_post = timeline[focal_pos]
            missing_spans = [span for span in evidence if span not in focal_post]
            if missing_spans:
                raise ValueError(f"{row_id}: assigned evidence is not verbatim: {missing_spans}")
            records.append(
                {
                    "row_id": row_id,
                    "focal_post": focal_post,
                    "assigned": {
                        "risk": target["risk"],
                        "evidence": evidence,
                        "factors": prep_target.true_factors(target),
                    },
                    "metadata": {
                        "pool": prov["pool"],
                        "bundle_id": row["synth"]["bundle_id"],
                        "su_id": row["synth"]["su_id"],
                        "post_pos": row["synth"]["post_pos"],
                        "target_factor": row["synth"]["target_factor"],
                        "is_target_post": bool(row["synth"]["is_target_post"]),
                        "prompt_version": row["synth"]["prompt_version"],
                        "model": row["synth"]["model"],
                    },
                }
            )
    if seen != set(provenance):
        missing = sorted(set(provenance) - seen)
        raise ValueError(f"{len(missing)} provenance rows missing from pools, e.g. {missing[:5]}")
    return records, source_paths


def _balanced_pick(
    candidates: Iterable[dict],
    n: int,
    seed: int,
    strata,
    scope: str,
) -> list[dict]:
    """Deterministic round-robin sampling across available strata."""
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for row in candidates:
        key = tuple(strata(row))
        buckets[key].append(row)
    for key, rows in buckets.items():
        rows.sort(key=lambda row: _stable_rank(seed, scope, key, row["row_id"]))

    selected: list[dict] = []
    keys = sorted(buckets, key=lambda key: _stable_rank(seed, scope, key))
    while len(selected) < n and keys:
        next_keys = []
        for key in keys:
            if buckets[key] and len(selected) < n:
                selected.append(buckets[key].pop(0))
            if buckets[key]:
                next_keys.append(key)
        keys = next_keys
    if len(selected) != n:
        raise ValueError(f"{scope}: requested {n} rows but only {len(selected)} are available")
    return selected


def select_subset(
    records: list[dict],
    *,
    per_factor: int,
    nontarget: int,
    seed: int,
) -> list[dict]:
    """Select target-positive rows per rare factor plus non-target siblings."""
    selected: list[dict] = []
    for factor in SC.RARE_FACTORS:
        candidates = [
            row
            for row in records
            if row["metadata"]["target_factor"] == factor
            and row["metadata"]["is_target_post"]
            and factor in row["assigned"]["factors"]
        ]
        selected.extend(
            _balanced_pick(
                candidates,
                per_factor,
                seed,
                lambda row: (row["assigned"]["risk"], row["metadata"]["pool"]),
                f"target:{factor}",
            )
        )

    already = {row["row_id"] for row in selected}
    non_candidates = [
        row
        for row in records
        if not row["metadata"]["is_target_post"] and row["row_id"] not in already
    ]
    selected.extend(
        _balanced_pick(
            non_candidates,
            nontarget,
            seed,
            lambda row: (
                row["metadata"]["target_factor"],
                row["assigned"]["risk"],
            ),
            "nontarget",
        )
    )
    if len({row["row_id"] for row in selected}) != len(selected):
        raise AssertionError("subset selection produced duplicate rows")
    selected.sort(key=lambda row: _stable_rank(seed, "packet-order", row["row_id"]))
    return selected


def prepare(args: argparse.Namespace) -> int:
    audit_dir = args.audit_dir.resolve()
    manifest_path = audit_dir / "manifest.json"
    if manifest_path.exists():
        raise SystemExit(
            f"{manifest_path} already exists; the frozen sample will not be overwritten. "
            "Choose a new --audit-dir."
        )
    records, source_paths = load_emitted_records(args.synth_dir.resolve())
    selected = select_subset(
        records,
        per_factor=args.per_factor,
        nontarget=args.nontarget,
        seed=args.seed,
    )
    packet_cases = []
    key_cases = []
    for index, row in enumerate(selected, 1):
        case_id = f"SA{index:03d}"
        packet_cases.append(
            {
                "case_id": case_id,
                "focal_post": row["focal_post"],
                "post_sha256": hashlib.sha256(
                    row["focal_post"].encode("utf-8")
                ).hexdigest(),
            }
        )
        key_cases.append(
            {
                "case_id": case_id,
                "source_row_id": row["row_id"],
                "post_sha256": packet_cases[-1]["post_sha256"],
                "assigned": row["assigned"],
                "metadata": row["metadata"],
            }
        )

    packet = {
        "protocol": PROTOCOL,
        "blind": (
            "Assigned labels, target factor, generator identity, and generation metadata "
            "are intentionally absent. Do not open assigned_key.DO_NOT_OPEN.json until "
            "both independent annotations and any consensus adjudication are complete."
        ),
        "task": {
            "risk_classes": list(PC.RISK_CLASSES),
            "risk_definitions": RISK_DEFINITIONS,
            "factor_taxonomy": list(PC.FACTORS_24),
            "factor_definitions": {
                factor: SC.FACTOR_DEFINITIONS[factor] for factor in PC.FACTORS_24
            },
            "cases": packet_cases,
        },
    }
    key = {
        "protocol": PROTOCOL,
        "warning": "DO NOT OPEN BEFORE BOTH INDEPENDENT ANNOTATIONS AND ADJUDICATION.",
        "cases": key_cases,
    }
    audit_dir.mkdir(parents=True, exist_ok=False)
    _atomic_json(audit_dir / "packet.json", packet)
    _atomic_json(audit_dir / "assigned_key.DO_NOT_OPEN.json", key)
    packet_sha = _sha256(audit_dir / "packet.json")
    key_sha = _sha256(audit_dir / "assigned_key.DO_NOT_OPEN.json")

    target_counts = Counter(
        case["metadata"]["target_factor"]
        for case in key_cases
        if case["metadata"]["is_target_post"]
    )
    risk_counts = Counter(case["assigned"]["risk"] for case in key_cases)
    manifest = {
        "protocol": PROTOCOL,
        "status": "frozen_before_annotation",
        "seed": args.seed,
        "selection": {
            "per_rare_target_factor": args.per_factor,
            "non_target_siblings": args.nontarget,
            "n_cases": len(selected),
            "target_counts": dict(sorted(target_counts.items())),
            "risk_counts": {risk: risk_counts[risk] for risk in PC.RISK_CLASSES},
        },
        "annotators": [_slug(name) for name in args.annotators],
        "blinding": {
            "shown": ["case_id", "focal_post", "task definitions"],
            "hidden_until_unlock": [
                "assigned risk",
                "assigned evidence",
                "assigned factors",
                "target factor",
                "target/non-target status",
                "generator model",
                "prompt version",
                "source row identity",
            ],
        },
        "source_files": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": _sha256(path),
            }
            for path in source_paths
        ],
        "packet": {"path": "packet.json", "sha256": packet_sha},
        "assigned_key": {
            "path": "assigned_key.DO_NOT_OPEN.json",
            "sha256": key_sha,
        },
    }
    _atomic_json(manifest_path, manifest)
    print(f"Frozen {len(selected)} blinded cases in {audit_dir}")
    print(f"Target-positive: {args.per_factor} × {len(SC.RARE_FACTORS)} rare factors")
    print(f"Non-target siblings: {args.nontarget}")
    print("Do not open assigned_key.DO_NOT_OPEN.json before annotation/adjudication.")
    for annotator in manifest["annotators"]:
        print(
            "Run: python audit/annotate.py annotate "
            f"--audit-dir {audit_dir} --annotator {annotator}"
        )
    return 0


def _load_audit(audit_dir: Path) -> tuple[dict, dict]:
    manifest = _read_json(audit_dir / "manifest.json")
    packet = _read_json(audit_dir / manifest["packet"]["path"])
    if manifest["protocol"] != PROTOCOL or packet["protocol"] != PROTOCOL:
        raise ValueError("unsupported audit protocol")
    if _sha256(audit_dir / manifest["packet"]["path"]) != manifest["packet"]["sha256"]:
        raise ValueError("packet hash mismatch")
    return manifest, packet


def _annotation_path(audit_dir: Path, annotator: str) -> Path:
    return audit_dir / "annotations" / f"{_slug(annotator)}.json"


def _new_annotation(annotator: str, display_name: str, packet_sha: str) -> dict:
    return {
        "protocol": PROTOCOL,
        "annotator": _slug(annotator),
        "display_name": display_name,
        "packet_sha256": packet_sha,
        "status": "in_progress",
        "responses": {},
    }


def _prompt(prompt: str, *, allow_empty: bool = False) -> str:
    while True:
        try:
            value = input(prompt).strip()
        except (EOFError, KeyboardInterrupt) as error:
            print()
            raise StopSession from error
        if value == ":quit":
            raise StopSession
        if value or allow_empty:
            return value


def _ask_risk() -> str:
    while True:
        value = _prompt("Risk [i=Indicator, d=Ideation, b=Behavior, a=Attempt, ?=unclear]: ")
        result = RISK_SHORTCUTS.get(value.lower())
        if result:
            return result
        print("Enter i, d, b, a, or ?.")


def _ask_evidence(post: str) -> tuple[list[str], bool]:
    while True:
        value = _prompt(
            "Verbatim risk evidence spans, separated by ' || ' "
            "[none, or ? if unclear]: "
        )
        if value.lower() == "none":
            return [], False
        if value == "?":
            return [], True
        spans = [span.strip() for span in value.split("||") if span.strip()]
        if not spans:
            print("Enter at least one span, 'none', or '?'.")
            continue
        missing = [span for span in spans if span not in post]
        if missing:
            print("Every span must be copied byte-for-byte from the focal post.")
            for span in missing:
                print(f"  Not found: {span!r}")
            continue
        return list(dict.fromkeys(spans)), False


def _print_numbered(items: Iterable[str]) -> None:
    for index, item in enumerate(items, 1):
        print(f"  {index:>2}. {item}")


def _print_factor_definitions() -> None:
    for index, factor in enumerate(PC.FACTORS_24, 1):
        print(f"  {index:>2}. {factor}")
        print(f"      {SC.FACTOR_DEFINITIONS[factor]}")


def _ask_factors() -> tuple[list[str], bool]:
    while True:
        value = _prompt(
            "Factor numbers, comma-separated [none, ?, or list to show taxonomy]: "
        )
        if value.lower() == "list":
            _print_factor_definitions()
            continue
        if value.lower() == "none":
            return [], False
        if value == "?":
            return [], True
        try:
            numbers = [int(part.strip()) for part in value.split(",") if part.strip()]
        except ValueError:
            print("Use comma-separated numbers, 'none', '?', or 'list'.")
            continue
        if not numbers or any(not 1 <= number <= len(PC.FACTORS_24) for number in numbers):
            print(f"Factor numbers must be between 1 and {len(PC.FACTORS_24)}.")
            continue
        return [PC.FACTORS_24[number - 1] for number in dict.fromkeys(numbers)], False


def _ask_scale(prompt: str, low: int, high: int) -> int:
    while True:
        value = _prompt(prompt)
        try:
            number = int(value)
        except ValueError:
            number = low - 1
        if low <= number <= high:
            return number
        print(f"Enter an integer from {low} to {high}.")


def _ask_issue_tags() -> list[str]:
    while True:
        value = _prompt(
            "Issue-tag numbers, comma-separated [none, or list to show tags]: "
        )
        if value.lower() == "list":
            _print_numbered(ISSUE_TAGS)
            continue
        if value.lower() == "none":
            return []
        try:
            numbers = [int(part.strip()) for part in value.split(",") if part.strip()]
        except ValueError:
            print("Use comma-separated numbers, 'none', or 'list'.")
            continue
        if not numbers or any(not 1 <= number <= len(ISSUE_TAGS) for number in numbers):
            print(f"Issue-tag numbers must be between 1 and {len(ISSUE_TAGS)}.")
            continue
        return [ISSUE_TAGS[number - 1] for number in dict.fromkeys(numbers)]


def _collect_semantic_response(case: dict) -> dict:
    post = case["focal_post"]
    print("\n" + "=" * 78)
    print(case["case_id"])
    print("-" * 78)
    print(post)
    print("-" * 78)
    risk = _ask_risk()
    evidence, evidence_unclear = _ask_evidence(post)
    factors, factors_unclear = _ask_factors()
    naturalness = _ask_scale(
        "Naturalness [1=clearly machine-like, 5=plausibly naturally authored]: ", 1, 5
    )
    confidence = _ask_scale("Confidence [1=low, 2=medium, 3=high]: ", 1, 3)
    issues = _ask_issue_tags()
    notes = _prompt("Optional notes [Enter to skip]: ", allow_empty=True)
    return {
        "risk": risk,
        "evidence": evidence,
        "evidence_unclear": evidence_unclear,
        "factors": factors,
        "factors_unclear": factors_unclear,
        "naturalness": naturalness,
        "confidence": confidence,
        "issue_tags": issues,
        "notes": notes,
    }


def annotate(args: argparse.Namespace) -> int:
    audit_dir = args.audit_dir.resolve()
    manifest, packet = _load_audit(audit_dir)
    annotator = _slug(args.annotator)
    if manifest.get("annotators") and annotator not in manifest["annotators"]:
        raise SystemExit(
            f"{annotator!r} is not a registered annotator: {manifest['annotators']}"
        )
    path = _annotation_path(audit_dir, annotator)
    if path.exists():
        annotation = _read_json(path)
        if annotation["packet_sha256"] != manifest["packet"]["sha256"]:
            raise ValueError(f"{path}: packet hash mismatch")
    else:
        annotation = _new_annotation(
            annotator,
            args.display_name or args.annotator,
            manifest["packet"]["sha256"],
        )
    cases = packet["task"]["cases"]
    remaining = [case for case in cases if case["case_id"] not in annotation["responses"]]
    if not remaining:
        annotation["status"] = "complete"
        _atomic_json(path, annotation)
        print(f"{annotator}: all {len(cases)} cases are already complete.")
        return 0

    print(f"Blinded semantic audit, annotator: {annotation['display_name']}")
    print(f"Completed {len(cases) - len(remaining)}/{len(cases)}. Type :quit to save and exit.")
    print("Content warning: focal posts may contain explicit suicide and self-harm language.")
    print("Assigned labels, target factors, and generator metadata are hidden.")
    print("\nRisk definitions:")
    for risk in PC.RISK_CLASSES:
        print(f"  {risk}: {RISK_DEFINITIONS[risk]}")
    print("\nFactor taxonomy and definitions:")
    _print_factor_definitions()
    print("\nIssue tags:")
    _print_numbered(ISSUE_TAGS)
    try:
        for case in remaining:
            response = _collect_semantic_response(case)
            annotation["responses"][case["case_id"]] = response
            annotation["status"] = (
                "complete"
                if len(annotation["responses"]) == len(cases)
                else "in_progress"
            )
            _atomic_json(path, annotation)
            print(
                f"Saved {len(annotation['responses'])}/{len(cases)} "
                f"to {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}"
            )
    except StopSession:
        _atomic_json(path, annotation)
        print(f"Saved progress: {len(annotation['responses'])}/{len(cases)}.")
        return 0
    print(f"Annotation complete: {path}")
    return 0


def _load_annotations(
    audit_dir: Path, manifest: dict, annotators: list[str]
) -> list[dict]:
    out = []
    for name in annotators:
        path = _annotation_path(audit_dir, name)
        if not path.exists():
            raise FileNotFoundError(path)
        annotation = _read_json(path)
        if annotation["packet_sha256"] != manifest["packet"]["sha256"]:
            raise ValueError(f"{path}: packet hash mismatch")
        out.append(annotation)
    return out


def status(args: argparse.Namespace) -> int:
    audit_dir = args.audit_dir.resolve()
    manifest, packet = _load_audit(audit_dir)
    total = len(packet["task"]["cases"])
    print(f"Audit: {audit_dir}")
    print(f"Protocol: {manifest['protocol']} | frozen cases: {total}")
    for annotator in manifest.get("annotators", []):
        path = _annotation_path(audit_dir, annotator)
        if not path.exists():
            print(f"  {annotator}: 0/{total} (not started)")
            continue
        annotation = _read_json(path)
        done = len(annotation.get("responses", {}))
        print(f"  {annotator}: {done}/{total} ({annotation.get('status', 'unknown')})")
    consensus_path = audit_dir / "adjudication" / "consensus.json"
    if consensus_path.exists():
        consensus = _read_json(consensus_path)
        done = len(consensus.get("responses", {}))
        print(f"  consensus: {done}/{total} ({consensus.get('status', 'unknown')})")
    else:
        print("  consensus: not started")
    return 0


def _semantic_fields_equal(left: dict, right: dict) -> bool:
    return (
        left["risk"] == right["risk"]
        and left["evidence_unclear"] == right["evidence_unclear"]
        and set(left["evidence"]) == set(right["evidence"])
        and left["factors_unclear"] == right["factors_unclear"]
        and set(left["factors"]) == set(right["factors"])
    )


def adjudicate(args: argparse.Namespace) -> int:
    audit_dir = args.audit_dir.resolve()
    manifest, packet = _load_audit(audit_dir)
    annotator_names = [_slug(name) for name in args.annotators]
    if len(annotator_names) != 2:
        raise SystemExit("adjudication currently requires exactly two annotators")
    annotations = _load_annotations(audit_dir, manifest, annotator_names)
    total = len(packet["task"]["cases"])
    for annotation in annotations:
        if len(annotation["responses"]) != total:
            raise SystemExit(
                f"{annotation['annotator']} is incomplete "
                f"({len(annotation['responses'])}/{total})"
            )

    path = audit_dir / "adjudication" / "consensus.json"
    if path.exists():
        consensus = _read_json(path)
        if consensus["packet_sha256"] != manifest["packet"]["sha256"]:
            raise ValueError(f"{path}: packet hash mismatch")
    else:
        consensus = {
            "protocol": PROTOCOL,
            "adjudicator": args.adjudicator,
            "source_annotators": annotator_names,
            "packet_sha256": manifest["packet"]["sha256"],
            "status": "in_progress",
            "responses": {},
        }

    cases = packet["task"]["cases"]
    try:
        for case in cases:
            case_id = case["case_id"]
            if case_id in consensus["responses"]:
                continue
            left = annotations[0]["responses"][case_id]
            right = annotations[1]["responses"][case_id]
            if _semantic_fields_equal(left, right):
                consensus["responses"][case_id] = {
                    "risk": left["risk"],
                    "evidence": left["evidence"],
                    "evidence_unclear": left["evidence_unclear"],
                    "factors": left["factors"],
                    "factors_unclear": left["factors_unclear"],
                    "confidence": min(left["confidence"], right["confidence"]),
                    "notes": "Auto-consensus: independent semantic labels agree.",
                    "source": "agreement",
                }
                _atomic_json(path, consensus)
                continue

            print("\n" + "=" * 78)
            print(f"{case_id}: disagreement adjudication")
            print("-" * 78)
            print(case["focal_post"])
            for annotation, response in zip(annotations, (left, right)):
                print("-" * 78)
                print(annotation["display_name"])
                print(f"  Risk: {response['risk']}")
                print(f"  Evidence: {response['evidence'] or 'none'}")
                print(f"  Factors: {response['factors'] or 'none'}")
                print(f"  Notes: {response['notes'] or 'none'}")
            print("-" * 78)
            response = _collect_semantic_response(case)
            consensus["responses"][case_id] = {
                "risk": response["risk"],
                "evidence": response["evidence"],
                "evidence_unclear": response["evidence_unclear"],
                "factors": response["factors"],
                "factors_unclear": response["factors_unclear"],
                "confidence": response["confidence"],
                "notes": response["notes"],
                "source": "adjudicated",
            }
            _atomic_json(path, consensus)
            print(f"Saved consensus {len(consensus['responses'])}/{total}.")
    except StopSession:
        _atomic_json(path, consensus)
        print(f"Saved consensus progress: {len(consensus['responses'])}/{total}.")
        return 0
    consensus["status"] = "complete"
    _atomic_json(path, consensus)
    print(f"Consensus complete: {path}")
    return 0


def _normalize_span(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _span_match(pred: str, gold: str) -> bool:
    p = _normalize_span(pred)
    g = _normalize_span(gold)
    if not p or not g:
        return False
    return (p in g or g in p) and len(p.split()) <= 3 * len(g.split())


def phrase_f1_post(preds: list[str], golds: list[str]) -> float:
    preds = [span for span in preds if span.strip()]
    golds = [span for span in golds if span.strip()]
    if not preds and not golds:
        return 1.0
    if not preds or not golds:
        return 0.0
    edges = [
        [index for index, gold in enumerate(golds) if _span_match(pred, gold)]
        for pred in preds
    ]
    matched_gold: dict[int, int] = {}

    def augment(pred_index: int, visited: set[int]) -> bool:
        for gold_index in edges[pred_index]:
            if gold_index in visited:
                continue
            visited.add(gold_index)
            if gold_index not in matched_gold or augment(matched_gold[gold_index], visited):
                matched_gold[gold_index] = pred_index
                return True
        return False

    matches = sum(augment(index, set()) for index in range(len(preds)))
    precision = matches / len(preds)
    recall = matches / len(golds)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _safe_mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _cohen_kappa(left: list[str], right: list[str]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    labels = sorted(set(left) | set(right))
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum(
        (left_counts[label] / len(left)) * (right_counts[label] / len(right))
        for label in labels
    )
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else None
    return (observed - expected) / (1 - expected)


def _set_micro(left_sets: list[set[str]], right_sets: list[set[str]]) -> dict:
    tp = sum(len(left & right) for left, right in zip(left_sets, right_sets))
    fp = sum(len(left - right) for left, right in zip(left_sets, right_sets))
    fn = sum(len(right - left) for left, right in zip(left_sets, right_sets))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _pairwise_report(left: dict, right: dict, case_ids: list[str]) -> dict:
    risk_left, risk_right = [], []
    factor_left, factor_right = [], []
    factor_exact = []
    evidence_scores = []
    natural_left, natural_right = [], []
    for case_id in case_ids:
        a = left["responses"][case_id]
        b = right["responses"][case_id]
        if a["risk"] != "Unclear" and b["risk"] != "Unclear":
            risk_left.append(a["risk"])
            risk_right.append(b["risk"])
        if not a["factors_unclear"] and not b["factors_unclear"]:
            aset, bset = set(a["factors"]), set(b["factors"])
            factor_left.append(aset)
            factor_right.append(bset)
            factor_exact.append(aset == bset)
        if not a["evidence_unclear"] and not b["evidence_unclear"]:
            evidence_scores.append(
                (
                    phrase_f1_post(a["evidence"], b["evidence"])
                    + phrase_f1_post(b["evidence"], a["evidence"])
                )
                / 2
            )
        natural_left.append(a["naturalness"])
        natural_right.append(b["naturalness"])
    return {
        "annotators": [left["annotator"], right["annotator"]],
        "risk_n": len(risk_left),
        "risk_exact_agreement": _safe_mean(
            [float(a == b) for a, b in zip(risk_left, risk_right)]
        ),
        "risk_cohen_kappa": _cohen_kappa(risk_left, risk_right),
        "factor_n": len(factor_exact),
        "factor_exact_set_agreement": _safe_mean([float(value) for value in factor_exact]),
        "factor_micro": _set_micro(factor_left, factor_right),
        "evidence_n": len(evidence_scores),
        "evidence_mean_bidirectional_phrase_f1": _safe_mean(evidence_scores),
        "naturalness_mean": {
            left["annotator"]: _safe_mean(natural_left),
            right["annotator"]: _safe_mean(natural_right),
        },
    }


def _fidelity_report(annotation: dict, key_by_id: dict, case_ids: list[str]) -> dict:
    risk_hits: list[float] = []
    evidence_scores: list[float] = []
    assigned_factor_sets: list[set[str]] = []
    annotated_factor_sets: list[set[str]] = []
    exact_factor_sets: list[float] = []
    per_factor = {
        factor: {
            "assigned_positive": 0,
            "assigned_positive_affirmed": 0,
            "annotator_extra_positive": 0,
        }
        for factor in PC.FACTORS_24
    }
    naturalness: list[int] = []
    confidence: list[int] = []
    issue_counts: Counter = Counter()
    unclear = Counter()
    flagged_cases: list[str] = []

    for case_id in case_ids:
        response = annotation["responses"][case_id]
        assigned = key_by_id[case_id]["assigned"]
        if response["risk"] == "Unclear":
            unclear["risk"] += 1
        else:
            risk_hits.append(float(response["risk"] == assigned["risk"]))
        if response["evidence_unclear"]:
            unclear["evidence"] += 1
        else:
            evidence_scores.append(phrase_f1_post(response["evidence"], assigned["evidence"]))
        if response["factors_unclear"]:
            unclear["factors"] += 1
        else:
            assigned_set = set(assigned["factors"])
            annotated_set = set(response["factors"])
            assigned_factor_sets.append(assigned_set)
            annotated_factor_sets.append(annotated_set)
            exact_factor_sets.append(float(assigned_set == annotated_set))
            for factor in PC.FACTORS_24:
                if factor in assigned_set:
                    per_factor[factor]["assigned_positive"] += 1
                    if factor in annotated_set:
                        per_factor[factor]["assigned_positive_affirmed"] += 1
                elif factor in annotated_set:
                    per_factor[factor]["annotator_extra_positive"] += 1
        naturalness.append(response["naturalness"])
        confidence.append(response["confidence"])
        issue_counts.update(response["issue_tags"])
        if response["issue_tags"] or response["notes"]:
            flagged_cases.append(case_id)

    factor_micro = _set_micro(annotated_factor_sets, assigned_factor_sets)
    assigned_total = factor_micro["tp"] + factor_micro["fn"]
    extra_total = factor_micro["fp"]
    for values in per_factor.values():
        denominator = values["assigned_positive"]
        values["assigned_positive_affirmation_rate"] = (
            values["assigned_positive_affirmed"] / denominator if denominator else None
        )
    return {
        "annotator": annotation["annotator"],
        "n_cases": len(case_ids),
        "unclear_counts": dict(unclear),
        "risk_exact_accuracy_vs_assigned": _safe_mean(risk_hits),
        "evidence_mean_phrase_f1_vs_assigned": _safe_mean(evidence_scores),
        "factor_exact_set_rate_vs_assigned": _safe_mean(exact_factor_sets),
        "factor_micro_vs_assigned": factor_micro,
        "assigned_factor_positive_affirmation_rate": (
            factor_micro["tp"] / assigned_total if assigned_total else None
        ),
        "extra_unassigned_factor_positives_per_case": extra_total / len(case_ids),
        "naturalness_mean": _safe_mean(naturalness),
        "confidence_mean": _safe_mean(confidence),
        "issue_tag_counts": dict(sorted(issue_counts.items())),
        "flagged_case_ids": flagged_cases,
        "per_factor": per_factor,
    }


def _fmt(value: float | None, digits: int = 3) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def _markdown_report(report: dict) -> str:
    lines = [
        "# Blinded synthetic semantic subset audit",
        "",
        f"- Protocol: `{report['protocol']}`",
        f"- Frozen cases: {report['n_cases']}",
        f"- Sample seed: `{report['seed']}`",
        "- Unit: focal synthetic post; generator and assigned labels hidden during annotation",
        "- Assigned labels are evaluated as generation claims, not treated as human gold",
        "",
        "## Inter-annotator agreement",
        "",
        "| Pair | Risk exact | Risk kappa | Factor exact-set | Factor micro-F1 | Evidence agreement |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for pair in report["pairwise"]:
        lines.append(
            "| "
            + " / ".join(pair["annotators"])
            + f" | {_fmt(pair['risk_exact_agreement'])}"
            + f" | {_fmt(pair['risk_cohen_kappa'])}"
            + f" | {_fmt(pair['factor_exact_set_agreement'])}"
            + f" | {_fmt(pair['factor_micro']['f1'])}"
            + f" | {_fmt(pair['evidence_mean_bidirectional_phrase_f1'])} |"
        )
    lines.extend(
        [
            "",
            "## Agreement with assigned synthetic labels",
            "",
            "| Reader | Risk accuracy | Evidence Phrase-F1 | Factor exact-set | "
            "Assigned factor affirmation | Extra factors/case | Naturalness (1–5) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for fidelity in report["fidelity"]:
        lines.append(
            f"| {fidelity['annotator']}"
            f" | {_fmt(fidelity['risk_exact_accuracy_vs_assigned'])}"
            f" | {_fmt(fidelity['evidence_mean_phrase_f1_vs_assigned'])}"
            f" | {_fmt(fidelity['factor_exact_set_rate_vs_assigned'])}"
            f" | {_fmt(fidelity['assigned_factor_positive_affirmation_rate'])}"
            f" | {_fmt(fidelity['extra_unassigned_factor_positives_per_case'])}"
            f" | {_fmt(fidelity['naturalness_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Rare-factor affirmation",
            "",
            "Assigned-positive denominator and independent-reader affirmation rate. "
            "These are semantic-fidelity diagnostics, not model-performance estimates.",
            "",
            "| Factor | "
            + " | ".join(
                f"{fidelity['annotator']} n / rate" for fidelity in report["fidelity"]
            )
            + " |",
            "|---|" + "|".join("---:" for _ in report["fidelity"]) + "|",
        ]
    )
    for factor in SC.RARE_FACTORS:
        cells = []
        for fidelity in report["fidelity"]:
            values = fidelity["per_factor"][factor]
            cells.append(
                f"{values['assigned_positive']} / "
                f"{_fmt(values['assigned_positive_affirmation_rate'])}"
            )
        lines.append(f"| {factor} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This subset audit estimates whether independent readers can recover the "
            "programmatically assigned risk, verbatim evidence, and factor labels from the "
            "generated focal text. It does not establish clinical validity, population "
            "representativeness, fairness, or a causal mechanism for the downstream utility "
            "result. Readers knew they were auditing synthetic material, so naturalness ratings "
            "are not source-blind discrimination estimates. The closed utility decision remains "
            "unchanged.",
            "",
        ]
    )
    return "\n".join(lines)


def report(args: argparse.Namespace) -> int:
    if not args.unlock_key:
        raise SystemExit(
            "Refusing to open assigned labels. Run only after both independent passes "
            "and any adjudication, then add --unlock-key."
        )
    audit_dir = args.audit_dir.resolve()
    manifest, packet = _load_audit(audit_dir)
    annotator_names = [_slug(name) for name in (args.annotators or manifest["annotators"])]
    annotations = _load_annotations(audit_dir, manifest, annotator_names)
    case_ids = [case["case_id"] for case in packet["task"]["cases"]]
    for annotation in annotations:
        if len(annotation["responses"]) != len(case_ids):
            raise SystemExit(
                f"{annotation['annotator']} incomplete: "
                f"{len(annotation['responses'])}/{len(case_ids)}"
            )

    key_path = audit_dir / manifest["assigned_key"]["path"]
    if _sha256(key_path) != manifest["assigned_key"]["sha256"]:
        raise ValueError("assigned-key hash mismatch")
    key = _read_json(key_path)
    key_by_id = {case["case_id"]: case for case in key["cases"]}
    if set(key_by_id) != set(case_ids):
        raise ValueError("packet/key case mismatch")

    pairwise = []
    for left_index in range(len(annotations)):
        for right_index in range(left_index + 1, len(annotations)):
            pairwise.append(
                _pairwise_report(
                    annotations[left_index], annotations[right_index], case_ids
                )
            )
    fidelity = [
        _fidelity_report(annotation, key_by_id, case_ids)
        for annotation in annotations
    ]
    consensus_path = audit_dir / "adjudication" / "consensus.json"
    consensus_fidelity = None
    if consensus_path.exists():
        consensus = _read_json(consensus_path)
        if len(consensus.get("responses", {})) == len(case_ids):
            # Consensus records intentionally omit naturalness and issue tags.
            consensus_for_metrics = {
                "annotator": "consensus",
                "responses": {
                    case_id: {
                        **response,
                        "naturalness": round(
                            sum(
                                annotation["responses"][case_id]["naturalness"]
                                for annotation in annotations
                            )
                            / len(annotations)
                        ),
                        "confidence": response["confidence"],
                        "issue_tags": [],
                        "notes": response.get("notes", ""),
                    }
                    for case_id, response in consensus["responses"].items()
                },
            }
            consensus_fidelity = _fidelity_report(
                consensus_for_metrics, key_by_id, case_ids
            )

    output = {
        "protocol": PROTOCOL,
        "n_cases": len(case_ids),
        "seed": manifest["seed"],
        "annotators": annotator_names,
        "sample": manifest["selection"],
        "pairwise": pairwise,
        "fidelity": fidelity,
        "consensus_fidelity": consensus_fidelity,
        "interpretation_boundary": (
            "Assigned labels are generation claims, not human gold. Results describe this "
            "frozen subset and do not establish clinical validity, fairness, or a causal "
            "mechanism for downstream utility."
        ),
    }
    _atomic_json(audit_dir / "report.json", output)
    (audit_dir / "report.md").write_text(_markdown_report(output), encoding="utf-8")
    print(f"Wrote {audit_dir / 'report.json'}")
    print(f"Wrote {audit_dir / 'report.md'}")
    return 0


def _as_int(value: object, field: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer from {low} to {high}")
    if not low <= value <= high:
        raise ValueError(f"{field} must be an integer from {low} to {high}")
    return value


def _as_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be true or false")
    return value


def _as_choice(value: object, field: str, allowed: Iterable[str]) -> str:
    allowed = list(allowed)
    if value not in allowed:
        raise ValueError(f"{field} must be one of {allowed}")
    return str(value)


def normalize_spans(post: str, raw_spans: object) -> list[dict]:
    """Validate client offsets against the frozen post and merge overlaps.

    Offsets are the UI's transport for evidence.  They are trusted only after the
    slice they name reproduces the text the browser claims to have highlighted,
    which is what keeps stored evidence byte-verbatim.
    """
    if not isinstance(raw_spans, list):
        raise ValueError("evidence_spans must be a list")
    bounds: list[tuple[int, int]] = []
    for item in raw_spans:
        if not isinstance(item, dict):
            raise ValueError("each evidence span must be an object")
        start = _as_int(item.get("start"), "span start", 0, len(post))
        end = _as_int(item.get("end"), "span end", 0, len(post))
        if end <= start:
            raise ValueError("span end must be greater than span start")
        raw = post[start:end]
        stripped = raw.strip()
        if not stripped:
            continue
        start += len(raw) - len(raw.lstrip())
        end = start + len(stripped)
        claimed = item.get("text")
        if claimed is not None and str(claimed).strip() != stripped:
            raise ValueError(
                "span text does not match the frozen post; reload the page"
            )
        bounds.append((start, end))

    merged: list[list[int]] = []
    for start, end in sorted(bounds):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [{"start": start, "end": end, "text": post[start:end]} for start, end in merged]


def build_ui_response(case: dict, payload: object) -> dict:
    """Turn one UI submission into a record identical in shape to the CLI's."""
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")
    post = case["focal_post"]
    if payload.get("post_sha256") != case["post_sha256"]:
        raise ValueError("stale case in the browser; reload the page")

    risk = _as_choice(payload.get("risk"), "risk", [*PC.RISK_CLASSES, "Unclear"])

    evidence_status = _as_choice(
        payload.get("evidence_status"), "evidence_status", ["spans", "none", "unclear"]
    )
    spans: list[dict] = []
    if evidence_status == "spans":
        spans = normalize_spans(post, payload.get("evidence_spans"))
        if not spans:
            raise ValueError(
                "highlight at least one evidence span, or record 'no evidence'/'unclear'"
            )

    factors_status = _as_choice(
        payload.get("factors_status"), "factors_status", ["selected", "none", "unclear"]
    )
    factors: list[str] = []
    if factors_status == "selected":
        raw_factors = payload.get("factors")
        if not isinstance(raw_factors, list):
            raise ValueError("factors must be a list")
        unknown = [factor for factor in raw_factors if factor not in PC.FACTORS_24]
        if unknown:
            raise ValueError(f"unknown factors: {unknown}")
        chosen = set(raw_factors)
        factors = [factor for factor in PC.FACTORS_24 if factor in chosen]
        if not factors:
            raise ValueError("select at least one factor, or record 'none'/'unclear'")

    raw_tags = payload.get("issue_tags", [])
    if not isinstance(raw_tags, list):
        raise ValueError("issue_tags must be a list")
    unknown_tags = [tag for tag in raw_tags if tag not in ISSUE_TAGS]
    if unknown_tags:
        raise ValueError(f"unknown issue tags: {unknown_tags}")
    tagged = set(raw_tags)
    issue_tags = [tag for tag in ISSUE_TAGS if tag in tagged]

    notes = payload.get("notes", "")
    if not isinstance(notes, str):
        raise ValueError("notes must be a string")
    notes = notes.strip()
    if len(notes) > MAX_NOTES_CHARS:
        raise ValueError(f"notes must be at most {MAX_NOTES_CHARS} characters")

    return {
        "risk": risk,
        "evidence": [span["text"] for span in spans],
        "evidence_unclear": evidence_status == "unclear",
        "factors": factors,
        "factors_unclear": factors_status == "unclear",
        "naturalness": _as_int(payload.get("naturalness"), "naturalness", 1, 5),
        "confidence": _as_int(payload.get("confidence"), "confidence", 1, 3),
        "issue_tags": issue_tags,
        "notes": notes,
        "evidence_spans": spans,
        "interface": UI_INTERFACE,
    }


class AuditSession:
    """One annotator's live session over a frozen packet."""

    def __init__(self, audit_dir: Path, annotator: str, display_name: str | None = None):
        self.audit_dir = audit_dir
        self.manifest, self.packet = _load_audit(audit_dir)
        self.annotator = _slug(annotator)
        registered = self.manifest.get("annotators")
        if registered and self.annotator not in registered:
            raise SystemExit(
                f"{self.annotator!r} is not a registered annotator: {registered}"
            )
        self.path = _annotation_path(audit_dir, self.annotator)
        if self.path.exists():
            annotation = _read_json(self.path)
            if annotation["packet_sha256"] != self.manifest["packet"]["sha256"]:
                raise ValueError(f"{self.path}: packet hash mismatch")
        else:
            annotation = _new_annotation(
                self.annotator,
                display_name or annotator,
                self.manifest["packet"]["sha256"],
            )
        self.annotation = annotation
        self.cases = self.packet["task"]["cases"]
        self.case_by_id = {case["case_id"]: case for case in self.cases}
        self._lock = threading.Lock()

    def task_payload(self) -> dict:
        """Everything the browser is allowed to know.  No assigned labels."""
        with self._lock:
            responses = json.loads(json.dumps(self.annotation["responses"]))
        return {
            "protocol": PROTOCOL,
            "interface": UI_INTERFACE,
            "annotator": self.annotator,
            "display_name": self.annotation["display_name"],
            "risk_classes": list(PC.RISK_CLASSES),
            "risk_definitions": RISK_DEFINITIONS,
            "risk_shortcuts": {
                key: label for key, label in RISK_SHORTCUTS.items() if key != "?"
            },
            "factors": [
                {"name": factor, "definition": SC.FACTOR_DEFINITIONS[factor]}
                for factor in PC.FACTORS_24
            ],
            "issue_tags": list(ISSUE_TAGS),
            "cases": self.cases,
            "responses": responses,
        }

    def save(self, payload: object) -> dict:
        if not isinstance(payload, dict):
            raise ValueError("payload must be a JSON object")
        case_id = payload.get("case_id")
        case = self.case_by_id.get(case_id)
        if case is None:
            raise ValueError(f"unknown case_id: {case_id!r}")
        response = build_ui_response(case, payload)
        with self._lock:
            self.annotation["responses"][case_id] = response
            self.annotation["status"] = (
                "complete"
                if len(self.annotation["responses"]) == len(self.cases)
                else "in_progress"
            )
            _atomic_json(self.path, self.annotation)
            done = len(self.annotation["responses"])
            status = self.annotation["status"]
        return {
            "case_id": case_id,
            "response": response,
            "completed": done,
            "total": len(self.cases),
            "status": status,
        }

    @property
    def completed(self) -> int:
        with self._lock:
            return len(self.annotation["responses"])


UI_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blinded semantic audit</title>
<style>
  :root { color-scheme: light dark; --line: #d5d5d9; --muted: #6b6b73; --hl: #ffd76e; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
  header { position: sticky; top: 0; z-index: 5; display: flex; flex-wrap: wrap; gap: 12px;
           align-items: center; padding: 10px 16px; border-bottom: 1px solid var(--line);
           background: Canvas; }
  header .grow { flex: 1; }
  main { max-width: 900px; margin: 0 auto; padding: 16px; }
  section { border: 1px solid var(--line); border-radius: 6px; padding: 12px 14px; margin-bottom: 14px; }
  h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted);
       margin: 0 0 10px; }
  .warn { color: var(--muted); font-size: 13px; }
  #post { white-space: pre-wrap; font-size: 17px; line-height: 1.75; padding: 12px;
          border: 1px solid var(--line); border-radius: 6px; user-select: text; cursor: text; }
  #post .hl { background: var(--hl); color: #1a1a1a; }
  .spanlist { list-style: none; margin: 10px 0 0; padding: 0; }
  .spanlist li { display: flex; gap: 8px; align-items: baseline; padding: 3px 0; }
  .spanlist code { background: var(--hl); color: #1a1a1a; padding: 1px 4px; border-radius: 3px;
                   font-size: 13px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(270px, 1fr)); gap: 6px 16px; }
  label.opt { display: block; padding: 2px 0; }
  label.opt .def { display: block; color: var(--muted); font-size: 12px; margin-left: 22px; }
  .row { display: flex; flex-wrap: wrap; gap: 16px; align-items: center; }
  .status { display: flex; gap: 10px; align-items: baseline; }
  button { font: inherit; padding: 5px 12px; border: 1px solid var(--line); border-radius: 5px;
           background: ButtonFace; color: inherit; cursor: pointer; }
  button.primary { background: #1f6feb; border-color: #1f6feb; color: #fff; }
  button:disabled { opacity: .5; cursor: default; }
  select, textarea { font: inherit; color: inherit; background: Canvas; border: 1px solid var(--line);
                     border-radius: 5px; padding: 5px; }
  textarea { width: 100%; min-height: 60px; }
  .msg { font-size: 13px; }
  .msg.err { color: #c22; }
  .msg.ok { color: var(--muted); }
  kbd { border: 1px solid var(--line); border-radius: 4px; padding: 0 4px; font-size: 12px; }
  .done { color: #16794a; font-weight: 600; }
  /* Wide screens: post pinned on the left, all controls scroll on the right. */
  @media (min-width: 1000px) {
    main { max-width: 1500px; display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
           column-gap: 20px; align-items: start; }
    main > :not(#postpane) { grid-column: 2; }
    #postpane { grid-column: 1; grid-row: 1 / span 20; position: sticky; top: 64px;
                max-height: calc(100vh - 80px); overflow: auto; }
  }
</style>
</head>
<body>
<header>
  <strong id="who">&hellip;</strong>
  <span class="status"><span id="progress"></span></span>
  <span class="grow"></span>
  <button id="prev" title="[">&larr;</button>
  <select id="jump"></select>
  <button id="next" title="]">&rarr;</button>
  <button id="save" class="primary" title="Ctrl/Cmd+Enter">Save &amp; next</button>
</header>
<main>
  <p class="warn">Blinded audit. Assigned labels, target factor, and generator identity are
    hidden. Focal posts may contain explicit suicide and self-harm language.</p>
  <div class="msg" id="msg"></div>

  <section id="postpane">
    <h2>Focal post <span id="caseid"></span></h2>
    <div id="post"></div>
    <div class="row" style="margin-top:10px">
      <label><input type="checkbox" id="snap" checked> snap to whole words</label>
      <span class="warn">Select text to mark risk evidence.</span>
    </div>
  </section>

  <section>
    <h2>Risk</h2>
    <div id="risk" class="row"></div>
  </section>

  <section>
    <h2>Risk evidence</h2>
    <div class="row" id="evidence_status"></div>
    <ul class="spanlist" id="spans"></ul>
  </section>

  <section>
    <h2>Factors</h2>
    <div class="row" id="factors_status" style="margin-bottom:8px"></div>
    <div class="grid" id="factors"></div>
  </section>

  <section>
    <h2>Ratings</h2>
    <div class="row"><span>Naturalness (1 = clearly machine-like, 5 = plausibly naturally
      authored)</span><span id="naturalness" class="row"></span></div>
    <div class="row" style="margin-top:6px"><span>Confidence</span>
      <span id="confidence" class="row"></span></div>
  </section>

  <section>
    <h2>Issue tags</h2>
    <div class="grid" id="issue_tags"></div>
  </section>

  <section>
    <h2>Notes</h2>
    <textarea id="notes" placeholder="Optional"></textarea>
  </section>

  <p class="warn"><kbd>i</kbd><kbd>d</kbd><kbd>b</kbd><kbd>a</kbd><kbd>u</kbd> risk &middot;
    <kbd>[</kbd><kbd>]</kbd> previous/next &middot; <kbd>Ctrl</kbd>+<kbd>Enter</kbd> save and next
    &middot; edits are saved to disk per case and can be revised any time.</p>
</main>
<script>
const $ = (id) => document.getElementById(id);
let TASK = null, index = 0, spans = [], dirty = false, post = "", chars = [];

function radios(host, name, values, labeller) {
  host.innerHTML = "";
  for (const value of values) {
    const label = document.createElement("label");
    label.className = "opt";
    label.innerHTML = '<input type="radio" name="' + name + '" value="' +
      String(value).replace(/"/g, "&quot;") + '"> ' + labeller(value);
    label.querySelector("input").addEventListener("change", () => { dirty = true; render(); });
    host.appendChild(label);
  }
}
function checks(host, name, items) {
  host.innerHTML = "";
  for (const item of items) {
    const label = document.createElement("label");
    label.className = "opt";
    label.innerHTML = '<input type="checkbox" name="' + name + '" value="' +
      item.name.replace(/"/g, "&quot;") + '"> ' + escapeHtml(item.name) +
      (item.definition ? '<span class="def">' + escapeHtml(item.definition) + "</span>" : "");
    label.querySelector("input").addEventListener("change", () => { dirty = true; render(); });
    host.appendChild(label);
  }
}
function escapeHtml(text) {
  return text.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}
function getRadio(name) {
  const node = document.querySelector('input[name="' + name + '"]:checked');
  return node ? node.value : null;
}
function setRadio(name, value) {
  for (const node of document.querySelectorAll('input[name="' + name + '"]')) {
    node.checked = String(node.value) === String(value);
  }
}
function getChecks(name) {
  return [...document.querySelectorAll('input[name="' + name + '"]:checked')].map((n) => n.value);
}
function setChecks(name, values) {
  const wanted = new Set(values || []);
  for (const node of document.querySelectorAll('input[name="' + name + '"]')) {
    node.checked = wanted.has(node.value);
  }
}

function renderPost() {
  const host = $("post");
  host.innerHTML = "";
  chars = [];
  for (let i = 0; i < post.length; i++) {
    const node = document.createElement("span");
    node.dataset.i = String(i);
    node.textContent = post[i];
    host.appendChild(node);
    chars.push(node);
  }
}
function paint() {
  const marked = new Uint8Array(post.length);
  for (const span of spans) { for (let i = span.start; i < span.end; i++) marked[i] = 1; }
  for (let i = 0; i < chars.length; i++) chars[i].className = marked[i] ? "hl" : "";
}
function offsetOf(container, offset) {
  if (container.nodeType === Node.TEXT_NODE) {
    const index = Number(container.parentElement.dataset.i);
    return Number.isNaN(index) ? 0 : index + (offset > 0 ? 1 : 0);
  }
  const child = container.childNodes[offset];
  if (!child) return post.length;
  const index = Number(child.dataset ? child.dataset.i : NaN);
  return Number.isNaN(index) ? 0 : index;
}
function addSelection() {
  const selection = window.getSelection();
  if (!selection || selection.isCollapsed || selection.rangeCount === 0) return;
  const range = selection.getRangeAt(0);
  if (!$("post").contains(range.commonAncestorContainer)) return;
  let start = offsetOf(range.startContainer, range.startOffset);
  let end = offsetOf(range.endContainer, range.endOffset);
  if (end < start) [start, end] = [end, start];
  while (start < end && /\s/.test(post[start])) start++;
  while (end > start && /\s/.test(post[end - 1])) end--;
  if (end <= start) { selection.removeAllRanges(); return; }
  if ($("snap").checked) {
    // Word characters only, so a half-selected word completes but trailing
    // punctuation is not pulled into the evidence span.
    const word = /[\w'’]/;
    while (start > 0 && word.test(post[start - 1])) start--;
    while (end < post.length && word.test(post[end])) end++;
  }
  spans.push({ start, end });
  spans.sort((a, b) => a.start - b.start);
  const merged = [];
  for (const span of spans) {
    if (merged.length && span.start <= merged[merged.length - 1].end) {
      merged[merged.length - 1].end = Math.max(merged[merged.length - 1].end, span.end);
    } else merged.push({ ...span });
  }
  spans = merged;
  selection.removeAllRanges();
  setRadio("evidence_status", "spans");
  dirty = true;
  render();
}

function render() {
  paint();
  const list = $("spans");
  list.innerHTML = "";
  spans.forEach((span, position) => {
    const item = document.createElement("li");
    const code = document.createElement("code");
    code.textContent = post.slice(span.start, span.end);
    const drop = document.createElement("button");
    drop.textContent = "remove";
    drop.addEventListener("click", () => {
      spans.splice(position, 1); dirty = true; render();
    });
    item.append(code, drop);
    list.appendChild(item);
  });
  if (!spans.length) {
    const item = document.createElement("li");
    item.className = "warn";
    item.textContent = "No spans marked.";
    list.appendChild(item);
  }
  const done = Object.keys(TASK.responses).length;
  $("progress").innerHTML = "<span" + (done === TASK.cases.length ? ' class="done"' : "") +
    ">" + done + " / " + TASK.cases.length + " saved</span>";
  $("prev").disabled = index === 0;
  $("next").disabled = index === TASK.cases.length - 1;
}

function loadCase(position) {
  index = Math.max(0, Math.min(TASK.cases.length - 1, position));
  const item = TASK.cases[index];
  post = item.focal_post;
  $("caseid").textContent = item.case_id + "  (" + (index + 1) + " of " + TASK.cases.length + ")";
  $("jump").value = String(index);
  renderPost();
  const saved = TASK.responses[item.case_id];
  if (saved) {
    setRadio("risk", saved.risk);
    spans = (saved.evidence_spans || restoreSpans(saved.evidence)).map(
      (span) => ({ start: span.start, end: span.end }));
    setRadio("evidence_status",
      saved.evidence_unclear ? "unclear" : (saved.evidence.length ? "spans" : "none"));
    setRadio("factors_status",
      saved.factors_unclear ? "unclear" : (saved.factors.length ? "selected" : "none"));
    setChecks("factors", saved.factors);
    setRadio("naturalness", saved.naturalness);
    setRadio("confidence", saved.confidence);
    setChecks("issue_tags", saved.issue_tags);
    $("notes").value = saved.notes || "";
  } else {
    setRadio("risk", null);
    spans = [];
    setRadio("evidence_status", "spans");
    setRadio("factors_status", "selected");
    setChecks("factors", []);
    setRadio("naturalness", null);
    setRadio("confidence", null);
    setChecks("issue_tags", []);
    $("notes").value = "";
  }
  dirty = false;
  message(saved ? "Saved response loaded; edits overwrite it." : "", "ok");
  render();
  window.scrollTo(0, 0);
}
function restoreSpans(texts) {
  const out = [];
  let from = 0;
  for (const text of texts || []) {
    const at = post.indexOf(text, from);
    if (at < 0) continue;
    out.push({ start: at, end: at + text.length });
    from = at + text.length;
  }
  return out;
}
function message(text, kind) {
  const node = $("msg");
  node.textContent = text;
  node.className = "msg " + (kind || "");
}

function payload() {
  const item = TASK.cases[index];
  return {
    case_id: item.case_id,
    post_sha256: item.post_sha256,
    risk: getRadio("risk"),
    evidence_status: getRadio("evidence_status"),
    evidence_spans: spans.map((span) => ({
      start: span.start, end: span.end, text: post.slice(span.start, span.end),
    })),
    factors_status: getRadio("factors_status"),
    factors: getChecks("factors"),
    naturalness: Number(getRadio("naturalness")),
    confidence: Number(getRadio("confidence")),
    issue_tags: getChecks("issue_tags"),
    notes: $("notes").value,
  };
}
async function save(advance) {
  const body = payload();
  if (!body.risk) { message("Choose a risk level first.", "err"); return false; }
  if (!body.naturalness || !body.confidence) {
    message("Rate naturalness and confidence first.", "err"); return false;
  }
  $("save").disabled = true;
  try {
    const reply = await fetch("/api/response", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await reply.json();
    if (!reply.ok) { message(data.error || "Save failed.", "err"); return false; }
    TASK.responses[data.case_id] = data.response;
    dirty = false;
    const saved = "Saved " + data.case_id + " (" + data.completed + "/" + data.total + ").";
    message(saved, "ok");
    render();
    if (advance) {
      const nextOpen = TASK.cases.findIndex(
        (item, position) => position > index && !TASK.responses[item.case_id]);
      if (nextOpen >= 0) loadCase(nextOpen);
      else if (index < TASK.cases.length - 1) loadCase(index + 1);
      // loadCase resets the banner, so restate the confirmation for the new case.
      message(
        data.completed === data.total ? saved + " All cases are saved." : saved, "ok");
    }
    return true;
  } catch (error) {
    message("Save failed: " + error, "err");
    return false;
  } finally {
    $("save").disabled = false;
  }
}
function leave(position) {
  if (dirty && !window.confirm("Discard unsaved edits on this case?")) return;
  loadCase(position);
}

(async function start() {
  TASK = await (await fetch("/api/task")).json();
  $("who").textContent = TASK.display_name;
  radios($("risk"), "risk", [...TASK.risk_classes, "Unclear"], (value) =>
    escapeHtml(value) + '<span class="def">' +
    escapeHtml(TASK.risk_definitions[value] || "Cannot decide from this post") + "</span>");
  radios($("evidence_status"), "evidence_status", ["spans", "none", "unclear"], (value) =>
    ({ spans: "highlighted spans", none: "no risk evidence in this post",
       unclear: "unclear" })[value]);
  radios($("factors_status"), "factors_status", ["selected", "none", "unclear"], (value) =>
    ({ selected: "checked factors", none: "no factors present", unclear: "unclear" })[value]);
  checks($("factors"), "factors", TASK.factors);
  checks($("issue_tags"), "issue_tags", TASK.issue_tags.map((name) => ({ name, definition: "" })));
  radios($("naturalness"), "naturalness", [1, 2, 3, 4, 5], (value) => String(value));
  radios($("confidence"), "confidence", [1, 2, 3], (value) =>
    ({ 1: "1 low", 2: "2 medium", 3: "3 high" })[value]);
  $("jump").innerHTML = TASK.cases.map((item, position) =>
    '<option value="' + position + '">' + item.case_id + "</option>").join("");
  $("jump").addEventListener("change", (event) => leave(Number(event.target.value)));
  $("prev").addEventListener("click", () => leave(index - 1));
  $("next").addEventListener("click", () => leave(index + 1));
  $("save").addEventListener("click", () => save(true));
  $("post").addEventListener("mouseup", addSelection);
  $("notes").addEventListener("input", () => { dirty = true; });
  window.addEventListener("beforeunload", (event) => {
    if (dirty) { event.preventDefault(); event.returnValue = ""; }
  });
  document.addEventListener("keydown", (event) => {
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(event.target.tagName);
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") { save(true); return; }
    if (typing || event.metaKey || event.ctrlKey || event.altKey) return;
    const risk = { i: "Indicator", d: "Ideation", b: "Behavior", a: "Attempt", u: "Unclear" };
    if (risk[event.key]) { setRadio("risk", risk[event.key]); dirty = true; render(); }
    else if (event.key === "[") leave(index - 1);
    else if (event.key === "]") leave(index + 1);
  });
  const firstOpen = TASK.cases.findIndex((item) => !TASK.responses[item.case_id]);
  loadCase(firstOpen >= 0 ? firstOpen : 0);
})();
</script>
</body>
</html>
"""


class _AuditHandler(BaseHTTPRequestHandler):
    """Serves the blinded packet only.  The assigned-label key is never read."""

    session: AuditSession | None = None
    server_version = "SemanticAuditUI/1"

    def log_message(self, fmt: str, *args) -> None:  # noqa: A002 - stdlib signature
        return

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, value: object) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802 - stdlib signature
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            self._send(200, UI_HTML.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/task":
            self._send_json(200, self.session.task_payload())
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib signature
        if self.path.split("?", 1)[0] != "/api/response":
            self._send_json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY_BYTES:
            self._send_json(400, {"error": "missing or oversized request body"})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(400, {"error": "request body is not valid JSON"})
            return
        try:
            result = self.session.save(payload)
        except ValueError as error:
            self._send_json(400, {"error": str(error)})
            return
        print(
            f"saved {result['case_id']} "
            f"({result['completed']}/{result['total']}) -> {self.session.path.name}"
        )
        self._send_json(200, result)


def serve(args: argparse.Namespace) -> int:
    audit_dir = args.audit_dir.resolve()
    session = AuditSession(audit_dir, args.annotator, args.display_name)
    handler = type("_BoundAuditHandler", (_AuditHandler,), {"session": session})
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    url = f"http://{args.host}:{httpd.server_address[1]}/"
    total = len(session.cases)
    print(f"Blinded semantic audit, annotator: {session.annotation['display_name']}")
    print(f"Completed {session.completed}/{total}. Saving to {session.path}")
    print("Assigned labels, target factors, and generator metadata are hidden.")
    print("Content warning: focal posts may contain explicit suicide and self-harm language.")
    print(f"Open {url}, then press Ctrl+C here when finished.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        httpd.server_close()
    print(f"Saved progress: {session.completed}/{total} in {session.path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    prepare_parser = sub.add_parser("prepare", help="freeze a deterministic blinded subset")
    prepare_parser.add_argument("--synth-dir", type=Path, default=DEFAULT_SYNTH_DIR)
    prepare_parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    prepare_parser.add_argument("--per-factor", type=int, default=6)
    prepare_parser.add_argument("--nontarget", type=int, default=12)
    prepare_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    prepare_parser.add_argument(
        "--annotators", nargs="+", default=list(DEFAULT_ANNOTATORS)
    )
    prepare_parser.set_defaults(func=prepare)

    serve_parser = sub.add_parser(
        "serve", help="open the blinded NER-style annotation UI in a browser"
    )
    serve_parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    serve_parser.add_argument("--annotator", required=True)
    serve_parser.add_argument("--display-name")
    serve_parser.add_argument("--host", default=DEFAULT_HOST)
    serve_parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve_parser.add_argument(
        "--no-browser", action="store_true", help="do not open a browser tab automatically"
    )
    serve_parser.set_defaults(func=serve)

    annotate_parser = sub.add_parser(
        "annotate", help="terminal fallback for a blinded annotation (see: serve)"
    )
    annotate_parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    annotate_parser.add_argument("--annotator", required=True)
    annotate_parser.add_argument("--display-name")
    annotate_parser.set_defaults(func=annotate)

    status_parser = sub.add_parser("status", help="show independent/adjudication progress")
    status_parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    status_parser.set_defaults(func=status)

    adjudicate_parser = sub.add_parser(
        "adjudicate", help="resolve disagreements without showing assigned labels"
    )
    adjudicate_parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    adjudicate_parser.add_argument("--annotators", nargs=2, required=True)
    adjudicate_parser.add_argument("--adjudicator", default="two-annotator consensus")
    adjudicate_parser.set_defaults(func=adjudicate)

    report_parser = sub.add_parser(
        "report", help="compare annotations with the hidden assigned-label key"
    )
    report_parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    report_parser.add_argument("--annotators", nargs="+")
    report_parser.add_argument(
        "--unlock-key",
        action="store_true",
        help="acknowledge that independent annotation/adjudication is complete",
    )
    report_parser.set_defaults(func=report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "per_factor", 1) < 1:
        raise SystemExit("--per-factor must be positive")
    if getattr(args, "nontarget", 0) < 0:
        raise SystemExit("--nontarget must be non-negative")
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
