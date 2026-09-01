"""Write and validate organizer-format predictions for unseen examples."""
import ast
import csv
from pathlib import Path

import pandas as pd

from prep import constants as C
from prep.clean import parse_evidence


def repair_coupling(preds: dict[str, dict]) -> dict[str, dict]:
    repaired = {row_id: {**pred, "spans": list(pred["spans"]),
                         "factors": list(pred["factors"])}
                for row_id, pred in preds.items()}
    for row_id, pred in repaired.items():
        if pred["risk"] == "Indicator":
            pred["spans"] = []
        elif not pred["spans"]:
            raise ValueError(
                f"{row_id}: risk={pred['risk']} but no evidence spans")
    return repaired


def validate_preds(preds: dict[str, dict], examples: pd.DataFrame) -> None:
    post_col = "post_raw" if "post_raw" in examples else "post"
    posts = dict(zip(examples["row_id"], examples[post_col]))
    assert set(preds) == set(posts), "row coverage mismatch"
    for row_id, pred in preds.items():
        assert pred["risk"] in C.RISK_CLASSES, (
            f"{row_id}: bad risk {pred['risk']!r}")
        factors = []
        for factor in pred["factors"]:
            factor = factor.replace("’", "'").replace("‘", "'").strip()
            assert factor in C.FACTOR_INDEX, (
                f"{row_id}: factor not byte-exact: {factor!r}")
            if factor not in factors:
                factors.append(factor)
        pred["factors"] = sorted(factors, key=C.FACTOR_INDEX.__getitem__)
        for span in pred["spans"]:
            assert span.strip(), f"{row_id}: empty/whitespace-only span"
            assert span in posts[row_id], (
                f"{row_id}: span not a verbatim substring: {span!r}")
            assert ";" not in span, f"{row_id}: span contains ';': {span!r}"
            assert "\n" not in span, f"{row_id}: span contains newline: {span!r}"


def write_submission(preds: dict[str, dict], examples: pd.DataFrame,
                     out_path: str | Path, empty_style: str = "none") -> Path:
    """Write row_id,risk_level,evidence,factors without public-set overrides."""
    preds = repair_coupling(preds)
    validate_preds(preds, examples)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["row_id", "risk_level", "evidence", "factors"])
        for row_id in examples["row_id"]:
            pred = preds[row_id]
            evidence = "; ".join(pred["spans"]) if pred["spans"] else (
                "none" if empty_style == "none" else "")
            writer.writerow([row_id, pred["risk"], evidence,
                             repr(pred["factors"])])
    validate_submission(out_path, examples)
    return out_path


def validate_submission(path: str | Path, examples: pd.DataFrame) -> None:
    submission = pd.read_csv(path, dtype=str).fillna("")
    expected_cols = ["row_id", "risk_level", "evidence", "factors"]
    assert list(submission.columns) == expected_cols
    assert len(submission) == len(examples)
    assert set(submission["row_id"]) == set(examples["row_id"])
    post_col = "post_raw" if "post_raw" in examples else "post"
    posts = dict(zip(examples["row_id"], examples[post_col]))
    for row in submission.itertuples():
        assert row.risk_level in C.RISK_CLASSES
        spans = parse_evidence(row.evidence)
        if row.risk_level == "Indicator":
            assert spans == [], f"{row.row_id}: Indicator must have no evidence"
        else:
            assert spans, f"{row.row_id}: non-Indicator must have evidence"
        for span in spans:
            assert span in posts[row.row_id], (
                f"{row.row_id}: span not verbatim after round-trip: {span!r}")
        factors = ast.literal_eval(row.factors)
        assert isinstance(factors, list)
        assert len(factors) == len(set(factors))
        assert all(factor in C.FACTOR_INDEX for factor in factors)
