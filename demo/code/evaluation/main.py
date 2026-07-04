"""
Evaluation workflow: score an existing predictions CSV against the labeled sample set.

This does NOT run the agent. It reads a predictions CSV (produced by running the main
pipeline over sample_claims.csv) and compares it to the expected-output columns in
dataset/sample_claims.csv, reporting per-field accuracy, a claim_status confusion matrix,
and a severity off-by-one breakdown.

Usage (paths default to the repo layout; run from anywhere):
    python evaluation/main.py
    python evaluation/main.py path/to/predictions.csv path/to/sample_claims.csv

Layout assumed:
    repo/
      code/
        main.py                 -> writes output.csv to dataset/ directory by default
        evaluation/main.py      <- this file
      dataset/
        output.csv              <- predictions (default)
        sample_claims.csv       <- labels (default)
"""

import os
import csv
import sys
import json
from collections import Counter, defaultdict

# --- anchored default paths (work regardless of current directory) ---
BASE = os.path.dirname(os.path.abspath(__file__))      # .../code/evaluation
CODE = os.path.dirname(BASE)                            # .../code
REPO = os.path.dirname(CODE)                            # .../repo
DATASET = os.path.join(REPO, "dataset")

DEFAULT_PRED = os.path.join(DATASET, "output.csv")
DEFAULT_LABELS = os.path.join(DATASET, "sample_claims.csv")

# Fields scored by exact categorical match
CATEGORICAL = [
    "evidence_standard_met", "valid_image", "claim_status",
    "issue_type", "object_part", "severity",
]
# Fields scored as order-independent sets (semicolon-joined)
SET_FIELDS = ["risk_flags", "supporting_image_ids"]
# Free-text fields: not accuracy-scored (no exact ground truth), only completeness
TEXT_FIELDS = ["evidence_standard_met_reason", "claim_status_justification"]

SEV_ORDER = {"none": 0, "low": 1, "medium": 2, "high": 3}   # 'unknown' is non-ordinal


def load_csv(path):
    """Read a CSV with encoding fallback (the dataset has some cp1252 bytes)."""
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            with open(path, encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("all", b"", 0, 1, f"could not decode {path}")


def lc(x):
    return (x or "").strip().lower()


def as_set(x):
    """'a;b' -> {'a','b'}; treats 'none'/'' as empty."""
    items = {t.strip().lower() for t in (x or "").split(";") if t.strip()}
    return items - {"none", ""}


def key(row):
    # user_id can repeat across rows, so align on (user_id, image_paths)
    return (row.get("user_id", ""), row.get("image_paths", ""))


def score(pred_rows, gold_rows):
    gmap = {key(g): g for g in gold_rows}

    cat_hits = {c: 0 for c in CATEGORICAL}
    cat_total = {c: 0 for c in CATEGORICAL}
    set_hits = {c: 0 for c in SET_FIELDS}
    set_total = {c: 0 for c in SET_FIELDS}
    text_present = {c: 0 for c in TEXT_FIELDS}

    confusion = defaultdict(Counter)        # confusion[gold_status][pred_status]
    sev_adjacent = 0
    sev_far = 0
    fully_correct = 0
    aligned = 0
    missing = 0

    for p in pred_rows:
        g = gmap.get(key(p))
        if g is None:
            missing += 1
            continue
        aligned += 1

        row_all_correct = True
        for c in CATEGORICAL:
            gv = lc(g.get(c))
            if gv == "":            # row has no label for this field -> skip
                continue
            cat_total[c] += 1
            if lc(p.get(c)) == gv:
                cat_hits[c] += 1
            else:
                row_all_correct = False

        for c in SET_FIELDS:
            if lc(g.get(c)) == "" and lc(p.get(c)) == "":
                continue
            set_total[c] += 1
            if as_set(p.get(c)) == as_set(g.get(c)):
                set_hits[c] += 1

        for c in TEXT_FIELDS:
            if (p.get(c) or "").strip():
                text_present[c] += 1

        # claim_status confusion + severity distance
        confusion[lc(g.get("claim_status"))][lc(p.get("claim_status"))] += 1
        gv_s, pv_s = lc(g.get("severity")), lc(p.get("severity"))
        if gv_s != pv_s and gv_s in SEV_ORDER and pv_s in SEV_ORDER:
            if abs(SEV_ORDER[gv_s] - SEV_ORDER[pv_s]) == 1:
                sev_adjacent += 1
            else:
                sev_far += 1

        if row_all_correct:
            fully_correct += 1

    return {
        "aligned": aligned, "missing": missing, "fully_correct": fully_correct,
        "cat_hits": cat_hits, "cat_total": cat_total,
        "set_hits": set_hits, "set_total": set_total,
        "text_present": text_present,
        "confusion": {k: dict(v) for k, v in confusion.items()},
        "sev_adjacent": sev_adjacent, "sev_far": sev_far,
    }


def pct(h, t):
    return round(100 * h / t) if t else 0


def report(m, pred_path, labels_path):
    print(f"\nPredictions : {pred_path}")
    print(f"Labels      : {labels_path}")
    print(f"Aligned rows: {m['aligned']}   (unmatched predictions: {m['missing']})")

    print("\n=== Categorical field accuracy ===")
    cat_pcts = []
    for c in CATEGORICAL:
        h, t = m["cat_hits"][c], m["cat_total"][c]
        cat_pcts.append(pct(h, t))
        print(f"  {c:24s} {h:3d}/{t:<3d}  {pct(h, t):3d}%")
    mean = round(sum(cat_pcts) / len(cat_pcts)) if cat_pcts else 0
    print(f"  {'MEAN':24s} {'':7s} {mean:3d}%")
    print(f"  fully-correct rows (all categorical): "
          f"{m['fully_correct']}/{m['aligned']}")

    print("\n=== Set-based field accuracy (order-independent) ===")
    for c in SET_FIELDS:
        h, t = m["set_hits"][c], m["set_total"][c]
        print(f"  {c:24s} {h:3d}/{t:<3d}  {pct(h, t):3d}%")

    print("\n=== claim_status confusion (rows: gold -> pred) ===")
    statuses = ["supported", "contradicted", "not_enough_information"]
    header = "  gold \\ pred         " + "".join(f"{s[:12]:>14s}" for s in statuses)
    print(header)
    for gs in statuses:
        line = f"  {gs:20s}"
        for ps in statuses:
            line += f"{m['confusion'].get(gs, {}).get(ps, 0):>14d}"
        print(line)

    print("\n=== severity error profile ===")
    print(f"  off-by-one (adjacent level, e.g. medium<->high): {m['sev_adjacent']}")
    print(f"  off-by-more or to/from unknown                : {m['sev_far']}")
    print("  (adjacent-level severity misses are largely label subjectivity)")

    print("\n=== free-text completeness (not accuracy) ===")
    for c in TEXT_FIELDS:
        print(f"  {c:30s} non-empty in {m['text_present'][c]}/{m['aligned']} rows")


def main():
    pred_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PRED
    labels_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_LABELS

    pred_rows = load_csv(pred_path)
    gold_rows = load_csv(labels_path)
    m = score(pred_rows, gold_rows)
    report(m, pred_path, labels_path)

    # also dump machine-readable metrics next to this file (handy for the write-up)
    try:
        out = os.path.join(BASE, "metrics.json")
        summary = {
            "aligned": m["aligned"], "fully_correct": m["fully_correct"],
            "categorical": {c: {"hits": m["cat_hits"][c], "total": m["cat_total"][c],
                                "pct": pct(m["cat_hits"][c], m["cat_total"][c])}
                            for c in CATEGORICAL},
            "set_fields": {c: {"hits": m["set_hits"][c], "total": m["set_total"][c],
                               "pct": pct(m["set_hits"][c], m["set_total"][c])}
                           for c in SET_FIELDS},
            "claim_status_confusion": m["confusion"],
            "severity_adjacent": m["sev_adjacent"], "severity_far": m["sev_far"],
        }
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote metrics to {out}")
    except OSError:
        pass


if __name__ == "__main__":
    main()