from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any

import sys

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from srrtd.utils.config import apply_overrides, load_yaml  # noqa: E402


_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin1")


def _open_text(path: Path):
    last_err: Exception | None = None
    for enc in _ENCODINGS:
        try:
            return path.open("r", encoding=enc, newline="")
        except UnicodeDecodeError as e:
            last_err = e
    if last_err is not None:
        raise last_err
    return path.open("r", encoding="utf-8", newline="")


def _norm_label(v: Any) -> str:
    if v is None:
        return ""
    return str(v).strip()


def _maybe_int(s: str) -> int | None:
    ss = str(s).strip()
    if not ss:
        return None
    try:
        if ss.isdigit() or (ss.startswith("-") and ss[1:].isdigit()):
            return int(ss)
    except Exception:
        return None
    return None


def inspect_csv_labels(
    path: Path,
    label_field: str,
    max_rows: int = 0,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    int_counts: Counter[int] = Counter()
    missing = 0
    total = 0

    with _open_text(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            raw = row.get(label_field)
            s = _norm_label(raw)
            if not s:
                missing += 1
            else:
                counts[s] += 1
                iv = _maybe_int(s)
                if iv is not None:
                    int_counts[iv] += 1

            if max_rows and total >= max_rows:
                break

    return {
        "path": str(path),
        "label_field": str(label_field),
        "total_rows_scanned": int(total),
        "missing": int(missing),
        "unique_labels": int(len(counts)),
        "counts": counts,
        "int_counts": int_counts,
    }


def _print_report(title: str, rep: dict[str, Any], label_map: dict[str, Any] | None, class_list: list[str] | None) -> None:
    print(f"\n== {title} ==")
    print(f"file: {rep['path']}")
    print(f"label_field: {rep['label_field']}")
    print(f"rows_scanned: {rep['total_rows_scanned']}")
    print(f"missing_labels: {rep['missing']}")
    print(f"unique_label_values: {rep['unique_labels']}")

    counts: Counter[str] = rep["counts"]
    if counts:
        print("\nTop label values:")
        for k, v in counts.most_common(20):
            print(f"  {k!r}: {v}")

    int_counts: Counter[int] = rep["int_counts"]
    if int_counts and len(int_counts) <= 50:
        print("\nInteger-like label values:")
        for k, v in sorted(int_counts.items(), key=lambda x: x[0]):
            print(f"  {k}: {v}")

    if label_map is not None:
        # Compare normalized keys to normalized observed labels
        norm_map = {str(k).strip().lower(): int(v) for k, v in label_map.items()}
        unmapped = []
        for lab in counts.keys():
            if str(lab).strip().lower() not in norm_map:
                unmapped.append(lab)
        if unmapped:
            print("\nWARN: These label values are NOT covered by label_map (first 30 shown):")
            for x in unmapped[:30]:
                print(f"  {x!r}")
        else:
            print("\nOK: All observed label values are covered by label_map.")

    if class_list is not None:
        norm_classes = [str(c).strip().lower() for c in class_list]
        not_in_classes = []
        for lab in counts.keys():
            # allow numeric labels through
            if _maybe_int(lab) is not None:
                continue
            if str(lab).strip().lower() not in norm_classes:
                not_in_classes.append(lab)
        if not_in_classes:
            print("\nWARN: These string labels are not present in labels.*_classes (first 30 shown):")
            for x in not_in_classes[:30]:
                print(f"  {x!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overrides", nargs="*", default=None)
    ap.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated splits to scan: train,val,test (default: train,val,test). Use 'all' for train+val+test.",
    )
    ap.add_argument("--max-rows", type=int, default=0, help="If >0, scan only first N rows (faster).")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args.overrides)

    root = _ROOT
    mode = str(cfg.get("data", {}).get("mode", "single")).lower()

    if mode != "multitask":
        print("This script currently supports data.mode=multitask only.")
        return 2

    mt = dict(cfg.get("data", {}).get("multitask", {}) or {})
    risk_cfg = dict(mt.get("risk", {}) or {})
    emo_cfg = dict(mt.get("emotion", {}) or {})

    labels_cfg = dict(cfg.get("labels", {}) or {})
    risk_classes = list(labels_cfg.get("risk_classes", []) or [])
    emo_classes = list(labels_cfg.get("emotion_classes", []) or [])

    splits_raw = str(args.splits).strip().lower()
    if splits_raw == "all":
        splits = ["train", "val", "test"]
    else:
        splits = [s.strip() for s in splits_raw.split(",") if s.strip()]
    if not splits:
        print("ERROR: --splits resolved to empty list")
        return 2

    risk_label_field = str(risk_cfg.get("label_field", "label"))
    emo_label_field = str(emo_cfg.get("label_field", "label"))

    risk_label_map = risk_cfg.get("label_map")
    emo_label_map = emo_cfg.get("label_map")

    def _split_path(task_cfg: dict[str, Any], split: str) -> Path | None:
        key = f"{split}_path"
        p = task_cfg.get(key)
        if not p:
            return None
        return (root / str(p)).resolve()

    for split in splits:
        risk_path = _split_path(risk_cfg, split)
        emo_path = _split_path(emo_cfg, split)

        if risk_path is None:
            print(f"\n== RISK {split.upper()} ==\nWARN: config missing risk.{split}_path; skipping")
        elif not risk_path.exists():
            print(f"\n== RISK {split.upper()} ==\nWARN: file not found: {risk_path}; skipping")
        else:
            risk_rep = inspect_csv_labels(risk_path, risk_label_field, max_rows=int(args.max_rows))
            _print_report(f"RISK {split}.csv", risk_rep, label_map=risk_label_map, class_list=risk_classes)

        if emo_path is None:
            print(f"\n== EMOTION {split.upper()} ==\nWARN: config missing emotion.{split}_path; skipping")
        elif not emo_path.exists():
            print(f"\n== EMOTION {split.upper()} ==\nWARN: file not found: {emo_path}; skipping")
        else:
            emo_rep = inspect_csv_labels(emo_path, emo_label_field, max_rows=int(args.max_rows))
            _print_report(f"EMOTION {split}.csv", emo_rep, label_map=emo_label_map, class_list=emo_classes)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
