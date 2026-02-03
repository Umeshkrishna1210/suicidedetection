from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _is_subpath(p: Path, parent: Path) -> bool:
    try:
        p.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def _load_metrics(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "metrics" in data:
        metrics = data.get("metrics") or {}
    else:
        metrics = data if isinstance(data, dict) else {}

    # Normalize to plain scalars
    out: dict[str, object] = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float, str)) or v is None:
            out[k] = v
        else:
            # drop nested structures from table
            continue

    return out


def _variant_name(outputs_root: Path, metrics_path: Path) -> str:
    # Preferred: outputs/<variant>/metrics.json
    # Also supports: outputs/experiments/<variant>/metrics.json, outputs/eval_<variant>/metrics.json
    rel = metrics_path.resolve().relative_to(outputs_root.resolve())
    parts = rel.parts
    if len(parts) >= 2 and parts[-1].lower() == "metrics.json":
        if parts[0] == "experiments" and len(parts) >= 3:
            return str(parts[1])
        return str(parts[-2])
    return metrics_path.parent.name


def _write_markdown_table(df: pd.DataFrame, path: Path) -> None:
    # Deterministic column order
    cols = ["name"] + [c for c in df.columns if c != "name"]
    df2 = df[cols]
    md = df2.to_markdown(index=False)
    path.write_text(md, encoding="utf-8")


def _plot_bars(df: pd.DataFrame, metric: str, out_path: Path, title: str) -> None:
    if metric not in df.columns:
        return

    d = df[["name", metric]].copy()
    d = d.dropna(subset=[metric])
    if d.empty:
        return

    d = d.sort_values(metric, ascending=False)

    plt.figure(figsize=(10, max(4, 0.35 * len(d))))
    plt.barh(d["name"], d[metric])
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel(metric)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs", help="Path to outputs directory")
    ap.add_argument(
        "--pattern",
        default="**/metrics.json",
        help="Glob pattern under outputs/ to find metrics files",
    )
    ap.add_argument("--out", default="outputs/comparisons", help="Where to write tables/plots")
    ap.add_argument(
        "--prefer",
        default="experiments",
        choices=["experiments", "all"],
        help="If 'experiments', only include outputs/experiments/*/metrics.json when present; otherwise include all metrics.json",
    )
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    outputs_root = (root / args.outputs).resolve() if not Path(args.outputs).is_absolute() else Path(args.outputs).resolve()
    out_dir = (root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_paths = sorted(outputs_root.glob(args.pattern))

    exp_root = outputs_root / "experiments"
    exp_paths = []
    if exp_root.exists():
        exp_paths = sorted(exp_root.glob("*/metrics.json"))

    if args.prefer == "experiments" and exp_paths:
        metric_paths = exp_paths
    else:
        # Filter out chroma internals
        metric_paths = [p for p in all_paths if "chroma" not in str(p).lower()]

    rows: list[dict] = []
    for p in metric_paths:
        try:
            metrics = _load_metrics(p)
        except Exception:
            continue
        name = _variant_name(outputs_root, p)
        rows.append({"name": name, **metrics, "_path": str(p)})

    if not rows:
        raise SystemExit(f"No metrics found under: {outputs_root}")

    df = pd.DataFrame(rows)

    # Keep only scalar metric columns + name
    # Move path to end
    cols = [c for c in df.columns if c not in {"_path"}]
    df = df[cols + ["_path"]]

    # Sort by a primary metric if present
    sort_metric = "risk_f1_macro" if "risk_f1_macro" in df.columns else None
    if sort_metric is not None:
        df = df.sort_values(sort_metric, ascending=False)
    else:
        df = df.sort_values("name")

    csv_path = out_dir / "comparison_table.csv"
    md_path = out_dir / "comparison_table.md"

    df.to_csv(csv_path, index=False)
    _write_markdown_table(df.drop(columns=["_path"], errors="ignore"), md_path)

    # Plots
    _plot_bars(df, "risk_f1_macro", out_dir / "risk_f1_macro.png", "Risk F1 (Macro)")
    _plot_bars(df, "emotion_f1_macro", out_dir / "emotion_f1_macro.png", "Emotion F1 (Macro)")
    _plot_bars(df, "risk_recall_high", out_dir / "risk_recall_high.png", "Risk Recall (High Class)")

    print(f"OK: wrote {csv_path}")
    print(f"OK: wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
