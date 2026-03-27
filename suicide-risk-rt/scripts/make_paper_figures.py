from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_metrics(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate paper-ready figures from evaluation outputs.")
    ap.add_argument("--eval-dir", default="outputs/eval_final_2day", help="Folder containing metrics.json")
    ap.add_argument("--out-dir", default="reports/figures", help="Where to write figures")
    ap.add_argument("--name", default="final_2day", help="Run identifier (used only for defaults)")
    ap.add_argument(
        "--style",
        choices=["classic", "clean"],
        default="clean",
        help="Figure style. 'classic' matches the original dense layout; 'clean' reduces clutter.",
    )
    ap.add_argument(
        "--out-stem",
        default=None,
        help="Output filename stem (no extension). Example: 'paper_results' -> paper_results.png/pdf",
    )
    ap.add_argument(
        "--show-suptitle",
        action="store_true",
        help="Include a compact suptitle with headline metrics (no run name).",
    )
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    eval_dir = (root / args.eval_dir).resolve() if not Path(args.eval_dir).is_absolute() else Path(args.eval_dir)
    out_dir = (root / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = eval_dir / "metrics.json"
    if not metrics_path.exists():
        raise SystemExit(f"Missing {metrics_path}")

    blob = _load_metrics(metrics_path)
    metrics = dict(blob.get("metrics", {}) or {})
    details = dict(blob.get("details", {}) or {})

    risk_classes = list((details.get("risk_report") or {}).keys())
    # risk_report includes class keys plus 'accuracy', 'macro avg', ...
    risk_classes = [c for c in risk_classes if c not in {"accuracy", "macro avg", "weighted avg"}]

    emo_report = dict(details.get("emotion_report", {}) or {})
    emo_classes = [c for c in emo_report.keys() if c not in {"accuracy", "macro avg", "weighted avg"}]
    emo_f1 = [
        _safe_float(((emo_report.get(c) or {}).get("f1-score"))) or 0.0
        for c in emo_classes
    ]

    risk_cm = np.array(details.get("risk_confusion") or [], dtype=float)
    if risk_cm.size == 0:
        risk_cm = np.zeros((2, 2), dtype=float)
    risk_cm_norm = risk_cm / np.maximum(risk_cm.sum(axis=1, keepdims=True), 1.0)

    # Import plotting libs lazily to keep script importable without GUI backends.
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")

    if args.style == "classic":
        fig = plt.figure(figsize=(11, 4.5), constrained_layout=True)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.35])

        ax0 = fig.add_subplot(gs[0, 0])
        sns.heatmap(
            risk_cm_norm,
            ax=ax0,
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            cbar=True,
            annot=risk_cm.astype(int),
            fmt="d",
            linewidths=0.5,
            linecolor="#DDDDDD",
            square=True,
        )
        ax0.set_title("Risk confusion (normalized)")
        ax0.set_xlabel("Predicted")
        ax0.set_ylabel("True")
        if len(risk_classes) == risk_cm.shape[0]:
            ax0.set_xticklabels(risk_classes, rotation=30, ha="right")
            ax0.set_yticklabels(risk_classes, rotation=0)

        ax1 = fig.add_subplot(gs[0, 1])
        emo_sorted = list(emo_classes)
        f1_sorted = list(emo_f1)
        bars = ax1.bar(emo_sorted, f1_sorted)
        ax1.set_ylim(0.0, 1.0)
        ax1.set_title("Emotion per-class F1")
        ax1.set_ylabel("F1 score")
        ax1.set_xlabel("Emotion")
        ax1.bar_label(bars, labels=[f"{v:.2f}" for v in f1_sorted], padding=3, fontsize=9)
        ax1.tick_params(axis="x", rotation=30)
    else:
        # 'clean' style: less crowded.
        fig = plt.figure(figsize=(12.5, 4.2), constrained_layout=True)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.4])

        ax0 = fig.add_subplot(gs[0, 0])
        sns.heatmap(
            risk_cm_norm,
            ax=ax0,
            cmap="Blues",
            vmin=0.0,
            vmax=1.0,
            cbar=True,
            annot=True,
            fmt=".2f",
            linewidths=0.4,
            linecolor="#EEEEEE",
            square=True,
        )
        ax0.set_title("Risk confusion (row-normalized)")
        ax0.set_xlabel("Predicted")
        ax0.set_ylabel("True")
        if len(risk_classes) == risk_cm.shape[0]:
            ax0.set_xticklabels(risk_classes, rotation=20, ha="right")
            ax0.set_yticklabels(risk_classes, rotation=0)

        ax1 = fig.add_subplot(gs[0, 1])
        # Horizontal bars reduce label crowding.
        order = list(np.argsort(np.array(emo_f1))[::-1]) if emo_classes else []
        emo_sorted = [emo_classes[i] for i in order]
        f1_sorted = [emo_f1[i] for i in order]

        ax1.barh(emo_sorted, f1_sorted)
        ax1.set_xlim(0.0, 1.0)
        ax1.set_title("Emotion per-class F1")
        ax1.set_xlabel("F1 score")
        ax1.set_ylabel("")
        ax1.grid(True, axis="x", alpha=0.25)

    # Optional compact suptitle.
    rf1 = _safe_float(metrics.get("risk_f1_macro"))
    ef1 = _safe_float(metrics.get("emotion_f1_macro"))
    racc = _safe_float(metrics.get("emotion_acc"))
    if args.show_suptitle and rf1 is not None and ef1 is not None and racc is not None:
        fig.suptitle(
            f"risk F1(macro)={rf1:.3f} • emotion F1(macro)={ef1:.3f} • emotion acc={racc:.3f}",
            fontsize=12,
        )

    out_stem = str(args.out_stem) if args.out_stem else f"paper_results_{args.name}_clean"
    png_path = out_dir / f"{out_stem}.png"
    pdf_path = out_dir / f"{out_stem}.pdf"
    fig.savefig(png_path, dpi=int(args.dpi), bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote: {png_path}")
    print(f"Wrote: {pdf_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
