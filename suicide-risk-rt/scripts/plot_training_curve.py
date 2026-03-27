from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


_EPOCH_RE = re.compile(r"val_metrics_epoch_(\d+)\.json$")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_epoch(path: Path) -> int | None:
    m = _EPOCH_RE.search(path.name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _auto_ylim(values: list[float], *, pad: float, hard_min: float, hard_max: float) -> tuple[float, float]:
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return hard_min, hard_max
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if vmax - vmin < pad:
        # If almost constant, widen slightly so the line is visible.
        vmin -= pad
        vmax += pad
    lo = max(hard_min, vmin - pad)
    hi = min(hard_max, vmax + pad)
    if hi <= lo:
        lo, hi = hard_min, hard_max
    return float(lo), float(hi)


def _demo_fill_series(
    epochs_obs: list[int],
    values_obs: list[float],
    *,
    fill_to: int,
    kind: str,
    noise: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a synthetic (demo) dense curve 1..fill_to.

    Returns:
      - x_dense: 1..fill_to
      - y_dense: filled values
      - x_obs: observed epochs
      - y_obs: observed values (nan allowed)

    This is intentionally *not* a projection of real training; it's for demos.
    """

    if fill_to < 1:
        raise ValueError("fill_to must be >= 1")

    x_obs = np.array(epochs_obs, dtype=float)
    y_obs = np.array(values_obs, dtype=float)
    x_dense = np.arange(1, fill_to + 1, dtype=float)

    # Keep only finite observed points for interpolation.
    mask = np.isfinite(x_obs) & np.isfinite(y_obs)
    xk = x_obs[mask]
    yk = y_obs[mask]
    if xk.size == 0:
        y_dense = np.full_like(x_dense, np.nan, dtype=float)
        return x_dense, y_dense, x_obs, y_obs

    # Sort, and drop duplicate epochs (keep last).
    order = np.argsort(xk)
    xk = xk[order]
    yk = yk[order]
    uniq_x: list[float] = []
    uniq_y: list[float] = []
    for xi, yi in zip(xk.tolist(), yk.tolist(), strict=False):
        if uniq_x and xi == uniq_x[-1]:
            uniq_y[-1] = yi
        else:
            uniq_x.append(xi)
            uniq_y.append(yi)
    xk = np.array(uniq_x, dtype=float)
    yk = np.array(uniq_y, dtype=float)

    if kind == "hold":
        y_dense = np.empty_like(x_dense, dtype=float)
        last = float(yk[0])
        j = 0
        for i, xi in enumerate(x_dense.tolist()):
            while j + 1 < xk.size and xi >= float(xk[j + 1]):
                j += 1
                last = float(yk[j])
            y_dense[i] = last
    elif kind == "linear":
        # Interpolate between observed points; hold last value after max observed.
        y_dense = np.interp(x_dense, xk, yk, left=float(yk[0]), right=float(yk[-1]))
    elif kind == "sat-exp":
        # A simple saturating curve anchored at the last observed value.
        # It rises quickly early on, then approaches the final observed value.
        y_start = float(yk[0])
        y_final = float(yk[-1])
        # Choose a timescale based on span (heuristic).
        span = max(1.0, float(xk[-1] - xk[0]))
        tau = max(1.0, span / 3.0)
        t = x_dense - float(xk[0])
        y_dense = y_final - (y_final - y_start) * np.exp(-t / tau)
        # Up to the last observed epoch, follow linear interpolation more closely.
        y_lin = np.interp(x_dense, xk, yk, left=y_start, right=y_final)
        cutoff = float(xk[-1])
        y_dense = np.where(x_dense <= cutoff, y_lin, y_dense)
    else:
        raise ValueError(f"Unknown kind: {kind}")

    if noise and noise > 0:
        rng = np.random.default_rng(int(seed))
        # Decay noise with epoch so it doesn't look too chaotic.
        decay = 1.0 / np.sqrt(np.maximum(1.0, x_dense))
        y_dense = y_dense + rng.normal(0.0, float(noise), size=y_dense.shape) * decay

    # Clip to metric bounds.
    y_dense = np.clip(y_dense, 0.0, 1.0)
    return x_dense, y_dense, x_obs, y_obs


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot training curves from val_metrics_epoch_*.json files.")
    ap.add_argument("--run-dir", default="outputs/final_2day", help="Folder containing val_metrics_epoch_*.json")
    ap.add_argument("--out-dir", default="reports/figures", help="Where to write figures")
    ap.add_argument("--out-stem", default="training_curve", help="Output filename stem")
    ap.add_argument(
        "--zoom-y",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Zoom y-axis around the observed metric range (makes small improvements visible).",
    )
    ap.add_argument(
        "--line-width",
        type=float,
        default=1.1,
        help="Line width for curves (smaller looks less like a thick bar).",
    )
    ap.add_argument(
        "--markers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show point markers at each epoch.",
    )
    ap.add_argument(
        "--f1-percent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Plot F1 as percent (F1×100).",
    )
    ap.add_argument(
        "--xmax",
        type=int,
        default=None,
        help="If set, force the x-axis to span epochs 1..xmax (plots only observed points; no projection).",
    )
    ap.add_argument(
        "--demo-fill-to",
        type=int,
        default=None,
        help="If set (e.g., 200), also write separate DEMO figures with synthetic filled epochs up to this value.",
    )
    ap.add_argument(
        "--demo-kind",
        choices=["linear", "hold", "sat-exp"],
        default="linear",
        help="How to fill missing epochs in demo mode.",
    )
    ap.add_argument(
        "--demo-noise",
        type=float,
        default=0.0,
        help="Optional small gaussian noise added in demo mode (e.g., 0.01).",
    )
    ap.add_argument(
        "--demo-seed",
        type=int,
        default=7,
        help="Random seed for demo noise (deterministic).",
    )
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    run_dir = (root / args.run_dir).resolve() if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
    out_dir = (root / args.out_dir).resolve() if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in run_dir.glob("val_metrics_epoch_*.json") if _extract_epoch(p) is not None], key=_extract_epoch)
    if not files:
        raise SystemExit(f"No val_metrics_epoch_*.json found in {run_dir}")

    epochs: list[int] = []
    risk_acc: list[float] = []
    emotion_acc: list[float] = []
    risk_f1_macro: list[float] = []
    emotion_f1_macro: list[float] = []

    for p in files:
        blob = _read_json(p)
        m = dict(blob.get("metrics", {}) or {})
        ep = _safe_float(m.get("epoch"))
        ep_i = int(ep) if ep is not None else _extract_epoch(p)
        if ep_i is None:
            continue

        epochs.append(ep_i)
        # Risk accuracy is nested under details.risk_report.accuracy.
        details = dict(blob.get("details", {}) or {})
        risk_report = dict(details.get("risk_report", {}) or {})
        ra = _safe_float(risk_report.get("accuracy"))
        risk_acc.append(float(ra) if ra is not None else float("nan"))

        ea = _safe_float(m.get("emotion_acc"))
        emotion_acc.append(float(ea) if ea is not None else float("nan"))

        rf1 = _safe_float(m.get("risk_f1_macro"))
        risk_f1_macro.append(float(rf1) if rf1 is not None else float("nan"))

        ef1 = _safe_float(m.get("emotion_f1_macro"))
        emotion_f1_macro.append(float(ef1) if ef1 is not None else float("nan"))

    x = np.array(epochs, dtype=float)

    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 110})

    def _save(fig, stem: str) -> None:
        png = out_dir / f"{stem}.png"
        pdf = out_dir / f"{stem}.pdf"
        fig.savefig(png, dpi=int(args.dpi), bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        print(f"Wrote: {png}")
        print(f"Wrote: {pdf}")

    # Accuracy curve (risk + emotion)
    fig1, ax = plt.subplots(figsize=(7.8, 4.2), constrained_layout=True)
    marker = "o" if args.markers else None
    ax.plot(x, risk_acc, marker=marker, markersize=3.0, linewidth=float(args.line_width), label="Risk accuracy")
    ax.plot(x, emotion_acc, marker=marker, markersize=3.0, linewidth=float(args.line_width), label="Emotion accuracy")
    ax.set_title("Validation accuracy vs epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    if args.zoom_y:
        lo, hi = _auto_ylim(risk_acc + emotion_acc, pad=0.01, hard_min=0.0, hard_max=1.0)
        ax.set_ylim(lo, hi)
    else:
        ax.set_ylim(0.0, 1.0)
    if args.xmax is not None:
        xmax = int(args.xmax)
        if len(x) > 0 and xmax < int(np.nanmax(x)):
            raise SystemExit(f"--xmax={xmax} is smaller than max observed epoch {int(np.nanmax(x))}")
        ax.set_xlim(1, xmax)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")

    # Annotate last points (kept minimal).
    if len(x) > 0:
        ax.annotate(f"{risk_acc[-1]:.3f}", (x[-1], risk_acc[-1]), textcoords="offset points", xytext=(6, -10))
        ax.annotate(f"{emotion_acc[-1]:.3f}", (x[-1], emotion_acc[-1]), textcoords="offset points", xytext=(6, 6))

    _save(fig1, args.out_stem + "_accuracy")
    plt.close(fig1)

    # Macro F1 curve (risk + emotion)
    f1_scale = 100.0 if bool(args.f1_percent) else 1.0
    risk_f1_plot = (np.array(risk_f1_macro, dtype=float) * f1_scale).tolist()
    emo_f1_plot = (np.array(emotion_f1_macro, dtype=float) * f1_scale).tolist()

    fig2, ax2 = plt.subplots(figsize=(7.8, 4.2), constrained_layout=True)
    ax2.plot(x, risk_f1_plot, marker=marker, markersize=3.0, linewidth=float(args.line_width), label="Risk F1 (macro)")
    ax2.plot(x, emo_f1_plot, marker=marker, markersize=3.0, linewidth=float(args.line_width), label="Emotion F1 (macro)")
    ax2.set_title("Validation F1 (macro) vs epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 (macro, %)" if args.f1_percent else "F1 (macro)")
    if args.zoom_y:
        lo, hi = _auto_ylim(risk_f1_plot + emo_f1_plot, pad=(1.0 if args.f1_percent else 0.01), hard_min=0.0, hard_max=(100.0 if args.f1_percent else 1.0))
        ax2.set_ylim(lo, hi)
    else:
        ax2.set_ylim(0.0, 100.0 if args.f1_percent else 1.0)
    if args.xmax is not None:
        xmax = int(args.xmax)
        if len(x) > 0 and xmax < int(np.nanmax(x)):
            raise SystemExit(f"--xmax={xmax} is smaller than max observed epoch {int(np.nanmax(x))}")
        ax2.set_xlim(1, xmax)
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="lower right")

    if len(x) > 0:
        fmt = ("{:.1f}" if args.f1_percent else "{:.3f}")
        ax2.annotate(fmt.format(risk_f1_plot[-1]), (x[-1], risk_f1_plot[-1]), textcoords="offset points", xytext=(6, -10))
        ax2.annotate(fmt.format(emo_f1_plot[-1]), (x[-1], emo_f1_plot[-1]), textcoords="offset points", xytext=(6, 6))

    _save(fig2, args.out_stem + "_f1")
    plt.close(fig2)

    # Optional demo figures with synthetic filled epochs.
    if args.demo_fill_to is not None:
        fill_to = int(args.demo_fill_to)
        if fill_to < 1:
            raise SystemExit("--demo-fill-to must be >= 1")

        def _demo_plot_pair(
            *,
            title: str,
            y1: list[float],
            y2: list[float],
            label1: str,
            label2: str,
            ylabel: str,
            out_suffix: str,
        ) -> None:
            xd1, yd1, xo, yo1 = _demo_fill_series(
                epochs,
                y1,
                fill_to=fill_to,
                kind=str(args.demo_kind),
                noise=float(args.demo_noise),
                seed=int(args.demo_seed) + 11,
            )
            xd2, yd2, _, yo2 = _demo_fill_series(
                epochs,
                y2,
                fill_to=fill_to,
                kind=str(args.demo_kind),
                noise=float(args.demo_noise),
                seed=int(args.demo_seed) + 23,
            )

            fig, axd = plt.subplots(figsize=(7.8, 4.2), constrained_layout=True)
            scale = 100.0 if (out_suffix == "f1" and bool(args.f1_percent)) else 1.0
            axd.plot(xd1, yd1 * scale, linewidth=float(args.line_width), alpha=0.9, label=f"{label1}")
            axd.plot(xd2, yd2 * scale, linewidth=float(args.line_width), alpha=0.9, label=f"{label2}")
            # Observed points on top
            axd.scatter(xo, np.array(yo1, dtype=float) * scale, s=12, alpha=0.75)
            axd.scatter(xo, np.array(yo2, dtype=float) * scale, s=12, alpha=0.75)

            axd.set_title(title)
            axd.set_xlabel("Epoch")
            if out_suffix == "f1" and bool(args.f1_percent):
                axd.set_ylabel("F1 (macro, %)")
            else:
                axd.set_ylabel(ylabel)

            if args.zoom_y:
                hard_max = 100.0 if (out_suffix == "f1" and bool(args.f1_percent)) else 1.0
                pad = 1.0 if (out_suffix == "f1" and bool(args.f1_percent)) else 0.01
                lo, hi = _auto_ylim((yd1 * scale).tolist() + (yd2 * scale).tolist(), pad=pad, hard_min=0.0, hard_max=hard_max)
                axd.set_ylim(lo, hi)
            else:
                axd.set_ylim(0.0, 100.0 if (out_suffix == "f1" and bool(args.f1_percent)) else 1.0)
            axd.set_xlim(1, fill_to)
            axd.grid(True, alpha=0.25)
            axd.legend(loc="lower right", ncols=1)
            axd.text(
                0.01,
                0.01,
                "DEMO curve (filled epochs)",
                transform=axd.transAxes,
                fontsize=9,
                alpha=0.75,
                va="bottom",
            )
            _save(fig, args.out_stem + f"_demo_{out_suffix}")
            plt.close(fig)

        _demo_plot_pair(
            title="Validation accuracy vs epoch",
            y1=risk_acc,
            y2=emotion_acc,
            label1="Risk accuracy",
            label2="Emotion accuracy",
            ylabel="Accuracy",
            out_suffix="accuracy",
        )
        _demo_plot_pair(
            title="Validation F1 (macro) vs epoch",
            y1=risk_f1_macro,
            y2=emotion_f1_macro,
            label1="Risk F1 (macro)",
            label2="Emotion F1 (macro)",
            ylabel="F1 (macro)",
            out_suffix="f1",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
