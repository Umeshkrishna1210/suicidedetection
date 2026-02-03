from __future__ import annotations

import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import argparse
import csv
import json
from pathlib import Path

from srrtd.utils.config import apply_overrides, load_yaml


def _flatten_overrides(d: dict, prefix: str = "") -> list[str]:
    out: list[str] = []
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.extend(_flatten_overrides(v, prefix=key))
        else:
            if isinstance(v, bool):
                out.append(f"{key}={'true' if v else 'false'}")
            else:
                out.append(f"{key}={v}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overrides", nargs="*", default=None)
    ap.add_argument("--out", default="outputs/experiments")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args.overrides)

    root = Path(__file__).resolve().parents[1]
    out_dir = (root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp = dict(cfg.get("experiments", {}) or {})
    baselines = list(exp.get("baselines", []) or [])
    ablations = list(exp.get("ablations", []) or [])

    results_csv = out_dir / "results.csv"
    rows = []

    base_cfg = load_yaml(args.config)
    base_cfg = apply_overrides(base_cfg, args.overrides)
    rag_index_built = False

    def run_one(name: str, overrides: list[str]):
        import subprocess
        import sys

        run_name = name
        merged_overrides = (args.overrides or []) + (overrides or [])

        # Build RAG index once if any run needs it.
        nonlocal rag_index_built
        variant_cfg = apply_overrides(load_yaml(args.config), merged_overrides)
        use_rag = bool(variant_cfg.get("model", {}).get("use_rag", False))
        if use_rag and not rag_index_built:
            rag_index_built = True
            idx_cmd = [sys.executable, str(root / "scripts" / "build_rag_index.py"), "--config", args.config]
            if merged_overrides:
                idx_cmd += ["--overrides", *merged_overrides]
            subprocess.check_call(idx_cmd, cwd=str(root))

        train_cmd = [sys.executable, str(root / "scripts" / "train.py"), "--config", args.config, "--run-name", run_name]
        if merged_overrides:
            train_cmd += ["--overrides", *merged_overrides]

        subprocess.check_call(train_cmd, cwd=str(root))

        ckpt = str((root / "outputs" / run_name / "best.pt").resolve())
        eval_out = str((out_dir / run_name).resolve())
        eval_cmd = [sys.executable, str(root / "scripts" / "evaluate.py"), "--config", args.config, "--ckpt", ckpt, "--out", eval_out]
        if merged_overrides:
            eval_cmd += ["--overrides", *merged_overrides]

        subprocess.check_call(eval_cmd, cwd=str(root))
        with open(Path(eval_out) / "metrics.json", "r", encoding="utf-8") as f:
            m = json.load(f)["metrics"]
        return m

    for b in baselines:
        name = str(b.get("name"))
        overrides = []
        b2 = dict(b)
        b2.pop("name", None)
        overrides.extend(_flatten_overrides({"model": b2}))
        m = run_one(name, overrides)
        rows.append({"name": name, **m})

    for a in ablations:
        name = str(a.get("name"))
        a2 = dict(a)
        a2.pop("name", None)
        overrides = _flatten_overrides({"model": a2})
        m = run_one(name, overrides)
        rows.append({"name": name, **m})

    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["name"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"OK: wrote {results_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
