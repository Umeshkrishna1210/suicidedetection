from __future__ import annotations

import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import argparse
from pathlib import Path

from srrtd.data.loader import DatasetConfirmationRequired, load_dataset_bundle
from srrtd.utils.config import apply_overrides, load_yaml, resolve_paths
from srrtd.utils.seed import set_global_seed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overrides", nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args.overrides)

    root = Path(__file__).resolve().parents[1]
    _paths = resolve_paths(cfg, root)

    seed = int(cfg.get("project", {}).get("seed", 42))
    set_global_seed(seed)

    try:
        bundle = load_dataset_bundle(cfg, root)
    except DatasetConfirmationRequired as e:
        raise SystemExit(str(e))

    print(f"OK: train={len(bundle.train.records)} val={len(bundle.val.records)} test={len(bundle.test.records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
