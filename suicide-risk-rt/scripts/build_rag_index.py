from __future__ import annotations

import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import argparse
from pathlib import Path

from srrtd.data.loader import DatasetConfirmationRequired, load_dataset_bundle
from srrtd.rag.indexer import RagIndexConfig, build_rag_index
from srrtd.utils.config import apply_overrides, load_yaml
from srrtd.utils.seed import set_global_seed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overrides", nargs="*", default=None)
    ap.add_argument("--no-reset", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args.overrides)

    root = Path(__file__).resolve().parents[1]
    seed = int(cfg.get("project", {}).get("seed", 42))
    set_global_seed(seed)

    try:
        bundle = load_dataset_bundle(cfg, root)
    except DatasetConfirmationRequired as e:
        raise SystemExit(str(e))

    rag_cfg = cfg.get("model", {}).get("rag", {}) or {}
    idx_cfg = RagIndexConfig(
        embed_model=str(rag_cfg.get("embed_model")),
        chroma_dir=str(rag_cfg.get("chroma_dir")),
        collection=str(rag_cfg.get("collection")),
    )

    info = build_rag_index(idx_cfg, root, bundle=bundle, reset=(not args.no_reset))
    print(f"OK: indexed={info['count']} embed_dim={info['embed_dim']} collection={info['collection']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
