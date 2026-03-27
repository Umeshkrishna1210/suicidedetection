from __future__ import annotations

import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import argparse
from pathlib import Path
from typing import Literal

from srrtd.data.loader import DatasetConfirmationRequired, load_dataset_bundle, load_multitask_bundles
from srrtd.data.schema import DatasetBundle, DatasetSplit
from srrtd.rag.indexer import RagIndexConfig, build_rag_index
from srrtd.utils.config import apply_overrides, load_yaml
from srrtd.utils.seed import set_global_seed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overrides", nargs="*", default=None)
    ap.add_argument("--no-reset", action="store_true")
    ap.add_argument(
        "--splits",
        default="train",
        choices=["train", "all"],
        help="Which splits to index. Use 'train' to avoid evaluation leakage (default).",
    )
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args.overrides)

    # If hard negatives are enabled for training via extra_train_paths + extra_train_repeat,
    # we do NOT want to index repeated duplicates into RAG. Clamp repeat to 1 for indexing.
    try:
        data_cfg = cfg.get("data", {}) or {}
        if str(data_cfg.get("mode", "single")).lower() == "multitask":
            mt = data_cfg.get("multitask", {}) or {}
            risk = mt.get("risk", {}) or {}
            if "extra_train_repeat" in risk:
                r = int(risk.get("extra_train_repeat", 1))
                if r != 1:
                    risk["extra_train_repeat"] = 1
    except Exception:
        pass

    root = Path(__file__).resolve().parents[1]
    seed = int(cfg.get("project", {}).get("seed", 42))
    set_global_seed(seed)

    def _splits_to_index(s: str) -> Literal["train", "all"]:
        s2 = str(s).strip().lower()
        return "all" if s2 == "all" else "train"

    splits = _splits_to_index(args.splits)

    try:
        mode = str(cfg.get("data", {}).get("mode", "single")).lower()
        if mode == "multitask":
            risk_bundle, emo_bundle = load_multitask_bundles(cfg, root)
            train_recs = list(risk_bundle.train.records) + list(emo_bundle.train.records)
            if splits == "all":
                val_recs = list(risk_bundle.val.records) + list(emo_bundle.val.records)
                test_recs = list(risk_bundle.test.records) + list(emo_bundle.test.records)
            else:
                val_recs = []
                test_recs = []
            bundle = DatasetBundle(
                train=DatasetSplit(records=train_recs),
                val=DatasetSplit(records=val_recs),
                test=DatasetSplit(records=test_recs),
                risk_classes=list(risk_bundle.risk_classes),
                emotion_classes=list(emo_bundle.emotion_classes),
            )
        else:
            b = load_dataset_bundle(cfg, root)
            if splits == "all":
                bundle = b
            else:
                bundle = DatasetBundle(
                    train=DatasetSplit(records=list(b.train.records)),
                    val=DatasetSplit(records=[]),
                    test=DatasetSplit(records=[]),
                    risk_classes=list(b.risk_classes),
                    emotion_classes=list(b.emotion_classes),
                )
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
