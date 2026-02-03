from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from srrtd.data.cache import cache_path, read_bundle, write_bundle
from srrtd.data.schema import DatasetBundle, DatasetSplit, PostRecord
from srrtd.data.toy import toy_splits
from srrtd.utils.privacy import privacy_cfg_from_dict, privacy_preprocess


class DatasetConfirmationRequired(RuntimeError):
    pass


def _require_confirmation(data_cfg: dict[str, Any]) -> None:
    source = str(data_cfg.get("source", "toy"))
    if source == "toy":
        return
    confirmed = bool(data_cfg.get("dataset_confirmed", False))
    if not confirmed:
        raise DatasetConfirmationRequired(
            "External dataset usage blocked. Set data.dataset_confirmed=true only after you explicitly confirm: "
            "dataset name, source, expected size, and license/access requirements."
        )


def load_dataset_bundle(cfg: dict[str, Any], project_root: Path) -> DatasetBundle:
    data_cfg = dict(cfg.get("data", {}) or {})
    privacy_cfg = privacy_cfg_from_dict(dict(cfg.get("privacy", {}) or {}))
    labels_cfg = dict(cfg.get("labels", {}) or {})
    train_cfg = dict(cfg.get("train", {}) or {})

    risk_classes = list(labels_cfg.get("risk_classes", ["low", "medium", "high"]))
    emotion_classes = list(
        labels_cfg.get("emotion_classes", ["neutral", "sadness", "anger", "fear", "joy", "surprise", "disgust"])
    )

    val_ratio = float(train_cfg.get("val_ratio", 0.1))
    test_ratio = float(train_cfg.get("test_ratio", 0.1))

    source = str(data_cfg.get("source", "toy")).lower()

    # Cache key includes privacy+labels+split ratios+source details
    cache_key = {
        "source": source,
        "data": {k: data_cfg.get(k) for k in sorted(data_cfg.keys())},
        "privacy": asdict(privacy_cfg),
        "labels": {"risk": risk_classes, "emotion": emotion_classes},
        "splits": {"val_ratio": val_ratio, "test_ratio": test_ratio},
    }

    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    cpath = cache_path(processed_dir, cache_key)
    if cpath.exists():
        return read_bundle(cpath)

    if source == "toy":
        bundle = toy_splits(
            seed=int(cfg.get("project", {}).get("seed", 42)),
            risk_classes=risk_classes,
            emotion_classes=emotion_classes,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
    else:
        _require_confirmation(data_cfg)
        if source == "hf":
            bundle = _load_from_hf(cfg, risk_classes=risk_classes, emotion_classes=emotion_classes, val_ratio=val_ratio, test_ratio=test_ratio)
        elif source == "csv":
            bundle = _load_from_csv(cfg, risk_classes=risk_classes, emotion_classes=emotion_classes, val_ratio=val_ratio, test_ratio=test_ratio)
        elif source == "jsonl":
            bundle = _load_from_jsonl(cfg, risk_classes=risk_classes, emotion_classes=emotion_classes, val_ratio=val_ratio, test_ratio=test_ratio)
        else:
            raise ValueError(f"Unknown data.source='{source}'")

    # Privacy preprocess text before caching
    for split in (bundle.train, bundle.val, bundle.test):
        for r in split.records:
            r.text = privacy_preprocess(r.text, privacy_cfg)

    write_bundle(cpath, bundle)
    return bundle


def _load_from_hf(
    cfg: dict[str, Any],
    risk_classes: list[str],
    emotion_classes: list[str],
    val_ratio: float,
    test_ratio: float,
) -> DatasetBundle:
    data_cfg = dict(cfg.get("data", {}) or {})
    name = str(data_cfg.get("hf_dataset_name", "")).strip()
    if not name:
        raise ValueError("data.hf_dataset_name is required for data.source=hf")

    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError("HuggingFace 'datasets' not installed") from e

    ds = load_dataset(name, data_cfg.get("hf_dataset_config") or None)
    # Expect a single split or 'train'
    if "train" in ds:
        full = ds["train"]
    else:
        # take first split
        first_key = list(ds.keys())[0]
        full = ds[first_key]

    text_field = str(data_cfg.get("hf_text_field", "text"))
    label_field = str(data_cfg.get("hf_label_field", "label"))
    user_field = str(data_cfg.get("user_field", "user_id"))
    time_field = str(data_cfg.get("time_field", "timestamp"))

    # If fields missing, fall back to defaults
    def get(x: dict[str, Any], k: str, default: Any) -> Any:
        return x[k] if k in x else default

    records: list[PostRecord] = []
    for i, row in enumerate(full):
        text = str(get(row, text_field, ""))
        label = int(get(row, label_field, 0))
        # If label space differs, map to 3-class (0/1/2) by clipping
        risk = max(0, min(2, int(label)))
        emotion = 0
        user_id = str(get(row, user_field, f"user_{i:08d}"))
        ts = int(get(row, time_field, 1_700_000_000 + i))
        lang = str(get(row, "lang", "und"))
        records.append(PostRecord(text=text, risk=risk, emotion=emotion, user_id=user_id, timestamp=ts, lang=lang, meta={"hf": name}))

    return _simple_splits(records, risk_classes, emotion_classes, val_ratio, test_ratio, seed=int(cfg.get("project", {}).get("seed", 42)))


def _load_from_csv(
    cfg: dict[str, Any],
    risk_classes: list[str],
    emotion_classes: list[str],
    val_ratio: float,
    test_ratio: float,
) -> DatasetBundle:
    import pandas as pd

    data_cfg = dict(cfg.get("data", {}) or {})
    path = str(data_cfg.get("path", "")).strip()
    if not path:
        raise ValueError("data.path is required for data.source=csv")

    df = pd.read_csv(path)
    return _load_from_frame(df, cfg, risk_classes, emotion_classes, val_ratio, test_ratio, source_meta={"csv": path})


def _load_from_jsonl(
    cfg: dict[str, Any],
    risk_classes: list[str],
    emotion_classes: list[str],
    val_ratio: float,
    test_ratio: float,
) -> DatasetBundle:
    import json

    data_cfg = dict(cfg.get("data", {}) or {})
    path = str(data_cfg.get("path", "")).strip()
    if not path:
        raise ValueError("data.path is required for data.source=jsonl")

    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    import pandas as pd

    df = pd.DataFrame(rows)
    return _load_from_frame(df, cfg, risk_classes, emotion_classes, val_ratio, test_ratio, source_meta={"jsonl": path})


def _load_from_frame(
    df,
    cfg: dict[str, Any],
    risk_classes: list[str],
    emotion_classes: list[str],
    val_ratio: float,
    test_ratio: float,
    source_meta: dict[str, Any],
) -> DatasetBundle:
    data_cfg = dict(cfg.get("data", {}) or {})
    text_field = str(data_cfg.get("text_field", "text"))
    label_field = str(data_cfg.get("label_field", "label"))
    user_field = str(data_cfg.get("user_field", "user_id"))
    time_field = str(data_cfg.get("time_field", "timestamp"))

    if text_field not in df.columns or label_field not in df.columns:
        raise ValueError(f"Missing required columns: '{text_field}', '{label_field}'")

    records: list[PostRecord] = []
    for i, row in df.iterrows():
        text = str(row[text_field])
        risk = max(0, min(2, int(row[label_field])))
        emotion = int(row["emotion"]) if "emotion" in df.columns else 0
        emotion = max(0, min(len(emotion_classes) - 1, int(emotion)))
        user_id = str(row[user_field]) if user_field in df.columns else f"user_{int(i):08d}"
        ts = int(row[time_field]) if time_field in df.columns else 1_700_000_000 + int(i)
        lang = str(row["lang"]) if "lang" in df.columns else "und"
        records.append(PostRecord(text=text, risk=risk, emotion=emotion, user_id=user_id, timestamp=ts, lang=lang, meta=dict(source_meta)))

    return _simple_splits(records, risk_classes, emotion_classes, val_ratio, test_ratio, seed=int(cfg.get("project", {}).get("seed", 42)))


def _simple_splits(
    records: list[PostRecord],
    risk_classes: list[str],
    emotion_classes: list[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> DatasetBundle:
    import random

    rng = random.Random(seed)
    rng.shuffle(records)
    n = len(records)
    n_test = max(1, int(round(n * test_ratio)))
    n_val = max(1, int(round(n * val_ratio)))

    test = records[:n_test]
    val = records[n_test : n_test + n_val]
    train = records[n_test + n_val :]

    return DatasetBundle(
        train=DatasetSplit(records=train),
        val=DatasetSplit(records=val),
        test=DatasetSplit(records=test),
        risk_classes=risk_classes,
        emotion_classes=emotion_classes,
    )
