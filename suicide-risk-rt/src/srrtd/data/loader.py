from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path
from typing import Any

from srrtd.data.cache import cache_path, read_bundle, write_bundle
from srrtd.data.schema import DatasetBundle, DatasetSplit, PostRecord
from srrtd.data.toy import toy_splits
from srrtd.utils.privacy import privacy_cfg_from_dict, privacy_preprocess


class DatasetConfirmationRequired(RuntimeError):
    pass


def _cleaning_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    d = dict(cfg.get("data", {}) or {})
    c = dict(d.get("cleaning", {}) or {})
    return {
        "drop_deleted": bool(c.get("drop_deleted", True)),
        "min_text_len": int(c.get("min_text_len", 10)),
        "deduplicate": bool(c.get("deduplicate", False)),
    }


def _is_deleted_placeholder(text: str) -> bool:
    s = text.strip().lower()
    return s in {"[deleted]", "[removed]", "deleted", "removed"}


def _normalize_for_dedupe(text: str) -> str:
    # Minimal, stable normalization for dedupe keys.
    return " ".join(str(text).split()).strip().lower()


def _require_confirmation(data_cfg: dict[str, Any]) -> None:
    mode = str(data_cfg.get("mode", "single")).lower()
    if mode == "single":
        source = str(data_cfg.get("source", "toy"))
        if source == "toy":
            return
    elif mode == "multitask":
        # multitask always implies external/local data
        pass
    else:
        raise ValueError(f"Unknown data.mode='{mode}'")
    confirmed = bool(data_cfg.get("dataset_confirmed", False))
    if not confirmed:
        raise DatasetConfirmationRequired(
            "External dataset usage blocked. Set data.dataset_confirmed=true only after you explicitly confirm: "
            "dataset name, source, expected size, and license/access requirements."
        )


def _stable_id(s: str) -> str:
    h = hashlib.sha1(s.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
    return h


def load_multitask_bundles(cfg: dict[str, Any], project_root: Path) -> tuple[DatasetBundle, DatasetBundle]:
    """Load two separate datasets for multitask learning.

    Returns: (risk_bundle, emotion_bundle)
    """

    data_cfg = dict(cfg.get("data", {}) or {})
    _require_confirmation(data_cfg)

    privacy_cfg = privacy_cfg_from_dict(dict(cfg.get("privacy", {}) or {}))
    labels_cfg = dict(cfg.get("labels", {}) or {})

    risk_classes = list(labels_cfg.get("risk_classes", ["low", "medium", "high"]))
    emotion_classes = list(
        labels_cfg.get(
            "emotion_classes",
            ["neutral", "sadness", "anger", "fear", "joy", "surprise", "disgust"],
        )
    )

    mt = dict(data_cfg.get("multitask", {}) or {})
    risk_cfg = dict(mt.get("risk", {}) or {})
    emo_cfg = dict(mt.get("emotion", {}) or {})

    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Cache per task
    common_cache_bits = {
        "mode": "multitask",
        "privacy": asdict(privacy_cfg),
        "labels": {"risk": risk_classes, "emotion": emotion_classes},
    }
    risk_key = {**common_cache_bits, "task": "risk", "cfg": {k: risk_cfg.get(k) for k in sorted(risk_cfg.keys())}}
    emo_key = {**common_cache_bits, "task": "emotion", "cfg": {k: emo_cfg.get(k) for k in sorted(emo_cfg.keys())}}

    risk_path = cache_path(processed_dir, risk_key)
    emo_path = cache_path(processed_dir, emo_key)
    if risk_path.exists() and emo_path.exists():
        return read_bundle(risk_path), read_bundle(emo_path)

    risk_bundle = _load_multitask_task_bundle(
        cfg=cfg,
        project_root=project_root,
        task="risk",
        task_cfg=risk_cfg,
        risk_classes=risk_classes,
        emotion_classes=emotion_classes,
    )
    emo_bundle = _load_multitask_task_bundle(
        cfg=cfg,
        project_root=project_root,
        task="emotion",
        task_cfg=emo_cfg,
        risk_classes=risk_classes,
        emotion_classes=emotion_classes,
    )

    write_bundle(risk_path, risk_bundle)
    write_bundle(emo_path, emo_bundle)
    return risk_bundle, emo_bundle


def _infer_format(path: str, explicit: str | None) -> str:
    if explicit and str(explicit).strip():
        return str(explicit).strip().lower()
    suf = Path(path).suffix.lower()
    if suf == ".csv":
        return "csv"
    if suf in (".jsonl", ".json"):
        return "jsonl"
    if suf in (".xlsx", ".xls"):
        return "xlsx"
    raise ValueError(f"Cannot infer file format from extension '{suf}' for path='{path}'")


def _read_table(path: str, fmt: str):
    import pandas as pd

    if fmt == "csv":
        encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
        last_err: Exception | None = None
        for enc in encodings:
            try:
                return pd.read_csv(path, encoding=enc, engine="python", on_bad_lines="skip")
            except UnicodeDecodeError as e:
                last_err = e
        if last_err is not None:
            raise last_err
        return pd.read_csv(path, engine="python", on_bad_lines="skip")
    if fmt == "jsonl":
        # json lines
        return pd.read_json(path, lines=True)
    if fmt == "xlsx":
        return pd.read_excel(path)
    raise ValueError(f"Unsupported format '{fmt}'")


def _maybe_cap(df, max_samples: int, seed: int):
    if max_samples is None:
        return df
    m = int(max_samples)
    if m <= 0 or len(df) <= m:
        return df
    return df.sample(n=m, random_state=seed).reset_index(drop=True)


def _map_risk_label(val: Any, label_map: dict[str, Any] | None) -> int:
    if val is None:
        raise ValueError("Risk label is missing")
    # numeric
    try:
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return int(val)
        s = str(val).strip()
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            return int(s)
    except Exception:
        pass

    s = str(val).strip().lower()
    if label_map:
        if s in label_map:
            return int(label_map[s])
        # allow original-case keys
        for k, v in label_map.items():
            if str(k).strip().lower() == s:
                return int(v)

    # conservative heuristics for common suicide datasets
    suicide_tokens = {
        "suicide",
        "suicidal",
        "suicidewatch",
        "self.suicidewatch",
        "sw",
        "1",
        "true",
        "yes",
        "positive",
    }
    nonsuicide_tokens = {
        "non-suicide",
        "nonsuicide",
        "non_suicide",
        "not_suicide",
        "0",
        "false",
        "no",
        "negative",
        "depression",
        "self.depression",
    }
    if s in suicide_tokens:
        return 1
    if s in nonsuicide_tokens:
        return 0

    raise ValueError(
        f"Unrecognized risk label '{val}'. Provide data.multitask.risk.label_map to map string labels -> integers."
    )


def _map_emotion_label(val: Any, emotion_classes: list[str], label_map: dict[str, Any] | None) -> int:
    if val is None:
        raise ValueError("Emotion label is missing")
    try:
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return int(val)
        s = str(val).strip()
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            return int(s)
    except Exception:
        pass

    s = str(val).strip().lower()
    if label_map:
        if s in label_map:
            return int(label_map[s])
        for k, v in label_map.items():
            if str(k).strip().lower() == s:
                return int(v)

    # fall back to labels.emotion_classes order
    norm = [c.strip().lower() for c in emotion_classes]
    if s in norm:
        return int(norm.index(s))

    raise ValueError(
        f"Unrecognized emotion label '{val}'. Add it to labels.emotion_classes or provide data.multitask.emotion.label_map."
    )


def _load_multitask_task_bundle(
    cfg: dict[str, Any],
    project_root: Path,
    task: str,
    task_cfg: dict[str, Any],
    risk_classes: list[str],
    emotion_classes: list[str],
) -> DatasetBundle:
    seed = int(cfg.get("project", {}).get("seed", 42))
    clean_cfg = _cleaning_cfg(cfg)

    train_path = str(task_cfg.get("train_path", "")).strip()
    val_path = str(task_cfg.get("val_path", "")).strip()
    test_path = str(task_cfg.get("test_path", "")).strip()
    if not train_path or not val_path or not test_path:
        raise ValueError(f"data.multitask.{task}.train_path/val_path/test_path are required")

    # resolve relative to project root
    def rp(p: str) -> str:
        pp = Path(p)
        return str((project_root / pp).resolve()) if not pp.is_absolute() else str(pp)

    train_path = rp(train_path)
    val_path = rp(val_path)
    test_path = rp(test_path)

    fmt_train = _infer_format(train_path, task_cfg.get("format"))
    fmt_val = _infer_format(val_path, task_cfg.get("format"))
    fmt_test = _infer_format(test_path, task_cfg.get("format"))
    if not (fmt_train == fmt_val == fmt_test):
        raise ValueError(f"Split formats must match for task '{task}'")
    fmt = fmt_train

    text_field = str(task_cfg.get("text_field", "text"))
    label_field = str(task_cfg.get("label_field", "label"))
    user_field = str(task_cfg.get("user_field", "user_id"))
    time_field = str(task_cfg.get("time_field", "timestamp"))

    max_train = int(task_cfg.get("max_train_samples", -1))
    max_val = int(task_cfg.get("max_val_samples", -1))
    max_test = int(task_cfg.get("max_test_samples", -1))

    label_map = dict(task_cfg.get("label_map", {}) or {})

    def to_records(df, split_name: str) -> list[PostRecord]:
        if text_field not in df.columns or label_field not in df.columns:
            raise ValueError(f"Missing required columns for {task}/{split_name}: '{text_field}', '{label_field}'")

        recs: list[PostRecord] = []
        seen: set[str] = set()
        for i, row in df.iterrows():
            raw_text = row[text_field]
            if raw_text is None:
                continue
            # pandas NaN -> float nan
            try:
                import pandas as _pd

                if _pd.isna(raw_text):
                    continue
            except Exception:
                pass

            text = str(raw_text).strip()
            if not text:
                continue
            if clean_cfg["drop_deleted"] and _is_deleted_placeholder(text):
                continue
            if len(text) < int(clean_cfg["min_text_len"]):
                continue
            if clean_cfg["deduplicate"]:
                key = _normalize_for_dedupe(text)
                if key in seen:
                    continue
                seen.add(key)

            raw_label = row[label_field]

            if task == "risk":
                risk = _map_risk_label(raw_label, label_map)
                # clip into available head size (commonly binary)
                risk = max(0, min(len(risk_classes) - 1, int(risk)))
                emotion = 0
            else:
                emotion = _map_emotion_label(raw_label, emotion_classes, label_map)
                emotion = max(0, min(len(emotion_classes) - 1, int(emotion)))
                risk = 0

            # If missing, create stable ids so memory/GNN can still function.
            user_id = str(row[user_field]) if user_field in df.columns else f"user_{_stable_id(task + ':' + split_name + ':' + str(i))}"
            ts = int(row[time_field]) if time_field in df.columns else 1_700_000_000 + int(i)
            lang = str(row["lang"]) if "lang" in df.columns else "und"
            recs.append(
                PostRecord(
                    text=text,
                    risk=risk,
                    emotion=emotion,
                    user_id=user_id,
                    timestamp=ts,
                    lang=lang,
                    meta={"task": task, "split": split_name},
                )
            )
        return recs

    df_train = _read_table(train_path, fmt)
    df_val = _read_table(val_path, fmt)
    df_test = _read_table(test_path, fmt)

    df_train = _maybe_cap(df_train, max_train, seed)
    df_val = _maybe_cap(df_val, max_val, seed)
    df_test = _maybe_cap(df_test, max_test, seed)

    train_records = to_records(df_train, "train")
    val_records = to_records(df_val, "val")
    test_records = to_records(df_test, "test")

    privacy_cfg = privacy_cfg_from_dict(dict(cfg.get("privacy", {}) or {}))
    for r in train_records:
        r.text = privacy_preprocess(r.text, privacy_cfg)
    for r in val_records:
        r.text = privacy_preprocess(r.text, privacy_cfg)
    for r in test_records:
        r.text = privacy_preprocess(r.text, privacy_cfg)

    return DatasetBundle(
        train=DatasetSplit(records=train_records),
        val=DatasetSplit(records=val_records),
        test=DatasetSplit(records=test_records),
        risk_classes=risk_classes,
        emotion_classes=emotion_classes,
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

    # Clean + privacy preprocess text before caching
    clean_cfg = _cleaning_cfg(cfg)
    for split in (bundle.train, bundle.val, bundle.test):
        cleaned: list[PostRecord] = []
        seen: set[str] = set()
        for r in split.records:
            t = str(r.text).strip()
            if not t:
                continue
            if clean_cfg["drop_deleted"] and _is_deleted_placeholder(t):
                continue
            if len(t) < int(clean_cfg["min_text_len"]):
                continue
            if clean_cfg["deduplicate"]:
                key = _normalize_for_dedupe(t)
                if key in seen:
                    continue
                seen.add(key)
            r.text = privacy_preprocess(t, privacy_cfg)
            cleaned.append(r)
        split.records = cleaned

    # Note: multitask path applies cleaning and privacy preprocessing inside its loader.

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
