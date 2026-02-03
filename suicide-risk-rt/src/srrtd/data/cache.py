from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from srrtd.data.schema import DatasetBundle, DatasetSplit, PostRecord


def _hash_obj(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:16]


def cache_path(data_processed_dir: Path, cache_key: dict[str, Any]) -> Path:
    h = _hash_obj(cache_key)
    return data_processed_dir / f"dataset_{h}.jsonl.gz"


def _rec_to_dict(r: PostRecord) -> dict[str, Any]:
    return {
        "text": r.text,
        "risk": int(r.risk),
        "emotion": int(r.emotion),
        "user_id": r.user_id,
        "timestamp": int(r.timestamp),
        "lang": r.lang,
        "meta": r.meta or {},
    }


def _dict_to_rec(d: dict[str, Any]) -> PostRecord:
    return PostRecord(
        text=str(d.get("text", "")),
        risk=int(d.get("risk", 0)),
        emotion=int(d.get("emotion", 0)),
        user_id=str(d.get("user_id", "")),
        timestamp=int(d.get("timestamp", 0)),
        lang=str(d.get("lang", "und")),
        meta=dict(d.get("meta", {}) or {}),
    )


def write_bundle(path: Path, bundle: DatasetBundle) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        header = {
            "_type": "header",
            "risk_classes": bundle.risk_classes,
            "emotion_classes": bundle.emotion_classes,
        }
        f.write(json.dumps(header, ensure_ascii=False) + "\n")
        for split_name, split in ("train", bundle.train), ("val", bundle.val), ("test", bundle.test):
            for r in split.records:
                row = _rec_to_dict(r)
                row["_type"] = split_name
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_bundle(path: Path) -> DatasetBundle:
    risk_classes: list[str] = []
    emotion_classes: list[str] = []
    train: list[PostRecord] = []
    val: list[PostRecord] = []
    test: list[PostRecord] = []

    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            t = d.get("_type")
            if t == "header":
                risk_classes = list(d.get("risk_classes", []))
                emotion_classes = list(d.get("emotion_classes", []))
                continue
            rec = _dict_to_rec(d)
            if t == "train":
                train.append(rec)
            elif t == "val":
                val.append(rec)
            elif t == "test":
                test.append(rec)

    return DatasetBundle(
        train=DatasetSplit(records=train),
        val=DatasetSplit(records=val),
        test=DatasetSplit(records=test),
        risk_classes=risk_classes,
        emotion_classes=emotion_classes,
    )
