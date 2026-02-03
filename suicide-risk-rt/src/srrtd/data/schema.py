from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PostRecord:
    text: str
    risk: int
    emotion: int
    user_id: str
    timestamp: int
    lang: str = "und"
    meta: dict[str, Any] | None = None


@dataclass
class DatasetSplit:
    records: list[PostRecord]


@dataclass
class DatasetBundle:
    train: DatasetSplit
    val: DatasetSplit
    test: DatasetSplit
    risk_classes: list[str]
    emotion_classes: list[str]
