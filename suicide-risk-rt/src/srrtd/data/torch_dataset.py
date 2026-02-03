from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from srrtd.data.schema import DatasetSplit, PostRecord


@dataclass
class TorchExample:
    text: str
    risk: int
    emotion: int
    user_id: str
    timestamp: int
    lang: str
    meta: dict[str, Any]


class StreamingPostDataset(Dataset):
    def __init__(self, split: DatasetSplit):
        self.records = sorted(split.records, key=lambda r: (int(r.timestamp), str(r.user_id)))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> TorchExample:
        r: PostRecord = self.records[idx]
        return TorchExample(
            text=r.text,
            risk=int(r.risk),
            emotion=int(r.emotion),
            user_id=str(r.user_id),
            timestamp=int(r.timestamp),
            lang=str(r.lang),
            meta=dict(r.meta or {}),
        )


def collate_tokenized(tokenizer, max_length: int):
    def _fn(batch: list[TorchExample]) -> dict[str, Any]:
        texts = [b.text for b in batch]
        enc = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
        return {
            "texts": texts,
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "risk": torch.tensor([b.risk for b in batch], dtype=torch.long),
            "emotion": torch.tensor([b.emotion for b in batch], dtype=torch.long),
            "user_ids": [b.user_id for b in batch],
            "timestamp": torch.tensor([b.timestamp for b in batch], dtype=torch.long),
            "lang": [b.lang for b in batch],
        }

    return _fn
