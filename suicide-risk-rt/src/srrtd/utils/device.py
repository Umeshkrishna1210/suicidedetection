from __future__ import annotations

import torch


def resolve_device(device: str) -> torch.device:
    d = str(device).lower().strip()
    if d == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(d)
