from __future__ import annotations

import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import argparse
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

from srrtd.models.factory import model_config_from_yaml
from srrtd.models.multitask import SrrtdMultiTaskModel
from srrtd.utils.config import apply_overrides, load_yaml
from srrtd.utils.device import resolve_device
from srrtd.utils.privacy import privacy_cfg_from_dict, privacy_preprocess


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1)
    user_id: str = Field(default="anon")
    update_memory: bool = Field(default=True)


class PredictResponse(BaseModel):
    user_id: str
    text: str
    risk_label: str
    risk_prob_suicide: float
    risk_probs: dict[str, float]
    emotion_top: str
    emotion_probs: dict[str, float]
    disclaimer: str


def _load_ckpt(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overrides", nargs="*", default=None)
    ap.add_argument("--ckpt", required=True, help="Path to a .pt checkpoint (e.g., outputs/<run>/best.pt)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", default=8000, type=int)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args.overrides)

    device = resolve_device(str(cfg.get("project", {}).get("device", "auto")))

    ckpt_path = (_ROOT / str(args.ckpt)).resolve() if not Path(str(args.ckpt)).is_absolute() else Path(str(args.ckpt))
    ckpt = _load_ckpt(ckpt_path)

    model_cfg = model_config_from_yaml(cfg)
    # Align head sizes to checkpoint label spaces.
    model_cfg = model_cfg.__class__(
        **{
            **model_cfg.__dict__,
            "risk_classes": list(ckpt.get("risk_classes", model_cfg.risk_classes)),
            "emotion_classes": list(ckpt.get("emotion_classes", model_cfg.emotion_classes)),
        }
    )

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt.get("tokenizer_name", model_cfg.encoder_name)))

    model = SrrtdMultiTaskModel(model_cfg)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.to(device)
    model.eval()

    privacy_cfg = privacy_cfg_from_dict(dict(cfg.get("privacy", {}) or {}))
    threshold_high = float(cfg.get("inference", {}).get("threshold_high", 0.55))

    app = FastAPI(title="SRRTD API", version="0.1")

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "device": str(device),
            "encoder": str(model_cfg.encoder_name),
            "risk_classes": list(model_cfg.risk_classes),
            "emotion_classes": list(model_cfg.emotion_classes),
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        text = privacy_preprocess(req.text, privacy_cfg)
        user_id = req.user_id or "anon"

        enc = tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(model_cfg.max_length),
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                user_ids=[user_id],
                update_memory=bool(req.update_memory),
                gnn_context=None,
                rag_context=None,
            )

        risk_probs_t = torch.softmax(out["risk_logits"], dim=-1)[0].detach().cpu()
        emo_probs_t = torch.softmax(out["emotion_logits"], dim=-1)[0].detach().cpu()

        risk_classes = list(model_cfg.risk_classes)
        emo_classes = list(model_cfg.emotion_classes)

        risk_probs = {c: float(risk_probs_t[i].item()) for i, c in enumerate(risk_classes)}
        emo_probs = {c: float(emo_probs_t[i].item()) for i, c in enumerate(emo_classes)}

        # By convention in this repo/config: index 1 is "suicide".
        suicide_idx = 1 if len(risk_classes) > 1 else 0
        p_suicide = float(risk_probs_t[suicide_idx].item())
        risk_label = "high" if p_suicide >= threshold_high else "low"

        top_emo_idx = int(torch.argmax(emo_probs_t).item())
        emotion_top = emo_classes[top_emo_idx] if 0 <= top_emo_idx < len(emo_classes) else "unknown"

        return PredictResponse(
            user_id=user_id,
            text=text,
            risk_label=risk_label,
            risk_prob_suicide=p_suicide,
            risk_probs=risk_probs,
            emotion_top=emotion_top,
            emotion_probs=emo_probs,
            disclaimer=(
                "This output is a research risk/emotion signal, not a diagnosis. "
                "If someone may be in immediate danger, contact local emergency services." 
            ),
        )

    import uvicorn

    uvicorn.run(app, host=str(args.host), port=int(args.port), log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
