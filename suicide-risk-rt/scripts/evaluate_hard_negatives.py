from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import sys

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import torch

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover
    from transformers.models.auto.tokenization_auto import AutoTokenizer

from srrtd.models.factory import model_config_from_yaml
from srrtd.models.multitask import SrrtdMultiTaskModel
from srrtd.rag.retriever import RagConfig, RagRetriever
from srrtd.utils.config import apply_overrides, load_yaml
from srrtd.utils.device import resolve_device
from srrtd.utils.privacy import privacy_cfg_from_dict, privacy_preprocess


def _load_ckpt(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def _read_texts(path: Path, text_field: str = "text") -> list[str]:
    out: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if text_field not in (reader.fieldnames or []):
            raise ValueError(f"Missing column '{text_field}' in {path}")
        for row in reader:
            t = (row.get(text_field) or "").strip()
            if t:
                out.append(t)
    return out


def _suicide_index(risk_classes: list[str]) -> int:
    if not risk_classes:
        return 0
    classes_norm = [str(c).strip().lower() for c in risk_classes]
    if "suicide" in classes_norm:
        return int(classes_norm.index("suicide"))
    return 1 if len(risk_classes) > 1 else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", default="data/raw/risk/hard_negatives.csv")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--overrides", nargs="*", default=None)
    ap.add_argument("--use-rag", action="store_true")
    ap.add_argument("--max-n", type=int, default=0, help="If >0, evaluate only first N rows")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    cfg = apply_overrides(cfg, args.overrides)

    device = resolve_device(str(cfg.get("project", {}).get("device", "auto")))

    ckpt_path = (_ROOT / str(args.ckpt)).resolve() if not Path(str(args.ckpt)).is_absolute() else Path(str(args.ckpt))
    ckpt = _load_ckpt(ckpt_path)

    model_cfg = model_config_from_yaml(cfg)
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

    rag = None
    if bool(args.use_rag) and bool(model_cfg.use_rag):
        rag_cfg_d = dict(cfg.get("model", {}).get("rag", {}) or {})
        rag_cfg = RagConfig(
            embed_model=str(rag_cfg_d.get("embed_model")),
            embed_dim=int(rag_cfg_d.get("embed_dim", model_cfg.rag_embed_dim)),
            top_k=int(rag_cfg_d.get("top_k", 5)),
            chroma_dir=str(rag_cfg_d.get("chroma_dir")),
            collection=str(rag_cfg_d.get("collection")),
        )
        rag = RagRetriever(rag_cfg, _ROOT)

    privacy_cfg = privacy_cfg_from_dict(dict(cfg.get("privacy", {}) or {}))
    threshold_high = float(cfg.get("inference", {}).get("threshold_high", 0.55))

    csv_path = (_ROOT / str(args.csv)).resolve() if not Path(str(args.csv)).is_absolute() else Path(str(args.csv))
    texts = _read_texts(csv_path, text_field=str(args.text_field))
    if args.max_n and args.max_n > 0:
        texts = texts[: int(args.max_n)]

    idx = _suicide_index(list(model_cfg.risk_classes))

    probs: list[float] = []
    with torch.no_grad():
        for t in texts:
            t2 = privacy_preprocess(t, privacy_cfg)
            enc = tokenizer([t2], return_tensors="pt", padding=True, truncation=True, max_length=int(model_cfg.max_length))
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            rag_ctx = None
            if rag is not None:
                rag_ctx = rag.retrieve_context_vec([t2], device=device)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                user_ids=["hardneg"],
                update_memory=False,
                gnn_context=None,
                rag_context=rag_ctx,
            )
            p = float(torch.softmax(out["risk_logits"], dim=-1)[0, idx].detach().cpu().item())
            probs.append(p)

    if not probs:
        print("No rows found.")
        return 2

    frac_high = sum(1 for p in probs if p >= threshold_high) / len(probs)
    mean_p = sum(probs) / len(probs)

    mode = "with_RAG" if rag is not None else "no_RAG"
    print(f"hard_negatives_rows={len(probs)} mode={mode}")
    print(f"threshold_high={threshold_high}")
    print(f"mean_p_suicide={mean_p:.4f}")
    print(f"false_positive_rate={frac_high:.3%}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
